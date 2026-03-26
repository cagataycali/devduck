"""
🦆 DevDuck TUI - Multi-conversation terminal UI with Textual.

Features:
  - Concurrent interleaved conversations with color-coded panels
  - Streaming markdown rendering (tables, code blocks, lists)
  - Tool call tracking with icons and timing
  - Collapsible conversation panels (click header to toggle)
  - Elapsed time per conversation
  - Slash commands (/clear, /status, /peers, /tools, /help)
  - Zenoh peer sidebar with live updates
  - Keyboard shortcuts (Ctrl+L, Ctrl+K, Ctrl+T toggle sidebar)
  - Search/filter conversations
  - Token-efficient: caches expensive lookups, batched scrolling

Usage:
    devduck --tui
    from devduck.tui import run_tui; run_tui()
"""

import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.css.query import NoMatches
from textual.widgets import (
    Header,
    Footer,
    Input,
    Static,
    RichLog,
    Collapsible,
    LoadingIndicator,
    OptionList,
)
from textual.widgets.option_list import Option
from textual.message import Message
from textual.events import Key

from rich.text import Text
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.columns import Columns
from rich.align import Align
from rich.console import Group
from rich import box


# ─── Color palette for concurrent conversations ────────────────
COLORS = [
    "#61afef",  # blue
    "#98c379",  # green
    "#e5c07b",  # yellow
    "#c678dd",  # purple
    "#56b6c2",  # cyan
    "#e06c75",  # red
    "#d19a66",  # orange
    "#be5046",  # dark red
    "#7ec8e3",  # light blue
    "#c3e88d",  # light green
]

TOOL_ICONS = {
    "shell": "🐚", "editor": "📝", "file_read": "📖", "file_write": "💾",
    "use_github": "🐙", "use_agent": "🤖", "retrieve": "🔍", "system_prompt": "⚙️",
    "manage_tools": "🔧", "zenoh_peer": "🌐", "tasks": "📋", "scheduler": "⏰",
    "sqlite_memory": "🧠", "use_computer": "🖥️", "apple_vision": "👁️",
    "list_issues": "📋", "list_pull_requests": "🔀", "add_comment": "💬",
    "gist": "📄", "tui": "🖥️", "store_in_kb": "💾", "telegram": "📱",
    "dialog": "💬", "listen": "🎤", "session_recorder": "🎬",
    "strands-docs_search_docs": "📚", "strands-docs_fetch_doc": "📚",
    "apple_nlp": "🧠", "apple_smc": "🌡️", "apple_wifi": "📶",
    "apple_sensors": "📡", "websocket": "🔌", "agentcore_proxy": "☁️",
    "manage_messages": "💬", "view_logs": "📋", "file_read": "📖",
    "speech_to_speech": "🎤", "speech_session": "🎙️",
}


# ─── Input History & Autocomplete ───────────────────────────────

def _load_input_history(max_entries: int = 500) -> List[str]:
    """Load input history from devduck shell history file (deduped, recent first)."""
    history: List[str] = []
    try:
        from devduck import get_shell_history_file
        history_file = get_shell_history_file()
        with open(history_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "# devduck:" in line:
                    try:
                        query = line.split("# devduck:")[-1].strip()
                        if query and query not in ("exit", "quit", "q"):
                            history.append(query)
                    except (ValueError, IndexError):
                        continue
    except Exception:
        pass
    # Dedupe keeping last occurrence, return recent-first
    seen = set()
    deduped = []
    for item in reversed(history):
        if item not in seen:
            seen.add(item)
            deduped.append(item)
        if len(deduped) >= max_entries:
            break
    return deduped  # most recent first


def _build_completions() -> List[str]:
    """Build autocomplete word list from commands + history."""
    base = [
        "exit", "quit", "q", "help", "clear", "status", "reload",
        "ambient", "auto", "autonomous", "record",
        "/help", "/clear", "/clearall", "/status", "/peers", "/tools",
        "/sidebar", "/ambient", "/auto", "/record", "/voice", "/logs",
        "/schedule",
    ]
    try:
        from devduck import extract_commands_from_history
        base.extend(extract_commands_from_history())
    except Exception:
        pass
    return sorted(set(base))


class HistoryInput(Input):
    """Input widget with up/down arrow history and inline autocomplete suggestions.

    - Up/Down arrows cycle through previous inputs (like bash/zsh)
    - Tab accepts the current ghost suggestion
    - Typing filters suggestions from history + commands
    """

    BINDINGS = [
        Binding("up", "history_prev", "Previous", show=False, priority=True),
        Binding("down", "history_next", "Next", show=False, priority=True),
        Binding("tab", "accept_suggestion", "Accept", show=False, priority=True),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history: List[str] = []
        self._history_index: int = -1  # -1 = not browsing history
        self._saved_value: str = ""  # value before entering history mode
        self._completions: List[str] = []
        self._current_suggestion: str = ""
        # Prefix index for O(1) suggestion lookup instead of O(n) linear scan
        self._prefix_index: Dict[str, List[str]] = {}
        self._suggest_timer = None  # Debounce timer for suggestions

    def on_mount(self) -> None:
        # Load history + completions on mount (fast, from file)
        self._history = _load_input_history()
        self._completions = _build_completions()
        self._rebuild_prefix_index()

    def _rebuild_prefix_index(self) -> None:
        """Build prefix lookup dict for O(1) autocomplete. Called once on mount + on new input."""
        idx: Dict[str, List[str]] = {}
        # Index history entries (higher priority, added first)
        for entry in self._history:
            for plen in range(2, min(len(entry) + 1, 6)):  # prefixes of length 2-5
                key = entry[:plen].lower()
                if key not in idx:
                    idx[key] = []
                if entry not in idx[key]:
                    idx[key].append(entry)
        # Index completions (lower priority)
        for cmd in self._completions:
            for plen in range(2, min(len(cmd) + 1, 6)):
                key = cmd[:plen].lower()
                if key not in idx:
                    idx[key] = []
                if cmd not in idx[key]:
                    idx[key].append(cmd)
        self._prefix_index = idx

    def record_input(self, value: str) -> None:
        """Record a submitted input into the in-memory history."""
        value = value.strip()
        if value and value not in ("exit", "quit", "q"):
            # Remove duplicate if exists, then prepend
            if value in self._history:
                self._history.remove(value)
            self._history.insert(0, value)
            # Add words to completions
            for word in value.split():
                if len(word) > 2 and word not in self._completions:
                    self._completions.append(word)
            # Update prefix index incrementally (just add the new entry)
            for plen in range(2, min(len(value) + 1, 6)):
                key = value[:plen].lower()
                if key not in self._prefix_index:
                    self._prefix_index[key] = []
                if value not in self._prefix_index[key]:
                    self._prefix_index[key].insert(0, value)

    # ── History navigation ──────────────────────────────────────

    def action_history_prev(self) -> None:
        """Go to previous (older) history entry."""
        if not self._history:
            return
        if self._history_index == -1:
            # Save current typed text before entering history mode
            self._saved_value = self.value
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self.value = self._history[self._history_index]
            self.cursor_position = len(self.value)
            self._current_suggestion = ""

    def action_history_next(self) -> None:
        """Go to next (newer) history entry, or restore typed text."""
        if self._history_index <= -1:
            return
        self._history_index -= 1
        if self._history_index == -1:
            # Restore the text user was typing before browsing history
            self.value = self._saved_value
        else:
            self.value = self._history[self._history_index]
        self.cursor_position = len(self.value)
        self._current_suggestion = ""

    # ── Autocomplete suggestion ─────────────────────────────────

    def _find_suggestion(self) -> str:
        """Find a ghost suggestion using prefix index — O(1) lookup instead of O(n) scan."""
        text = self.value.strip()
        if not text or len(text) < 2:
            return ""
        text_lower = text.lower()
        # Use the longest prefix we indexed (up to 5 chars) for best filtering
        lookup_len = min(len(text), 5)
        key = text_lower[:lookup_len]
        candidates = self._prefix_index.get(key, [])
        for entry in candidates:
            if entry.lower().startswith(text_lower) and entry != text:
                return entry
        # Fallback: try shorter prefixes
        if lookup_len > 2:
            for plen in range(lookup_len - 1, 1, -1):
                key = text_lower[:plen]
                candidates = self._prefix_index.get(key, [])
                for entry in candidates:
                    if entry.lower().startswith(text_lower) and entry != text:
                        return entry
        return ""

    def action_accept_suggestion(self) -> None:
        """Tab key: accept the current ghost suggestion."""
        if self._current_suggestion:
            self.value = self._current_suggestion
            self.cursor_position = len(self.value)
            self._current_suggestion = ""
            self.placeholder = self._default_placeholder()
        # If no suggestion, Tab does nothing (don't insert tab char)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update ghost suggestion as user types — debounced for fast typing."""
        # Reset history browsing when user types
        if self._history_index != -1:
            self._history_index = -1

        # Cancel previous debounce timer and set a new one (80ms delay)
        # This means suggestions only compute AFTER you pause typing briefly,
        # so rapid keystrokes never block the input.
        if self._suggest_timer is not None:
            self._suggest_timer.stop()
        self._suggest_timer = self.set_timer(0.08, self._deferred_suggestion)

    def _deferred_suggestion(self) -> None:
        """Run suggestion lookup after debounce period."""
        self._suggest_timer = None
        suggestion = self._find_suggestion()
        self._current_suggestion = suggestion
        if suggestion:
            self.placeholder = suggestion
        else:
            self.placeholder = self._default_placeholder()

    def _default_placeholder(self) -> str:
        return "  Ask anything… | ! shell cmd | /help | ↑↓ history | Tab complete"


# ─── Message types for thread-safe TUI updates ─────────────────

class StreamChunk(Message):
    """Streamed text from the agent."""
    def __init__(self, conv_id: int, text: str) -> None:
        super().__init__()
        self.conv_id = conv_id
        self.text = text


class ToolEvent(Message):
    """Tool start/end event."""
    def __init__(self, conv_id: int, tool_name: str, status: str, detail: str = "") -> None:
        super().__init__()
        self.conv_id = conv_id
        self.tool_name = tool_name
        self.status = status
        self.detail = detail


class ConversationDone(Message):
    """Conversation finished."""
    def __init__(self, conv_id: int, error: str = "") -> None:
        super().__init__()
        self.conv_id = conv_id
        self.error = error


# ─── TUI Callback Handler ──────────────────────────────────────

class TUICallbackHandler:
    """Routes agent events to TUI messages. Runs in worker thread."""

    def __init__(self, app: "DevDuckTUI", conv_id: int):
        self.app = app
        self.conv_id = conv_id
        self._current_tool_id = None
        self._current_tool_name = None
        self._tool_times: Dict[str, float] = {}

    def __call__(self, **kwargs: Any) -> None:
        data = kwargs.get("data", "")
        current_tool_use = kwargs.get("current_tool_use", {})
        message = kwargs.get("message", {})
        reasoningText = kwargs.get("reasoningText", "")
        force_stop = kwargs.get("force_stop", False)
        event_loop_throttled_delay = kwargs.get("event_loop_throttled_delay", None)

        # Stream text
        if data:
            self.app.post_message(StreamChunk(self.conv_id, data))

        # Reasoning / thinking
        if reasoningText:
            self.app.post_message(StreamChunk(self.conv_id, reasoningText))

        # Tool use detection via input streaming
        if current_tool_use and current_tool_use.get("input"):
            tid = current_tool_use.get("toolUseId", "")
            tname = current_tool_use.get("name", "unknown")
            if tid != self._current_tool_id:
                self._current_tool_id = tid
                self._current_tool_name = tname
                self._tool_times[tid] = time.time()
                self.app.post_message(ToolEvent(self.conv_id, tname, "start"))

        # Tool results
        if isinstance(message, dict):
            if message.get("role") == "user":
                for content in message.get("content", []):
                    if isinstance(content, dict):
                        tr = content.get("toolResult")
                        if tr:
                            tid = tr.get("toolUseId", "")
                            status = tr.get("status", "unknown")
                            dur = ""
                            if tid in self._tool_times:
                                dur = f"{time.time() - self._tool_times.pop(tid):.1f}s"
                            name = self._current_tool_name or "tool"
                            self.app.post_message(ToolEvent(self.conv_id, name, status, dur))
                            self._current_tool_id = None
                            self._current_tool_name = None

            if message.get("role") == "assistant":
                for content in message.get("content", []):
                    if isinstance(content, dict) and content.get("toolUse"):
                        tu = content["toolUse"]
                        tid = tu.get("toolUseId", "")
                        tname = tu.get("name", "tool")
                        self._current_tool_name = tname
                        if tid and tid not in self._tool_times:
                            self._tool_times[tid] = time.time()
                            self.app.post_message(ToolEvent(self.conv_id, tname, "start"))

        if event_loop_throttled_delay:
            self.app.post_message(
                StreamChunk(self.conv_id, f"\n⏳ Throttled {event_loop_throttled_delay}s…\n")
            )

        if force_stop:
            self.app.post_message(ConversationDone(self.conv_id))


# ─── Conversation Panel ────────────────────────────────────────

class ConversationPanel(Static):
    """A single conversation with streaming markdown output."""

    def __init__(self, conv_id: int, query: str, color: str, **kwargs):
        super().__init__(**kwargs)
        self.conv_id = conv_id
        self.query = query
        self.color = color
        self.is_done = False
        self._tool_count = 0
        self._stream_buffer = ""
        self._full_response = ""
        self._start_time = time.time()
        self._end_time: Optional[float] = None

    def compose(self) -> ComposeResult:
        yield RichLog(
            id=f"log-{self.conv_id}",
            highlight=True,
            markup=True,
            auto_scroll=True,
            min_width=40,
            wrap=True,
        )

    def on_mount(self) -> None:
        log = self.query_one(f"#log-{self.conv_id}", RichLog)
        header = Text()
        header.append(f" #{self.conv_id} ", style=f"bold on {self.color}")
        header.append(f"  {self.query}", style="bold white")
        log.write(header)
        log.write(Rule(style=self.color))

    @property
    def elapsed(self) -> float:
        end = self._end_time or time.time()
        return end - self._start_time

    @property
    def elapsed_str(self) -> str:
        e = self.elapsed
        if e < 60:
            return f"{e:.1f}s"
        return f"{int(e // 60)}m{int(e % 60)}s"

    # ── Streaming markdown ──────────────────────────────────────

    def append_text(self, text: str) -> None:
        """Buffer streamed text and render as Markdown at paragraph boundaries."""
        try:
            log = self.query_one(f"#log-{self.conv_id}", RichLog)
        except NoMatches:
            return

        self._stream_buffer += text
        self._full_response += text

        # Render at paragraph boundaries or when buffer gets large
        if "\n\n" in self._stream_buffer or len(self._stream_buffer) > 400:
            self._render_chunk(log)

    def _render_chunk(self, log: RichLog) -> None:
        """Render buffered text as Markdown."""
        if not self._stream_buffer.strip():
            return

        content = self._stream_buffer
        remaining = ""

        # Break at paragraph boundary
        if "\n\n" in content:
            idx = content.rfind("\n\n")
            remaining = content[idx + 2:]
            content = content[:idx + 2]
        elif content.endswith("\n"):
            pass  # clean line break
        elif len(content) < 400:
            return  # keep buffering
        else:
            # Force break at last newline
            idx = content.rfind("\n")
            if idx > 0:
                remaining = content[idx + 1:]
                content = content[:idx + 1]

        if content.strip():
            try:
                log.write(Markdown(content.rstrip()))
            except Exception:
                log.write(Text(content))

        self._stream_buffer = remaining
        self._scroll_parent()

    def _flush(self) -> None:
        """Flush remaining buffer as Markdown."""
        if not self._stream_buffer.strip():
            self._stream_buffer = ""
            return
        try:
            log = self.query_one(f"#log-{self.conv_id}", RichLog)
            try:
                log.write(Markdown(self._stream_buffer.rstrip()))
            except Exception:
                log.write(Text(self._stream_buffer))
            self._stream_buffer = ""
            self._scroll_parent()
        except NoMatches:
            pass

    def _scroll_parent(self) -> None:
        try:
            scroll = self.app.query_one("#conversations-scroll", ScrollableContainer)
            scroll.scroll_end(animate=False)
        except Exception:
            pass

    # ── Tool events ─────────────────────────────────────────────

    def append_tool_event(self, tool_name: str, status: str, detail: str = "") -> None:
        self._flush()
        try:
            log = self.query_one(f"#log-{self.conv_id}", RichLog)
            icon = TOOL_ICONS.get(tool_name, "🔧")

            if status == "start":
                self._tool_count += 1
                t = Text()
                t.append(f"  {icon} ", style="bold")
                t.append(tool_name, style=f"bold {self.color}")
                t.append(" ⟳", style="yellow")
                log.write(t)
            elif status == "success":
                t = Text()
                t.append("    ✓ ", style="bold green")
                t.append(tool_name, style="green")
                if detail:
                    t.append(f" ({detail})", style="dim")
                log.write(t)
            elif status == "error":
                t = Text()
                t.append("    ✗ ", style="bold red")
                t.append(tool_name, style="red")
                if detail:
                    t.append(f" ({detail})", style="dim red")
                log.write(t)
        except NoMatches:
            pass

    # ── Completion ──────────────────────────────────────────────

    def mark_done(self, error: str = "") -> None:
        self.is_done = True
        self._end_time = time.time()
        self._flush()
        try:
            log = self.query_one(f"#log-{self.conv_id}", RichLog)
            log.write(Text(""))
            if error:
                t = Text()
                t.append("  ✗ Error: ", style="bold red")
                t.append(error[:300], style="red")
                log.write(t)
            else:
                t = Text()
                t.append("  ✓ Done", style=f"bold {self.color}")
                if self._tool_count > 0:
                    t.append(f" ({self._tool_count} tools", style="dim")
                    t.append(f", {self.elapsed_str})", style="dim")
                else:
                    t.append(f" ({self.elapsed_str})", style="dim")
                log.write(t)
            log.write(Rule(style="dim"))
            self._scroll_parent()
        except NoMatches:
            pass


# ─── Sidebar ────────────────────────────────────────────────────

# ─── Sidebar ────────────────────────────────────────────────────
# No separate widget — sidebar sections are managed directly by the app
# for better performance and interactivity.


# ─── Thread-safe shared message list ────────────────────────────

import threading as _threading
import copy as _copy


class SharedMessages(list):
    """Thread-safe list that multiple Agent instances share as their `messages`.

    Each concurrent Agent points its `.messages` to the SAME SharedMessages
    instance. All reads/writes are serialized via a lock so message ordering
    is preserved regardless of which thread appends.

    This gives:
      - True concurrency (each Agent has its own callback handler)
      - Shared awareness (agents see each other's messages in real-time)
      - Correct ordering (lock serialises mutations)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = _threading.Lock()

    # ── Mutating methods (need lock) ────────────────────────────

    def append(self, item):
        with self._lock:
            super().append(item)

    def extend(self, items):
        with self._lock:
            super().extend(items)

    def insert(self, index, item):
        with self._lock:
            super().insert(index, item)

    def pop(self, index=-1):
        with self._lock:
            return super().pop(index)

    def remove(self, item):
        with self._lock:
            super().remove(item)

    def clear(self):
        with self._lock:
            super().clear()

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with self._lock:
            super().__delitem__(key)

    def __iadd__(self, other):
        with self._lock:
            super().__iadd__(other)
            return self

    # ── Read methods (need lock for consistency) ────────────────

    def __len__(self):
        with self._lock:
            return super().__len__()

    def __getitem__(self, key):
        with self._lock:
            return super().__getitem__(key)

    def __iter__(self):
        """Return a snapshot iterator that excludes trailing orphaned toolUse.

        When the model reads messages for inference, an orphaned toolUse
        (assistant message with toolUse but no following user+toolResult)
        causes API errors. This happens when Agent #1 is mid-tool-execution
        and Agent #2 iterates the shared list concurrently.

        We trim trailing messages that form an incomplete tool turn:
        - If the last message is assistant with toolUse → exclude it
        - If the last messages are assistant+toolUse followed by user+toolResult
          but no final assistant response → that's fine (model will continue)
        """
        with self._lock:
            msgs = list(super().__iter__())

        # Trim trailing incomplete tool turns from the snapshot
        while msgs:
            last = msgs[-1]
            if not isinstance(last, dict):
                break
            role = last.get("role", "")
            content = last.get("content", [])

            # Check if last message is an assistant message containing toolUse
            has_tool_use = False
            if role == "assistant":
                for c in content:
                    if isinstance(c, dict) and "toolUse" in c:
                        has_tool_use = True
                        break

            if has_tool_use:
                # Orphaned toolUse — another agent is mid-execution.
                # Remove it so model doesn't see an incomplete turn.
                msgs.pop()
            else:
                break

        return iter(msgs)

    def __contains__(self, item):
        with self._lock:
            return super().__contains__(item)

    def snapshot(self) -> list:
        """Return a plain-list copy (for serialization, export, etc.)."""
        with self._lock:
            return list(self)

    def trim_to(self, max_size: int):
        """Keep only the last max_size messages."""
        with self._lock:
            if super().__len__() > max_size:
                excess = super().__len__() - max_size
                del self[:excess]


# ─── Main TUI App ──────────────────────────────────────────────

class DevDuckTUI(App):
    """DevDuck multi-conversation TUI."""

    TITLE = "🦆"
    SUB_TITLE = "DevDuck"

    CSS = """
    Screen {
        layout: horizontal;
    }

    #main-area {
        width: 1fr;
        height: 100%;
        layout: vertical;
    }

    #sidebar {
        width: 30;
        height: 100%;
        background: $surface-darken-1;
        border-left: thick $primary-background;
        padding: 1 1 0 1;
    }

    .sidebar-hidden #sidebar {
        display: none;
    }

    #sidebar-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        padding-bottom: 1;
    }

    #sidebar-scroll {
        height: 1fr;
        scrollbar-size: 1 1;
    }

    .sidebar-section {
        height: auto;
        padding: 0 0 1 0;
    }

    .sidebar-section-title {
        text-style: bold;
        color: $text-muted;
        padding: 0 0 0 0;
    }

    #conversations-scroll {
        height: 1fr;
        scrollbar-size: 1 1;
    }

    ConversationPanel {
        height: auto;
        margin: 0 0 1 0;
        border: round $primary-background;
        padding: 0 1;
    }

    ConversationPanel.done {
        border: round $success-darken-2;
        opacity: 0.85;
    }

    ConversationPanel.error {
        border: round $error;
    }

    #input-area {
        height: auto;
        max-height: 5;
        dock: bottom;
        padding: 0 1;
        background: $surface;
        border-top: thick $primary-background;
    }

    #query-input {
        margin: 0;
    }

    #status-bar {
        height: 1;
        dock: bottom;
        background: $primary-background;
        color: $text;
        padding: 0 1;
    }

    RichLog {
        height: auto;
        max-height: 80;
        scrollbar-size: 1 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True, priority=True),
        Binding("ctrl+l", "clear_done", "Clear Done", show=True),
        Binding("ctrl+k", "clear_all", "Clear All", show=True),
        Binding("ctrl+t", "toggle_sidebar", "Sidebar", show=True),
        Binding("ctrl+v", "toggle_voice", "🎤 Voice", show=True),
        Binding("space", "ptt_press", "Push-to-Talk", show=False),
        Binding("escape", "focus_input", "Focus Input", show=False),
    ]

    def __init__(self, devduck_instance=None, **kwargs):
        super().__init__(**kwargs)
        self._devduck = devduck_instance
        self._conv_counter = 0
        self._active_conversations: Dict[int, ConversationPanel] = {}
        self._total_queries = 0
        self._sidebar_visible = True
        # Speech-to-speech state
        self._speech_session_id: Optional[str] = None
        self._speech_provider: str = "novasonic"
        self._speech_panel_id: Optional[int] = None
        # Shared message history — all concurrent conversations read/write to this
        # single thread-safe list. Each conv gets a fresh Agent whose .messages
        # is pointed at this shared instance. This gives:
        #   1. True concurrency (separate Agent instances, no callback handler conflicts)
        #   2. Real-time shared awareness (agents see each other's messages as they stream)
        #   3. Correct ordering (SharedMessages lock serialises all mutations)
        self._shared_messages = SharedMessages()
        # Cache
        self._zenoh_mod = None
        self._zenoh_checked = False
        self._ring_last_count = 0  # Track last seen ring entries
        self._log_last_pos = 0  # Track last read position in log file
        model_name = str(getattr(devduck_instance, "model", "?"))
        self._model_display = ("…" + model_name[-29:]) if len(model_name) > 30 else model_name

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():
            with Vertical(id="main-area"):
                yield ScrollableContainer(id="conversations-scroll")
                with Horizontal(id="input-area"):
                    yield HistoryInput(
                        placeholder="  Ask anything… | ! shell cmd | /help | ↑↓ history | Tab complete",
                        id="query-input",
                    )
                yield Static(id="status-bar")

            with Vertical(id="sidebar"):
                yield Static("🦆 DevDuck", id="sidebar-title")
                with ScrollableContainer(id="sidebar-scroll"):
                    yield Static(id="sb-convos")
                    yield Static(id="sb-voice")
                    yield Static(id="sb-tools")
                    yield Static(id="sb-schedules")
                    yield Static(id="sb-peers")
                    yield Static(id="sb-net-feed")
                    yield Static(id="sb-stats")

        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#query-input", HistoryInput).focus()
        self._update_status_bar()
        self._ptt_timer = None  # PTT debounce timer
        self.set_interval(5.0, self._update_status_bar)
        self.set_interval(10.0, self._update_sidebar_stats)
        self.set_interval(3.0, self._update_sb_net_feed)
        self.set_interval(1.0, self._update_sb_voice)  # Voice sidebar — fast only when active
        self._show_welcome()

        # Register for tui tool
        try:
            from devduck.tools.tui import set_tui_app
            set_tui_app(self)
        except ImportError:
            pass

    def _build_landing_panel(self):
        """Build the rich landing dashboard panel (mirrors landing.py style)."""
        from devduck import get_session_recorder

        def _status_dot(ok: bool) -> str:
            return "[green]●[/green]" if ok else "[dim]○[/dim]"

        # ── Duck art + title ────────────────────────────────────
        duck_lines = [
            "        .__       ",
            "     __/  .\\     ",
            "   <(o  )___\\   ",
            "    ( ._>   /    ",
            "     `----'`     ",
        ]
        duck_colors = ["bright_yellow", "yellow", "bright_yellow", "yellow", "bright_yellow"]
        duck_text = Text()
        for i, line in enumerate(duck_lines):
            duck_text.append(line + "\n", style=duck_colors[i])

        title_text = Text()
        title_text.append("D", style="bold bright_yellow")
        title_text.append("ev", style="bold white")
        title_text.append("D", style="bold bright_yellow")
        title_text.append("uck", style="bold white")
        title_text.append("  ", style="")
        try:
            from importlib.metadata import version as pkg_version
            ver = pkg_version("devduck")
        except Exception:
            ver = "dev"
        title_text.append(f"v{ver}", style="dim")
        title_text.append("\n", style="")
        title_text.append("Self-modifying AI agent\n", style="dim italic")
        title_text.append("Multi-conversation TUI", style="dim italic bright_cyan")

        header_table = Table.grid(padding=(0, 2))
        header_table.add_row(duck_text, title_text)

        # ── Info cards ──────────────────────────────────────────
        model_display = self._model_display
        tool_count = len(self._devduck.tools) if self._devduck and hasattr(self._devduck, "tools") else 0

        env = getattr(self._devduck, "env_info", {})
        os_str = f"{env.get('os', '?')} {env.get('arch', '')}"
        py_info = env.get("python", sys.version_info)
        py_str = f"{py_info.major}.{py_info.minor}.{py_info.micro}" if hasattr(py_info, "major") else str(py_info)

        cards = []
        cards.append(Panel(
            f"[bold bright_cyan]{model_display}[/]",
            title="[bold]🧠 Model[/]",
            border_style="cyan",
            box=box.ROUNDED,
            expand=True,
        ))
        cards.append(Panel(
            f"[bold]{os_str}[/]\nPython {py_str}",
            title="[bold]💻 Env[/]",
            border_style="green",
            box=box.ROUNDED,
            expand=True,
        ))
        cards.append(Panel(
            f"[bold bright_green]{tool_count}[/] [dim]tools loaded[/]",
            title="[bold]🛠️  Tools[/]",
            border_style="bright_green",
            box=box.ROUNDED,
            expand=True,
        ))
        info_row = Columns(cards, equal=True, expand=True)

        # ── Services status ─────────────────────────────────────
        svc_table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
            expand=True,
            padding=(0, 1),
        )
        svc_table.add_column("Service", style="bold", min_width=12)
        svc_table.add_column("Status", justify="center", min_width=4)
        svc_table.add_column("Endpoint", style="dim")
        svc_table.add_column("Info", style="dim italic")

        servers = getattr(self._devduck, "servers", {})

        # Zenoh
        zenoh_enabled = servers.get("zenoh_peer", {}).get("enabled", False)
        zenoh_id = ""
        zenoh_peers = 0
        if zenoh_enabled:
            try:
                _zp_mod = sys.modules.get("devduck.tools.zenoh_peer")
                if _zp_mod:
                    zenoh_id = _zp_mod.ZENOH_STATE.get("instance_id", "")
                    zenoh_peers = len(_zp_mod.ZENOH_STATE.get("peers", {}))
            except Exception:
                pass
        svc_table.add_row(
            "Zenoh P2P", _status_dot(zenoh_enabled),
            f"[bright_magenta]{zenoh_id}[/]" if zenoh_id else "—",
            f"{zenoh_peers} peer(s)" if zenoh_enabled else "",
        )

        # Mesh Relay
        ac_cfg = servers.get("agentcore_proxy", {})
        ac_enabled = ac_cfg.get("enabled", False)
        ac_port = ac_cfg.get("port", 10000)
        svc_table.add_row(
            "Mesh Relay", _status_dot(ac_enabled),
            f"ws://localhost:{ac_port}" if ac_enabled else "—",
            "browser + cloud",
        )

        # WebSocket
        ws_cfg = servers.get("ws", {})
        ws_enabled = ws_cfg.get("enabled", False)
        ws_port = ws_cfg.get("port", 10001)
        svc_table.add_row(
            "WebSocket", _status_dot(ws_enabled),
            f"ws://localhost:{ws_port}" if ws_enabled else "—",
            "per-msg streaming",
        )

        # TCP
        tcp_cfg = servers.get("tcp", {})
        tcp_enabled = tcp_cfg.get("enabled", False)
        tcp_port = tcp_cfg.get("port", 10002)
        svc_table.add_row("TCP", _status_dot(tcp_enabled),
            f"localhost:{tcp_port}" if tcp_enabled else "—", "")

        # MCP
        mcp_cfg = servers.get("mcp", {})
        mcp_enabled = mcp_cfg.get("enabled", False)
        mcp_port = mcp_cfg.get("port", 10003)
        svc_table.add_row("MCP", _status_dot(mcp_enabled),
            f"http://localhost:{mcp_port}/mcp" if mcp_enabled else "—", "")

        # IPC
        ipc_cfg = servers.get("ipc", {})
        ipc_enabled = ipc_cfg.get("enabled", False)
        ipc_path = ipc_cfg.get("socket_path", "/tmp/devduck_main.sock")
        svc_table.add_row("IPC", _status_dot(ipc_enabled),
            ipc_path if ipc_enabled else "—", "")

        svc_panel = Panel(svc_table, title="[bold]⚡ Services[/]", border_style="bright_blue", box=box.ROUNDED)

        # ── Features row ────────────────────────────────────────
        ambient = getattr(self._devduck, "ambient", None)
        ambient_on = ambient and ambient.running
        ambient_mode = "AUTONOMOUS" if (ambient and ambient.autonomous) else "standard"
        recorder = get_session_recorder()
        recording = recorder and recorder.recording
        watcher = hasattr(self._devduck, "_watcher_running") and self._devduck._watcher_running

        feat_parts = []
        if ambient_on:
            feat_parts.append(f"[bright_yellow]🌙 Ambient[/]: [green]ON[/] ({ambient_mode})")
        else:
            feat_parts.append("[dim]🌙 Ambient: OFF[/]")
        if recording:
            feat_parts.append(f"[red]🎬 Recording[/]: [green]ON[/]")
        else:
            feat_parts.append("[dim]🎬 Recording: OFF[/]")
        if watcher:
            feat_parts.append("[green]🔥 Hot-Reload[/]: [green]watching[/]")
        else:
            feat_parts.append("[dim]🔥 Hot-Reload: OFF[/]")

        feat_table = Table.grid(padding=(0, 3))
        feat_table.add_row(*feat_parts)
        feat_panel = Panel(Align.center(feat_table), border_style="dim", box=box.ROUNDED, padding=(0, 1))

        # ── Quick reference ─────────────────────────────────────
        help_table = Table(box=None, show_header=False, padding=(0, 2), expand=True)
        help_table.add_column("Key", style="bold bright_yellow", min_width=16)
        help_table.add_column("Description", style="dim")

        help_table.add_row("ask anything", "natural language → agent executes concurrently")
        help_table.add_row("[bright_cyan]![/]command", "run shell command directly")
        help_table.add_row("[bright_cyan]ambient[/]", "toggle background thinking")
        help_table.add_row("[bright_cyan]auto[/]", "autonomous mode (runs until done)")
        help_table.add_row("[bright_cyan]record[/]", "toggle session recording")
        help_table.add_row("[bright_cyan]/help[/]", "show this dashboard")
        help_table.add_row("[bright_cyan]/clear[/] [bright_cyan]/tools[/] [bright_cyan]/peers[/]", "manage · Ctrl+L Ctrl+K Ctrl+T")
        help_table.add_row("[bright_cyan]Ctrl+V[/]", "🎤 toggle voice (speech-to-speech)")
        help_table.add_row("[bright_cyan]/voice[/] [dim]provider[/]", "configure voice provider")
        help_table.add_row("[bright_cyan]exit[/] / [bright_cyan]q[/]", "quit  ·  Ctrl+C to force")

        help_panel = Panel(help_table, title="[bold]⌨️  Commands[/]", border_style="dim yellow", box=box.ROUNDED, padding=(0, 0))

        # ── Compose all sections ────────────────────────────────
        return Group(
            Panel(Align.center(header_table), border_style="bright_yellow", box=box.DOUBLE_EDGE, padding=(0, 1)),
            info_row,
            svc_panel,
            feat_panel,
            help_panel,
        )

    def _show_welcome(self) -> None:
        scroll = self.query_one("#conversations-scroll", ScrollableContainer)

        welcome = Static(
            self._build_landing_panel(),
            id="welcome-panel",
        )
        scroll.mount(welcome)

    # ── Cached helpers ──────────────────────────────────────────

    def _get_peer_count(self) -> int:
        if not self._zenoh_checked:
            self._zenoh_mod = sys.modules.get("devduck.tools.zenoh_peer")
            self._zenoh_checked = True
        if self._zenoh_mod:
            try:
                return len(self._zenoh_mod.ZENOH_STATE.get("peers", {}))
            except Exception:
                pass
        return 0

    def _get_next_color(self) -> str:
        return COLORS[self._conv_counter % len(COLORS)]

    # ── Status updates ──────────────────────────────────────────

    def _update_status_bar(self) -> None:
        active = sum(1 for p in self._active_conversations.values() if not p.is_done)
        total = self._total_queries
        peer_count = self._get_peer_count()

        bar = Text()
        bar.append(" 🦆 ", style="bold bright_yellow")
        bar.append(self._model_display, style="bold")
        bar.append("  │  ", style="dim")
        if active > 0:
            bar.append(f"⚡ {active} running", style="bold yellow")
        else:
            bar.append("⚡ idle", style="dim")
        bar.append(f"  📊 {total}", style="dim")
        bar.append("  │  ", style="dim")
        bar.append(f"🌐 {peer_count}", style="cyan" if peer_count > 0 else "dim")

        # Ambient/autonomous indicator
        dd = self._devduck
        if dd and dd.ambient and dd.ambient.running:
            bar.append("  │  ", style="dim")
            if dd.ambient.autonomous:
                bar.append("🚀 auto", style="bold bright_magenta")
            else:
                bar.append("🌙 ambient", style="bright_yellow")

        # Recording indicator
        try:
            from devduck import get_session_recorder
            rec = get_session_recorder()
            if rec and rec.recording:
                bar.append("  │  ", style="dim")
                bar.append("🎬 rec", style="bold red")
        except ImportError:
            pass

        # Speech-to-speech indicator
        try:
            from devduck.tools.speech_to_speech import _active_sessions, _session_lock
            with _session_lock:
                if _active_sessions:
                    bar.append("  │  ", style="dim")
                    bar.append("🎙️ LIVE", style="bold bright_red")
        except ImportError:
            pass

        try:
            self.query_one("#status-bar", Static).update(bar)
        except NoMatches:
            pass

    def _update_sidebar_stats(self) -> None:
        """Update all sidebar sections."""
        self._update_sb_convos()
        self._update_sb_voice()
        self._update_sb_tools()
        self._update_sb_schedules()
        self._update_sb_peers()
        self._update_sb_stats()

    def _update_sb_convos(self) -> None:
        """Active conversations section."""
        try:
            w = self.query_one("#sb-convos", Static)
        except NoMatches:
            return

        if not self._active_conversations:
            w.update(Text(""))
            return

        t = Text()
        t.append("── Conversations ──\n", style="bold dim")
        for cid, panel in self._active_conversations.items():
            icon = "⟳" if not panel.is_done else "✓"
            style = "yellow" if not panel.is_done else "green"
            t.append(f" {icon} ", style=style)
            t.append(f"#{cid} ", style=f"bold {panel.color}")
            preview = panel.query[:18] + "…" if len(panel.query) > 18 else panel.query
            t.append(f"{preview}", style="dim")
            if not panel.is_done:
                t.append(f" {panel.elapsed_str}", style="dim yellow")
            t.append("\n")
        w.update(t)

    def _update_sb_voice(self) -> None:
        """Voice/speech-to-speech section in sidebar with live VAD meter."""
        try:
            w = self.query_one("#sb-voice", Static)
        except NoMatches:
            return

        t = Text()
        t.append("── Voice ──\n", style="bold dim")

        # Check for active speech sessions
        try:
            from devduck.tools.speech_to_speech import _active_sessions, _session_lock
            with _session_lock:
                active_count = len(_active_sessions)
                if active_count > 0:
                    for sid, session in _active_sessions.items():
                        # Status indicator
                        if session.is_transmitting:
                            t.append(" 🎙️ ", style="bold bright_red")
                            t.append("TRANSMIT", style="bold bright_red")
                        else:
                            t.append(" 🎤 ", style="bold bright_yellow")
                            t.append("LISTENING", style="bold bright_yellow")
                        t.append(f"\n   {sid[:16]}\n", style="dim")

                        # VAD meter bar
                        prob = session.speech_probability
                        bar_w = 18
                        filled = int(prob * bar_w)
                        bar_style = "bright_green" if prob > 0.5 else "bright_yellow" if prob > 0.2 else "dim"
                        t.append("   ")
                        t.append("█" * filled, style=bar_style)
                        t.append("░" * (bar_w - filled), style="dim")
                        t.append(f" {prob:.0%}\n", style=bar_style)

                        # PTT hint
                        if session.push_to_talk:
                            if session.is_transmitting:
                                t.append("   ● hold Space…\n", style="bright_red")
                            else:
                                t.append("   ○ press Space\n", style="dim")
                        else:
                            t.append("   ● always-on\n", style="green")

                        # AEC indicator
                        if hasattr(session, '_audio_io') and session._audio_io and session._audio_io._has_aec:
                            t.append("   🔇 AEC+NS on\n", style="dim green")

                    t.append(f" Ctrl+V to stop\n", style="dim")
                else:
                    t.append(" 🎤 ", style="dim")
                    t.append("inactive\n", style="dim")
                    t.append(f" Ctrl+V to start\n", style="dim")
                    t.append(f" /voice to configure\n", style="dim")
        except ImportError:
            t.append(" 🎤 ", style="dim")
            t.append("not loaded\n", style="dim italic")
            t.append(" load speech_to_speech\n", style="dim")

        w.update(t)

    def _update_sb_tools(self) -> None:
        """Tools section — shows loaded tools with counts."""
        try:
            w = self.query_one("#sb-tools", Static)
        except NoMatches:
            return

        if not self._devduck:
            w.update(Text(""))
            return

        tool_names = []
        if hasattr(self._devduck, 'agent') and self._devduck.agent:
            try:
                tool_names = sorted(self._devduck.agent.tool_names)
            except Exception:
                pass

        t = Text()
        t.append("── Tools ", style="bold dim")
        t.append(f"({len(tool_names)})", style="dim")
        t.append(" ──\n", style="bold dim")

        # Show tools in compact format
        for name in tool_names:
            icon = TOOL_ICONS.get(name, "·")
            t.append(f" {icon} ", style="dim")
            # Truncate long tool names
            display = name[:22] + "…" if len(name) > 22 else name
            t.append(f"{display}\n", style="")
        w.update(t)

    def _update_sb_schedules(self) -> None:
        """Schedules section — shows active scheduled jobs."""
        try:
            w = self.query_one("#sb-schedules", Static)
        except NoMatches:
            return

        t = Text()
        t.append("── Schedules ──\n", style="bold dim")

        # Try to get scheduler state
        has_jobs = False
        try:
            sched_mod = sys.modules.get("devduck.tools.scheduler")
            if sched_mod and hasattr(sched_mod, "_SCHEDULER_STATE"):
                state = sched_mod._SCHEDULER_STATE
                jobs = state.get("jobs", {})
                if jobs:
                    has_jobs = True
                    for name, job in jobs.items():
                        enabled = job.get("enabled", True)
                        schedule = job.get("schedule", job.get("run_at", "?"))
                        icon = "✓" if enabled else "○"
                        style = "green" if enabled else "dim"
                        t.append(f" {icon} ", style=style)
                        t.append(f"{name[:18]}", style="bold" if enabled else "dim")
                        t.append(f"\n   {schedule}\n", style="dim")
        except Exception:
            pass

        if not has_jobs:
            t.append(" No scheduled jobs\n", style="dim italic")
            t.append(" /schedule to add\n", style="dim")

        w.update(t)

    def _update_sb_peers(self) -> None:
        """Peers section — compact zenoh peers."""
        try:
            w = self.query_one("#sb-peers", Static)
        except NoMatches:
            return

        if not self._zenoh_checked:
            self._zenoh_mod = sys.modules.get("devduck.tools.zenoh_peer")
            self._zenoh_checked = True

        peers = {}
        my_id = ""
        if self._zenoh_mod:
            try:
                zs = self._zenoh_mod.ZENOH_STATE
                peers = zs.get("peers", {})
                my_id = zs.get("instance_id", "")
            except Exception:
                pass

        t = Text()
        t.append("── Peers ", style="bold dim")
        t.append(f"({len(peers)})", style="dim")
        t.append(" ──\n", style="bold dim")

        if my_id:
            t.append(" 🦆 ", style="bold")
            t.append(my_id[:12], style="bold bright_yellow")
            t.append(" me\n", style="dim")

        for pid, info in peers.items():
            age = time.time() - info.get("last_seen", 0)
            t.append(" 🦆 ", style="")
            t.append(pid[:12], style="cyan")
            t.append(f" {age:.0f}s\n", style="dim")

        if not peers and not my_id:
            t.append(" No peers\n", style="dim italic")

        w.update(t)

    def _update_sb_net_feed(self) -> None:
        """Network activity feed — ring context + log tail."""
        try:
            w = self.query_one("#sb-net-feed", Static)
        except NoMatches:
            return

        t = Text()
        t.append("── Network Feed ──\n", style="bold dim")

        entries_shown = 0
        max_entries = 8

        # 1. Ring context (mesh activity from all agents)
        try:
            mesh_mod = sys.modules.get("devduck.tools.unified_mesh")
            if mesh_mod:
                ring = mesh_mod.MESH_STATE.get("ring_context", [])
                # Show newest entries
                recent = ring[-max_entries:]
                for entry in recent:
                    ts = entry.get("timestamp", 0)
                    agent_id = entry.get("agent_id", "?")
                    text = entry.get("text", "")
                    agent_type = entry.get("agent_type", "")

                    # Format timestamp
                    if ts:
                        time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                    else:
                        time_str = "?"

                    # Pick icon based on source
                    source = entry.get("metadata", {}).get("source", "")
                    if "telegram" in agent_id.lower() or "telegram" in source:
                        icon = "📱"
                    elif "whatsapp" in agent_id.lower() or "whatsapp" in source:
                        icon = "💬"
                    elif "browser" in agent_id.lower() or "ws" in source:
                        icon = "🌐"
                    elif "zenoh" in agent_type:
                        icon = "🦆"
                    else:
                        icon = "→"

                    # Compact display
                    t.append(f" {icon} ", style="dim")
                    t.append(f"{time_str} ", style="dim")
                    # Agent name shortened
                    short_agent = agent_id.split(":")[-1][:10]
                    t.append(f"{short_agent}\n", style="bold cyan" if agent_type != "local" else "dim")
                    # Text preview (very short for sidebar)
                    preview = text[:35].replace("\n", " ")
                    if len(text) > 35:
                        preview += "…"
                    t.append(f"   {preview}\n", style="dim")
                    entries_shown += 1

                # Track if new entries arrived
                new_count = len(ring)
                if new_count > self._ring_last_count and self._ring_last_count > 0:
                    diff = new_count - self._ring_last_count
                    t.append(f" ✦ {diff} new event(s)\n", style="bold bright_yellow")
                self._ring_last_count = new_count
        except Exception:
            pass

        # 2. Tail recent log lines for network events
        try:
            from devduck import LOG_FILE
            if LOG_FILE.exists():
                with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(0, 2)  # end
                    size = f.tell()
                    # Read last 4KB
                    read_from = max(0, size - 4096)
                    f.seek(read_from)
                    tail = f.read()

                # Filter for network-related log lines
                net_keywords = ["peer", "zenoh", "mesh", "telegram", "whatsapp", "websocket", "proxy", "browser", "ring"]
                for line in tail.strip().split("\n")[-6:]:
                    line_lower = line.lower()
                    if any(kw in line_lower for kw in net_keywords):
                        # Extract just the message part
                        parts = line.split(" - ", 3)
                        if len(parts) >= 4:
                            ts_part = parts[0].split(",")[0].split(" ")[-1]  # HH:MM:SS
                            msg = parts[3][:40]
                            level = parts[2].strip()
                            lvl_style = "yellow" if level == "WARNING" else "red" if level == "ERROR" else "dim"
                            t.append(f" ‣ ", style=lvl_style)
                            t.append(f"{ts_part} ", style="dim")
                            t.append(f"{msg}\n", style=lvl_style)
                            entries_shown += 1
                            if entries_shown >= max_entries + 4:
                                break
        except Exception:
            pass

        if entries_shown == 0:
            t.append(" No network activity yet\n", style="dim italic")
            t.append(" Events from zenoh, mesh,\n", style="dim")
            t.append(" telegram, whatsapp appear\n", style="dim")
            t.append(" here in real-time\n", style="dim")

        w.update(t)

    def _update_sb_stats(self) -> None:
        """Stats section at bottom."""
        try:
            w = self.query_one("#sb-stats", Static)
        except NoMatches:
            return

        active = sum(1 for p in self._active_conversations.values() if not p.is_done)
        done = sum(1 for p in self._active_conversations.values() if p.is_done)

        t = Text()
        t.append("── Stats ──\n", style="bold dim")
        t.append(" ⚡ ", style="yellow" if active else "dim")
        t.append(f"{active} running  ", style="bold yellow" if active else "dim")
        t.append("✓ ", style="green" if done else "dim")
        t.append(f"{done} done\n", style="bold green" if done else "dim")
        t.append(f" 📊 {self._total_queries} total", style="dim")
        t.append(f"  ⏰ {datetime.now().strftime('%H:%M')}\n", style="dim")

        # Ambient indicator
        dd = self._devduck
        if dd and dd.ambient and dd.ambient.running:
            if dd.ambient.autonomous:
                t.append(" 🚀 autonomous", style="bold bright_magenta")
                t.append(f" {dd.ambient.ambient_iterations}/{dd.ambient.autonomous_max_iterations}\n", style="dim")
            else:
                t.append(" 🌙 ambient", style="bright_yellow")
                t.append(f" {dd.ambient.ambient_iterations}/{dd.ambient.max_iterations}\n", style="dim")

        # Recording indicator
        try:
            from devduck import get_session_recorder
            rec = get_session_recorder()
            if rec and rec.recording:
                dur = time.time() - rec.start_time if rec.start_time else 0
                t.append(f" 🎬 recording {dur:.0f}s\n", style="bold red")
        except ImportError:
            pass

        w.update(t)

    # ── Input handling ──────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return

        event.input.value = ""

        # Record in the input widget's history for up/down recall
        try:
            input_widget = self.query_one("#query-input", HistoryInput)
            input_widget.record_input(query)
            input_widget._history_index = -1  # Reset history browsing
            input_widget.placeholder = input_widget._default_placeholder()
        except NoMatches:
            pass

        # Exit commands
        if query.lower() in ("exit", "quit", "q", "/quit", "/exit"):
            self.exit()
            return

        # Slash commands
        if query.startswith("/"):
            self._handle_slash_command(query)
            return

        # Shell commands with ! prefix — run directly without agent
        if query.startswith("!"):
            self._run_shell_command(query[1:].strip())
            return

        # Ambient mode toggle
        if query.lower() == "ambient":
            self._toggle_ambient(autonomous=False)
            return

        # Autonomous mode toggle
        if query.lower() in ("auto", "autonomous"):
            self._toggle_ambient(autonomous=True)
            return

        # Recording toggle
        if query.lower() == "record":
            self._toggle_recording()
            return

        # Remove welcome panel
        try:
            self.query_one("#welcome-panel").remove()
        except NoMatches:
            pass

        # Create conversation
        self._conv_counter += 1
        self._total_queries += 1
        conv_id = self._conv_counter
        color = self._get_next_color()

        panel = ConversationPanel(conv_id=conv_id, query=query, color=color)
        self._active_conversations[conv_id] = panel

        scroll = self.query_one("#conversations-scroll", ScrollableContainer)
        scroll.mount(panel)
        scroll.scroll_end(animate=False)

        self._run_conversation(conv_id, query)
        self._update_status_bar()
        self._update_sidebar_stats()

    # ── Shell commands (!prefix) ────────────────────────────────

    def _run_shell_command(self, cmd: str) -> None:
        """Run a shell command directly via the shell tool (no agent)."""
        if not cmd:
            return

        # Remove welcome
        try:
            self.query_one("#welcome-panel").remove()
        except NoMatches:
            pass

        # Create a panel for the shell output
        self._conv_counter += 1
        self._total_queries += 1
        conv_id = self._conv_counter
        color = "#d19a66"  # orange for shell

        panel = ConversationPanel(conv_id=conv_id, query=f"! {cmd}", color=color)
        self._active_conversations[conv_id] = panel

        scroll = self.query_one("#conversations-scroll", ScrollableContainer)
        scroll.mount(panel)
        scroll.scroll_end(animate=False)

        self._run_shell_worker(conv_id, cmd)
        self._update_status_bar()

    @work(thread=True)
    def _run_shell_worker(self, conv_id: int, cmd: str) -> None:
        """Execute shell command in background thread."""
        try:
            if not self._devduck or not self._devduck.agent:
                self.post_message(ConversationDone(conv_id, "Agent not available"))
                return

            self.post_message(ToolEvent(conv_id, "shell", "start"))
            # Disable pagers for TUI — git diff, man, etc. hang without a real TTY.
            # Set as env vars so they don't appear in the command display.
            pager_env = "GIT_PAGER=cat PAGER=cat DELTA_PAGER=cat BAT_PAGER=cat LESS= "
            result = self._devduck.agent.tool.shell(
                command=f"{pager_env}{cmd}",
                timeout=9000,
                non_interactive=True,
            )
            self.post_message(ToolEvent(conv_id, "shell", "success"))

            # Parse shell result — extract just the command output, strip metadata.
            # Shell tool returns content like:
            #   [0] "Execution Summary:\nTotal commands: 1\nSuccessful: 1\nFailed: 0"
            #   [1] "Command: ...\nStatus: success\nExit Code: 0\nOutput: <actual>\n\nError: <stderr>"
            raw_parts = []
            if result and "content" in result:
                for item in result["content"]:
                    if isinstance(item, dict) and "text" in item:
                        raw_parts.append(item["text"])

            raw_text = "\n".join(raw_parts)
            cmd_output = ""
            cmd_error = ""

            if "\nOutput:" in raw_text:
                # Extract between "Output:" and "\nError:" (or end)
                after_output = raw_text.split("\nOutput:", 1)[1]
                if "\nError:" in after_output:
                    cmd_output, cmd_error = after_output.split("\nError:", 1)
                else:
                    cmd_output = after_output
                cmd_output = cmd_output.strip()
                cmd_error = cmd_error.strip()
            elif "Output:" in raw_text:
                # "Output:" at start of a part
                after_output = raw_text.split("Output:", 1)[1]
                if "\nError:" in after_output:
                    cmd_output, cmd_error = after_output.split("\nError:", 1)
                else:
                    cmd_output = after_output
                cmd_output = cmd_output.strip()
                cmd_error = cmd_error.strip()
            else:
                # No structured format — show everything
                cmd_output = raw_text.strip()

            # Render clean output
            if cmd_output:
                self.post_message(StreamChunk(conv_id, f"\n```\n{cmd_output}\n```\n"))
            elif not cmd_error:
                self.post_message(StreamChunk(conv_id, "\n*(no output)*\n"))

            if cmd_error:
                self.post_message(StreamChunk(conv_id, f"\n**stderr:**\n```\n{cmd_error}\n```\n"))

            # Inject shell result into shared messages so agents have context.
            # Format as a user→assistant pair so the model sees what happened.
            display_output = cmd_output or "(no output)"
            if cmd_error:
                display_output += f"\nstderr: {cmd_error}"
            self._shared_messages.append({
                "role": "user",
                "content": [{"text": f"[Shell command]: {cmd}"}],
            })
            self._shared_messages.append({
                "role": "assistant",
                "content": [{"text": f"[Shell output]:\n{display_output}"}],
            })

            self.post_message(ConversationDone(conv_id))

            # Save to history
            try:
                from devduck import append_to_shell_history
                append_to_shell_history(f"! {cmd}", cmd_output or raw_text)
            except Exception:
                pass

        except Exception as e:
            self.post_message(ConversationDone(conv_id, str(e)[:300]))

    # ── Ambient / Autonomous mode ───────────────────────────────

    def _toggle_ambient(self, autonomous: bool = False) -> None:
        """Toggle ambient or autonomous mode."""
        scroll = self.query_one("#conversations-scroll", ScrollableContainer)

        try:
            from devduck import AmbientMode
        except ImportError:
            scroll.mount(Static(Panel("[red]AmbientMode not available[/]", border_style="red")))
            scroll.scroll_end(animate=False)
            return

        dd = self._devduck
        if not dd:
            return

        if autonomous:
            if dd.ambient and dd.ambient.autonomous:
                dd.ambient.stop()
                msg = "🌙 Autonomous mode **disabled**"
            elif dd.ambient and dd.ambient.running:
                dd.ambient.start(autonomous=True)
                msg = "🚀 Switched to **AUTONOMOUS** mode — agent works until `[AMBIENT_DONE]`"
            else:
                if not dd.ambient:
                    dd.ambient = AmbientMode(dd)
                dd.ambient.start(autonomous=True)
                msg = "🚀 **AUTONOMOUS** mode enabled — agent works continuously until done"
        else:
            if dd.ambient and dd.ambient.running:
                dd.ambient.stop()
                msg = "🌙 Ambient mode **disabled**"
            else:
                if not dd.ambient:
                    dd.ambient = AmbientMode(dd)
                dd.ambient.start()
                msg = "🌙 Ambient mode **enabled** (thinks in background when idle)"

        scroll.mount(Static(Panel(Markdown(msg), border_style="bright_yellow", padding=(0, 1))))
        scroll.scroll_end(animate=False)
        self._update_status_bar()

    # ── Session recording ───────────────────────────────────────

    def _toggle_recording(self) -> None:
        """Toggle session recording."""
        scroll = self.query_one("#conversations-scroll", ScrollableContainer)

        try:
            from devduck import get_session_recorder, start_recording, stop_recording
        except ImportError:
            scroll.mount(Static(Panel("[red]Recording not available[/]", border_style="red")))
            scroll.scroll_end(animate=False)
            return

        recorder = get_session_recorder()
        if recorder and recorder.recording:
            export_path = stop_recording()
            if self._devduck:
                self._devduck._recording = False
            msg = f"🎬 Recording **stopped** and exported:\n`{export_path}`"
        else:
            start_recording()
            if self._devduck:
                self._devduck._recording = True
            msg = "🎬 Recording **started** — type `record` again to stop and export"

        scroll.mount(Static(Panel(Markdown(msg), border_style="bright_magenta", padding=(0, 1))))
        scroll.scroll_end(animate=False)

    # ── Slash commands ──────────────────────────────────────────

    def _handle_slash_command(self, cmd: str) -> None:
        """Handle /commands without spawning an agent."""
        cmd_lower = cmd.lower().strip()
        scroll = self.query_one("#conversations-scroll", ScrollableContainer)

        if cmd_lower in ("/help", "/h", "/?"):
            self._show_welcome()
            scroll.scroll_end(animate=False)

        elif cmd_lower in ("/clear", "/cl"):
            self.action_clear_done()

        elif cmd_lower in ("/clearall", "/ca"):
            self.action_clear_all()

        elif cmd_lower in ("/status", "/s"):
            self._show_status()

        elif cmd_lower in ("/peers", "/p"):
            self._show_peers()

        elif cmd_lower in ("/tools", "/t"):
            self._show_tools()

        elif cmd_lower in ("/sidebar", "/sb"):
            self.action_toggle_sidebar()

        elif cmd_lower in ("/ambient", "/am"):
            self._toggle_ambient(autonomous=False)

        elif cmd_lower in ("/auto", "/autonomous"):
            self._toggle_ambient(autonomous=True)

        elif cmd_lower in ("/record", "/rec"):
            self._toggle_recording()

        elif cmd_lower in ("/voice", "/v", "/speech"):
            self._show_voice_config()

        elif cmd_lower.startswith("/voice ") or cmd_lower.startswith("/v "):
            # /voice novasonic, /voice openai, /voice gemini_live, /voice stop
            parts = cmd.strip().split(None, 1)
            if len(parts) > 1:
                arg = parts[1].strip().lower()
                if arg in ("stop", "off", "end"):
                    self._toggle_voice(force_stop=True)
                elif arg in ("novasonic", "openai", "gemini_live", "gemini"):
                    provider = "gemini_live" if arg == "gemini" else arg
                    self._speech_provider = provider
                    self._toggle_voice(force_start=True)
                else:
                    self._show_voice_config()

        elif cmd_lower.startswith("/schedule") or cmd_lower.startswith("/sched"):
            self._show_schedule_help()

        elif cmd_lower in ("/logs", "/log"):
            self._show_logs()

        else:
            scroll.mount(
                Static(
                    Panel(
                        f"[dim]Unknown command:[/] [bold]{cmd}[/]\n"
                        f"[dim]Try /help for available commands[/]",
                        border_style="yellow",
                    )
                )
            )
            scroll.scroll_end(animate=False)

    def _show_logs(self) -> None:
        """Show recent network/system logs in a panel."""
        scroll = self.query_one("#conversations-scroll", ScrollableContainer)

        lines = ["## 📋 Recent Logs\n"]

        try:
            from devduck import LOG_FILE
            if LOG_FILE.exists():
                with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                    all_lines = f.readlines()

                # Last 40 lines, filtered for interesting stuff
                recent = all_lines[-60:]
                net_keywords = ["peer", "zenoh", "mesh", "telegram", "whatsapp",
                                "websocket", "proxy", "browser", "ring", "agentcore",
                                "scheduler", "ambient", "recording", "tool", "error", "warning"]

                shown = 0
                for line in recent:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    line_lower = line_stripped.lower()
                    if any(kw in line_lower for kw in net_keywords):
                        lines.append(f"    {line_stripped}")
                        shown += 1
                        if shown >= 30:
                            break

                if shown == 0:
                    lines.append("*No network-related log entries found.*")
            else:
                lines.append("*Log file not found.*")
        except Exception as e:
            lines.append(f"*Error reading logs: {e}*")

        scroll.mount(Static(Panel(Markdown("\n".join(lines)), border_style="dim")))
        scroll.scroll_end(animate=False)


    def _show_schedule_help(self) -> None:
        """Show schedule management help."""
        scroll = self.query_one("#conversations-scroll", ScrollableContainer)

        md = (
            "## ⏰ Scheduler\n\n"
            "Manage scheduled jobs directly — just ask the agent:\n\n"
            "```\n"
            "schedule a job called 'backup' to run every hour: git status\n"
            "list all scheduled jobs\n"
            "disable the backup job\n"
            "run the backup job now\n"
            "remove the backup job\n"
            "```\n\n"
            "Or use the scheduler tool directly:\n"
            "- `scheduler(action='add', name='test', schedule='*/5 * * * *', prompt='echo hi')`\n"
            "- `scheduler(action='list')`\n"
            "- `scheduler(action='enable', name='test')` / `disable`\n"
            "- `scheduler(action='run_now', name='test')`\n"
            "- `scheduler(action='remove', name='test')`\n"
            "- `scheduler(action='history')`\n\n"
            "Active jobs appear in the sidebar → **Schedules** section."
        )

        scroll.mount(Static(Panel(Markdown(md), border_style="bright_yellow")))
        scroll.scroll_end(animate=False)

    def _show_status(self) -> None:
        scroll = self.query_one("#conversations-scroll", ScrollableContainer)
        status_info = self._devduck.status() if self._devduck else {}

        active = sum(1 for p in self._active_conversations.values() if not p.is_done)
        done = sum(1 for p in self._active_conversations.values() if p.is_done)

        # Ambient state
        dd = self._devduck
        ambient_str = "off"
        if dd and dd.ambient and dd.ambient.running:
            if dd.ambient.autonomous:
                ambient_str = f"🚀 autonomous ({dd.ambient.ambient_iterations}/{dd.ambient.autonomous_max_iterations})"
            else:
                ambient_str = f"🌙 standard ({dd.ambient.ambient_iterations}/{dd.ambient.max_iterations})"

        # Recording state
        rec_str = "off"
        try:
            from devduck import get_session_recorder
            rec = get_session_recorder()
            if rec and rec.recording:
                dur = time.time() - rec.start_time if rec.start_time else 0
                rec_str = f"🎬 recording ({rec.event_buffer.count} events, {dur:.0f}s)"
        except ImportError:
            pass

        md = (
            f"## 📊 Status\n\n"
            f"| | |\n"
            f"|---|---|\n"
            f"| **Model** | `{status_info.get('model', '?')}` |\n"
            f"| **Tools** | {status_info.get('tools', 0)} |\n"
            f"| **Active** | {active} |\n"
            f"| **Done** | {done} |\n"
            f"| **Total** | {self._total_queries} |\n"
            f"| **Peers** | {self._get_peer_count()} |\n"
            f"| **Ambient** | {ambient_str} |\n"
            f"| **Recording** | {rec_str} |\n"
        )

        scroll.mount(Static(Panel(Markdown(md), border_style="cyan")))
        scroll.scroll_end(animate=False)

    def _show_peers(self) -> None:
        scroll = self.query_one("#conversations-scroll", ScrollableContainer)

        peers = {}
        my_id = ""
        if not self._zenoh_checked:
            self._zenoh_mod = sys.modules.get("devduck.tools.zenoh_peer")
            self._zenoh_checked = True
        if self._zenoh_mod:
            try:
                zs = self._zenoh_mod.ZENOH_STATE
                peers = zs.get("peers", {})
                my_id = zs.get("instance_id", "")
            except Exception:
                pass

        lines = [f"## 🌐 Zenoh Peers\n", f"**My ID:** `{my_id}`\n"]
        if peers:
            lines.append("| Peer | Age |")
            lines.append("|------|-----|")
            for pid, info in peers.items():
                age = time.time() - info.get("last_seen", 0)
                lines.append(f"| `{pid}` | {age:.0f}s |")
        else:
            lines.append("*No peers connected*")

        scroll.mount(Static(Panel(Markdown("\n".join(lines)), border_style="cyan")))
        scroll.scroll_end(animate=False)

    def _show_tools(self) -> None:
        scroll = self.query_one("#conversations-scroll", ScrollableContainer)

        if not self._devduck:
            scroll.mount(Static(Panel("[dim]Agent not available[/]", border_style="red")))
            return

        tool_names = []
        if hasattr(self._devduck, 'agent') and self._devduck.agent:
            try:
                tool_names = sorted(self._devduck.agent.tool_names)
            except Exception:
                pass

        if not tool_names:
            tool_names = [f"tool_{i}" for i in range(len(self._devduck.tools))]

        lines = [f"## 🔧 Tools ({len(tool_names)})\n"]
        # Render as compact grid
        row = []
        for name in tool_names:
            icon = TOOL_ICONS.get(name, "🔧")
            row.append(f"{icon} `{name}`")
            if len(row) == 3:
                lines.append(" · ".join(row))
                row = []
        if row:
            lines.append(" · ".join(row))

        scroll.mount(Static(Panel(Markdown("\n".join(lines)), border_style="cyan")))
        scroll.scroll_end(animate=False)

    # ── Agent runner ────────────────────────────────────────────

    @work(thread=True)
    def _run_conversation(self, conv_id: int, query: str) -> None:
        try:
            if not self._devduck or not self._devduck.agent:
                self.post_message(ConversationDone(conv_id, "Agent not available"))
                return

            tui_handler = TUICallbackHandler(self, conv_id)
            original_query = query

            # Create a FRESH Agent per conversation for true concurrency.
            # Each agent has its own callback handler (no conflicts), but ALL agents
            # share the SAME SharedMessages list as their .messages. This means:
            #   - Messages from conv1 are visible to conv2 in real-time
            #   - Ordering is preserved by the SharedMessages lock
            #   - No snapshot/merge needed — it's live
            from strands import Agent

            tools = list(self._devduck.tools)
            try:
                from devduck.tools.tui import tui as tui_tool
                tools.append(tui_tool)
            except ImportError:
                pass

            conv_agent = Agent(
                model=self._devduck.agent_model,
                tools=tools,
                system_prompt=self._devduck._build_system_prompt(),
                callback_handler=tui_handler,
                load_tools_from_directory=True,
            )

            # Point this agent's messages at the shared thread-safe list
            conv_agent.messages = self._shared_messages

            # 🔒 Mark agent as executing to prevent hot-reload interruption
            dd = self._devduck
            if dd:
                dd._agent_executing = True

            # 🎬 Record user query if recording active
            try:
                from devduck import get_session_recorder
                recorder = get_session_recorder()
                if recorder and recorder.recording:
                    recorder.record_agent_message("user", query)
                    recorder.snapshot(conv_agent, "before_tui_call", last_query=query)
            except Exception:
                recorder = None

            # 🌙 Inject ambient result if available
            if dd and dd.ambient:
                ambient_result = dd.ambient.get_and_clear_result()
                if ambient_result:
                    self.post_message(StreamChunk(conv_id, "\n🌙 *Injecting ambient background work…*\n"))
                    query = f"{ambient_result}\n\n[New user query]:\n{query}"

            # 📚 Knowledge Base Retrieval (BEFORE agent runs)
            knowledge_base_id = os.getenv("DEVDUCK_KNOWLEDGE_BASE_ID")
            if knowledge_base_id and conv_agent:
                try:
                    if "retrieve" in conv_agent.tool_names:
                        conv_agent.tool.retrieve(
                            text=query, knowledgeBaseId=knowledge_base_id
                        )
                except Exception:
                    pass

            # 🔗 Inject dynamic context (zenoh + ring + ambient + recording events)
            try:
                from devduck import (
                    get_zenoh_peers_context, get_unified_ring_context,
                    get_ambient_status_context,
                )
                zenoh_ctx = get_zenoh_peers_context()
                ring_ctx = get_unified_ring_context()
                ambient_ctx = get_ambient_status_context()

                # 🎬 Inject recent recorded events
                recording_ctx = ""
                if recorder and recorder.recording:
                    recording_ctx = recorder.event_buffer.get_recent_context(
                        seconds=10.0, max_events=15
                    )

                dynamic_context = zenoh_ctx + ring_ctx + ambient_ctx + recording_ctx
                if dynamic_context:
                    query = f"[Dynamic Context]{dynamic_context}\n\n[User Query]\n{query}"
            except ImportError:
                pass
            except Exception:
                pass

            # Run query on the per-conversation agent
            try:
                result = conv_agent(query)
            except Exception as e:
                # Context window overflow — clear shared history and retry
                error_str = str(e).lower()
                if any(kw in error_str for kw in ("trim conversation", "context window", "too many tokens", "input is too long")):
                    self.post_message(StreamChunk(conv_id, "\n⚠️ Context overflow — clearing shared history and retrying…\n"))
                    self._shared_messages.clear()
                    result = conv_agent(original_query)
                else:
                    raise

            self.post_message(ConversationDone(conv_id))

            # Trim shared history to prevent unbounded growth / context overflow.
            # Messages were already added live by the agent — no merge needed.
            max_shared = int(os.getenv("DEVDUCK_TUI_MAX_SHARED_MESSAGES", "100"))
            self._shared_messages.trim_to(max_shared)

            # 🎬 Record agent response if recording active
            try:
                if recorder and recorder.recording:
                    recorder.record_agent_message("assistant", str(result)[:2000])
                    recorder.snapshot(
                        conv_agent,
                        "after_tui_call",
                        last_query=original_query,
                        last_result=str(result)[:5000],
                    )
            except Exception:
                pass

            # 🌙 Record interaction for ambient mode
            if dd and dd.ambient:
                dd.ambient.record_interaction(original_query, result)

            # 🔗 Push to unified mesh ring (bidirectional sync)
            try:
                from devduck.tools.unified_mesh import add_to_ring
                result_preview = str(result)
                add_to_ring(
                    "local:devduck-tui",
                    "local",
                    f"Q: {original_query} → {result_preview}",
                    {"source": "tui"},
                )
            except (ImportError, Exception):
                pass

            # 💾 Knowledge Base Storage (AFTER agent runs)
            if knowledge_base_id and conv_agent:
                try:
                    if "store_in_kb" in conv_agent.tool_names:
                        from datetime import datetime as _dt
                        conv_agent.tool.store_in_kb(
                            content=f"Input: {original_query}, Result: {result!s}",
                            title=f"DevDuck TUI: {_dt.now().strftime('%Y-%m-%d')} | {original_query[:500]}",
                            knowledge_base_id=knowledge_base_id,
                        )
                except Exception:
                    pass

            try:
                from devduck import append_to_shell_history
                append_to_shell_history(original_query, str(result))
            except Exception:
                pass

            # 🔒 Clear executing flag + check pending hot-reload
            if dd:
                dd._agent_executing = False
                if dd._reload_pending:
                    try:
                        from devduck import logger as _logger
                        _logger.info("Hot-reload pending but skipped in TUI mode")
                    except Exception:
                        pass

        except Exception as e:
            # 🔒 Reset executing flag on error
            if self._devduck:
                self._devduck._agent_executing = False
            # 🎬 Record error if recording active
            try:
                from devduck import get_session_recorder
                rec = get_session_recorder()
                if rec and rec.recording:
                    rec.record_agent_message("error", str(e))
            except Exception:
                pass
            self.post_message(ConversationDone(conv_id, str(e)[:300]))

    # ── Message handlers ────────────────────────────────────────

    def on_stream_chunk(self, event: StreamChunk) -> None:
        panel = self._active_conversations.get(event.conv_id)
        if panel:
            panel.append_text(event.text)

    def on_tool_event(self, event: ToolEvent) -> None:
        panel = self._active_conversations.get(event.conv_id)
        if panel:
            panel.append_tool_event(event.tool_name, event.status, event.detail)

    def on_conversation_done(self, event: ConversationDone) -> None:
        panel = self._active_conversations.get(event.conv_id)
        if panel:
            panel.mark_done(event.error)
            panel.add_class("error" if event.error else "done")
        self._update_status_bar()
        self._update_sidebar_stats()

    # ── Actions ─────────────────────────────────────────────────

    def action_focus_input(self) -> None:
        self.query_one("#query-input", HistoryInput).focus()

    def action_clear_done(self) -> None:
        to_remove = [cid for cid, p in self._active_conversations.items() if p.is_done]
        for cid in to_remove:
            self._active_conversations.pop(cid).remove()
        self._update_status_bar()
        self._update_sidebar_stats()

    def action_clear_all(self) -> None:
        for panel in self._active_conversations.values():
            panel.remove()
        self._active_conversations.clear()
        self._update_status_bar()
        self._update_sidebar_stats()

    def action_toggle_sidebar(self) -> None:
        self._sidebar_visible = not self._sidebar_visible
        try:
            sidebar = self.query_one("#sidebar")
            sidebar.display = self._sidebar_visible
        except NoMatches:
            pass

    # ── Voice / Speech-to-Speech ────────────────────────────────

    def action_toggle_voice(self) -> None:
        """Ctrl+V handler — toggle speech-to-speech session."""
        self._toggle_voice()

    def action_ptt_press(self) -> None:
        """Space press — open mic gate if voice session active and input not focused."""
        # Only activate PTT if input is NOT focused (so typing works normally)
        input_widget = self.query_one("#query-input", HistoryInput)
        if input_widget.has_focus:
            # Input is focused — let space type normally
            input_widget.insert_text_at_cursor(" ")
            return

        # PTT: open mic gate
        self._set_ptt(True)

    def on_key(self, event) -> None:
        """Handle key release for push-to-talk (Space release = mic off)."""
        # Textual doesn't have native key-release events, so we use a timer approach:
        # Every Space press resets a debounce timer. When the timer fires, mic closes.
        if event.key == "space":
            input_widget = self.query_one("#query-input", HistoryInput)
            if not input_widget.has_focus and self._speech_session_id:
                # Reset the PTT release timer
                if hasattr(self, "_ptt_timer") and self._ptt_timer is not None:
                    self._ptt_timer.stop()
                self._set_ptt(True)
                # Auto-release after 200ms of no space presses (simulates key release)
                self._ptt_timer = self.set_timer(0.3, self._ptt_release_callback)

    def _ptt_release_callback(self) -> None:
        """Called when PTT timer expires — close mic."""
        self._set_ptt(False)
        self._ptt_timer = None

    def _set_ptt(self, pressed: bool) -> None:
        """Control push-to-talk mic gate on active speech session."""
        if not self._speech_session_id:
            return
        try:
            from devduck.tools.speech_to_speech import _active_sessions, _session_lock
            with _session_lock:
                session = _active_sessions.get(self._speech_session_id)
                if session:
                    if pressed:
                        session.mic_on()
                    else:
                        session.mic_off()
        except (ImportError, Exception):
            pass

    def _toggle_voice(self, force_start: bool = False, force_stop: bool = False) -> None:
        """Toggle speech-to-speech on/off with visual feedback in TUI."""
        scroll = self.query_one("#conversations-scroll", ScrollableContainer)

        # Check if speech_to_speech is available
        try:
            from devduck.tools.speech_to_speech import (
                _active_sessions, _session_lock, _stop_speech_session,
                _start_speech_session, _list_audio_devices,
            )
        except ImportError:
            scroll.mount(Static(Panel(
                "[red]speech_to_speech tool not loaded.[/]\n"
                "[dim]Load it with: manage_tools(action='add', tools='devduck.tools.speech_to_speech')[/]",
                border_style="red",
                title="🎤 Voice Error",
            )))
            scroll.scroll_end(animate=False)
            return

        # Check current state
        with _session_lock:
            has_active = len(_active_sessions) > 0

        if (has_active and not force_start) or force_stop:
            # ── STOP ──
            self._stop_voice_session(scroll)
        else:
            # ── START ──
            self._start_voice_session(scroll)

    def _start_voice_session(self, scroll: ScrollableContainer) -> None:
        """Start a new speech-to-speech session with TUI feedback."""
        from datetime import datetime as dt

        # Remove welcome panel if present
        try:
            self.query_one("#welcome-panel").remove()
        except NoMatches:
            pass

        # Create a visual panel for the voice session
        self._conv_counter += 1
        conv_id = self._conv_counter
        self._speech_panel_id = conv_id
        color = "#e06c75"  # red for voice

        panel = ConversationPanel(
            conv_id=conv_id,
            query=f"🎙️ Voice Session ({self._speech_provider})",
            color=color,
        )
        self._active_conversations[conv_id] = panel
        scroll.mount(panel)
        scroll.scroll_end(animate=False)

        # Generate session ID
        session_id = f"tui_voice_{dt.now().strftime('%H%M%S')}"
        self._speech_session_id = session_id

        # Show starting message
        panel.append_text(
            f"**Starting speech-to-speech session...**\n\n"
            f"- **Provider:** `{self._speech_provider}`\n"
            f"- **Session:** `{session_id}`\n"
            f"- **Hold Space** to talk, release to listen\n"
            f"- **Ctrl+V** to stop\n\n"
            f"🎤 Hold Space and speak...\n"
        )
        panel.append_tool_event("speech_to_speech", "start")

        # Start in background thread
        self._start_voice_worker(conv_id, session_id, self._speech_provider)
        self._update_status_bar()
        self._update_sidebar_stats()

    @work(thread=True)
    def _start_voice_worker(self, conv_id: int, session_id: str, provider: str) -> None:
        """Background worker to start the speech session."""
        try:
            from devduck.tools.speech_to_speech import _start_speech_session

            # Get parent agent for tool inheritance
            parent_agent = None
            if self._devduck and hasattr(self._devduck, "agent"):
                parent_agent = self._devduck.agent

            # Build a concise system prompt for voice (token-efficient)
            voice_system_prompt = (
                "You are DevDuck, a helpful AI voice assistant. "
                "Keep responses brief and conversational. "
                "You have access to tools - use them when needed. "
                "When the user says 'stop' or 'goodbye', use the speech_session tool to stop."
            )

            # Transcript callback → routes to TUI conversation panel
            def on_transcript(role: str, text: str, is_final: bool) -> None:
                if not text.strip():
                    return
                prefix = "🗣️" if role == "user" else "🤖" if role == "assistant" else "⚡"
                marker = "" if is_final else " ..."
                self.post_message(StreamChunk(conv_id, f"\n{prefix} **{role}**: {text}{marker}\n"))

            result = _start_speech_session(
                provider=provider,
                system_prompt=voice_system_prompt,
                session_id=session_id,
                model_settings=None,
                tool_names=None,  # Inherit all tools
                parent_agent=parent_agent,
                load_history_from=None,
                inherit_system_prompt=False,  # Use concise prompt for voice
                input_device_index=None,
                output_device_index=None,
                push_to_talk=True,  # TUI uses hold-Space PTT
                echo_cancellation=True,
                noise_suppression=True,
                transcript_callback=on_transcript,
            )

            # Report result
            if "✅" in result:
                self.post_message(StreamChunk(conv_id, f"\n✅ **Voice session active!**\n\n"))
                self.post_message(ToolEvent(conv_id, "speech_to_speech", "success"))

                # Start monitoring the session for auto-cleanup
                self._monitor_voice_session(conv_id, session_id)
            else:
                self.post_message(StreamChunk(conv_id, f"\n{result}\n"))
                self.post_message(ConversationDone(conv_id, "Failed to start"))
                self._speech_session_id = None
                self._speech_panel_id = None

        except Exception as e:
            self.post_message(StreamChunk(conv_id, f"\n❌ Error: {e}\n"))
            self.post_message(ConversationDone(conv_id, str(e)[:200]))
            self._speech_session_id = None
            self._speech_panel_id = None

    @work(thread=True)
    def _monitor_voice_session(self, conv_id: int, session_id: str) -> None:
        """Monitor the voice session and update TUI when it ends."""
        import time as _time

        try:
            from devduck.tools.speech_to_speech import _active_sessions, _session_lock

            while True:
                _time.sleep(2)
                with _session_lock:
                    session = _active_sessions.get(session_id)
                    if not session or not session.active:
                        break

            # Session ended (either by voice command or externally)
            self.post_message(StreamChunk(conv_id, "\n\n🎤 **Voice session ended.**\n"))
            self.post_message(ConversationDone(conv_id))

            # Try to show conversation history summary
            try:
                from devduck.tools.speech_to_speech import HISTORY_DIR
                history_file = HISTORY_DIR / f"{session_id}.json"
                if history_file.exists():
                    import json
                    with open(history_file, "r") as f:
                        data = json.load(f)
                    msg_count = len(data.get("messages", []))
                    self.post_message(StreamChunk(
                        conv_id,
                        f"📝 **Transcript saved:** {msg_count} messages → `{history_file}`\n"
                    ))
            except Exception:
                pass

            self._speech_session_id = None
            self._speech_panel_id = None
            self.call_from_thread(self._update_status_bar)
            self.call_from_thread(self._update_sidebar_stats)

        except Exception as e:
            self.post_message(ConversationDone(conv_id, f"Monitor error: {e}"))
            self._speech_session_id = None
            self._speech_panel_id = None

    def _stop_voice_session(self, scroll: ScrollableContainer) -> None:
        """Stop the active speech-to-speech session."""
        try:
            from devduck.tools.speech_to_speech import _stop_speech_session

            session_id = self._speech_session_id
            result = _stop_speech_session(session_id)

            # Update the voice panel if it exists
            if self._speech_panel_id and self._speech_panel_id in self._active_conversations:
                panel = self._active_conversations[self._speech_panel_id]
                panel.append_text(f"\n\n🛑 **Session stopped.**\n{result}\n")
                panel.mark_done()
                panel.add_class("done")
            else:
                # Show inline confirmation
                scroll.mount(Static(Panel(
                    Markdown(f"🛑 Voice session stopped.\n\n{result}"),
                    border_style="bright_red",
                    title="🎤 Voice",
                )))
                scroll.scroll_end(animate=False)

            self._speech_session_id = None
            self._speech_panel_id = None
            self._update_status_bar()
            self._update_sidebar_stats()

        except Exception as e:
            scroll.mount(Static(Panel(
                f"[red]Error stopping voice: {e}[/]",
                border_style="red",
            )))
            scroll.scroll_end(animate=False)

    def _show_voice_config(self) -> None:
        """Show voice configuration panel with provider options."""
        scroll = self.query_one("#conversations-scroll", ScrollableContainer)

        # Check current state
        active_info = ""
        try:
            from devduck.tools.speech_to_speech import _active_sessions, _session_lock
            with _session_lock:
                if _active_sessions:
                    for sid, session in _active_sessions.items():
                        active_info = f"\n\n🎙️ **Active Session:** `{sid}` — Ctrl+V to stop"
        except ImportError:
            pass

        md = (
            "## 🎤 Voice — Speech-to-Speech\n\n"
            f"**Current Provider:** `{self._speech_provider}`{active_info}\n\n"
            "### Quick Start\n"
            "- **Ctrl+V** — Toggle voice on/off\n"
            "- `/voice novasonic` — Switch to Nova Sonic (AWS)\n"
            "- `/voice openai` — Switch to OpenAI Realtime\n"
            "- `/voice gemini` — Switch to Gemini Live\n"
            "- `/voice stop` — Force stop session\n\n"
            "### Providers\n\n"
            "| Provider | Voices | Requires |\n"
            "|----------|--------|----------|\n"
            "| `novasonic` | tiffany, matthew, amy, ambre, florian | AWS credentials |\n"
            "| `openai` | coral, default | OPENAI_API_KEY |\n"
            "| `gemini_live` | Kore, default | GOOGLE_API_KEY |\n\n"
            "### Features\n"
            "- 🔧 **Full tool access** — voice agent inherits all tools\n"
            "- 💬 **Natural conversation** — VAD auto-interruption\n"
            "- 📝 **Auto-transcript** — saved to history after session\n"
            "- 🔄 **Background** — TUI stays responsive during voice\n"
        )

        scroll.mount(Static(Panel(Markdown(md), border_style="bright_red", title="🎤 Voice Config")))
        scroll.scroll_end(animate=False)

    def action_quit(self) -> None:
        # Stop active voice sessions
        try:
            from devduck.tools.speech_to_speech import _stop_speech_session
            _stop_speech_session(None)  # Stop all
        except (ImportError, Exception):
            pass
        try:
            from devduck.tools.tui import set_tui_app
            set_tui_app(None)
        except ImportError:
            pass
        self.exit()


# ─── Entry point ────────────────────────────────────────────────

def run_tui(devduck_instance=None):
    """Launch the DevDuck TUI."""
    if devduck_instance is None:
        from devduck import devduck as dd
        devduck_instance = dd

    app = DevDuckTUI(devduck_instance=devduck_instance)
    app.run()


if __name__ == "__main__":
    run_tui()
