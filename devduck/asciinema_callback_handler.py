"""
🎬 Asciinema Callback Handler for DevDuck

A drop-in replacement for callback_handler.py that:
1. Prints everything to the console exactly like the original
2. Simultaneously records all output as asciicast v2 format (.cast files)

Play recordings with: asciinema play recording.cast
Share on: asciinema.org
Embed in docs, READMEs, blog posts - agent trajectories everywhere!

Usage:
    # In __init__.py, swap the import:
    # from devduck.asciinema_callback_handler import callback_handler
    
    # Or enable via env var:
    # DEVDUCK_ASCIINEMA=true devduck

    # Recordings saved to: /tmp/devduck/casts/
    # Custom dir: DEVDUCK_CAST_DIR=/path/to/casts
"""

import json
import os
import shutil
import sys
import time
import threading
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from colorama import Fore, Style, init
from halo import Halo

# Initialize Colorama
init(autoreset=True)

# Reuse spinner config from original
SPINNERS = {
    "dots": {
        "interval": 80,
        "frames": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    }
}

TOOL_COLORS = {
    "running": Fore.GREEN,
    "success": Fore.GREEN,
    "error": Fore.RED,
    "info": Fore.CYAN,
}


# =============================================================================
# 🎬 Asciicast v2 Writer - Zero dependencies, pure JSONL
# =============================================================================

class AsciicastWriter:
    """Writes asciicast v2 format files (.cast).
    
    Format spec: https://docs.asciinema.org/manual/asciicast/v2/
    
    Line 1: Header JSON object
    Line 2+: Event arrays [timestamp, event_type, data]
    
    Event types:
        "o" - output (data written to terminal stdout)
        "i" - input (data read from terminal stdin)
    """

    def __init__(
        self,
        output_dir: str = None,
        width: int = None,
        height: int = None,
        title: str = None,
        idle_time_limit: float = 2.0,
    ):
        self.output_dir = Path(
            output_dir or os.getenv("DEVDUCK_CAST_DIR", "/tmp/devduck/casts")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect terminal size
        term_size = shutil.get_terminal_size((120, 40))
        self.width = width or term_size.columns
        self.height = height or term_size.lines

        self.title = title
        self.idle_time_limit = idle_time_limit

        # Recording state
        self._file = None
        self._filepath: Optional[Path] = None
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()
        self._event_count = 0
        self._recording = False

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def filepath(self) -> Optional[Path]:
        return self._filepath

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def duration(self) -> float:
        if self._start_time:
            return time.time() - self._start_time
        return 0.0

    def start(self, filename: str = None) -> Path:
        """Start a new recording session."""
        with self._lock:
            if self._recording:
                self.stop()

            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"devduck-{timestamp}"

            self._filepath = self.output_dir / f"{filename}.cast"
            self._start_time = time.time()
            self._event_count = 0
            self._recording = True

            # Open file and write header
            self._file = open(self._filepath, "w", encoding="utf-8")

            header = {
                "version": 2,
                "width": self.width,
                "height": self.height,
                "timestamp": int(self._start_time),
                "title": self.title or f"DevDuck Session - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "env": {
                    "SHELL": os.environ.get("SHELL", "/bin/zsh"),
                    "TERM": os.environ.get("TERM", "xterm-256color"),
                },
                "idle_time_limit": self.idle_time_limit,
            }

            self._file.write(json.dumps(header) + "\n")
            self._file.flush()

            return self._filepath

    def write_output(self, data: str):
        """Record terminal output event."""
        if not self._recording or not data:
            return

        with self._lock:
            if not self._file or not self._start_time:
                return

            offset = time.time() - self._start_time
            event = [round(offset, 6), "o", data]
            try:
                self._file.write(json.dumps(event) + "\n")
                self._file.flush()
                self._event_count += 1
            except (ValueError, IOError):
                pass

    def write_input(self, data: str):
        """Record terminal input event."""
        if not self._recording or not data:
            return

        with self._lock:
            if not self._file or not self._start_time:
                return

            offset = time.time() - self._start_time
            event = [round(offset, 6), "i", data]
            try:
                self._file.write(json.dumps(event) + "\n")
                self._file.flush()
                self._event_count += 1
            except (ValueError, IOError):
                pass

    def write_marker(self, label: str):
        """Write a visual marker/separator into the recording."""
        marker_text = f"\033[90m{'─' * 50} {label} {'─' * 10}\033[0m\r\n"
        self.write_output(marker_text)

    def stop(self) -> Optional[Path]:
        """Stop recording and close the file."""
        with self._lock:
            if not self._recording:
                return None

            self._recording = False
            filepath = self._filepath

            if self._file:
                try:
                    self._file.flush()
                    self._file.close()
                except (IOError, ValueError):
                    pass
                self._file = None

            return filepath

    def __del__(self):
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass


# =============================================================================
# 🛠️ ToolSpinner (with cast recording)
# =============================================================================

class ToolSpinner:
    def __init__(self, text: str = "", color: str = TOOL_COLORS["running"], cast_writer: AsciicastWriter = None):
        self.spinner = Halo(
            text=text,
            spinner=SPINNERS["dots"],
            color="green",
            text_color="green",
            interval=80,
        )
        self.color = color
        self.current_text = text
        self._cast = cast_writer

    def _record(self, text: str, prefix: str = ""):
        if self._cast:
            self._cast.write_output(f"{prefix}{text}\r\n")

    def start(self, text: str = None):
        if text:
            self.current_text = text
        print()
        self.spinner.start(f"{self.color}{self.current_text}{Style.RESET_ALL}")

    def update(self, text: str):
        self.current_text = text
        self.spinner.text = f"{self.color}{text}{Style.RESET_ALL}"

    def succeed(self, text: str = None):
        if text:
            self.current_text = text
        self.spinner.succeed(
            f"{TOOL_COLORS['success']}{self.current_text}{Style.RESET_ALL}"
        )
        self._record(self.current_text, prefix="✔ ")

    def fail(self, text: str = None):
        if text:
            self.current_text = text
        self.spinner.fail(f"{TOOL_COLORS['error']}{self.current_text}{Style.RESET_ALL}")
        self._record(self.current_text, prefix="✖ ")

    def info(self, text: str = None):
        if text:
            self.current_text = text
        self.spinner.info(f"{TOOL_COLORS['info']}{self.current_text}{Style.RESET_ALL}")
        self._record(self.current_text, prefix="ℹ ")

    def stop(self):
        self.spinner.stop()


# =============================================================================
# 🎬 AsciinemaCallbackHandler - Console + Recording
# =============================================================================

class AsciinemaCallbackHandler:
    """Callback handler that mirrors console output AND records asciicast.
    
    Key fix: asciicast virtual terminals need explicit \\r\\n (carriage return +
    line feed) for proper line breaks. Regular \\n alone causes the cursor to
    move down without returning to column 0, resulting in staircase/garbled text.
    
    The real terminal handles this via the tty driver (which translates \\n to
    \\r\\n in cooked mode), but asciicast replay is raw.
    """

    def __init__(self, auto_record: bool = True):
        self.thinking_spinner = None
        self.current_spinner = None
        self.current_tool = None
        self.tool_histories = {}

        # Track column position to know when we need \r\n vs just appending
        self._col = 0

        # 🎬 Asciicast writer
        self.cast_writer = AsciicastWriter()
        self._auto_record = auto_record

        if auto_record:
            cast_path = self.cast_writer.start()
            self.cast_writer.write_output(
                f"\033[1;33m🦆 DevDuck Session Recording\033[0m\r\n"
                f"\033[90m   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\033[0m\r\n\r\n"
            )

    @property
    def recording(self) -> bool:
        return self.cast_writer.recording

    @property
    def cast_path(self) -> Optional[Path]:
        return self.cast_writer.filepath

    def start_recording(self, filename: str = None) -> Path:
        return self.cast_writer.start(filename)

    def stop_recording(self) -> Optional[Path]:
        path = self.cast_writer.stop()
        if path:
            print(f"\n\033[33m🎬 Recording saved: {path}\033[0m")
            print(f"\033[90m   Play with: asciinema play {path}\033[0m")
            print(f"\033[90m   Upload:    asciinema upload {path}\033[0m")
        return path

    def _normalize_newlines(self, text: str) -> str:
        """Convert \\n to \\r\\n for proper asciicast rendering.
        
        Asciicast virtual terminals are raw — they need explicit CR+LF.
        Without \\r, the cursor moves down but stays at the same column,
        causing staircase/garbled output on replay.
        
        We avoid double-converting existing \\r\\n sequences.
        """
        # First, normalize any existing \r\n to just \n to avoid double-conversion
        text = text.replace("\r\n", "\n")
        # Then convert all \n to \r\n
        text = text.replace("\n", "\r\n")
        return text

    def _record_output(self, text: str, newline: bool = False):
        """Record text output to asciicast with proper line handling.
        
        Args:
            text: Raw text to record
            newline: If True, append \\r\\n after the text (mirrors print()'s default end='\\n')
        """
        if not text and not newline:
            return

        # Normalize any embedded newlines in the text itself
        normalized = self._normalize_newlines(text) if text else ""

        if newline:
            normalized += "\r\n"

        self.cast_writer.write_output(normalized)

        # Track column position for debugging
        if normalized:
            last_line = normalized.rsplit("\n", 1)[-1]
            if "\r" in last_line:
                self._col = len(last_line.rsplit("\r", 1)[-1])
            else:
                self._col += len(last_line)

    def _record_input(self, text: str):
        self.cast_writer.write_input(text)

    def callback_handler(self, **kwargs: Any) -> None:
        """Main callback - mirrors original behavior + records asciicast."""
        reasoningText = kwargs.get("reasoningText", False)
        data = kwargs.get("data", "")
        complete = kwargs.get("complete", False)
        force_stop = kwargs.get("force_stop", False)
        message = kwargs.get("message", {})
        current_tool_use = kwargs.get("current_tool_use", {})
        init_event_loop = kwargs.get("init_event_loop", False)
        start_event_loop = kwargs.get("start_event_loop", False)
        event_loop_throttled_delay = kwargs.get("event_loop_throttled_delay", None)
        console = kwargs.get("console", None)

        try:
            if self.thinking_spinner and (data or current_tool_use):
                self.thinking_spinner.stop()

            if init_event_loop:
                from rich.status import Status
                self.thinking_spinner = Status(
                    "[blue] retrieving memories...[/blue]",
                    spinner="dots",
                    console=console,
                )
                self.thinking_spinner.start()
                self._record_output("⠋ retrieving memories...", newline=True)

            if reasoningText:
                print(reasoningText, end="")
                self._record_output(reasoningText)

            if start_event_loop:
                self.thinking_spinner.update("[blue] thinking...[/blue]")
                self._record_output("⠋ thinking...", newline=True)
        except BaseException:
            pass

        if event_loop_throttled_delay and console:
            if self.current_spinner:
                self.current_spinner.stop()
            msg = f"Throttled! Waiting {event_loop_throttled_delay} seconds before retrying..."
            console.print(f"[red]{msg}[/red]")
            self._record_output(msg, newline=True)

        if force_stop:
            if self.thinking_spinner:
                self.thinking_spinner.stop()
            if self.current_spinner:
                self.current_spinner.stop()

        # Handle regular output - THE MAIN TEXT STREAM
        if data:
            if complete:
                # print() adds \n — mirror that as \r\n in cast
                print(f"{Fore.WHITE}{data}{Style.RESET_ALL}")
                self._record_output(data, newline=True)
            else:
                # Streaming token — no trailing newline from print, but the
                # data itself may contain \n chars (e.g. markdown tables).
                print(f"{Fore.WHITE}{data}{Style.RESET_ALL}", end="")
                self._record_output(data)

        # Handle tool input streaming
        if current_tool_use and current_tool_use.get("input"):
            tool_id = current_tool_use.get("toolUseId")
            tool_name = current_tool_use.get("name")
            tool_input = current_tool_use.get("input", "")

            if tool_id != self.current_tool:
                if self.current_spinner:
                    self.current_spinner.stop()

                self.current_tool = tool_id

                self.current_spinner = ToolSpinner(
                    f"🛠️  {tool_name}: Preparing...",
                    TOOL_COLORS["running"],
                    cast_writer=self.cast_writer,
                )
                self.current_spinner.start()

                self.cast_writer.write_marker(f"🛠️  {tool_name}")

                self.tool_histories[tool_id] = {
                    "name": tool_name,
                    "start_time": time.time(),
                    "input_size": 0,
                }

            if tool_id in self.tool_histories:
                current_size = len(tool_input)
                if current_size > self.tool_histories[tool_id]["input_size"]:
                    self.tool_histories[tool_id]["input_size"] = current_size
                    if self.current_spinner:
                        self.current_spinner.update(
                            f"🛠️  {tool_name}: {current_size} chars"
                        )

        # Process messages (tool results)
        if isinstance(message, dict):
            if message.get("role") == "assistant":
                for content in message.get("content", []):
                    if isinstance(content, dict):
                        tool_use = content.get("toolUse")
                        if tool_use:
                            tool_name = tool_use.get("name")
                            if self.current_spinner:
                                self.current_spinner.info(f"🔧 Starting {tool_name}...")

            elif message.get("role") == "user":
                for content in message.get("content", []):
                    if isinstance(content, dict):
                        tool_result = content.get("toolResult")
                        if tool_result:
                            tool_id = tool_result.get("toolUseId")
                            status = tool_result.get("status")

                            if tool_id in self.tool_histories:
                                tool_info = self.tool_histories[tool_id]
                                duration = round(
                                    time.time() - tool_info["start_time"], 2
                                )

                                if status == "success":
                                    msg = f"{tool_info['name']} completed in {duration}s"
                                else:
                                    msg = f"{tool_info['name']} failed after {duration}s"

                                if self.current_spinner:
                                    if status == "success":
                                        self.current_spinner.succeed(msg)
                                    else:
                                        self.current_spinner.fail(msg)

                                del self.tool_histories[tool_id]
                                self.current_spinner = None
                                self.current_tool = None


# =============================================================================
# 🦆 Module-level instances - drop-in replacement
# =============================================================================

_asciinema_enabled = os.getenv("DEVDUCK_ASCIINEMA", "false").lower() == "true"

if _asciinema_enabled:
    callback_handler_instance = AsciinemaCallbackHandler(auto_record=True)
    callback_handler = callback_handler_instance.callback_handler
else:
    from devduck.callback_handler import callback_handler_instance, callback_handler
