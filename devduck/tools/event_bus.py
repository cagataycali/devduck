"""🔔 Unified Event Bus for DevDuck.

Central event stream that ALL background tools push into:
- telegram: incoming/outgoing messages
- whatsapp: incoming/outgoing messages
- scheduler: job executions, completions
- tasks: task state changes, results
- listen: voice transcriptions
- notify: notifications sent
- zenoh: peer events
- speech: voice session events

The TUI sidebar reads from this bus for the live event feed.
The agent context builder reads from this for awareness injection.

Thread-safe, capped ring buffer, zero-config.
"""

import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from strands import tool

# ── Event Types ─────────────────────────────────────────────────
# Standardized event type constants for consistency

# Messaging
EVT_TELEGRAM_IN = "telegram.in"
EVT_TELEGRAM_OUT = "telegram.out"
EVT_WHATSAPP_IN = "whatsapp.in"
EVT_WHATSAPP_OUT = "whatsapp.out"

# Voice
EVT_LISTEN_TRANSCRIPT = "listen.transcript"
EVT_LISTEN_START = "listen.start"
EVT_LISTEN_STOP = "listen.stop"
EVT_SPEECH_START = "speech.start"
EVT_SPEECH_STOP = "speech.stop"
EVT_SPEECH_TRANSCRIPT = "speech.transcript"

# Scheduler & Tasks
EVT_SCHEDULE_FIRE = "schedule.fire"
EVT_SCHEDULE_DONE = "schedule.done"
EVT_SCHEDULE_ERROR = "schedule.error"
EVT_TASK_CREATE = "task.create"
EVT_TASK_DONE = "task.done"
EVT_TASK_ERROR = "task.error"
EVT_TASK_MESSAGE = "task.message"

# Notifications
EVT_NOTIFY = "notify"

# Network
EVT_ZENOH_PEER_JOIN = "zenoh.join"
EVT_ZENOH_PEER_LEAVE = "zenoh.leave"
EVT_ZENOH_MESSAGE = "zenoh.message"

# System
EVT_SYSTEM = "system"

# Icons for each event type (for TUI rendering)
EVENT_ICONS = {
    EVT_TELEGRAM_IN: "📱→",
    EVT_TELEGRAM_OUT: "📱←",
    EVT_WHATSAPP_IN: "💬→",
    EVT_WHATSAPP_OUT: "💬←",
    EVT_LISTEN_TRANSCRIPT: "🎤",
    EVT_LISTEN_START: "🎤▶",
    EVT_LISTEN_STOP: "🎤⏹",
    EVT_SPEECH_START: "🎙️▶",
    EVT_SPEECH_STOP: "🎙️⏹",
    EVT_SPEECH_TRANSCRIPT: "🎙️",
    EVT_SCHEDULE_FIRE: "⏰▶",
    EVT_SCHEDULE_DONE: "⏰✓",
    EVT_SCHEDULE_ERROR: "⏰✗",
    EVT_TASK_CREATE: "📋▶",
    EVT_TASK_DONE: "📋✓",
    EVT_TASK_ERROR: "📋✗",
    EVT_TASK_MESSAGE: "📋💬",
    EVT_NOTIFY: "🔔",
    EVT_ZENOH_PEER_JOIN: "🌐+",
    EVT_ZENOH_PEER_LEAVE: "🌐-",
    EVT_ZENOH_MESSAGE: "🌐💬",
    EVT_SYSTEM: "⚡",
}


class Event:
    """A single event in the unified stream."""
    __slots__ = ("timestamp", "event_type", "source", "summary", "detail", "metadata")

    def __init__(
        self,
        event_type: str,
        source: str,
        summary: str,
        detail: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.timestamp = time.time()
        self.event_type = event_type
        self.source = source
        self.summary = summary  # Short one-liner for sidebar
        self.detail = detail    # Longer text for context injection
        self.metadata = metadata or {}

    @property
    def icon(self) -> str:
        return EVENT_ICONS.get(self.event_type, "→")

    @property
    def time_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "time_str": self.time_str,
            "event_type": self.event_type,
            "source": self.source,
            "summary": self.summary,
            "detail": self.detail,
            "metadata": self.metadata,
        }


class EventBus:
    """Thread-safe capped ring buffer for unified events.

    All tools push here. TUI and context builder read from here.
    """

    def __init__(self, max_events: int = 200):
        self._events: deque = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._subscribers: List[Callable[[Event], None]] = []
        self._counter = 0

    def emit(
        self,
        event_type: str,
        source: str,
        summary: str,
        detail: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """Push an event into the bus. Thread-safe."""
        event = Event(event_type, source, summary, detail, metadata)
        with self._lock:
            self._events.append(event)
            self._counter += 1

        # Notify subscribers (non-blocking, fire-and-forget)
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:
                pass

        return event

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        """Register a callback for new events."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Event], None]) -> None:
        """Remove a callback."""
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    def recent(self, count: int = 30) -> List[Event]:
        """Get the last N events."""
        with self._lock:
            items = list(self._events)
        return items[-count:]

    def recent_by_type(self, event_type: str, count: int = 10) -> List[Event]:
        """Get recent events of a specific type."""
        with self._lock:
            items = [e for e in self._events if e.event_type == event_type]
        return items[-count:]

    def recent_since(self, seconds: float) -> List[Event]:
        """Get events from the last N seconds."""
        cutoff = time.time() - seconds
        with self._lock:
            return [e for e in self._events if e.timestamp > cutoff]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    @property
    def count(self) -> int:
        return self._counter

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._events)

    def get_context_string(self, max_events: int = 20, max_age_seconds: float = 300) -> str:
        """Build a context string for agent injection.

        Returns a formatted block of recent events that can be injected
        into the agent's dynamic context so it's aware of what's happening.
        """
        cutoff = time.time() - max_age_seconds
        with self._lock:
            events = [e for e in self._events if e.timestamp > cutoff]

        events = events[-max_events:]
        if not events:
            return ""

        lines = ["\n\n## 🔔 Live Event Stream (recent activity):"]
        for e in events:
            detail_preview = f": {e.detail[:150]}" if e.detail else ""
            lines.append(
                f"- [{e.time_str}] {e.icon} **{e.source}** {e.summary}{detail_preview}"
            )

        # Add counts by type for quick summary
        type_counts = {}
        for e in events:
            bucket = e.event_type.split(".")[0]  # "telegram", "whatsapp", etc.
            type_counts[bucket] = type_counts.get(bucket, 0) + 1

        if type_counts:
            summary_parts = [f"{k}:{v}" for k, v in sorted(type_counts.items())]
            lines.append(f"\n*Event summary: {', '.join(summary_parts)}*")

        return "\n".join(lines)


# ── Global singleton ────────────────────────────────────────────
# All tools import and use this single instance.

bus = EventBus(max_events=200)


# ── Convenience emit functions ──────────────────────────────────
# These make it easy for tools to push events with minimal code.

def emit(event_type: str, source: str, summary: str, detail: str = "",
         metadata: Optional[Dict[str, Any]] = None) -> Event:
    """Shorthand: push an event to the global bus."""
    return bus.emit(event_type, source, summary, detail, metadata)

@tool
def event_bus(
    action: str = "recent",
    event_type: str = None,
    source: str = "agent",
    summary: str = "",
    detail: str = "",
    count: int = 30,
    seconds: float = 300,
    max_age_seconds: float = 300,
) -> Dict[str, Any]:
    """🔔 Inspect and control the unified DevDuck event bus.

    All background tools (telegram, whatsapp, scheduler, tasks, listen,
    notify, zenoh, speech) push events here. The TUI sidebar and agent
    context reads from it.

    Actions:
        - "recent":     List last N events (count=30)
        - "by_type":    Filter by event_type (e.g. "telegram.in")
        - "since":      Events from last N seconds
        - "context":    Formatted context block (as injected to agent)
        - "emit":       Push a custom event (requires event_type+summary)
        - "clear":      Wipe the buffer
        - "stats":      Counter, size, event type distribution

    Args:
        action: One of the actions above
        event_type: For "by_type" filter or "emit" (e.g. "custom.note")
        source: Source label when emitting (default: "agent")
        summary: Short one-liner (required for emit)
        detail: Longer detail text (optional for emit)
        count: How many events to return (recent/by_type)
        seconds: Time window for "since"
        max_age_seconds: Time window for "context"

    Returns:
        Dict with status and content.
    """
    try:
        if action == "recent":
            events = bus.recent(count=count)
            if not events:
                return {"status": "success", "content": [{"text": "No events in bus."}]}
            lines = [f"📋 {len(events)} recent events (total emitted: {bus.count}):"]
            for e in events:
                detail_preview = f" — {e.detail[:100]}" if e.detail else ""
                lines.append(f"[{e.time_str}] {e.icon} {e.source} | {e.event_type}: {e.summary}{detail_preview}")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "by_type":
            if not event_type:
                return {"status": "error", "content": [{"text": "event_type required for by_type"}]}
            events = bus.recent_by_type(event_type, count=count)
            if not events:
                return {"status": "success", "content": [{"text": f"No events of type '{event_type}'."}]}
            lines = [f"📋 {len(events)} events of type '{event_type}':"]
            for e in events:
                lines.append(f"[{e.time_str}] {e.icon} {e.source}: {e.summary}")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "since":
            events = bus.recent_since(seconds)
            if not events:
                return {"status": "success", "content": [{"text": f"No events in last {seconds}s."}]}
            lines = [f"📋 {len(events)} events in last {seconds}s:"]
            for e in events:
                lines.append(f"[{e.time_str}] {e.icon} {e.source} | {e.event_type}: {e.summary}")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "context":
            ctx = bus.get_context_string(max_events=count, max_age_seconds=max_age_seconds)
            if not ctx:
                return {"status": "success", "content": [{"text": "No recent events for context."}]}
            return {"status": "success", "content": [{"text": ctx}]}

        elif action == "emit":
            if not event_type or not summary:
                return {"status": "error", "content": [{"text": "emit requires event_type + summary"}]}
            e = bus.emit(event_type, source, summary, detail)
            return {"status": "success", "content": [{"text": f"✅ Emitted [{e.time_str}] {e.icon} {source} | {event_type}: {summary}"}]}

        elif action == "clear":
            prev = bus.size
            bus.clear()
            return {"status": "success", "content": [{"text": f"🗑  Cleared {prev} events."}]}

        elif action == "stats":
            events = bus.recent(count=bus.size)
            type_counts: Dict[str, int] = {}
            source_counts: Dict[str, int] = {}
            for e in events:
                type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
                source_counts[e.source] = source_counts.get(e.source, 0) + 1
            lines = [
                f"📊 Event Bus Stats",
                f"  Total emitted: {bus.count}",
                f"  Currently buffered: {bus.size}",
                f"  Subscribers: {len(bus._subscribers)}",
                "",
                "By type:",
            ]
            for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {c:4d}  {t}")
            lines.append("\nBy source:")
            for s, c in sorted(source_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {c:4d}  {s}")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        else:
            return {
                "status": "error",
                "content": [{"text": f"Unknown action '{action}'. Valid: recent, by_type, since, context, emit, clear, stats"}],
            }

    except Exception as e:
        return {"status": "error", "content": [{"text": f"event_bus error: {e}"}]}
