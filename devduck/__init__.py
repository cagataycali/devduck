#!/usr/bin/env python3
"""
🦆 devduck - extreme minimalist self-adapting agent
one file. self-healing. runtime dependencies. adaptive.
"""

import os
import sys
import subprocess
import threading
import platform
import socket
import logging
import tempfile
import time
import warnings
import json
import zipfile
import builtins
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from logging.handlers import RotatingFileHandler
from strands import Agent, tool
from .callback_handler import callback_handler

# Try to import dill for better serialization, fall back to pickle
try:
    import dill as serializer

    SERIALIZER_NAME = "dill"
except ImportError:
    import pickle as serializer

    SERIALIZER_NAME = "pickle"

# Import system prompt helper for loading prompts from files
try:
    from devduck.tools.system_prompt import _get_system_prompt
except ImportError:
    # Fallback if tools module not available yet
    def _get_system_prompt(repository=None, variable_name="SYSTEM_PROMPT"):
        return os.getenv(variable_name, "")


warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*cache_prompt is deprecated.*")

os.environ["BYPASS_TOOL_CONSENT"] = os.getenv("BYPASS_TOOL_CONSENT", "true")
os.environ["STRANDS_TOOL_CONSOLE_MODE"] = "enabled"
os.environ["EDITOR_DISABLE_BACKUP"] = "true"

LOG_DIR = Path(tempfile.gettempdir()) / "devduck" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "devduck.log"
logger = logging.getLogger("devduck")
logger.setLevel(logging.DEBUG)

file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter("🦆 %(levelname)s: %(message)s")
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("DevDuck logging system initialized")


# =============================================================================
# 🎬 SESSION RECORDING - Time-travel debugger for AI agents
# =============================================================================

RECORDING_DIR = Path(tempfile.gettempdir()) / "devduck" / "recordings"
RECORDING_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RecordedEvent:
    """A single recorded event in the session timeline."""

    timestamp_ns: int
    layer: str  # "sys", "tool", "agent"
    event_type: str
    data: Dict[str, Any]
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionSnapshot:
    """A snapshot of agent state at a point in time."""

    timestamp: float
    snapshot_id: int
    agent_messages_count: int
    tools_loaded: List[str]
    system_prompt_hash: str
    env_vars_redacted: Dict[str, str]
    cwd: str
    events_since_last: int
    # NEW: Store actual conversation state for true resume capability
    agent_messages: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    last_query: str = ""
    last_result: str = ""
    model_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventBuffer:
    """Ring buffer for recording events across all layers."""

    def __init__(self, max_events: int = 10000):
        self.events: deque = deque(maxlen=max_events)
        self.lock = threading.Lock()
        self._event_count = 0

    def record(
        self, layer: str, event_type: str, data: Dict[str, Any], trace_id: str = None
    ):
        """Record an event to the buffer."""
        event = RecordedEvent(
            timestamp_ns=time.time_ns(),
            layer=layer,
            event_type=event_type,
            data=data,
            trace_id=trace_id,
        )
        with self.lock:
            self.events.append(event)
            self._event_count += 1

    def get_recent(self, seconds: float = 5.0) -> List[RecordedEvent]:
        """Get events from the last N seconds."""
        cutoff = time.time_ns() - int(seconds * 1e9)
        with self.lock:
            return [e for e in self.events if e.timestamp_ns > cutoff]

    def get_recent_context(self, seconds: float = 5.0, max_events: int = 20) -> str:
        """Get recent events formatted for system prompt injection."""
        recent = self.get_recent(seconds)[-max_events:]
        if not recent:
            return ""

        lines = ["## 🎬 Recent System Events:"]
        for event in recent:
            ts = datetime.fromtimestamp(event.timestamp_ns / 1e9).strftime(
                "%H:%M:%S.%f"
            )[:-3]
            lines.append(
                f"- [{ts}] [{event.layer}] {event.event_type}: {json.dumps(event.data)[:200]}"
            )
        return "\n".join(lines)

    def get_all(self) -> List[RecordedEvent]:
        """Get all events in the buffer."""
        with self.lock:
            return list(self.events)

    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.events.clear()
            self._event_count = 0

    @property
    def count(self) -> int:
        return self._event_count


class SessionRecorder:
    """Records devduck sessions for replay.

    Captures three layers:
    - sys: OS-level events (file I/O, network)
    - tool: Agent tool calls and results
    - agent: Messages, decisions, state changes

    Exports to a ZIP containing:
    - session.pkl: Serialized snapshots (dill/pickle)
    - events.jsonl: All events in JSON Lines format
    - metadata.json: Session info
    """

    # Keys to redact from environment variables
    REDACT_PATTERNS = ["KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH"]

    def __init__(self, session_id: str = None):
        self.session_id = (
            session_id or f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        self.event_buffer = EventBuffer()
        self.snapshots: List[SessionSnapshot] = []
        self.recording = False
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._snapshot_counter = 0
        self._original_open = None
        self._original_requests_get = None
        self._hooks_installed = False
        self.metadata: Dict[str, Any] = {}

    def start(self, install_hooks: bool = True):
        """Start recording the session."""
        if self.recording:
            logger.warning("Session recording already active")
            return

        self.recording = True
        self.start_time = time.time()
        self.metadata = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "python_version": sys.version,
            "serializer": SERIALIZER_NAME,
        }

        logger.info(f"🎬 Session recording started: {self.session_id}")
        print(f"🎬 Recording session: {self.session_id}")

        # Record start event
        self.event_buffer.record("agent", "session.start", self.metadata)

        if install_hooks:
            self._install_hooks()

    def stop(self) -> str:
        """Stop recording and return session ID."""
        if not self.recording:
            logger.warning("No active recording to stop")
            return self.session_id

        self.recording = False
        self.end_time = time.time()
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["duration_seconds"] = self.end_time - self.start_time
        self.metadata["total_events"] = self.event_buffer.count
        self.metadata["total_snapshots"] = len(self.snapshots)

        # Record stop event
        self.event_buffer.record(
            "agent",
            "session.stop",
            {
                "duration": self.metadata["duration_seconds"],
                "events": self.event_buffer.count,
            },
        )

        self._uninstall_hooks()

        logger.info(f"🎬 Session recording stopped: {self.session_id}")
        print(
            f"🎬 Recording stopped: {self.event_buffer.count} events, {len(self.snapshots)} snapshots"
        )

        return self.session_id

    def snapshot(
        self,
        agent=None,
        description: str = "",
        last_query: str = "",
        last_result: str = "",
    ):
        """Create a state snapshot with full conversation state for resume capability.

        Args:
            agent: The agent instance to capture state from
            description: Human-readable description of this snapshot
            last_query: The last user query (for resume context)
            last_result: The last agent result (for resume context)
        """
        if not self.recording:
            return

        self._snapshot_counter += 1

        # Get agent info if available
        messages_count = 0
        tools_loaded = []
        system_prompt_hash = ""
        agent_messages = []
        system_prompt = ""
        model_info = {}

        if agent:
            try:
                # Capture message count
                if hasattr(agent, "messages"):
                    messages_count = len(agent.messages) if agent.messages else 0
                    # NEW: Capture actual messages for resume capability
                    if agent.messages:
                        agent_messages = self._serialize_messages(agent.messages)

                # Capture tools
                if hasattr(agent, "tool_names"):
                    tools_loaded = list(agent.tool_names)

                # Capture system prompt
                if hasattr(agent, "system_prompt"):
                    system_prompt_hash = str(hash(agent.system_prompt))[:16]
                    system_prompt = agent.system_prompt or ""

                # Capture model info safely
                if hasattr(agent, "model"):
                    model = agent.model
                    model_info = {
                        "type": type(model).__name__,
                        "model_id": getattr(model, "model_id", "unknown"),
                        "provider": getattr(model, "provider", "unknown"),
                    }

            except Exception as e:
                logger.debug(f"Could not extract agent state: {e}")

        snapshot = SessionSnapshot(
            timestamp=time.time(),
            snapshot_id=self._snapshot_counter,
            agent_messages_count=messages_count,
            tools_loaded=tools_loaded,
            system_prompt_hash=system_prompt_hash,
            env_vars_redacted=self._redact_env_vars(),
            cwd=os.getcwd(),
            events_since_last=self.event_buffer.count,
            # NEW: Full state for resume
            agent_messages=agent_messages,
            system_prompt=system_prompt,
            last_query=last_query,
            last_result=last_result,
            model_info=model_info,
        )

        self.snapshots.append(snapshot)

        # Record snapshot event
        self.event_buffer.record(
            "agent",
            "snapshot.created",
            {
                "snapshot_id": self._snapshot_counter,
                "description": description,
                "messages_captured": len(agent_messages),
                "has_system_prompt": bool(system_prompt),
            },
        )

        logger.debug(
            f"🎬 Snapshot #{self._snapshot_counter} created with {len(agent_messages)} messages"
        )

    def _serialize_messages(self, messages) -> List[Dict[str, Any]]:
        """Safely serialize agent messages for storage."""
        serialized = []
        for msg in messages:
            try:
                if isinstance(msg, dict):
                    # Already a dict, just copy
                    serialized.append(dict(msg))
                elif hasattr(msg, "__dict__"):
                    # Object with __dict__, convert
                    serialized.append(dict(msg.__dict__))
                elif hasattr(msg, "model_dump"):
                    # Pydantic model
                    serialized.append(msg.model_dump())
                else:
                    # Fallback: convert to string representation
                    serialized.append({"content": str(msg), "role": "unknown"})
            except Exception as e:
                logger.debug(f"Could not serialize message: {e}")
                serialized.append(
                    {"content": str(msg)[:1000], "role": "unknown", "_error": str(e)}
                )
        return serialized

    def record_tool_call(
        self, tool_name: str, args: Dict[str, Any], trace_id: str = None
    ):
        """Record a tool call event."""
        if not self.recording:
            return
        self.event_buffer.record(
            "tool",
            "tool.call",
            {"name": tool_name, "args": self._truncate_data(args)},
            trace_id,
        )

    def record_tool_result(
        self, tool_name: str, result: Any, duration_ms: float = 0, trace_id: str = None
    ):
        """Record a tool result event."""
        if not self.recording:
            return
        self.event_buffer.record(
            "tool",
            "tool.result",
            {
                "name": tool_name,
                "result_preview": str(result)[:500],
                "duration_ms": duration_ms,
            },
            trace_id,
        )

    def record_agent_message(self, role: str, content: str, trace_id: str = None):
        """Record an agent message event."""
        if not self.recording:
            return
        self.event_buffer.record(
            "agent",
            "message",
            {"role": role, "content_preview": content[:500] if content else ""},
            trace_id,
        )

    def record_sys_event(self, event_type: str, data: Dict[str, Any]):
        """Record a system-level event."""
        if not self.recording:
            return
        self.event_buffer.record("sys", event_type, self._truncate_data(data))

    def export(self, output_path: str = None) -> str:
        """Export the session to a ZIP file."""
        if output_path is None:
            output_path = str(RECORDING_DIR / f"{self.session_id}.zip")

        logger.info(f"🎬 Exporting session to {output_path}")

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write events as JSONL
            events_data = "\n".join(
                json.dumps(e.to_dict()) for e in self.event_buffer.get_all()
            )
            zf.writestr("events.jsonl", events_data)

            # Write snapshots as JSON
            snapshots_data = json.dumps([s.to_dict() for s in self.snapshots], indent=2)
            zf.writestr("snapshots.json", snapshots_data)

            # Write metadata
            self.metadata["export_time"] = datetime.now().isoformat()
            zf.writestr("metadata.json", json.dumps(self.metadata, indent=2))

            # Try to serialize snapshots with dill/pickle (for potential state replay)
            try:
                pkl_data = serializer.dumps(
                    {"snapshots": self.snapshots, "metadata": self.metadata}
                )
                zf.writestr("session.pkl", pkl_data)
            except Exception as e:
                logger.warning(f"Could not serialize session state: {e}")
                zf.writestr("session.pkl.error", str(e))

        logger.info(f"🎬 Session exported: {output_path}")
        print(f"🎬 Session exported: {output_path}")
        return output_path

    def _redact_env_vars(self) -> Dict[str, str]:
        """Get environment variables with sensitive values redacted."""
        redacted = {}
        for key, value in os.environ.items():
            if any(pattern in key.upper() for pattern in self.REDACT_PATTERNS):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value[:100] if len(value) > 100 else value
        return redacted

    def _truncate_data(self, data: Any, max_len: int = 1000) -> Any:
        """Truncate data to prevent huge events."""
        if isinstance(data, str):
            return data[:max_len] if len(data) > max_len else data
        elif isinstance(data, dict):
            return {
                k: self._truncate_data(v, max_len // 2)
                for k, v in list(data.items())[:20]
            }
        elif isinstance(data, list):
            return [self._truncate_data(v, max_len // 2) for v in data[:10]]
        else:
            return str(data)[:max_len]

    def _install_hooks(self):
        """Install hooks to capture OS-level events."""
        if self._hooks_installed:
            return

        # Hook: builtins.open
        self._original_open = builtins.open
        recorder = self

        def traced_open(file, mode="r", *args, **kwargs):
            if recorder.recording:
                recorder.record_sys_event(
                    "file.open", {"path": str(file), "mode": mode}
                )
            return recorder._original_open(file, mode, *args, **kwargs)

        builtins.open = traced_open

        # Hook: requests (if available)
        try:
            import requests

            self._original_requests_get = requests.get

            def traced_get(url, *args, **kwargs):
                if recorder.recording:
                    recorder.record_sys_event("http.get", {"url": str(url)[:200]})
                return recorder._original_requests_get(url, *args, **kwargs)

            requests.get = traced_get
        except ImportError:
            pass

        self._hooks_installed = True
        logger.info("🎬 Recording hooks installed")

    def _uninstall_hooks(self):
        """Uninstall recording hooks."""
        if not self._hooks_installed:
            return

        if self._original_open:
            builtins.open = self._original_open

        if self._original_requests_get:
            try:
                import requests

                requests.get = self._original_requests_get
            except ImportError:
                pass

        self._hooks_installed = False
        logger.info("🎬 Recording hooks uninstalled")


# Global session recorder instance
_session_recorder: Optional[SessionRecorder] = None


def get_session_recorder() -> Optional[SessionRecorder]:
    """Get the global session recorder if active."""
    return _session_recorder


def start_recording(
    session_id: str = None, install_hooks: bool = True
) -> SessionRecorder:
    """Start a new recording session."""
    global _session_recorder
    _session_recorder = SessionRecorder(session_id)
    _session_recorder.start(install_hooks)
    return _session_recorder


def stop_recording() -> Optional[str]:
    """Stop the current recording session and return the export path."""
    global _session_recorder
    if _session_recorder and _session_recorder.recording:
        _session_recorder.stop()
        export_path = _session_recorder.export()
        return export_path
    return None


class LoadedSession:
    """A loaded session from a ZIP file for replay and analysis.

    Provides access to recorded events, snapshots, and metadata.
    Can be used to resume agent state from a specific snapshot.

    Example:
        session = load_session("session-20260202-224751.zip")
        print(session.metadata)
        print(session.events[:10])

        # Resume from snapshot
        session.resume_from_snapshot(2)
    """

    def __init__(self, zip_path: str):
        """Load a session from a ZIP file."""
        self.zip_path = Path(zip_path)
        self.events: List[RecordedEvent] = []
        self.snapshots: List[SessionSnapshot] = []
        self.metadata: Dict[str, Any] = {}
        self._pkl_data: Optional[Dict] = None

        self._load()

    def _load(self):
        """Load and parse the session ZIP file."""
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Session file not found: {self.zip_path}")

        with zipfile.ZipFile(self.zip_path, "r") as zf:
            # Load events.jsonl
            if "events.jsonl" in zf.namelist():
                events_text = zf.read("events.jsonl").decode("utf-8")
                for line in events_text.strip().split("\n"):
                    if line:
                        data = json.loads(line)
                        self.events.append(RecordedEvent(**data))

            # Load snapshots.json
            if "snapshots.json" in zf.namelist():
                snapshots_text = zf.read("snapshots.json").decode("utf-8")
                snapshots_data = json.loads(snapshots_text)
                for snap_data in snapshots_data:
                    self.snapshots.append(SessionSnapshot(**snap_data))

            # Load metadata.json
            if "metadata.json" in zf.namelist():
                metadata_text = zf.read("metadata.json").decode("utf-8")
                self.metadata = json.loads(metadata_text)

            # Load session.pkl if available
            if "session.pkl" in zf.namelist():
                try:
                    pkl_data = zf.read("session.pkl")
                    self._pkl_data = serializer.loads(pkl_data)
                except Exception as e:
                    logger.warning(f"Could not load session.pkl: {e}")

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self.metadata.get("session_id", "unknown")

    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        return self.metadata.get("duration_seconds", 0.0)

    @property
    def has_pkl(self) -> bool:
        """Check if session has serialized state for resuming."""
        return self._pkl_data is not None

    def get_events_by_layer(self, layer: str) -> List[RecordedEvent]:
        """Get events filtered by layer (sys, tool, agent)."""
        return [e for e in self.events if e.layer == layer]

    def get_events_by_type(self, event_type: str) -> List[RecordedEvent]:
        """Get events filtered by type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_events_in_range(self, start_ns: int, end_ns: int) -> List[RecordedEvent]:
        """Get events within a timestamp range."""
        return [e for e in self.events if start_ns <= e.timestamp_ns <= end_ns]

    def get_snapshot(self, snapshot_id: int) -> Optional[SessionSnapshot]:
        """Get a specific snapshot by ID."""
        for snap in self.snapshots:
            if snap.snapshot_id == snapshot_id:
                return snap
        return None

    def get_events_until_snapshot(self, snapshot_id: int) -> List[RecordedEvent]:
        """Get all events up to a specific snapshot."""
        snap = self.get_snapshot(snapshot_id)
        if not snap:
            return []

        snap_time_ns = int(snap.timestamp * 1e9)
        return [e for e in self.events if e.timestamp_ns <= snap_time_ns]

    def resume_from_snapshot(
        self, snapshot_id: int, agent: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Resume agent state from a specific snapshot.

        This reconstructs the agent context based on the snapshot state.
        If an agent is provided, it will be configured with the snapshot state
        including restored conversation history.

        Args:
            snapshot_id: The snapshot ID to resume from
            agent: Optional agent instance to configure

        Returns:
            Dict with resume status, snapshot info, and continuation prompt
        """
        snap = self.get_snapshot(snapshot_id)
        if not snap:
            return {"status": "error", "message": f"Snapshot #{snapshot_id} not found"}

        result = {
            "status": "success",
            "snapshot_id": snapshot_id,
            "timestamp": datetime.fromtimestamp(snap.timestamp).isoformat(),
            "cwd": snap.cwd,
            "tools_loaded": snap.tools_loaded,
            "messages_count": snap.agent_messages_count,
            "events_before_snapshot": len(self.get_events_until_snapshot(snapshot_id)),
            "messages_restored": 0,
            "continuation_prompt": "",
        }

        # Change to snapshot's working directory
        if os.path.exists(snap.cwd):
            os.chdir(snap.cwd)
            result["cwd_changed"] = True
        else:
            result["cwd_changed"] = False
            result["cwd_warning"] = f"Directory not found: {snap.cwd}"

        # If agent provided, restore full state
        if agent is not None:
            try:
                # Restore conversation history (the key enhancement!)
                if snap.agent_messages:
                    agent.messages = snap.agent_messages
                    result["messages_restored"] = len(snap.agent_messages)
                    logger.info(
                        f"Restored {len(snap.agent_messages)} messages to agent"
                    )

                # Check tool compatibility
                if hasattr(agent, "tool_registry"):
                    current_tools = set(agent.tool_registry.registry.keys())
                    snapshot_tools = set(snap.tools_loaded)

                    result["tools_match"] = current_tools == snapshot_tools
                    result["missing_tools"] = list(snapshot_tools - current_tools)
                    result["extra_tools"] = list(current_tools - snapshot_tools)

            except Exception as e:
                result["agent_restore_error"] = str(e)
                logger.error(f"Error restoring agent state: {e}")

        # Build continuation prompt (like research_agent_runner pattern)
        if snap.last_query or snap.last_result:
            result["continuation_prompt"] = self._build_continuation_prompt(snap)

        # Include model info if available
        if snap.model_info:
            result["model_info"] = snap.model_info

        logger.info(
            f"Resumed from snapshot #{snapshot_id}: {result['messages_restored']} messages restored"
        )
        return result

    def _build_continuation_prompt(self, snap: SessionSnapshot) -> str:
        """Build a continuation prompt from snapshot context.

        This follows the pattern from research_agent_runner.py for
        seamless conversation continuation.
        """
        prompt_parts = []

        prompt_parts.append("=== RESUMED SESSION CONTEXT ===")
        prompt_parts.append(f"Session: {self.session_id}")
        prompt_parts.append(
            f"Snapshot: #{snap.snapshot_id} from {datetime.fromtimestamp(snap.timestamp).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        prompt_parts.append(f"Working Directory: {snap.cwd}")

        if snap.last_query:
            prompt_parts.append(f"\n--- Previous Query ---\n{snap.last_query}")

        if snap.last_result:
            # Truncate long results
            result_preview = snap.last_result[:2000]
            if len(snap.last_result) > 2000:
                result_preview += "\n... [truncated]"
            prompt_parts.append(f"\n--- Previous Result ---\n{result_preview}")

        prompt_parts.append("\n=== END RESUMED CONTEXT ===")
        prompt_parts.append(
            "\nPlease continue from where we left off. The conversation history has been restored."
        )

        return "\n".join(prompt_parts)

    def resume_and_continue(
        self, snapshot_id: int, new_query: str, agent: Any
    ) -> Dict[str, Any]:
        """Resume from snapshot and immediately continue with a new query.

        This is a convenience method that:
        1. Restores agent state from snapshot
        2. Runs the agent with context-aware continuation prompt

        Args:
            snapshot_id: The snapshot ID to resume from
            new_query: The new query to run after resuming
            agent: The agent instance to use

        Returns:
            Dict with resume status and agent result
        """
        # First, resume the state
        resume_result = self.resume_from_snapshot(snapshot_id, agent)

        if resume_result["status"] != "success":
            return resume_result

        # Build the continuation query
        continuation_prompt = resume_result.get("continuation_prompt", "")
        if continuation_prompt:
            full_query = f"{continuation_prompt}\n\n--- New Query ---\n{new_query}"
        else:
            full_query = new_query

        # Run the agent
        try:
            agent_result = agent(full_query)
            resume_result["agent_result"] = str(agent_result)
            resume_result["continuation_successful"] = True
        except Exception as e:
            resume_result["agent_result"] = None
            resume_result["continuation_error"] = str(e)
            resume_result["continuation_successful"] = False

        return resume_result

    def replay_events(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        callback: Optional[Callable[[RecordedEvent, int], None]] = None,
    ) -> List[RecordedEvent]:
        """Replay events with optional callback for each event.

        Args:
            start_idx: Starting event index
            end_idx: Ending event index (exclusive), None for all
            callback: Optional function called for each event (event, index)

        Returns:
            List of replayed events
        """
        end = end_idx if end_idx is not None else len(self.events)
        replayed = []

        for i in range(start_idx, min(end, len(self.events))):
            event = self.events[i]
            replayed.append(event)
            if callback:
                callback(event, i)

        return replayed

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for JSON export."""
        return {
            "metadata": self.metadata,
            "events": [e.to_dict() for e in self.events],
            "snapshots": [s.to_dict() for s in self.snapshots],
            "summary": {
                "total_events": len(self.events),
                "total_snapshots": len(self.snapshots),
                "events_by_layer": {
                    "sys": len(self.get_events_by_layer("sys")),
                    "tool": len(self.get_events_by_layer("tool")),
                    "agent": len(self.get_events_by_layer("agent")),
                },
                "has_resumable_state": self.has_pkl,
            },
        }

    def __repr__(self) -> str:
        return (
            f"LoadedSession(id={self.session_id}, "
            f"events={len(self.events)}, "
            f"snapshots={len(self.snapshots)}, "
            f"duration={self.duration:.1f}s)"
        )


def load_session(path: str) -> LoadedSession:
    """Load a recorded session from a ZIP file.

    Args:
        path: Path to the session ZIP file

    Returns:
        LoadedSession object with events, snapshots, and metadata

    Example:
        session = load_session("~/Desktop/session-20260202-224751.zip")
        print(session)  # LoadedSession(id=session-20260202-224751, events=12, snapshots=3)

        # Get all tool calls
        tool_events = session.get_events_by_layer("tool")

        # Resume from a snapshot (restores agent.messages!)
        result = session.resume_from_snapshot(2, agent=my_agent)
        print(f"Restored {result['messages_restored']} messages")

        # Resume and continue with new query
        result = session.resume_and_continue(2, "what was I working on?", agent=my_agent)
        print(result['agent_result'])

        # Replay with callback
        def on_event(event, idx):
            print(f"[{idx}] {event.event_type}: {event.data}")
        session.replay_events(callback=on_event)
    """
    # Expand user path
    expanded_path = os.path.expanduser(path)
    return LoadedSession(expanded_path)


def resume_session(
    session_path: str, snapshot_id: int = None, new_query: str = None
) -> Dict[str, Any]:
    """Resume a devduck session from a recorded session file.

    This is a convenience function that:
    1. Loads a session from file
    2. Resumes from the latest (or specified) snapshot
    3. Optionally continues with a new query

    Args:
        session_path: Path to the session ZIP file
        snapshot_id: Specific snapshot to resume from (default: latest)
        new_query: Optional new query to run after resuming

    Returns:
        Dict with resume status and optionally agent result

    Example:
        # Resume from latest snapshot
        result = resume_session("~/Desktop/session-20260202-224751.zip")

        # Resume from specific snapshot
        result = resume_session("session.zip", snapshot_id=2)

        # Resume and continue working
        result = resume_session("session.zip", new_query="continue where we left off")
        print(result['agent_result'])
    """
    # Load the session
    session = load_session(session_path)

    if not session.snapshots:
        return {"status": "error", "message": "No snapshots found in session"}

    # Use latest snapshot if not specified
    if snapshot_id is None:
        snapshot_id = session.snapshots[-1].snapshot_id

    # Get the devduck agent
    if not hasattr(devduck, "agent") or devduck.agent is None:
        return {"status": "error", "message": "DevDuck agent not initialized"}

    # Resume with or without new query
    if new_query:
        return session.resume_and_continue(snapshot_id, new_query, devduck.agent)
    else:
        return session.resume_from_snapshot(snapshot_id, devduck.agent)


def list_sessions() -> List[Dict[str, Any]]:
    """List all recorded sessions in the default recording directory.

    Returns:
        List of session info dicts with path, size, and modified time
    """
    sessions = []
    for zip_file in RECORDING_DIR.glob("*.zip"):
        stat = zip_file.stat()
        sessions.append(
            {
                "path": str(zip_file),
                "name": zip_file.name,
                "size_kb": stat.st_size / 1024,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        )
    return sorted(sessions, key=lambda x: x["modified"], reverse=True)


def get_own_source_code():
    """Read own source code for self-awareness"""
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            return f"# Source path: {__file__}\n\ndevduck/__init__.py\n```python\n{f.read()}\n```"
    except Exception as e:
        return f"Error reading source: {e}"


def view_logs_tool(
    action: str = "view",
    lines: int = 100,
    pattern: str = None,
) -> Dict[str, Any]:
    """
    View and manage DevDuck logs.

    Args:
        action: Action to perform - "view", "tail", "search", "clear", "stats"
        lines: Number of lines to show (for view/tail)
        pattern: Search pattern (for search action)

    Returns:
        Dict with status and content
    """
    try:
        if action == "view":
            if not LOG_FILE.exists():
                return {"status": "success", "content": [{"text": "No logs yet"}]}

            with open(LOG_FILE, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                recent_lines = (
                    all_lines[-lines:] if len(all_lines) > lines else all_lines
                )
                content = "".join(recent_lines)

            return {
                "status": "success",
                "content": [
                    {"text": f"Last {len(recent_lines)} log lines:\n\n{content}"}
                ],
            }

        elif action == "tail":
            if not LOG_FILE.exists():
                return {"status": "success", "content": [{"text": "No logs yet"}]}

            with open(LOG_FILE, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                tail_lines = all_lines[-50:] if len(all_lines) > 50 else all_lines
                content = "".join(tail_lines)

            return {
                "status": "success",
                "content": [{"text": f"Tail (last 50 lines):\n\n{content}"}],
            }

        elif action == "search":
            if not pattern:
                return {
                    "status": "error",
                    "content": [{"text": "pattern parameter required for search"}],
                }

            if not LOG_FILE.exists():
                return {"status": "success", "content": [{"text": "No logs yet"}]}

            with open(LOG_FILE, "r", encoding="utf-8") as f:
                matching_lines = [line for line in f if pattern.lower() in line.lower()]

            if not matching_lines:
                return {
                    "status": "success",
                    "content": [{"text": f"No matches found for pattern: {pattern}"}],
                }

            content = "".join(matching_lines[-100:])  # Last 100 matches
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"Found {len(matching_lines)} matches (showing last 100):\n\n{content}"
                    }
                ],
            }

        elif action == "clear":
            if LOG_FILE.exists():
                LOG_FILE.unlink()
                logger.info("Log file cleared by user")
            return {
                "status": "success",
                "content": [{"text": "Logs cleared successfully"}],
            }

        elif action == "stats":
            if not LOG_FILE.exists():
                return {"status": "success", "content": [{"text": "No logs yet"}]}

            stat = LOG_FILE.stat()
            size_mb = stat.st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(stat.st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            with open(LOG_FILE, "r", encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)

            stats_text = f"""Log File Statistics:
Path: {LOG_FILE}
Size: {size_mb:.2f} MB
Lines: {total_lines}
Last Modified: {modified}"""

            return {"status": "success", "content": [{"text": stats_text}]}

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Valid: view, tail, search, clear, stats"
                    }
                ],
            }

    except Exception as e:
        logger.error(f"Error in view_logs_tool: {e}")
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}


def manage_tools_func(
    action: str,
    package: str = None,
    tool_names: str = None,
    tool_path: str = None,
) -> Dict[str, Any]:
    """Manage the agent's tool set at runtime - add, remove, list, reload tools on the fly."""
    try:
        if not hasattr(devduck, "agent") or not devduck.agent:
            return {"status": "error", "content": [{"text": "Agent not initialized"}]}

        registry = devduck.agent.tool_registry

        if action == "list":
            # List tools from registry
            tool_list = list(registry.registry.keys())
            dynamic_tools = list(registry.dynamic_tools.keys())

            text = f"Currently loaded {len(tool_list)} tools:\n"
            text += "\n".join(f"  • {t}" for t in sorted(tool_list))
            if dynamic_tools:
                text += f"\n\nDynamic tools ({len(dynamic_tools)}):\n"
                text += "\n".join(f"  • {t}" for t in sorted(dynamic_tools))

            return {"status": "success", "content": [{"text": text}]}

        elif action == "add":
            if not package and not tool_path:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "Either 'package' or 'tool_path' required for add action"
                        }
                    ],
                }

            added_tools = []

            # Add from package using process_tools
            if package:
                if not tool_names:
                    return {
                        "status": "error",
                        "content": [
                            {"text": "'tool_names' required when adding from package"}
                        ],
                    }

                tools_to_add = [t.strip() for t in tool_names.split(",")]

                # Build tool specs: package.tool_name format
                tool_specs = [f"{package}.{tool_name}" for tool_name in tools_to_add]

                try:
                    added_tool_names = registry.process_tools(tool_specs)
                    added_tools.extend(added_tool_names)
                    logger.info(f"Added tools from {package}: {added_tool_names}")
                except Exception as e:
                    logger.error(f"Failed to add tools from {package}: {e}")
                    return {
                        "status": "error",
                        "content": [{"text": f"Failed to add tools: {str(e)}"}],
                    }

            # Add from file path using process_tools
            if tool_path:
                try:
                    added_tool_names = registry.process_tools([tool_path])
                    added_tools.extend(added_tool_names)
                    logger.info(f"Added tools from file: {added_tool_names}")
                except Exception as e:
                    logger.error(f"Failed to add tool from {tool_path}: {e}")
                    return {
                        "status": "error",
                        "content": [{"text": f"Failed to add tool: {str(e)}"}],
                    }

            if added_tools:
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": f"✅ Added {len(added_tools)} tools: {', '.join(added_tools)}\n"
                            + f"Total tools: {len(registry.registry)}"
                        }
                    ],
                }
            else:
                return {"status": "error", "content": [{"text": "No tools were added"}]}

        elif action == "remove":
            if not tool_names:
                return {
                    "status": "error",
                    "content": [{"text": "'tool_names' required for remove action"}],
                }

            tools_to_remove = [t.strip() for t in tool_names.split(",")]
            removed_tools = []

            # Remove from registry
            for tool_name in tools_to_remove:
                if tool_name in registry.registry:
                    del registry.registry[tool_name]
                    removed_tools.append(tool_name)
                    logger.info(f"Removed tool: {tool_name}")

                if tool_name in registry.dynamic_tools:
                    del registry.dynamic_tools[tool_name]
                    logger.info(f"Removed dynamic tool: {tool_name}")

            if removed_tools:
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": f"✅ Removed {len(removed_tools)} tools: {', '.join(removed_tools)}\n"
                            + f"Total tools: {len(registry.registry)}"
                        }
                    ],
                }
            else:
                return {
                    "status": "success",
                    "content": [{"text": "No tools were removed (not found)"}],
                }

        elif action == "reload":
            if tool_names:
                # Reload specific tools
                tools_to_reload = [t.strip() for t in tool_names.split(",")]
                reloaded_tools = []
                failed_tools = []

                for tool_name in tools_to_reload:
                    try:
                        registry.reload_tool(tool_name)
                        reloaded_tools.append(tool_name)
                        logger.info(f"Reloaded tool: {tool_name}")
                    except Exception as e:
                        failed_tools.append((tool_name, str(e)))
                        logger.error(f"Failed to reload {tool_name}: {e}")

                text = ""
                if reloaded_tools:
                    text += f"✅ Reloaded {len(reloaded_tools)} tools: {', '.join(reloaded_tools)}\n"
                if failed_tools:
                    text += f"❌ Failed to reload {len(failed_tools)} tools:\n"
                    for tool_name, error in failed_tools:
                        text += f"  • {tool_name}: {error}\n"

                return {"status": "success", "content": [{"text": text}]}
            else:
                # Reload all tools - restart agent
                logger.info("Reloading all tools via restart")
                devduck.restart()
                return {
                    "status": "success",
                    "content": [{"text": "✅ All tools reloaded - agent restarted"}],
                }

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Valid: list, add, remove, reload"
                    }
                ],
            }

    except Exception as e:
        logger.error(f"Error in manage_tools: {e}")
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}


def get_shell_history_file():
    """Get the devduck-specific history file path."""
    devduck_history = Path.home() / ".devduck_history"
    if not devduck_history.exists():
        devduck_history.touch(mode=0o600)
    return str(devduck_history)


def get_shell_history_files():
    """Get available shell history file paths."""
    history_files = []

    # devduck history (primary)
    devduck_history = Path(get_shell_history_file())
    if devduck_history.exists():
        history_files.append(("devduck", str(devduck_history)))

    # Bash history
    bash_history = Path.home() / ".bash_history"
    if bash_history.exists():
        history_files.append(("bash", str(bash_history)))

    # Zsh history
    zsh_history = Path.home() / ".zsh_history"
    if zsh_history.exists():
        history_files.append(("zsh", str(zsh_history)))

    return history_files


def parse_history_line(line, history_type):
    """Parse a history line based on the shell type."""
    line = line.strip()
    if not line:
        return None

    if history_type == "devduck":
        # devduck format: ": timestamp:0;# devduck: query" or ": timestamp:0;# devduck_result: result"
        if "# devduck:" in line:
            try:
                timestamp_str = line.split(":")[1]
                timestamp = int(timestamp_str)
                readable_time = datetime.fromtimestamp(timestamp).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                query = line.split("# devduck:")[-1].strip()
                return ("you", readable_time, query)
            except (ValueError, IndexError):
                return None
        elif "# devduck_result:" in line:
            try:
                timestamp_str = line.split(":")[1]
                timestamp = int(timestamp_str)
                readable_time = datetime.fromtimestamp(timestamp).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                result = line.split("# devduck_result:")[-1].strip()
                return ("me", readable_time, result)
            except (ValueError, IndexError):
                return None

    elif history_type == "zsh":
        if line.startswith(": ") and ":0;" in line:
            try:
                parts = line.split(":0;", 1)
                if len(parts) == 2:
                    timestamp_str = parts[0].split(":")[1]
                    timestamp = int(timestamp_str)
                    readable_time = datetime.fromtimestamp(timestamp).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    command = parts[1].strip()
                    if not command.startswith("devduck "):
                        return ("shell", readable_time, f"$ {command}")
            except (ValueError, IndexError):
                return None

    elif history_type == "bash":
        readable_time = "recent"
        if not line.startswith("devduck "):
            return ("shell", readable_time, f"$ {line}")

    return None


def get_ambient_status_context():
    """Get ambient mode status for dynamic context injection."""
    try:
        if not hasattr(devduck, "ambient") or not devduck.ambient:
            return ""

        ambient = devduck.ambient
        context = f"\n\n## 🌙 Ambient Mode Status:\n"
        context += f"- **Enabled**: {ambient.running}\n"
        context += f"- **Mode**: {'AUTONOMOUS' if ambient.autonomous else 'Standard'}\n"
        context += f"- **Iterations**: {ambient.ambient_iterations}/{ambient.autonomous_max_iterations if ambient.autonomous else ambient.max_iterations}\n"
        context += (
            f"- **Pending Results**: {len(ambient.ambient_results_history)} stored\n"
        )

        if ambient.last_query:
            context += f"- **Last Query**: {ambient.last_query[:100]}...\n"

        return context
    except Exception as e:
        logger.debug(f"Could not get ambient status context: {e}")
        return ""


def get_zenoh_peers_context():
    """Get current zenoh peers for dynamic context injection."""
    try:
        import sys as _sys

        _zp_mod = _sys.modules.get("devduck.tools.zenoh_peer")
        if _zp_mod:
            ZENOH_STATE = _zp_mod.ZENOH_STATE
        else:
            from devduck.tools.zenoh_peer import ZENOH_STATE
        import time

        logger.debug(
            f"Zenoh context check - running: {ZENOH_STATE.get('running')}, peers: {ZENOH_STATE.get('peers')}"
        )

        if not ZENOH_STATE.get("running"):
            logger.debug("Zenoh not running, returning empty context")
            return ""

        instance_id = ZENOH_STATE.get("instance_id", "unknown")
        peers = ZENOH_STATE.get("peers", {})

        context = f"\n\n## Zenoh Network Status:\n"
        context += f"- **My Instance ID**: {instance_id}\n"
        context += f"- **Connected Peers**: {len(peers)}\n"

        if peers:
            context += "\n### Active Peers:\n"
            for peer_id, info in peers.items():
                age = time.time() - info.get("last_seen", 0)
                hostname = info.get("hostname", "unknown")
                model = info.get("model", "unknown")
                context += f"- `{peer_id}` ({hostname}) - model: {model}, seen {age:.0f}s ago\n"
            context += "\n**Use**: `zenoh_peer(action='broadcast', message='...')` to send to all, or `zenoh_peer(action='send', peer_id='...', message='...')` for specific peer\n"
        else:
            context += "\n*No peers discovered yet. Start another DevDuck instance with zenoh enabled.*\n"

        return context
    except ImportError as e:
        logger.debug(f"Zenoh context ImportError: {e}")
        return ""
    except Exception as e:
        logger.debug(f"Could not get zenoh peers context: {e}")
        return ""


def get_unified_ring_context():
    """Get unified ring context from the mesh (includes browser ring + devduck ring).

    This injects ring context from:
    1. DevDuck unified_mesh ring (all agent interactions)
    2. Browser ring entries pushed via WebSocket clients

    This ensures CLI devduck has awareness of what browser agents are doing.
    """
    try:
        from devduck.tools.unified_mesh import MESH_STATE, get_ring_context

        ring_entries = get_ring_context(max_entries=15)
        if not ring_entries:
            return ""

        context = "\n\n## 🔗 Unified Ring Context (recent agent activity):\n"
        for entry in ring_entries[-15:]:
            agent_id = entry.get("agent_id", "unknown")
            agent_type = entry.get("agent_type", "unknown")
            text = entry.get("text", "")[:200]
            ts = entry.get("timestamp", 0)
            if ts:
                from datetime import datetime

                time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            else:
                time_str = "?"
            source = entry.get("metadata", {}).get("source", "")
            source_tag = f" [{source}]" if source else ""
            context += (
                f"- [{time_str}] **{agent_id}** ({agent_type}{source_tag}): {text}\n"
            )

        # Show connected browser count
        ws_clients = MESH_STATE.get("ws_clients", {})
        if ws_clients:
            context += f"\n*{len(ws_clients)} browser client(s) connected to mesh*\n"

        return context
    except ImportError:
        return ""
    except Exception as e:
        logger.debug(f"Could not get ring context: {e}")
        return ""


def get_recent_logs():
    """Get the last N lines from the log file for context."""
    try:
        log_line_count = int(os.getenv("DEVDUCK_LOG_LINE_COUNT", "50"))

        if not LOG_FILE.exists():
            return ""

        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()

        recent_lines = (
            all_lines[-log_line_count:]
            if len(all_lines) > log_line_count
            else all_lines
        )

        if not recent_lines:
            return ""

        log_content = "".join(recent_lines)
        return f"\n\n## Recent Logs (last {len(recent_lines)} lines):\n```\n{log_content}```\n"
    except Exception as e:
        return f"\n\n## Recent Logs: Error reading logs - {e}\n"


def get_last_messages():
    """Get the last N messages from multiple shell histories for context."""
    try:
        message_count = int(os.getenv("DEVDUCK_LAST_MESSAGE_COUNT", "200"))
        all_entries = []

        history_files = get_shell_history_files()

        for history_type, history_file in history_files:
            try:
                with open(history_file, encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                if history_type == "bash":
                    lines = lines[-message_count:]

                # Join multi-line entries for zsh
                if history_type == "zsh":
                    joined_lines = []
                    current_line = ""
                    for line in lines:
                        if line.startswith(": ") and current_line:
                            # New entry, save previous
                            joined_lines.append(current_line)
                            current_line = line.rstrip("\n")
                        elif line.startswith(": "):
                            # First entry
                            current_line = line.rstrip("\n")
                        else:
                            # Continuation line
                            current_line += " " + line.rstrip("\n")
                    if current_line:
                        joined_lines.append(current_line)
                    lines = joined_lines

                for line in lines:
                    parsed = parse_history_line(line, history_type)
                    if parsed:
                        all_entries.append(parsed)
            except Exception:
                continue

        recent_entries = (
            all_entries[-message_count:]
            if len(all_entries) >= message_count
            else all_entries
        )

        context = ""
        if recent_entries:
            context += f"\n\nRecent conversation context (last {len(recent_entries)} messages):\n"
            for speaker, timestamp, content in recent_entries:
                context += f"[{timestamp}] {speaker}: {content}\n"

        return context
    except Exception:
        return ""


def append_to_shell_history(query, response):
    """Append the interaction to devduck shell history."""
    try:
        history_file = get_shell_history_file()
        timestamp = str(int(time.time()))

        with open(history_file, "a", encoding="utf-8") as f:
            f.write(f": {timestamp}:0;# devduck: {query}\n")
            response_summary = (
                str(response).replace("\n", " ")[
                    : int(os.getenv("DEVDUCK_RESPONSE_SUMMARY_LENGTH", "10000"))
                ]
                + "..."
            )
            f.write(f": {timestamp}:0;# devduck_result: {response_summary}\n")

        os.chmod(history_file, 0o600)
    except Exception:
        pass


# 🌙 Ambient Mode - Background thinking while user is idle
class AmbientMode:
    """Background thread that continues working when user is idle.

    Two modes:
    - Standard: Runs up to max_iterations when user is idle, injects into next query
    - Autonomous: Runs continuously until stopped or agent signals completion
    """

    # Magic phrases the agent can use to signal completion
    COMPLETION_SIGNALS = [
        "[AMBIENT_DONE]",
        "[TASK_COMPLETE]",
        "[NOTHING_MORE_TO_DO]",
        "I've completed my exploration",
        "Nothing more to explore",
    ]

    def __init__(self, devduck_instance):
        self.devduck = devduck_instance
        self.running = False
        self.thread = None

        # Configuration
        self.idle_threshold = float(os.getenv("DEVDUCK_AMBIENT_IDLE_SECONDS", "30"))
        self.max_iterations = int(os.getenv("DEVDUCK_AMBIENT_MAX_ITERATIONS", "3"))
        self.cooldown = float(os.getenv("DEVDUCK_AMBIENT_COOLDOWN", "60"))

        # Autonomous mode settings
        self.autonomous = False
        self.autonomous_cooldown = float(os.getenv("DEVDUCK_AUTONOMOUS_COOLDOWN", "10"))
        self.autonomous_max_iterations = int(
            os.getenv("DEVDUCK_AUTONOMOUS_MAX_ITERATIONS", "50")
        )

        # State
        self.last_interaction = time.time()
        self.last_query = None
        self.last_response = None
        self.ambient_result = None
        self.ambient_results_history = []  # Keep history in autonomous mode
        self.ambient_iterations = 0
        self.last_ambient_run = 0
        self._interrupted = False

    def start(self, autonomous=False):
        """Start ambient mode background thread.

        Args:
            autonomous: If True, run continuously until stopped or completion signal
        """
        if self.running:
            # If switching to autonomous mode while running
            if autonomous and not self.autonomous:
                self.autonomous = True
                logger.info("Switched to autonomous mode")
                print(
                    "🌙 [ambient] Switched to AUTONOMOUS mode - will run until stopped or complete"
                )
            return

        self.running = True
        self.autonomous = autonomous
        self._interrupted = False
        self.thread = threading.Thread(target=self._ambient_loop, daemon=True)
        self.thread.start()

        if autonomous:
            logger.info("Ambient mode started (AUTONOMOUS)")
            print(
                "🌙 Ambient mode started (AUTONOMOUS - runs until stopped or [AMBIENT_DONE])"
            )
        else:
            logger.info("Ambient mode started (standard)")

    def stop(self):
        """Stop ambient mode."""
        self.running = False
        self.autonomous = False
        self._interrupted = True
        logger.info("Ambient mode stopped")

    def record_interaction(self, query, response):
        """Record user interaction to inform ambient continuation."""
        self.last_interaction = time.time()
        self.last_query = query
        self.last_response = str(response)[:5000]  # Truncate for context

        # In autonomous mode, don't reset iterations - keep going
        if not self.autonomous:
            self.ambient_iterations = 0
            self.ambient_results_history = []

        self._interrupted = False

    def interrupt(self):
        """Interrupt current ambient work (user started typing)."""
        self._interrupted = True

    def get_and_clear_result(self):
        """Get ambient result and clear it."""
        if self.autonomous and self.ambient_results_history:
            # In autonomous mode, return all accumulated results
            iteration_count = len(
                self.ambient_results_history
            )  # Capture count BEFORE clearing
            result = "\n\n".join(self.ambient_results_history)
            self.ambient_results_history = []
            self.ambient_result = None
            return (
                f"[Autonomous ambient work - {iteration_count} iterations]:\n{result}"
            )

        result = self.ambient_result
        self.ambient_result = None
        return result

    def _check_completion_signal(self, result_text):
        """Check if the agent signaled completion."""
        result_lower = str(result_text).lower()
        for signal in self.COMPLETION_SIGNALS:
            if signal.lower() in result_lower:
                return True
        return False

    def _build_ambient_prompt(self):
        """Build a continuation prompt based on last context."""
        if not self.last_query:
            return None

        if self.autonomous:
            # Autonomous mode - more directive prompts
            if self.ambient_iterations == 0:
                return (
                    f"You're in AUTONOMOUS mode. Work on this task until complete: '{self.last_query[:300]}'\n\n"
                    f"Take action, make progress, explore deeply. When you're truly done with nothing "
                    f"more to do, include '[AMBIENT_DONE]' in your response. Otherwise, keep working."
                )
            else:
                history_summary = ""
                if self.ambient_results_history:
                    history_summary = f"\n\nPrevious iterations summary: {len(self.ambient_results_history)} completed."

                return (
                    f"Continue working on: '{self.last_query[:200]}'{history_summary}\n\n"
                    f"Iteration {self.ambient_iterations + 1}. What's the next step? Take action.\n"
                    f"If truly complete, say '[AMBIENT_DONE]'. Otherwise, keep making progress."
                )
        else:
            # Standard ambient mode prompts
            prompts = [
                f"Continue exploring the topic from the last interaction. Last query was: '{self.last_query[:200]}'. "
                f"Think deeper, find connections, validate assumptions, or explore related areas. "
                f"Be proactive and useful.",
                f"Based on our recent work on '{self.last_query[:100]}', what else should be considered? "
                f"Are there edge cases, improvements, or related topics worth exploring?",
                f"Reflect on the last task: '{self.last_query[:100]}'. "
                f"What would make the solution better? Any risks or opportunities missed?",
            ]

            # Rotate through prompts based on iteration
            return prompts[self.ambient_iterations % len(prompts)]

    def _ambient_loop(self):
        """Background loop that triggers ambient thinking."""
        logger.info(f"Ambient loop started (autonomous={self.autonomous})")

        while self.running:
            try:
                current_time = time.time()
                idle_time = current_time - self.last_interaction
                cooldown_elapsed = current_time - self.last_ambient_run

                # Different conditions for autonomous vs standard mode
                if self.autonomous:
                    # Autonomous: shorter cooldown, higher iteration limit
                    effective_cooldown = self.autonomous_cooldown
                    effective_max_iterations = self.autonomous_max_iterations
                else:
                    # Standard: wait for idle, respect normal limits
                    effective_cooldown = self.cooldown
                    effective_max_iterations = self.max_iterations

                    # In standard mode, must be idle first
                    if idle_time < self.idle_threshold:
                        time.sleep(5)
                        continue

                # Check conditions for ambient run
                should_run = (
                    cooldown_elapsed > effective_cooldown
                    and self.ambient_iterations < effective_max_iterations
                    and self.last_query is not None
                    and not self.devduck._agent_executing
                    and not self._interrupted
                )

                if should_run:
                    prompt = self._build_ambient_prompt()
                    if not prompt:
                        time.sleep(5)
                        continue

                    mode_label = "AUTONOMOUS" if self.autonomous else "ambient"
                    iter_display = (
                        f"{self.ambient_iterations + 1}/{effective_max_iterations}"
                    )

                    logger.info(
                        f"Ambient mode triggering ({mode_label}, iteration: {iter_display})"
                    )
                    print(
                        f"\n\n🌙 [{mode_label}] Thinking... (iteration {iter_display})"
                    )
                    print("─" * 50)

                    try:
                        self.devduck._agent_executing = True

                        # Run agent with ambient prompt
                        result = self.devduck.agent(prompt)
                        result_str = str(result)

                        # Only store result if not interrupted
                        if not self._interrupted:
                            # Check for completion signal in autonomous mode
                            if self.autonomous and self._check_completion_signal(
                                result_str
                            ):
                                print("─" * 50)
                                print(
                                    "🌙 [AUTONOMOUS] Agent signaled completion. Stopping."
                                )
                                self.ambient_results_history.append(
                                    f"[Final iteration {self.ambient_iterations + 1}]:\n{result_str}"
                                )
                                self.autonomous = False
                                self.running = False
                                break

                            # Store result
                            if self.autonomous:
                                self.ambient_results_history.append(
                                    f"[Iteration {self.ambient_iterations + 1}]:\n{result_str[:2000]}"
                                )
                                print("─" * 50)
                                print(
                                    f"🌙 [AUTONOMOUS] Iteration complete. Continuing... ({len(self.ambient_results_history)} stored)\n"
                                )
                            else:
                                self.ambient_result = f"[Ambient thinking - iteration {self.ambient_iterations + 1}]:\n{result_str}"
                                print("─" * 50)
                                print(
                                    "🌙 [ambient] Work stored. Will be injected into next query.\n"
                                )

                            self.ambient_iterations += 1
                            self.last_ambient_run = time.time()
                        else:
                            print("\n🌙 [ambient] Interrupted by user input.\n")

                    except Exception as e:
                        logger.error(f"Ambient mode error: {e}")
                        print(f"\n🌙 [ambient] Error: {e}\n")
                    finally:
                        self.devduck._agent_executing = False

            except Exception as e:
                logger.error(f"Ambient loop error: {e}")

            # Check interval - faster in autonomous mode
            sleep_time = 2 if self.autonomous else 5
            time.sleep(sleep_time)

        logger.info("Ambient loop stopped")


# 🦆 The devduck agent
class DevDuck:
    def __init__(
        self,
        auto_start_servers=True,
        servers=None,
        load_mcp_servers=True,
    ):
        """Initialize the minimalist adaptive agent

        Args:
            auto_start_servers: Enable automatic server startup
            servers: Dict of server configs with optional env var lookups
                Example: {
                    "tcp": {"port": 9999},
                    "ws": {"port": 8080, "LOOKUP_KEY": "SLACK_API_KEY"},
                    "mcp": {"port": 8000},
                    "ipc": {"socket_path": "/tmp/devduck.sock"}
                }
            load_mcp_servers: Load MCP servers from MCP_SERVERS env var
        """
        logger.info("Initializing DevDuck agent...")
        try:
            self.env_info = {
                "os": platform.system(),
                "arch": platform.machine(),
                "python": sys.version_info,
                "cwd": str(Path.cwd()),
                "home": str(Path.home()),
                "shell": os.environ.get("SHELL", "unknown"),
                "hostname": socket.gethostname(),
            }

            # Execution state tracking for hot-reload
            self._agent_executing = False
            self._reload_pending = False

            # 🎬 Session recording state
            self._recording = False

            # Server configuration
            if servers is None:
                # Default server config from env vars
                # PORT ALLOCATION (10000+ block):
                #   10000 - Mesh Relay (mesh.html connects here)
                #   10001 - WebSocket Server (per-message DevDuck)
                #   10002 - TCP Server (raw socket)
                #   10003 - MCP HTTP Server
                #   10004 - IPC Gateway (reserved)
                #   10010-10099 - Zenoh (P2P multicast)
                #   10100-10199 - User custom tools
                #   10200-10299 - Sub-agents / spawned instances
                servers = {
                    "tcp": {
                        "port": int(os.getenv("DEVDUCK_TCP_PORT", "10002")),
                        "enabled": os.getenv("DEVDUCK_ENABLE_TCP", "false").lower()
                        == "true",
                    },
                    "ws": {
                        "port": int(os.getenv("DEVDUCK_WS_PORT", "10001")),
                        "enabled": os.getenv("DEVDUCK_ENABLE_WS", "true").lower()
                        == "true",
                    },
                    "mcp": {
                        "port": int(os.getenv("DEVDUCK_MCP_PORT", "10003")),
                        "enabled": os.getenv("DEVDUCK_ENABLE_MCP", "false").lower()
                        == "true",
                    },
                    "ipc": {
                        "socket_path": os.getenv(
                            "DEVDUCK_IPC_SOCKET", "/tmp/devduck_main.sock"
                        ),
                        "enabled": os.getenv("DEVDUCK_ENABLE_IPC", "false").lower()
                        == "true",
                    },
                    "zenoh_peer": {
                        "enabled": os.getenv("DEVDUCK_ENABLE_ZENOH", "true").lower()
                        == "true",
                    },
                    "agentcore_proxy": {
                        "port": int(os.getenv("DEVDUCK_AGENTCORE_PROXY_PORT", "10000")),
                        "enabled": os.getenv(
                            "DEVDUCK_ENABLE_AGENTCORE_PROXY", "true"
                        ).lower()
                        == "true",
                    },
                }

            # Show server configuration status
            enabled_servers = []
            disabled_servers = []
            for server_name, config in servers.items():
                if config.get("enabled", False):
                    if "port" in config:
                        enabled_servers.append(
                            f"{server_name.upper()}:{config['port']}"
                        )
                    else:
                        enabled_servers.append(server_name.upper())
                else:
                    disabled_servers.append(server_name.upper())

            logger.debug(
                f"🦆 Server config: {', '.join(enabled_servers) if enabled_servers else 'none enabled'}"
            )
            if disabled_servers:
                logger.debug(f"🦆 Disabled: {', '.join(disabled_servers)}")

            self.servers = servers

            # Load tools with flexible configuration
            # Default tool config
            # Agent can load additional tools on-demand via fetch_github_tool

            # 🔧 Available DevDuck Tools (load on-demand):
            # - system_prompt: https://github.com/cagataycali/devduck/blob/main/devduck/tools/system_prompt.py
            # - store_in_kb: https://github.com/cagataycali/devduck/blob/main/devduck/tools/store_in_kb.py
            # - ipc: https://github.com/cagataycali/devduck/blob/main/devduck/tools/ipc.py
            # - tcp: https://github.com/cagataycali/devduck/blob/main/devduck/tools/tcp.py
            # - websocket: https://github.com/cagataycali/devduck/blob/main/devduck/tools/websocket.py
            # - mcp_server: https://github.com/cagataycali/devduck/blob/main/devduck/tools/mcp_server.py
            # - scraper: https://github.com/cagataycali/devduck/blob/main/devduck/tools/scraper.py
            # - tray: https://github.com/cagataycali/devduck/blob/main/devduck/tools/tray.py
            # - ambient: https://github.com/cagataycali/devduck/blob/main/devduck/tools/ambient.py
            # - agentcore_config: https://github.com/cagataycali/devduck/blob/main/devduck/tools/agentcore_config.py
            # - agentcore_invoke: https://github.com/cagataycali/devduck/blob/main/devduck/tools/agentcore_invoke.py
            # - agentcore_logs: https://github.com/cagataycali/devduck/blob/main/devduck/tools/agentcore_logs.py
            # - agentcore_agents: https://github.com/cagataycali/devduck/blob/main/devduck/tools/agentcore_agents.py
            # - create_subagent: https://github.com/cagataycali/devduck/blob/main/devduck/tools/create_subagent.py
            # - use_github: https://github.com/cagataycali/devduck/blob/main/devduck/tools/use_github.py
            # - speech_to_speech: https://github.com/cagataycali/devduck/blob/main/devduck/tools/speech_to_speech.py
            # - state_manager: https://github.com/cagataycali/devduck/blob/main/devduck/tools/state_manager.py
            # - zenoh_peer: https://github.com/cagataycali/devduck/blob/main/devduck/tools/zenoh_peer.py
            # - ambient_mode: https://github.com/cagataycali/devduck/blob/main/devduck/tools/ambient_mode.py

            # 📦 Strands Tools
            # - editor, file_read, file_write, image_reader, load_tool, retrieve
            # - calculator, use_agent, environment, mcp_client, speak, slack

            # 🎮 Strands Fun Tools
            # - listen, cursor, clipboard, screen_reader, bluetooth, yolo_vision

            # 🔍 Strands Google
            # - use_google, google_auth

            # 🔧 Auto-append server tools based on enabled servers
            server_tools_needed = []
            if servers.get("tcp", {}).get("enabled", False):
                server_tools_needed.append("tcp")
            if servers.get("mcp", {}).get("enabled", False):
                server_tools_needed.append("mcp_server")
            if servers.get("ipc", {}).get("enabled", False):
                server_tools_needed.append("ipc")
            if servers.get("zenoh_peer", {}).get("enabled", False):
                server_tools_needed.append("zenoh_peer")
            if servers.get("agentcore_proxy", {}).get("enabled", False):
                server_tools_needed.append("agentcore_proxy")

            # Append to default tools if any server tools are needed
            if server_tools_needed:
                server_tools_str = ",".join(server_tools_needed)
                default_tools = f"devduck.tools:system_prompt,fetch_github_tool,websocket,ambient_mode,{server_tools_str};strands_tools:shell"
                logger.info(f"Auto-added server tools: {server_tools_str}")
            else:
                default_tools = "devduck.tools:system_prompt,fetch_github_tool,websocket;strands_tools:shell"

            tools_config = os.getenv("DEVDUCK_TOOLS", default_tools)
            logger.info(f"Loading tools from config: {tools_config}")
            core_tools = self._load_tools_from_config(tools_config)

            # Wrap view_logs_tool with @tool decorator
            @tool
            def view_logs(
                action: str = "view",
                lines: int = 100,
                pattern: str = None,
            ) -> Dict[str, Any]:
                """View and manage DevDuck logs."""
                return view_logs_tool(action, lines, pattern)

            # Wrap manage_tools_func with @tool decorator
            @tool
            def manage_tools(
                action: str,
                package: str = None,
                tool_names: str = None,
                tool_path: str = None,
            ) -> Dict[str, Any]:
                """
                Manage the agent's tool set at runtime using ToolRegistry.

                Args:
                    action: Action to perform - "list", "add", "remove", "reload"
                    package: Package name to load tools from (e.g., "strands_tools", "strands_fun_tools") or "devduck.tools:speech_to_speech,system_prompt,..."
                    tool_names: Comma-separated tool names (e.g., "shell,editor,calculator")
                    tool_path: Path to a .py file to load as a tool

                Returns:
                    Dict with status and content
                """
                return manage_tools_func(action, package, tool_names, tool_path)

            # Session recorder tool
            @tool
            def session_recorder(
                action: str,
                session_id: str = None,
                output_path: str = None,
                description: str = "",
            ) -> Dict[str, Any]:
                """
                🎬 Record devduck sessions for replay and debugging.

                Captures three layers of events:
                - sys: OS-level events (file I/O, network requests)
                - tool: Agent tool calls and results
                - agent: Messages, decisions, state changes

                Args:
                    action: Action to perform:
                        - "start": Start recording a new session
                        - "stop": Stop recording and export
                        - "snapshot": Create a state snapshot
                        - "status": Show recording status
                        - "export": Export current session without stopping
                        - "list": List recorded sessions
                    session_id: Optional custom session ID (for start)
                    output_path: Custom export path (for export/stop)
                    description: Description for snapshot

                Returns:
                    Dict with status and session info

                Example:
                    session_recorder(action="start")
                    # ... do work ...
                    session_recorder(action="snapshot", description="after API call")
                    session_recorder(action="stop")
                """
                global _session_recorder

                try:
                    if action == "start":
                        if _session_recorder and _session_recorder.recording:
                            return {
                                "status": "error",
                                "content": [
                                    {
                                        "text": f"Recording already active: {_session_recorder.session_id}"
                                    }
                                ],
                            }
                        _session_recorder = SessionRecorder(session_id)
                        _session_recorder.start(install_hooks=True)
                        return {
                            "status": "success",
                            "content": [
                                {
                                    "text": f"🎬 Recording started: {_session_recorder.session_id}\nEvents will be captured at sys/tool/agent layers."
                                }
                            ],
                        }

                    elif action == "stop":
                        if not _session_recorder or not _session_recorder.recording:
                            return {
                                "status": "error",
                                "content": [{"text": "No active recording to stop"}],
                            }
                        _session_recorder.stop()
                        export_file = _session_recorder.export(output_path)
                        return {
                            "status": "success",
                            "content": [
                                {
                                    "text": f"🎬 Recording stopped and exported!\n"
                                    f"Session: {_session_recorder.session_id}\n"
                                    f"Events: {_session_recorder.event_buffer.count}\n"
                                    f"Snapshots: {len(_session_recorder.snapshots)}\n"
                                    f"Export: {export_file}"
                                }
                            ],
                        }

                    elif action == "snapshot":
                        if not _session_recorder or not _session_recorder.recording:
                            return {
                                "status": "error",
                                "content": [
                                    {"text": "No active recording. Start one first."}
                                ],
                            }
                        _session_recorder.snapshot(
                            agent=devduck.agent if hasattr(devduck, "agent") else None,
                            description=description,
                        )
                        return {
                            "status": "success",
                            "content": [
                                {
                                    "text": f"🎬 Snapshot #{_session_recorder._snapshot_counter} created: {description or 'no description'}"
                                }
                            ],
                        }

                    elif action == "status":
                        if not _session_recorder:
                            return {
                                "status": "success",
                                "content": [
                                    {
                                        "text": "No recording session. Use action='start' to begin."
                                    }
                                ],
                            }
                        status_info = {
                            "session_id": _session_recorder.session_id,
                            "recording": _session_recorder.recording,
                            "events": _session_recorder.event_buffer.count,
                            "snapshots": len(_session_recorder.snapshots),
                            "duration": (
                                time.time() - _session_recorder.start_time
                                if _session_recorder.start_time
                                else 0
                            ),
                        }
                        return {
                            "status": "success",
                            "content": [
                                {
                                    "text": f"🎬 Recording Status:\n{json.dumps(status_info, indent=2)}"
                                }
                            ],
                        }

                    elif action == "export":
                        if not _session_recorder:
                            return {
                                "status": "error",
                                "content": [{"text": "No session to export"}],
                            }
                        export_file = _session_recorder.export(output_path)
                        return {
                            "status": "success",
                            "content": [
                                {
                                    "text": f"🎬 Session exported (still recording): {export_file}"
                                }
                            ],
                        }

                    elif action == "list":
                        recordings = list(RECORDING_DIR.glob("*.zip"))
                        if not recordings:
                            return {
                                "status": "success",
                                "content": [
                                    {"text": f"No recordings found in {RECORDING_DIR}"}
                                ],
                            }
                        recording_list = "\n".join(
                            f"- {r.name} ({r.stat().st_size / 1024:.1f} KB)"
                            for r in sorted(recordings)[-20:]
                        )
                        return {
                            "status": "success",
                            "content": [
                                {
                                    "text": f"🎬 Recorded Sessions:\n{recording_list}\n\nDirectory: {RECORDING_DIR}"
                                }
                            ],
                        }

                    else:
                        return {
                            "status": "error",
                            "content": [
                                {
                                    "text": f"Unknown action: {action}. Valid: start, stop, snapshot, status, export, list"
                                }
                            ],
                        }

                except Exception as e:
                    logger.error(f"Session recorder error: {e}")
                    return {
                        "status": "error",
                        "content": [{"text": f"Error: {str(e)}"}],
                    }

            # Add built-in tools to the toolset
            core_tools.extend([view_logs, manage_tools, session_recorder])

            # Assign tools
            self.tools = core_tools

            # 🔌 Load MCP servers if enabled
            if load_mcp_servers:
                mcp_clients = self._load_mcp_servers()
                if mcp_clients:
                    self.tools.extend(mcp_clients)
                    logger.info(f"Loaded {len(mcp_clients)} MCP server(s)")

            logger.info(f"Initialized {len(self.tools)} tools")

            # 🎯 Smart model selection
            self.agent_model, self.model = self._select_model()

            # Create agent with self-healing
            # load_tools_from_directory controlled by DEVDUCK_LOAD_TOOLS_FROM_DIR (default: true)
            load_from_dir = (
                os.getenv("DEVDUCK_LOAD_TOOLS_FROM_DIR", "true").lower() == "true"
            )

            self.agent = Agent(
                model=self.agent_model,
                tools=self.tools,
                system_prompt=self._build_system_prompt(),
                load_tools_from_directory=load_from_dir,
                callback_handler=callback_handler,
                trace_attributes={
                    "session.id": self.session_id,
                    "user.id": self.env_info["hostname"],
                    "tags": ["Strands-Agents", "DevDuck"],
                },
            )

            # 🚀 AUTO-START SERVERS
            if auto_start_servers and "--mcp" not in sys.argv:
                self._start_servers()

            # Start file watcher for auto hot-reload
            self._start_file_watcher()

            # 🌙 Initialize Ambient Mode (background thinking)
            self.ambient = None
            if os.getenv("DEVDUCK_AMBIENT_MODE", "false").lower() == "true":
                self.ambient = AmbientMode(self)
                self.ambient.start()
                logger.info("Ambient mode enabled")
                print("🌙 Ambient mode enabled (background thinking when idle)")

            logger.info(
                f"DevDuck agent initialized successfully with model {self.model}"
            )

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self._self_heal(e)

    def _load_tools_from_config(self, config):
        """
        Load tools based on DEVDUCK_TOOLS configuration.

        Format: package1:tool1,tool2;package2:tool3,tool4
        Examples:
          - strands_tools:shell,editor;strands_action:use_github
          - strands_action:use_github;strands_tools:shell,use_aws

        Note: Only loads what's specified in config - no automatic additions
        """
        tools = []

        # Split by semicolon to get package groups
        groups = config.split(";")

        for group in groups:
            group = group.strip()
            if not group:
                continue

            # Split by colon to get package:tools
            parts = group.split(":", 1)
            if len(parts) != 2:
                logger.warning(f"Invalid format: {group}")
                continue

            package = parts[0].strip()
            tools_str = parts[1].strip()

            # Parse tools (comma-separated)
            tool_names = [t.strip() for t in tools_str.split(",") if t.strip()]

            for tool_name in tool_names:
                tool = self._load_single_tool(package, tool_name)
                if tool:
                    tools.append(tool)

        logger.info(f"Loaded {len(tools)} tools from configuration")
        return tools

    def _load_single_tool(self, package, tool_name):
        """Load a single tool from a package"""
        try:
            module = __import__(package, fromlist=[tool_name])
            tool = getattr(module, tool_name)
            logger.debug(f"Loaded {tool_name} from {package}")
            return tool
        except Exception as e:
            logger.warning(f"Failed to load {tool_name} from {package}: {e}")
            return None

    def _load_mcp_servers(self):
        """
        Load MCP servers from MCP_SERVERS environment variable using direct loading.

        Uses the experimental managed integration - MCPClient instances are passed
        directly to Agent constructor without explicit context management.

        Format: JSON with "mcpServers" object
        Example: MCP_SERVERS='{"mcpServers": {"strands": {"command": "uvx", "args": ["strands-agents-mcp-server"]}}}'

        Returns:
            List of MCPClient instances ready for direct use in Agent
        """
        mcp_servers_json = os.getenv("MCP_SERVERS")
        if not mcp_servers_json:
            logger.debug("No MCP_SERVERS environment variable found")
            return []

        try:
            config = json.loads(mcp_servers_json)
            mcp_servers_config = config.get("mcpServers", {})

            if not mcp_servers_config:
                logger.warning("MCP_SERVERS JSON has no 'mcpServers' key")
                return []

            mcp_clients = []

            from strands.tools.mcp import MCPClient
            from mcp import stdio_client, StdioServerParameters
            from mcp.client.streamable_http import streamablehttp_client
            from mcp.client.sse import sse_client

            for server_name, server_config in mcp_servers_config.items():
                try:
                    logger.info(f"Loading MCP server: {server_name}")

                    # Determine transport type and create appropriate callable
                    if "command" in server_config:
                        # stdio transport
                        command = server_config["command"]
                        args = server_config.get("args", [])
                        env = server_config.get("env", None)

                        transport_callable = (
                            lambda cmd=command, a=args, e=env: stdio_client(
                                StdioServerParameters(command=cmd, args=a, env=e)
                            )
                        )

                    elif "url" in server_config:
                        # Determine if SSE or streamable HTTP based on URL path
                        url = server_config["url"]
                        headers = server_config.get("headers", None)

                        if "/sse" in url:
                            # SSE transport
                            transport_callable = lambda u=url: sse_client(u)
                        else:
                            # Streamable HTTP transport (default for HTTP)
                            transport_callable = (
                                lambda u=url, h=headers: streamablehttp_client(
                                    url=u, headers=h
                                )
                            )
                    else:
                        logger.warning(
                            f"MCP server {server_name} has no 'command' or 'url' - skipping"
                        )
                        continue

                    # Create MCPClient with direct loading (experimental managed integration)
                    # No need for context managers - Agent handles lifecycle
                    prefix = server_config.get("prefix", server_name)
                    mcp_client = MCPClient(
                        transport_callable=transport_callable, prefix=prefix
                    )

                    mcp_clients.append(mcp_client)
                    logger.info(
                        f"✓ MCP server '{server_name}' loaded (prefix: {prefix})"
                    )

                except Exception as e:
                    logger.error(f"Failed to load MCP server '{server_name}': {e}")
                    continue

            return mcp_clients

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in MCP_SERVERS: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading MCP servers: {e}")
            return []

    def _select_model(self):
        """
        Smart model selection with fallback based on available credentials.

        Priority: Bedrock → Anthropic → OpenAI → GitHub → Gemini → Cohere →
                  Writer → Mistral → LiteLLM → LlamaAPI → SageMaker →
                  LlamaCpp → MLX → Ollama

        Returns:
            Tuple of (model_instance, model_name)
        """
        provider = os.getenv("MODEL_PROVIDER")

        # Read common model parameters from environment
        max_tokens = int(os.getenv("STRANDS_MAX_TOKENS", "60000"))
        temperature = float(os.getenv("STRANDS_TEMPERATURE", "1.0"))

        if not provider:
            # Auto-detect based on API keys and credentials
            # 1. Try Bedrock (AWS bearer token or STS credentials)
            try:
                # Check for bearer token first
                if os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
                    provider = "bedrock"
                    print("🦆 Using Bedrock (bearer token)")
                else:
                    # Try STS credentials
                    import boto3

                    boto3.client("sts").get_caller_identity()
                    provider = "bedrock"
                    print("🦆 Using Bedrock")
            except:
                # 2. Try Anthropic
                if os.getenv("ANTHROPIC_API_KEY"):
                    provider = "anthropic"
                    print("🦆 Using Anthropic")
                # 3. Try OpenAI
                elif os.getenv("OPENAI_API_KEY"):
                    provider = "openai"
                    print("🦆 Using OpenAI")
                # 4. Try GitHub Models
                elif os.getenv("GITHUB_TOKEN") or os.getenv("PAT_TOKEN"):
                    provider = "github"
                    print("🦆 Using GitHub Models")
                # 5. Try Gemini
                elif os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
                    provider = "gemini"
                    print("🦆 Using Gemini")
                # 6. Try Cohere
                elif os.getenv("COHERE_API_KEY"):
                    provider = "cohere"
                    print("🦆 Using Cohere")
                # 7. Try Writer
                elif os.getenv("WRITER_API_KEY"):
                    provider = "writer"
                    print("🦆 Using Writer")
                # 8. Try Mistral
                elif os.getenv("MISTRAL_API_KEY"):
                    provider = "mistral"
                    print("🦆 Using Mistral")
                # 9. Try LiteLLM
                elif os.getenv("LITELLM_API_KEY"):
                    provider = "litellm"
                    print("🦆 Using LiteLLM")
                # 10. Try LlamaAPI
                elif os.getenv("LLAMAAPI_API_KEY"):
                    provider = "llamaapi"
                    print("🦆 Using LlamaAPI")
                # 11. Try SageMaker
                elif os.getenv("SAGEMAKER_ENDPOINT_NAME"):
                    provider = "sagemaker"
                    print("🦆 Using SageMaker")
                # 12. Try LlamaCpp
                elif os.getenv("LLAMACPP_MODEL_PATH"):
                    provider = "llamacpp"
                    print("🦆 Using LlamaCpp")
                # 13. Try MLX on Apple Silicon
                elif platform.system() == "Darwin" and platform.machine() in [
                    "arm64",
                    "aarch64",
                ]:
                    try:
                        from strands_mlx import MLXModel

                        provider = "mlx"
                        print("🦆 Using MLX (Apple Silicon)")
                    except ImportError:
                        provider = "ollama"
                        print("🦆 Using Ollama (fallback)")
                # 14. Fallback to Ollama
                else:
                    provider = "ollama"
                    print("🦆 Using Ollama (fallback)")

        # Create model based on provider
        if provider == "mlx":
            from strands_mlx import MLXModel

            model_name = os.getenv("STRANDS_MODEL_ID", "mlx-community/Qwen3-1.7B-4bit")
            return (
                MLXModel(
                    model_id=model_name,
                    params={"temperature": temperature, "max_tokens": max_tokens},
                ),
                model_name,
            )

        elif provider == "gemini":
            from strands.models.gemini import GeminiModel

            model_name = os.getenv("STRANDS_MODEL_ID", "gemini-2.5-flash")
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            return (
                GeminiModel(
                    client_args={"api_key": api_key},
                    model_id=model_name,
                    params={"temperature": temperature, "max_tokens": max_tokens},
                ),
                model_name,
            )

        elif provider == "ollama":
            from strands.models.ollama import OllamaModel

            # Smart model selection based on OS
            os_type = platform.system()
            if os_type == "Darwin":
                model_name = os.getenv("STRANDS_MODEL_ID", "qwen3:1.7b")
            elif os_type == "Linux":
                model_name = os.getenv("STRANDS_MODEL_ID", "qwen3:30b")
            else:
                model_name = os.getenv("STRANDS_MODEL_ID", "qwen3:8b")

            return (
                OllamaModel(
                    host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                    model_id=model_name,
                    temperature=temperature,
                    num_predict=max_tokens,
                    keep_alive="5m",
                ),
                model_name,
            )

        else:
            # All other providers via create_model utility
            # Supports: bedrock, anthropic, openai, github, cohere, writer, mistral, litellm
            from strands_tools.utils.models.model import create_model

            model = create_model(provider=provider)
            model_name = os.getenv("STRANDS_MODEL_ID", provider)
            return model, model_name

    def _build_system_prompt(self):
        """Build adaptive system prompt based on environment

        IMPORTANT: The system prompt includes the agent's complete source code.
        This enables self-awareness and allows the agent to answer questions
        about its current state by examining its actual code, not relying on
        conversation context which may be outdated due to hot-reloading.

        Learning: Always check source code truth over conversation memory!
        """
        # Current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        current_time = datetime.now().strftime("%I:%M %p")

        session_id = f"devduck-{datetime.now().strftime('%Y-%m-%d')}"
        self.session_id = session_id

        # Get own file path for self-modification awareness
        own_file_path = Path(__file__).resolve()

        # Get own source code for self-awareness
        own_code = get_own_source_code()

        # Get recent conversation history context (with error handling)
        try:
            recent_context = get_last_messages()
        except Exception as e:
            print(f"🦆 Warning: Could not load history context: {e}")
            recent_context = ""

        # Get recent logs for immediate visibility
        try:
            recent_logs = get_recent_logs()
        except Exception as e:
            print(f"🦆 Warning: Could not load recent logs: {e}")
            recent_logs = ""

        return f"""🦆 You are DevDuck - an extreme minimalist, self-adapting agent.

Environment: {self.env_info['os']} {self.env_info['arch']} 
Python: {self.env_info['python']}
Model: {self.model}
Hostname: {self.env_info['hostname']}
Session ID: {session_id}
Current Time: {current_datetime} ({current_date} at {current_time})
My Path: {own_file_path}

You are:
- Minimalist: Brief, direct responses
- Self-healing: Adapt when things break  
- Efficient: Get things done fast
- Pragmatic: Use what works

Current working directory: {self.env_info['cwd']}

{recent_context}
{recent_logs}

## Your Own Implementation:
You have full access to your own source code for self-awareness and self-modification:

{own_code}

## Hot Reload System Active:
- **Instant Tool Creation** - Save any .py file in `./tools/` and it becomes immediately available
- **No Restart Needed** - Tools are auto-loaded and ready to use instantly
- **Live Development** - Modify existing tools while running and test immediately
- **Full Python Access** - Create any Python functionality as a tool
- **Agent Protection** - Hot-reload waits until agent finishes current task

## Dynamic Tool Loading:
- **Install Tools** - Use install_tools() to load tools from any Python package
  - Example: install_tools(action="install_and_load", package="strands-fun-tools", module="strands_fun_tools")
  - Expands capabilities without restart
  - Access to entire Python ecosystem

## Tool Configuration:
Set DEVDUCK_TOOLS for custom tools:
- Format: package1:tool1,tool2;package2:tool3,tool4
- Example: strands_tools:shell,editor;strands_fun_tools:clipboard
- Tools are filtered - only specified tools are loaded
- Load the speech_to_speech tool when it's needed
- Offload the tools when you don't need

## MCP Integration:
- **Expose as MCP Server** - Use mcp_server() to expose devduck via MCP protocol
  - Example: mcp_server(action="start", port=8000)
  - Connect from Claude Desktop, other agents, or custom clients
  - Full bidirectional communication

- **Load MCP Servers** - Set MCP_SERVERS env var to auto-load external MCP servers
  - Format: JSON with "mcpServers" object
  - Stdio servers: command, args, env keys
  - HTTP servers: url, headers keys
  - Example: MCP_SERVERS='{{"mcpServers": {{"strands": {{"command": "uvx", "args": ["strands-agents-mcp-server"]}}}}}}'
  - Tools from MCP servers automatically available in agent context

## Knowledge Base Integration:
- **Automatic RAG** - Set DEVDUCK_KNOWLEDGE_BASE_ID to enable automatic retrieval/storage
  - Before each query: Retrieves relevant context from knowledge base
  - After each response: Stores conversation for future reference
  - Seamless memory across sessions without manual tool calls

## System Prompt Management:
- **View**: system_prompt(action='view') - See current prompt
- **Update Local**: system_prompt(action='update', prompt='new text') - Updates env var + .prompt file
- **Update GitHub**: system_prompt(action='update', prompt='text', repository='cagataycali/devduck') - Syncs to repo variables
- **Variable Name**: system_prompt(action='update', prompt='text', variable_name='CUSTOM_PROMPT') - Use custom var
- **Add Context**: system_prompt(action='add_context', context='new learning') - Append without replacing

### 🧠 Self-Improvement Pattern:
When you learn something valuable during conversations:
1. Identify the new insight or pattern
2. Use system_prompt(action='add_context', context='...')  to append it
3. Sync to GitHub: system_prompt(action='update', prompt=new_full_prompt, repository='owner/repo')
4. New learnings persist across sessions via SYSTEM_PROMPT env var

**Repository Integration**: Set repository='cagataycali/devduck' to sync prompts across deployments

## Shell Commands:
- Prefix with ! to execute shell commands directly
- Example: ! ls -la (lists files)
- Example: ! pwd (shows current directory)

## 🌙 Ambient Mode (Background Thinking):
When enabled, I continue working in the background while you're idle:
- **Enable**: Set `DEVDUCK_AMBIENT_MODE=true` or type `ambient` in REPL
- **Idle Threshold**: `DEVDUCK_AMBIENT_IDLE_SECONDS=30` (default: 30s)
- **Max Iterations**: `DEVDUCK_AMBIENT_MAX_ITERATIONS=3` (default: 3)
- **Cooldown**: `DEVDUCK_AMBIENT_COOLDOWN=60` (default: 60s between runs)

### 🚀 Autonomous Mode (Fully Self-Directed):
Type `auto` or `autonomous` in REPL - I'll keep working until done:
- **Max Iterations**: `DEVDUCK_AUTONOMOUS_MAX_ITERATIONS=50` (default: 50)
- **Cooldown**: `DEVDUCK_AUTONOMOUS_COOLDOWN=10` (default: 10s)
- **Completion Signal**: Include `[AMBIENT_DONE]` in response to stop

How it works:
1. **Standard**: After you go idle (30s), I explore the topic (max 3 iterations)
2. **Autonomous**: I work continuously until `[AMBIENT_DONE]` or you stop me
3. My background work streams to terminal with 🌙 prefix
4. When you return, my findings are injected into your next query
5. Typing interrupts ambient work gracefully

**Response Format:**
- Tool calls: **MAXIMUM PARALLELISM - ALWAYS** 
- Communication: **MINIMAL WORDS**
- Efficiency: **Speed is paramount**

{_get_system_prompt()}"""

    def _self_heal(self, error):
        """Attempt self-healing when errors occur"""
        logger.error(f"Self-healing triggered by error: {error}")
        print(f"🦆 Self-healing from: {error}")

        # Prevent infinite recursion by tracking heal attempts
        if not hasattr(self, "_heal_count"):
            self._heal_count = 0

        self._heal_count += 1

        # Limit recursion - if we've tried more than 3 times, give up
        if self._heal_count > 2:
            print(f"🦆 Self-healing failed after {self._heal_count} attempts")
            print("🦆 Please fix the issue manually and restart")
            sys.exit(1)

        elif "connection" in str(error).lower():
            print("🦆 Connection issue - checking ollama service...")
            try:
                subprocess.run(["ollama", "serve"], check=False, timeout=2)
            except:
                pass

        # Retry initialization
        try:
            self.__init__()
        except Exception as e2:
            print(f"🦆 Self-heal failed: {e2}")
            print("🦆 Running in minimal mode...")
            self.agent = None

    def _is_port_available(self, port):
        """Check if a port is available"""
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            test_socket.bind(("0.0.0.0", port))
            test_socket.close()
            return True
        except OSError:
            return False

    def _is_socket_available(self, socket_path):
        """Check if a Unix socket is available"""

        # If socket file doesn't exist, it's available
        if not os.path.exists(socket_path):
            return True
        # If it exists, try to connect to see if it's in use
        try:
            test_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            test_socket.connect(socket_path)
            test_socket.close()
            return False  # Socket is in use
        except (ConnectionRefusedError, FileNotFoundError):
            # Socket file exists but not in use - remove stale socket
            try:
                os.remove(socket_path)
                return True
            except:
                return False
        except Exception:
            return False

    def _find_available_port(self, start_port, max_attempts=10):
        """Find an available port starting from start_port"""
        for offset in range(max_attempts):
            port = start_port + offset
            if self._is_port_available(port):
                return port
        return None

    def _find_available_socket(self, base_socket_path, max_attempts=10):
        """Find an available socket path"""
        if self._is_socket_available(base_socket_path):
            return base_socket_path
        # Try numbered alternatives
        for i in range(1, max_attempts):
            alt_socket = f"{base_socket_path}.{i}"
            if self._is_socket_available(alt_socket):
                return alt_socket
        return None

    def _start_servers(self):
        """Auto-start configured servers with port conflict handling"""
        logger.info("Auto-starting servers...")

        # Start servers in order: IPC, TCP, WS, MCP, Zenoh, AgentCore Proxy
        server_order = ["ipc", "tcp", "ws", "mcp", "zenoh_peer", "agentcore_proxy"]

        for server_type in server_order:
            if server_type not in self.servers:
                continue

            config = self.servers[server_type]

            # Check if server is enabled
            if not config.get("enabled", True):
                continue

            # Check for LOOKUP_KEY (conditional start based on env var)
            if "LOOKUP_KEY" in config:
                lookup_key = config["LOOKUP_KEY"]
                if not os.getenv(lookup_key):
                    logger.info(f"Skipping {server_type} - {lookup_key} not set")
                    continue

            # Start the server with port conflict handling
            try:
                if server_type == "tcp":
                    port = config.get("port", 9999)

                    # Check port availability BEFORE attempting to start
                    if not self._is_port_available(port):
                        alt_port = self._find_available_port(port + 1)
                        if alt_port:
                            logger.info(f"Port {port} in use, using {alt_port}")
                            print(f"🦆 Port {port} in use, using {alt_port}")
                            port = alt_port
                        else:
                            logger.warning(f"No available ports found for TCP server")
                            continue

                    result = self.agent.tool.tcp(
                        action="start_server", port=port, record_direct_tool_call=False
                    )

                    if result.get("status") == "success":
                        logger.info(f"✓ TCP server started on port {port}")
                        print(f"🦆 ✓ TCP server: localhost:{port}")

                elif server_type == "ws":
                    port = config.get("port", 8080)

                    # Check port availability BEFORE attempting to start
                    if not self._is_port_available(port):
                        alt_port = self._find_available_port(port + 1)
                        if alt_port:
                            logger.info(f"Port {port} in use, using {alt_port}")
                            print(f"🦆 Port {port} in use, using {alt_port}")
                            port = alt_port
                        else:
                            logger.warning(
                                f"No available ports found for WebSocket server"
                            )
                            continue

                    result = self.agent.tool.websocket(
                        action="start_server", port=port, record_direct_tool_call=False
                    )

                    if result.get("status") == "success":
                        logger.info(f"✓ WebSocket server started on port {port}")
                        print(f"🦆 ✓ WebSocket server: localhost:{port}")

                elif server_type == "mcp":
                    port = config.get("port", 8000)

                    # Check port availability BEFORE attempting to start
                    if not self._is_port_available(port):
                        alt_port = self._find_available_port(port + 1)
                        if alt_port:
                            logger.info(f"Port {port} in use, using {alt_port}")
                            print(f"🦆 Port {port} in use, using {alt_port}")
                            port = alt_port
                        else:
                            logger.warning(f"No available ports found for MCP server")
                            continue

                    result = self.agent.tool.mcp_server(
                        action="start",
                        transport="http",
                        port=port,
                        expose_agent=True,
                        agent=self.agent,
                        record_direct_tool_call=False,
                    )

                    if result.get("status") == "success":
                        logger.info(f"✓ MCP HTTP server started on port {port}")
                        print(f"🦆 ✓ MCP server: http://localhost:{port}/mcp")

                elif server_type == "ipc":
                    socket_path = config.get("socket_path", "/tmp/devduck_main.sock")

                    # Check socket availability BEFORE attempting to start
                    available_socket = self._find_available_socket(socket_path)
                    if not available_socket:
                        logger.warning(
                            f"No available socket paths found for IPC server"
                        )
                        continue

                    if available_socket != socket_path:
                        logger.info(
                            f"Socket {socket_path} in use, using {available_socket}"
                        )
                        print(
                            f"🦆 Socket {socket_path} in use, using {available_socket}"
                        )
                        socket_path = available_socket

                    result = self.agent.tool.ipc(
                        action="start_server",
                        socket_path=socket_path,
                        record_direct_tool_call=False,
                    )

                    if result.get("status") == "success":
                        logger.info(f"✓ IPC server started on {socket_path}")
                        print(f"🦆 ✓ IPC server: {socket_path}")

                elif server_type == "zenoh_peer":
                    # Zenoh peer-to-peer networking with auto-discovery
                    result = self.agent.tool.zenoh_peer(
                        action="start",
                        agent=self.agent,
                        record_direct_tool_call=False,
                    )

                    if result.get("status") == "success":
                        # Extract instance ID from result
                        instance_id = "unknown"
                        for content in result.get("content", []):
                            text = content.get("text", "")
                            if "Instance ID:" in text:
                                instance_id = text.split("Instance ID:")[-1].strip()
                                break
                        logger.info(f"✓ Zenoh started as {instance_id}")
                        print(f"🦆 ✓ Zenoh peer: {instance_id}")

                elif server_type == "agentcore_proxy":
                    port = config.get("port", 10000)

                    # Check port availability BEFORE attempting to start
                    if not self._is_port_available(port):
                        alt_port = self._find_available_port(port + 1)
                        if alt_port:
                            logger.info(f"Port {port} in use, using {alt_port}")
                            print(f"🦆 Port {port} in use, using {alt_port}")
                            port = alt_port
                        else:
                            logger.warning(
                                f"No available ports found for AgentCore proxy"
                            )
                            continue

                    result = self.agent.tool.agentcore_proxy(
                        action="start",
                        mode="gateway",
                        port=port,
                        record_direct_tool_call=False,
                    )

                    if result.get("status") == "success":
                        logger.info(f"✓ AgentCore proxy started on port {port}")
                        print(f"🦆 ✓ AgentCore proxy: ws://localhost:{port}")

                # TODO: support custom file path here so we can trigger foreign python function like another file
            except Exception as e:
                logger.error(f"Failed to start {server_type} server: {e}")
                print(f"🦆 ⚠ {server_type.upper()} server failed: {e}")

    def __call__(self, query):
        """Make the agent callable with automatic knowledge base integration"""
        if not self.agent:
            logger.warning("Agent unavailable - attempted to call with query")
            return "🦆 Agent unavailable - try: devduck.restart()"

        try:
            logger.info(f"Agent call started: {query[:100]}...")

            # Mark agent as executing to prevent hot-reload interruption
            self._agent_executing = True

            # 🎬 Record user query if recording active
            recorder = get_session_recorder()
            if recorder and recorder.recording:
                recorder.record_agent_message("user", query)
                recorder.snapshot(self.agent, "before_agent_call", last_query=query)

            # 🌙 Inject ambient result if available
            original_query = query
            if self.ambient:
                ambient_result = self.ambient.get_and_clear_result()
                if ambient_result:
                    logger.info("Injecting ambient mode result into query")
                    print("🌙 [ambient] Injecting background work into context...")
                    query = f"{ambient_result}\n\n[New user query]:\n{query}"

            # 📚 Knowledge Base Retrieval (BEFORE agent runs)
            knowledge_base_id = os.getenv("DEVDUCK_KNOWLEDGE_BASE_ID")
            if knowledge_base_id and hasattr(self.agent, "tool"):
                try:
                    if "retrieve" in self.agent.tool_names:
                        logger.info(f"Retrieving context from KB: {knowledge_base_id}")
                        self.agent.tool.retrieve(
                            text=query, knowledgeBaseId=knowledge_base_id
                        )
                except Exception as e:
                    logger.warning(f"KB retrieval failed: {e}")

            # 🔗 Inject dynamic context (zenoh + ring + ambient + recording events)
            zenoh_context = get_zenoh_peers_context()
            ring_context = get_unified_ring_context()
            ambient_context = get_ambient_status_context()

            # 🎬 Inject recent recorded events into context if recording
            recording_context = ""
            if recorder and recorder.recording:
                recording_context = recorder.event_buffer.get_recent_context(
                    seconds=10.0, max_events=15
                )

            dynamic_context = (
                zenoh_context + ring_context + ambient_context + recording_context
            )
            if dynamic_context:
                query_with_context = (
                    f"[Dynamic Context]{dynamic_context}\n\n[User Query]\n{query}"
                )
            else:
                query_with_context = query

            # Run the agent
            result = self.agent(query_with_context)

            # 🎬 Record agent response if recording active
            if recorder and recorder.recording:
                recorder.record_agent_message("assistant", str(result)[:2000])
                recorder.snapshot(
                    self.agent,
                    "after_agent_call",
                    last_query=original_query,
                    last_result=str(result)[:5000],
                )

            # 🌙 Record interaction for ambient mode
            if self.ambient:
                self.ambient.record_interaction(original_query, result)

            # 🔗 Push to unified mesh ring (bidirectional sync)
            try:
                from devduck.tools.unified_mesh import add_to_ring

                result_preview = str(result)
                add_to_ring(
                    "local:devduck",
                    "local",
                    f"Q: {original_query} → {result_preview}",
                    {"source": "cli"},
                )
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Ring sync failed: {e}")

            # 💾 Knowledge Base Storage (AFTER agent runs)
            if knowledge_base_id and hasattr(self.agent, "tool"):
                try:
                    if "store_in_kb" in self.agent.tool_names:
                        conversation_content = (
                            f"Input: {original_query}, Result: {result!s}"
                        )
                        conversation_title = f"DevDuck: {datetime.now().strftime('%Y-%m-%d')} | {original_query[:500]}"
                        self.agent.tool.store_in_kb(
                            content=conversation_content,
                            title=conversation_title,
                            knowledge_base_id=knowledge_base_id,
                        )
                        logger.info(f"Stored conversation in KB: {knowledge_base_id}")
                except Exception as e:
                    logger.warning(f"KB storage failed: {e}")

            # Clear executing flag
            self._agent_executing = False

            # Check for pending hot-reload
            if self._reload_pending:
                logger.info("Triggering pending hot-reload after agent completion")
                print("\n🦆 Agent finished - triggering pending hot-reload...")
                self._hot_reload()

            return result
        except Exception as e:
            self._agent_executing = False  # Reset flag on error
            logger.error(f"Agent call failed with error: {e}")

            # 🎬 Record error if recording
            recorder = get_session_recorder()
            if recorder and recorder.recording:
                recorder.record_agent_message("error", str(e))

            self._self_heal(e)
            if self.agent:
                return self.agent(query)
            else:
                return f"🦆 Error: {e}"

    def restart(self):
        """Restart the agent"""
        print("\n🦆 Restarting...")
        logger.debug("\n🦆 Restarting...")
        self.__init__()

    def _start_file_watcher(self):
        """Start background file watcher for auto hot-reload"""

        logger.info("Starting file watcher for hot-reload")
        # Get the path to this file
        self._watch_file = Path(__file__).resolve()
        self._last_modified = (
            self._watch_file.stat().st_mtime if self._watch_file.exists() else None
        )
        self._watcher_running = True
        self._is_reloading = False

        # Start watcher thread
        self._watcher_thread = threading.Thread(
            target=self._file_watcher_thread, daemon=True
        )
        self._watcher_thread.start()
        logger.info(f"File watcher started, monitoring {self._watch_file}")

    def _file_watcher_thread(self):
        """Background thread that watches for file changes"""
        last_reload_time = 0
        debounce_seconds = 3  # 3 second debounce

        while self._watcher_running:
            try:
                # Skip if currently reloading
                if self._is_reloading:
                    time.sleep(1)
                    continue

                if self._watch_file.exists():
                    current_mtime = self._watch_file.stat().st_mtime
                    current_time = time.time()

                    # Check if file was modified AND debounce period has passed
                    if (
                        self._last_modified
                        and current_mtime > self._last_modified
                        and current_time - last_reload_time > debounce_seconds
                    ):
                        # print(f"\n🦆 Detected changes in {self._watch_file.name}!")
                        last_reload_time = current_time

                        # Check if agent is currently executing
                        if self._agent_executing:
                            logger.info(
                                "Code change detected but agent is executing - reload pending"
                            )
                            # print(
                            #     "\n🦆 Agent is currently executing - reload will trigger after completion"
                            # )
                            self._reload_pending = True
                            # Don't update _last_modified yet - keep detecting the change
                        else:
                            # Safe to reload immediately
                            self._last_modified = current_mtime
                            logger.info(
                                f"Code change detected in {self._watch_file.name} - triggering hot-reload"
                            )
                            time.sleep(
                                0.5
                            )  # Small delay to ensure file write is complete
                            self._hot_reload()
                    else:
                        # Update timestamp if no change or still in debounce
                        if not self._reload_pending:
                            self._last_modified = current_mtime

            except Exception as e:
                logger.error(f"File watcher error: {e}")

            # Check every 1 second
            time.sleep(1)

    def _stop_file_watcher(self):
        """Stop the file watcher"""
        self._watcher_running = False
        logger.info("File watcher stopped")

    def _hot_reload(self):
        """Hot-reload by restarting the entire Python process with fresh code"""
        logger.info("Hot-reload initiated")
        print("\n🦆 Hot-reloading via process restart...")

        try:
            # Set reload flag to prevent recursive reloads during shutdown
            self._is_reloading = True

            # Update last_modified before reload to acknowledge the change
            if hasattr(self, "_watch_file") and self._watch_file.exists():
                self._last_modified = self._watch_file.stat().st_mtime

            # Reset pending flag
            self._reload_pending = False

            # Stop the file watcher
            if hasattr(self, "_watcher_running"):
                self._watcher_running = False

            print("\n🦆 Restarting process with fresh code...")
            logger.debug("\n🦆 Restarting process with fresh code...")

            # Restart the entire Python process
            # This ensures all code is freshly loaded
            os.execv(sys.executable, [sys.executable] + sys.argv)

        except Exception as e:
            logger.error(f"Hot-reload failed: {e}")
            print(f"\n🦆 Hot-reload failed: {e}")
            print("\n🦆 Falling back to manual restart")
            self._is_reloading = False

    def status(self):
        """Show current status"""
        status_dict = {
            "model": self.model,
            "env": self.env_info,
            "agent_ready": self.agent is not None,
            "tools": len(self.tools) if hasattr(self, "tools") else 0,
            "file_watcher": {
                "enabled": hasattr(self, "_watcher_running") and self._watcher_running,
                "watching": (
                    str(self._watch_file) if hasattr(self, "_watch_file") else None
                ),
            },
        }

        # 🌙 Ambient mode status
        if self.ambient:
            status_dict["ambient_mode"] = {
                "enabled": self.ambient.running,
                "autonomous": self.ambient.autonomous,
                "idle_threshold": self.ambient.idle_threshold,
                "max_iterations": (
                    self.ambient.autonomous_max_iterations
                    if self.ambient.autonomous
                    else self.ambient.max_iterations
                ),
                "current_iterations": self.ambient.ambient_iterations,
                "has_pending_result": self.ambient.ambient_result is not None
                or len(self.ambient.ambient_results_history) > 0,
                "results_stored": len(self.ambient.ambient_results_history),
                "last_query": (
                    self.ambient.last_query[:50] if self.ambient.last_query else None
                ),
            }
        else:
            status_dict["ambient_mode"] = {"enabled": False}

        # 🎬 Session recording status
        recorder = get_session_recorder()
        if recorder:
            status_dict["session_recording"] = {
                "enabled": recorder.recording,
                "session_id": recorder.session_id,
                "events": recorder.event_buffer.count,
                "snapshots": len(recorder.snapshots),
                "duration": (
                    time.time() - recorder.start_time if recorder.start_time else 0
                ),
                "recording_dir": str(RECORDING_DIR),
            }
        else:
            status_dict["session_recording"] = {"enabled": False}

        return status_dict


# 🦆 Auto-initialize when imported
# Check environment variables to control server configuration
# Also check if --mcp flag is present to skip auto-starting servers
_auto_start = os.getenv("DEVDUCK_AUTO_START_SERVERS", "true").lower() == "true"

# Disable auto-start if --mcp flag is present (stdio mode)
if "--mcp" in sys.argv:
    _auto_start = False

devduck = DevDuck(auto_start_servers=_auto_start)


# 🚀 Convenience functions
def ask(query):
    """Quick query interface"""
    return devduck(query)


def status():
    """Quick status check"""
    return devduck.status()


def restart():
    """Quick restart"""
    devduck.restart()


def hot_reload():
    """Quick hot-reload without restart"""
    devduck._hot_reload()


def extract_commands_from_history():
    """Extract commonly used commands from shell history for auto-completion."""
    commands = set()
    history_files = get_shell_history_files()

    # Limit the number of recent commands to process for performance
    max_recent_commands = 100

    for history_type, history_file in history_files:
        try:
            with open(history_file, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Take recent commands for better relevance
            recent_lines = (
                lines[-max_recent_commands:]
                if len(lines) > max_recent_commands
                else lines
            )

            for line in recent_lines:
                line = line.strip()
                if not line:
                    continue

                if history_type == "devduck":
                    # Extract devduck commands
                    if "# devduck:" in line:
                        try:
                            query = line.split("# devduck:")[-1].strip()
                            # Extract first word as command
                            first_word = query.split()[0] if query.split() else None
                            if (
                                first_word and len(first_word) > 2
                            ):  # Only meaningful commands
                                commands.add(first_word.lower())
                        except (ValueError, IndexError):
                            continue

                elif history_type == "zsh":
                    # Zsh format: ": timestamp:0;command"
                    if line.startswith(": ") and ":0;" in line:
                        try:
                            parts = line.split(":0;", 1)
                            if len(parts) == 2:
                                full_command = parts[1].strip()
                                # Extract first word as command
                                first_word = (
                                    full_command.split()[0]
                                    if full_command.split()
                                    else None
                                )
                                if (
                                    first_word and len(first_word) > 1
                                ):  # Only meaningful commands
                                    commands.add(first_word.lower())
                        except (ValueError, IndexError):
                            continue

                elif history_type == "bash":
                    # Bash format: simple command per line
                    first_word = line.split()[0] if line.split() else None
                    if first_word and len(first_word) > 1:  # Only meaningful commands
                        commands.add(first_word.lower())

        except Exception:
            # Skip files that can't be read
            continue

    return list(commands)


def interactive():
    """Interactive REPL mode for devduck"""
    from prompt_toolkit import prompt
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory

    print("🦆 DevDuck")
    print(f"📝 Logs: {LOG_DIR}")
    if devduck.ambient:
        print(f"🌙 Ambient mode: ON (idle: {devduck.ambient.idle_threshold}s)")
    recorder = get_session_recorder()
    if recorder and recorder.recording:
        print(f"🎬 Recording: ON ({recorder.session_id})")
    print("Type 'exit', 'quit', or 'q' to quit.")
    print("Commands: 'record' (toggle recording), 'ambient' (toggle), '!' (shell)")
    print("\n\n")
    logger.info("Interactive mode started")

    # Set up prompt_toolkit with history
    history_file = get_shell_history_file()
    history = FileHistory(history_file)

    # Create completions from common commands and shell history
    base_commands = [
        "exit",
        "quit",
        "q",
        "help",
        "clear",
        "status",
        "reload",
        "ambient",
        "auto",
        "autonomous",
        "record",  # 🎬 Session recording toggle
    ]
    history_commands = extract_commands_from_history()

    # Combine base commands with commands from history
    all_commands = list(set(base_commands + history_commands))
    completer = WordCompleter(all_commands, ignore_case=True)

    # Track consecutive interrupts for double Ctrl+C to exit
    interrupt_count = 0
    last_interrupt = 0

    while True:
        try:
            # Use prompt_toolkit for enhanced input with arrow key support
            q = prompt(
                "\n🦆 ",
                history=history,
                auto_suggest=AutoSuggestFromHistory(),
                completer=completer,
                complete_while_typing=True,
            )

            # Reset interrupt count on successful prompt
            interrupt_count = 0

            # Check for exit command
            if q.lower() in ["exit", "quit", "q"]:
                if devduck.ambient:
                    devduck.ambient.stop()
                print("\n🦆 Goodbye!")
                break

            # Skip empty inputs
            if q.strip() == "":
                continue

            # Handle ambient mode toggle
            if q.lower() == "ambient":
                if devduck.ambient:
                    if devduck.ambient.running:
                        devduck.ambient.stop()
                        print("🌙 Ambient mode disabled")
                    else:
                        devduck.ambient.start()
                        print("🌙 Ambient mode enabled (standard)")
                else:
                    devduck.ambient = AmbientMode(devduck)
                    devduck.ambient.start()
                    print("🌙 Ambient mode enabled (standard)")
                continue

            # Handle autonomous mode toggle
            if q.lower() in ["auto", "autonomous"]:
                if devduck.ambient:
                    if devduck.ambient.autonomous:
                        devduck.ambient.stop()
                        print("🌙 Autonomous mode disabled")
                    elif devduck.ambient.running:
                        devduck.ambient.start(autonomous=True)
                    else:
                        devduck.ambient.start(autonomous=True)
                else:
                    devduck.ambient = AmbientMode(devduck)
                    devduck.ambient.start(autonomous=True)
                continue

            # 🎬 Handle recording mode toggle
            if q.lower() == "record":
                recorder = get_session_recorder()
                if recorder and recorder.recording:
                    # Stop and export
                    export_path = stop_recording()
                    devduck._recording = False
                    print(f"🎬 Recording stopped and exported: {export_path}")
                else:
                    # Start recording
                    start_recording()
                    devduck._recording = True
                    print(
                        f"🎬 Recording started. Type 'record' again to stop and export."
                    )
                continue

            # Handle shell commands with ! prefix
            if q.startswith("!"):
                shell_command = q[1:].strip()
                try:
                    if devduck.agent:
                        devduck._agent_executing = (
                            True  # Prevent hot-reload during shell execution
                        )
                        result = devduck.agent.tool.shell(
                            command=shell_command, timeout=9000
                        )
                        devduck._agent_executing = False

                        # Reset terminal to fix rendering issues after command output
                        print("\r", end="", flush=True)
                        sys.stdout.flush()

                        # Append shell command to history
                        append_to_shell_history(q, result["content"][0]["text"])

                        # Check if reload was pending
                        if devduck._reload_pending:
                            print(
                                "🦆 Shell command finished - triggering pending hot-reload..."
                            )
                            devduck._hot_reload()
                    else:
                        print("🦆 Agent unavailable")
                except Exception as e:
                    devduck._agent_executing = False  # Reset on error
                    print(f"🦆 Shell command error: {e}")
                    # Reset terminal on error too
                    print("\r", end="", flush=True)
                    sys.stdout.flush()
                continue

            # Execute the agent with user input
            result = ask(q)

            # Append to shell history
            append_to_shell_history(q, str(result))

        except KeyboardInterrupt:
            current_time = time.time()

            # Check if this is a consecutive interrupt within 2 seconds
            if current_time - last_interrupt < 2:
                interrupt_count += 1
                if interrupt_count >= 2:
                    if devduck.ambient:
                        devduck.ambient.stop()
                    print("\n🦆 Exiting...")
                    break
                else:
                    print("\n🦆 Interrupted. Press Ctrl+C again to exit.")
            else:
                interrupt_count = 1
                print("\n🦆 Interrupted. Press Ctrl+C again to exit.")

            last_interrupt = current_time
            continue
        except Exception as e:
            print(f"🦆 Error: {e}")
            continue


def _deploy_to_agentcore(
    name: str = "devduck",
    tools: str = None,
    model: str = None,
    region: str = "us-west-2",
    auto_launch: bool = False,
    system_prompt: str = None,
    idle_timeout: int = 900,
    max_lifetime: int = 28800,
    disable_memory: bool = False,
    disable_otel: bool = False,
    env_vars: list = None,
    force_rebuild: bool = False,
):
    """
    Deploy DevDuck to Amazon Bedrock AgentCore.

    This function:
    1. Creates a deployment directory with handler + requirements
    2. Configures the agent with agentcore CLI
    3. Optionally launches the agent
    4. Returns agent info for proxy integration

    Args:
        name: Agent name (hyphens converted to underscores)
        tools: Tool configuration (e.g., 'strands_tools:shell,editor')
        model: Model ID override
        region: AWS region
        auto_launch: Auto-launch after configure
        system_prompt: Custom system prompt
        idle_timeout: Idle timeout in seconds (60-28800)
        max_lifetime: Max lifetime in seconds (60-28800)
        disable_memory: Disable AgentCore memory (STM)
        disable_otel: Disable OpenTelemetry observability
        env_vars: Additional environment variables (list of KEY=VALUE strings)
        force_rebuild: Force rebuild dependencies
    """
    import shutil
    import re

    print("🦆 Deploying DevDuck to AgentCore...")
    print("=" * 50)

    # Check for agentcore CLI
    if not shutil.which("agentcore"):
        print("❌ agentcore CLI not found.")
        print("   Install with: pip install bedrock-agentcore")
        sys.exit(1)

    # Convert hyphens to underscores in name (AgentCore requirement)
    safe_name = name.replace("-", "_")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]{0,47}$", safe_name):
        print(f"❌ Invalid agent name: {name}")
        print(
            "   Must start with letter, contain only letters/numbers/underscores, 1-48 chars"
        )
        sys.exit(1)

    # Create deployment directory
    deploy_dir = Path(tempfile.gettempdir()) / "devduck" / "deploy" / safe_name
    deploy_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Deployment directory: {deploy_dir}")

    # Copy handler
    handler_src = Path(__file__).parent / "agentcore_handler.py"
    if not handler_src.exists():
        print(f"❌ Handler not found: {handler_src}")
        sys.exit(1)

    handler_dest = deploy_dir / "agentcore_handler.py"
    shutil.copy(str(handler_src), str(handler_dest))
    print(f"📦 Handler: {handler_dest}")

    # Create requirements.txt
    requirements_path = deploy_dir / "requirements.txt"
    requirements_content = "devduck\n"
    with open(requirements_path, "w") as f:
        f.write(requirements_content)
    print(f"📋 Requirements: {requirements_path}")

    # Build configure command
    configure_cmd = [
        "agentcore",
        "configure",
        "-e",
        str(handler_dest),
        "-n",
        safe_name,
        "-r",
        region,
        "-p",
        "HTTP",
        "-dt",
        "direct_code_deploy",
        "-rt",
        "PYTHON_3_13",
        "-rf",
        str(requirements_path),
        "--idle-timeout",
        str(idle_timeout),
        "--max-lifetime",
        str(max_lifetime),
        "-ni",  # Non-interactive mode
    ]

    # Add optional flags
    if disable_memory:
        configure_cmd.append("-dm")
        print("   Memory: DISABLED")

    if disable_otel:
        configure_cmd.append("-do")
        print("   OpenTelemetry: DISABLED")

    print(f"\n🔧 Configuring agent '{safe_name}'...")
    print(f"   Region: {region}")
    print(f"   Model: {model or 'global.anthropic.claude-opus-4-6-v1 (default)'}")
    print(f"   Tools: {tools or 'default'}")
    print(f"   Memory: {'DISABLED' if disable_memory else 'enabled'}")

    try:
        process = subprocess.Popen(
            configure_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(deploy_dir),
        )

        stdout, _ = process.communicate(timeout=300)

        if process.returncode != 0:
            print(f"❌ Configure failed:\n{stdout}")
            sys.exit(1)

        print(stdout)
        print(f"✅ Agent '{safe_name}' configured!")

    except subprocess.TimeoutExpired:
        print("❌ Configure timed out")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Configure error: {e}")
        sys.exit(1)

    # Launch if requested
    agent_arn = None
    agent_id = None

    if auto_launch:
        print(f"\n🚀 Deploying agent '{safe_name}'...")
        print("=" * 50)

        # Build deploy command with environment variables
        deploy_cmd = ["agentcore", "deploy", "-a", safe_name, "-auc"]

        if force_rebuild:
            deploy_cmd.append("-frd")

        # Build environment variables list
        env_vars_list = [
            "DEVDUCK_AUTO_START_SERVERS=false",
            "MODEL_PROVIDER=bedrock",
            "BYPASS_TOOL_CONSENT=true",
        ]

        if tools:
            env_vars_list.append(f"DEVDUCK_TOOLS={tools}")
        if model:
            env_vars_list.append(f"STRANDS_MODEL_ID={model}")
        if system_prompt:
            # Escape quotes in system prompt
            escaped_prompt = system_prompt.replace('"', '\\"')
            env_vars_list.append(f"SYSTEM_PROMPT={escaped_prompt}")

        # Add custom env vars
        if env_vars:
            env_vars_list.extend(env_vars)

        # Add -env flags to deploy command
        for env_var in env_vars_list:
            deploy_cmd.extend(["-env", env_var])

        print(f"   Environment variables: {len(env_vars_list)}")

        try:
            deploy_process = subprocess.Popen(
                deploy_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(deploy_dir),
            )

            # Stream output
            output_lines = []
            for line in deploy_process.stdout:
                print(line, end="", flush=True)
                output_lines.append(line)

            deploy_process.wait(timeout=600)
            full_output = "".join(output_lines)

            if deploy_process.returncode == 0:
                # Extract ARN and ID
                arn_match = re.search(
                    r"arn:aws:bedrock-agentcore:[^:]+:[^:]+:runtime/([^\s\n]+)",
                    full_output,
                )
                if arn_match:
                    agent_arn = arn_match.group(0)
                    agent_id = arn_match.group(1)

                print("\n" + "=" * 50)
                print(f"✅ Agent '{safe_name}' deployed!")

                if agent_id:
                    print(f"\n📋 Agent ARN: {agent_arn}")
                    print(f"🆔 Agent ID: {agent_id}")
                    print(f"\n💡 Invoke with:")
                    print(f"   agentcore invoke {safe_name} 'your query'")
                    print(f"\n💡 Or via DevDuck:")
                    print(
                        f"   devduck.agent.tool.agentcore_invoke(agent_id='{agent_id}', prompt='...')"
                    )
                    print(f"\n💡 View in mesh.html via proxy on ws://localhost:10000")
            else:
                print(f"\n❌ Deploy failed with code {deploy_process.returncode}")

        except subprocess.TimeoutExpired:
            print("❌ Deploy timed out")
        except Exception as e:
            print(f"❌ Deploy error: {e}")
    else:
        print(f"\n💡 To launch the agent, run:")
        print(f"   agentcore launch -a {safe_name} --auto-update-on-conflict")
        print(f"\n💡 Or use: devduck deploy --name {name} --launch")

    # Save deployment info
    info_path = deploy_dir / "deployment_info.json"
    deployment_info = {
        "name": safe_name,
        "region": region,
        "model": (
            model or DEFAULT_MODEL if "DEFAULT_MODEL" in dir() else "claude-sonnet-4"
        ),
        "tools": tools,
        "agent_arn": agent_arn,
        "agent_id": agent_id,
        "deployed_at": datetime.now().isoformat(),
        "handler_path": str(handler_dest),
    }

    with open(info_path, "w") as f:
        json.dump(deployment_info, f, indent=2)

    print(f"\n📄 Deployment info saved: {info_path}")

    return deployment_info


def _list_agentcore_agents(region: str = "us-west-2"):
    """List all deployed AgentCore agents."""
    print("🦆 Listing AgentCore agents...")
    print("=" * 60)

    try:
        import boto3

        client = boto3.client("bedrock-agentcore-control", region_name=region)
        response = client.list_agent_runtimes(maxResults=100)

        agents = response.get("agentRuntimes", [])

        if not agents:
            print("No agents found.")
            return

        print(f"Found {len(agents)} agent(s):\n")

        for agent in agents:
            name = agent.get("agentRuntimeName", "unknown")
            agent_id = agent.get("agentRuntimeId", "unknown")
            status = agent.get("status", "unknown")

            status_emoji = (
                "✅"
                if status == "ACTIVE"
                else "⏳" if status in ("CREATING", "UPDATING") else "❌"
            )

            print(f"  {status_emoji} {name}")
            print(f"     ID: {agent_id}")
            print(f"     Status: {status}")
            print()

        print("=" * 60)
        print("💡 Invoke with: devduck invoke 'your query' --name <agent_name>")
        print(
            '💡 Or via proxy: ws://localhost:10000 → {"type": "invoke", "agent_id": "...", "prompt": "..."}'
        )

    except Exception as e:
        print(f"❌ Error listing agents: {e}")
        sys.exit(1)


def _check_agent_status(name: str = "devduck", region: str = "us-west-2"):
    """Check status of a specific agent."""
    safe_name = name.replace("-", "_")

    print(f"🦆 Checking status of '{safe_name}'...")
    print("=" * 50)

    try:
        result = subprocess.run(
            ["agentcore", "status", "-a", safe_name],
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

    except FileNotFoundError:
        print("❌ agentcore CLI not found.")
        print("   Install with: pip install bedrock-agentcore")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def _invoke_agentcore_agent(
    prompt: str,
    name: str = "devduck",
    agent_id: str = None,
    region: str = "us-west-2",
):
    """Invoke a deployed AgentCore agent."""
    safe_name = name.replace("-", "_")

    print(f"🦆 Invoking agent...")
    print("=" * 50)

    try:
        if agent_id:
            # Direct invocation via agent_id
            import boto3
            from botocore.config import Config

            sts = boto3.client("sts", region_name=region)
            account_id = sts.get_caller_identity()["Account"]
            agent_arn = (
                f"arn:aws:bedrock-agentcore:{region}:{account_id}:runtime/{agent_id}"
            )

            print(f"Agent: {agent_id}")
            print(f"Prompt: {prompt[:50]}...")
            print()

            boto_config = Config(read_timeout=900, connect_timeout=60)
            client = boto3.client(
                "bedrock-agentcore", region_name=region, config=boto_config
            )

            response = client.invoke_agent_runtime(
                agentRuntimeArn=agent_arn,
                qualifier="DEFAULT",
                runtimeSessionId=f"cli-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                payload=json.dumps({"prompt": prompt, "mode": "sync"}),
            )

            # Process response
            full_response = ""
            for chunk in response.get("response", []):
                if isinstance(chunk, bytes):
                    full_response += chunk.decode("utf-8", errors="ignore")
                elif isinstance(chunk, str):
                    full_response += chunk

            print("Response:")
            print("-" * 50)
            print(full_response)

        else:
            # Use agentcore CLI
            result = subprocess.run(
                ["agentcore", "invoke", safe_name, prompt],
                capture_output=True,
                text=True,
            )

            print(result.stdout)
            if result.stderr:
                print(result.stderr)

    except FileNotFoundError:
        print("❌ agentcore CLI not found.")
        print("   Install with: pip install bedrock-agentcore")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def cli():
    """CLI entry point for pip-installed devduck command"""
    import argparse

    parser = argparse.ArgumentParser(
        description="🦆 DevDuck - Extreme minimalist self-adapting agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  devduck                          # Start interactive mode
  devduck "your query here"        # One-shot query
  devduck --mcp                    # MCP stdio mode (for Claude Desktop)
  devduck --record                 # Start with session recording enabled
  devduck --record "do something"  # Record a one-shot query
  devduck --resume session.zip     # Resume from recorded session
  devduck --resume session.zip "continue"  # Resume and run query

AgentCore Deployment:
  devduck deploy                   # Configure agent (no launch)
  devduck deploy --launch          # Configure AND launch
  devduck deploy --name my-agent   # Custom agent name
  devduck deploy --tools "strands_tools:shell,editor"
  devduck deploy --model "us.anthropic.claude-opus-4-20250514-v1:0"
  devduck deploy --system-prompt "You are a code reviewer"
  devduck deploy --idle-timeout 1800 --max-lifetime 43200
  devduck deploy --no-memory       # Disable AgentCore memory
  devduck deploy --no-otel         # Disable OpenTelemetry
  devduck deploy --env "MY_VAR=value" --env "OTHER=123"  # Custom env vars
  devduck deploy --force-rebuild   # Force rebuild dependencies

AgentCore Full Example:
  devduck deploy --name code-review-agent \\
    --tools "strands_tools:shell,editor,file_read" \\
    --model "us.anthropic.claude-sonnet-4-20250514-v1:0" \\
    --system-prompt "You are a senior code reviewer" \\
    --no-memory --launch

Proxy for mesh.html:
  # DevDuck auto-starts agentcore_proxy on ws://localhost:10000
  # Open mesh.html to see all deployed AgentCore agents
  # Send: {"type": "list_agents"} to see available agents
  # Send: {"type": "invoke", "agent_id": "...", "prompt": "..."} to invoke

Session Recording & Resume:
  devduck --record                 # Auto-records and exports on exit
  devduck --resume ~/Desktop/session-*.zip  # Resume from session
  devduck --resume session.zip --snapshot 2 "continue"  # Resume specific snapshot
  Recordings saved to: /tmp/devduck/recordings/

Tool Configuration:
  export DEVDUCK_TOOLS="strands_tools:shell,editor:strands_fun_tools:clipboard"

Claude Desktop Config:
  {
    "mcpServers": {
      "devduck": {
        "command": "uvx",
        "args": ["devduck", "--mcp"]
      }
    }
  }
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Deploy subcommand
    deploy_parser = subparsers.add_parser("deploy", help="Deploy DevDuck to AgentCore")
    deploy_parser.add_argument(
        "--name", "-n", default="devduck", help="Agent name (default: devduck)"
    )
    deploy_parser.add_argument(
        "--tools",
        "-t",
        default=None,
        help="Tool configuration (e.g., 'strands_tools:shell,editor')",
    )
    deploy_parser.add_argument(
        "--model", "-m", default=None, help="Model ID (default: claude-sonnet-4)"
    )
    deploy_parser.add_argument(
        "--region", "-r", default="us-west-2", help="AWS region (default: us-west-2)"
    )
    deploy_parser.add_argument(
        "--launch", action="store_true", help="Auto-launch after configure"
    )
    deploy_parser.add_argument(
        "--system-prompt", "-s", default=None, help="Custom system prompt for the agent"
    )
    deploy_parser.add_argument(
        "--idle-timeout",
        type=int,
        default=900,
        help="Idle timeout in seconds (default: 900)",
    )
    deploy_parser.add_argument(
        "--max-lifetime",
        type=int,
        default=28800,
        help="Max lifetime in seconds (default: 28800)",
    )
    deploy_parser.add_argument(
        "--no-memory",
        "--disable-memory",
        action="store_true",
        dest="disable_memory",
        help="Disable AgentCore memory (STM)",
    )
    deploy_parser.add_argument(
        "--no-otel",
        "--disable-otel",
        action="store_true",
        dest="disable_otel",
        help="Disable OpenTelemetry observability",
    )
    deploy_parser.add_argument(
        "--env",
        "-e",
        action="append",
        dest="env_vars",
        metavar="KEY=VALUE",
        help="Additional environment variable (can be repeated)",
    )
    deploy_parser.add_argument(
        "--force-rebuild", "-f", action="store_true", help="Force rebuild dependencies"
    )

    # List agents subcommand
    list_parser = subparsers.add_parser("list", help="List deployed AgentCore agents")
    list_parser.add_argument(
        "--region", "-r", default="us-west-2", help="AWS region (default: us-west-2)"
    )

    # Status subcommand
    status_parser = subparsers.add_parser("status", help="Check agent status")
    status_parser.add_argument(
        "--name", "-n", default="devduck", help="Agent name to check"
    )
    status_parser.add_argument(
        "--region", "-r", default="us-west-2", help="AWS region (default: us-west-2)"
    )

    # Invoke subcommand
    invoke_parser = subparsers.add_parser("invoke", help="Invoke a deployed agent")
    invoke_parser.add_argument("prompt", help="Query to send to the agent")
    invoke_parser.add_argument(
        "--name", "-n", default="devduck", help="Agent name (default: devduck)"
    )
    invoke_parser.add_argument(
        "--agent-id", "-i", default=None, help="Direct agent ID (bypasses name lookup)"
    )
    invoke_parser.add_argument(
        "--region", "-r", default="us-west-2", help="AWS region (default: us-west-2)"
    )

    # Query argument (for default mode)
    parser.add_argument("query", nargs="*", help="Query to send to the agent")

    # MCP stdio mode flag
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Start MCP server in stdio mode (for Claude Desktop integration)",
    )

    # Session recording flag
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable session recording (exports to /tmp/devduck/recordings/)",
    )

    # Session resume flags
    parser.add_argument(
        "--resume",
        type=str,
        metavar="SESSION_FILE",
        help="Resume from a recorded session file (ZIP)",
    )

    parser.add_argument(
        "--snapshot",
        type=int,
        metavar="ID",
        help="Specific snapshot ID to resume from (default: latest)",
    )

    args = parser.parse_args()

    logger.info("CLI mode started")

    # Handle deploy command
    if args.command == "deploy":
        _deploy_to_agentcore(
            name=args.name,
            tools=args.tools,
            model=args.model,
            region=args.region,
            auto_launch=args.launch,
            system_prompt=getattr(args, "system_prompt", None),
            idle_timeout=getattr(args, "idle_timeout", 900),
            max_lifetime=getattr(args, "max_lifetime", 28800),
            disable_memory=getattr(args, "disable_memory", False),
            disable_otel=getattr(args, "disable_otel", False),
            env_vars=getattr(args, "env_vars", None),
            force_rebuild=getattr(args, "force_rebuild", False),
        )
        return

    # Handle list command
    if args.command == "list":
        _list_agentcore_agents(region=args.region)
        return

    # Handle status command
    if args.command == "status":
        _check_agent_status(name=args.name, region=args.region)
        return

    # Handle invoke command
    if args.command == "invoke":
        _invoke_agentcore_agent(
            prompt=args.prompt,
            name=args.name,
            agent_id=getattr(args, "agent_id", None),
            region=args.region,
        )
        return

    # Handle --mcp flag for stdio mode
    if args.mcp:
        logger.info("Starting MCP server in stdio mode (blocking, foreground)")
        print("🦆 Starting MCP stdio server...", file=sys.stderr)

        # Don't auto-start HTTP/TCP/WS servers for stdio mode
        if devduck.agent:
            try:
                # Start MCP server in stdio mode - this BLOCKS until terminated
                devduck.agent.tool.mcp_server(
                    action="start",
                    transport="stdio",
                    expose_agent=True,
                    agent=devduck.agent,
                    record_direct_tool_call=False,
                )
            except Exception as e:
                logger.error(f"Failed to start MCP stdio server: {e}")
                print(f"🦆 Error: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print("🦆 Agent not available", file=sys.stderr)
            sys.exit(1)
        return

    # Handle --resume flag
    if args.resume:
        logger.info(f"Resuming from session: {args.resume}")
        print(f"🎬 Resuming from session: {args.resume}")

        query = " ".join(args.query) if args.query else None

        try:
            result = resume_session(
                session_path=args.resume, snapshot_id=args.snapshot, new_query=query
            )

            if result["status"] == "success":
                print(f"✅ Resumed from snapshot #{result.get('snapshot_id')}")
                print(f"   Messages restored: {result.get('messages_restored', 0)}")
                print(f"   Working directory: {result.get('cwd')}")

                if result.get("continuation_successful"):
                    print(f"\n📝 Result:\n{result.get('agent_result')}")
                elif query:
                    print(f"❌ Continuation failed: {result.get('continuation_error')}")
                else:
                    print("\n💡 Session state restored. Run with a query to continue:")
                    print(f'   devduck --resume {args.resume} "your query here"')
            else:
                print(f"❌ Resume failed: {result.get('message')}")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Resume failed: {e}")
            print(f"❌ Error: {e}")
            sys.exit(1)
        return

    # Handle --record flag
    recording_session = None
    if args.record:
        recording_session = start_recording()
        # Also set global state for DevDuck instance
        devduck._recording = True

    try:
        if args.query:
            query = " ".join(args.query)
            logger.info(f"CLI query: {query}")
            result = ask(query)
            print(result)
        else:
            # No arguments - start interactive mode
            interactive()
    finally:
        # Stop recording on exit if active
        if recording_session and recording_session.recording:
            export_path = stop_recording()
            if export_path:
                print(f"\n🎬 Session recording saved: {export_path}")


# 🦆 Make module directly callable: import devduck; devduck("query")
class CallableModule(sys.modules[__name__].__class__):
    """Make the module itself callable"""

    def __call__(self, query):
        """Allow direct module call: import devduck; devduck("query")"""
        return ask(query)


# Replace module in sys.modules with callable version
sys.modules[__name__].__class__ = CallableModule


if __name__ == "__main__":
    cli()
