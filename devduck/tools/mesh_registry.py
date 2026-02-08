"""
🦆 Mesh Registry — File-based agent discovery with zero race conditions.

Every running DevDuck (CLI, browser proxy, AgentCore) writes itself into
a single JSON file. Stale entries auto-expire via TTL. No central server needed.

Race condition prevention:
1. fcntl.flock() for exclusive write access
2. Atomic write: write to .tmp, then os.rename() (atomic on POSIX)
3. Heartbeat TTL: entries expire after STALE_SECONDS — no cleanup needed on crash

Usage:
    from devduck.tools.mesh_registry import registry

    # Register (call once on startup)
    registry.register(agent_id="80a99724-abc123", agent_type="zenoh", metadata={...})

    # Heartbeat (call every 5s from existing heartbeat thread)
    registry.heartbeat("80a99724-abc123")

    # Read all live agents (from ANY process — file is the source of truth)
    agents = registry.get_all()

    # Unregister (call on shutdown — but crash-safe via TTL)
    registry.unregister("80a99724-abc123")
"""

import fcntl
import json
import os
import tempfile
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("devduck.tools.mesh_registry")

# Registry lives in /tmp so all processes on the machine can see it
REGISTRY_PATH = Path(tempfile.gettempdir()) / "devduck" / "mesh_registry.json"
REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

# Entries older than this are considered dead (crash-safe cleanup)
# 30s gives enough headroom for process restarts (zenoh heartbeat is 5s)
STALE_SECONDS = 30.0


class MeshRegistry:
    """File-based agent registry. Lock-free reads, locked writes, crash-safe."""

    def __init__(
        self, path: Path = REGISTRY_PATH, stale_seconds: float = STALE_SECONDS
    ):
        self.path = path
        self.stale_seconds = stale_seconds

    # ── READ (no lock needed — we read whatever's on disk) ──

    def _read_raw(self) -> Dict[str, Any]:
        """Read registry file. Returns empty dict if missing/corrupt."""
        try:
            if not self.path.exists():
                return {}
            with open(self.path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, OSError):
            return {}

    def get_all(self, include_stale: bool = False) -> Dict[str, Dict[str, Any]]:
        """Get all live agents. Stale entries filtered by default.

        Returns: {agent_id: {type, metadata, last_seen, registered_at, ...}}
        """
        data = self._read_raw()
        agents = data.get("agents", {})
        if include_stale:
            return agents
        now = time.time()
        return {
            aid: info
            for aid, info in agents.items()
            if now - info.get("last_seen", 0) < self.stale_seconds
        }

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a single agent by ID. Returns None if not found or stale."""
        agents = self.get_all()
        return agents.get(agent_id)

    def get_by_type(self, agent_type: str) -> Dict[str, Dict[str, Any]]:
        """Get all live agents of a specific type."""
        return {
            aid: info
            for aid, info in self.get_all().items()
            if info.get("type") == agent_type
        }

    # ── WRITE (locked + atomic) ──

    def _write_locked(self, mutator):
        """Execute a mutator function while holding an exclusive file lock.

        Pattern:
            1. Open lock file (separate from data file)
            2. fcntl.flock(LOCK_EX) — blocks until lock acquired
            3. Read current data
            4. mutator(data) — modify in place
            5. Write to .tmp file
            6. os.rename(.tmp → real) — atomic on POSIX
            7. Release lock

        This guarantees no two processes can corrupt the file.
        """
        lock_path = self.path.with_suffix(".lock")
        lock_path.touch(exist_ok=True)

        with open(lock_path, "w") as lock_fd:
            # Exclusive lock — blocks until acquired
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                # Read current state
                data = self._read_raw()
                if "agents" not in data:
                    data["agents"] = {}

                # Apply mutation
                mutator(data)

                # Atomic write: tmp → rename
                tmp_path = self.path.with_suffix(".tmp")
                with open(tmp_path, "w") as f:
                    json.dump(data, f, indent=2)
                os.rename(str(tmp_path), str(self.path))

            finally:
                # Release lock
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

    def register(self, agent_id: str, agent_type: str, metadata: Dict[str, Any] = None):
        """Register an agent. Idempotent — safe to call multiple times.

        Args:
            agent_id: Unique ID (e.g., "80a99724-abc123", "browser:xyz")
            agent_type: One of "zenoh", "browser", "agentcore", "local", "github"
            metadata: Arbitrary metadata (hostname, model, tools, cwd, etc.)
        """
        now = time.time()

        def mutate(data):
            data["agents"][agent_id] = {
                "type": agent_type,
                "registered_at": data["agents"]
                .get(agent_id, {})
                .get("registered_at", now),
                "last_seen": now,
                "metadata": metadata or {},
            }
            # While we have the lock, prune stale entries
            self._prune(data)

        self._write_locked(mutate)
        logger.debug(f"Registry: registered {agent_id} ({agent_type})")

    def heartbeat(self, agent_id: str, metadata_update: Dict[str, Any] = None):
        """Update last_seen timestamp. Call from heartbeat thread.

        Optionally update metadata (e.g., new peer count, cwd change).
        """
        now = time.time()

        def mutate(data):
            if agent_id in data["agents"]:
                data["agents"][agent_id]["last_seen"] = now
                if metadata_update:
                    data["agents"][agent_id]["metadata"].update(metadata_update)
            # Prune while we're here
            self._prune(data)

        self._write_locked(mutate)

    def unregister(self, agent_id: str):
        """Remove an agent. Called on graceful shutdown.

        If the process crashes, the TTL cleanup handles it automatically.
        """

        def mutate(data):
            data["agents"].pop(agent_id, None)

        self._write_locked(mutate)
        logger.debug(f"Registry: unregistered {agent_id}")

    def _prune(self, data: Dict):
        """Remove stale entries. Called inside locked context."""
        now = time.time()
        stale = [
            aid
            for aid, info in data.get("agents", {}).items()
            if now - info.get("last_seen", 0) > self.stale_seconds
        ]
        for aid in stale:
            del data["agents"][aid]
            logger.debug(f"Registry: pruned stale {aid}")

    # ── CONVENIENCE ──

    def summary(self) -> str:
        """Human-readable summary of live agents."""
        agents = self.get_all()
        if not agents:
            return "No agents registered"
        by_type = {}
        for aid, info in agents.items():
            t = info.get("type", "unknown")
            by_type.setdefault(t, []).append(aid)
        lines = [f"Mesh Registry: {len(agents)} live agents"]
        for t, ids in sorted(by_type.items()):
            lines.append(f"  {t}: {len(ids)}")
            for aid in ids:
                meta = agents[aid].get("metadata", {})
                age = time.time() - agents[aid].get("last_seen", 0)
                name = meta.get("hostname") or meta.get("name") or aid
                lines.append(f"    • {aid} ({name}) — {age:.0f}s ago")
        return "\n".join(lines)

    def clear(self):
        """Clear all entries. For testing."""

        def mutate(data):
            data["agents"] = {}

        self._write_locked(mutate)


# Global singleton
registry = MeshRegistry()


# ── @tool interface for agent access ──

from strands import tool


@tool
def mesh_registry(
    action: str,
    agent_id: str = None,
    agent_type: str = None,
    metadata: dict = None,
    include_stale: bool = False,
) -> Dict[str, Any]:
    """Manage the mesh agent registry — discover, register, and monitor agents.

    The mesh registry is a file-based agent discovery system. Every running DevDuck
    (CLI, browser, AgentCore, Zenoh peer) registers itself here. Stale entries
    auto-expire via TTL. No central server needed.

    Args:
        action: Action to perform:
            - "list": List all live agents (default view)
            - "get": Get a single agent by ID (requires agent_id)
            - "list_by_type": List agents of a specific type (requires agent_type)
            - "register": Register an agent (requires agent_id, agent_type)
            - "unregister": Remove an agent (requires agent_id)
            - "heartbeat": Update last_seen for an agent (requires agent_id)
            - "summary": Human-readable summary of the mesh
            - "clear": Clear all entries (for testing)
        agent_id: Agent ID for get/register/unregister/heartbeat
        agent_type: Agent type for register/list_by_type (zenoh, browser, agentcore, local, github)
        metadata: Optional metadata dict for register/heartbeat
        include_stale: If True, include expired entries in list

    Returns:
        Dict with status and content

    Examples:
        mesh_registry(action="list")
        mesh_registry(action="summary")
        mesh_registry(action="get", agent_id="80a99724-abc123")
        mesh_registry(action="list_by_type", agent_type="browser")
        mesh_registry(action="register", agent_id="my-agent", agent_type="local", metadata={"name": "test"})
        mesh_registry(action="unregister", agent_id="my-agent")
    """
    try:
        if action == "list":
            agents = registry.get_all(include_stale=include_stale)
            if not agents:
                return {
                    "status": "success",
                    "content": [{"text": "No live agents in mesh registry."}],
                }
            lines = [f"Mesh Registry: {len(agents)} live agent(s)\n"]
            for aid, info in agents.items():
                age = time.time() - info.get("last_seen", 0)
                meta = info.get("metadata", {})
                name = meta.get("hostname") or meta.get("name") or ""
                label = f" ({name})" if name else ""
                lines.append(
                    f"• **{aid}**{label} — type: {info.get('type')}, seen {age:.0f}s ago"
                )
                if meta:
                    for k, v in meta.items():
                        if k not in ("hostname", "name"):
                            lines.append(f"    {k}: {str(v)[:120]}")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "get":
            if not agent_id:
                return {
                    "status": "error",
                    "content": [{"text": "agent_id required for 'get'"}],
                }
            agent = registry.get_agent(agent_id)
            if not agent:
                return {
                    "status": "success",
                    "content": [{"text": f"Agent '{agent_id}' not found or stale."}],
                }
            return {
                "status": "success",
                "content": [
                    {"text": f"Agent {agent_id}:\n{json.dumps(agent, indent=2)}"}
                ],
            }

        elif action == "list_by_type":
            if not agent_type:
                return {
                    "status": "error",
                    "content": [{"text": "agent_type required for 'list_by_type'"}],
                }
            agents = registry.get_by_type(agent_type)
            if not agents:
                return {
                    "status": "success",
                    "content": [{"text": f"No live '{agent_type}' agents."}],
                }
            lines = [f"{len(agents)} '{agent_type}' agent(s):"]
            for aid, info in agents.items():
                age = time.time() - info.get("last_seen", 0)
                lines.append(f"• {aid} — {age:.0f}s ago")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "register":
            if not agent_id or not agent_type:
                return {
                    "status": "error",
                    "content": [
                        {"text": "agent_id and agent_type required for 'register'"}
                    ],
                }
            registry.register(agent_id, agent_type, metadata)
            return {
                "status": "success",
                "content": [{"text": f"✅ Registered {agent_id} as {agent_type}"}],
            }

        elif action == "unregister":
            if not agent_id:
                return {
                    "status": "error",
                    "content": [{"text": "agent_id required for 'unregister'"}],
                }
            registry.unregister(agent_id)
            return {
                "status": "success",
                "content": [{"text": f"✅ Unregistered {agent_id}"}],
            }

        elif action == "heartbeat":
            if not agent_id:
                return {
                    "status": "error",
                    "content": [{"text": "agent_id required for 'heartbeat'"}],
                }
            registry.heartbeat(agent_id, metadata)
            return {
                "status": "success",
                "content": [{"text": f"✅ Heartbeat sent for {agent_id}"}],
            }

        elif action == "summary":
            return {"status": "success", "content": [{"text": registry.summary()}]}

        elif action == "clear":
            registry.clear()
            return {"status": "success", "content": [{"text": "✅ Registry cleared."}]}

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Valid: list, get, list_by_type, register, unregister, heartbeat, summary, clear"
                    }
                ],
            }

    except Exception as e:
        logger.error(f"mesh_registry tool error: {e}")
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}
