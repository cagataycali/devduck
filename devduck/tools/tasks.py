"""📋 Background task management for DevDuck.

Create, manage, and monitor parallel background tasks.
Each task runs its own agent instance in a separate thread.
State persists to disk for recovery.

Examples:
    tasks(action="create", task_id="research", prompt="Research quantum computing", system_prompt="You are a researcher")
    tasks(action="status", task_id="research")
    tasks(action="add_message", task_id="research", message="Focus on healthcare applications")
    tasks(action="list")
    tasks(action="get_result", task_id="research")
    tasks(action="stop", task_id="research")
"""

import json
import logging
import queue
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from strands import Agent, tool

logger = logging.getLogger(__name__)

# Task storage
TASKS_DIR = Path.home() / ".devduck" / "tasks"
TASKS_DIR.mkdir(parents=True, exist_ok=True)

# Global state
_task_threads: Dict[str, threading.Thread] = {}
_task_agents: Dict[str, Any] = {}
_task_states: Dict[str, "TaskState"] = {}
_task_queues: Dict[str, queue.Queue] = {}


class TaskState:
    """Tracks task lifecycle and persists to disk."""

    @staticmethod
    def _emit_event(event_type: str, summary: str, detail: str = "", metadata: dict = None):
        """Push event to the unified event bus (if available)."""
        try:
            from devduck.tools.event_bus import emit
            emit(event_type, "tasks", summary, detail, metadata)
        except ImportError:
            pass

    def __init__(
        self,
        task_id: str,
        prompt: str,
        system_prompt: str,
        tools: List[str] = None,
        timeout: int = 900,
    ):
        self.task_id = task_id
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.timeout = timeout
        self.status = "initializing"
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.paused = False
        self.messages = [{"role": "user", "content": [{"text": prompt}]}]
        self.queue = queue.Queue()
        _task_queues[task_id] = self.queue
        self._save()

    @property
    def state_path(self) -> Path:
        return TASKS_DIR / f"{self.task_id}_state.json"

    @property
    def result_path(self) -> Path:
        return TASKS_DIR / f"{self.task_id}_result.txt"

    @property
    def messages_path(self) -> Path:
        return TASKS_DIR / f"{self.task_id}_messages.json"

    def _save(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "task_id": self.task_id,
                "status": self.status,
                "created_at": self.created_at,
                "updated_at": datetime.now().isoformat(),
                "prompt": self.prompt,
                "system_prompt": self.system_prompt,
                "tools": self.tools,
                "timeout": self.timeout,
                "paused": self.paused,
            }
            self.state_path.write_text(json.dumps(state, indent=2))
            self.messages_path.write_text(json.dumps(self.messages, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save task state {self.task_id}: {e}")

    def set_status(self, status: str):
        self.status = status
        self.updated_at = datetime.now().isoformat()
        self._save()
        # 🔔 Emit task state change event
        if status in ("completed", "error", "stopped"):
            evt = "task.done" if status == "completed" else "task.error" if status == "error" else "task.done"
            self._emit_event(evt, f"'{self.task_id}' → {status}", self.prompt[:150],
                             {"task_id": self.task_id, "status": status})

    def append_result(self, content: str):
        try:
            with open(self.result_path, "a") as f:
                f.write(f"--- {datetime.now().isoformat()} ---\n{content}\n\n")
        except Exception:
            pass

    @classmethod
    def load(cls, task_id: str) -> Optional["TaskState"]:
        state_path = TASKS_DIR / f"{task_id}_state.json"
        if not state_path.exists():
            return None
        try:
            data = json.loads(state_path.read_text())
            ts = cls(
                data["task_id"],
                data["prompt"],
                data["system_prompt"],
                data.get("tools", []),
                data.get("timeout", 900),
            )
            ts.status = data["status"]
            ts.created_at = data["created_at"]
            ts.paused = data.get("paused", False)
            msgs_path = TASKS_DIR / f"{task_id}_messages.json"
            if msgs_path.exists():
                ts.messages = json.loads(msgs_path.read_text())
            return ts
        except Exception:
            return None


def _run_task(ts: TaskState, parent_agent: Optional[Any] = None):
    """Execute task in background thread."""
    start = time.time()
    try:
        ts.set_status("running")

        # Build tools from parent agent
        tools = []
        if parent_agent:
            if ts.tools:
                for name in ts.tools:
                    if name in parent_agent.tool_registry.registry:
                        tools.append(parent_agent.tool_registry.registry[name])
            else:
                tools = list(parent_agent.tool_registry.registry.values())

        agent = Agent(
            model=parent_agent.model if parent_agent else None,
            messages=ts.messages.copy(),
            tools=tools,
            system_prompt=ts.system_prompt,
            callback_handler=None,
        )
        _task_agents[ts.task_id] = agent

        # Process initial prompt
        if len(ts.messages) == 1 and ts.messages[0]["role"] == "user":
            result = agent(ts.prompt)
            ts.messages = list(agent.messages)
            ts._save()
            ts.append_result(f"Initial response: {str(result)[:5000]}")

        # Message queue loop
        empty_checks = 0
        while ts.status == "running" and not ts.paused:
            try:
                msg = ts.queue.get(timeout=2.0)
                empty_checks = 0
                result = agent(msg)
                ts.messages = list(agent.messages)
                ts._save()
                ts.append_result(f"Response to '{msg[:100]}': {str(result)[:5000]}")
                ts.queue.task_done()
            except queue.Empty:
                empty_checks += 1
                if empty_checks >= 5:
                    ts.set_status("completed")
                    break

        elapsed = time.time() - start
        if ts.status == "running":
            ts.set_status("completed")
        ts.append_result(f"Task finished in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"Task {ts.task_id} error: {e}\n{traceback.format_exc()}")
        ts.set_status("error")
        ts.append_result(f"ERROR: {e}\n{traceback.format_exc()}")


@tool
def tasks(
    action: str,
    task_id: Optional[str] = None,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    message: Optional[str] = None,
    tools: Optional[List[str]] = None,
    timeout: int = 900,
    agent: Any = None,
) -> Dict[str, Any]:
    """📋 Background task manager - run parallel agent tasks.

    Args:
        action: create, status, list, stop, pause, resume, add_message, get_result, get_messages
        task_id: Unique task identifier (auto-generated for create if omitted)
        prompt: Initial prompt (required for create)
        system_prompt: System prompt for task agent (required for create)
        message: Message to add (for add_message)
        tools: Tool names to enable (empty = all parent tools)
        timeout: Task timeout in seconds (default: 900)
        agent: Parent agent instance

    Returns:
        Dict with status and content
    """
    action = action.lower()

    if action == "create":
        if not prompt:
            return {
                "status": "error",
                "content": [{"text": "prompt required for create"}],
            }
        if not system_prompt:
            return {
                "status": "error",
                "content": [{"text": "system_prompt required for create"}],
            }

        task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"

        if task_id in _task_threads and _task_threads[task_id].is_alive():
            return {
                "status": "error",
                "content": [{"text": f"Task '{task_id}' already running"}],
            }

        ts = TaskState(task_id, prompt, system_prompt, tools, timeout)
        _task_states[task_id] = ts

        # 🔔 Emit task create event
        TaskState._emit_event("task.create", f"'{task_id}' started", prompt[:150],
                              {"task_id": task_id})

        t = threading.Thread(
            target=_run_task, args=(ts, agent), name=f"task-{task_id}", daemon=True
        )
        t.start()
        _task_threads[task_id] = t

        return {
            "status": "success",
            "content": [
                {"text": f"📋 Task '{task_id}' created and running in background"},
                {"text": f"Prompt: {prompt[:200]}"},
                {"text": f"Results: {ts.result_path}"},
            ],
        }

    elif action == "status":
        if not task_id:
            return {"status": "error", "content": [{"text": "task_id required"}]}

        ts = _task_states.get(task_id) or TaskState.load(task_id)
        if not ts:
            return {
                "status": "error",
                "content": [{"text": f"Task '{task_id}' not found"}],
            }

        is_alive = task_id in _task_threads and _task_threads[task_id].is_alive()
        return {
            "status": "success",
            "content": [
                {
                    "text": f"📋 Task '{task_id}': {ts.status} | Thread: {'alive' if is_alive else 'dead'} | Created: {ts.created_at}"
                },
            ],
        }

    elif action == "list":
        all_ids = set(_task_states.keys())
        for p in TASKS_DIR.glob("*_state.json"):
            all_ids.add(p.name.replace("_state.json", ""))

        if not all_ids:
            return {"status": "success", "content": [{"text": "No tasks found."}]}

        lines = []
        for tid in sorted(all_ids):
            ts = _task_states.get(tid) or TaskState.load(tid)
            if not ts:
                continue
            alive = tid in _task_threads and _task_threads[tid].is_alive()
            lines.append(
                f"  {'🟢' if alive else '⚪'} {tid}: {ts.status} (created {ts.created_at[:19]})"
            )

        return {
            "status": "success",
            "content": [{"text": f"📋 Tasks ({len(lines)}):\n" + "\n".join(lines)}],
        }

    elif action == "add_message":
        if not task_id or not message:
            return {
                "status": "error",
                "content": [{"text": "task_id and message required"}],
            }

        ts = _task_states.get(task_id) or TaskState.load(task_id)
        if not ts:
            return {
                "status": "error",
                "content": [{"text": f"Task '{task_id}' not found"}],
            }

        is_alive = task_id in _task_threads and _task_threads[task_id].is_alive()
        if is_alive:
            ts.queue.put(message)
            return {
                "status": "success",
                "content": [{"text": f"Message queued for running task '{task_id}'"}],
            }
        else:
            ts.messages.append({"role": "user", "content": [{"text": message}]})
            ts._save()
            t = threading.Thread(
                target=_run_task, args=(ts, agent), name=f"task-{task_id}", daemon=True
            )
            t.start()
            _task_threads[task_id] = t
            return {
                "status": "success",
                "content": [{"text": f"Message added and task '{task_id}' restarted"}],
            }

    elif action == "get_result":
        if not task_id:
            return {"status": "error", "content": [{"text": "task_id required"}]}
        result_path = TASKS_DIR / f"{task_id}_result.txt"
        if not result_path.exists():
            return {
                "status": "error",
                "content": [{"text": f"No results for '{task_id}'"}],
            }
        content = result_path.read_text()[-10000:]  # Last 10k chars
        return {
            "status": "success",
            "content": [{"text": f"📋 Results for '{task_id}':\n{content}"}],
        }

    elif action == "get_messages":
        if not task_id:
            return {"status": "error", "content": [{"text": "task_id required"}]}
        ts = _task_states.get(task_id) or TaskState.load(task_id)
        if not ts:
            return {
                "status": "error",
                "content": [{"text": f"Task '{task_id}' not found"}],
            }

        lines = []
        for i, msg in enumerate(ts.messages):
            role = msg.get("role", "?")
            blocks = msg.get("content", [])
            parts = []
            for b in blocks:
                if "text" in b:
                    parts.append(b["text"][:200])
                elif "toolUse" in b:
                    parts.append(f"[tool: {b['toolUse'].get('name', '?')}]")
                elif "toolResult" in b:
                    parts.append(f"[result: {b['toolResult'].get('status', '?')}]")
            lines.append(f"  [{i}] {role}: {' | '.join(parts)}")

        return {
            "status": "success",
            "content": [
                {
                    "text": f"📋 Messages for '{task_id}' ({len(ts.messages)}):\n"
                    + "\n".join(lines)
                }
            ],
        }

    elif action == "stop":
        if not task_id:
            return {"status": "error", "content": [{"text": "task_id required"}]}
        ts = _task_states.get(task_id) or TaskState.load(task_id)
        if ts:
            ts.set_status("stopped")
        return {"status": "success", "content": [{"text": f"Task '{task_id}' stopped"}]}

    elif action == "pause":
        if not task_id:
            return {"status": "error", "content": [{"text": "task_id required"}]}
        ts = _task_states.get(task_id) or TaskState.load(task_id)
        if not ts:
            return {
                "status": "error",
                "content": [{"text": f"Task '{task_id}' not found"}],
            }
        ts.paused = True
        ts.set_status("paused")
        return {"status": "success", "content": [{"text": f"Task '{task_id}' paused"}]}

    elif action == "resume":
        if not task_id:
            return {"status": "error", "content": [{"text": "task_id required"}]}
        ts = _task_states.get(task_id) or TaskState.load(task_id)
        if not ts:
            return {
                "status": "error",
                "content": [{"text": f"Task '{task_id}' not found"}],
            }
        ts.paused = False
        ts.set_status("resuming")

        is_alive = task_id in _task_threads and _task_threads[task_id].is_alive()
        if not is_alive:
            t = threading.Thread(
                target=_run_task, args=(ts, agent), name=f"task-{task_id}", daemon=True
            )
            t.start()
            _task_threads[task_id] = t
        return {"status": "success", "content": [{"text": f"Task '{task_id}' resumed"}]}

    else:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"Unknown action: {action}. Valid: create, status, list, stop, pause, resume, add_message, get_result, get_messages"
                }
            ],
        }
