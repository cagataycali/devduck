"""⏰ Scheduler tool for DevDuck — Python-native cron & one-time job scheduling.

Persists jobs to /tmp/.devduck/scheduler/jobs.json (survives on Android/constrained devices).
Uses Python's sched + threading for precise timing — no external dependencies.

Each scheduled job spawns a full DevDuck session via use_agent with configurable:
- system_prompt, tools, model, max_tokens, context
- cron (recurring) or run_at (one-time)
- Catch-up window for missed jobs after restart
- Standard cron DOW (Sunday=0)

The scheduler auto-starts on DevDuck init and catches up on missed jobs.

Examples:
    scheduler(action="add", name="review", schedule="0 9 * * *",
              prompt="Review open PRs", system_prompt="You are a code reviewer",
              tools="strands_tools:shell;devduck.tools:use_github")
    scheduler(action="add", name="deploy", run_at="2026-03-28T15:00:00",
              prompt="Deploy to prod", once=True)
    scheduler(action="list")
    scheduler(action="run_now", name="review")
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from strands import tool

logger = logging.getLogger(__name__)

# ── Persistence ─────────────────────────────────────────────────
# /tmp/.devduck survives on Android/constrained devices better than ~/.devduck
SCHEDULER_DIR = Path(os.getenv(
    "DEVDUCK_SCHEDULER_DIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / ".devduck" / "scheduler"),
))
SCHEDULER_DIR.mkdir(parents=True, exist_ok=True)
JOBS_FILE = SCHEDULER_DIR / "jobs.json"
HISTORY_FILE = SCHEDULER_DIR / "history.json"

# Catch-up window: run missed jobs up to this many seconds old on startup
CATCH_UP_WINDOW = int(os.getenv("DEVDUCK_SCHEDULER_CATCHUP", "86400"))  # 24h default
LOCK_DIR = SCHEDULER_DIR / "locks"
LOCK_DIR.mkdir(parents=True, exist_ok=True)

# ── Runtime state (module-level so TUI sidebar can read it) ─────
_state: Dict[str, Any] = {
    "running": False,
    "thread": None,
    "stop_event": None,
    "jobs": {},      # live mirror of jobs.json
    "agent": None,
}


# ── JSON persistence ────────────────────────────────────────────

def _load_jobs() -> Dict[str, dict]:
    """Load jobs from disk. Thread-safe via atomic read."""
    if JOBS_FILE.exists():
        try:
            data = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
            # Migrate old format: ensure all jobs have new fields
            for name, job in data.items():
                job.setdefault("tools", None)
                job.setdefault("model", None)
                job.setdefault("max_tokens", None)
                job.setdefault("context", None)
                job.setdefault("last_triggered", 0)
                job.setdefault("run_count", 0)
                job.setdefault("last_result", None)
                job.setdefault("last_duration", None)
                job.setdefault("last_status", None)
                job.setdefault("created_at", None)
            return data
        except Exception as e:
            logger.warning(f"Failed to load jobs.json: {e}")
    return {}


def _save_jobs(jobs: dict):
    """Save jobs to disk atomically (write tmp + rename)."""
    tmp = JOBS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, indent=2, default=str), encoding="utf-8")
    tmp.replace(JOBS_FILE)
    # Update live mirror
    _state["jobs"] = jobs


def _load_history() -> List[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_history(history: list):
    HISTORY_FILE.write_text(
        json.dumps(history[-500:], indent=2, default=str), encoding="utf-8"
    )


# ── Cron parsing (Sunday=0, standard cron) ──────────────────────

def _parse_cron(expr: str) -> Optional[dict]:
    """Parse 5-field cron: min hour dom month dow. Returns None on error."""
    parts = expr.strip().split()
    if len(parts) != 5:
        return None
    return dict(zip(("minute", "hour", "dom", "month", "dow"), parts))


def _field_matches(pattern: str, value: int) -> bool:
    """Check if a cron field pattern matches value."""
    if pattern == "*":
        return True
    for part in pattern.split(","):
        if "/" in part:
            base, step = part.split("/", 1)
            base_val = 0 if base == "*" else int(base)
            step_val = int(step)
            if step_val > 0 and value >= base_val and (value - base_val) % step_val == 0:
                return True
        elif "-" in part:
            lo, hi = part.split("-", 1)
            if int(lo) <= value <= int(hi):
                return True
        else:
            if int(part) == value:
                return True
    return False


def _cron_matches(cron: dict, dt: datetime) -> bool:
    """Check if datetime matches cron. DOW: Sunday=0 (standard cron)."""
    # Convert Python weekday (Mon=0) to cron (Sun=0)
    cron_dow = (dt.weekday() + 1) % 7
    return (
        _field_matches(cron["minute"], dt.minute)
        and _field_matches(cron["hour"], dt.hour)
        and _field_matches(cron["dom"], dt.day)
        and _field_matches(cron["month"], dt.month)
        and _field_matches(cron["dow"], cron_dow)
    )


# ── Distributed file-based lock ─────────────────────────────────

def _try_acquire_job_lock(name: str, minute_key: str) -> bool:
    """Try to acquire a file-based lock for a job+minute combination.

    Uses atomic O_CREAT|O_EXCL to ensure only one process wins.
    Lock files are named: {job_name}_{YYYYMMDD_HHMM}.lock
    Returns True if this process acquired the lock (should execute).
    """
    lock_file = LOCK_DIR / f"{name}_{minute_key}.lock"
    try:
        # O_CREAT | O_EXCL = atomic create-if-not-exists (fails if file exists)
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        # Write our PID so we can debug who holds the lock
        os.write(fd, f"{os.getpid()}:{time.time()}".encode())
        os.close(fd)
        logger.debug(f"⏰ Lock acquired for {name} @ {minute_key} (pid {os.getpid()})")
        return True
    except FileExistsError:
        logger.debug(f"⏰ Lock already held for {name} @ {minute_key} — skipping")
        return False
    except Exception as e:
        logger.warning(f"⏰ Lock error for {name}: {e}")
        return False


def _cleanup_old_locks(max_age_seconds: int = 7200):
    """Remove lock files older than max_age to prevent disk buildup."""
    try:
        now = time.time()
        for lock_file in LOCK_DIR.glob("*.lock"):
            try:
                if now - lock_file.stat().st_mtime > max_age_seconds:
                    lock_file.unlink()
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Lock cleanup error: {e}")


# ── Event bus integration ───────────────────────────────────────

def _emit(event_type: str, summary: str, detail: str = "", metadata: dict = None):
    try:
        from devduck.tools.event_bus import emit
        emit(event_type, "scheduler", summary, detail, metadata)
    except ImportError:
        pass


# ── Job execution ───────────────────────────────────────────────

def _push_to_mesh(event_type: str, name: str, detail: str):
    """Push scheduler events to Zenoh peers and unified ring context."""
    # Push to unified ring
    try:
        from devduck.tools.unified_mesh import add_to_ring
        add_to_ring(
            f"scheduler:{name}",
            "scheduler",
            f"[{event_type}] {detail}",
            {"source": "scheduler", "job": name},
        )
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Ring push failed: {e}")

    # Broadcast to Zenoh peers
    try:
        import sys as _sys
        _zp_mod = _sys.modules.get("devduck.tools.zenoh_peer")
        if _zp_mod:
            ZENOH_STATE = _zp_mod.ZENOH_STATE
            if ZENOH_STATE.get("running") and ZENOH_STATE.get("session"):
                import zenoh
                session = ZENOH_STATE["session"]
                instance_id = ZENOH_STATE.get("instance_id", "unknown")
                import json as _json
                payload = _json.dumps({
                    "type": "scheduler_event",
                    "event": event_type,
                    "job": name,
                    "detail": detail[:500],
                    "from": instance_id,
                }).encode()
                session.put(f"devduck/scheduler/{event_type}", payload)
                logger.debug(f"Zenoh scheduler event published: {event_type} for {name}")
    except Exception as e:
        logger.debug(f"Zenoh push failed: {e}")


class _EnvOverride:
    """Context manager to temporarily override env vars for per-job DevDuck spawning."""

    def __init__(self, overrides: Dict[str, Optional[str]]):
        self.overrides = {k: v for k, v in overrides.items() if v is not None}
        self.original: Dict[str, Optional[str]] = {}

    def __enter__(self):
        for k, v in self.overrides.items():
            self.original[k] = os.environ.get(k)
            os.environ[k] = str(v)
        return self

    def __exit__(self, *exc):
        for k, original in self.original.items():
            if original is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = original


def _spawn_devduck_for_job(job: dict):
    """Spawn a fresh DevDuck instance configured for this scheduled job.

    Applies per-job overrides (model, tools, system_prompt) via env vars so
    the spawned DevDuck picks them up during __init__. The returned instance
    goes through the full DevDuck __call__ path, which means it gets all
    dynamic context injection (zenoh peers, ring, ambient, listen transcripts,
    event bus, AGENTS.md, recording events, KB retrieval/storage, mesh sync).
    """
    from devduck import DevDuck

    name = job["name"]
    sys_prompt_base = (
        job.get("system_prompt")
        or "You are executing a scheduled DevDuck job. Be concise and efficient."
    )

    # Augment system prompt with scheduler context
    run_count = job.get("run_count", 0)
    last_status = job.get("last_status") or "never"
    last_result = (job.get("last_result") or "")[:500]
    schedule_info = job.get("schedule") or f"run_at={job.get('run_at', '?')}"

    sys_prompt = f"""{sys_prompt_base}

## Scheduled Job Context:
- **Job Name**: {name}
- **Schedule**: {schedule_info}
- **Run Count**: {run_count} (this is execution #{run_count + 1})
- **Last Status**: {last_status}
- **Triggered At**: {datetime.now().isoformat()}
"""
    if last_result:
        sys_prompt += f"- **Last Result Preview**: {last_result[:300]}\n"

    # Per-job env overrides
    overrides: Dict[str, Optional[str]] = {
        "SYSTEM_PROMPT": sys_prompt,
    }
    if job.get("model"):
        overrides["STRANDS_MODEL_ID"] = job["model"]
    if job.get("tools"):
        # tools can be either comma-separated tool names OR full DEVDUCK_TOOLS config
        tools_str = job["tools"].strip()
        if ":" in tools_str or ";" in tools_str:
            # Full DEVDUCK_TOOLS config (e.g., "strands_tools:shell;devduck.tools:use_github")
            overrides["DEVDUCK_TOOLS"] = tools_str
        else:
            # Simple comma list → assume devduck.tools + strands_tools:shell fallback
            # Let the user be explicit for anything fancier
            overrides["DEVDUCK_TOOLS"] = f"devduck.tools:{tools_str};strands_tools:shell"
    if job.get("max_tokens"):
        overrides["STRANDS_MAX_TOKENS"] = str(job["max_tokens"])

    with _EnvOverride(overrides):
        # auto_start_servers=False — scheduled jobs shouldn't fight for ports
        return DevDuck(auto_start_servers=False)


def _execute_job(job: dict, agent: Any = None) -> dict:
    """Execute a scheduled job as a full DevDuck session.

    Spawns a fresh DevDuck(auto_start_servers=False) per job. The spawned
    instance is invoked via its __call__ method (NOT the bare agent), which
    means it receives all dynamic context injection: zenoh peers, ring
    context, ambient status, recording events, listen transcripts, event
    bus context, AGENTS.md, own source code, KB retrieval/storage, and
    pushes results to the unified ring.

    `agent` parameter kept for API compatibility but no longer required —
    each job gets its own fresh DevDuck. If provided, can be used as fallback.

    Returns execution record dict.
    """
    name = job["name"]
    prompt = job["prompt"]

    logger.info(f"⏰ Executing scheduled job: {name}")
    print(f"\n⏰ [scheduler] Running '{name}'...")
    _emit("schedule.fire", f"Running '{name}'", prompt[:200], {"job": name})
    _push_to_mesh("fire", name, f"Running: {prompt[:200]}")

    record = {
        "name": name,
        "started_at": datetime.now().isoformat(),
        "prompt": prompt[:200],
        "status": "running",
    }

    t0 = time.time()

    try:
        # Spawn a fresh DevDuck instance for this job
        job_devduck = _spawn_devduck_for_job(job)

        if not job_devduck.agent:
            record["result"] = "DevDuck instance initialization failed"
            record["status"] = "error"
            raise RuntimeError("DevDuck agent unavailable in spawned instance")

        # Invoke via wrapper (__call__) — this triggers full context injection
        result = job_devduck(prompt)
        result_text = str(result)

        record["result"] = result_text[:3000]
        record["status"] = "success"
        print(f"⏰ [scheduler] '{name}' completed.")
        _emit("schedule.done", f"'{name}' completed", result_text[:200], {"job": name})
        _push_to_mesh("done", name, f"Completed: {result_text[:300]}")

    except Exception as e:
        record["result"] = str(e)[:1000]
        record["status"] = "error"
        logger.error(f"Scheduler job '{name}' failed: {e}", exc_info=True)
        print(f"⏰ [scheduler] '{name}' failed: {e}")
        _emit("schedule.error", f"'{name}' failed", str(e)[:200], {"job": name})
        _push_to_mesh("error", name, f"Failed: {str(e)[:300]}")

    duration = time.time() - t0
    record["finished_at"] = datetime.now().isoformat()
    record["duration_seconds"] = round(duration, 1)

    # Save history
    history = _load_history()
    history.append(record)
    _save_history(history)

    return record


# ── Catch-up: run missed jobs on startup ────────────────────────

def _catch_up_jobs(jobs: dict, agent: Any):
    """Check for jobs that should have run while we were down."""
    now = datetime.now()
    now_epoch = time.time()
    updated = False

    for name, job in list(jobs.items()):
        if not job.get("enabled", True):
            continue
        last_triggered = job.get("last_triggered", 0)
        if not last_triggered:
            continue  # Never ran, don't catch up

        # Only catch up if within window
        age = now_epoch - last_triggered
        if age > CATCH_UP_WINDOW:
            continue

        should_catch_up = False

        # Cron jobs: check if a trigger was missed
        if job.get("cron_parsed"):
            cron = job["cron_parsed"]
            if _cron_matches(cron, now):
                should_catch_up = True

        # One-time jobs: check if run_at was missed
        if job.get("run_at") and not job.get("executed"):
            try:
                run_at = datetime.fromisoformat(job["run_at"])
                if now >= run_at:
                    should_catch_up = True
            except ValueError:
                pass

        if should_catch_up:
            # Distributed lock: only one process catches up a given job
            catchup_key = f"catchup_{now.strftime('%Y%m%d_%H%M')}"
            if not _try_acquire_job_lock(name, catchup_key):
                logger.info(f"⏰ Catch-up for '{name}' already claimed — skipping")
                continue

            logger.info(f"⏰ Catching up missed job: {name}")
            print(f"⏰ [scheduler] Catching up missed job: {name}")
            record = _execute_job(job, agent)
            job["last_triggered"] = int(now_epoch)
            job["run_count"] = job.get("run_count", 0) + 1
            job["last_result"] = (record.get("result") or "")[:500]
            job["last_duration"] = record.get("duration_seconds")
            job["last_status"] = record.get("status")
            if job.get("run_at") and job.get("once"):
                job["executed"] = True
            updated = True

    if updated:
        _save_jobs(jobs)


# ── Scheduler loop ──────────────────────────────────────────────

def _scheduler_loop(stop_event: threading.Event):
    """Main scheduler loop — checks every 15 seconds, fires once per minute."""
    last_check_minute = -1
    last_lock_cleanup = 0

    while not stop_event.is_set():
        try:
            now = datetime.now()
            now_epoch = time.time()

            # Only fire once per minute
            if now.minute == last_check_minute:
                stop_event.wait(15)
                continue
            last_check_minute = now.minute

            # Periodic lock cleanup (every 30 minutes)
            if now_epoch - last_lock_cleanup > 1800:
                _cleanup_old_locks()
                last_lock_cleanup = now_epoch

            jobs = _load_jobs()
            _state["jobs"] = jobs  # keep live mirror fresh
            agent = _state.get("agent")

            for name, job in list(jobs.items()):
                if not job.get("enabled", True):
                    continue

                # One-time job
                if job.get("run_at"):
                    try:
                        run_at = datetime.fromisoformat(job["run_at"])
                    except ValueError:
                        continue
                    if now >= run_at and not job.get("executed"):
                        # Distributed lock: only one process executes one-time jobs
                        once_key = f"once_{name}"
                        if not _try_acquire_job_lock(name, once_key):
                            logger.info(f"⏰ One-time job '{name}' already claimed — skipping")
                            continue

                        record = _execute_job(job, agent)
                        job["executed"] = True
                        job["last_triggered"] = int(now_epoch)
                        job["run_count"] = job.get("run_count", 0) + 1
                        job["last_result"] = (record.get("result") or "")[:500]
                        job["last_duration"] = record.get("duration_seconds")
                        job["last_status"] = record.get("status")
                        _save_jobs(jobs)
                    continue

                # Cron job
                cron = job.get("cron_parsed")
                if cron and _cron_matches(cron, now):
                    # Prevent double-fire in same minute (local check)
                    last = job.get("last_triggered", 0)
                    if last:
                        lt = datetime.fromtimestamp(last)
                        if lt.minute == now.minute and lt.hour == now.hour and lt.date() == now.date():
                            continue

                    # Distributed lock: only one process executes per job per minute
                    minute_key = now.strftime("%Y%m%d_%H%M")
                    if not _try_acquire_job_lock(name, minute_key):
                        logger.info(f"⏰ Job '{name}' already claimed by another instance — skipping")
                        continue

                    record = _execute_job(job, agent)
                    job["last_triggered"] = int(now_epoch)
                    job["run_count"] = job.get("run_count", 0) + 1
                    job["last_result"] = (record.get("result") or "")[:500]
                    job["last_duration"] = record.get("duration_seconds")
                    job["last_status"] = record.get("status")
                    _save_jobs(jobs)

        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")

        stop_event.wait(15)

    logger.info("Scheduler loop stopped")


# ── Auto-start helper (called from __init__.py) ────────────────

def auto_start_scheduler(agent: Any = None):
    """Start the scheduler daemon if not already running. Called on DevDuck init."""
    if _state["running"]:
        return
    jobs = _load_jobs()
    if not jobs:
        _state["jobs"] = {}
        return  # No jobs, no need to start

    _state["agent"] = agent
    _state["jobs"] = jobs
    stop_event = threading.Event()
    _state["stop_event"] = stop_event
    _state["running"] = True

    t = threading.Thread(target=_scheduler_loop, args=(stop_event,), daemon=True)
    t.start()
    _state["thread"] = t

    logger.info(f"⏰ Scheduler auto-started with {len(jobs)} jobs")

    # Catch up missed jobs
    try:
        _catch_up_jobs(jobs, agent)
    except Exception as e:
        logger.error(f"Catch-up failed: {e}")


# ── Formatting helpers ──────────────────────────────────────────

def _format_epoch(epoch: int) -> str:
    """Format epoch to human readable, or 'never'."""
    if not epoch:
        return "never"
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")


def _format_job_detail(name: str, j: dict) -> str:
    """Format a single job for display."""
    enabled = "✅" if j.get("enabled", True) else "⏸️"
    jtype = "🔄 cron" if j.get("schedule") else "📅 once"
    sched = j.get("schedule") or f"at {j.get('run_at', '?')}"
    runs = j.get("run_count", 0)
    last = _format_epoch(j.get("last_triggered", 0))
    last_status = j.get("last_status") or "—"
    last_dur = f"{j['last_duration']:.1f}s" if j.get("last_duration") else "—"
    prompt_preview = (j.get("prompt") or "")[:80]

    lines = [
        f"  {enabled} **{name}** [{jtype}] `{sched}`",
        f"     runs: {runs} | last: {last} | status: {last_status} | dur: {last_dur}",
        f"     → {prompt_preview}",
    ]
    if j.get("system_prompt"):
        lines.append(f"     📝 sys: {j['system_prompt'][:60]}…")
    if j.get("tools"):
        lines.append(f"     🔧 tools: {j['tools']}")
    if j.get("model"):
        lines.append(f"     🧠 model: {j['model']}")
    return "\n".join(lines)


# ── The tool ────────────────────────────────────────────────────

@tool
def scheduler(
    action: str,
    name: Optional[str] = None,
    schedule: Optional[str] = None,
    run_at: Optional[str] = None,
    once: bool = False,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    context: Optional[str] = None,
    enabled: bool = True,
    agent: Any = None,
) -> Dict[str, Any]:
    """⏰ Job scheduler - cron and one-time tasks with persistence.

    Schedules future DevDuck sessions with full configuration per job.
    Jobs persist to /tmp/.devduck/scheduler/jobs.json and survive restarts.
    Missed jobs are caught up automatically on startup (24h window).

    Args:
        action: Action to perform:
            - start: Start the scheduler daemon
            - stop: Stop the scheduler
            - status: Show scheduler status
            - add: Add a new job (requires name + schedule/run_at + prompt)
            - remove: Remove a job (requires name)
            - list: List all jobs with full details
            - enable: Enable a job (requires name)
            - disable: Disable a job (requires name)
            - history: Show execution history (optional: name to filter)
            - run_now: Execute a job immediately (requires name)
            - clear_history: Clear execution history
        name: Job name (unique identifier)
        schedule: Cron expression (e.g., "*/5 * * * *" = every 5 min, "0 9 * * 1" = Mon 9am)
            DOW: 0=Sunday (standard cron)
        run_at: ISO datetime for one-time job (e.g., "2026-03-04T15:00:00")
        once: If True, auto-disable after execution (for one-time jobs)
        prompt: Agent prompt to execute when job triggers
        system_prompt: Custom system prompt for the job agent
        tools: Comma-separated tool names to make available (e.g., "shell,file_read,use_github")
        model: Model ID override for this job (e.g., "us.anthropic.claude-sonnet-4-20250514-v1:0")
        max_tokens: Max tokens for model response
        context: Additional context injected into the prompt
        enabled: Whether the job is enabled (default: True)
        agent: Parent agent instance

    Returns:
        Dict with status and content
    """
    action = action.lower().strip()

    # ── START ────────────────────────────────────────────────
    if action == "start":
        if _state["running"]:
            jobs = _state.get("jobs", {})
            return _ok(f"⏰ Scheduler already running. {len(jobs)} jobs loaded.")

        _state["agent"] = agent
        jobs = _load_jobs()
        _state["jobs"] = jobs
        stop_event = threading.Event()
        _state["stop_event"] = stop_event
        _state["running"] = True

        t = threading.Thread(target=_scheduler_loop, args=(stop_event,), daemon=True)
        t.start()
        _state["thread"] = t

        # Catch up
        if jobs:
            try:
                _catch_up_jobs(jobs, agent)
            except Exception as e:
                logger.error(f"Catch-up error: {e}")

        return _ok(f"⏰ Scheduler started. {len(jobs)} jobs loaded. Catch-up window: {CATCH_UP_WINDOW}s")

    # ── STOP ─────────────────────────────────────────────────
    elif action == "stop":
        if not _state["running"]:
            return _ok("Scheduler not running.")
        _state["stop_event"].set()
        _state["running"] = False
        return _ok("⏰ Scheduler stopped.")

    # ── STATUS ───────────────────────────────────────────────
    elif action == "status":
        jobs = _load_jobs()
        active = sum(1 for j in jobs.values() if j.get("enabled", True))
        lines = [
            f"⏰ Scheduler: **{'running' if _state['running'] else 'stopped'}**",
            f"Jobs: {len(jobs)} total, {active} active",
            f"Storage: `{JOBS_FILE}`",
            f"Catch-up window: {CATCH_UP_WINDOW}s",
        ]
        return _ok("\n".join(lines))

    # ── ADD ──────────────────────────────────────────────────
    elif action == "add":
        if not name or not prompt:
            return _err("name and prompt required")
        if not schedule and not run_at:
            return _err("schedule (cron) or run_at (datetime) required")

        jobs = _load_jobs()

        job: Dict[str, Any] = {
            "name": name,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "tools": tools,
            "model": model,
            "max_tokens": max_tokens,
            "context": context,
            "enabled": enabled,
            "created_at": datetime.now().isoformat(),
            "run_count": 0,
            "last_triggered": 0,
            "last_result": None,
            "last_duration": None,
            "last_status": None,
        }

        if schedule:
            cron = _parse_cron(schedule)
            if not cron:
                return _err(f"Invalid cron: {schedule}. Format: min hour dom month dow (DOW: 0=Sun)")
            job["schedule"] = schedule
            job["cron_parsed"] = cron
            job["type"] = "cron"
        else:
            try:
                datetime.fromisoformat(run_at)
            except ValueError:
                return _err(f"Invalid datetime: {run_at}. Use ISO format.")
            job["run_at"] = run_at
            job["type"] = "once"
            job["once"] = once
            job["executed"] = False

        # Prepend context to prompt if provided
        if context:
            job["prompt"] = f"{context}\n\n{prompt}"

        is_update = name in jobs
        jobs[name] = job
        _save_jobs(jobs)

        # Auto-start scheduler if not running and we have jobs
        if not _state["running"] and agent:
            auto_start_scheduler(agent)

        verb = "updated" if is_update else "added"
        sched_info = schedule or f"at {run_at}"
        extras = []
        if system_prompt:
            extras.append(f"sys_prompt: {system_prompt[:50]}…")
        if tools:
            extras.append(f"tools: {tools}")
        if model:
            extras.append(f"model: {model}")
        extra_str = f"\n{' | '.join(extras)}" if extras else ""

        return _ok(f"⏰ Job '{name}' {verb} ({sched_info}){extra_str}")

    # ── REMOVE ───────────────────────────────────────────────
    elif action == "remove":
        if not name:
            return _err("name required")
        jobs = _load_jobs()
        if name not in jobs:
            return _err(f"Job '{name}' not found")
        del jobs[name]
        _save_jobs(jobs)
        return _ok(f"⏰ Job '{name}' removed")

    # ── LIST ─────────────────────────────────────────────────
    elif action == "list":
        jobs = _load_jobs()
        if not jobs:
            return _ok("No scheduled jobs. Use action='add' to create one.")

        lines = [f"⏰ Scheduled Jobs ({len(jobs)}):\n"]
        for n, j in jobs.items():
            lines.append(_format_job_detail(n, j))
            lines.append("")

        lines.append(f"Storage: `{JOBS_FILE}`")
        return _ok("\n".join(lines))

    # ── ENABLE / DISABLE ─────────────────────────────────────
    elif action in ("enable", "disable"):
        if not name:
            return _err("name required")
        jobs = _load_jobs()
        if name not in jobs:
            return _err(f"Job '{name}' not found")
        jobs[name]["enabled"] = action == "enable"
        _save_jobs(jobs)
        return _ok(f"⏰ Job '{name}' {'enabled' if action == 'enable' else 'disabled'}")

    # ── HISTORY ──────────────────────────────────────────────
    elif action == "history":
        history = _load_history()
        if name:
            history = [h for h in history if h.get("name") == name]
        if not history:
            return _ok(f"No history{f' for {name}' if name else ''}.")

        lines = [f"⏰ History (last {min(len(history), 20)}):\n"]
        for h in history[-20:]:
            emoji = {"success": "✅", "error": "❌", "skipped": "⏭️"}.get(h.get("status", ""), "❓")
            dur = f" ({h['duration_seconds']:.1f}s)" if h.get("duration_seconds") else ""
            lines.append(
                f"  {emoji} [{h.get('started_at', '?')[:19]}] {h.get('name', '?')}{dur}: "
                f"{(h.get('result') or '')[:120]}"
            )
        return _ok("\n".join(lines))

    # ── RUN_NOW ──────────────────────────────────────────────
    elif action == "run_now":
        if not name:
            return _err("name required")
        jobs = _load_jobs()
        if name not in jobs:
            return _err(f"Job '{name}' not found")

        run_agent = agent or _state.get("agent")
        if not run_agent:
            return _err("No agent available. Start scheduler first or pass agent=.")

        job = jobs[name]
        record = _execute_job(job, run_agent)
        jobs[name]["last_triggered"] = int(time.time())
        jobs[name]["run_count"] = jobs[name].get("run_count", 0) + 1
        jobs[name]["last_result"] = (record.get("result") or "")[:500]
        jobs[name]["last_duration"] = record.get("duration_seconds")
        jobs[name]["last_status"] = record.get("status")
        _save_jobs(jobs)

        return _ok(f"⏰ Job '{name}' executed. Status: {record.get('status')} ({record.get('duration_seconds', 0):.1f}s)")

    # ── CLEAR_HISTORY ────────────────────────────────────────
    elif action == "clear_history":
        _save_history([])
        return _ok("⏰ History cleared.")

    else:
        return _err(f"Unknown action: {action}. Valid: start, stop, status, add, remove, list, enable, disable, history, run_now, clear_history")


def _ok(text: str) -> Dict[str, Any]:
    return {"status": "success", "content": [{"text": text}]}


def _err(text: str) -> Dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}
