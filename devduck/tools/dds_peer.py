"""DDS peer tool — CycloneDDS-based native ROS2 interop.

Scaffold (first of several surgical commits):
- Start/stop a DDS DomainParticipant
- Track tool state in DDS_STATE (mirrors ZENOH_STATE / ZCM_STATE patterns)
- Emit lifecycle events to event_bus (dds.start, dds.stop)
- Status action exposes liveness to the agent

Discovery, pub/sub, and ROS2 type handling arrive in subsequent commits.

Design notes:
- CycloneDDS is imported lazily inside the start() path so that DevDuck
  instances without the `cyclonedds` package still load this tool and
  report a clean "not available" error — no import-time crashes.
- One process = one DomainParticipant per domain_id. The DDS domain
  defaults to 0 (the ROS2 default). Override with ROS_DOMAIN_ID env var
  or the domain_id argument.
- Threading model matches zcm_peer.py: background daemon threads owned
  by the tool, state mutations behind a single module-level lock.
"""

import logging
import os
import socket
import threading
import time
import uuid
from typing import Any, Dict, Optional

from strands import tool

logger = logging.getLogger(__name__)

# ── Global state (mirrors ZENOH_STATE / ZCM_STATE) ──────────────────
DDS_STATE: Dict[str, Any] = {
    "running": False,
    "participant": None,          # cyclonedds.domain.DomainParticipant
    "subscriber": None,           # cyclonedds.sub.Subscriber for builtin topics
    "instance_id": None,
    "domain_id": 0,
    "start_time": None,
    "agent": None,
    # Populated by later commits:
    "participants": {},           # guid -> participant info
    "publications": {},           # (topic, type) -> publisher info
    "subscriptions": {},          # (topic, type) -> subscriber info
    "topic_types": {},            # topic_name -> type_name (observed)
    "discovery_thread": None,
    "discovery_running": False,
}

_STATE_LOCK = threading.RLock()


def _get_instance_id() -> str:
    """Deterministic-per-process DDS peer id (hostname + short uuid)."""
    if DDS_STATE["instance_id"]:
        return DDS_STATE["instance_id"]
    host = socket.gethostname().split(".")[0][:8]
    suffix = uuid.uuid4().hex[:6]
    iid = f"{host}-dds-{suffix}"
    DDS_STATE["instance_id"] = iid
    return iid


def _emit(event_type: str, payload: Dict[str, Any]) -> None:
    """Best-effort emit to devduck.tools.event_bus (no hard dependency)."""
    try:
        from devduck.tools.event_bus import bus  # type: ignore
        bus.emit(event_type, payload, source="dds_peer")
    except Exception:  # pragma: no cover - event_bus is optional at import time
        pass


def _cyclonedds_available() -> Optional[str]:
    """Return None if cyclonedds importable, else a human-readable reason."""
    try:
        import cyclonedds  # noqa: F401
        from cyclonedds.domain import DomainParticipant  # noqa: F401
        return None
    except ImportError as e:
        return f"cyclonedds not installed: {e}. Install with: pip install cyclonedds"
    except Exception as e:  # pragma: no cover
        return f"cyclonedds import error: {e}"


# ── Core lifecycle ──────────────────────────────────────────────────
def _start(domain_id: int, agent: Any = None) -> Dict[str, Any]:
    reason = _cyclonedds_available()
    if reason is not None:
        return {"status": "error", "content": [{"text": f"🦆 dds_peer: {reason}"}]}

    with _STATE_LOCK:
        if DDS_STATE["running"]:
            iid = DDS_STATE["instance_id"]
            return {
                "status": "success",
                "content": [{"text": f"🦆 dds_peer already running as {iid} (domain {DDS_STATE['domain_id']})"}],
            }

        # Respect ROS_DOMAIN_ID if the caller didn't set domain_id explicitly.
        if domain_id == 0 and os.environ.get("ROS_DOMAIN_ID"):
            try:
                domain_id = int(os.environ["ROS_DOMAIN_ID"])
            except ValueError:
                logger.warning("Invalid ROS_DOMAIN_ID=%r, falling back to 0", os.environ["ROS_DOMAIN_ID"])

        # Lazy import — we already verified availability above.
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.sub import Subscriber

        try:
            participant = DomainParticipant(domain_id)
            subscriber = Subscriber(participant)
        except Exception as e:
            logger.exception("Failed to create DDS participant")
            return {"status": "error", "content": [{"text": f"🦆 dds_peer start failed: {e}"}]}

        DDS_STATE.update(
            {
                "running": True,
                "participant": participant,
                "subscriber": subscriber,
                "domain_id": domain_id,
                "start_time": time.time(),
                "agent": agent,
            }
        )
        iid = _get_instance_id()

    _emit(
        "dds.start",
        {"instance_id": iid, "domain_id": domain_id, "guid": str(participant.guid)},
    )
    logger.info("dds_peer started as %s on domain %d (guid=%s)", iid, domain_id, participant.guid)

    return {
        "status": "success",
        "content": [
            {
                "text": (
                    f"🦆 dds_peer started\n"
                    f"  Instance ID: {iid}\n"
                    f"  Domain: {domain_id}\n"
                    f"  GUID: {participant.guid}\n"
                    f"  (discovery/pub/sub arrive in next commits)"
                )
            }
        ],
    }


def _stop() -> Dict[str, Any]:
    with _STATE_LOCK:
        if not DDS_STATE["running"]:
            return {"status": "success", "content": [{"text": "🦆 dds_peer not running"}]}

        iid = DDS_STATE.get("instance_id") or "unknown"
        # Signal any background loops spawned by future commits.
        DDS_STATE["discovery_running"] = False

        # Drop references — CycloneDDS Python cleans up via __del__.
        DDS_STATE["subscriber"] = None
        DDS_STATE["participant"] = None
        DDS_STATE["running"] = False
        DDS_STATE["start_time"] = None
        DDS_STATE["participants"].clear()
        DDS_STATE["publications"].clear()
        DDS_STATE["subscriptions"].clear()
        DDS_STATE["topic_types"].clear()

    _emit("dds.stop", {"instance_id": iid})
    logger.info("dds_peer stopped (%s)", iid)
    return {"status": "success", "content": [{"text": f"🦆 dds_peer stopped ({iid})"}]}


def _status() -> Dict[str, Any]:
    with _STATE_LOCK:
        if not DDS_STATE["running"]:
            return {"status": "success", "content": [{"text": "🦆 dds_peer: stopped"}]}
        uptime = time.time() - (DDS_STATE["start_time"] or time.time())
        lines = [
            "🦆 dds_peer status:",
            f"  Instance ID : {DDS_STATE['instance_id']}",
            f"  Domain      : {DDS_STATE['domain_id']}",
            f"  GUID        : {DDS_STATE['participant'].guid if DDS_STATE['participant'] else 'n/a'}",
            f"  Uptime      : {uptime:.1f}s",
            f"  Participants: {len(DDS_STATE['participants'])} (populated by discovery — next commit)",
            f"  Topics seen : {len(DDS_STATE['topic_types'])}",
        ]
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}


# ── Tool entry point ────────────────────────────────────────────────
@tool
def dds_peer(
    action: str,
    domain_id: int = 0,
    agent: Any = None,
) -> Dict[str, Any]:
    """DDS peer: native ROS2 fleet interop over CycloneDDS.

    Scaffold actions (more added in follow-up commits):
        start   — create a DomainParticipant on the given domain
        stop    — tear down the participant
        status  — current liveness + counters

    Args:
        action: start | stop | status
        domain_id: DDS domain id (default 0; honors ROS_DOMAIN_ID env)
        agent: optional reference to the parent DevDuck agent (for future
            actions that need to invoke other tools)

    Returns:
        Standard Strands tool result dict.
    """
    a = (action or "").strip().lower()
    if a == "start":
        return _start(domain_id=domain_id, agent=agent)
    if a == "stop":
        return _stop()
    if a == "status":
        return _status()
    return {
        "status": "error",
        "content": [{"text": f"🦆 dds_peer: unknown action '{action}' (use start|stop|status)"}],
    }
