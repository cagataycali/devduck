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
    # Discovery registries (populated by the discovery thread):
    "participants": {},           # guid -> {"first_seen": ts, "last_seen": ts}
    "publications": {},           # (topic, type) -> {"participant": guid, "last_seen": ts}
    "subscriptions": {},          # (topic, type) -> {"participant": guid, "last_seen": ts}
    "topic_types": {},            # topic_name -> type_name (observed)
    # Background threads:
    "discovery_thread": None,
    "discovery_running": False,
    "builtin_readers": {},        # label -> BuiltinDataReader
}

# Discovery loop tuning.
DISCOVERY_POLL_INTERVAL = 1.0      # seconds between take() calls
DISCOVERY_STALE_AFTER = 30.0       # drop entries not seen in this window

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


# ── Discovery loop ──────────────────────────────────────────────────
def _strip_ros_prefix(topic_name: str) -> str:
    """Map DDS topic names back to their ROS2 names.

    ROS2 prefixes topics on the DDS wire so they don't clash with raw DDS:
        "rt/cmd_vel"  -> "/cmd_vel"   (topic)
        "rq/add_ints" -> service request (left alone; we handle in use_ros)
        "rr/add_ints" -> service reply
    """
    for prefix, replacement in (("rt/", "/"),):
        if topic_name.startswith(prefix):
            return replacement + topic_name[len(prefix):]
    return topic_name


def _classify_topic(topic_name: str) -> str:
    """Return 'topic' | 'service_req' | 'service_rep' | 'raw'."""
    if topic_name.startswith("rt/"):
        return "topic"
    if topic_name.startswith("rq/"):
        return "service_req"
    if topic_name.startswith("rr/"):
        return "service_rep"
    return "raw"


def _discovery_loop() -> None:
    """Poll DDS built-in readers, keep participants/topics registry fresh.

    CycloneDDS surfaces SPDP (participant) and SEDP (endpoint) discovery
    data as the built-in topics DCPSParticipant / DCPSPublication /
    DCPSSubscription. Reading them gives us an always-live picture of
    every ROS2 node, every publisher, every subscriber on the LAN.
    """
    from cyclonedds.builtin import (  # lazy: cyclonedds may be absent
        BuiltinDataReader,
        BuiltinTopicDcpsParticipant,
        BuiltinTopicDcpsPublication,
        BuiltinTopicDcpsSubscription,
    )

    with _STATE_LOCK:
        sub = DDS_STATE["subscriber"]

    readers = {
        "participant":  BuiltinDataReader(sub, BuiltinTopicDcpsParticipant),
        "publication":  BuiltinDataReader(sub, BuiltinTopicDcpsPublication),
        "subscription": BuiltinDataReader(sub, BuiltinTopicDcpsSubscription),
    }
    with _STATE_LOCK:
        DDS_STATE["builtin_readers"] = readers

    logger.info("dds_peer discovery loop started")
    while DDS_STATE.get("discovery_running"):
        now = time.time()
        try:
            _ingest_participants(readers["participant"], now)
            _ingest_endpoints(readers["publication"], now, kind="publication")
            _ingest_endpoints(readers["subscription"], now, kind="subscription")
            _reap_stale(now)
        except Exception:
            logger.exception("dds_peer discovery iteration failed")
        time.sleep(DISCOVERY_POLL_INTERVAL)

    logger.info("dds_peer discovery loop stopped")


def _ingest_participants(reader, now: float) -> None:
    samples = reader.take(N=128) or []
    for s in samples:
        guid = str(getattr(s, "key", ""))
        if not guid:
            continue
        with _STATE_LOCK:
            entry = DDS_STATE["participants"].get(guid)
            if entry is None:
                DDS_STATE["participants"][guid] = {"first_seen": now, "last_seen": now}
                _emit("dds.participant.join", {"guid": guid})
                logger.info("dds_peer: new participant %s", guid)
            else:
                entry["last_seen"] = now


def _ingest_endpoints(reader, now: float, *, kind: str) -> None:
    samples = reader.take(N=128) or []
    bucket = "publications" if kind == "publication" else "subscriptions"
    for s in samples:
        topic_name = getattr(s, "topic_name", None)
        type_name = getattr(s, "type_name", None)
        part_key = str(getattr(s, "participant_key", "")) or None
        if not topic_name or not type_name:
            continue
        key = (topic_name, type_name)
        with _STATE_LOCK:
            entry = DDS_STATE[bucket].get(key)
            if entry is None:
                DDS_STATE[bucket][key] = {
                    "participant": part_key,
                    "first_seen": now,
                    "last_seen": now,
                    "classification": _classify_topic(topic_name),
                }
                # Keep a quick topic -> type index for the agent UX.
                DDS_STATE["topic_types"].setdefault(topic_name, type_name)
                _emit(
                    "dds.endpoint.new",
                    {
                        "kind": kind,
                        "topic": topic_name,
                        "type": type_name,
                        "participant": part_key,
                    },
                )
                logger.info("dds_peer: new %s %s [%s]", kind, topic_name, type_name)
            else:
                entry["last_seen"] = now


def _reap_stale(now: float) -> None:
    """Drop registry entries that haven't been refreshed within the stale window."""
    cutoff = now - DISCOVERY_STALE_AFTER
    with _STATE_LOCK:
        for bucket in ("participants", "publications", "subscriptions"):
            stale = [k for k, v in DDS_STATE[bucket].items() if v["last_seen"] < cutoff]
            for k in stale:
                DDS_STATE[bucket].pop(k, None)
                if bucket == "participants":
                    _emit("dds.participant.leave", {"guid": k})
                    logger.info("dds_peer: participant %s gone", k)
        # Rebuild topic_types from the live publications+subscriptions sets.
        live_topics: Dict[str, str] = {}
        for (topic, type_name), _v in DDS_STATE["publications"].items():
            live_topics.setdefault(topic, type_name)
        for (topic, type_name), _v in DDS_STATE["subscriptions"].items():
            live_topics.setdefault(topic, type_name)
        DDS_STATE["topic_types"] = live_topics


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

        # Kick off discovery thread (reads DCPS built-in topics every second).
        DDS_STATE["discovery_running"] = True
        t = threading.Thread(target=_discovery_loop, name="dds_peer-discovery", daemon=True)
        DDS_STATE["discovery_thread"] = t
        t.start()

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
                    f"  Discovery: running (polls every {DISCOVERY_POLL_INTERVAL:.0f}s)"
                )
            }
        ],
    }


def _stop() -> Dict[str, Any]:
    with _STATE_LOCK:
        if not DDS_STATE["running"]:
            return {"status": "success", "content": [{"text": "🦆 dds_peer not running"}]}

        iid = DDS_STATE.get("instance_id") or "unknown"
        # Signal the discovery loop to exit; it will notice within one poll.
        DDS_STATE["discovery_running"] = False
        t = DDS_STATE.get("discovery_thread")

    # Join outside the lock so the loop can finish cleanly.
    if t is not None:
        t.join(timeout=2.0)

    with _STATE_LOCK:
        # Drop references — CycloneDDS Python cleans up via __del__.
        DDS_STATE["builtin_readers"] = {}
        DDS_STATE["discovery_thread"] = None
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
            f"  Participants: {len(DDS_STATE['participants'])}",
            f"  Publishers  : {len(DDS_STATE['publications'])}",
            f"  Subscribers : {len(DDS_STATE['subscriptions'])}",
            f"  Topics seen : {len(DDS_STATE['topic_types'])}",
        ]
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def _list_participants() -> Dict[str, Any]:
    with _STATE_LOCK:
        if not DDS_STATE["running"]:
            return {"status": "error", "content": [{"text": "🦆 dds_peer: start() first"}]}
        items = sorted(DDS_STATE["participants"].items(), key=lambda kv: kv[1]["first_seen"])
    if not items:
        return {"status": "success", "content": [{"text": "🦆 no DDS participants discovered yet"}]}
    now = time.time()
    lines = [f"🦆 DDS participants ({len(items)}):"]
    for guid, info in items:
        lines.append(f"  {guid}  age={now - info['first_seen']:.0f}s  last={now - info['last_seen']:.1f}s")
    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def _list_topics() -> Dict[str, Any]:
    with _STATE_LOCK:
        if not DDS_STATE["running"]:
            return {"status": "error", "content": [{"text": "🦆 dds_peer: start() first"}]}
        topics = dict(DDS_STATE["topic_types"])
    if not topics:
        return {"status": "success", "content": [{"text": "🦆 no DDS topics discovered yet"}]}
    lines = [f"🦆 DDS topics ({len(topics)}):"]
    for topic in sorted(topics):
        ros_name = _strip_ros_prefix(topic)
        tag = f"  ({_classify_topic(topic)})"
        if ros_name != topic:
            lines.append(f"  {topic}  ↪ {ros_name}  [{topics[topic]}]{tag}")
        else:
            lines.append(f"  {topic}  [{topics[topic]}]{tag}")
    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def _list_endpoints(*, kind: str) -> Dict[str, Any]:
    bucket = "publications" if kind == "publication" else "subscriptions"
    with _STATE_LOCK:
        if not DDS_STATE["running"]:
            return {"status": "error", "content": [{"text": "🦆 dds_peer: start() first"}]}
        items = list(DDS_STATE[bucket].items())
    if not items:
        return {"status": "success", "content": [{"text": f"🦆 no DDS {kind}s discovered yet"}]}
    lines = [f"🦆 DDS {kind}s ({len(items)}):"]
    for (topic, type_name), info in sorted(items, key=lambda kv: kv[0][0]):
        part = info.get("participant") or "?"
        lines.append(f"  {topic}  [{type_name}]  via {part}")
    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


# ── Tool entry point ────────────────────────────────────────────────
@tool
def dds_peer(
    action: str,
    domain_id: int = 0,
    agent: Any = None,
) -> Dict[str, Any]:
    """DDS peer: native ROS2 fleet interop over CycloneDDS.

    Actions:
        start               — create a DomainParticipant on the given domain
        stop                — tear down the participant
        status              — current liveness + discovery counters
        list_participants   — every DDS participant (ROS2 node) on the LAN
        list_topics         — every DDS topic we've seen (ROS2 topics too)
        list_publications   — all publishers, indexed by (topic, type)
        list_subscriptions  — all subscribers, indexed by (topic, type)

    Not yet implemented in this commit (coming up in use_ros):
        subscribe / publish / call / tail

    Args:
        action: one of the actions listed above
        domain_id: DDS domain id (default 0; honors ROS_DOMAIN_ID env)
        agent: optional reference to the parent DevDuck agent

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
    if a in ("list_participants", "participants"):
        return _list_participants()
    if a in ("list_topics", "topics"):
        return _list_topics()
    if a in ("list_publications", "publications", "list_publishers"):
        return _list_endpoints(kind="publication")
    if a in ("list_subscriptions", "subscriptions", "list_subscribers"):
        return _list_endpoints(kind="subscription")
    return {
        "status": "error",
        "content": [
            {
                "text": (
                    f"🦆 dds_peer: unknown action '{action}'. "
                    "Use: start | stop | status | list_participants | list_topics | "
                    "list_publications | list_subscriptions"
                )
            }
        ],
    }
