"""Agent-facing ROS2 tool — opinionated wrapper over dds_peer + _ros_msgs.

Where `dds_peer` is the low-level DDS transport (participants, discovery,
raw pub/sub), `use_ros` is the high-level tool the agent actually reaches
for. It speaks ROS2 vocabulary (topics, messages, nodes, services) and
hides DDS typename mangling, CDR details, QoS knobs.

Actions shipped in this commit:
    list_nodes      — every DDS participant on the domain (ROS2 nodes)
    list_topics     — every ROS2 topic (filtered to rt/*, mapped to /X)
    echo            — read a single sample from a topic (one-shot)
    pub             — publish one sample to a topic
    types           — list the bundled IDL types we know how to decode

Later commits add:
    tail            — background subscription that streams into event_bus
    call            — ROS2 service request / reply
    bag_record      — short rosbag-style capture to disk
    vision          — auto-wire camera topics to use_aws / use_google

Design:
    - Always uses the dds_peer's DomainParticipant (auto-starts if idle).
    - Topic-type resolution order:
         1. Explicit `type` arg passed by the caller
         2. Live DDS discovery (topic -> type observed on the wire)
         3. Error with helpful message
    - Message-class resolution: _ros_msgs.ros_type_to_idl(type_name).
      Unknown types currently error out (opaque byte fallback lands with
      dynamic type support in a later commit).
    - JSON friendly: echo returns dict form of the message, pub accepts
      dict form. Agents can pipe strings through without knowing CDR.
"""

import logging
import threading
import time
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Optional

from strands import tool

from devduck.tools import _ros_msgs
from devduck.tools.dds_peer import (
    DDS_STATE,
    _STATE_LOCK,
    _classify_topic,
    _start as dds_start,
    _strip_ros_prefix,
)

logger = logging.getLogger(__name__)

# Cache of topic -> (DataReader, DataWriter) so repeated echo/pub on the
# same topic don't churn the DDS machinery. Keyed by (topic_name, idl_cls).
_ENDPOINTS: Dict[tuple, Dict[str, Any]] = {}

# Active tail loops: key = ros_topic_name (e.g. "/cmd_vel"),
#                    value = {"thread": Thread, "running": bool,
#                             "dds_topic": str, "type": str, "count": int,
#                             "last": float, "rate_hz": float}
_TAILS: Dict[str, Dict[str, Any]] = {}
_TAILS_LOCK = threading.RLock()


# ── Helpers ─────────────────────────────────────────────────────────
def _ensure_started() -> Optional[Dict[str, Any]]:
    """Make sure dds_peer is running. Return an error response if not."""
    with _STATE_LOCK:
        if DDS_STATE["running"]:
            return None
    res = dds_start(domain_id=0)
    if res.get("status") != "success":
        return res
    return None


def _to_ros_topic(name: str) -> str:
    """Accept '/scan' or 'rt/scan' or 'scan' and produce the DDS name."""
    if name.startswith("rt/") or name.startswith("rq/") or name.startswith("rr/"):
        return name
    if name.startswith("/"):
        return "rt" + name
    return "rt/" + name


def _from_ros_topic(dds_name: str) -> str:
    """Display helper: rt/scan -> /scan."""
    return _strip_ros_prefix(dds_name)


def _resolve_type(topic_name_dds: str, explicit: Optional[str]) -> Optional[str]:
    """Pick the type for this topic. Explicit wins, else live discovery."""
    if explicit:
        return explicit
    with _STATE_LOCK:
        return DDS_STATE["topic_types"].get(topic_name_dds)


def _message_to_dict(msg: Any) -> Any:
    """Recursive IdlStruct -> plain dict converter (JSON-safe)."""
    if is_dataclass(msg):
        return {f.name: _message_to_dict(getattr(msg, f.name)) for f in fields(msg)}
    if isinstance(msg, (list, tuple)):
        return [_message_to_dict(v) for v in msg]
    if isinstance(msg, (bytes, bytearray)):
        return list(msg)  # uint8 arrays (Image.data etc.)
    return msg


def _dict_to_message(cls: type, data: Dict[str, Any]) -> Any:
    """dict -> IdlStruct instance (nested)."""
    if not is_dataclass(cls):
        # Leaf (plain Python type)
        return data
    if not isinstance(data, dict):
        # Caller passed a non-dict for a struct field — fall through and let
        # dataclass construction raise a helpful TypeError.
        return cls(data)  # type: ignore[arg-type]

    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        child = data[f.name]
        sub_cls = f.type
        # Resolve nested dataclass targets by walking into our registry.
        if isinstance(sub_cls, type) and is_dataclass(sub_cls):
            kwargs[f.name] = _dict_to_message(sub_cls, child)
        else:
            kwargs[f.name] = child
    return cls(**kwargs)


def _get_or_build_endpoints(topic_name_dds: str, idl_cls: type) -> Dict[str, Any]:
    """Create (once, cached) a reader + writer pair for this topic/type."""
    key = (topic_name_dds, idl_cls)
    cached = _ENDPOINTS.get(key)
    if cached is not None:
        return cached

    from cyclonedds.pub import Publisher, DataWriter
    from cyclonedds.sub import Subscriber, DataReader
    from cyclonedds.topic import Topic

    with _STATE_LOCK:
        participant = DDS_STATE["participant"]
    if participant is None:
        raise RuntimeError("dds_peer participant unavailable")

    topic = Topic(participant, topic_name_dds, idl_cls)
    publisher = Publisher(participant)
    subscriber = Subscriber(participant)
    writer = DataWriter(publisher, topic)
    reader = DataReader(subscriber, topic)

    entry = {
        "topic": topic,
        "publisher": publisher,
        "subscriber": subscriber,
        "writer": writer,
        "reader": reader,
    }
    _ENDPOINTS[key] = entry
    return entry


# ── Actions ─────────────────────────────────────────────────────────
def _list_nodes() -> Dict[str, Any]:
    err = _ensure_started()
    if err is not None:
        return err
    with _STATE_LOCK:
        items = sorted(DDS_STATE["participants"].items(), key=lambda kv: kv[1]["first_seen"])
        my_guid = str(DDS_STATE["participant"].guid) if DDS_STATE["participant"] else None
    if not items:
        return {"status": "success", "content": [{"text": "🦆 use_ros: no ROS2/DDS nodes found yet"}]}
    now = time.time()
    lines = [f"🦆 ROS2/DDS nodes ({len(items)}):"]
    for guid, info in items:
        tag = "  (me)" if guid == my_guid else ""
        lines.append(f"  {guid}  up {now - info['first_seen']:.0f}s{tag}")
    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def _list_topics(include_raw: bool = False) -> Dict[str, Any]:
    err = _ensure_started()
    if err is not None:
        return err
    with _STATE_LOCK:
        types_map = dict(DDS_STATE["topic_types"])

    rows = []
    for dds_name, type_name in sorted(types_map.items()):
        cls = _classify_topic(dds_name)
        if cls != "topic" and not include_raw:
            continue
        rows.append((_from_ros_topic(dds_name), type_name, cls))
    if not rows:
        return {"status": "success", "content": [{"text": "🦆 use_ros: no ROS2 topics discovered yet"}]}
    lines = [f"🦆 ROS2 topics ({len(rows)}):"]
    for ros_name, type_name, cls in rows:
        known = "known" if _ros_msgs.ros_type_to_idl(type_name) else "unknown"
        lines.append(f"  {ros_name}  [{type_name}]  ({cls}, {known})")
    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def _types() -> Dict[str, Any]:
    known = _ros_msgs.known_types()
    lines = [f"🦆 use_ros bundled ROS2 types ({len(known)}):"]
    lines.extend(f"  {t}" for t in known)
    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def _echo(topic: str, type_name: Optional[str], timeout: float) -> Dict[str, Any]:
    err = _ensure_started()
    if err is not None:
        return err
    dds_topic = _to_ros_topic(topic)
    resolved = _resolve_type(dds_topic, type_name)
    if not resolved:
        return {
            "status": "error",
            "content": [{"text": f"🦆 use_ros: cannot determine type for '{topic}'. Pass type='pkg/msg/Name' or wait for discovery."}],
        }
    idl_cls = _ros_msgs.ros_type_to_idl(resolved)
    if idl_cls is None:
        return {
            "status": "error",
            "content": [{"text": f"🦆 use_ros: type '{resolved}' not in bundled registry (opaque-byte fallback coming in a later commit)"}],
        }

    endpoints = _get_or_build_endpoints(dds_topic, idl_cls)
    reader = endpoints["reader"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        samples = reader.take(N=1)
        if samples:
            sample = samples[0]
            payload = _message_to_dict(sample)
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🦆 {topic} [{resolved}]\n"
                            f"{_format_preview(payload)}"
                        )
                    }
                ],
            }
        time.sleep(0.05)
    return {
        "status": "success",
        "content": [{"text": f"🦆 use_ros: no sample received on {topic} within {timeout:.1f}s"}],
    }


def _pub(topic: str, type_name: Optional[str], msg: Dict[str, Any]) -> Dict[str, Any]:
    err = _ensure_started()
    if err is not None:
        return err
    dds_topic = _to_ros_topic(topic)
    resolved = _resolve_type(dds_topic, type_name)
    if not resolved:
        return {
            "status": "error",
            "content": [{"text": f"🦆 use_ros: cannot determine type for '{topic}'. Pass type='pkg/msg/Name'."}],
        }
    idl_cls = _ros_msgs.ros_type_to_idl(resolved)
    if idl_cls is None:
        return {
            "status": "error",
            "content": [{"text": f"🦆 use_ros: type '{resolved}' not in bundled registry"}],
        }
    if not isinstance(msg, dict):
        return {
            "status": "error",
            "content": [{"text": "🦆 use_ros: msg must be a JSON-style dict matching the ROS2 message layout"}],
        }

    try:
        instance = _dict_to_message(idl_cls, msg)
    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"🦆 use_ros: failed to build {resolved} from dict: {e}"}],
        }

    endpoints = _get_or_build_endpoints(dds_topic, idl_cls)
    writer = endpoints["writer"]
    try:
        writer.write(instance)
    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"🦆 use_ros: DDS write failed: {e}"}],
        }

    return {
        "status": "success",
        "content": [{"text": f"🦆 use_ros: published to {topic} [{resolved}]"}],
    }


def _format_preview(payload: Any, max_len: int = 1200) -> str:
    """Pretty-print a dict payload, truncating big arrays (images etc.)."""
    import json

    def _truncate(v):
        if isinstance(v, list) and len(v) > 16:
            return v[:8] + [f"...({len(v) - 8} more)"]
        if isinstance(v, dict):
            return {k: _truncate(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_truncate(x) for x in v]
        return v

    compact = _truncate(payload)
    text = json.dumps(compact, indent=2, default=str)
    if len(text) > max_len:
        text = text[:max_len] + "\n... [truncated]"
    return text


def _emit_ros_event(ros_topic: str, type_name: str, payload: Dict[str, Any]) -> None:
    """Push a ROS2 message into the shared event_bus as 'ros.<topic>'."""
    try:
        from devduck.tools.event_bus import bus  # type: ignore
    except Exception:
        return
    try:
        summary = f"{ros_topic} [{type_name}]"
        detail = _format_preview(payload, max_len=400)
    except Exception:
        summary = ros_topic
        detail = ""
    try:
        bus.emit(
            event_type=f"ros.{ros_topic.lstrip('/')}",
            source="use_ros",
            summary=summary,
            detail=detail,
            metadata={"topic": ros_topic, "type": type_name, "payload": payload},
        )
    except Exception:
        pass


def _tail_loop(ros_topic: str, dds_topic: str, idl_cls: type, resolved_type: str,
               max_hz: float, stop_flag_key: str) -> None:
    """Background subscriber that streams samples into event_bus.

    Rate-limits to `max_hz` to keep high-frequency topics from flooding the
    agent's dynamic context. We always keep the latest sample per window.
    """
    endpoints = _get_or_build_endpoints(dds_topic, idl_cls)
    reader = endpoints["reader"]

    min_interval = 1.0 / max_hz if max_hz > 0 else 0.0
    next_emit = 0.0
    logger.info("use_ros tail start: %s [%s] max_hz=%.1f", ros_topic, resolved_type, max_hz)

    while True:
        with _TAILS_LOCK:
            entry = _TAILS.get(stop_flag_key)
            if entry is None or not entry.get("running"):
                break

        samples = reader.take(N=8)
        now = time.time()
        if samples:
            latest = samples[-1]
            with _TAILS_LOCK:
                entry = _TAILS.get(stop_flag_key)
                if entry is not None:
                    entry["count"] += len(samples)
                    # Exponential moving average of receive rate.
                    last = entry.get("last") or now
                    dt = max(now - last, 1e-3)
                    inst = len(samples) / dt
                    entry["rate_hz"] = 0.8 * entry.get("rate_hz", inst) + 0.2 * inst
                    entry["last"] = now
            if now >= next_emit:
                payload = _message_to_dict(latest)
                _emit_ros_event(ros_topic, resolved_type, payload)
                next_emit = now + min_interval
        time.sleep(0.02)

    logger.info("use_ros tail stop: %s", ros_topic)


def _tail_start(topic: str, type_name: Optional[str], max_hz: float) -> Dict[str, Any]:
    err = _ensure_started()
    if err is not None:
        return err
    ros_topic = "/" + topic.lstrip("/") if not topic.startswith("rt/") else _from_ros_topic(topic)
    dds_topic = _to_ros_topic(topic)
    resolved = _resolve_type(dds_topic, type_name)
    if not resolved:
        return {
            "status": "error",
            "content": [{"text": f"🦆 use_ros: cannot determine type for '{topic}'. Pass type='pkg/msg/Name'."}],
        }
    idl_cls = _ros_msgs.ros_type_to_idl(resolved)
    if idl_cls is None:
        return {
            "status": "error",
            "content": [{"text": f"🦆 use_ros: type '{resolved}' not in bundled registry"}],
        }

    with _TAILS_LOCK:
        existing = _TAILS.get(ros_topic)
        if existing and existing.get("running"):
            return {
                "status": "success",
                "content": [{"text": f"🦆 use_ros tail already running on {ros_topic}"}],
            }
        entry = {
            "running": True,
            "dds_topic": dds_topic,
            "type": resolved,
            "max_hz": max_hz,
            "count": 0,
            "rate_hz": 0.0,
            "last": 0.0,
            "started": time.time(),
        }
        t = threading.Thread(
            target=_tail_loop,
            args=(ros_topic, dds_topic, idl_cls, resolved, max_hz, ros_topic),
            name=f"use_ros-tail:{ros_topic}",
            daemon=True,
        )
        entry["thread"] = t
        _TAILS[ros_topic] = entry
        t.start()

    return {
        "status": "success",
        "content": [
            {
                "text": (
                    f"🦆 use_ros tail started on {ros_topic} [{resolved}]\n"
                    f"  Events emitted as 'ros.{ros_topic.lstrip('/')}' on event_bus, capped at {max_hz:.1f} Hz"
                )
            }
        ],
    }


def _tail_stop(topic: str) -> Dict[str, Any]:
    ros_topic = "/" + topic.lstrip("/") if not topic.startswith("rt/") else _from_ros_topic(topic)
    with _TAILS_LOCK:
        entry = _TAILS.get(ros_topic)
        if not entry:
            return {"status": "success", "content": [{"text": f"🦆 use_ros: no tail running on {ros_topic}"}]}
        entry["running"] = False
        t = entry.get("thread")
    if t is not None:
        t.join(timeout=1.0)
    with _TAILS_LOCK:
        _TAILS.pop(ros_topic, None)
    return {"status": "success", "content": [{"text": f"🦆 use_ros tail stopped on {ros_topic}"}]}


def _tail_list() -> Dict[str, Any]:
    with _TAILS_LOCK:
        items = list(_TAILS.items())
    if not items:
        return {"status": "success", "content": [{"text": "🦆 use_ros: no active tails"}]}
    now = time.time()
    lines = [f"🦆 use_ros active tails ({len(items)}):"]
    for ros_topic, info in sorted(items):
        uptime = now - info.get("started", now)
        lines.append(
            f"  {ros_topic}  [{info['type']}]  "
            f"recv={info['count']}  rate≈{info['rate_hz']:.2f} Hz  "
            f"up={uptime:.0f}s  cap={info['max_hz']:.1f} Hz"
        )
    return {"status": "success", "content": [{"text": "\n".join(lines)}]}




# ── ROS2 services: `call` action (rmw_cyclonedds wire format) ───────
#
# ROS2 services transport request/reply over two DDS topics:
#     rq/<service>Request   -> client sends request here
#     rr/<service>Reply     -> server replies here
#
# rmw_cyclonedds_cpp prepends each request/reply struct with a sample
# identity header (16-byte client GUID + 8-byte sequence number) so
# clients can correlate replies to the request that triggered them.
# Every IDL struct in _ros_msgs for a service type MUST declare those
# three header fields first; see AddTwoIntsRequest for the pattern.
#
# For plain DDS clients (us), the strategy is:
#     1. Pick a pseudo-random 128-bit client GUID (once per process).
#     2. For each call, bump a monotonic sequence number.
#     3. Write the request on rq/...Request.
#     4. Read rr/...Reply samples until one arrives whose
#        client_guid_* and sequence_number match ours.
#     5. Return the payload fields (everything after the header).

import os as _os
import struct as _struct
_CLIENT_GUID = _struct.unpack("<Q", _os.urandom(8))[0]
_SERVICE_SEQ_LOCK = threading.Lock()
_SERVICE_SEQ = 0


def _next_service_seq() -> int:
    global _SERVICE_SEQ
    with _SERVICE_SEQ_LOCK:
        _SERVICE_SEQ += 1
        return _SERVICE_SEQ


def _service_topic_names(service: str) -> tuple:
    """'/add_two_ints' -> ('rq/add_two_intsRequest', 'rr/add_two_intsReply')."""
    name = service.lstrip("/")
    return f"rq/{name}Request", f"rr/{name}Reply"


def _resolve_service_types(
    service: str,
    srv_type: Optional[str],
) -> tuple:
    """Resolve request + response IDL classes from a ROS2 service type.

    Accepts:
        explicit: 'example_interfaces/srv/AddTwoInts' — builds request +
                  response type names by convention.
        discovered: looks for rq/... and rr/... entries in DDS_STATE.

    Returns (req_type_name, res_type_name, req_idl_cls, res_idl_cls) or
    a tuple with None entries and an error message in [4].
    """
    rq_topic, rr_topic = _service_topic_names(service)

    # 1) Explicit type wins.
    if srv_type:
        req_name = f"{srv_type}_Request"
        res_name = f"{srv_type}_Response"
    else:
        # 2) Lookup from live discovery.
        with _STATE_LOCK:
            req_name = DDS_STATE["topic_types"].get(rq_topic)
            res_name = DDS_STATE["topic_types"].get(rr_topic)
        if not req_name or not res_name:
            return None, None, None, None, (
                f"cannot determine service type for '{service}'. "
                "Pass srv_type='pkg/srv/Name' or wait for discovery."
            )

    req_cls = _ros_msgs.ros_type_to_idl(req_name)
    res_cls = _ros_msgs.ros_type_to_idl(res_name)
    if req_cls is None or res_cls is None:
        return None, None, None, None, (
            f"service type '{srv_type or 'discovered'}' not in bundled registry "
            f"(missing: {[n for n, c in [(req_name, req_cls), (res_name, res_cls)] if c is None]}). "
            "Add it to devduck/tools/_ros_msgs.py."
        )
    return req_name, res_name, req_cls, res_cls, None


def _call(
    service: str,
    srv_type: Optional[str],
    msg: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    """Invoke a ROS2 service and wait for its reply.

    Args:
        service: ROS2 service name like '/add_two_ints'.
        srv_type: ROS2 service type like 'example_interfaces/srv/AddTwoInts'.
                  Optional when discovery has already observed the service.
        msg: dict of request fields (header fields will be filled in for you).
        timeout: seconds to wait for a matching reply.

    Returns:
        Standard Strands tool result; on success `content[0].text` contains
        a pretty-printed dict of the response payload.
    """
    err = _ensure_started()
    if err is not None:
        return err

    req_name, res_name, req_cls, res_cls, err_msg = _resolve_service_types(service, srv_type)
    if err_msg:
        return {"status": "error", "content": [{"text": f"🦆 use_ros: {err_msg}"}]}

    rq_topic, rr_topic = _service_topic_names(service)

    # Build the request: fill in correlation header + user payload.
    seq = _next_service_seq()
    try:
        request = _dict_to_message(req_cls, {
            "client_guid": _CLIENT_GUID,
            "sequence_number": seq,
            **(msg or {}),
        })
    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"🦆 use_ros: failed to build {req_name} request: {e}"}],
        }

    # Create endpoints on both service topics.
    try:
        req_endpoints = _get_or_build_endpoints(rq_topic, req_cls)
        res_endpoints = _get_or_build_endpoints(rr_topic, res_cls)
    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"🦆 use_ros: failed to bind service endpoints: {e}"}],
        }

    writer = req_endpoints["writer"]
    reader = res_endpoints["reader"]

    # Drain any stale replies so we don't match a previous client's correlation.
    try:
        reader.take(N=64)
    except Exception:
        pass

    # Fire the request.
    try:
        writer.write(request)
    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"🦆 use_ros: DDS write failed on {rq_topic}: {e}"}],
        }

    # Wait for a reply whose header matches our (guid, seq).
    deadline = time.time() + timeout
    poll = 0.02
    while time.time() < deadline:
        try:
            samples = reader.take(N=8)
        except Exception:
            samples = []
        for s in samples or []:
            if (
                getattr(s, "client_guid", None) == _CLIENT_GUID
                and getattr(s, "sequence_number", None) == seq
            ):
                payload = _message_to_dict(s)
                # Strip the correlation header from the visible response.
                for hdr in ("client_guid", "sequence_number"):
                    payload.pop(hdr, None)
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"🦆 {service} [{res_name}]\n"
                                f"{_format_preview(payload)}"
                            )
                        }
                    ],
                }
        time.sleep(poll)

    return {
        "status": "error",
        "content": [
            {
                "text": (
                    f"🦆 use_ros: no matching reply on {rr_topic} within {timeout:.1f}s. "
                    f"Server may be offline, type may be wrong, or the server is not on rmw_cyclonedds_cpp "
                    f"(fastrtps uses a different service wire format)."
                )
            }
        ],
    }


def _list_services() -> Dict[str, Any]:
    """List discovered ROS2 services (joined rq/rr pairs)."""
    err = _ensure_started()
    if err is not None:
        return err
    with _STATE_LOCK:
        types_map = dict(DDS_STATE["topic_types"])

    services = {}
    for topic, tname in types_map.items():
        if topic.startswith("rq/") and topic.endswith("Request"):
            base = topic[3:-len("Request")]
            services.setdefault(base, {})["req"] = tname
        elif topic.startswith("rr/") and topic.endswith("Reply"):
            base = topic[3:-len("Reply")]
            services.setdefault(base, {})["res"] = tname

    if not services:
        return {"status": "success", "content": [{"text": "🦆 use_ros: no ROS2 services discovered yet"}]}

    lines = [f"🦆 ROS2 services ({len(services)}):"]
    for name in sorted(services):
        info = services[name]
        req = info.get("req", "?")
        res = info.get("res", "?")
        # Try to derive a clean pkg/srv/Name for the call hint.
        srv_hint = ""
        if req.endswith("_Request_"):
            # req looks like: pkg::srv::dds_::Name_Request_
            pieces = req.replace("::", "/").split("/")
            if len(pieces) >= 4:
                pkg = pieces[0]
                nm = pieces[-1][:-len("_Request_")]
                srv_hint = f"  → call as '{pkg}/srv/{nm}'"
        lines.append(f"  /{name}  req={req} res={res}{srv_hint}")
    return {"status": "success", "content": [{"text": "\n".join(lines)}]}

# ── Tool entry point ────────────────────────────────────────────────
@tool
def use_ros(
    action: str,
    topic: str = "",
    type: Optional[str] = None,  # noqa: A002 - matches ROS2 nomenclature
    msg: Optional[Dict[str, Any]] = None,
    timeout: float = 2.0,
    include_raw: bool = False,
    max_hz: float = 5.0,
    service: str = "",
    srv_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Talk to any ROS2 robot / node / topic over DDS — no ROS2 install needed.

    Actions:
        list_nodes      every DDS participant / ROS2 node on the domain
        list_topics     every ROS2 topic (use include_raw=True for rq/rr too)
        echo            one-shot read from a topic
        pub             publish one sample to a topic
        tail            start a background subscriber that streams messages
                        into event_bus as 'ros.<topic_name>' events
        untail          stop a running tail
        list_tails      inspect active tails + receive rate
        types           list bundled ROS2 message types we can decode
        call            invoke a ROS2 service and wait for the reply
        list_services   list discovered ROS2 services (rq/rr topic pairs)

    Args:
        action: one of the actions above
        topic: ROS2 topic name ('/cmd_vel', '/scan', or 'rt/scan' — any form)
        type: ROS2 message type ('geometry_msgs/msg/Twist'). Optional when
            live discovery has already observed the topic.
        msg: dict payload for `pub`, structured like the message
            (e.g. {"linear": {"x": 0.2}, "angular": {"z": 0.1}})
        timeout: echo wait time in seconds (default 2.0)
        include_raw: also show service/raw DDS topics in list_topics
        max_hz: tail rate cap in Hz (default 5.0); samples above this rate
            are coalesced to the latest
        service: ROS2 service name for `call` (e.g. '/add_two_ints')
        srv_type: ROS2 service type for `call` (e.g. 'example_interfaces/srv/AddTwoInts')

    Examples:
        use_ros(action="list_nodes")
        use_ros(action="list_topics")
        use_ros(action="echo", topic="/cmd_vel", timeout=3)
        use_ros(action="pub",  topic="/cmd_vel",
                type="geometry_msgs/msg/Twist",
                msg={"linear": {"x": 0.2}})
        use_ros(action="tail", topic="/scan", max_hz=2.0)
        use_ros(action="list_tails")
        use_ros(action="untail", topic="/scan")
        use_ros(action="list_services")
        use_ros(action="call", service="/add_two_ints",
                srv_type="example_interfaces/srv/AddTwoInts",
                msg={"a": 17, "b": 25}, timeout=5.0)

    Returns:
        Standard Strands tool result dict.
    """
    a = (action or "").strip().lower()
    if a in ("list_nodes", "nodes"):
        return _list_nodes()
    if a in ("list_topics", "topics"):
        return _list_topics(include_raw=include_raw)
    if a == "types":
        return _types()
    if a == "echo":
        if not topic:
            return {"status": "error", "content": [{"text": "🦆 use_ros: echo requires a `topic`"}]}
        return _echo(topic, type, timeout)
    if a in ("pub", "publish"):
        if not topic:
            return {"status": "error", "content": [{"text": "🦆 use_ros: pub requires a `topic`"}]}
        return _pub(topic, type, msg or {})
    if a == "tail":
        if not topic:
            return {"status": "error", "content": [{"text": "🦆 use_ros: tail requires a `topic`"}]}
        return _tail_start(topic, type, max_hz)
    if a in ("untail", "stop_tail"):
        if not topic:
            return {"status": "error", "content": [{"text": "🦆 use_ros: untail requires a `topic`"}]}
        return _tail_stop(topic)
    if a in ("list_tails", "tails"):
        return _tail_list()
    if a in ("call", "service_call"):
        if not service:
            return {"status": "error", "content": [{"text": "🦆 use_ros: call requires a `service` (e.g. '/add_two_ints')"}]}
        return _call(service, srv_type, msg or {}, timeout)
    if a in ("list_services", "services"):
        return _list_services()
    return {
        "status": "error",
        "content": [
            {
                "text": (
                    f"🦆 use_ros: unknown action '{action}'. "
                    "Use: list_nodes | list_topics | types | echo | pub | "
                    "tail | untail | list_tails | call | list_services"
                )
            }
        ],
    }
