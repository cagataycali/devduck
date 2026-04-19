"""CycloneDDS peer tool for DevDuck — ROS2-native robot interop.

This module exposes a minimal CycloneDDS peer that lets DevDuck agents
participate in a DDS domain as a first-class citizen. The killer
feature: ROS2 uses DDS under the hood, so a DevDuck instance on the
**same DOMAIN_ID** with CycloneDDS can see, publish, and subscribe to
ROS2 topics natively — no ``rclpy`` required.

Key Ideas
---------
* A single ``DomainParticipant`` is created per DevDuck instance.
* Two ``BuiltinDataReader`` instances on the ``DCPSParticipant`` and
  ``DCPSTopic`` builtin topics provide passive auto-discovery of every
  participant and every advertised topic on the domain.
* A background thread refreshes discovery every few seconds.
* User code can then publish/subscribe to any topic; for simple string
  payloads we dynamically register a ``std_msgs/String``-compatible
  IDL type so ROS2 nodes can consume it.

Environment
-----------
* ``DEVDUCK_DDS_DOMAIN`` — Domain id (default ``0``, same as ROS2 default)
* ``DEVDUCK_ENABLE_DDS`` — Set to ``true`` to auto-start at boot.

ROS2 Interop
------------
To make ROS2 itself use CycloneDDS (so it's on the same wire protocol)::

    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export ROS_DOMAIN_ID=0  # must match DEVDUCK_DDS_DOMAIN

After that, the DevDuck ``dds_peer`` tool can:
- ``list_participants`` — see every ROS2 node running on the domain
- ``list_topics``       — see every ROS2 topic
- ``publish`` / ``subscribe`` — inject/read messages directly
"""

import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from strands import tool

logger = logging.getLogger("devduck.dds_peer")


# ---------------------------------------------------------------------------
# Global state (mirrors the zenoh_peer pattern so __init__.py can introspect)
# ---------------------------------------------------------------------------

DDS_STATE: Dict[str, Any] = {
    "participant": None,
    "readers": {},               # topic_name -> DataReader
    "writers": {},               # topic_name -> DataWriter
    "topics": {},                # topic_name -> Topic object (to keep refs alive)
    "types": {},                 # topic_name -> IdlStruct class used
    "discovered_topics": {},     # topic_name -> {type_name, last_seen}
    "discovered_participants": {},  # key_str -> {hostname, last_seen, qos}
    "received": {},              # topic_name -> list[{"payload": str, "ts": float}]
    "lock": threading.RLock(),
    "running": False,
    "domain_id": 0,
    "instance_id": None,
    "discovery_thread": None,
    "discovery_stop": None,
    "builtin_participant_reader": None,
    "builtin_topic_reader": None,
    "started_at": None,
}

# Max per-topic samples retained in memory for inspection
_MAX_RETAINED_SAMPLES = 50


def _get_instance_id() -> str:
    hostname = socket.gethostname().split(".")[0]
    pid = os.getpid()
    return f"dds-{hostname}-{pid}"


# ---------------------------------------------------------------------------
# Discovery — drains the two DCPS builtin topics
# ---------------------------------------------------------------------------

def _discovery_loop(stop_event: threading.Event, interval: float = 5.0) -> None:
    """Background thread that drains builtin readers and updates state."""
    logger.debug("dds_peer discovery loop started")
    while not stop_event.is_set():
        try:
            _refresh_discovery_once()
        except Exception as exc:  # noqa: BLE001
            logger.debug("dds_peer discovery tick failed: %s", exc)
        stop_event.wait(interval)
    logger.debug("dds_peer discovery loop stopped")


def _refresh_discovery_once() -> Dict[str, Any]:
    """One-shot pass: drain builtin readers, update DDS_STATE."""
    rp = DDS_STATE.get("builtin_participant_reader")
    rt = DDS_STATE.get("builtin_topic_reader")
    now = time.time()
    found_participants = 0
    found_topics = 0

    if rp is not None:
        try:
            for sample in rp.take(N=50):
                key = str(getattr(sample, "key", "unknown"))
                with DDS_STATE["lock"]:
                    entry = DDS_STATE["discovered_participants"].setdefault(
                        key, {"first_seen": now, "hostname": "unknown"}
                    )
                    entry["last_seen"] = now
                    qos = getattr(sample, "qos", None)
                    if qos is not None:
                        entry["qos"] = repr(qos)[:200]
                found_participants += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("participant reader take failed: %s", exc)

    if rt is not None:
        try:
            for sample in rt.take(N=100):
                topic_name = getattr(sample, "topic_name", None) or "?"
                type_name = getattr(sample, "type_name", None) or "?"
                with DDS_STATE["lock"]:
                    entry = DDS_STATE["discovered_topics"].setdefault(
                        topic_name, {"first_seen": now}
                    )
                    entry["type_name"] = type_name
                    entry["last_seen"] = now
                found_topics += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("topic reader take failed: %s", exc)

    return {"participants": found_participants, "topics": found_topics}


# ---------------------------------------------------------------------------
# Default message type — std_msgs/String compatible
#
# ROS2's ``std_msgs/msg/String`` has a single ``string data`` field; the
# DDS-encoded wire format is identical when CycloneDDS is the RMW.
# We declare it at module scope (not inside a function) so ``@dataclass``
# can look up the owning module during type resolution.
# ---------------------------------------------------------------------------

try:
    from cyclonedds.idl import IdlStruct as _IdlStruct

    @dataclass
    class DevDuckString(_IdlStruct, typename="std_msgs::msg::dds_::String_"):
        """Wire-compatible with ``std_msgs/msg/String`` when RMW is CycloneDDS."""

        data: str = ""

except ImportError:  # cyclonedds missing — tool will error cleanly at start()
    DevDuckString = None  # type: ignore[assignment]


def _default_string_type():
    if DevDuckString is None:
        raise RuntimeError("cyclonedds is not installed; cannot build default type")
    return DevDuckString


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discovery_loop(stop_event: threading.Event, interval: float = 5.0) -> None:
    """Background thread that drains builtin readers and updates state."""
    logger.debug("dds_peer discovery loop started")
    while not stop_event.is_set():
        try:
            _refresh_discovery_once()
        except Exception as exc:  # noqa: BLE001 — discovery should never crash caller
            logger.debug("dds_peer discovery tick failed: %s", exc)
        stop_event.wait(interval)
    logger.debug("dds_peer discovery loop stopped")


def _refresh_discovery_once() -> Dict[str, Any]:
    """One-shot pass: drain builtin readers, update DDS_STATE."""
    rp = DDS_STATE.get("builtin_participant_reader")
    rt = DDS_STATE.get("builtin_topic_reader")
    now = time.time()
    found_participants = 0
    found_topics = 0

    if rp is not None:
        try:
            for sample in rp.take(N=50):
                key = str(getattr(sample, "key", "unknown"))
                with DDS_STATE["lock"]:
                    entry = DDS_STATE["discovered_participants"].setdefault(
                        key,
                        {"first_seen": now, "hostname": "unknown"},
                    )
                    entry["last_seen"] = now
                    # Cyclone exposes QoS policies on the sample; try to surface
                    # hostname-ish info if present.
                    qos = getattr(sample, "qos", None)
                    if qos is not None:
                        entry["qos"] = repr(qos)[:200]
                found_participants += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("participant reader take failed: %s", exc)

    if rt is not None:
        try:
            for sample in rt.take(N=100):
                topic_name = getattr(sample, "topic_name", None) or "?"
                type_name = getattr(sample, "type_name", None) or "?"
                with DDS_STATE["lock"]:
                    entry = DDS_STATE["discovered_topics"].setdefault(
                        topic_name, {"first_seen": now}
                    )
                    entry["type_name"] = type_name
                    entry["last_seen"] = now
                found_topics += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("topic reader take failed: %s", exc)

    return {"participants": found_participants, "topics": found_topics}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def start_dds(domain_id: Optional[int] = None, agent: Any = None) -> Dict[str, Any]:
    """Create a DomainParticipant and begin background discovery."""
    with DDS_STATE["lock"]:
        if DDS_STATE["running"]:
            return {
                "status": "success",
                "content": [
                    {"text": f"DDS peer already running on domain {DDS_STATE['domain_id']}"},
                    {"text": f"Instance ID: {DDS_STATE['instance_id']}"},
                ],
            }

        try:
            from cyclonedds.builtin import (
                BuiltinDataReader,
                BuiltinTopicDcpsParticipant,
                BuiltinTopicDcpsTopic,
            )
            from cyclonedds.domain import DomainParticipant
        except ImportError as exc:
            return {
                "status": "error",
                "content": [
                    {"text": f"cyclonedds not installed: {exc}"},
                    {"text": "Install with: pip install cyclonedds"},
                ],
            }

        if domain_id is None:
            domain_id = int(os.getenv("DEVDUCK_DDS_DOMAIN", "0"))

        try:
            participant = DomainParticipant(domain_id)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "content": [{"text": f"Failed to create DomainParticipant: {exc}"}],
            }

        try:
            r_part = BuiltinDataReader(participant, BuiltinTopicDcpsParticipant)
            r_topic = BuiltinDataReader(participant, BuiltinTopicDcpsTopic)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "content": [{"text": f"Failed to attach builtin readers: {exc}"}],
            }

        instance_id = _get_instance_id()
        stop_event = threading.Event()
        thread = threading.Thread(
            target=_discovery_loop,
            args=(stop_event,),
            name="devduck-dds-discovery",
            daemon=True,
        )

        DDS_STATE.update(
            {
                "participant": participant,
                "builtin_participant_reader": r_part,
                "builtin_topic_reader": r_topic,
                "running": True,
                "domain_id": domain_id,
                "instance_id": instance_id,
                "discovery_thread": thread,
                "discovery_stop": stop_event,
                "started_at": time.time(),
            }
        )
        thread.start()

    logger.info("dds_peer started on domain %d as %s", domain_id, instance_id)
    # Give the reactor a moment to observe the first round of discovery.
    time.sleep(0.2)
    _refresh_discovery_once()

    return {
        "status": "success",
        "content": [
            {"text": f"🦆 DDS peer started on domain {domain_id}"},
            {"text": f"Instance ID: {instance_id}"},
            {
                "text": "Tip: set RMW_IMPLEMENTATION=rmw_cyclonedds_cpp on ROS2 nodes "
                "sharing this domain to enable bidirectional interop."
            },
        ],
    }


def stop_dds() -> Dict[str, Any]:
    with DDS_STATE["lock"]:
        if not DDS_STATE["running"]:
            return {"status": "success", "content": [{"text": "DDS peer not running"}]}

        stop_event = DDS_STATE.get("discovery_stop")
        if stop_event:
            stop_event.set()
        thread = DDS_STATE.get("discovery_thread")
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

        # Drop references in the right order so Cyclone cleans up cleanly.
        DDS_STATE["readers"].clear()
        DDS_STATE["writers"].clear()
        DDS_STATE["topics"].clear()
        DDS_STATE["types"].clear()
        DDS_STATE["builtin_participant_reader"] = None
        DDS_STATE["builtin_topic_reader"] = None
        DDS_STATE["participant"] = None
        DDS_STATE["running"] = False
        DDS_STATE["discovery_thread"] = None
        DDS_STATE["discovery_stop"] = None

    logger.info("dds_peer stopped")
    return {"status": "success", "content": [{"text": "🦆 DDS peer stopped"}]}


# ---------------------------------------------------------------------------
# Pub / Sub
# ---------------------------------------------------------------------------

def _get_or_create_topic(topic_name: str, type_cls):
    from cyclonedds.topic import Topic

    key = f"{topic_name}::{type_cls.__name__}"
    existing = DDS_STATE["topics"].get(key)
    if existing is not None:
        return existing
    topic = Topic(DDS_STATE["participant"], topic_name, type_cls)
    DDS_STATE["topics"][key] = topic
    DDS_STATE["types"][topic_name] = type_cls
    return topic


def publish_message(topic_name: str, message: str) -> Dict[str, Any]:
    from cyclonedds.pub import DataWriter, Publisher

    if not DDS_STATE["running"]:
        return {"status": "error", "content": [{"text": "DDS peer not running; call start first"}]}

    with DDS_STATE["lock"]:
        type_cls = DDS_STATE["types"].get(topic_name) or _default_string_type()
        topic = _get_or_create_topic(topic_name, type_cls)
        writer = DDS_STATE["writers"].get(topic_name)
        if writer is None:
            publisher = Publisher(DDS_STATE["participant"])
            writer = DataWriter(publisher, topic)
            DDS_STATE["writers"][topic_name] = writer

    try:
        writer.write(type_cls(data=message))
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "content": [{"text": f"publish failed: {exc}"}]}

    return {
        "status": "success",
        "content": [{"text": f"📡 Published to '{topic_name}': {message[:120]}"}],
    }


def subscribe_topic(topic_name: str, wait_time: float = 1.0) -> Dict[str, Any]:
    """Create a reader for ``topic_name`` and drain any already-queued samples."""
    from cyclonedds.sub import DataReader, Subscriber

    if not DDS_STATE["running"]:
        return {"status": "error", "content": [{"text": "DDS peer not running; call start first"}]}

    with DDS_STATE["lock"]:
        type_cls = DDS_STATE["types"].get(topic_name) or _default_string_type()
        topic = _get_or_create_topic(topic_name, type_cls)
        reader = DDS_STATE["readers"].get(topic_name)
        if reader is None:
            subscriber = Subscriber(DDS_STATE["participant"])
            reader = DataReader(subscriber, topic)
            DDS_STATE["readers"][topic_name] = reader

    if wait_time > 0:
        time.sleep(wait_time)

    collected: List[Dict[str, Any]] = []
    try:
        for sample in reader.take(N=_MAX_RETAINED_SAMPLES):
            payload = getattr(sample, "data", None)
            if payload is None:
                payload = str(sample)
            collected.append({"payload": payload, "ts": time.time()})
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "content": [{"text": f"take failed: {exc}"}]}

    with DDS_STATE["lock"]:
        buf = DDS_STATE["received"].setdefault(topic_name, [])
        buf.extend(collected)
        # Trim so the buffer stays bounded.
        if len(buf) > _MAX_RETAINED_SAMPLES:
            del buf[:-_MAX_RETAINED_SAMPLES]

    return {
        "status": "success",
        "content": [
            {"text": f"📥 '{topic_name}': {len(collected)} new sample(s) this call"},
            {"text": f"Latest: {collected[-1]['payload'] if collected else 'none'}"},
        ],
    }


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def get_status() -> Dict[str, Any]:
    with DDS_STATE["lock"]:
        running = DDS_STATE["running"]
        if not running:
            return {"status": "success", "content": [{"text": "DDS peer: stopped"}]}
        uptime = time.time() - (DDS_STATE["started_at"] or time.time())
        return {
            "status": "success",
            "content": [
                {"text": f"🦆 DDS peer: running"},
                {"text": f"Domain: {DDS_STATE['domain_id']}"},
                {"text": f"Instance ID: {DDS_STATE['instance_id']}"},
                {"text": f"Uptime: {uptime:.1f}s"},
                {"text": f"Discovered participants: {len(DDS_STATE['discovered_participants'])}"},
                {"text": f"Discovered topics: {len(DDS_STATE['discovered_topics'])}"},
                {"text": f"Local readers: {len(DDS_STATE['readers'])}"},
                {"text": f"Local writers: {len(DDS_STATE['writers'])}"},
            ],
        }


def list_participants() -> Dict[str, Any]:
    _refresh_discovery_once()
    with DDS_STATE["lock"]:
        parts = dict(DDS_STATE["discovered_participants"])
    lines = [f"🛰  {len(parts)} participant(s) on domain {DDS_STATE['domain_id']}:"]
    now = time.time()
    for key, info in parts.items():
        age = now - info.get("last_seen", now)
        lines.append(f"  • {key}  (last seen {age:.1f}s ago)")
    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def list_topics() -> Dict[str, Any]:
    _refresh_discovery_once()
    with DDS_STATE["lock"]:
        topics = dict(DDS_STATE["discovered_topics"])
    lines = [f"📋 {len(topics)} topic(s) on domain {DDS_STATE['domain_id']}:"]
    now = time.time()
    for name, info in sorted(topics.items()):
        age = now - info.get("last_seen", now)
        lines.append(f"  • {name}  [{info.get('type_name', '?')}]  ({age:.1f}s ago)")
    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def send_to_peer(peer_id: str, message: str) -> Dict[str, Any]:
    """Publish a message on a per-peer topic convention.

    Convention: ``devduck/cmd/{peer_id}``. The receiving DevDuck is
    expected to have subscribed to the same name.
    """
    topic_name = f"devduck/cmd/{peer_id}"
    return publish_message(topic_name, message)


# ---------------------------------------------------------------------------
# Strands @tool entrypoint
# ---------------------------------------------------------------------------

@tool
def dds_peer(
    action: str,
    topic: str = "",
    message: str = "",
    peer_id: str = "",
    domain_id: Optional[int] = None,
    wait_time: float = 1.0,
    agent: Any = None,
) -> Dict[str, Any]:
    """CycloneDDS peer — ROS2-native pub/sub and participant discovery.

    DDS is the messaging backbone of ROS2. When ROS2 nodes use
    ``rmw_cyclonedds_cpp``, a DevDuck agent running this tool on the
    **same domain_id** appears to ROS2 as just another DDS participant
    — it can see every node, every topic, and pub/sub to them without
    rclpy.

    Actions
    -------
    - ``start``              — create participant + start discovery
    - ``stop``               — tear everything down
    - ``status``             — summary of discovered peers/topics
    - ``list_participants``  — participants on the domain
    - ``list_topics``        — topics on the domain
    - ``publish``            — publish ``message`` to ``topic``
    - ``subscribe``          — take pending samples from ``topic``
    - ``discover``           — force a discovery refresh
    - ``send_to_peer``       — publish to ``devduck/cmd/{peer_id}``

    Environment
    -----------
    - ``DEVDUCK_DDS_DOMAIN`` (default ``0``) — DDS domain id.

    Args:
        action: one of the actions above
        topic: DDS topic name (required for publish/subscribe)
        message: payload string (for publish/send_to_peer)
        peer_id: target peer id (for send_to_peer)
        domain_id: override the environment default at start time
        wait_time: seconds to wait before draining samples on subscribe
        agent: DevDuck agent instance (injected automatically)

    Returns:
        Standard DevDuck tool response dict.
    """
    if action == "start":
        return start_dds(domain_id=domain_id, agent=agent)
    if action == "stop":
        return stop_dds()
    if action == "status":
        return get_status()
    if action == "list_participants":
        return list_participants()
    if action == "list_topics":
        return list_topics()
    if action == "discover":
        stats = _refresh_discovery_once()
        return {
            "status": "success",
            "content": [
                {"text": f"discovery tick: {stats['participants']} participants, {stats['topics']} topics seen"}
            ],
        }
    if action == "publish":
        if not topic:
            return {"status": "error", "content": [{"text": "topic is required for publish"}]}
        return publish_message(topic, message)
    if action == "subscribe":
        if not topic:
            return {"status": "error", "content": [{"text": "topic is required for subscribe"}]}
        return subscribe_topic(topic, wait_time=wait_time)
    if action == "send_to_peer":
        if not peer_id or not message:
            return {"status": "error", "content": [{"text": "peer_id and message are required"}]}
        return send_to_peer(peer_id, message)

    return {
        "status": "error",
        "content": [
            {"text": f"Unknown action: {action}"},
            {
                "text": "Valid actions: start, stop, status, list_participants, "
                "list_topics, publish, subscribe, discover, send_to_peer"
            },
        ],
    }
