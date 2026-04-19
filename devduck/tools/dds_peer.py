"""CycloneDDS peer tool for DevDuck — skeleton (start/stop/status).

This module exposes a minimal CycloneDDS peer that lets DevDuck agents
participate in a DDS domain as a first-class citizen. The killer
feature: ROS2 uses DDS under the hood, so a DevDuck instance on the
**same DOMAIN_ID** with CycloneDDS can see, publish, and subscribe to
ROS2 topics natively — no ``rclpy`` required.

Environment
-----------
* ``DEVDUCK_DDS_DOMAIN`` — Domain id (default ``0``, same as ROS2 default)
* ``DEVDUCK_ENABLE_DDS`` — Set to ``true`` to auto-start at boot.
"""

import logging
import os
import socket
import threading
import time
from typing import Any, Dict, Optional

from strands import tool

logger = logging.getLogger("devduck.dds_peer")


# ---------------------------------------------------------------------------
# Global state (mirrors the zenoh_peer pattern so __init__.py can introspect)
# ---------------------------------------------------------------------------

DDS_STATE: Dict[str, Any] = {
    "participant": None,
    "readers": {},
    "writers": {},
    "topics": {},
    "types": {},
    "discovered_topics": {},
    "discovered_participants": {},
    "received": {},
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


def _get_instance_id() -> str:
    hostname = socket.gethostname().split(".")[0]
    pid = os.getpid()
    return f"dds-{hostname}-{pid}"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def start_dds(domain_id: Optional[int] = None, agent: Any = None) -> Dict[str, Any]:
    """Create a DomainParticipant ready for discovery and pub/sub."""
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

        instance_id = _get_instance_id()
        DDS_STATE.update(
            {
                "participant": participant,
                "running": True,
                "domain_id": domain_id,
                "instance_id": instance_id,
                "started_at": time.time(),
            }
        )

    logger.info("dds_peer started on domain %d as %s", domain_id, instance_id)
    return {
        "status": "success",
        "content": [
            {"text": f"🦆 DDS peer started on domain {domain_id}"},
            {"text": f"Instance ID: {instance_id}"},
        ],
    }


def stop_dds() -> Dict[str, Any]:
    with DDS_STATE["lock"]:
        if not DDS_STATE["running"]:
            return {"status": "success", "content": [{"text": "DDS peer not running"}]}
        DDS_STATE["readers"].clear()
        DDS_STATE["writers"].clear()
        DDS_STATE["topics"].clear()
        DDS_STATE["types"].clear()
        DDS_STATE["participant"] = None
        DDS_STATE["running"] = False
    logger.info("dds_peer stopped")
    return {"status": "success", "content": [{"text": "🦆 DDS peer stopped"}]}


def get_status() -> Dict[str, Any]:
    with DDS_STATE["lock"]:
        if not DDS_STATE["running"]:
            return {"status": "success", "content": [{"text": "DDS peer: stopped"}]}
        uptime = time.time() - (DDS_STATE["started_at"] or time.time())
        return {
            "status": "success",
            "content": [
                {"text": f"🦆 DDS peer: running"},
                {"text": f"Domain: {DDS_STATE['domain_id']}"},
                {"text": f"Instance ID: {DDS_STATE['instance_id']}"},
                {"text": f"Uptime: {uptime:.1f}s"},
            ],
        }


# ---------------------------------------------------------------------------
# Strands @tool entrypoint (skeleton — discovery/pub/sub land in later commits)
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
    """CycloneDDS peer — ROS2-native participant (skeleton stage)."""
    if action == "start":
        return start_dds(domain_id=domain_id, agent=agent)
    if action == "stop":
        return stop_dds()
    if action == "status":
        return get_status()
    return {
        "status": "error",
        "content": [{"text": f"Unknown action: {action}"}],
    }
