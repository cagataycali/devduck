"""
🦆 use_mavlink - Universal MAVLink drone control via pymavlink

One tool, every MAVLink-speaking vehicle:
- ArduPilot (Copter, Plane, Rover, Sub)
- PX4
- DJI (via MAVSDK bridge)
- Any drone speaking MAVLink 2.0

Design: mirrors `use_ros` — getattr-driven, future-proof, action-based.

Connection strings (pymavlink format):
    udpin:0.0.0.0:14550     # SITL / companion computer
    udpout:192.168.4.1:14550
    tcp:127.0.0.1:5760      # SITL TCP
    /dev/ttyUSB0,57600      # Serial radio
    /dev/ttyACM0,115200     # USB direct

Examples:
    use_mavlink(action="connect", connection="udpin:0.0.0.0:14550")
    use_mavlink(action="status")
    use_mavlink(action="arm")
    use_mavlink(action="takeoff", altitude=10.0)
    use_mavlink(action="goto", lat=47.397, lon=8.545, alt=20)
    use_mavlink(action="set_mode", mode="GUIDED")
    use_mavlink(action="send", message="SET_POSITION_TARGET_LOCAL_NED", fields={...})
    use_mavlink(action="stream", message="ATTITUDE", rate=10, samples=50)
    use_mavlink(action="disconnect")
"""

from __future__ import annotations

import os
import time
import threading
from typing import Any, Dict, List, Optional

from strands import tool


# ─────────────────────────── Global connection state ──────────────────────────

_STATE: Dict[str, Any] = {
    "conn": None,              # mavutil.mavlink_connection instance
    "connection_str": None,
    "target_system": 1,
    "target_component": 1,
    "heartbeat_thread": None,
    "streams": {},             # {msg_name: {"thread": t, "stop": Event, "samples": [...]}}
    "lock": threading.Lock(),
}


# ─────────────────────────── Helpers ──────────────────────────

def _err(msg: str) -> Dict[str, Any]:
    return {"status": "error", "content": [{"text": f"🦆 use_mavlink: {msg}"}]}


def _ok(msg: str) -> Dict[str, Any]:
    return {"status": "success", "content": [{"text": f"🦆 {msg}"}]}


def _require_conn():
    """Raise if not connected."""
    if _STATE["conn"] is None:
        raise RuntimeError("not connected — call action='connect' first")
    return _STATE["conn"]


def _import_mav():
    try:
        from pymavlink import mavutil
        return mavutil
    except ImportError:
        raise RuntimeError("pymavlink not installed — pip install pymavlink")


def _msg_to_dict(msg) -> Dict[str, Any]:
    """Convert pymavlink message to plain dict."""
    if msg is None:
        return {}
    d = msg.to_dict()
    # Remove mavpackettype noise, keep everything else
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ─────────────────────────── Tool ──────────────────────────

@tool
def use_mavlink(
    action: str,
    connection: Optional[str] = None,
    message: Optional[str] = None,
    fields: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
    altitude: Optional[float] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    alt: Optional[float] = None,
    vx: Optional[float] = None,
    vy: Optional[float] = None,
    vz: Optional[float] = None,
    yaw: Optional[float] = None,
    rate: float = 4.0,
    samples: int = 1,
    timeout: float = 5.0,
    target_system: Optional[int] = None,
    target_component: Optional[int] = None,
) -> Dict[str, Any]:
    """
    🦆 Universal MAVLink drone control.

    Connection management:
        action='connect' connection='udpin:0.0.0.0:14550'
        action='disconnect'
        action='status'

    Flight control (requires connection):
        action='arm'
        action='disarm'
        action='takeoff' altitude=10.0
        action='land'
        action='rtl'                     # Return to launch
        action='set_mode' mode='GUIDED'  # GUIDED, AUTO, LOITER, RTL, STABILIZE…
        action='goto' lat=... lon=... alt=...
        action='velocity' vx=... vy=... vz=... [yaw=...]

    Introspection:
        action='list_messages'           # Recently received messages
        action='get_message' message='ATTITUDE' timeout=5
        action='stream' message='ATTITUDE' rate=10 samples=50

    Raw MAVLink:
        action='send' message='COMMAND_LONG' fields={...}
    """
    try:
        mavutil = _import_mav()
    except RuntimeError as e:
        return _err(str(e))

    # Override target system/component if provided
    if target_system is not None:
        _STATE["target_system"] = target_system
    if target_component is not None:
        _STATE["target_component"] = target_component

    try:
        # ─── Connection management ───
        if action == "connect":
            if not connection:
                return _err("connection string required (e.g., 'udpin:0.0.0.0:14550')")

            with _STATE["lock"]:
                if _STATE["conn"] is not None:
                    _STATE["conn"].close()

                conn = mavutil.mavlink_connection(connection)
                _STATE["conn"] = conn
                _STATE["connection_str"] = connection

                # Wait for heartbeat to confirm link
                hb = conn.wait_heartbeat(timeout=timeout)
                if hb is None:
                    _STATE["conn"] = None
                    return _err(f"no heartbeat from {connection} within {timeout}s")

                _STATE["target_system"] = conn.target_system
                _STATE["target_component"] = conn.target_component

            return _ok(
                f"connected to {connection}\n"
                f"   system={conn.target_system} component={conn.target_component}\n"
                f"   autopilot={mavutil.mavlink.enums['MAV_AUTOPILOT'][hb.autopilot].name}\n"
                f"   type={mavutil.mavlink.enums['MAV_TYPE'][hb.type].name}"
            )

        if action == "disconnect":
            with _STATE["lock"]:
                # Stop all streams
                for name, info in list(_STATE["streams"].items()):
                    info["stop"].set()
                _STATE["streams"].clear()

                if _STATE["conn"]:
                    _STATE["conn"].close()
                    _STATE["conn"] = None
                    _STATE["connection_str"] = None
            return _ok("disconnected")

        if action == "status":
            if _STATE["conn"] is None:
                return _ok("not connected")
            conn = _STATE["conn"]
            return _ok(
                f"connected: {_STATE['connection_str']}\n"
                f"   target_system={_STATE['target_system']}\n"
                f"   target_component={_STATE['target_component']}\n"
                f"   active_streams={list(_STATE['streams'].keys())}"
            )

        # All below need an active connection
        conn = _require_conn()

        # ─── High-level flight control ───
        if action == "arm":
            conn.mav.command_long_send(
                _STATE["target_system"], _STATE["target_component"],
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                1, 0, 0, 0, 0, 0, 0,
            )
            ack = conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=timeout)
            return _ok(f"arm command sent: {_msg_to_dict(ack)}")

        if action == "disarm":
            conn.mav.command_long_send(
                _STATE["target_system"], _STATE["target_component"],
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                0, 0, 0, 0, 0, 0, 0,
            )
            ack = conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=timeout)
            return _ok(f"disarm command sent: {_msg_to_dict(ack)}")

        if action == "takeoff":
            if altitude is None:
                return _err("altitude required for takeoff")
            conn.mav.command_long_send(
                _STATE["target_system"], _STATE["target_component"],
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
                0, 0, 0, 0, 0, 0, float(altitude),
            )
            ack = conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=timeout)
            return _ok(f"takeoff to {altitude}m sent: {_msg_to_dict(ack)}")

        if action == "land":
            conn.mav.command_long_send(
                _STATE["target_system"], _STATE["target_component"],
                mavutil.mavlink.MAV_CMD_NAV_LAND, 0,
                0, 0, 0, 0, 0, 0, 0,
            )
            ack = conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=timeout)
            return _ok(f"land sent: {_msg_to_dict(ack)}")

        if action == "rtl":
            conn.mav.command_long_send(
                _STATE["target_system"], _STATE["target_component"],
                mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0,
                0, 0, 0, 0, 0, 0, 0,
            )
            ack = conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=timeout)
            return _ok(f"RTL sent: {_msg_to_dict(ack)}")

        if action == "set_mode":
            if not mode:
                return _err("mode required (e.g. 'GUIDED', 'AUTO', 'LOITER', 'RTL')")
            mode_id = conn.mode_mapping().get(mode.upper())
            if mode_id is None:
                available = list(conn.mode_mapping().keys())
                return _err(f"unknown mode '{mode}'. available: {available}")
            conn.mav.set_mode_send(
                _STATE["target_system"],
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id,
            )
            return _ok(f"mode set to {mode.upper()} (id={mode_id})")

        if action == "goto":
            if lat is None or lon is None:
                return _err("lat and lon required for goto")
            alt_m = float(alt if alt is not None else 10.0)
            # SET_POSITION_TARGET_GLOBAL_INT
            conn.mav.set_position_target_global_int_send(
                0,  # time_boot_ms
                _STATE["target_system"], _STATE["target_component"],
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                0b110111111000,  # position only
                int(lat * 1e7), int(lon * 1e7), alt_m,
                0, 0, 0, 0, 0, 0, 0, 0,
            )
            return _ok(f"goto ({lat}, {lon}, alt={alt_m}m) sent")

        if action == "velocity":
            if vx is None or vy is None or vz is None:
                return _err("vx, vy, vz required for velocity command (m/s, NED)")
            # SET_POSITION_TARGET_LOCAL_NED — velocity in body frame
            type_mask = 0b110111000111  # velocity only
            conn.mav.set_position_target_local_ned_send(
                0,
                _STATE["target_system"], _STATE["target_component"],
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                type_mask,
                0, 0, 0,
                float(vx), float(vy), float(vz),
                0, 0, 0,
                float(yaw or 0), 0,
            )
            return _ok(f"velocity ({vx}, {vy}, {vz}) m/s sent")

        # ─── Introspection ───
        if action == "list_messages":
            # Collect messages for a short window
            seen = {}
            end = time.time() + min(timeout, 3.0)
            while time.time() < end:
                m = conn.recv_match(blocking=False)
                if m is None:
                    time.sleep(0.01)
                    continue
                t = m.get_type()
                seen[t] = seen.get(t, 0) + 1
            lines = [f"   {name}: {count}x" for name, count in sorted(seen.items())]
            return _ok(f"messages observed in {min(timeout, 3.0)}s:\n" + "\n".join(lines))

        if action == "get_message":
            if not message:
                return _err("message name required")
            m = conn.recv_match(type=message, blocking=True, timeout=timeout)
            if m is None:
                return _err(f"no {message} received within {timeout}s")
            return _ok(f"{message}:\n{_msg_to_dict(m)}")

        if action == "stream":
            if not message:
                return _err("message name required")
            # Collect N samples at given rate
            results = []
            interval = 1.0 / max(rate, 0.1)
            end = time.time() + (samples / rate) + timeout
            while len(results) < samples and time.time() < end:
                m = conn.recv_match(type=message, blocking=True, timeout=interval * 2)
                if m is not None:
                    results.append(_msg_to_dict(m))
            return _ok(
                f"streamed {len(results)}/{samples} × {message}:\n"
                + "\n".join(f"   [{i}] {r}" for i, r in enumerate(results[:10]))
                + (f"\n   ... +{len(results)-10} more" if len(results) > 10 else "")
            )

        # ─── Raw MAVLink send ───
        if action == "send":
            if not message:
                return _err("message name required (e.g. 'COMMAND_LONG')")
            # Build message dynamically via getattr — future-proof like use_ros
            send_fn_name = f"{message.lower()}_send"
            send_fn = getattr(conn.mav, send_fn_name, None)
            if send_fn is None:
                return _err(f"unknown message type: {message} (no {send_fn_name})")
            send_fn(**(fields or {}))
            return _ok(f"sent {message} with {fields or {}}")

        return _err(
            f"unknown action: {action}\n"
            "   valid: connect, disconnect, status, arm, disarm, takeoff, land, rtl,\n"
            "          set_mode, goto, velocity, list_messages, get_message, stream, send"
        )

    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")
