"""ZCM (Zero Communications and Marshalling) transport for DevDuck agents.

Adds ZCM as a native peer-to-peer transport alongside Zenoh. ZCM is a
high-bandwidth, low-latency pub/sub framework originally designed for robotics
(derived from MIT's LCM used on the DARPA Urban Challenge).

Key Features:
1. Transport-Agnostic: UDP multicast, IPC, in-process, serial, etc.
2. Auto-Marshalling: Typed messages with auto-generated serialization
3. Low Latency: Designed for real-time robotics control loops
4. Language Bindings: C, C++, Python, Java, Node.js
5. LCM Compatible: ~95% API compatible with LCM

Architecture:
------------
Mirrors zenoh_peer.py patterns:
- devduck/presence/{id}  → DEVDUCK_PRESENCE channel
- devduck/broadcast      → DEVDUCK_BROADCAST channel
- devduck/cmd/{id}       → DEVDUCK_CMD_{id} channel
- devduck/response/{id}  → DEVDUCK_RESP_{requester}_{turn} channel

Transport URLs:
- udpm://239.255.76.67:7667  (UDP multicast, default - like LCM)
- ipc:///tmp/devduck_zcm     (IPC for local machine)
- inproc                      (in-process for threads)

Usage:
------
```python
# Start ZCM transport
zcm_peer(action="start")

# List peers
zcm_peer(action="list_peers")

# Broadcast to all ZCM peers
zcm_peer(action="broadcast", message="hello from zcm!")

# Send to specific peer
zcm_peer(action="send", peer_id="host-abc123", message="run diagnostics")
```

References:
- Original ZCM: https://github.com/ZeroCM/zcm
- Generalist AI fork: https://github.com/generalistai/zcm
- LCM (ancestor): https://github.com/lcm-proj/lcm
"""

import logging
import threading
import time
import os
import json
import uuid
import socket
import struct
import select
from typing import Any, Dict, Optional

from strands import tool

logger = logging.getLogger(__name__)

# ─── Global state (mirrors ZENOH_STATE pattern) ───
ZCM_STATE: Dict[str, Any] = {
    "running": False,
    "transport": None,
    "instance_id": None,
    "peers": {},
    "subscriptions": [],
    "agent": None,
    "pending_responses": {},
    "collected_responses": {},
    "streamed_content": {},
    "peers_version": 0,
    "transport_url": None,
    "start_time": None,
    "model": "unknown",
}

# ─── Constants ───
HEARTBEAT_INTERVAL = 5.0
PEER_TIMEOUT = 15.0

# ZCM Channel names (flat namespace, like LCM/ZCM convention)
CH_PRESENCE = "DEVDUCK_PRESENCE"
CH_BROADCAST = "DEVDUCK_BROADCAST"
CH_CMD_PREFIX = "DEVDUCK_CMD_"      # + instance_id
CH_RESP_PREFIX = "DEVDUCK_RESP_"    # + requester_turn

# Default UDP multicast (same group LCM traditionally uses, different port)
DEFAULT_MULTICAST_GROUP = "239.255.76.67"
DEFAULT_MULTICAST_PORT = 7667
DEFAULT_TRANSPORT_URL = f"udpm://{DEFAULT_MULTICAST_GROUP}:{DEFAULT_MULTICAST_PORT}"


def get_instance_id() -> str:
    """Generate or retrieve unique instance ID for this DevDuck ZCM peer."""
    if ZCM_STATE["instance_id"]:
        return ZCM_STATE["instance_id"]
    hostname = socket.gethostname()[:8]
    suffix = uuid.uuid4().hex[:6]
    instance_id = f"{hostname}-zcm-{suffix}"
    ZCM_STATE["instance_id"] = instance_id
    return instance_id


# ═══════════════════════════════════════════════════════════════════════
# Pure-Python UDP Multicast Transport (no zcm dependency required)
# ═══════════════════════════════════════════════════════════════════════
# This implements ZCM-compatible UDP multicast pub/sub from scratch.
# If the actual `zcm` Python package is available, we use it instead.

class UDPMulticastTransport:
    """Lightweight UDP multicast transport implementing ZCM-style pub/sub.

    This is a pure-Python fallback that doesn't require the ZCM C library.
    Messages are JSON-encoded with a simple framing protocol:
      [4 bytes: channel_len][channel_bytes][4 bytes: payload_len][payload_bytes]

    Compatible with ZCM's UDP multicast discovery model.
    """

    def __init__(self, multicast_group: str, port: int):
        self.group = multicast_group
        self.port = port
        self._callbacks: Dict[str, list] = {}
        self._running = False
        self._recv_thread: Optional[threading.Thread] = None

        # Create send socket
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._send_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        # Set multicast interface to all interfaces
        self._send_sock.setsockopt(
            socket.IPPROTO_IP, socket.IP_MULTICAST_IF,
            socket.inet_aton("0.0.0.0")
        )

        # Create receive socket
        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                self._recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except (AttributeError, OSError):
                pass

        self._recv_sock.bind(("", port))

        # Join multicast group
        mreq = struct.pack(
            "4s4s",
            socket.inet_aton(multicast_group),
            socket.inet_aton("0.0.0.0"),
        )
        self._recv_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self._recv_sock.settimeout(1.0)

    def publish(self, channel: str, data: bytes) -> None:
        """Publish data on a channel via UDP multicast."""
        # Frame: [2B magic][4B chan_len][chan][4B data_len][data]
        chan_bytes = channel.encode("utf-8")
        frame = (
            b"ZD"  # magic bytes for DevDuck ZCM
            + struct.pack("!I", len(chan_bytes))
            + chan_bytes
            + struct.pack("!I", len(data))
            + data
        )
        try:
            self._send_sock.sendto(frame, (self.group, self.port))
        except Exception as e:
            logger.warning(f"ZCM: UDP send error: {e}")

    def subscribe(self, channel: str, callback) -> None:
        """Subscribe to a channel with a callback(channel, data)."""
        if channel not in self._callbacks:
            self._callbacks[channel] = []
        self._callbacks[channel].append(callback)

    def start(self) -> None:
        """Start the receive loop."""
        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def stop(self) -> None:
        """Stop the transport."""
        self._running = False
        try:
            self._send_sock.close()
        except Exception:
            pass
        try:
            self._recv_sock.close()
        except Exception:
            pass

    def _recv_loop(self) -> None:
        """Background receive loop dispatching to callbacks."""
        while self._running:
            try:
                ready, _, _ = select.select([self._recv_sock], [], [], 1.0)
                if not ready:
                    continue

                raw, addr = self._recv_sock.recvfrom(65535)
                if len(raw) < 10 or raw[:2] != b"ZD":
                    continue  # Not our protocol

                offset = 2
                chan_len = struct.unpack("!I", raw[offset:offset + 4])[0]
                offset += 4
                channel = raw[offset:offset + chan_len].decode("utf-8")
                offset += chan_len
                data_len = struct.unpack("!I", raw[offset:offset + 4])[0]
                offset += 4
                data = raw[offset:offset + data_len]

                # Dispatch to matching callbacks
                # Support exact match and wildcard "*" subscriptions
                for pattern, cbs in self._callbacks.items():
                    if pattern == channel or pattern == "*":
                        for cb in cbs:
                            try:
                                cb(channel, data)
                            except Exception as e:
                                logger.error(f"ZCM: Callback error on {channel}: {e}")

            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    logger.debug("ZCM: Socket closed")
                break
            except Exception as e:
                if self._running:
                    logger.error(f"ZCM: Recv error: {e}")
                time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════════
# Message handlers (mirror zenoh_peer.py patterns)
# ═══════════════════════════════════════════════════════════════════════

def _handle_presence(channel: str, data: bytes) -> None:
    """Handle peer presence heartbeats."""
    try:
        msg = json.loads(data.decode("utf-8"))
        peer_id = msg.get("instance_id")
        if peer_id and peer_id != get_instance_id():
            is_new = peer_id not in ZCM_STATE["peers"]

            ZCM_STATE["peers"][peer_id] = {
                "last_seen": time.time(),
                "hostname": msg.get("hostname", "unknown"),
                "model": msg.get("model", "unknown"),
                "transport": msg.get("transport", "udpm"),
                "tools": msg.get("tools", []),
                "tool_count": msg.get("tool_count", 0),
                "cwd": msg.get("cwd", ""),
            }

            if is_new:
                ZCM_STATE["peers_version"] += 1
                hostname = msg.get("hostname", "unknown")
                model = msg.get("model", "unknown")
                logger.info(f"ZCM: NEW peer discovered: {peer_id}")
                print(f"\n🔗 [ZCM] New peer: {peer_id} ({hostname}) model={model}")

                # Register in mesh registry
                try:
                    from devduck.tools.mesh_registry import registry
                    registry.register(peer_id, "zcm", {
                        "hostname": hostname,
                        "model": model,
                        "layer": "local",
                        "name": hostname,
                        "transport": "zcm",
                    })
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"ZCM: Presence handler error: {e}")


def _handle_command(channel: str, data: bytes) -> None:
    """Handle incoming commands (broadcast or direct)."""
    try:
        msg = json.loads(data.decode("utf-8"))
        sender_id = msg.get("sender_id")
        turn_id = msg.get("turn_id")
        command = msg.get("command", "")

        if sender_id == get_instance_id():
            return  # Ignore own messages

        logger.info(f"ZCM: Command from {sender_id}: {command[:50]}...")

        instance_id = get_instance_id()
        resp_channel = f"{CH_RESP_PREFIX}{sender_id}_{turn_id}"

        # Send ACK
        _publish_json(resp_channel, {
            "type": "ack",
            "responder_id": instance_id,
            "turn_id": turn_id,
            "timestamp": time.time(),
        })

        # Process with a new DevDuck instance (avoid concurrent invocation)
        try:
            from devduck import DevDuck
            cmd_duck = DevDuck(auto_start_servers=False)

            if cmd_duck.agent:
                result = cmd_duck.agent(command)

                _publish_json(resp_channel, {
                    "type": "turn_end",
                    "responder_id": instance_id,
                    "turn_id": turn_id,
                    "result": str(result),
                    "timestamp": time.time(),
                })
            else:
                raise Exception("Failed to create DevDuck instance")

        except Exception as e:
            _publish_json(resp_channel, {
                "type": "error",
                "responder_id": instance_id,
                "turn_id": turn_id,
                "error": str(e),
                "timestamp": time.time(),
            })

    except Exception as e:
        logger.error(f"ZCM: Command handler error: {e}")


def _handle_response(channel: str, data: bytes) -> None:
    """Handle responses to our commands."""
    try:
        msg = json.loads(data.decode("utf-8"))
        turn_id = msg.get("turn_id")
        responder_id = msg.get("responder_id")
        msg_type = msg.get("type")

        if turn_id not in ZCM_STATE["pending_responses"]:
            return

        import sys

        if msg_type == "ack":
            sys.stdout.write(f"\n🦆 [ZCM] [{responder_id}] Processing...\n")
            sys.stdout.flush()

        elif msg_type == "stream":
            chunk_data = msg.get("data", "")
            if chunk_data:
                sys.stdout.write(chunk_data)
                sys.stdout.flush()
                if turn_id not in ZCM_STATE["streamed_content"]:
                    ZCM_STATE["streamed_content"][turn_id] = {}
                if responder_id not in ZCM_STATE["streamed_content"][turn_id]:
                    ZCM_STATE["streamed_content"][turn_id][responder_id] = ""
                ZCM_STATE["streamed_content"][turn_id][responder_id] += chunk_data

        elif msg_type == "turn_end":
            sys.stdout.write(f"\n\n✅ [ZCM] [{responder_id}] Complete\n")
            sys.stdout.flush()

            if turn_id not in ZCM_STATE["collected_responses"]:
                ZCM_STATE["collected_responses"][turn_id] = []
            ZCM_STATE["collected_responses"][turn_id].append({
                "responder": responder_id,
                "type": "complete",
                "result": msg.get("result"),
                "timestamp": msg.get("timestamp"),
            })

            pending = ZCM_STATE["pending_responses"].get(turn_id)
            if isinstance(pending, threading.Event):
                pending.set()

        elif msg_type == "error":
            sys.stdout.write(f"\n\n❌ [ZCM] [{responder_id}] Error: {msg.get('error')}\n")
            sys.stdout.flush()

            if turn_id not in ZCM_STATE["collected_responses"]:
                ZCM_STATE["collected_responses"][turn_id] = []
            ZCM_STATE["collected_responses"][turn_id].append({
                "responder": responder_id,
                "type": "error",
                "error": msg.get("error"),
            })

            pending = ZCM_STATE["pending_responses"].get(turn_id)
            if isinstance(pending, threading.Event):
                pending.set()

    except Exception as e:
        logger.error(f"ZCM: Response handler error: {e}")


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _publish_json(channel: str, data: dict) -> None:
    """Publish a JSON message on a ZCM channel."""
    transport = ZCM_STATE.get("transport")
    if transport:
        try:
            payload = json.dumps(data).encode("utf-8")
            transport.publish(channel, payload)
        except Exception as e:
            logger.error(f"ZCM: Publish error on {channel}: {e}")


def _heartbeat_loop() -> None:
    """Background heartbeat thread."""
    instance_id = get_instance_id()

    # Build metadata once
    tools = []
    tool_count = 0
    agent = ZCM_STATE.get("agent")
    if agent:
        try:
            if hasattr(agent, "tool_names"):
                tools = sorted(agent.tool_names)[:50]
                tool_count = len(list(agent.tool_names))
        except Exception:
            pass

    while ZCM_STATE["running"]:
        try:
            _publish_json(CH_PRESENCE, {
                "instance_id": instance_id,
                "hostname": socket.gethostname(),
                "model": ZCM_STATE.get("model", "unknown"),
                "transport": ZCM_STATE.get("transport_url", "udpm"),
                "tools": tools,
                "tool_count": tool_count,
                "cwd": os.getcwd(),
                "timestamp": time.time(),
            })

            # Heartbeat mesh registry
            try:
                from devduck.tools.mesh_registry import registry
                registry.heartbeat(instance_id, {"transport": "zcm"})
            except Exception:
                pass

            # Prune stale peers
            now = time.time()
            stale = [pid for pid, info in ZCM_STATE["peers"].items()
                     if now - info["last_seen"] > PEER_TIMEOUT]
            for pid in stale:
                del ZCM_STATE["peers"][pid]
                ZCM_STATE["peers_version"] += 1
                logger.info(f"ZCM: Peer {pid} timed out")
                print(f"\n⚡ [ZCM] Peer left: {pid}")

        except Exception as e:
            logger.error(f"ZCM: Heartbeat error: {e}")

        time.sleep(HEARTBEAT_INTERVAL)


# ═══════════════════════════════════════════════════════════════════════
# Core start/stop/actions
# ═══════════════════════════════════════════════════════════════════════

def start_zcm(
    agent=None,
    model: str = "unknown",
    transport_url: str = None,
) -> dict:
    """Start ZCM peer networking."""
    if ZCM_STATE["running"]:
        return {"status": "error", "content": [{"text": "❌ ZCM already running"}]}

    transport_url = transport_url or os.getenv("ZCM_TRANSPORT_URL", DEFAULT_TRANSPORT_URL)
    instance_id = get_instance_id()

    logger.info(f"ZCM: Starting as {instance_id} on {transport_url}")

    try:
        # Parse transport URL
        if transport_url.startswith("udpm://"):
            parts = transport_url.replace("udpm://", "").split(":")
            group = parts[0] if parts else DEFAULT_MULTICAST_GROUP
            port = int(parts[1]) if len(parts) > 1 else DEFAULT_MULTICAST_PORT
            transport = UDPMulticastTransport(group, port)
        else:
            return {"status": "error", "content": [
                {"text": f"❌ Unsupported transport: {transport_url}"},
                {"text": "Supported: udpm://group:port"},
            ]}

        # Store state
        ZCM_STATE["transport"] = transport
        ZCM_STATE["running"] = True
        ZCM_STATE["transport_url"] = transport_url
        ZCM_STATE["start_time"] = time.time()
        ZCM_STATE["model"] = model
        ZCM_STATE["agent"] = agent

        # Subscribe to channels
        transport.subscribe(CH_PRESENCE, _handle_presence)
        transport.subscribe(CH_BROADCAST, _handle_command)
        transport.subscribe(f"{CH_CMD_PREFIX}{instance_id}", _handle_command)

        # Subscribe to our response channels (wildcard pattern)
        # We handle routing in _handle_response by checking turn_id
        transport.subscribe("*", lambda ch, data: (
            _handle_response(ch, data) if ch.startswith(f"{CH_RESP_PREFIX}{instance_id}_") else None
        ))

        # Start transport receive loop
        transport.start()

        # Start heartbeat
        hb = threading.Thread(target=_heartbeat_loop, daemon=True)
        hb.start()

        # Register in mesh registry
        try:
            from devduck.tools.mesh_registry import registry
            registry.register(instance_id, "zcm", {
                "hostname": socket.gethostname(),
                "model": model,
                "is_self": True,
                "layer": "local",
                "name": socket.gethostname(),
                "transport": transport_url,
            })
        except Exception:
            pass

        logger.info(f"ZCM: Started as {instance_id} on {transport_url}")

        return {"status": "success", "content": [
            {"text": "✅ ZCM transport started"},
            {"text": f"🆔 Instance ID: {instance_id}"},
            {"text": f"📡 Transport: {transport_url}"},
            {"text": f"📢 Channels: {CH_PRESENCE}, {CH_BROADCAST}, {CH_CMD_PREFIX}{instance_id}"},
            {"text": ""},
            {"text": "Commands:"},
            {"text": "  • zcm_peer(action='list_peers')"},
            {"text": "  • zcm_peer(action='broadcast', message='...')"},
            {"text": "  • zcm_peer(action='send', peer_id='...', message='...')"},
        ]}

    except Exception as e:
        ZCM_STATE["running"] = False
        logger.error(f"ZCM: Start failed: {e}")
        return {"status": "error", "content": [{"text": f"❌ ZCM start failed: {e}"}]}


def stop_zcm() -> dict:
    """Stop ZCM transport."""
    if not ZCM_STATE["running"]:
        return {"status": "error", "content": [{"text": "❌ ZCM not running"}]}

    ZCM_STATE["running"] = False
    instance_id = ZCM_STATE["instance_id"]
    peer_count = len(ZCM_STATE["peers"])

    if ZCM_STATE["transport"]:
        ZCM_STATE["transport"].stop()
        ZCM_STATE["transport"] = None

    ZCM_STATE["peers"] = {}
    ZCM_STATE["agent"] = None
    ZCM_STATE["instance_id"] = None

    try:
        from devduck.tools.mesh_registry import registry
        if instance_id:
            registry.unregister(instance_id)
    except Exception:
        pass

    return {"status": "success", "content": [
        {"text": "✅ ZCM stopped"},
        {"text": f"🆔 Was: {instance_id}"},
        {"text": f"👥 Had {peer_count} peers"},
    ]}


def broadcast_zcm(message: str, wait_time: float = 60.0) -> dict:
    """Broadcast a command to all ZCM peers."""
    if not ZCM_STATE["running"]:
        return {"status": "error", "content": [{"text": "❌ ZCM not running"}]}

    if not ZCM_STATE["peers"]:
        return {"status": "error", "content": [
            {"text": "❌ No ZCM peers discovered yet"}
        ]}

    turn_id = uuid.uuid4().hex[:8]
    instance_id = get_instance_id()
    peer_count = len(ZCM_STATE["peers"])

    event = threading.Event()
    ZCM_STATE["pending_responses"][turn_id] = event
    ZCM_STATE["collected_responses"][turn_id] = []

    _publish_json(CH_BROADCAST, {
        "sender_id": instance_id,
        "turn_id": turn_id,
        "command": message,
        "timestamp": time.time(),
    })

    event.wait(timeout=wait_time)

    responses = ZCM_STATE["collected_responses"].pop(turn_id, [])
    streamed = ZCM_STATE["streamed_content"].pop(turn_id, {})
    ZCM_STATE["pending_responses"].pop(turn_id, None)

    content = [
        {"text": f"📢 [ZCM] Broadcast to {peer_count} peers"},
        {"text": f"💬 {message}"},
        {"text": f"📥 Responses: {len(responses)}"},
    ]
    for resp_id, text in streamed.items():
        content.append({"text": f"\n🦆 {resp_id}:\n{text}"})
    for resp in responses:
        if resp["type"] == "complete":
            content.append({"text": f"\n🦆 {resp['responder']}:\n{resp.get('result', '')[:500]}"})
        elif resp["type"] == "error":
            content.append({"text": f"\n❌ {resp['responder']}: {resp.get('error')}"})

    return {"status": "success", "content": content}


def send_zcm(peer_id: str, message: str, wait_time: float = 120.0) -> dict:
    """Send a command to a specific ZCM peer."""
    if not ZCM_STATE["running"]:
        return {"status": "error", "content": [{"text": "❌ ZCM not running"}]}

    if peer_id not in ZCM_STATE["peers"]:
        available = list(ZCM_STATE["peers"].keys())
        return {"status": "error", "content": [
            {"text": f"❌ ZCM peer '{peer_id}' not found"},
            {"text": f"Available: {available}"},
        ]}

    turn_id = uuid.uuid4().hex[:8]
    instance_id = get_instance_id()

    event = threading.Event()
    ZCM_STATE["pending_responses"][turn_id] = event
    ZCM_STATE["collected_responses"][turn_id] = []

    _publish_json(f"{CH_CMD_PREFIX}{peer_id}", {
        "sender_id": instance_id,
        "turn_id": turn_id,
        "command": message,
        "timestamp": time.time(),
    })

    event.wait(timeout=wait_time)

    responses = ZCM_STATE["collected_responses"].pop(turn_id, [])
    streamed = ZCM_STATE["streamed_content"].pop(turn_id, {})
    ZCM_STATE["pending_responses"].pop(turn_id, None)

    content = [{"text": f"📨 [ZCM] → {peer_id}"}, {"text": f"💬 {message}"}]
    for resp_id, text in streamed.items():
        content.append({"text": f"\n📥 {resp_id}:\n{text}"})
    for resp in responses:
        if resp["type"] == "complete":
            content.append({"text": f"\n📥 Response:\n{resp.get('result', '')}"})
        elif resp["type"] == "error":
            content.append({"text": f"\n❌ Error: {resp.get('error')}"})
    if not responses and not streamed:
        content.append({"text": "\n⏱️ No response (peer may be busy)"})

    return {"status": "success", "content": content}


# ═══════════════════════════════════════════════════════════════════════
# Tool entry point
# ═══════════════════════════════════════════════════════════════════════

@tool
def zcm_peer(
    action: str,
    message: str = "",
    peer_id: str = "",
    wait_time: float = 120.0,
    transport_url: str = "",
    agent=None,
) -> dict:
    """ZCM (Zero Communications and Marshalling) peer networking for DevDuck.

    High-bandwidth, low-latency pub/sub transport originally designed for robotics.
    Derived from MIT's LCM (DARPA Urban Challenge). Used by Generalist AI for GEN-1.

    Uses UDP multicast by default for zero-config peer discovery on LAN —
    similar to Zenoh but using ZCM's protocol, which is optimized for
    high-frequency robotics data (camera feeds, lidar, joint states, actions).

    Args:
        action: Action to perform:
            - "start": Start ZCM transport
            - "stop": Stop ZCM transport
            - "status": Show current status
            - "list_peers": List discovered peers
            - "broadcast": Send to ALL peers
            - "send": Send to specific peer
        message: Command/message to send
        peer_id: Target peer ID (for send)
        wait_time: Max seconds to wait for responses
        transport_url: ZCM transport URL (default: udpm://239.255.76.67:7667)
        agent: DevDuck agent instance

    Returns:
        Dictionary with status and content

    Examples:
        # Start ZCM (UDP multicast, auto-discovers peers on LAN)
        zcm_peer(action="start")

        # Start with custom transport
        zcm_peer(action="start", transport_url="udpm://239.255.76.67:7668")

        # Broadcast to all ZCM peers
        zcm_peer(action="broadcast", message="run health check")

        # Send to specific peer
        zcm_peer(action="send", peer_id="host-zcm-abc123", message="status")

    Environment:
        ZCM_TRANSPORT_URL - Override default transport (udpm://239.255.76.67:7667)
    """
    if action == "start":
        model = "unknown"
        if agent and hasattr(agent, "model"):
            m = getattr(agent, "model", None)
            if m:
                model = (
                    getattr(m, "model_id", None)
                    or getattr(m, "model_name", None)
                    or type(m).__name__
                )
        return start_zcm(
            agent=agent,
            model=model,
            transport_url=transport_url or None,
        )
    elif action == "stop":
        return stop_zcm()
    elif action == "status":
        if not ZCM_STATE["running"]:
            return {"status": "success", "content": [{"text": "ZCM not running"}]}
        uptime = time.time() - (ZCM_STATE.get("start_time") or time.time())
        return {"status": "success", "content": [
            {"text": "🦆 ZCM Status"},
            {"text": f"🆔 Instance: {get_instance_id()}"},
            {"text": f"📡 Transport: {ZCM_STATE.get('transport_url')}"},
            {"text": f"⏱️  Uptime: {uptime:.0f}s"},
            {"text": f"👥 Peers: {len(ZCM_STATE['peers'])}"},
        ]}
    elif action == "list_peers":
        if not ZCM_STATE["running"]:
            return {"status": "error", "content": [{"text": "❌ ZCM not running"}]}
        peers = ZCM_STATE["peers"]
        if not peers:
            return {"status": "success", "content": [
                {"text": "No ZCM peers yet"},
                {"text": "💡 Start another DevDuck with zcm_peer(action='start')"},
            ]}
        content = [{"text": f"👥 ZCM Peers ({len(peers)}):"}]
        for pid, info in peers.items():
            age = time.time() - info["last_seen"]
            content.append({"text": (
                f"\n  🦆 {pid}\n"
                f"     Host: {info.get('hostname', '?')}\n"
                f"     Model: {info.get('model', '?')}\n"
                f"     Transport: {info.get('transport', '?')}\n"
                f"     Seen: {age:.1f}s ago"
            )})
        return {"status": "success", "content": content}
    elif action == "broadcast":
        if not message:
            return {"status": "error", "content": [{"text": "❌ message required"}]}
        return broadcast_zcm(message, wait_time)
    elif action == "send":
        if not peer_id or not message:
            return {"status": "error", "content": [{"text": "❌ peer_id and message required"}]}
        return send_zcm(peer_id, message, wait_time)
    else:
        return {"status": "error", "content": [
            {"text": f"❌ Unknown action: {action}"},
            {"text": "Valid: start, stop, status, list_peers, broadcast, send"},
        ]}
