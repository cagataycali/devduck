"""Zenoh tool for DevDuck agents with automatic peer discovery.

This module provides Zenoh-based peer-to-peer communication for DevDuck agents,
allowing multiple DevDuck instances to automatically discover and communicate
with each other using Zenoh's multicast scouting.

Key Features:
1. Auto-Discovery: DevDuck instances find each other automatically via multicast
2. Peer-to-Peer: Direct communication without central server
3. Broadcast: Send commands to ALL connected DevDuck instances at once
4. Direct Message: Send to specific peer by instance ID
5. Real-time Streaming: Responses stream as they're generated

How It Works:
------------
When a DevDuck instance starts with Zenoh enabled:
1. Joins the Zenoh peer network via multicast scouting (224.0.0.224:7446)
2. Subscribes to "devduck/**" key expressions for messages
3. Publishes its presence to "devduck/presence/{instance_id}"
4. Listens for commands on "devduck/cmd/{instance_id}" and "devduck/broadcast"
5. Responds on "devduck/response/{requester_id}/{turn_id}"

Key Expressions:
---------------
- devduck/presence/{id}  - Peer announcements (heartbeat)
- devduck/broadcast      - Messages to all peers
- devduck/cmd/{id}       - Direct messages to specific peer
- devduck/response/{requester}/{turn_id} - Responses

Usage:
------
```python
# Terminal 1
devduck "start zenoh"  # or auto-starts if DEVDUCK_ENABLE_ZENOH=true

# Terminal 2
devduck "start zenoh"  # Auto-discovers Terminal 1

# Terminal 1: Broadcast to all
devduck "zenoh broadcast 'list all files'"

# Terminal 2: Send to specific peer
devduck "zenoh send peer-abc123 'what is 2+2?'"

# Check discovered peers
devduck "zenoh list peers"
```

Note: This file is named zenoh_peer.py to avoid shadowing the eclipse-zenoh package.
The tool is exported as 'zenoh' for backward compatibility.
"""

import logging
import importlib
import threading
import time
import os
import json
import uuid
import socket
from typing import Any
from datetime import datetime

from strands import tool

logger = logging.getLogger(__name__)

# Global state for Zenoh
ZENOH_STATE: dict[str, Any] = {
    "running": False,
    "session": None,
    "instance_id": None,
    "peers": {},  # {peer_id: {last_seen, hostname, ...}}
    "subscribers": [],
    "publisher": None,
    "agent": None,
    "pending_responses": {},  # {turn_id: asyncio.Future or threading.Event}
    "collected_responses": {},  # {turn_id: [responses]}
    "streamed_content": {},  # {turn_id: {responder_id: "accumulated text"}}
    "peers_version": 0,  # Bumped on every peer change — lets proxy detect changes cheaply
}

# Heartbeat interval in seconds
HEARTBEAT_INTERVAL = 5.0
PEER_TIMEOUT = 15.0  # Consider peer dead after this many seconds


def get_instance_id() -> str:
    """Generate or retrieve unique instance ID for this DevDuck."""
    if ZENOH_STATE["instance_id"]:
        return ZENOH_STATE["instance_id"]

    # Generate a unique ID based on hostname + random suffix
    hostname = socket.gethostname()[:8]
    suffix = uuid.uuid4().hex[:6]
    instance_id = f"{hostname}-{suffix}"
    ZENOH_STATE["instance_id"] = instance_id
    return instance_id


def _push_zenoh_state_to_browsers(reason: str = "update") -> None:
    """Push current Zenoh state directly to all connected browser clients.

    This is the SOLE mechanism for Zenoh → Browser updates. Called immediately
    when peers join, leave, or on first heartbeat. No polling needed.

    Args:
        reason: Why this push happened (for logging)
    """
    try:
        from devduck.tools.unified_mesh import MESH_STATE
        from devduck.tools.agentcore_proxy import _GATEWAY_STATE
        import asyncio

        ws_clients = MESH_STATE.get("ws_clients", {})
        if not ws_clients:
            return

        # Get the event loop from agentcore_proxy
        loop = _GATEWAY_STATE.get("loop")
        if not loop:
            return

        my_id = get_instance_id()
        if not my_id:
            return

        # Build full peer list (self + remote peers)
        peer_list = []

        # Self
        self_meta = {}
        try:
            self_meta = _build_rich_presence()
        except Exception:
            pass

        peer_list.append(
            {
                "id": my_id,
                "hostname": socket.gethostname(),
                "model": ZENOH_STATE.get("model", "unknown"),
                "last_seen": time.time(),
                "is_self": True,
                "tools": self_meta.get("tools", []),
                "tool_count": self_meta.get("tool_count", 0),
                "system_prompt_preview": self_meta.get("system_prompt_preview", ""),
                "cwd": self_meta.get("cwd", ""),
                "platform": self_meta.get("platform", ""),
            }
        )

        # Remote peers
        for pid, info in ZENOH_STATE.get("peers", {}).items():
            peer_list.append(
                {
                    "id": pid,
                    "hostname": info.get("hostname", "unknown"),
                    "model": info.get("model", "unknown"),
                    "last_seen": info.get("last_seen", 0),
                    "tools": info.get("tools", []),
                    "tool_count": info.get("tool_count", 0),
                    "system_prompt_preview": info.get("system_prompt_preview", ""),
                    "cwd": info.get("cwd", ""),
                    "platform": info.get("platform", ""),
                }
            )

        update_msg = json.dumps(
            {
                "type": "zenoh_peers_update",
                "instance_id": my_id,
                "peers": peer_list,
                "timestamp": time.time(),
            }
        )

        peer_ids = [p["id"] for p in peer_list]
        logger.info(f"Zenoh→Browser push ({reason}): {len(peer_list)} peers {peer_ids}")

        for ws_id, websocket in list(ws_clients.items()):
            try:
                asyncio.run_coroutine_threadsafe(websocket.send(update_msg), loop)
            except Exception as e:
                logger.debug(f"Failed to push zenoh state to browser {ws_id}: {e}")

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Zenoh browser push error: {e}")


def _forward_zenoh_event_to_browsers(data: dict) -> None:
    """Forward a Zenoh response event to all connected browser clients.

    This gives browsers real-time visibility into Zenoh agent responses
    (stream chunks, tool events, turn_end) without going through the proxy poll.

    Args:
        data: The zenoh response event dict
    """
    try:
        from devduck.tools.unified_mesh import MESH_STATE
        from devduck.tools.agentcore_proxy import _GATEWAY_STATE
        import asyncio

        ws_clients = MESH_STATE.get("ws_clients", {})
        if not ws_clients:
            return

        loop = _GATEWAY_STATE.get("loop")
        if not loop:
            return

        msg_type = data.get("type")
        responder_id = data.get("responder_id", "")
        turn_id = data.get("turn_id", "")

        # Map zenoh event types to browser-compatible message types
        if msg_type == "stream":
            ws_msg = {
                "type": "chunk",
                "turn_id": turn_id,
                "agent_id": responder_id,
                "agent_type": "zenoh",
                "data": data.get("data", ""),
                "chunk_type": data.get("chunk_type", "text"),
                "timestamp": data.get("timestamp", time.time()),
            }
        elif msg_type == "ack":
            ws_msg = {
                "type": "turn_start",
                "turn_id": turn_id,
                "agent_id": responder_id,
                "agent_type": "zenoh",
                "timestamp": data.get("timestamp", time.time()),
            }
        elif msg_type == "turn_end":
            ws_msg = {
                "type": "turn_end",
                "turn_id": turn_id,
                "agent_id": responder_id,
                "agent_type": "zenoh",
                "response": data.get("result", ""),
                "chunks_sent": data.get("chunks_sent", 0),
                "timestamp": data.get("timestamp", time.time()),
            }
        elif msg_type == "error":
            ws_msg = {
                "type": "turn_end",
                "turn_id": turn_id,
                "agent_id": responder_id,
                "agent_type": "zenoh",
                "response": f"❌ {data.get('error', 'unknown error')}",
                "timestamp": data.get("timestamp", time.time()),
            }
        else:
            return  # Skip unknown types

        msg_json = json.dumps(ws_msg)
        for ws_id, websocket in list(ws_clients.items()):
            try:
                asyncio.run_coroutine_threadsafe(websocket.send(msg_json), loop)
            except Exception as e:
                logger.debug(f"Failed to forward zenoh event to browser {ws_id}: {e}")

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Zenoh event forward error: {e}")


def handle_presence(sample) -> None:
    """Handle peer presence announcements.

    Updates ZENOH_STATE and pushes directly to browsers on change.
    No polling — immediate push on new peer or metadata update.

    Args:
        sample: Zenoh sample containing peer info
    """
    try:
        key = str(sample.key_expr)
        payload = sample.payload.to_bytes().decode()
        data = json.loads(payload)

        peer_id = data.get("instance_id")
        if peer_id and peer_id != get_instance_id():
            is_new = peer_id not in ZENOH_STATE["peers"]

            # Update peer info with all available metadata
            ZENOH_STATE["peers"][peer_id] = {
                "last_seen": time.time(),
                "hostname": data.get("hostname", "unknown"),
                "started": data.get("started"),
                "model": data.get("model", "unknown"),
                # Rich metadata from peer
                "tools": data.get("tools", []),
                "tool_count": data.get("tool_count", 0),
                "system_prompt_preview": data.get("system_prompt_preview", ""),
                "cwd": data.get("cwd", ""),
                "python_version": data.get("python_version", ""),
                "platform": data.get("platform", ""),
            }

            if is_new:
                ZENOH_STATE["peers_version"] += 1
                hostname = data.get("hostname", "unknown")
                model = data.get("model", "unknown")
                tool_count = data.get("tool_count", 0)
                cwd = data.get("cwd", "")
                logger.info(f"Zenoh: NEW peer discovered: {peer_id}")
                print(
                    f"\n🔗 New peer joined: {peer_id} ({hostname}) — model: {model}, tools: {tool_count}, cwd: {cwd}"
                )
                # Register discovered peer in file-based registry
                try:
                    from devduck.tools.mesh_registry import registry

                    registry.register(
                        peer_id,
                        "zenoh",
                        {
                            "hostname": hostname,
                            "model": model,
                            "tools": data.get("tools", []),
                            "tool_count": tool_count,
                            "cwd": cwd,
                            "layer": "local",
                            "name": hostname,
                            "platform": data.get("platform", ""),
                            "system_prompt_preview": data.get(
                                "system_prompt_preview", ""
                            ),
                        },
                    )
                except Exception:
                    pass
                _push_zenoh_state_to_browsers(f"new_peer:{peer_id}")
            else:
                # Update existing peer's timestamp in registry
                try:
                    from devduck.tools.mesh_registry import registry

                    registry.heartbeat(peer_id)
                except Exception:
                    pass
                logger.debug(f"Zenoh: Peer heartbeat: {peer_id}")
    except Exception as e:
        logger.error(f"Zenoh: Error handling presence: {e}")


class ZenohStreamingCallbackHandler:
    """Callback handler that streams agent responses over Zenoh.

    This handler implements real-time streaming of:
    - Assistant responses (text chunks as they're generated)
    - Tool invocations (names and status)
    - Reasoning text (if enabled)
    - Tool results (success/error status)

    All data is published immediately to Zenoh for the requester to receive.
    """

    def __init__(self, response_key: str, turn_id: str, responder_id: str):
        """Initialize the streaming handler.

        Args:
            response_key: Zenoh key expression to publish responses to
            turn_id: Unique turn ID for this conversation
            responder_id: This instance's ID
        """
        self.response_key = response_key
        self.turn_id = turn_id
        self.responder_id = responder_id
        self.tool_count = 0
        self.previous_tool_use = None
        self.chunk_count = 0

    def _publish(self, data: str, chunk_type: str = "text") -> None:
        """Publish a streaming chunk over Zenoh.

        Args:
            data: String data to publish
            chunk_type: Type of chunk (text, tool, reasoning)
        """
        try:
            self.chunk_count += 1
            chunk_msg = {
                "type": "stream",
                "chunk_type": chunk_type,
                "responder_id": self.responder_id,
                "turn_id": self.turn_id,
                "chunk_num": self.chunk_count,
                "data": data,
                "timestamp": time.time(),
            }
            publish_message(self.response_key, chunk_msg)
        except Exception as e:
            logger.warning(f"Zenoh: Failed to publish stream chunk: {e}")

    def __call__(self, **kwargs) -> None:
        """Stream events to Zenoh in real-time.

        Args:
            **kwargs: Callback event data including:
                - reasoningText (Optional[str]): Reasoning text to stream
                - data (str): Text content to stream
                - complete (bool): Whether this is the final chunk
                - current_tool_use (dict): Current tool being invoked
                - message (dict): Full message objects (for tool results)
        """
        reasoningText = kwargs.get("reasoningText", False)
        data = kwargs.get("data", "")
        complete = kwargs.get("complete", False)
        current_tool_use = kwargs.get("current_tool_use", {})
        message = kwargs.get("message", {})

        # Stream reasoning text
        if reasoningText:
            self._publish(reasoningText, "reasoning")

        # Stream response text chunks
        if data:
            self._publish(data, "text")
            if complete:
                self._publish("\n", "text")

        # Stream tool invocation notifications
        if current_tool_use and current_tool_use.get("name"):
            tool_name = current_tool_use.get("name", "Unknown tool")
            if self.previous_tool_use != current_tool_use:
                self.previous_tool_use = current_tool_use
                self.tool_count += 1
                self._publish(f"\n🛠️  Tool #{self.tool_count}: {tool_name}\n", "tool")

        # Stream tool results
        if isinstance(message, dict) and message.get("role") == "user":
            for content in message.get("content", []):
                if isinstance(content, dict):
                    tool_result = content.get("toolResult")
                    if tool_result:
                        status = tool_result.get("status", "unknown")
                        if status == "success":
                            self._publish("✅ Tool completed successfully\n", "tool")
                        else:
                            self._publish("❌ Tool failed\n", "tool")


def handle_command(sample) -> None:
    """Handle incoming commands (broadcast or direct).

    Creates a NEW DevDuck instance for each command to avoid concurrent
    invocation errors (Strands Agent doesn't support concurrent requests).

    Uses ZenohStreamingCallbackHandler to stream responses in real-time,
    just like the TCP implementation.

    Args:
        sample: Zenoh sample containing command
    """
    try:
        key = str(sample.key_expr)
        payload = sample.payload.to_bytes().decode()
        data = json.loads(payload)

        sender_id = data.get("sender_id")
        turn_id = data.get("turn_id")
        command = data.get("command", "")

        # Don't process our own messages
        if sender_id == get_instance_id():
            return

        logger.info(f"Zenoh: Received command from {sender_id}: {command[:50]}...")

        # Process the command with a NEW DevDuck instance
        # This avoids concurrent invocation errors on the main agent
        try:
            # Create response topic
            response_key = f"devduck/response/{sender_id}/{turn_id}"
            instance_id = get_instance_id()

            # Send acknowledgment
            ack = {
                "type": "ack",
                "responder_id": instance_id,
                "turn_id": turn_id,
                "timestamp": time.time(),
            }
            publish_message(response_key, ack)

            # Create a NEW DevDuck instance for this command
            # auto_start_servers=False prevents recursion
            from devduck import DevDuck

            command_devduck = DevDuck(auto_start_servers=False)

            if command_devduck.agent:
                # Create streaming callback handler for real-time response streaming
                streaming_handler = ZenohStreamingCallbackHandler(
                    response_key=response_key,
                    turn_id=turn_id,
                    responder_id=instance_id,
                )

                # Attach streaming handler to the agent
                command_devduck.agent.callback_handler = streaming_handler

                # Process with the new agent instance
                # Responses stream automatically via callback_handler
                result = command_devduck.agent(command)

                # Send turn_end AFTER agent completes and all chunks are sent
                # This is the definitive signal that streaming is complete
                turn_end = {
                    "type": "turn_end",
                    "responder_id": instance_id,
                    "turn_id": turn_id,
                    "result": str(result),
                    "chunks_sent": streaming_handler.chunk_count,
                    "timestamp": time.time(),
                }
                publish_message(response_key, turn_end)

                logger.info(
                    f"Zenoh: Sent turn_end to {sender_id} for turn {turn_id} ({streaming_handler.chunk_count} chunks)"
                )
            else:
                raise Exception("Failed to create DevDuck instance")

        except Exception as e:
            # Send error response
            error_response = {
                "type": "error",
                "responder_id": get_instance_id(),
                "turn_id": turn_id,
                "error": str(e),
                "timestamp": time.time(),
            }
            publish_message(f"devduck/response/{sender_id}/{turn_id}", error_response)
            logger.error(f"Zenoh: Error processing command: {e}")

    except Exception as e:
        logger.error(f"Zenoh: Error handling command: {e}")


def handle_response(sample) -> None:
    """Handle responses to our commands.

    Streams chunks to terminal in real-time and collects final response.
    Waits for explicit 'turn_end' message which indicates all streaming is complete.

    ALSO forwards all stream events to WebSocket clients for browser visibility.

    Args:
        sample: Zenoh sample containing response
    """
    try:
        key = str(sample.key_expr)
        payload = sample.payload.to_bytes().decode()
        data = json.loads(payload)

        turn_id = data.get("turn_id")
        responder_id = data.get("responder_id")
        msg_type = data.get("type")

        # Forward ALL zenoh response events to browsers for real-time visibility
        _forward_zenoh_event_to_browsers(data)

        if turn_id in ZENOH_STATE["pending_responses"]:
            # Handle streaming chunks - print to terminal AND forward to browser
            if msg_type == "stream":
                chunk_data = data.get("data", "")
                chunk_type = data.get("chunk_type", "text")

                # Print streaming content directly to terminal
                # This gives the same experience as TCP streaming
                import sys

                if chunk_data:
                    sys.stdout.write(chunk_data)
                    sys.stdout.flush()

                    # Also collect streamed content for tool return value
                    if turn_id not in ZENOH_STATE["streamed_content"]:
                        ZENOH_STATE["streamed_content"][turn_id] = {}
                    if responder_id not in ZENOH_STATE["streamed_content"][turn_id]:
                        ZENOH_STATE["streamed_content"][turn_id][responder_id] = ""
                    ZENOH_STATE["streamed_content"][turn_id][responder_id] += chunk_data

                logger.debug(f"Zenoh: Stream chunk from {responder_id}: {chunk_type}")
                return  # Continue to next chunk

            # Handle ACK - show peer is processing
            if msg_type == "ack":
                import sys

                sys.stdout.write(f"\n🦆 [{responder_id}] Processing...\n")
                sys.stdout.flush()

                logger.debug(f"Zenoh: ACK from {responder_id} for turn {turn_id}")
                return

            # Handle turn_end - THIS is the real completion signal
            # Sent AFTER all stream chunks have been published
            if msg_type == "turn_end":
                import sys

                chunks_sent = data.get("chunks_sent", 0)
                sys.stdout.write(
                    f"\n\n✅ [{responder_id}] Complete ({chunks_sent} chunks)\n"
                )
                sys.stdout.flush()

                # Store final result if present
                if turn_id not in ZENOH_STATE["collected_responses"]:
                    ZENOH_STATE["collected_responses"][turn_id] = []

                ZENOH_STATE["collected_responses"][turn_id].append(
                    {
                        "responder": responder_id,
                        "type": "complete",
                        "result": data.get("result"),
                        "chunks_sent": chunks_sent,
                        "timestamp": data.get("timestamp"),
                    }
                )

                # Signal completion - all chunks have been sent
                pending = ZENOH_STATE["pending_responses"].get(turn_id)
                if isinstance(pending, threading.Event):
                    pending.set()

                logger.debug(
                    f"Zenoh: Turn ended from {responder_id} for turn {turn_id}"
                )
                return

            # Handle errors
            if msg_type == "error":
                import sys

                sys.stdout.write(
                    f"\n\n❌ [{responder_id}] Error: {data.get('error', 'unknown')}\n"
                )
                sys.stdout.flush()

                if turn_id not in ZENOH_STATE["collected_responses"]:
                    ZENOH_STATE["collected_responses"][turn_id] = []

                ZENOH_STATE["collected_responses"][turn_id].append(
                    {
                        "responder": responder_id,
                        "type": "error",
                        "error": data.get("error"),
                        "timestamp": data.get("timestamp"),
                    }
                )

                # Signal completion on error too
                pending = ZENOH_STATE["pending_responses"].get(turn_id)
                if isinstance(pending, threading.Event):
                    pending.set()

                logger.debug(f"Zenoh: Error from {responder_id} for turn {turn_id}")
                return

            # Legacy "response" type - treat as turn_end for backward compatibility
            if msg_type == "response":
                # Old-style response, treat as completion
                if turn_id not in ZENOH_STATE["collected_responses"]:
                    ZENOH_STATE["collected_responses"][turn_id] = []

                ZENOH_STATE["collected_responses"][turn_id].append(
                    {
                        "responder": responder_id,
                        "type": msg_type,
                        "result": data.get("result"),
                        "chunks_sent": data.get("chunks_sent", 0),
                        "timestamp": data.get("timestamp"),
                    }
                )

                # Don't signal completion here - wait for turn_end
                logger.debug(
                    f"Zenoh: Legacy response from {responder_id} for turn {turn_id}"
                )

    except Exception as e:
        logger.error(f"Zenoh: Error handling response: {e}")


def publish_message(key_expr: str, data: dict) -> None:
    """Publish a message to a Zenoh key expression.

    Args:
        key_expr: The key expression to publish to
        data: Dictionary to publish as JSON
    """
    if ZENOH_STATE["session"]:
        try:
            payload = json.dumps(data).encode()
            ZENOH_STATE["session"].put(key_expr, payload)
        except Exception as e:
            logger.error(f"Zenoh: Error publishing to {key_expr}: {e}")


def heartbeat_thread() -> None:
    """Background thread that sends periodic presence announcements.

    Publishes rich presence data including tools, system prompt summary,
    working directory, and other metadata so peers and the UI can show
    meaningful info about each DevDuck instance.

    On first heartbeat, pushes self to browsers immediately.
    On peer timeout, pushes updated state to browsers immediately.
    """
    instance_id = get_instance_id()

    # Build rich metadata once (these don't change between heartbeats)
    _rich_meta = _build_rich_presence()

    first_beat = True

    while ZENOH_STATE["running"]:
        try:
            # Publish presence with rich metadata
            presence_data = {
                "instance_id": instance_id,
                "hostname": socket.gethostname(),
                "started": ZENOH_STATE.get("start_time"),
                "model": ZENOH_STATE.get("model", "unknown"),
                "timestamp": time.time(),
                # Rich metadata
                "tools": _rich_meta.get("tools", []),
                "tool_count": _rich_meta.get("tool_count", 0),
                "system_prompt_preview": _rich_meta.get("system_prompt_preview", ""),
                "cwd": _rich_meta.get("cwd", ""),
                "python_version": _rich_meta.get("python_version", ""),
                "platform": _rich_meta.get("platform", ""),
            }
            publish_message(f"devduck/presence/{instance_id}", presence_data)

            # Also heartbeat the file-based registry (visible to proxy, browser, etc.)
            try:
                from devduck.tools.mesh_registry import registry

                registry.heartbeat(
                    instance_id,
                    {
                        "peers_count": len(ZENOH_STATE.get("peers", {})),
                        "cwd": os.getcwd(),
                    },
                )
            except Exception:
                pass

            # On first heartbeat, push self to browsers immediately
            if first_beat:
                first_beat = False
                _push_zenoh_state_to_browsers("first_heartbeat")

            # Clean up stale peers
            current_time = time.time()
            stale_peers = [
                peer_id
                for peer_id, info in ZENOH_STATE["peers"].items()
                if current_time - info["last_seen"] > PEER_TIMEOUT
            ]
            if stale_peers:
                for peer_id in stale_peers:
                    del ZENOH_STATE["peers"][peer_id]
                    logger.info(f"Zenoh: Peer {peer_id} timed out")
                    print(f"\n⚡ Peer left: {peer_id}")
                ZENOH_STATE["peers_version"] += 1
                # Push to browsers IMMEDIATELY on peer removal
                _push_zenoh_state_to_browsers(f"peer_timeout:{','.join(stale_peers)}")

        except Exception as e:
            logger.error(f"Zenoh: Heartbeat error: {e}")

        time.sleep(HEARTBEAT_INTERVAL)


def _build_rich_presence() -> dict:
    """Extract rich metadata from the current DevDuck agent for presence announcements.

    Called once when heartbeat starts. Returns tools, system prompt preview, etc.
    """
    import platform as _platform
    import sys

    meta = {
        "cwd": os.getcwd(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": f"{_platform.system()} {_platform.machine()}",
        "tools": [],
        "tool_count": 0,
        "system_prompt_preview": "",
    }

    agent = ZENOH_STATE.get("agent")
    if agent:
        try:
            # Extract tool names
            if hasattr(agent, "tool_names"):
                tool_names = sorted(agent.tool_names)
                meta["tools"] = tool_names[:50]  # Cap at 50 to keep heartbeat small
                meta["tool_count"] = len(tool_names)
            elif hasattr(agent, "tool_registry") and hasattr(
                agent.tool_registry, "registry"
            ):
                tool_names = sorted(agent.tool_registry.registry.keys())
                meta["tools"] = tool_names[:50]
                meta["tool_count"] = len(tool_names)
        except Exception as e:
            logger.debug(f"Could not extract tools for presence: {e}")

        try:
            # Extract system prompt preview (first 200 chars, skip boilerplate)
            if hasattr(agent, "system_prompt") and agent.system_prompt:
                sp = agent.system_prompt
                # Try to find meaningful part after the boilerplate header
                for marker in ["You are:", "You are ", "##"]:
                    idx = sp.find(marker)
                    if idx > 0 and idx < 500:
                        sp = sp[idx:]
                        break
                meta["system_prompt_preview"] = sp[:200].replace("\n", " ").strip()
        except Exception as e:
            logger.debug(f"Could not extract system prompt for presence: {e}")

    return meta


def start_zenoh(
    agent=None,
    model: str = "unknown",
    connect: str = None,
    listen: str = None,
) -> dict:
    """Start Zenoh peer networking for DevDuck.

    Args:
        agent: The DevDuck agent instance to use for processing commands
        model: Model name for peer info
        connect: Remote endpoint(s) to connect to (e.g., "tcp/1.2.3.4:7447" or comma-separated)
        listen: Endpoint(s) to listen on (e.g., "tcp/0.0.0.0:7447" for public access)

    Returns:
        Status dictionary
    """
    if ZENOH_STATE["running"]:
        return {
            "status": "error",
            "content": [{"text": "❌ Zenoh already running"}],
        }

    try:
        # Use importlib to avoid shadowing by this file's name
        zenoh_pkg = importlib.import_module("zenoh")
    except ImportError:
        return {
            "status": "error",
            "content": [
                {"text": "❌ Zenoh not installed. Run: pip install eclipse-zenoh"}
            ],
        }

    try:
        instance_id = get_instance_id()
        logger.info(f"Zenoh: Starting with instance ID: {instance_id}")

        # Check for env vars for remote connection
        connect = connect or os.getenv("ZENOH_CONNECT")
        listen = listen or os.getenv("ZENOH_LISTEN")

        # Configure Zenoh for peer mode with multicast scouting
        # API changed in zenoh 1.x - handle both versions
        try:
            # New API (zenoh >= 1.0)
            config = zenoh_pkg.Config.default()
        except AttributeError:
            try:
                # Old API (zenoh < 1.0)
                config = zenoh_pkg.Config()
            except AttributeError:
                # Fallback - open without config
                config = None

        # Configure remote endpoints if provided
        endpoints_info = []
        if config is not None:
            # Add connect endpoints (for connecting to remote peers/routers)
            if connect:
                connect_endpoints = [e.strip() for e in connect.split(",")]
                try:
                    config.insert_json5(
                        "connect/endpoints", json.dumps(connect_endpoints)
                    )
                    endpoints_info.append(
                        f"🔗 Connecting to: {', '.join(connect_endpoints)}"
                    )
                    logger.info(
                        f"Zenoh: Configured connect endpoints: {connect_endpoints}"
                    )
                except Exception as e:
                    logger.warning(f"Zenoh: Failed to set connect endpoints: {e}")

            # Add listen endpoints (for accepting remote connections)
            if listen:
                listen_endpoints = [e.strip() for e in listen.split(",")]
                try:
                    config.insert_json5(
                        "listen/endpoints", json.dumps(listen_endpoints)
                    )
                    endpoints_info.append(
                        f"👂 Listening on: {', '.join(listen_endpoints)}"
                    )
                    logger.info(
                        f"Zenoh: Configured listen endpoints: {listen_endpoints}"
                    )
                except Exception as e:
                    logger.warning(f"Zenoh: Failed to set listen endpoints: {e}")

        # Open Zenoh session
        if config is not None:
            session = zenoh_pkg.open(config)
        else:
            session = zenoh_pkg.open()
        ZENOH_STATE["session"] = session
        ZENOH_STATE["running"] = True
        ZENOH_STATE["start_time"] = datetime.now().isoformat()
        ZENOH_STATE["model"] = model
        ZENOH_STATE["agent"] = agent

        # Subscribe to presence announcements
        presence_sub = session.declare_subscriber("devduck/presence/*", handle_presence)
        ZENOH_STATE["subscribers"].append(presence_sub)

        # Subscribe to broadcast commands
        broadcast_sub = session.declare_subscriber("devduck/broadcast", handle_command)
        ZENOH_STATE["subscribers"].append(broadcast_sub)

        # Subscribe to direct commands for this instance
        direct_sub = session.declare_subscriber(
            f"devduck/cmd/{instance_id}", handle_command
        )
        ZENOH_STATE["subscribers"].append(direct_sub)

        # Subscribe to responses for this instance
        response_sub = session.declare_subscriber(
            f"devduck/response/{instance_id}/*", handle_response
        )
        ZENOH_STATE["subscribers"].append(response_sub)

        # Start heartbeat thread
        heartbeat = threading.Thread(target=heartbeat_thread, daemon=True)
        heartbeat.start()
        ZENOH_STATE["heartbeat_thread"] = heartbeat

        # Register in file-based mesh registry (visible to ALL processes)
        try:
            from devduck.tools.mesh_registry import registry

            _rich_meta = _build_rich_presence()
            registry.register(
                instance_id,
                "zenoh",
                {
                    "hostname": socket.gethostname(),
                    "model": model,
                    "is_self": True,
                    "layer": "local",
                    "name": socket.gethostname(),
                    **_rich_meta,
                },
            )
            logger.info(f"Zenoh: Registered in mesh registry as {instance_id}")
        except Exception as e:
            logger.warning(f"Zenoh: Registry registration failed (non-fatal): {e}")

        logger.info(f"Zenoh: Started successfully as {instance_id}")

        # Build response content
        content = [
            {"text": f"✅ Zenoh started successfully"},
            {"text": f"🆔 Instance ID: {instance_id}"},
        ]

        # Add endpoint info if remote connections configured
        if endpoints_info:
            for info in endpoints_info:
                content.append({"text": info})
        else:
            content.append({"text": "🔍 Multicast scouting enabled (224.0.0.224:7446)"})

        content.extend(
            [
                {"text": "📡 Listening for peers..."},
                {"text": ""},
                {"text": "Commands:"},
                {"text": "  • zenoh_peer(action='list_peers') - See discovered peers"},
                {
                    "text": "  • zenoh_peer(action='broadcast', message='...') - Send to all"
                },
                {
                    "text": "  • zenoh_peer(action='send', peer_id='...', message='...') - Send to one"
                },
            ]
        )

        return {
            "status": "success",
            "content": content,
        }

    except Exception as e:
        logger.error(f"Zenoh: Failed to start: {e}")
        ZENOH_STATE["running"] = False
        return {
            "status": "error",
            "content": [{"text": f"❌ Failed to start Zenoh: {e}"}],
        }


def stop_zenoh() -> dict:
    """Stop Zenoh peer networking.

    Returns:
        Status dictionary
    """
    if not ZENOH_STATE["running"]:
        return {
            "status": "error",
            "content": [{"text": "❌ Zenoh not running"}],
        }

    try:
        ZENOH_STATE["running"] = False

        # Unsubscribe all
        for sub in ZENOH_STATE["subscribers"]:
            try:
                sub.undeclare()
            except:
                pass
        ZENOH_STATE["subscribers"] = []

        # Close session
        if ZENOH_STATE["session"]:
            ZENOH_STATE["session"].close()
            ZENOH_STATE["session"] = None

        # Clear state
        peer_count = len(ZENOH_STATE["peers"])
        ZENOH_STATE["peers"] = {}
        ZENOH_STATE["agent"] = None

        instance_id = ZENOH_STATE["instance_id"]
        ZENOH_STATE["instance_id"] = None

        # Unregister from file-based registry
        try:
            from devduck.tools.mesh_registry import registry

            if instance_id:
                registry.unregister(instance_id)
        except Exception:
            pass

        logger.info("Zenoh: Stopped")

        return {
            "status": "success",
            "content": [
                {"text": f"✅ Zenoh stopped"},
                {"text": f"🆔 Was: {instance_id}"},
                {"text": f"👥 Had {peer_count} connected peers"},
            ],
        }

    except Exception as e:
        logger.error(f"Zenoh: Error stopping: {e}")
        return {
            "status": "error",
            "content": [{"text": f"❌ Error stopping Zenoh: {e}"}],
        }


def get_zenoh_status() -> dict:
    """Get current Zenoh status.

    Returns:
        Status dictionary
    """
    if not ZENOH_STATE["running"]:
        return {
            "status": "success",
            "content": [{"text": "Zenoh not running"}],
        }

    instance_id = get_instance_id()
    peer_count = len(ZENOH_STATE["peers"])
    start_time = ZENOH_STATE.get("start_time", "unknown")

    peer_list = []
    for peer_id, info in ZENOH_STATE["peers"].items():
        age = time.time() - info["last_seen"]
        peer_list.append(f"  • {peer_id} ({info['hostname']}) - seen {age:.1f}s ago")

    content = [
        {"text": "🦆 Zenoh Status"},
        {"text": f"🆔 Instance: {instance_id}"},
        {"text": f"⏱️  Started: {start_time}"},
        {"text": f"👥 Peers: {peer_count}"},
    ]

    if peer_list:
        content.append({"text": "\nDiscovered Peers:"})
        content.append({"text": "\n".join(peer_list)})
    else:
        content.append({"text": "\nNo peers discovered yet"})

    return {
        "status": "success",
        "content": content,
    }


def list_peers() -> dict:
    """List all discovered Zenoh peers.

    Returns:
        Status dictionary with peer list
    """
    if not ZENOH_STATE["running"]:
        return {
            "status": "error",
            "content": [{"text": "❌ Zenoh not running"}],
        }

    peers = ZENOH_STATE["peers"]

    # Also get browser peers
    browser_peers = []
    try:
        from devduck.tools.agentcore_proxy import get_browser_peers

        browser_peers = get_browser_peers()
    except:
        pass

    if not peers and not browser_peers:
        return {
            "status": "success",
            "content": [
                {"text": "No peers discovered yet"},
                {
                    "text": "💡 Start another DevDuck instance with Zenoh, or open one.html to see browser peers"
                },
            ],
        }

    peer_info = []
    for peer_id, info in peers.items():
        age = time.time() - info["last_seen"]
        peer_info.append(
            {
                "id": peer_id,
                "hostname": info.get("hostname", "unknown"),
                "model": info.get("model", "unknown"),
                "last_seen": f"{age:.1f}s ago",
                "type": "zenoh",
            }
        )

    for bp in browser_peers:
        peer_info.append(
            {
                "id": bp["id"],
                "hostname": "browser",
                "model": bp.get("model", "browser-agent"),
                "last_seen": "connected",
                "type": "browser",
            }
        )

    content = [{"text": f"👥 Discovered Peers ({len(peer_info)}):"}]
    for p in peer_info:
        icon = "🦆" if p["type"] == "zenoh" else "🧠"
        content.append(
            {
                "text": f"\n  {icon} {p['id']}\n     Host: {p['hostname']}\n     Model: {p['model']}\n     Seen: {p['last_seen']}\n     Type: {p['type']}"
            }
        )

    return {
        "status": "success",
        "content": content,
    }


def broadcast_message(message: str, wait_time: float = 60.0) -> dict:
    """Broadcast a command to ALL connected DevDuck peers.

    Args:
        message: The command/message to send
        wait_time: Maximum time to wait for responses (seconds, default: 60)

    Returns:
        Status dictionary with collected responses
    """
    if not ZENOH_STATE["running"]:
        return {
            "status": "error",
            "content": [{"text": "❌ Zenoh not running"}],
        }

    if not ZENOH_STATE["peers"]:
        return {
            "status": "error",
            "content": [
                {
                    "text": "❌ No peers discovered. Start another DevDuck instance first."
                }
            ],
        }

    turn_id = uuid.uuid4().hex[:8]
    instance_id = get_instance_id()
    peer_count = len(ZENOH_STATE["peers"])

    # Prepare for responses - use threading.Event for completion signal
    completion_event = threading.Event()
    ZENOH_STATE["pending_responses"][turn_id] = completion_event
    ZENOH_STATE["collected_responses"][turn_id] = []

    # Broadcast the command
    command_data = {
        "sender_id": instance_id,
        "turn_id": turn_id,
        "command": message,
        "timestamp": time.time(),
    }
    publish_message("devduck/broadcast", command_data)

    logger.info(
        f"Zenoh: Broadcast '{message[:50]}...' to {peer_count} peers (turn: {turn_id})"
    )

    # Wait for responses - could be multiple, so wait for timeout or all peers
    # For broadcast, wait for at least one response or timeout
    completed = completion_event.wait(timeout=wait_time)

    if not completed:
        logger.warning(
            f"Zenoh: Broadcast timeout after {wait_time}s for turn {turn_id}"
        )

    # Collect responses
    responses = ZENOH_STATE["collected_responses"].get(turn_id, [])

    # Get streamed content
    streamed = ZENOH_STATE["streamed_content"].get(turn_id, {})

    # Cleanup
    del ZENOH_STATE["pending_responses"][turn_id]
    if turn_id in ZENOH_STATE["collected_responses"]:
        del ZENOH_STATE["collected_responses"][turn_id]
    if turn_id in ZENOH_STATE["streamed_content"]:
        del ZENOH_STATE["streamed_content"][turn_id]

    content = [
        {"text": f"📢 Broadcast sent to {peer_count} peers"},
        {"text": f"💬 Message: {message}"},
        {"text": f"⏱️  Waited: {wait_time}s"},
        {"text": f"📥 Responses: {len(responses)}, Streamed: {len(streamed)}"},
    ]

    # Include streamed content first (real-time responses)
    if streamed:
        for responder, text in streamed.items():
            content.append({"text": f"\n🦆 {responder} (streamed):\n{text}"})

    # Then include formal responses
    for resp in responses:
        resp_type = resp.get("type", "unknown")
        responder = resp.get("responder", "unknown")

        if resp_type == "response":
            result = resp.get("result", "")[:500]  # Truncate long responses
            content.append({"text": f"\n🦆 {responder}:\n{result}"})
        elif resp_type == "error":
            error = resp.get("error", "unknown error")
            content.append({"text": f"\n❌ {responder}: {error}"})
        elif resp_type == "ack":
            content.append({"text": f"\n✓ {responder}: acknowledged"})

    return {
        "status": "success",
        "content": content,
    }


def send_to_peer(peer_id: str, message: str, wait_time: float = 120.0) -> dict:
    """Send a command to a specific DevDuck peer.

    Args:
        peer_id: The target peer's instance ID
        message: The command/message to send
        wait_time: Maximum time to wait for response (seconds, default: 120)

    Returns:
        Status dictionary with response
    """
    if not ZENOH_STATE["running"]:
        return {
            "status": "error",
            "content": [{"text": "❌ Zenoh not running"}],
        }

    # Check if target is a browser peer (browser:ws_id format)
    if peer_id.startswith("browser:"):
        ws_id = peer_id.replace("browser:", "")
        try:
            from devduck.tools.agentcore_proxy import send_to_browser_peer

            result = send_to_browser_peer(
                ws_id=ws_id,
                prompt=message,
                zenoh_requester_id=get_instance_id(),
            )
            return result
        except ImportError:
            return {
                "status": "error",
                "content": [
                    {"text": "agentcore_proxy not available for browser peer routing"}
                ],
            }
        except Exception as e:
            return {
                "status": "error",
                "content": [{"text": f"Browser peer error: {e}"}],
            }

    # Check if sending to SELF — process locally instead of via pubsub
    if peer_id == get_instance_id():
        try:
            agent = ZENOH_STATE.get("agent")
            if agent:
                result = agent(message)
                return {
                    "status": "success",
                    "content": [{"text": f"\n📥 Response from self:\n{str(result)}"}],
                }
            else:
                return {
                    "status": "error",
                    "content": [{"text": "❌ No agent available for self-processing"}],
                }
        except Exception as e:
            return {
                "status": "error",
                "content": [{"text": f"❌ Self-processing error: {e}"}],
            }

    if peer_id not in ZENOH_STATE["peers"]:
        available = list(ZENOH_STATE["peers"].keys())
        # Also check browser peers
        browser_peers = []
        try:
            from devduck.tools.agentcore_proxy import _GATEWAY_STATE

            browser_peers = [
                f"browser:{ws_id}"
                for ws_id in _GATEWAY_STATE.get("browser_peers", {}).keys()
            ]
        except:
            pass
        return {
            "status": "error",
            "content": [
                {"text": f"❌ Peer '{peer_id}' not found"},
                {"text": f"Available zenoh peers: {available}"},
                (
                    {"text": f"Available browser peers: {browser_peers}"}
                    if browser_peers
                    else {"text": "No browser peers"}
                ),
            ],
        }

    turn_id = uuid.uuid4().hex[:8]
    instance_id = get_instance_id()

    # Prepare for response - use threading.Event for completion signal
    completion_event = threading.Event()
    ZENOH_STATE["pending_responses"][turn_id] = completion_event
    ZENOH_STATE["collected_responses"][turn_id] = []

    # Send direct command
    command_data = {
        "sender_id": instance_id,
        "turn_id": turn_id,
        "command": message,
        "timestamp": time.time(),
    }
    publish_message(f"devduck/cmd/{peer_id}", command_data)

    logger.info(f"Zenoh: Sent '{message[:50]}...' to {peer_id} (turn: {turn_id})")

    # Wait for completion signal OR timeout
    # This waits until handle_response sets the event (on "turn_end" or "error")
    completed = completion_event.wait(timeout=wait_time)

    if not completed:
        logger.warning(f"Zenoh: Response timeout after {wait_time}s for turn {turn_id}")

    # Get response
    responses = ZENOH_STATE["collected_responses"].get(turn_id, [])

    # Get streamed content
    streamed = ZENOH_STATE["streamed_content"].get(turn_id, {})

    # Cleanup
    del ZENOH_STATE["pending_responses"][turn_id]
    if turn_id in ZENOH_STATE["collected_responses"]:
        del ZENOH_STATE["collected_responses"][turn_id]
    if turn_id in ZENOH_STATE["streamed_content"]:
        del ZENOH_STATE["streamed_content"][turn_id]

    content = [
        {"text": f"📨 Sent to: {peer_id}"},
        {"text": f"💬 Message: {message}"},
    ]

    # Include streamed content in response
    if streamed:
        for responder, text in streamed.items():
            content.append({"text": f"\n📥 Streamed from {responder}:\n{text}"})
    elif responses:
        for resp in responses:
            resp_type = resp.get("type", "unknown")
            if resp_type == "response":
                result = resp.get("result", "")
                content.append({"text": f"\n📥 Response:\n{result}"})
            elif resp_type == "error":
                error = resp.get("error", "unknown error")
                content.append({"text": f"\n❌ Error: {error}"})
    else:
        content.append(
            {"text": "\n⏱️ No response received (peer may be busy or timed out)"}
        )

    return {
        "status": "success",
        "content": content,
    }


@tool
def zenoh_peer(
    action: str,
    message: str = "",
    peer_id: str = "",
    wait_time: float = 120.0,
    connect: str = "",
    listen: str = "",
    agent=None,
) -> dict:
    """Zenoh peer-to-peer networking for DevDuck auto-discovery and communication.

    This tool enables multiple DevDuck instances to automatically discover each other
    and communicate using Zenoh's multicast scouting. No manual configuration needed -
    just start Zenoh on multiple terminals and they find each other!

    How It Works:
    ------------
    1. Each DevDuck instance joins a Zenoh peer network
    2. Multicast scouting (224.0.0.224:7446) auto-discovers peers on local network
    3. Peers exchange heartbeats to maintain presence awareness
    4. Commands can be broadcast to ALL peers or sent to specific peers
    5. Responses stream back from all responding peers

    Remote Connections:
    ------------------
    To connect DevDuck instances across different networks:
    - Use 'connect' to specify remote peer/router endpoints
    - Use 'listen' to accept incoming remote connections
    - Or set ZENOH_CONNECT / ZENOH_LISTEN environment variables

    Use Cases:
    ---------
    - Multi-terminal coordination: "zenoh broadcast 'git pull && npm install'"
    - Distributed task execution: One command triggers all instances
    - Peer monitoring: See all active DevDuck instances
    - Direct messaging: Send specific tasks to specific instances
    - Cross-network collaboration: Connect home and office DevDucks

    Args:
        action: Action to perform:
            - "start": Start Zenoh networking (auto-joins peer mesh)
            - "stop": Stop Zenoh networking
            - "status": Show current status and peer count
            - "list_peers": List all discovered peers
            - "broadcast": Send command to ALL peers
            - "send": Send command to specific peer
        message: Command/message to send (for broadcast/send actions)
        peer_id: Target peer ID (for send action)
        wait_time: Seconds to wait for responses (default: 5.0)
        connect: Remote endpoint(s) to connect to (e.g., "tcp/1.2.3.4:7447")
        listen: Endpoint(s) to listen on for remote connections (e.g., "tcp/0.0.0.0:7447")
        agent: DevDuck agent instance (passed automatically on start)

    Returns:
        Dictionary containing status and response content

    Examples:
        # Terminal 1: Start Zenoh (local network only)
        zenoh_peer(action="start")

        # Terminal 2: Start Zenoh (auto-discovers Terminal 1)
        zenoh_peer(action="start")

        # Start with remote connection (connect to peer at home)
        zenoh_peer(action="start", connect="tcp/home.example.com:7447")

        # Start listening for remote connections
        zenoh_peer(action="start", listen="tcp/0.0.0.0:7447")

        # Terminal 1: See peers
        zenoh_peer(action="list_peers")
        # Shows: Terminal 2's instance

        # Terminal 1: Broadcast to all
        zenoh_peer(action="broadcast", message="echo 'Hello from all DevDucks!'")
        # Terminal 2 executes the command and responds

        # Send to specific peer
        zenoh_peer(action="send", peer_id="hostname-abc123", message="what files are here?")

    Environment:
        DEVDUCK_ENABLE_ZENOH=true  - Auto-start Zenoh on DevDuck launch
        ZENOH_CONNECT=tcp/1.2.3.4:7447  - Auto-connect to remote endpoint
        ZENOH_LISTEN=tcp/0.0.0.0:7447   - Auto-listen for remote connections
    """
    if action == "start":
        # Get model info if agent provided
        model = "unknown"
        if agent and hasattr(agent, "model"):
            agent_model = getattr(agent, "model", None)
            if agent_model:
                # Try to get model_id attribute (most model providers have this)
                model = (
                    getattr(agent_model, "model_id", None)
                    or getattr(agent_model, "model_name", None)
                    or getattr(agent_model, "name", None)
                    or type(agent_model).__name__
                )
        return start_zenoh(
            agent=agent,
            model=model,
            connect=connect if connect else None,
            listen=listen if listen else None,
        )

    elif action == "stop":
        return stop_zenoh()

    elif action == "status":
        return get_zenoh_status()

    elif action == "list_peers":
        return list_peers()

    elif action == "broadcast":
        if not message:
            return {
                "status": "error",
                "content": [{"text": "❌ message parameter required for broadcast"}],
            }
        return broadcast_message(message, wait_time)

    elif action == "send":
        if not peer_id:
            return {
                "status": "error",
                "content": [{"text": "❌ peer_id parameter required for send"}],
            }
        if not message:
            return {
                "status": "error",
                "content": [{"text": "❌ message parameter required for send"}],
            }
        return send_to_peer(peer_id, message, wait_time)

    else:
        return {
            "status": "error",
            "content": [
                {"text": f"❌ Unknown action: {action}"},
                {
                    "text": "Valid actions: start, stop, status, list_peers, broadcast, send"
                },
            ],
        }
