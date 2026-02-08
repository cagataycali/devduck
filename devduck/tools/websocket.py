"""WebSocket tool for DevDuck agents with real-time streaming support.

This module provides WebSocket server functionality for DevDuck agents,
allowing them to communicate over WebSocket protocol with real-time response streaming.
The tool runs server operations in background threads, enabling concurrent
communication without blocking the main agent.

Key Features:
1. WebSocket Server: Listen for incoming connections and process them with a DevDuck agent
2. Real-time Streaming: Responses stream to clients as they're generated (non-blocking)
3. Concurrent Processing: Handle multiple messages simultaneously
4. Background Processing: Server runs in a background thread
5. Per-Message DevDuck: Creates a fresh DevDuck instance for EACH message to avoid concurrency errors
6. Callback Handler: Uses Strands callback system for efficient streaming
7. Browser Compatible: Works with browser WebSocket clients

Message Format:
```json
{
  "type": "turn_start" | "chunk" | "tool_start" | "tool_end" | "turn_end",
  "turn_id": "uuid",
  "data": "text content",
  "timestamp": 1234567890.123
}
```

Usage with DevDuck Agent:

```python
from devduck import devduck

# Start a streaming WebSocket server
result = devduck.agent.tool.websocket(
    action="start_server",
    host="127.0.0.1",
    port=8080,
    system_prompt="You are a helpful WebSocket server assistant.",
)

# Stop the WebSocket server
result = devduck.agent.tool.websocket(action="stop_server", port=8080)
```

For testing with browser:
```javascript
const ws = new WebSocket('ws://localhost:8080');
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(`[${msg.turn_id}] ${msg.type}: ${msg.data}`);
};
ws.send('Hello DevDuck!');
```
"""

import logging
import threading
import time
import os
import asyncio
import json
import uuid
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from strands import Agent, tool

logger = logging.getLogger(__name__)

# Global registry to store server threads
WS_SERVER_THREADS: dict[int, dict[str, Any]] = {}


class WebSocketStreamingCallbackHandler:
    """Callback handler that streams agent responses directly over WebSocket with turn tracking."""

    def __init__(self, websocket, loop, turn_id: str):
        """Initialize the streaming handler.

        Args:
            websocket: The WebSocket connection to stream data to
            loop: The event loop to use for async operations
            turn_id: Unique identifier for this conversation turn
        """
        self.websocket = websocket
        self.loop = loop
        self.turn_id = turn_id
        self.tool_count = 0
        self.previous_tool_use = None

    async def _send_message(
        self, msg_type: str, data: str = "", metadata: dict = None
    ) -> None:
        """Send a structured message over WebSocket.

        Args:
            msg_type: Message type (turn_start, chunk, tool_start, tool_end, turn_end)
            data: Text content
            metadata: Additional metadata
        """
        try:
            message = {
                "type": msg_type,
                "turn_id": self.turn_id,
                "data": data,
                "timestamp": time.time(),
            }
            if metadata:
                message.update(metadata)

            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to send message over WebSocket: {e}")

    def _schedule_message(
        self, msg_type: str, data: str = "", metadata: dict = None
    ) -> None:
        """Schedule an async message send from sync context.

        Args:
            msg_type: Message type
            data: Text content
            metadata: Additional metadata
        """
        asyncio.run_coroutine_threadsafe(
            self._send_message(msg_type, data, metadata), self.loop
        )

    def __call__(self, **kwargs: Any) -> None:
        """Stream events to WebSocket in real-time with turn tracking."""
        reasoningText = kwargs.get("reasoningText", False)
        data = kwargs.get("data", "")
        complete = kwargs.get("complete", False)
        current_tool_use = kwargs.get("current_tool_use", {})
        message = kwargs.get("message", {})

        # Stream reasoning text
        if reasoningText:
            self._schedule_message("chunk", reasoningText, {"reasoning": True})

        # Stream response text chunks
        if data:
            self._schedule_message("chunk", data)

        # Stream tool invocation notifications
        if current_tool_use and current_tool_use.get("name"):
            tool_name = current_tool_use.get("name", "Unknown tool")
            if self.previous_tool_use != current_tool_use:
                self.previous_tool_use = current_tool_use
                self.tool_count += 1
                self._schedule_message(
                    "tool_start", tool_name, {"tool_number": self.tool_count}
                )

        # Stream tool results
        if isinstance(message, dict) and message.get("role") == "user":
            for content in message.get("content", []):
                if isinstance(content, dict):
                    tool_result = content.get("toolResult")
                    if tool_result:
                        status = tool_result.get("status", "unknown")
                        self._schedule_message(
                            "tool_end", status, {"success": status == "success"}
                        )


async def process_message_async(system_prompt, message, websocket, loop, turn_id):
    """Process a message in a concurrent task.

    Creates a NEW DevDuck instance for each message to avoid concurrent
    invocation errors (Strands Agent doesn't support concurrent requests).

    Args:
        system_prompt: System prompt for the DevDuck agent
        message: The message to process
        websocket: WebSocket connection
        loop: Event loop
        turn_id: Unique turn ID
    """
    try:
        # Send turn start notification
        turn_start = {
            "type": "turn_start",
            "turn_id": turn_id,
            "data": message,
            "timestamp": time.time(),
        }
        await websocket.send(json.dumps(turn_start))

        # Create a NEW DevDuck instance for THIS message
        # This avoids concurrent invocation errors on shared agent instances
        try:
            from devduck import DevDuck

            # Create a new DevDuck instance with auto_start_servers=False to avoid recursion
            message_devduck = DevDuck(auto_start_servers=False)

            # Override system prompt if provided
            if message_devduck.agent and system_prompt:
                message_devduck.agent.system_prompt += (
                    "\nCustom system prompt:" + system_prompt
                )

            message_agent = message_devduck.agent

        except Exception as e:
            logger.error(f"Failed to create DevDuck instance: {e}", exc_info=True)
            # Fallback to basic Agent if DevDuck fails
            from strands import Agent
            from strands.models.ollama import OllamaModel

            agent_model = OllamaModel(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                model_id=os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
                temperature=1,
                keep_alive="5m",
            )

            message_agent = Agent(
                model=agent_model,
                tools=[],
                system_prompt=system_prompt
                or "You are a helpful WebSocket server assistant.",
            )

        # Create callback handler for this turn
        streaming_handler = WebSocketStreamingCallbackHandler(websocket, loop, turn_id)
        message_agent.callback_handler = streaming_handler

        # Process message in a thread to avoid blocking the event loop
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, message_agent, message)

        # Send turn end notification
        turn_end = {"type": "turn_end", "turn_id": turn_id, "timestamp": time.time()}
        await websocket.send(json.dumps(turn_end))

    except Exception as e:
        logger.error(f"Error processing message in turn {turn_id}: {e}", exc_info=True)
        error_msg = {
            "type": "error",
            "turn_id": turn_id,
            "data": f"Error processing message: {e}",
            "timestamp": time.time(),
        }
        await websocket.send(json.dumps(error_msg))


async def push_zenoh_updates(websocket, loop):
    """Background task that pushes Zenoh peer updates to the browser.

    This bridges the Zenoh P2P network to WebSocket clients,
    enabling browsers to see all connected DevDuck terminals.

    Args:
        websocket: WebSocket connection to push updates to
        loop: Event loop for async operations
    """
    last_peers = set()
    first_run = True  # Send immediate update on first run

    while True:
        try:
            if not first_run:
                await asyncio.sleep(2)  # Check every 2 seconds after first run
            first_run = False

            try:
                from devduck.tools.zenoh_peer import ZENOH_STATE, get_instance_id

                if not ZENOH_STATE.get("running"):
                    continue

                current_peers = set(ZENOH_STATE.get("peers", {}).keys())

                # Detect changes OR force send on first check with peers
                new_peers = current_peers - last_peers
                lost_peers = last_peers - current_peers
                force_send = (
                    last_peers == set() and current_peers
                )  # First time with peers

                if new_peers or lost_peers or force_send:
                    # Send peer update
                    update = {
                        "type": "zenoh_peers_update",
                        "instance_id": get_instance_id(),
                        "peers": [
                            {
                                "id": pid,
                                "hostname": ZENOH_STATE["peers"][pid].get(
                                    "hostname", "unknown"
                                ),
                                "model": ZENOH_STATE["peers"][pid].get(
                                    "model", "unknown"
                                ),
                                "last_seen": ZENOH_STATE["peers"][pid].get(
                                    "last_seen", 0
                                ),
                            }
                            for pid in current_peers
                        ],
                        "new_peers": list(new_peers),
                        "lost_peers": list(lost_peers),
                        "timestamp": time.time(),
                    }

                    await websocket.send(json.dumps(update))
                    logger.debug(
                        f"Pushed Zenoh update: +{len(new_peers)} -{len(lost_peers)} peers (total: {len(current_peers)})"
                    )

                    last_peers = current_peers

            except ImportError:
                pass

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug(f"Zenoh push error: {e}")
            await asyncio.sleep(5)


async def handle_websocket_client(websocket, system_prompt: str):
    """Handle a WebSocket client connection with streaming responses.

    Each message creates a NEW DevDuck instance to avoid concurrent invocation errors.
    This follows the same pattern as zenoh_peer.py.

    Also bridges Zenoh peer updates to browser clients for unified mesh.

    Args:
        websocket: WebSocket connection object
        system_prompt: System prompt for the DevDuck agent instances
    """
    client_address = websocket.remote_address
    logger.info(f"WebSocket connection established with {client_address}")

    # Get the current event loop
    loop = asyncio.get_running_loop()

    # Track active tasks for concurrent processing
    active_tasks = set()

    # Register this client for mesh broadcasts
    ws_id = str(uuid.uuid4())[:8]
    try:
        from devduck.tools.unified_mesh import MESH_STATE

        MESH_STATE["ws_clients"][ws_id] = websocket
    except ImportError:
        pass

    try:
        # Send welcome message with mesh info
        welcome = {
            "type": "connected",
            "data": "🦆 Welcome to DevDuck!",
            "ws_id": ws_id,
            "timestamp": time.time(),
        }

        # Include Zenoh peers in welcome message (full peer objects for mesh.html)
        try:
            from devduck.tools.zenoh_peer import ZENOH_STATE, get_instance_id

            if ZENOH_STATE.get("running"):
                peers_dict = ZENOH_STATE.get("peers", {})
                welcome["zenoh"] = {
                    "instance_id": get_instance_id(),
                    "peers": [
                        {
                            "id": pid,
                            "hostname": peers_dict[pid].get("hostname", "unknown"),
                            "model": peers_dict[pid].get("model", "unknown"),
                            "last_seen": peers_dict[pid].get("last_seen", 0),
                        }
                        for pid in peers_dict.keys()
                    ],
                    "peer_count": len(peers_dict),
                }
        except ImportError:
            pass

        await websocket.send(json.dumps(welcome))

        # Start background task to push Zenoh updates
        zenoh_update_task = asyncio.create_task(push_zenoh_updates(websocket, loop))
        active_tasks.add(zenoh_update_task)
        zenoh_update_task.add_done_callback(active_tasks.discard)

        async for message in websocket:
            message = message.strip()
            logger.info(f"Received from {client_address}: {message}")

            if message.lower() == "exit":
                bye = {
                    "type": "disconnected",
                    "data": "Connection closed by client request.",
                    "timestamp": time.time(),
                }
                await websocket.send(json.dumps(bye))
                logger.info(f"Client {client_address} requested to exit")
                break

            # Filter out relay/mesh protocol messages (meant for agentcore_proxy, not raw WS)
            # These come from mesh.html auto-discovery hitting the wrong port
            try:
                parsed = json.loads(message)
                if isinstance(parsed, dict) and parsed.get("type") in (
                    "configure",
                    "list_agents",
                    "get_ring",
                    "trigger_github",
                    "invoke",
                    "broadcast",
                    "list_peers",
                    "get_status",
                ):
                    logger.info(
                        f"Ignoring relay protocol message: {parsed.get('type')}"
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "data": f"This is a raw DevDuck WebSocket server, not a relay. Message type '{parsed.get('type')}' is not supported here.",
                                "timestamp": time.time(),
                            }
                        )
                    )
                    continue
            except (json.JSONDecodeError, ValueError):
                pass  # Not JSON - treat as normal text prompt

            # Generate unique turn ID for this conversation turn
            turn_id = str(uuid.uuid4())

            # Launch message processing as concurrent task (don't await)
            # Each task creates its OWN DevDuck instance to avoid concurrency issues
            task = asyncio.create_task(
                process_message_async(system_prompt, message, websocket, loop, turn_id)
            )
            active_tasks.add(task)

            # Clean up completed tasks
            task.add_done_callback(active_tasks.discard)

        # Wait for all active tasks to complete before closing
        if active_tasks:
            logger.info(f"Waiting for {len(active_tasks)} active tasks to complete...")
            await asyncio.gather(*active_tasks, return_exceptions=True)

    except Exception as e:
        logger.error(
            f"Error handling WebSocket client {client_address}: {e}", exc_info=True
        )
    finally:
        # Unregister from mesh
        try:
            from devduck.tools.unified_mesh import MESH_STATE

            if ws_id in MESH_STATE["ws_clients"]:
                del MESH_STATE["ws_clients"][ws_id]
        except:
            pass
        logger.info(f"WebSocket connection with {client_address} closed")


def run_websocket_server(
    host: str,
    port: int,
    system_prompt: str,
) -> None:
    """Run a WebSocket server that processes client requests with DevDuck instances."""
    import websockets

    WS_SERVER_THREADS[port]["running"] = True
    WS_SERVER_THREADS[port]["connections"] = 0
    WS_SERVER_THREADS[port]["start_time"] = time.time()

    async def server_handler(websocket):
        """Handle incoming WebSocket connections.

        Args:
            websocket: WebSocket connection object
        """
        WS_SERVER_THREADS[port]["connections"] += 1
        await handle_websocket_client(websocket, system_prompt)

    async def start_server():
        stop_future = asyncio.Future()
        WS_SERVER_THREADS[port]["stop_future"] = stop_future

        server = await websockets.serve(server_handler, host, port)
        logger.info(f"WebSocket Server listening on {host}:{port}")

        # Wait for stop signal
        await stop_future

        # Close the server
        server.close()
        await server.wait_closed()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        WS_SERVER_THREADS[port]["loop"] = loop
        loop.run_until_complete(start_server())
    except OSError as e:
        # Port conflict - handled upstream, no need for scary errors
        if "Address already in use" in str(e) or "address already in use" in str(e):
            logger.debug(f"Port {port} unavailable (handled upstream)")
        else:
            logger.error(f"WebSocket server error on {host}:{port}: {e}")
    except Exception as e:
        logger.error(f"WebSocket server error on {host}:{port}: {e}")
    finally:
        logger.info(f"WebSocket Server on {host}:{port} stopped")
        WS_SERVER_THREADS[port]["running"] = False


@tool
def websocket(
    action: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    system_prompt: str = "You are a helpful WebSocket server assistant.",
) -> dict:
    """Create and manage WebSocket servers with real-time streaming.

    Args:
        action: Action to perform (start_server, stop_server, get_status)
        host: Host address for server
        port: Port number for server
        system_prompt: System prompt for the server DevDuck instances

    Returns:
        Dictionary containing status and response content
    """
    if action == "start_server":
        if port in WS_SERVER_THREADS and WS_SERVER_THREADS[port].get("running", False):
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"❌ Error: WebSocket Server already running on port {port}"
                    }
                ],
            }

        WS_SERVER_THREADS[port] = {"running": False}
        server_thread = threading.Thread(
            target=run_websocket_server,
            args=(host, port, system_prompt),
        )
        server_thread.daemon = True
        server_thread.start()

        time.sleep(0.5)

        if not WS_SERVER_THREADS[port].get("running", False):
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"❌ Error: Failed to start WebSocket Server on {host}:{port}"
                    }
                ],
            }

        return {
            "status": "success",
            "content": [
                {"text": f"✅ WebSocket Server started successfully on {host}:{port}"},
                {"text": f"System prompt: {system_prompt}"},
                {"text": "🌊 Real-time streaming with concurrent message processing"},
                {"text": "📦 Structured JSON messages with turn_id"},
                {"text": "🦆 Server creates a new DevDuck instance for each message"},
                {"text": "⚡ Send multiple messages without waiting!"},
                {"text": f"📝 Test with: ws://localhost:{port}"},
            ],
        }

    elif action == "stop_server":
        if port not in WS_SERVER_THREADS or not WS_SERVER_THREADS[port].get(
            "running", False
        ):
            return {
                "status": "error",
                "content": [
                    {"text": f"❌ Error: No WebSocket Server running on port {port}"}
                ],
            }

        WS_SERVER_THREADS[port]["running"] = False

        # Signal the server to stop
        if "stop_future" in WS_SERVER_THREADS[port]:
            loop = WS_SERVER_THREADS[port]["loop"]
            loop.call_soon_threadsafe(
                lambda: WS_SERVER_THREADS[port]["stop_future"].set_result(None)
            )

        time.sleep(1.0)

        connections = WS_SERVER_THREADS[port].get("connections", 0)
        uptime = time.time() - WS_SERVER_THREADS[port].get("start_time", time.time())

        del WS_SERVER_THREADS[port]

        return {
            "status": "success",
            "content": [
                {"text": f"✅ WebSocket Server on port {port} stopped successfully"},
                {
                    "text": f"Statistics: {connections} connections handled, uptime {uptime:.2f} seconds"
                },
            ],
        }

    elif action == "get_status":
        if not WS_SERVER_THREADS:
            return {
                "status": "success",
                "content": [{"text": "No WebSocket Servers running"}],
            }

        status_info = []
        for port, data in WS_SERVER_THREADS.items():
            if data.get("running", False):
                uptime = time.time() - data.get("start_time", time.time())
                connections = data.get("connections", 0)
                status_info.append(
                    f"Port {port}: Running - {connections} connections, uptime {uptime:.2f}s"
                )
            else:
                status_info.append(f"Port {port}: Stopped")

        return {
            "status": "success",
            "content": [
                {"text": "WebSocket Server Status:"},
                {"text": "\n".join(status_info)},
            ],
        }

    else:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"Error: Unknown action '{action}'. Supported: start_server, stop_server, get_status"
                }
            ],
        }
