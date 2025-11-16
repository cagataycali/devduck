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
5. Per-Connection DevDuck: Creates a fresh DevDuck instance for each client connection
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


async def process_message_async(connection_agent, message, websocket, loop, turn_id):
    """Process a message in a concurrent task.

    Args:
        connection_agent: The agent instance to process the message
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

        # Create callback handler for this turn
        streaming_handler = WebSocketStreamingCallbackHandler(websocket, loop, turn_id)
        connection_agent.callback_handler = streaming_handler

        # Process message in a thread to avoid blocking the event loop
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, connection_agent, message)

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


async def handle_websocket_client(websocket, system_prompt: str):
    """Handle a WebSocket client connection with streaming responses.

    Args:
        websocket: WebSocket connection object
        system_prompt: System prompt for the DevDuck agent
    """
    client_address = websocket.remote_address
    logger.info(f"WebSocket connection established with {client_address}")

    # Get the current event loop
    loop = asyncio.get_running_loop()

    # Import DevDuck and create a new instance for this connection
    try:
        from devduck import DevDuck

        # Create a new DevDuck instance with auto_start_servers=False to avoid recursion
        connection_devduck = DevDuck(auto_start_servers=False)

        # Override system prompt if provided
        if connection_devduck.agent and system_prompt:
            connection_devduck.agent.system_prompt += (
                "\nCustom system prompt:" + system_prompt
            )

        connection_agent = connection_devduck.agent

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

        connection_agent = Agent(
            model=agent_model,
            tools=[],
            system_prompt=system_prompt
            or "You are a helpful WebSocket server assistant.",
        )

    # Track active tasks for concurrent processing
    active_tasks = set()

    try:
        # Send welcome message
        welcome = {
            "type": "connected",
            "data": "🦆 Welcome to DevDuck!",
            "timestamp": time.time(),
        }
        await websocket.send(json.dumps(welcome))

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

            # Generate unique turn ID for this conversation turn
            turn_id = str(uuid.uuid4())

            # Launch message processing as concurrent task (don't await)
            task = asyncio.create_task(
                process_message_async(
                    connection_agent, message, websocket, loop, turn_id
                )
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
                {
                    "text": "🦆 Server creates a new DevDuck instance for each connection"
                },
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
