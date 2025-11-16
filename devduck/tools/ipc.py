"""IPC tool for DevDuck agents with real-time streaming support.

This module provides Unix socket IPC server functionality for DevDuck agents,
allowing local processes (tray, ambient, etc.) to communicate with real-time streaming.
Similar to websocket.py but uses Unix sockets for inter-process communication.

Key Features:
1. IPC Server: Listen on Unix socket for local process connections
2. Real-time Streaming: Responses stream to clients as they're generated
3. Concurrent Processing: Handle multiple connections simultaneously
4. Background Processing: Server runs in a background thread
5. Per-Connection DevDuck: Creates a fresh DevDuck instance for each connection
6. Callback Handler: Uses Strands callback system for efficient streaming
7. Bidirectional: Clients can send commands AND receive streaming responses

Message Format:
```json
{
  "type": "turn_start" | "chunk" | "tool_start" | "tool_end" | "turn_end" | "command",
  "turn_id": "uuid",
  "data": "text content",
  "timestamp": 1234567890.123,
  "command": "optional_command_name",
  "params": {"optional": "parameters"}
}
```

Usage with DevDuck Agent:

```python
from devduck import devduck

# Start IPC server
result = devduck.agent.tool.ipc(
    action="start_server",
    socket_path="/tmp/devduck_main.sock",
    system_prompt="You are a helpful IPC server assistant.",
)

# Stop IPC server
result = devduck.agent.tool.ipc(
    action="stop_server",
    socket_path="/tmp/devduck_main.sock"
)
```

Client Example (Python):
```python
import socket
import json

# Connect to IPC server
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/tmp/devduck_main.sock")

# Send message
message = json.dumps({"message": "Hello DevDuck!", "turn_id": "123"})
sock.sendall(message.encode() + b'\n')

# Receive streaming response
buffer = b''
while True:
    chunk = sock.recv(4096)
    if not chunk:
        break
    buffer += chunk
    # Process complete JSON messages (newline delimited)
    while b'\n' in buffer:
        line, buffer = buffer.split(b'\n', 1)
        msg = json.loads(line.decode())
        print(f"[{msg['type']}] {msg.get('data', '')}")
```
"""

import logging
import threading
import time
import os
import asyncio
import json
import uuid
import tempfile
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from strands import Agent, tool

logger = logging.getLogger(__name__)

# Global registry to store server threads
IPC_SERVER_THREADS: dict[str, dict[str, Any]] = {}


class IPCStreamingCallbackHandler:
    """Callback handler that streams agent responses directly over Unix socket with turn tracking."""

    def __init__(self, client_socket, turn_id: str):
        """Initialize the streaming handler.

        Args:
            client_socket: The Unix socket connection to stream data to
            turn_id: Unique identifier for this conversation turn
        """
        self.socket = client_socket
        self.turn_id = turn_id
        self.tool_count = 0
        self.previous_tool_use = None

    def _send_message(
        self, msg_type: str, data: str = "", metadata: dict = None
    ) -> None:
        """Send a structured message over Unix socket.

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

            # Send as newline-delimited JSON for easy parsing
            self.socket.sendall(json.dumps(message).encode() + b"\n")
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning(f"Failed to send message over IPC: {e}")

    def __call__(self, **kwargs: Any) -> None:
        """Stream events to Unix socket in real-time with turn tracking."""
        reasoningText = kwargs.get("reasoningText", False)
        data = kwargs.get("data", "")
        complete = kwargs.get("complete", False)
        current_tool_use = kwargs.get("current_tool_use", {})
        message = kwargs.get("message", {})

        # Stream reasoning text
        if reasoningText:
            self._send_message("chunk", reasoningText, {"reasoning": True})

        # Stream response text chunks
        if data:
            self._send_message("chunk", data)

        # Stream tool invocation notifications
        if current_tool_use and current_tool_use.get("name"):
            tool_name = current_tool_use.get("name", "Unknown tool")
            if self.previous_tool_use != current_tool_use:
                self.previous_tool_use = current_tool_use
                self.tool_count += 1
                self._send_message(
                    "tool_start", tool_name, {"tool_number": self.tool_count}
                )

        # Stream tool results
        if isinstance(message, dict) and message.get("role") == "user":
            for content in message.get("content", []):
                if isinstance(content, dict):
                    tool_result = content.get("toolResult")
                    if tool_result:
                        status = tool_result.get("status", "unknown")
                        self._send_message(
                            "tool_end", status, {"success": status == "success"}
                        )


def process_ipc_message(connection_agent, message_data, client_socket, turn_id):
    """Process an IPC message and stream response.

    Args:
        connection_agent: The agent instance to process the message
        message_data: Parsed message data
        client_socket: Unix socket connection
        turn_id: Unique turn ID
    """
    try:
        message = message_data.get("message", "")

        # Send turn start notification
        turn_start = {
            "type": "turn_start",
            "turn_id": turn_id,
            "data": message,
            "timestamp": time.time(),
        }
        client_socket.sendall(json.dumps(turn_start).encode() + b"\n")

        # Create callback handler for this turn
        streaming_handler = IPCStreamingCallbackHandler(client_socket, turn_id)
        connection_agent.callback_handler = streaming_handler

        # Process message (synchronous, runs in thread pool)
        connection_agent(message)

        # Send turn end notification
        turn_end = {"type": "turn_end", "turn_id": turn_id, "timestamp": time.time()}
        client_socket.sendall(json.dumps(turn_end).encode() + b"\n")

    except Exception as e:
        logger.error(f"Error processing message in turn {turn_id}: {e}", exc_info=True)
        error_msg = {
            "type": "error",
            "turn_id": turn_id,
            "data": f"Error processing message: {e}",
            "timestamp": time.time(),
        }
        try:
            client_socket.sendall(json.dumps(error_msg).encode() + b"\n")
        except:
            pass


def handle_ipc_client(client_socket, client_id, system_prompt: str, socket_path: str):
    """Handle an IPC client connection with streaming responses.

    Args:
        client_socket: Unix socket connection object
        client_id: Unique client identifier
        system_prompt: System prompt for the DevDuck agent
        socket_path: Socket path (for logging)
    """
    logger.info(f"IPC connection established with client {client_id}")

    # Import DevDuck and create a new instance for this connection
    try:
        from devduck import DevDuck

        # Create a new DevDuck instance with auto_start_servers=False to avoid recursion
        connection_devduck = DevDuck(auto_start_servers=False)

        # Override system prompt if provided
        if connection_devduck.agent and system_prompt:
            connection_devduck.agent.system_prompt += (
                "\nCustom system prompt: " + system_prompt
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
            system_prompt=system_prompt or "You are a helpful IPC server assistant.",
        )

    try:
        # Send welcome message
        welcome = {
            "type": "connected",
            "data": "🦆 Welcome to DevDuck IPC!",
            "timestamp": time.time(),
        }
        client_socket.sendall(json.dumps(welcome).encode() + b"\n")

        # Track active tasks
        with ThreadPoolExecutor(max_workers=5) as executor:
            buffer = b""

            while True:
                # Receive data
                chunk = client_socket.recv(4096)
                if not chunk:
                    break

                buffer += chunk

                # Process complete messages (newline delimited)
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)

                    try:
                        message_data = json.loads(line.decode())

                        # Check for exit command
                        if message_data.get("message", "").lower() == "exit":
                            bye = {
                                "type": "disconnected",
                                "data": "Connection closed by client request.",
                                "timestamp": time.time(),
                            }
                            client_socket.sendall(json.dumps(bye).encode() + b"\n")
                            logger.info(f"Client {client_id} requested to exit")
                            return

                        # Generate unique turn ID
                        turn_id = message_data.get("turn_id") or str(uuid.uuid4())

                        logger.info(
                            f"Received from client {client_id}: {message_data.get('message', '')[:100]}"
                        )

                        # Process message in thread pool (concurrent)
                        executor.submit(
                            process_ipc_message,
                            connection_agent,
                            message_data,
                            client_socket,
                            turn_id,
                        )

                    except json.JSONDecodeError:
                        logger.warning(
                            f"Invalid JSON from client {client_id}: {line[:100]}"
                        )
                        continue

    except BrokenPipeError:
        # Normal disconnect - client closed connection
        logger.debug(f"IPC client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Error handling IPC client {client_id}: {e}")
    finally:
        try:
            client_socket.close()
        except:
            pass
        logger.info(f"IPC connection with client {client_id} closed")


def run_ipc_server(socket_path: str, system_prompt: str) -> None:
    """Run an IPC server that processes client requests with DevDuck instances.

    Args:
        socket_path: Unix socket path to bind
        system_prompt: System prompt for DevDuck agents
    """
    IPC_SERVER_THREADS[socket_path]["running"] = True
    IPC_SERVER_THREADS[socket_path]["connections"] = 0
    IPC_SERVER_THREADS[socket_path]["start_time"] = time.time()

    # Remove existing socket if it exists
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    import socket as unix_socket

    server_socket = unix_socket.socket(unix_socket.AF_UNIX, unix_socket.SOCK_STREAM)

    try:
        server_socket.bind(socket_path)
        server_socket.listen(10)
        logger.info(f"IPC Server listening on {socket_path}")

        IPC_SERVER_THREADS[socket_path]["socket"] = server_socket

        client_counter = 0

        while IPC_SERVER_THREADS[socket_path]["running"]:
            # Set timeout to check periodically if server should stop
            server_socket.settimeout(1.0)

            try:
                client_socket, _ = server_socket.accept()
                IPC_SERVER_THREADS[socket_path]["connections"] += 1
                client_counter += 1

                client_id = f"client_{client_counter}"

                # Handle client in a new thread
                client_thread = threading.Thread(
                    target=handle_ipc_client,
                    args=(client_socket, client_id, system_prompt, socket_path),
                    daemon=True,
                )
                client_thread.start()

            except TimeoutError:
                # Expected timeout for checking stop condition
                pass
            except Exception as e:
                if IPC_SERVER_THREADS[socket_path]["running"]:
                    logger.error(f"Error accepting connection: {e}")

    except Exception as e:
        logger.error(f"IPC server error on {socket_path}: {e}", exc_info=True)
    finally:
        try:
            server_socket.close()
        except:
            pass

        # Clean up socket file
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        logger.info(f"IPC Server on {socket_path} stopped")
        IPC_SERVER_THREADS[socket_path]["running"] = False


@tool
def ipc(
    action: str,
    socket_path: str = None,
    system_prompt: str = "You are a helpful IPC server assistant.",
) -> dict:
    """Create and manage IPC servers with real-time streaming.

    This tool creates a Unix socket server for inter-process communication,
    similar to the WebSocket server but for local processes (tray, ambient, etc.).

    Args:
        action: Action to perform (start_server, stop_server, get_status)
        socket_path: Unix socket path (default: /tmp/devduck_main.sock)
        system_prompt: System prompt for the server DevDuck instances

    Returns:
        Dictionary containing status and response content
    """
    # Default socket path
    if not socket_path:
        socket_path = os.path.join(tempfile.gettempdir(), "devduck_main.sock")

    if action == "start_server":
        if socket_path in IPC_SERVER_THREADS and IPC_SERVER_THREADS[socket_path].get(
            "running", False
        ):
            return {
                "status": "error",
                "content": [
                    {"text": f"❌ Error: IPC Server already running on {socket_path}"}
                ],
            }

        IPC_SERVER_THREADS[socket_path] = {"running": False}
        server_thread = threading.Thread(
            target=run_ipc_server, args=(socket_path, system_prompt), daemon=True
        )
        server_thread.start()

        time.sleep(0.5)

        if not IPC_SERVER_THREADS[socket_path].get("running", False):
            return {
                "status": "error",
                "content": [
                    {"text": f"❌ Error: Failed to start IPC Server on {socket_path}"}
                ],
            }

        return {
            "status": "success",
            "content": [
                {"text": f"✅ IPC Server started successfully on {socket_path}"},
                {"text": f"System prompt: {system_prompt}"},
                {"text": "🌊 Real-time streaming with concurrent message processing"},
                {"text": "📦 Newline-delimited JSON messages with turn_id"},
                {
                    "text": "🦆 Server creates a new DevDuck instance for each connection"
                },
                {"text": "⚡ Send multiple messages without waiting!"},
                {"text": f"📝 Connect from local processes to: {socket_path}"},
            ],
        }

    elif action == "stop_server":
        if socket_path not in IPC_SERVER_THREADS or not IPC_SERVER_THREADS[
            socket_path
        ].get("running", False):
            return {
                "status": "error",
                "content": [
                    {"text": f"❌ Error: No IPC Server running on {socket_path}"}
                ],
            }

        IPC_SERVER_THREADS[socket_path]["running"] = False

        # Close socket if exists
        if "socket" in IPC_SERVER_THREADS[socket_path]:
            try:
                IPC_SERVER_THREADS[socket_path]["socket"].close()
            except:
                pass

        time.sleep(1.0)

        connections = IPC_SERVER_THREADS[socket_path].get("connections", 0)
        uptime = time.time() - IPC_SERVER_THREADS[socket_path].get(
            "start_time", time.time()
        )

        del IPC_SERVER_THREADS[socket_path]

        return {
            "status": "success",
            "content": [
                {"text": f"✅ IPC Server on {socket_path} stopped successfully"},
                {
                    "text": f"Statistics: {connections} connections handled, uptime {uptime:.2f} seconds"
                },
            ],
        }

    elif action == "get_status":
        if not IPC_SERVER_THREADS:
            return {
                "status": "success",
                "content": [{"text": "No IPC Servers running"}],
            }

        status_info = []
        for path, data in IPC_SERVER_THREADS.items():
            if data.get("running", False):
                uptime = time.time() - data.get("start_time", time.time())
                connections = data.get("connections", 0)
                status_info.append(
                    f"Socket {path}: Running - {connections} connections, uptime {uptime:.2f}s"
                )
            else:
                status_info.append(f"Socket {path}: Stopped")

        return {
            "status": "success",
            "content": [
                {"text": "IPC Server Status:"},
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
