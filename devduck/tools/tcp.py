"""TCP tool for DevDuck agents with real-time streaming support.

This module provides TCP server and client functionality for DevDuck agents,
allowing them to communicate over TCP/IP networks with real-time response streaming.
The tool runs server operations in background threads, enabling concurrent
communication without blocking the main agent.

Key Features:
1. TCP Server: Listen for incoming connections and process them with a DevDuck agent
2. Real-time Streaming: Responses stream to clients as they're generated (non-blocking)
3. TCP Client: Connect to remote TCP servers and exchange messages
4. Background Processing: Server runs in a background thread
5. Per-Connection DevDuck: Creates a fresh DevDuck instance for each client connection
6. Callback Handler: Uses Strands callback system for efficient streaming

How Streaming Works:
-------------------
Instead of blocking until the full response is ready, this implementation uses
Strands' callback_handler mechanism to stream data as it's generated:

- Text chunks stream immediately as the model generates them
- Tool invocations are announced in real-time
- Tool results are sent as they complete
- No buffering delays - everything is instant

Usage with DevDuck Agent:

```python
from devduck import devduck

# Start a streaming TCP server (each connection gets its own DevDuck instance)
result = devduck.agent.tool.tcp(
    action="start_server",
    host="127.0.0.1",
    port=8000,
    system_prompt="You are a helpful TCP server assistant.",
)

# Connect as a client and receive streaming responses
result = devduck.agent.tool.tcp(
    action="client_send",
    host="127.0.0.1",
    port=8000,
    message="What's 2+2?"
)

# Stop the TCP server
result = devduck.agent.tool.tcp(action="stop_server", port=8000)
```

For testing with netcat:
```bash
# Start server from devduck
devduck "start a tcp server on port 8000"

# Connect with netcat and chat in real-time
nc localhost 8000
```

See the tcp function docstring for more details on configuration options and parameters.
"""

import logging
import socket
import threading
import time
import os
from typing import Any

from strands import Agent, tool

logger = logging.getLogger(__name__)

# Global registry to store server threads
SERVER_THREADS: dict[int, dict[str, Any]] = {}


class TCPStreamingCallbackHandler:
    """Callback handler that streams agent responses directly over TCP socket.

    This handler implements real-time streaming of:
    - Assistant responses (text chunks as they're generated)
    - Tool invocations (names and status)
    - Reasoning text (if enabled)
    - Tool results (success/error status)

    All data is sent immediately to the TCP client without buffering.
    """

    def __init__(self, client_socket: socket.socket):
        """Initialize the streaming handler.

        Args:
            client_socket: The TCP socket to stream data to
        """
        self.socket = client_socket
        self.tool_count = 0
        self.previous_tool_use = None

    def _send(self, data: str) -> None:
        """Safely send data over TCP socket.

        Args:
            data: String data to send
        """
        try:
            self.socket.sendall(data.encode())
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning(f"Failed to send data over TCP: {e}")

    def __call__(self, **kwargs: Any) -> None:
        """Stream events to TCP socket in real-time.

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

        # Skip reasoning text to keep output clean
        if reasoningText:
            self._send(reasoningText)

        # Stream response text chunks
        if data:
            self._send(data)
            if complete:
                self._send("\n")

        # Stream tool invocation notifications
        if current_tool_use and current_tool_use.get("name"):
            tool_name = current_tool_use.get("name", "Unknown tool")
            if self.previous_tool_use != current_tool_use:
                self.previous_tool_use = current_tool_use
                self.tool_count += 1
                self._send(f"\n🛠️  Tool #{self.tool_count}: {tool_name}\n")

        # Stream tool results
        if isinstance(message, dict) and message.get("role") == "user":
            for content in message.get("content", []):
                if isinstance(content, dict):
                    tool_result = content.get("toolResult")
                    if tool_result:
                        status = tool_result.get("status", "unknown")
                        if status == "success":
                            self._send(f"✅ Tool completed successfully\n")
                        else:
                            self._send(f"❌ Tool failed\n")


def handle_client(
    client_socket: socket.socket,
    client_address: tuple,
    system_prompt: str,
    buffer_size: int,
    model: Any,
    parent_tools: list | None = None,
    callback_handler: Any = None,
    trace_attributes: dict | None = None,
) -> None:
    """Handle a client connection in the TCP server with streaming responses.

    Args:
        client_socket: The socket for the client connection
        client_address: The address of the client
        system_prompt: System prompt for creating a new agent for this connection
        buffer_size: Size of the message buffer
        model: Model instance from parent agent (unused with DevDuck)
        parent_tools: Tools inherited from the parent agent (unused with DevDuck)
        callback_handler: Callback handler from parent agent (unused with DevDuck)
        trace_attributes: Trace attributes from the parent agent (unused with DevDuck)
    """
    logger.info(f"Connection established with {client_address}")

    # Create a streaming callback handler for this connection
    streaming_handler = TCPStreamingCallbackHandler(client_socket)

    # Create agent directly with callback handler (bypass DevDuck wrapper for proper callback support)
    from strands import Agent
    from strands.models.ollama import OllamaModel
    from strands_tools.utils.models.model import create_model

    # Check if MODEL_PROVIDER env variable is set
    model_provider = os.getenv("MODEL_PROVIDER")

    if model_provider:
        agent_model = create_model(provider=model_provider)
    else:
        # Fallback to Ollama
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
        agent_model = OllamaModel(
            host=ollama_host,
            model_id=ollama_model,
            temperature=1,
            keep_alive="5m",
        )

    # Import all tools
    from strands_tools import (
        shell,
        editor,
        file_read,
        file_write,
        python_repl,
        current_time,
        calculator,
        journal,
        image_reader,
        use_agent,
        load_tool,
        environment,
    )

    # Create Agent with callback handler at initialization
    connection_agent = Agent(
        model=agent_model,
        tools=[
            shell,
            editor,
            file_read,
            file_write,
            python_repl,
            current_time,
            calculator,
            journal,
            image_reader,
            use_agent,
            load_tool,
            environment,
        ],
        system_prompt=(
            system_prompt
            if system_prompt
            else "You are a helpful TCP server assistant."
        ),
        callback_handler=streaming_handler,  # Pass callback during init!
        load_tools_from_directory=True,
    )

    try:
        # Send welcome message
        welcome_msg = "🦆 Welcome to DevDuck TCP Server!\n"
        welcome_msg += (
            "Real-time streaming enabled - responses stream as they're generated.\n"
        )
        welcome_msg += "Send a message or 'exit' to close the connection.\n\n"
        streaming_handler._send(welcome_msg)

        while True:
            # Receive data from the client
            data = client_socket.recv(buffer_size)

            if not data:
                logger.info(f"Client {client_address} disconnected")
                break

            message = data.decode().strip()
            logger.info(f"Received from {client_address}: {message}")

            if message.lower() == "exit":
                streaming_handler._send("Connection closed by client request.\n")
                logger.info(f"Client {client_address} requested to exit")
                break

            # Process the message - responses stream automatically via callback_handler
            try:
                streaming_handler._send(f"\n\n🦆: {message}\n\n")

                # The agent call will stream responses directly to the socket
                # through the callback_handler - no need to collect the response
                connection_agent(message)

                # Send completion marker
                streaming_handler._send("\n\n🦆\n\n")

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                streaming_handler._send(f"\n❌ Error processing message: {e}\n\n")

    except Exception as e:
        logger.error(f"Error handling client {client_address}: {e}")
    finally:
        client_socket.close()
        logger.info(f"Connection with {client_address} closed")


def run_server(
    host: str,
    port: int,
    system_prompt: str,
    max_connections: int,
    buffer_size: int,
    parent_agent: Agent | None = None,
) -> None:
    """Run a TCP server that processes client requests with per-connection Strands agents.

    Args:
        host: Host address to bind the server
        port: Port number to bind the server
        system_prompt: System prompt for the server agents
        max_connections: Maximum number of concurrent connections
        buffer_size: Size of the message buffer
        parent_agent: Parent agent to inherit tools from
    """
    # Store server state
    SERVER_THREADS[port]["running"] = True
    SERVER_THREADS[port]["connections"] = 0
    SERVER_THREADS[port]["start_time"] = time.time()

    # Get model, tools, callback_handler and trace attributes from parent agent
    model = None
    callback_handler = None
    parent_tools = []
    trace_attributes = {}
    if parent_agent:
        model = parent_agent.model
        callback_handler = parent_agent.callback_handler
        parent_tools = list(parent_agent.tool_registry.registry.values())
        trace_attributes = parent_agent.trace_attributes

    # Create server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((host, port))
        server_socket.listen(max_connections)
        logger.info(f"TCP Server listening on {host}:{port}")

        SERVER_THREADS[port]["socket"] = server_socket

        while SERVER_THREADS[port]["running"]:
            # Set a timeout to check periodically if the server should stop
            server_socket.settimeout(1.0)

            try:
                # Accept client connection
                client_socket, client_address = server_socket.accept()
                SERVER_THREADS[port]["connections"] += 1

                # Handle client in a new thread with a fresh agent
                client_thread = threading.Thread(
                    target=handle_client,
                    args=(
                        client_socket,
                        client_address,
                        system_prompt,
                        buffer_size,
                        model,
                        parent_tools,
                        callback_handler,
                        trace_attributes,
                    ),
                )
                client_thread.daemon = True
                client_thread.start()

            except TimeoutError:
                # This is expected due to the timeout, allows checking if server should stop
                pass
            except Exception as e:
                if SERVER_THREADS[port]["running"]:
                    logger.error(f"Error accepting connection: {e}")

    except Exception as e:
        logger.error(f"Server error on {host}:{port}: {e}")
    finally:
        try:
            server_socket.close()
        except OSError:
            # Socket already closed, safe to ignore
            pass
        logger.info(f"TCP Server on {host}:{port} stopped")
        SERVER_THREADS[port]["running"] = False


@tool
def tcp(
    action: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    system_prompt: str = "You are a helpful TCP server assistant.",
    message: str = "",
    timeout: int = 90,
    buffer_size: int = 4096,
    max_connections: int = 5,
    agent: Any = None,
) -> dict:
    """Create and manage TCP servers and clients with real-time streaming for DevDuck instances.

    This function provides TCP server and client functionality for DevDuck agents,
    allowing them to communicate over TCP/IP networks. Servers run in background
    threads with a new, fresh DevDuck instance for each client connection.

    **Real-time Streaming:** Unlike traditional blocking TCP responses, this
    implementation streams data as it's generated using Strands' callback_handler
    mechanism. Clients receive:
    - Text chunks as the model generates them (no buffering)
    - Tool invocation notifications in real-time
    - Tool completion status immediately
    - Reasoning text (if enabled)

    How It Works:
    ------------
    1. Server Mode:
       - Starts a TCP server in a background thread
       - Creates a dedicated DevDuck instance for EACH client connection
       - Attaches a streaming callback handler to send data immediately
       - Each DevDuck has full self-healing, hot-reload, and all tools
       - Processes client messages with non-blocking streaming responses

    2. Client Mode:
       - Connects to a TCP server
       - Sends messages and receives responses
       - Maintains stateless connections (no persistent sessions)

    3. Management:
       - Track server status and statistics
       - Stop servers gracefully
       - Monitor connections and performance

    Common Use Cases:
    ---------------
    - Network service automation with real-time feedback
    - Inter-agent communication with streaming
    - Remote command and control (instant responsiveness)
    - API gateway implementation with SSE-like behavior
    - IoT device management with live updates
    - Interactive chat services over raw TCP

    Args:
        action: Action to perform (start_server, stop_server, get_status, client_send)
        host: Host address for server or client connection
        port: Port number for server or client connection
        system_prompt: System prompt for the server DevDuck instances (for start_server)
        message: Message to send to the TCP server (for client_send action)
        timeout: Connection timeout in seconds (default: 90)
        buffer_size: Size of the message buffer in bytes (default: 4096)
        max_connections: Maximum number of concurrent connections (default: 5)

    Returns:
        Dictionary containing status and response content

    Notes:
        - Server instances persist until explicitly stopped
        - Each client connection gets its own DevDuck instance
        - Connection DevDuck instances have all standard DevDuck capabilities
        - Streaming is automatic via callback_handler (no configuration needed)
        - Client connections are stateless
        - Compatible with any TCP client (netcat, telnet, custom clients)

    Examples:
        # Start a streaming server
        devduck("start a tcp server on port 9000")

        # Test with netcat
        nc localhost 9000
        > what is 2+2?
        [Streaming response appears in real-time]

        # Send message from another devduck instance
        devduck("send 'hello world' to tcp server at localhost:9000")
    """
    # Get parent agent from tool context if available
    parent_agent = agent

    if action == "start_server":
        # Check if server already running on this port
        if port in SERVER_THREADS and SERVER_THREADS[port].get("running", False):
            return {
                "status": "error",
                "content": [
                    {"text": f"❌ Error: TCP Server already running on port {port}"}
                ],
            }

        # Create server thread
        SERVER_THREADS[port] = {"running": False}
        server_thread = threading.Thread(
            target=run_server,
            args=(
                host,
                port,
                system_prompt,
                max_connections,
                buffer_size,
                parent_agent,
            ),
        )
        server_thread.daemon = True
        server_thread.start()

        # Wait briefly to ensure server starts
        time.sleep(0.5)

        if not SERVER_THREADS[port].get("running", False):
            return {
                "status": "error",
                "content": [
                    {"text": f"❌ Error: Failed to start TCP Server on {host}:{port}"}
                ],
            }

        return {
            "status": "success",
            "content": [
                {"text": f"✅ TCP Server started successfully on {host}:{port}"},
                {"text": f"System prompt: {system_prompt}"},
                {"text": "🌊 Real-time streaming enabled (non-blocking responses)"},
                {
                    "text": "🦆 Server creates a new DevDuck instance for each connection"
                },
                {
                    "text": "🛠️  Each DevDuck has full self-healing, hot-reload, and all tools"
                },
                {"text": f"📝 Test with: nc localhost {port}"},
            ],
        }

    elif action == "stop_server":
        if port not in SERVER_THREADS or not SERVER_THREADS[port].get("running", False):
            return {
                "status": "error",
                "content": [
                    {"text": f"❌ Error: No TCP Server running on port {port}"}
                ],
            }

        # Stop the server
        SERVER_THREADS[port]["running"] = False

        # Close socket if it exists
        if "socket" in SERVER_THREADS[port]:
            try:
                SERVER_THREADS[port]["socket"].close()
            except OSError:
                # Socket already closed, safe to ignore
                pass

        # Wait briefly to ensure server stops
        time.sleep(1.0)

        connections = SERVER_THREADS[port].get("connections", 0)
        uptime = time.time() - SERVER_THREADS[port].get("start_time", time.time())

        # Clean up server thread data
        del SERVER_THREADS[port]

        return {
            "status": "success",
            "content": [
                {"text": f"✅ TCP Server on port {port} stopped successfully"},
                {
                    "text": f"Statistics: {connections} connections handled, uptime {uptime:.2f} seconds"
                },
            ],
        }

    elif action == "get_status":
        if not SERVER_THREADS:
            return {
                "status": "success",
                "content": [{"text": "No TCP Servers running"}],
            }

        status_info = []
        for port, data in SERVER_THREADS.items():
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
                {"text": "TCP Server Status:"},
                {"text": "\n".join(status_info)},
            ],
        }

    elif action == "client_send":
        host = host
        port = port
        message = message
        timeout = timeout
        buffer_size = buffer_size

        if not message:
            return {
                "status": "error",
                "content": [
                    {"text": "Error: No message provided for client_send action"}
                ],
            }

        # Create client socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(timeout)

        try:
            # Connect to server
            client_socket.connect((host, port))

            # Receive welcome message
            _welcome = client_socket.recv(buffer_size).decode()

            # Send message to server
            client_socket.sendall(message.encode())

            # Receive response
            response = client_socket.recv(buffer_size).decode()

            # Send exit message and close connection
            client_socket.sendall(b"exit")
            client_socket.close()

            return {
                "status": "success",
                "content": [
                    {"text": f"Connected to {host}:{port} successfully"},
                    {"text": f"Received welcome message: {_welcome}"},
                    {"text": f"Sent message: {message}"},
                    {"text": "Response received:"},
                    {"text": response},
                ],
            }

        except TimeoutError:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Error: Connection to {host}:{port} timed out after {timeout} seconds"
                    }
                ],
            }
        except ConnectionRefusedError:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Error: Connection to {host}:{port} refused - no server running on that port"
                    }
                ],
            }
        except Exception as e:
            return {
                "status": "error",
                "content": [{"text": f"Error connecting to {host}:{port}: {e!s}"}],
            }
        finally:
            try:
                client_socket.close()
            except OSError:
                # Socket already closed, safe to ignore
                pass

    else:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"Error: Unknown action '{action}'. Supported actions are: "
                    f"start_server, stop_server, get_status, client_send"
                }
            ],
        }
