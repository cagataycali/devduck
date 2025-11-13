"""MCP Server Tool for DevDuck.

Transforms DevDuck into an MCP (Model Context Protocol) server, exposing devduck
tools and capabilities to any MCP-compatible client (Claude Desktop, other agents, etc.).
"""

import contextlib
import logging
import threading
import time
import traceback
from collections.abc import AsyncIterator
from typing import Any, Optional

from strands import tool

logger = logging.getLogger(__name__)

# MCP imports with error handling
MCP_IMPORT_ERROR = ""
try:
    import uvicorn
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.middleware.cors import CORSMiddleware
    from starlette.routing import Mount
    from starlette.types import Receive, Scope, Send

    HAS_MCP = True
except ImportError as e:
    HAS_MCP = False
    MCP_IMPORT_ERROR = str(e)


# Global state to track MCP servers
_server_state = {
    "servers": {},  # Map of server_id -> server_info
    "default_server": None,  # ID of the default server
}


@tool
def mcp_server(
    action: str,
    server_id: str = "default",
    transport: str = "http",
    port: int = 8000,
    tools: Optional[list[str]] = None,
    expose_agent: bool = True,
    stateless: bool = False,
    agent: Any = None,
) -> dict[str, Any]:
    """Turn devduck into an MCP server, exposing tools as MCP tools.

    Args:
        action: Action to perform - "start", "stop", "status", "list"
        server_id: Unique identifier for this server instance (default: "default")
        transport: Transport type - "http" (StreamableHTTP, background) or "stdio" (foreground, blocking)
        port: Port for HTTP server (only used when transport="http", default: 8000)
        tools: Optional list of tool names to expose. If None, exposes all tools except mcp_server itself
        expose_agent: Whether to expose "devduck" tool for full agent conversations (default: True)
        stateless: If True, creates fresh transport per request with no session state (default: False)
        agent: Parent agent instance (auto-injected by Strands framework)

    Returns:
        Result dictionary with status and content
    """
    try:
        # Check if MCP is installed
        if not HAS_MCP:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"❌ MCP not installed: {MCP_IMPORT_ERROR}\n\n"
                        f"Install with: pip install strands-mcp-server"
                    }
                ],
            }

        # Route to appropriate handler
        if action == "start":
            return _start_mcp_server(
                server_id, transport, port, tools, expose_agent, stateless, agent
            )
        elif action == "stop":
            return _stop_mcp_server(server_id)
        elif action == "status":
            return _get_mcp_status()
        elif action == "list":
            return _list_mcp_servers()
        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"❌ Unknown action: {action}\n\nValid actions: start, stop, status, list"
                    }
                ],
            }

    except Exception as e:
        logger.exception("MCP server tool error")
        return {
            "status": "error",
            "content": [{"text": f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"}],
        }


def _start_mcp_server(
    server_id: str,
    transport: str,
    port: int,
    tools_filter: Optional[list[str]],
    expose_agent: bool,
    stateless: bool,
    agent: Any,
) -> dict[str, Any]:
    """Start an MCP server exposing devduck tools."""
    if server_id in _server_state["servers"]:
        return {
            "status": "error",
            "content": [{"text": f"❌ Server '{server_id}' is already running"}],
        }

    if not agent:
        return {
            "status": "error",
            "content": [{"text": "❌ Tool context not available"}],
        }

    # Get all agent tools
    all_tools = agent.tool_registry.get_all_tools_config()
    if not all_tools:
        return {"status": "error", "content": [{"text": "❌ No tools found in agent"}]}

    # Filter tools based on tools_filter parameter
    if tools_filter:
        agent_tools = {
            name: spec
            for name, spec in all_tools.items()
            if name in tools_filter and name != "mcp_server"
        }
        if not agent_tools and not expose_agent:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"❌ No matching tools found. Available: {list(all_tools.keys())}"
                    }
                ],
            }
    else:
        agent_tools = {
            name: spec for name, spec in all_tools.items() if name != "mcp_server"
        }

    logger.debug(
        f"Creating MCP server with {len(agent_tools)} tools: {list(agent_tools.keys())}"
    )

    try:
        # Create low-level MCP server
        server = Server(f"devduck-{server_id}")

        # Create MCP Tool objects from agent tools
        mcp_tools = []
        for tool_name, tool_spec in agent_tools.items():
            description = tool_spec.get("description", f"DevDuck tool: {tool_name}")
            input_schema = {}

            if "inputSchema" in tool_spec:
                if "json" in tool_spec["inputSchema"]:
                    input_schema = tool_spec["inputSchema"]["json"]
                else:
                    input_schema = tool_spec["inputSchema"]

            mcp_tools.append(
                types.Tool(
                    name=tool_name,
                    description=description,
                    inputSchema=input_schema,
                )
            )

        # Add agent invocation tool if requested
        if expose_agent:
            agent_invoke_tool = types.Tool(
                name="devduck",
                description=(
                    "Invoke a FULL DevDuck instance with complete capabilities. "
                    "Each invocation creates a fresh DevDuck agent with self-healing, "
                    "hot-reload, all tools, knowledge base integration, and system prompt building. "
                    "Use this for complex queries requiring reasoning, multi-tool orchestration, "
                    "or when you need the complete DevDuck experience via MCP."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt or query to send to DevDuck",
                        }
                    },
                    "required": ["prompt"],
                },
            )
            mcp_tools.append(agent_invoke_tool)

        logger.debug(
            f"Created {len(mcp_tools)} MCP tools (agent invocation: {expose_agent})"
        )

        # Capture transport in closure
        _transport = transport

        # Register list_tools handler
        @server.list_tools()
        async def list_tools() -> list[types.Tool]:
            """Return list of available MCP tools."""
            logger.debug(f"list_tools called, returning {len(mcp_tools)} tools")
            return mcp_tools

        # Register call_tool handler
        @server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            """Handle tool calls from MCP clients."""
            try:
                logger.debug(f"call_tool: name={name}, arguments={arguments}")

                # Handle agent invocation tool - create a full DevDuck instance
                if name == "devduck" and expose_agent:
                    prompt = arguments.get("prompt")
                    if not prompt:
                        return [
                            types.TextContent(
                                type="text",
                                text="❌ Error: 'prompt' parameter is required",
                            )
                        ]

                    logger.debug(f"Invoking devduck with prompt: {prompt[:100]}...")

                    # Create a NEW DevDuck instance for this MCP invocation
                    # This gives full DevDuck power: self-healing, hot-reload, all tools, etc.
                    try:
                        from devduck import DevDuck

                        # Create fresh DevDuck instance (no auto-start to avoid recursion)
                        mcp_devduck = DevDuck(auto_start_servers=False)
                        mcp_agent = mcp_devduck.agent

                        if not mcp_agent:
                            return [
                                types.TextContent(
                                    type="text",
                                    text="❌ Error: Failed to create DevDuck instance",
                                )
                            ]

                        # Execute with full DevDuck capabilities
                        result = mcp_agent(prompt)

                    except Exception as e:
                        logger.error(f"DevDuck creation failed: {e}", exc_info=True)
                        return [
                            types.TextContent(
                                type="text",
                                text=f"❌ Error creating DevDuck instance: {str(e)}",
                            )
                        ]

                    # Extract text response from agent result
                    response_text = str(result)

                    logger.debug(
                        f"DevDuck invocation complete, response length: {len(response_text)}"
                    )

                    return [types.TextContent(type="text", text=response_text)]

                # Check if tool exists in agent
                if name not in agent_tools:
                    return [
                        types.TextContent(type="text", text=f"❌ Unknown tool: {name}")
                    ]

                # Call the agent tool
                tool_caller = getattr(agent.tool, name.replace("-", "_"))

                # For stdio transport, try to pass non_interactive=True if the tool supports it
                if _transport == "stdio":
                    try:
                        result = tool_caller(**arguments, non_interactive=True)
                    except TypeError:
                        logger.debug(
                            f"Tool '{name}' doesn't support non_interactive parameter"
                        )
                        result = tool_caller(**arguments)
                else:
                    result = tool_caller(**arguments)

                logger.debug(f"Tool '{name}' execution complete")

                # Convert result to MCP TextContent format
                mcp_content = []
                if isinstance(result, dict) and "content" in result:
                    for item in result.get("content", []):
                        if isinstance(item, dict) and "text" in item:
                            mcp_content.append(
                                types.TextContent(type="text", text=item["text"])
                            )
                else:
                    mcp_content.append(types.TextContent(type="text", text=str(result)))

                return (
                    mcp_content
                    if mcp_content
                    else [
                        types.TextContent(
                            type="text", text="✅ Tool executed successfully"
                        )
                    ]
                )

            except Exception as e:
                logger.exception(f"Error calling tool '{name}'")
                return [types.TextContent(type="text", text=f"❌ Error: {str(e)}")]

        # Record server state
        _server_state["servers"][server_id] = {
            "server": server,
            "transport": transport,
            "port": port,
            "stateless": stateless,
            "tools": list(agent_tools.keys()),
            "start_time": time.time(),
            "status": "starting",
            "expose_agent": expose_agent,
        }

        if _server_state["default_server"] is None:
            _server_state["default_server"] = server_id

        # For stdio transport: Run in FOREGROUND
        if transport == "stdio":
            logger.info(
                f"Starting MCP server '{server_id}' in stdio mode (foreground, blocking)"
            )
            _server_state["servers"][server_id]["status"] = "running"

            import asyncio
            from mcp.server.stdio import stdio_server

            async def run_stdio() -> None:
                """Run stdio server in foreground."""
                async with stdio_server() as streams:
                    await server.run(
                        streams[0], streams[1], server.create_initialization_options()
                    )

            asyncio.run(run_stdio())

            if server_id in _server_state["servers"]:
                del _server_state["servers"][server_id]

            return {
                "status": "success",
                "content": [{"text": f"✅ MCP server '{server_id}' stopped"}],
            }

        # For http transport: Run in BACKGROUND
        server_thread = threading.Thread(
            target=_run_mcp_server,
            args=(server, transport, port, stateless, server_id, len(mcp_tools)),
            daemon=True,
        )

        _server_state["servers"][server_id]["thread"] = server_thread
        server_thread.start()

        # Give server time to start
        time.sleep(2)

        # Check status
        if server_id not in _server_state["servers"]:
            return {
                "status": "error",
                "content": [{"text": f"❌ Server '{server_id}' failed to start"}],
            }

        server_info = _server_state["servers"][server_id]
        if server_info["status"] == "error":
            error_msg = server_info.get("error", "Unknown error")
            return {
                "status": "error",
                "content": [{"text": f"❌ Server '{server_id}' failed: {error_msg}"}],
            }

        # Update to running
        _server_state["servers"][server_id]["status"] = "running"

        # Build status message
        tool_list = "\n".join(f"  • {tool.name}" for tool in mcp_tools[:10])
        if len(mcp_tools) > 10:
            tool_list += f"\n  ... and {len(mcp_tools) - 10} more"

        if expose_agent:
            tool_list += "\n  • devduck (full devduck invocation) ✨"

        mode_desc = (
            "stateless (multi-node ready)"
            if stateless
            else "stateful (session persistence)"
        )
        message = (
            f"✅ MCP server '{server_id}' started on port {port}\n\n"
            f"📊 Mode: {mode_desc}\n"
            f"🔧 Exposed {len(mcp_tools)} tools:\n"
            f"{tool_list}\n\n"
            f"🔗 Connect at: http://localhost:{port}/mcp"
        )

        return {"status": "success", "content": [{"text": message}]}

    except Exception as e:
        logger.exception("Error starting MCP server")

        if server_id in _server_state["servers"]:
            _server_state["servers"][server_id]["status"] = "error"
            _server_state["servers"][server_id]["error"] = str(e)

        return {
            "status": "error",
            "content": [{"text": f"❌ Failed to start MCP server: {str(e)}"}],
        }


def _run_mcp_server(
    server: "Server",
    transport: str,
    port: int,
    stateless: bool,
    server_id: str,
    tool_count: int,
) -> None:
    """Run MCP server in background thread."""
    try:
        logger.debug(
            f"Starting MCP server: server_id={server_id}, transport={transport}, port={port}, stateless={stateless}"
        )

        if transport == "http":
            session_manager = StreamableHTTPSessionManager(
                app=server,
                event_store=None,
                json_response=False,
                stateless=stateless,
            )

            async def handle_streamable_http(
                scope: Scope, receive: Receive, send: Send
            ) -> None:
                await session_manager.handle_request(scope, receive, send)

            @contextlib.asynccontextmanager
            async def lifespan(app: Starlette) -> AsyncIterator[None]:
                async with session_manager.run():
                    logger.info(
                        f"MCP server '{server_id}' running (stateless={stateless})"
                    )
                    try:
                        yield
                    finally:
                        logger.info(f"MCP server '{server_id}' shutting down...")

            starlette_app = Starlette(
                debug=True,
                routes=[
                    Mount("/mcp", app=handle_streamable_http),
                ],
                lifespan=lifespan,
            )

            starlette_app = CORSMiddleware(
                starlette_app,
                allow_origins=["*"],
                allow_methods=["GET", "POST", "DELETE"],
                expose_headers=["Mcp-Session-Id"],
            )

            logger.debug(f"Starting Uvicorn server on 0.0.0.0:{port}")
            uvicorn.run(starlette_app, host="0.0.0.0", port=port, log_level="info")
        else:
            logger.error(f"Unsupported transport: {transport}")

    except Exception as e:
        logger.exception("Error in _run_mcp_server")

        if server_id in _server_state["servers"]:
            _server_state["servers"][server_id]["status"] = "error"
            _server_state["servers"][server_id]["error"] = str(e)


def _stop_mcp_server(server_id: str) -> dict[str, Any]:
    """Stop a running MCP server."""
    if server_id not in _server_state["servers"]:
        return {
            "status": "error",
            "content": [{"text": f"❌ Server '{server_id}' is not running"}],
        }

    server_info = _server_state["servers"][server_id]
    server_info["status"] = "stopping"

    del _server_state["servers"][server_id]

    if _server_state["default_server"] == server_id:
        _server_state["default_server"] = (
            next(iter(_server_state["servers"])) if _server_state["servers"] else None
        )

    return {
        "status": "success",
        "content": [{"text": f"✅ MCP server '{server_id}' stopped"}],
    }


def _get_mcp_status() -> dict[str, Any]:
    """Get status of all MCP servers."""
    if not _server_state["servers"]:
        return {"status": "success", "content": [{"text": "ℹ️ No MCP servers running"}]}

    lines = ["📡 **MCP Server Status**\n"]

    for server_id, server_info in _server_state["servers"].items():
        uptime = time.time() - server_info["start_time"]
        uptime_str = f"{int(uptime // 60)}m {int(uptime % 60)}s"

        default_marker = (
            " (default)" if server_id == _server_state["default_server"] else ""
        )
        status_emoji = {
            "running": "✅",
            "starting": "🔄",
            "stopping": "⏸️",
            "error": "❌",
        }.get(server_info["status"], "❓")

        lines.append(f"\n**{server_id}{default_marker}**")
        lines.append(f"  • Status: {status_emoji} {server_info['status']}")
        lines.append(f"  • Transport: {server_info['transport']}")

        if server_info["transport"] == "http":
            lines.append(f"  • Port: {server_info['port']}")
            lines.append(f"  • Connect: http://localhost:{server_info['port']}/mcp")
            mode_type = (
                "stateless (multi-node)"
                if server_info.get("stateless", False)
                else "stateful (single-node)"
            )
            lines.append(f"  • Type: {mode_type}")

        lines.append(f"  • Uptime: {uptime_str}")
        lines.append(f"  • Tools: {len(server_info['tools'])} exposed")

        if server_info.get("expose_agent"):
            lines.append(f"  • Agent Invocation: ✅ Enabled")

        if server_info["status"] == "error" and "error" in server_info:
            lines.append(f"  • Error: {server_info['error']}")

    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def _list_mcp_servers() -> dict[str, Any]:
    """List running MCP servers."""
    if not _server_state["servers"]:
        return {"status": "success", "content": [{"text": "ℹ️ No MCP servers running"}]}

    lines = ["📋 **Running MCP Servers**\n"]

    for server_id, server_info in _server_state["servers"].items():
        default_marker = (
            " (default)" if server_id == _server_state["default_server"] else ""
        )
        mode_info = (
            f"port {server_info['port']}"
            if server_info["transport"] == "http"
            else "stdio"
        )

        lines.append(
            f"• {server_id}{default_marker}: {server_info['status']}, "
            f"{server_info['transport']} ({mode_info})"
        )

    return {"status": "success", "content": [{"text": "\n".join(lines)}]}
