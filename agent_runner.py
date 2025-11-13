#!/usr/bin/env python3
"""
Minimal DevDuck GitHub Agent Runner
Uses DevDuck with GitHub context, MCP tools, and knowledge base integration.

Examples:
  echo "analyze this data" | python agent_runner.py
  python agent_runner.py "what can you do"
  INPUT_TASK="search papers" python agent_runner.py
"""

import json
import os
import sys
import datetime

# Disable auto-start servers for DevDuck singleton before import
os.environ["DEVDUCK_AUTO_START_SERVERS"] = "false"
os.environ["BYPASS_TOOL_CONSENT"] = "true"
os.environ["STRANDS_TOOL_CONSOLE_MODE"] = "enabled"

from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient

# HTTP transport imports
try:
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamablehttp_client

    HTTP_TRANSPORT_AVAILABLE = True
except ImportError:
    HTTP_TRANSPORT_AVAILABLE = False

# DevDuck is required
from devduck import DevDuck


def load_mcp_config() -> dict:
    """Load MCP server configuration from JSON file or environment variable."""
    mcp_servers_env = os.getenv("MCP_SERVERS")
    if mcp_servers_env:
        try:
            config = json.loads(mcp_servers_env)
            return config.get("mcpServers", {})
        except json.JSONDecodeError:
            pass

    # Fallback: hardcoded strands-agents-mcp-server
    return {"strands-agents": {"command": "uvx", "args": ["strands-agents-mcp-server"]}}


def create_mcp_clients() -> list:
    """Create MCP clients from configuration."""
    mcp_config = load_mcp_config()
    clients = []

    for server_name, server_config in mcp_config.items():
        try:
            command = server_config.get("command")
            if command:
                args = server_config.get("args", [])
                env = server_config.get("env", {})

                client = MCPClient(
                    lambda cmd=command, arguments=args, environment=env: stdio_client(
                        StdioServerParameters(
                            command=cmd,
                            args=arguments,
                            env=environment if environment else None,
                        )
                    )
                )
                clients.append((server_name, client))

            elif server_config.get("url") and HTTP_TRANSPORT_AVAILABLE:
                url = server_config.get("url")
                headers = server_config.get("headers", {})

                try:
                    client = MCPClient(lambda: sse_client(url, headers=headers))
                    clients.append((server_name, client))
                except Exception:
                    try:
                        client = MCPClient(
                            lambda: streamablehttp_client(url, headers=headers)
                        )
                        clients.append((server_name, client))
                    except Exception:
                        continue
        except Exception:
            continue

    return clients


def build_system_prompt() -> str:
    """Build system prompt from environment variables."""
    base_prompt = os.getenv("SYSTEM_PROMPT", "")
    if not base_prompt:
        base_prompt = "You are an autonomous GitHub agent powered by DevDuck."

    input_system_prompt = os.getenv("INPUT_SYSTEM_PROMPT", "")
    if input_system_prompt:
        base_prompt = f"{base_prompt}\n\n{input_system_prompt}"

    github_context = os.environ.get("GITHUB_CONTEXT", "")
    if github_context:
        base_prompt = f"{base_prompt}\n\nGitHub Context:\n{github_context}"

    return base_prompt


def run_agent(query: str) -> str:
    """Run DevDuck agent with GitHub context and MCP tools."""

    print("🦆 Creating DevDuck agent...")

    # Create a new DevDuck instance without auto-starting servers
    duck = DevDuck(auto_start_servers=False)
    agent = duck.agent

    # Build and append GitHub context to DevDuck's system prompt
    github_system_prompt = build_system_prompt()
    if agent and github_system_prompt:
        agent.system_prompt += "\n\nCustom system prompt:" + github_system_prompt
        print("✅ GitHub context appended to DevDuck's system prompt")

    # Get MCP clients
    mcp_clients = create_mcp_clients()

    # Run agent with MCP context
    # NOTE: KB retrieval/storage now handled automatically in DevDuck.__call__
    result = None
    if mcp_clients:
        print(f"🔗 Loading {len(mcp_clients)} MCP servers...")

        def run_with_mcp(clients, current_agent):
            if not clients:
                return current_agent(query)

            server_name, client = clients[0]
            remaining = clients[1:]

            with client:
                try:
                    mcp_tools = client.list_tools_sync()
                    print(f"✓ {server_name}: {len(mcp_tools)} tools loaded")
                except Exception as e:
                    print(f"⚠️  {server_name} failed: {e}")

                return run_with_mcp(remaining, current_agent)

        result = run_with_mcp(mcp_clients, agent)
    else:
        result = agent(query)

    return result


def main():
    """Main entry point."""

    # Multi-input task collection (priority-based)
    tasks = {}

    # Priority 1: Piped input (stdin)
    if not sys.stdin.isatty():
        try:
            pipe_task = sys.stdin.read().strip()
            if pipe_task:
                tasks["pipe"] = pipe_task
        except Exception:
            pass

    # Priority 2: Command line arguments
    if len(sys.argv) > 1:
        cmd_task = " ".join(sys.argv[1:])
        if cmd_task:
            tasks["command_line"] = cmd_task

    # Priority 3: Environment variable
    env_task = os.getenv("INPUT_TASK")
    if env_task:
        tasks["environment"] = env_task

    # No tasks found
    if not tasks:
        print(
            "❌ Error: No task provided. Use command line args, stdin, or INPUT_TASK env var."
        )
        sys.exit(1)

    # Combine all tasks
    combined_task = "\n\n".join(tasks.values())

    try:
        result = run_agent(combined_task)
        print(f"\n✅ Agent Execution Complete!")
        print(f"📝 Result: {result}")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
