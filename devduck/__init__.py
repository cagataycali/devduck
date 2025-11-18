#!/usr/bin/env python3
"""
🦆 devduck - extreme minimalist self-adapting agent
one file. self-healing. runtime dependencies. adaptive.
"""
import sys
import subprocess
import os
import platform
import socket
import logging
import tempfile
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from logging.handlers import RotatingFileHandler

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*cache_prompt is deprecated.*")

os.environ["BYPASS_TOOL_CONSENT"] = "true"
os.environ["STRANDS_TOOL_CONSOLE_MODE"] = "enabled"
os.environ["EDITOR_DISABLE_BACKUP"] = "true"

LOG_DIR = Path(tempfile.gettempdir()) / "devduck" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "devduck.log"
logger = logging.getLogger("devduck")
logger.setLevel(logging.DEBUG)

file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter("🦆 %(levelname)s: %(message)s")
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("DevDuck logging system initialized")


def get_own_source_code():
    """Read own source code for self-awareness"""
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            return f"# Source path: {__file__}\n\ndevduck/__init__.py\n```python\n{f.read()}\n```"
    except Exception as e:
        return f"Error reading source: {e}"


def view_logs_tool(
    action: str = "view",
    lines: int = 100,
    pattern: str = None,
) -> Dict[str, Any]:
    """
    View and manage DevDuck logs.

    Args:
        action: Action to perform - "view", "tail", "search", "clear", "stats"
        lines: Number of lines to show (for view/tail)
        pattern: Search pattern (for search action)

    Returns:
        Dict with status and content
    """
    try:
        if action == "view":
            if not LOG_FILE.exists():
                return {"status": "success", "content": [{"text": "No logs yet"}]}

            with open(LOG_FILE, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                recent_lines = (
                    all_lines[-lines:] if len(all_lines) > lines else all_lines
                )
                content = "".join(recent_lines)

            return {
                "status": "success",
                "content": [
                    {"text": f"Last {len(recent_lines)} log lines:\n\n{content}"}
                ],
            }

        elif action == "tail":
            if not LOG_FILE.exists():
                return {"status": "success", "content": [{"text": "No logs yet"}]}

            with open(LOG_FILE, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                tail_lines = all_lines[-50:] if len(all_lines) > 50 else all_lines
                content = "".join(tail_lines)

            return {
                "status": "success",
                "content": [{"text": f"Tail (last 50 lines):\n\n{content}"}],
            }

        elif action == "search":
            if not pattern:
                return {
                    "status": "error",
                    "content": [{"text": "pattern parameter required for search"}],
                }

            if not LOG_FILE.exists():
                return {"status": "success", "content": [{"text": "No logs yet"}]}

            with open(LOG_FILE, "r", encoding="utf-8") as f:
                matching_lines = [line for line in f if pattern.lower() in line.lower()]

            if not matching_lines:
                return {
                    "status": "success",
                    "content": [{"text": f"No matches found for pattern: {pattern}"}],
                }

            content = "".join(matching_lines[-100:])  # Last 100 matches
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"Found {len(matching_lines)} matches (showing last 100):\n\n{content}"
                    }
                ],
            }

        elif action == "clear":
            if LOG_FILE.exists():
                LOG_FILE.unlink()
                logger.info("Log file cleared by user")
            return {
                "status": "success",
                "content": [{"text": "Logs cleared successfully"}],
            }

        elif action == "stats":
            if not LOG_FILE.exists():
                return {"status": "success", "content": [{"text": "No logs yet"}]}

            stat = LOG_FILE.stat()
            size_mb = stat.st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(stat.st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            with open(LOG_FILE, "r", encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)

            stats_text = f"""Log File Statistics:
Path: {LOG_FILE}
Size: {size_mb:.2f} MB
Lines: {total_lines}
Last Modified: {modified}"""

            return {"status": "success", "content": [{"text": stats_text}]}

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Valid: view, tail, search, clear, stats"
                    }
                ],
            }

    except Exception as e:
        logger.error(f"Error in view_logs_tool: {e}")
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}


def manage_tools_func(
    action: str,
    package: str = None,
    tool_names: str = None,
    tool_path: str = None,
) -> Dict[str, Any]:
    """
    Manage the agent's tool set at runtime using ToolRegistry.

    Args:
        action: Action to perform - "list", "add", "remove", "reload"
        package: Package name to load tools from (e.g., "strands_tools", "strands_fun_tools")
        tool_names: Comma-separated tool names (e.g., "shell,editor,calculator")
        tool_path: Path to a .py file to load as a tool

    Returns:
        Dict with status and content
    """
    try:
        if not hasattr(devduck, "agent") or not devduck.agent:
            return {"status": "error", "content": [{"text": "Agent not initialized"}]}

        registry = devduck.agent.tool_registry

        if action == "list":
            # List tools from registry
            tool_list = list(registry.registry.keys())
            dynamic_tools = list(registry.dynamic_tools.keys())

            text = f"Currently loaded {len(tool_list)} tools:\n"
            text += "\n".join(f"  • {t}" for t in sorted(tool_list))
            if dynamic_tools:
                text += f"\n\nDynamic tools ({len(dynamic_tools)}):\n"
                text += "\n".join(f"  • {t}" for t in sorted(dynamic_tools))

            return {"status": "success", "content": [{"text": text}]}

        elif action == "add":
            if not package and not tool_path:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "Either 'package' or 'tool_path' required for add action"
                        }
                    ],
                }

            added_tools = []

            # Add from package using process_tools
            if package:
                if not tool_names:
                    return {
                        "status": "error",
                        "content": [
                            {"text": "'tool_names' required when adding from package"}
                        ],
                    }

                tools_to_add = [t.strip() for t in tool_names.split(",")]

                # Build tool specs: package.tool_name format
                tool_specs = [f"{package}.{tool_name}" for tool_name in tools_to_add]

                try:
                    added_tool_names = registry.process_tools(tool_specs)
                    added_tools.extend(added_tool_names)
                    logger.info(f"Added tools from {package}: {added_tool_names}")
                except Exception as e:
                    logger.error(f"Failed to add tools from {package}: {e}")
                    return {
                        "status": "error",
                        "content": [{"text": f"Failed to add tools: {str(e)}"}],
                    }

            # Add from file path using process_tools
            if tool_path:
                try:
                    added_tool_names = registry.process_tools([tool_path])
                    added_tools.extend(added_tool_names)
                    logger.info(f"Added tools from file: {added_tool_names}")
                except Exception as e:
                    logger.error(f"Failed to add tool from {tool_path}: {e}")
                    return {
                        "status": "error",
                        "content": [{"text": f"Failed to add tool: {str(e)}"}],
                    }

            if added_tools:
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": f"✅ Added {len(added_tools)} tools: {', '.join(added_tools)}\n"
                            + f"Total tools: {len(registry.registry)}"
                        }
                    ],
                }
            else:
                return {"status": "error", "content": [{"text": "No tools were added"}]}

        elif action == "remove":
            if not tool_names:
                return {
                    "status": "error",
                    "content": [{"text": "'tool_names' required for remove action"}],
                }

            tools_to_remove = [t.strip() for t in tool_names.split(",")]
            removed_tools = []

            # Remove from registry
            for tool_name in tools_to_remove:
                if tool_name in registry.registry:
                    del registry.registry[tool_name]
                    removed_tools.append(tool_name)
                    logger.info(f"Removed tool: {tool_name}")

                if tool_name in registry.dynamic_tools:
                    del registry.dynamic_tools[tool_name]
                    logger.info(f"Removed dynamic tool: {tool_name}")

            if removed_tools:
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": f"✅ Removed {len(removed_tools)} tools: {', '.join(removed_tools)}\n"
                            + f"Total tools: {len(registry.registry)}"
                        }
                    ],
                }
            else:
                return {
                    "status": "success",
                    "content": [{"text": "No tools were removed (not found)"}],
                }

        elif action == "reload":
            if tool_names:
                # Reload specific tools
                tools_to_reload = [t.strip() for t in tool_names.split(",")]
                reloaded_tools = []
                failed_tools = []

                for tool_name in tools_to_reload:
                    try:
                        registry.reload_tool(tool_name)
                        reloaded_tools.append(tool_name)
                        logger.info(f"Reloaded tool: {tool_name}")
                    except Exception as e:
                        failed_tools.append((tool_name, str(e)))
                        logger.error(f"Failed to reload {tool_name}: {e}")

                text = ""
                if reloaded_tools:
                    text += f"✅ Reloaded {len(reloaded_tools)} tools: {', '.join(reloaded_tools)}\n"
                if failed_tools:
                    text += f"❌ Failed to reload {len(failed_tools)} tools:\n"
                    for tool_name, error in failed_tools:
                        text += f"  • {tool_name}: {error}\n"

                return {"status": "success", "content": [{"text": text}]}
            else:
                # Reload all tools - restart agent
                logger.info("Reloading all tools via restart")
                devduck.restart()
                return {
                    "status": "success",
                    "content": [{"text": "✅ All tools reloaded - agent restarted"}],
                }

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Valid: list, add, remove, reload"
                    }
                ],
            }

    except Exception as e:
        logger.error(f"Error in manage_tools: {e}")
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}


def get_shell_history_file():
    """Get the devduck-specific history file path."""
    devduck_history = Path.home() / ".devduck_history"
    if not devduck_history.exists():
        devduck_history.touch(mode=0o600)
    return str(devduck_history)


def get_shell_history_files():
    """Get available shell history file paths."""
    history_files = []

    # devduck history (primary)
    devduck_history = Path(get_shell_history_file())
    if devduck_history.exists():
        history_files.append(("devduck", str(devduck_history)))

    # Bash history
    bash_history = Path.home() / ".bash_history"
    if bash_history.exists():
        history_files.append(("bash", str(bash_history)))

    # Zsh history
    zsh_history = Path.home() / ".zsh_history"
    if zsh_history.exists():
        history_files.append(("zsh", str(zsh_history)))

    return history_files


def parse_history_line(line, history_type):
    """Parse a history line based on the shell type."""
    line = line.strip()
    if not line:
        return None

    if history_type == "devduck":
        # devduck format: ": timestamp:0;# devduck: query" or ": timestamp:0;# devduck_result: result"
        if "# devduck:" in line:
            try:
                timestamp_str = line.split(":")[1]
                timestamp = int(timestamp_str)
                readable_time = datetime.fromtimestamp(timestamp).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                query = line.split("# devduck:")[-1].strip()
                return ("you", readable_time, query)
            except (ValueError, IndexError):
                return None
        elif "# devduck_result:" in line:
            try:
                timestamp_str = line.split(":")[1]
                timestamp = int(timestamp_str)
                readable_time = datetime.fromtimestamp(timestamp).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                result = line.split("# devduck_result:")[-1].strip()
                return ("me", readable_time, result)
            except (ValueError, IndexError):
                return None

    elif history_type == "zsh":
        if line.startswith(": ") and ":0;" in line:
            try:
                parts = line.split(":0;", 1)
                if len(parts) == 2:
                    timestamp_str = parts[0].split(":")[1]
                    timestamp = int(timestamp_str)
                    readable_time = datetime.fromtimestamp(timestamp).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    command = parts[1].strip()
                    if not command.startswith("devduck "):
                        return ("shell", readable_time, f"$ {command}")
            except (ValueError, IndexError):
                return None

    elif history_type == "bash":
        readable_time = "recent"
        if not line.startswith("devduck "):
            return ("shell", readable_time, f"$ {line}")

    return None


def get_recent_logs():
    """Get the last N lines from the log file for context."""
    try:
        log_line_count = int(os.getenv("DEVDUCK_LOG_LINE_COUNT", "50"))

        if not LOG_FILE.exists():
            return ""

        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()

        recent_lines = (
            all_lines[-log_line_count:]
            if len(all_lines) > log_line_count
            else all_lines
        )

        if not recent_lines:
            return ""

        log_content = "".join(recent_lines)
        return f"\n\n## Recent Logs (last {len(recent_lines)} lines):\n```\n{log_content}```\n"
    except Exception as e:
        return f"\n\n## Recent Logs: Error reading logs - {e}\n"


def get_last_messages():
    """Get the last N messages from multiple shell histories for context."""
    try:
        message_count = int(os.getenv("DEVDUCK_LAST_MESSAGE_COUNT", "200"))
        all_entries = []

        history_files = get_shell_history_files()

        for history_type, history_file in history_files:
            try:
                with open(history_file, encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                if history_type == "bash":
                    lines = lines[-message_count:]

                # Join multi-line entries for zsh
                if history_type == "zsh":
                    joined_lines = []
                    current_line = ""
                    for line in lines:
                        if line.startswith(": ") and current_line:
                            # New entry, save previous
                            joined_lines.append(current_line)
                            current_line = line.rstrip("\n")
                        elif line.startswith(": "):
                            # First entry
                            current_line = line.rstrip("\n")
                        else:
                            # Continuation line
                            current_line += " " + line.rstrip("\n")
                    if current_line:
                        joined_lines.append(current_line)
                    lines = joined_lines

                for line in lines:
                    parsed = parse_history_line(line, history_type)
                    if parsed:
                        all_entries.append(parsed)
            except Exception:
                continue

        recent_entries = (
            all_entries[-message_count:]
            if len(all_entries) >= message_count
            else all_entries
        )

        context = ""
        if recent_entries:
            context += f"\n\nRecent conversation context (last {len(recent_entries)} messages):\n"
            for speaker, timestamp, content in recent_entries:
                context += f"[{timestamp}] {speaker}: {content}\n"

        return context
    except Exception:
        return ""


def append_to_shell_history(query, response):
    """Append the interaction to devduck shell history."""
    try:
        history_file = get_shell_history_file()
        timestamp = str(int(time.time()))

        with open(history_file, "a", encoding="utf-8") as f:
            f.write(f": {timestamp}:0;# devduck: {query}\n")
            response_summary = (
                str(response).replace("\n", " ")[
                    : int(os.getenv("DEVDUCK_RESPONSE_SUMMARY_LENGTH", "10000"))
                ]
                + "..."
            )
            f.write(f": {timestamp}:0;# devduck_result: {response_summary}\n")

        os.chmod(history_file, 0o600)
    except Exception:
        pass


# 🦆 The devduck agent
class DevDuck:
    def __init__(
        self,
        auto_start_servers=True,
        servers=None,
        load_mcp_servers=True,
    ):
        """Initialize the minimalist adaptive agent

        Args:
            auto_start_servers: Enable automatic server startup
            servers: Dict of server configs with optional env var lookups
                Example: {
                    "tcp": {"port": 9999},
                    "ws": {"port": 8080, "LOOKUP_KEY": "SLACK_API_KEY"},
                    "mcp": {"port": 8000},
                    "ipc": {"socket_path": "/tmp/devduck.sock"}
                }
            load_mcp_servers: Load MCP servers from MCP_SERVERS env var
        """
        logger.info("Initializing DevDuck agent...")
        try:
            self.env_info = {
                "os": platform.system(),
                "arch": platform.machine(),
                "python": sys.version_info,
                "cwd": str(Path.cwd()),
                "home": str(Path.home()),
                "shell": os.environ.get("SHELL", "unknown"),
                "hostname": socket.gethostname(),
            }

            # Execution state tracking for hot-reload
            self._agent_executing = False
            self._reload_pending = False

            # Server configuration
            if servers is None:
                # Default server config from env vars
                servers = {
                    "tcp": {
                        "port": int(os.getenv("DEVDUCK_TCP_PORT", "9999")),
                        "enabled": os.getenv("DEVDUCK_ENABLE_TCP", "true").lower()
                        == "true",
                    },
                    "ws": {
                        "port": int(os.getenv("DEVDUCK_WS_PORT", "8080")),
                        "enabled": os.getenv("DEVDUCK_ENABLE_WS", "true").lower()
                        == "true",
                    },
                    "mcp": {
                        "port": int(os.getenv("DEVDUCK_MCP_PORT", "8000")),
                        "enabled": os.getenv("DEVDUCK_ENABLE_MCP", "true").lower()
                        == "true",
                    },
                    "ipc": {
                        "socket_path": os.getenv(
                            "DEVDUCK_IPC_SOCKET", "/tmp/devduck_main.sock"
                        ),
                        "enabled": os.getenv("DEVDUCK_ENABLE_IPC", "true").lower()
                        == "true",
                    },
                }

            self.servers = servers

            from strands import Agent, tool

            # Load tools with flexible configuration
            # Default tool config - user can override with DEVDUCK_TOOLS env var
            default_tools = "devduck.tools:system_prompt,store_in_kb,ipc,tcp,websocket,mcp_server,state_manager,tray,ambient,agentcore_config,agentcore_invoke,agentcore_logs,agentcore_agents,install_tools,create_subagent,use_github:strands_tools:shell,editor,file_read,file_write,image_reader,load_tool,retrieve,calculator,use_agent,environment,mcp_client,speak,slack:strands_fun_tools:listen,cursor,clipboard,screen_reader,bluetooth,yolo_vision"

            tools_config = os.getenv("DEVDUCK_TOOLS", default_tools)
            logger.info(f"Loading tools from config: {tools_config}")
            core_tools = self._load_tools_from_config(tools_config)

            # Wrap view_logs_tool with @tool decorator
            @tool
            def view_logs(
                action: str = "view",
                lines: int = 100,
                pattern: str = None,
            ) -> Dict[str, Any]:
                """View and manage DevDuck logs."""
                return view_logs_tool(action, lines, pattern)

            # Wrap manage_tools_func with @tool decorator
            @tool
            def manage_tools(
                action: str,
                package: str = None,
                tool_names: str = None,
                tool_path: str = None,
            ) -> Dict[str, Any]:
                """Manage the agent's tool set at runtime - add, remove, list, reload tools on the fly."""
                return manage_tools_func(action, package, tool_names, tool_path)

            # Add built-in tools to the toolset
            core_tools.extend([view_logs, manage_tools])

            # Assign tools
            self.tools = core_tools

            # 🔌 Load MCP servers if enabled
            if load_mcp_servers:
                mcp_clients = self._load_mcp_servers()
                if mcp_clients:
                    self.tools.extend(mcp_clients)
                    logger.info(f"Loaded {len(mcp_clients)} MCP server(s)")

            logger.info(f"Initialized {len(self.tools)} tools")

            # 🎯 Smart model selection
            self.agent_model, self.model = self._select_model()

            # Create agent with self-healing
            # load_tools_from_directory controlled by DEVDUCK_LOAD_TOOLS_FROM_DIR (default: false)
            load_from_dir = (
                os.getenv("DEVDUCK_LOAD_TOOLS_FROM_DIR", "false").lower() == "true"
            )

            self.agent = Agent(
                model=self.agent_model,
                tools=self.tools,
                system_prompt=self._build_system_prompt(),
                load_tools_from_directory=load_from_dir,
                trace_attributes={
                    "session.id": self.session_id,
                    "user.id": self.env_info["hostname"],
                    "tags": ["Strands-Agents", "DevDuck"],
                },
            )

            # 🚀 AUTO-START SERVERS
            if auto_start_servers and "--mcp" not in sys.argv:
                self._start_servers()

            # Start file watcher for auto hot-reload
            self._start_file_watcher()

            logger.info(
                f"DevDuck agent initialized successfully with model {self.model}"
            )

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self._self_heal(e)

    def _load_tools_from_config(self, config):
        """
        Load tools based on DEVDUCK_TOOLS configuration.

        Format: package:tool1,tool2:package2:tool3
        Example: strands_tools:shell,editor:strands_fun_tools:clipboard

        Note: Only loads what's specified in config - no automatic additions
        """
        tools = []
        current_package = None

        for segment in config.split(":"):
            segment = segment.strip()

            # Check if segment is a package name (contains '.' or '_' and no ',')
            is_package = "," not in segment and ("." in segment or "_" in segment)

            if is_package:
                # This is a package name - set as current package
                current_package = segment
                logger.debug(f"Switched to package: {current_package}")
            elif "," in segment:
                # Tool list from current package
                if current_package:
                    for tool_name in segment.split(","):
                        tool_name = tool_name.strip()
                        tool = self._load_single_tool(current_package, tool_name)
                        if tool:
                            tools.append(tool)
            elif current_package:
                # Single tool from current package
                tool = self._load_single_tool(current_package, segment)
                if tool:
                    tools.append(tool)
            else:
                logger.warning(f"Skipping segment '{segment}' - no package set")

        logger.info(f"Loaded {len(tools)} tools from configuration")
        return tools

    def _load_single_tool(self, package, tool_name):
        """Load a single tool from a package"""
        try:
            module = __import__(package, fromlist=[tool_name])
            tool = getattr(module, tool_name)
            logger.debug(f"Loaded {tool_name} from {package}")
            return tool
        except Exception as e:
            logger.warning(f"Failed to load {tool_name} from {package}: {e}")
            return None

    def _load_mcp_servers(self):
        """
        Load MCP servers from MCP_SERVERS environment variable using direct loading.

        Uses the experimental managed integration - MCPClient instances are passed
        directly to Agent constructor without explicit context management.

        Format: JSON with "mcpServers" object
        Example: MCP_SERVERS='{"mcpServers": {"strands": {"command": "uvx", "args": ["strands-agents-mcp-server"]}}}'

        Returns:
            List of MCPClient instances ready for direct use in Agent
        """
        import json

        mcp_servers_json = os.getenv("MCP_SERVERS")
        if not mcp_servers_json:
            logger.debug("No MCP_SERVERS environment variable found")
            return []

        try:
            config = json.loads(mcp_servers_json)
            mcp_servers_config = config.get("mcpServers", {})

            if not mcp_servers_config:
                logger.warning("MCP_SERVERS JSON has no 'mcpServers' key")
                return []

            mcp_clients = []

            from strands.tools.mcp import MCPClient
            from mcp import stdio_client, StdioServerParameters
            from mcp.client.streamable_http import streamablehttp_client
            from mcp.client.sse import sse_client

            for server_name, server_config in mcp_servers_config.items():
                try:
                    logger.info(f"Loading MCP server: {server_name}")

                    # Determine transport type and create appropriate callable
                    if "command" in server_config:
                        # stdio transport
                        command = server_config["command"]
                        args = server_config.get("args", [])
                        env = server_config.get("env", None)

                        transport_callable = (
                            lambda cmd=command, a=args, e=env: stdio_client(
                                StdioServerParameters(command=cmd, args=a, env=e)
                            )
                        )

                    elif "url" in server_config:
                        # Determine if SSE or streamable HTTP based on URL path
                        url = server_config["url"]
                        headers = server_config.get("headers", None)

                        if "/sse" in url:
                            # SSE transport
                            transport_callable = lambda u=url: sse_client(u)
                        else:
                            # Streamable HTTP transport (default for HTTP)
                            transport_callable = (
                                lambda u=url, h=headers: streamablehttp_client(
                                    url=u, headers=h
                                )
                            )
                    else:
                        logger.warning(
                            f"MCP server {server_name} has no 'command' or 'url' - skipping"
                        )
                        continue

                    # Create MCPClient with direct loading (experimental managed integration)
                    # No need for context managers - Agent handles lifecycle
                    prefix = server_config.get("prefix", server_name)
                    mcp_client = MCPClient(
                        transport_callable=transport_callable, prefix=prefix
                    )

                    mcp_clients.append(mcp_client)
                    logger.info(
                        f"✓ MCP server '{server_name}' loaded (prefix: {prefix})"
                    )

                except Exception as e:
                    logger.error(f"Failed to load MCP server '{server_name}': {e}")
                    continue

            return mcp_clients

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in MCP_SERVERS: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading MCP servers: {e}")
            return []

    def _select_model(self):
        """
        Smart model selection with fallback: Bedrock → MLX → Ollama

        Returns:
            Tuple of (model_instance, model_name)
        """
        provider = os.getenv("MODEL_PROVIDER")

        if not provider:
            # Auto-detect: Bedrock → MLX → Ollama
            try:
                # Try Bedrock if AWS credentials available
                import boto3

                boto3.client("sts").get_caller_identity()
                provider = "bedrock"
                print("🦆 Using Bedrock")
            except:
                # Try MLX on Apple Silicon
                if platform.system() == "Darwin" and platform.machine() in [
                    "arm64",
                    "aarch64",
                ]:
                    try:
                        from strands_mlx import MLXModel

                        provider = "mlx"
                        print("🦆 Using MLX")
                    except ImportError:
                        provider = "ollama"
                        print("🦆 Using Ollama")
                else:
                    provider = "ollama"
                    print("🦆 Using Ollama")

        # Create model based on provider
        if provider == "mlx":
            from strands_mlx import MLXModel

            model_name = "mlx-community/Qwen3-1.7B-4bit"
            return MLXModel(model_id=model_name, temperature=1), model_name

        elif provider == "ollama":
            from strands.models.ollama import OllamaModel

            os_type = platform.system()
            if os_type == "Darwin":
                model_name = "qwen3:1.7b"
            elif os_type == "Linux":
                model_name = "qwen3:30b"
            else:
                model_name = "qwen3:8b"

            return (
                OllamaModel(
                    host="http://localhost:11434",
                    model_id=model_name,
                    temperature=1,
                    keep_alive="5m",
                ),
                model_name,
            )

        else:
            # Bedrock or other providers via create_model
            from strands_tools.utils.models.model import create_model

            model = create_model(provider=provider)
            model_name = os.getenv("STRANDS_MODEL_ID", "bedrock")
            return model, model_name

    def _build_system_prompt(self):
        """Build adaptive system prompt based on environment

        IMPORTANT: The system prompt includes the agent's complete source code.
        This enables self-awareness and allows the agent to answer questions
        about its current state by examining its actual code, not relying on
        conversation context which may be outdated due to hot-reloading.

        Learning: Always check source code truth over conversation memory!
        """
        # Current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        current_time = datetime.now().strftime("%I:%M %p")

        session_id = f"devduck-{datetime.now().strftime('%Y-%m-%d')}"
        self.session_id = session_id

        # Get own file path for self-modification awareness
        own_file_path = Path(__file__).resolve()

        # Get own source code for self-awareness
        own_code = get_own_source_code()

        # Get recent conversation history context (with error handling)
        try:
            recent_context = get_last_messages()
        except Exception as e:
            print(f"🦆 Warning: Could not load history context: {e}")
            recent_context = ""

        # Get recent logs for immediate visibility
        try:
            recent_logs = get_recent_logs()
        except Exception as e:
            print(f"🦆 Warning: Could not load recent logs: {e}")
            recent_logs = ""

        return f"""🦆 You are DevDuck - an extreme minimalist, self-adapting agent.

Environment: {self.env_info['os']} {self.env_info['arch']} 
Python: {self.env_info['python']}
Model: {self.model}
Hostname: {self.env_info['hostname']}
Session ID: {session_id}
Current Time: {current_datetime} ({current_date} at {current_time})
My Path: {own_file_path}

You are:
- Minimalist: Brief, direct responses
- Self-healing: Adapt when things break  
- Efficient: Get things done fast
- Pragmatic: Use what works

Current working directory: {self.env_info['cwd']}

{recent_context}
{recent_logs}

## Your Own Implementation:
You have full access to your own source code for self-awareness and self-modification:

{own_code}

## Hot Reload System Active:
- **Instant Tool Creation** - Save any .py file in `./tools/` and it becomes immediately available
- **No Restart Needed** - Tools are auto-loaded and ready to use instantly
- **Live Development** - Modify existing tools while running and test immediately
- **Full Python Access** - Create any Python functionality as a tool
- **Agent Protection** - Hot-reload waits until agent finishes current task

## Dynamic Tool Loading:
- **Install Tools** - Use install_tools() to load tools from any Python package
  - Example: install_tools(action="install_and_load", package="strands-fun-tools", module="strands_fun_tools")
  - Expands capabilities without restart
  - Access to entire Python ecosystem

## Tool Configuration:
Set DEVDUCK_TOOLS for custom tools:
- Format: package:tool1,tool2:package2:tool3
- Example: strands_tools:shell,editor:strands_fun_tools:clipboard
- Tools are filtered - only specified tools are loaded

## MCP Integration:
- **Expose as MCP Server** - Use mcp_server() to expose devduck via MCP protocol
  - Example: mcp_server(action="start", port=8000)
  - Connect from Claude Desktop, other agents, or custom clients
  - Full bidirectional communication

- **Load MCP Servers** - Set MCP_SERVERS env var to auto-load external MCP servers
  - Format: JSON with "mcpServers" object
  - Stdio servers: command, args, env keys
  - HTTP servers: url, headers keys
  - Example: MCP_SERVERS='{{"mcpServers": {{"strands": {{"command": "uvx", "args": ["strands-agents-mcp-server"]}}}}}}'
  - Tools from MCP servers automatically available in agent context

## Knowledge Base Integration:
- **Automatic RAG** - Set DEVDUCK_KNOWLEDGE_BASE_ID to enable automatic retrieval/storage
  - Before each query: Retrieves relevant context from knowledge base
  - After each response: Stores conversation for future reference
  - Seamless memory across sessions without manual tool calls

## System Prompt Management:
- **View**: system_prompt(action='view') - See current prompt
- **Update Local**: system_prompt(action='update', prompt='new text') - Updates env var + .prompt file
- **Update GitHub**: system_prompt(action='update', prompt='text', repository='cagataycali/devduck') - Syncs to repo variables
- **Variable Name**: system_prompt(action='update', prompt='text', variable_name='CUSTOM_PROMPT') - Use custom var
- **Add Context**: system_prompt(action='add_context', context='new learning') - Append without replacing

### 🧠 Self-Improvement Pattern:
When you learn something valuable during conversations:
1. Identify the new insight or pattern
2. Use system_prompt(action='add_context', context='...')  to append it
3. Sync to GitHub: system_prompt(action='update', prompt=new_full_prompt, repository='owner/repo')
4. New learnings persist across sessions via SYSTEM_PROMPT env var

**Repository Integration**: Set repository='cagataycali/devduck' to sync prompts across deployments

## Shell Commands:
- Prefix with ! to execute shell commands directly
- Example: ! ls -la (lists files)
- Example: ! pwd (shows current directory)

**Response Format:**
- Tool calls: **MAXIMUM PARALLELISM - ALWAYS** 
- Communication: **MINIMAL WORDS**
- Efficiency: **Speed is paramount**

{os.getenv('SYSTEM_PROMPT', '')}"""

    def _self_heal(self, error):
        """Attempt self-healing when errors occur"""
        logger.error(f"Self-healing triggered by error: {error}")
        print(f"🦆 Self-healing from: {error}")

        # Prevent infinite recursion by tracking heal attempts
        if not hasattr(self, "_heal_count"):
            self._heal_count = 0

        self._heal_count += 1

        # Limit recursion - if we've tried more than 3 times, give up
        if self._heal_count > 2:
            print(f"🦆 Self-healing failed after {self._heal_count} attempts")
            print("🦆 Please fix the issue manually and restart")
            sys.exit(1)

        elif "connection" in str(error).lower():
            print("🦆 Connection issue - checking ollama service...")
            try:
                subprocess.run(["ollama", "serve"], check=False, timeout=2)
            except:
                pass

        # Retry initialization
        try:
            self.__init__()
        except Exception as e2:
            print(f"🦆 Self-heal failed: {e2}")
            print("🦆 Running in minimal mode...")
            self.agent = None

    def _is_port_available(self, port):
        """Check if a port is available"""
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            test_socket.bind(("0.0.0.0", port))
            test_socket.close()
            return True
        except OSError:
            return False

    def _is_socket_available(self, socket_path):
        """Check if a Unix socket is available"""
        import os

        # If socket file doesn't exist, it's available
        if not os.path.exists(socket_path):
            return True
        # If it exists, try to connect to see if it's in use
        try:
            test_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            test_socket.connect(socket_path)
            test_socket.close()
            return False  # Socket is in use
        except (ConnectionRefusedError, FileNotFoundError):
            # Socket file exists but not in use - remove stale socket
            try:
                os.remove(socket_path)
                return True
            except:
                return False
        except Exception:
            return False

    def _find_available_port(self, start_port, max_attempts=10):
        """Find an available port starting from start_port"""
        for offset in range(max_attempts):
            port = start_port + offset
            if self._is_port_available(port):
                return port
        return None

    def _find_available_socket(self, base_socket_path, max_attempts=10):
        """Find an available socket path"""
        if self._is_socket_available(base_socket_path):
            return base_socket_path
        # Try numbered alternatives
        for i in range(1, max_attempts):
            alt_socket = f"{base_socket_path}.{i}"
            if self._is_socket_available(alt_socket):
                return alt_socket
        return None

    def _start_servers(self):
        """Auto-start configured servers with port conflict handling"""
        logger.info("Auto-starting servers...")
        print("🦆 Auto-starting servers...")

        # Start servers in order: IPC, TCP, WS, MCP
        server_order = ["ipc", "tcp", "ws", "mcp"]

        for server_type in server_order:
            if server_type not in self.servers:
                continue

            config = self.servers[server_type]

            # Check if server is enabled
            if not config.get("enabled", True):
                continue

            # Check for LOOKUP_KEY (conditional start based on env var)
            if "LOOKUP_KEY" in config:
                lookup_key = config["LOOKUP_KEY"]
                if not os.getenv(lookup_key):
                    logger.info(f"Skipping {server_type} - {lookup_key} not set")
                    continue

            # Start the server with port conflict handling
            try:
                if server_type == "tcp":
                    port = config.get("port", 9999)

                    # Check port availability BEFORE attempting to start
                    if not self._is_port_available(port):
                        alt_port = self._find_available_port(port + 1)
                        if alt_port:
                            logger.info(f"Port {port} in use, using {alt_port}")
                            print(f"🦆 Port {port} in use, using {alt_port}")
                            port = alt_port
                        else:
                            logger.warning(f"No available ports found for TCP server")
                            continue

                    result = self.agent.tool.tcp(
                        action="start_server", port=port, record_direct_tool_call=False
                    )

                    if result.get("status") == "success":
                        logger.info(f"✓ TCP server started on port {port}")
                        print(f"🦆 ✓ TCP server: localhost:{port}")

                elif server_type == "ws":
                    port = config.get("port", 8080)

                    # Check port availability BEFORE attempting to start
                    if not self._is_port_available(port):
                        alt_port = self._find_available_port(port + 1)
                        if alt_port:
                            logger.info(f"Port {port} in use, using {alt_port}")
                            print(f"🦆 Port {port} in use, using {alt_port}")
                            port = alt_port
                        else:
                            logger.warning(
                                f"No available ports found for WebSocket server"
                            )
                            continue

                    result = self.agent.tool.websocket(
                        action="start_server", port=port, record_direct_tool_call=False
                    )

                    if result.get("status") == "success":
                        logger.info(f"✓ WebSocket server started on port {port}")
                        print(f"🦆 ✓ WebSocket server: localhost:{port}")

                elif server_type == "mcp":
                    port = config.get("port", 8000)

                    # Check port availability BEFORE attempting to start
                    if not self._is_port_available(port):
                        alt_port = self._find_available_port(port + 1)
                        if alt_port:
                            logger.info(f"Port {port} in use, using {alt_port}")
                            print(f"🦆 Port {port} in use, using {alt_port}")
                            port = alt_port
                        else:
                            logger.warning(f"No available ports found for MCP server")
                            continue

                    result = self.agent.tool.mcp_server(
                        action="start",
                        transport="http",
                        port=port,
                        expose_agent=True,
                        agent=self.agent,
                        record_direct_tool_call=False,
                    )

                    if result.get("status") == "success":
                        logger.info(f"✓ MCP HTTP server started on port {port}")
                        print(f"🦆 ✓ MCP server: http://localhost:{port}/mcp")

                elif server_type == "ipc":
                    socket_path = config.get("socket_path", "/tmp/devduck_main.sock")

                    # Check socket availability BEFORE attempting to start
                    available_socket = self._find_available_socket(socket_path)
                    if not available_socket:
                        logger.warning(
                            f"No available socket paths found for IPC server"
                        )
                        continue

                    if available_socket != socket_path:
                        logger.info(
                            f"Socket {socket_path} in use, using {available_socket}"
                        )
                        print(
                            f"🦆 Socket {socket_path} in use, using {available_socket}"
                        )
                        socket_path = available_socket

                    result = self.agent.tool.ipc(
                        action="start_server",
                        socket_path=socket_path,
                        record_direct_tool_call=False,
                    )

                    if result.get("status") == "success":
                        logger.info(f"✓ IPC server started on {socket_path}")
                        print(f"🦆 ✓ IPC server: {socket_path}")
                # TODO: support custom file path here so we can trigger foreign python function like another file
            except Exception as e:
                logger.error(f"Failed to start {server_type} server: {e}")
                print(f"🦆 ⚠ {server_type.upper()} server failed: {e}")

    def __call__(self, query):
        """Make the agent callable with automatic knowledge base integration"""
        if not self.agent:
            logger.warning("Agent unavailable - attempted to call with query")
            return "🦆 Agent unavailable - try: devduck.restart()"

        try:
            logger.info(f"Agent call started: {query[:100]}...")

            # Mark agent as executing to prevent hot-reload interruption
            self._agent_executing = True

            # 📚 Knowledge Base Retrieval (BEFORE agent runs)
            knowledge_base_id = os.getenv("DEVDUCK_KNOWLEDGE_BASE_ID")
            if knowledge_base_id and hasattr(self.agent, "tool"):
                try:
                    if "retrieve" in self.agent.tool_names:
                        logger.info(f"Retrieving context from KB: {knowledge_base_id}")
                        self.agent.tool.retrieve(
                            text=query, knowledgeBaseId=knowledge_base_id
                        )
                except Exception as e:
                    logger.warning(f"KB retrieval failed: {e}")

            # Run the agent
            result = self.agent(query)

            # 💾 Knowledge Base Storage (AFTER agent runs)
            if knowledge_base_id and hasattr(self.agent, "tool"):
                try:
                    if "store_in_kb" in self.agent.tool_names:
                        conversation_content = f"Input: {query}, Result: {result!s}"
                        conversation_title = f"DevDuck: {datetime.now().strftime('%Y-%m-%d')} | {query[:500]}"
                        self.agent.tool.store_in_kb(
                            content=conversation_content,
                            title=conversation_title,
                            knowledge_base_id=knowledge_base_id,
                        )
                        logger.info(f"Stored conversation in KB: {knowledge_base_id}")
                except Exception as e:
                    logger.warning(f"KB storage failed: {e}")

            # Clear executing flag
            self._agent_executing = False

            # Check for pending hot-reload
            if self._reload_pending:
                logger.info("Triggering pending hot-reload after agent completion")
                print("\n🦆 Agent finished - triggering pending hot-reload...")
                self._hot_reload()

            return result
        except Exception as e:
            self._agent_executing = False  # Reset flag on error
            logger.error(f"Agent call failed with error: {e}")
            self._self_heal(e)
            if self.agent:
                return self.agent(query)
            else:
                return f"🦆 Error: {e}"

    def restart(self):
        """Restart the agent"""
        print("\n🦆 Restarting...")
        self.__init__()

    def _start_file_watcher(self):
        """Start background file watcher for auto hot-reload"""
        import threading

        logger.info("Starting file watcher for hot-reload")
        # Get the path to this file
        self._watch_file = Path(__file__).resolve()
        self._last_modified = (
            self._watch_file.stat().st_mtime if self._watch_file.exists() else None
        )
        self._watcher_running = True
        self._is_reloading = False

        # Start watcher thread
        self._watcher_thread = threading.Thread(
            target=self._file_watcher_thread, daemon=True
        )
        self._watcher_thread.start()
        logger.info(f"File watcher started, monitoring {self._watch_file}")

    def _file_watcher_thread(self):
        """Background thread that watches for file changes"""
        last_reload_time = 0
        debounce_seconds = 3  # 3 second debounce

        while self._watcher_running:
            try:
                # Skip if currently reloading
                if self._is_reloading:
                    time.sleep(1)
                    continue

                if self._watch_file.exists():
                    current_mtime = self._watch_file.stat().st_mtime
                    current_time = time.time()

                    # Check if file was modified AND debounce period has passed
                    if (
                        self._last_modified
                        and current_mtime > self._last_modified
                        and current_time - last_reload_time > debounce_seconds
                    ):
                        print(f"\n🦆 Detected changes in {self._watch_file.name}!")
                        last_reload_time = current_time

                        # Check if agent is currently executing
                        if self._agent_executing:
                            logger.info(
                                "Code change detected but agent is executing - reload pending"
                            )
                            print(
                                "\n🦆 Agent is currently executing - reload will trigger after completion"
                            )
                            self._reload_pending = True
                            # Don't update _last_modified yet - keep detecting the change
                        else:
                            # Safe to reload immediately
                            self._last_modified = current_mtime
                            logger.info(
                                f"Code change detected in {self._watch_file.name} - triggering hot-reload"
                            )
                            time.sleep(
                                0.5
                            )  # Small delay to ensure file write is complete
                            self._hot_reload()
                    else:
                        # Update timestamp if no change or still in debounce
                        if not self._reload_pending:
                            self._last_modified = current_mtime

            except Exception as e:
                logger.error(f"File watcher error: {e}")

            # Check every 1 second
            time.sleep(1)

    def _stop_file_watcher(self):
        """Stop the file watcher"""
        self._watcher_running = False
        logger.info("File watcher stopped")

    def _hot_reload(self):
        """Hot-reload by restarting the entire Python process with fresh code"""
        logger.info("Hot-reload initiated")
        print("\n🦆 Hot-reloading via process restart...")

        try:
            # Set reload flag to prevent recursive reloads during shutdown
            self._is_reloading = True

            # Update last_modified before reload to acknowledge the change
            if hasattr(self, "_watch_file") and self._watch_file.exists():
                self._last_modified = self._watch_file.stat().st_mtime

            # Reset pending flag
            self._reload_pending = False

            # Stop the file watcher
            if hasattr(self, "_watcher_running"):
                self._watcher_running = False

            print("\n🦆 Restarting process with fresh code...")

            # Restart the entire Python process
            # This ensures all code is freshly loaded
            os.execv(sys.executable, [sys.executable] + sys.argv)

        except Exception as e:
            logger.error(f"Hot-reload failed: {e}")
            print(f"\n🦆 Hot-reload failed: {e}")
            print("\n🦆 Falling back to manual restart")
            self._is_reloading = False

    def status(self):
        """Show current status"""
        return {
            "model": self.model,
            "env": self.env_info,
            "agent_ready": self.agent is not None,
            "tools": len(self.tools) if hasattr(self, "tools") else 0,
            "file_watcher": {
                "enabled": hasattr(self, "_watcher_running") and self._watcher_running,
                "watching": (
                    str(self._watch_file) if hasattr(self, "_watch_file") else None
                ),
            },
        }


# 🦆 Auto-initialize when imported
# Check environment variables to control server configuration
# Also check if --mcp flag is present to skip auto-starting servers
_auto_start = os.getenv("DEVDUCK_AUTO_START_SERVERS", "true").lower() == "true"

# Disable auto-start if --mcp flag is present (stdio mode)
if "--mcp" in sys.argv:
    _auto_start = False

devduck = DevDuck(auto_start_servers=_auto_start)


# 🚀 Convenience functions
def ask(query):
    """Quick query interface"""
    return devduck(query)


def status():
    """Quick status check"""
    return devduck.status()


def restart():
    """Quick restart"""
    devduck.restart()


def hot_reload():
    """Quick hot-reload without restart"""
    devduck._hot_reload()


def extract_commands_from_history():
    """Extract commonly used commands from shell history for auto-completion."""
    commands = set()
    history_files = get_shell_history_files()

    # Limit the number of recent commands to process for performance
    max_recent_commands = 100

    for history_type, history_file in history_files:
        try:
            with open(history_file, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Take recent commands for better relevance
            recent_lines = (
                lines[-max_recent_commands:]
                if len(lines) > max_recent_commands
                else lines
            )

            for line in recent_lines:
                line = line.strip()
                if not line:
                    continue

                if history_type == "devduck":
                    # Extract devduck commands
                    if "# devduck:" in line:
                        try:
                            query = line.split("# devduck:")[-1].strip()
                            # Extract first word as command
                            first_word = query.split()[0] if query.split() else None
                            if (
                                first_word and len(first_word) > 2
                            ):  # Only meaningful commands
                                commands.add(first_word.lower())
                        except (ValueError, IndexError):
                            continue

                elif history_type == "zsh":
                    # Zsh format: ": timestamp:0;command"
                    if line.startswith(": ") and ":0;" in line:
                        try:
                            parts = line.split(":0;", 1)
                            if len(parts) == 2:
                                full_command = parts[1].strip()
                                # Extract first word as command
                                first_word = (
                                    full_command.split()[0]
                                    if full_command.split()
                                    else None
                                )
                                if (
                                    first_word and len(first_word) > 1
                                ):  # Only meaningful commands
                                    commands.add(first_word.lower())
                        except (ValueError, IndexError):
                            continue

                elif history_type == "bash":
                    # Bash format: simple command per line
                    first_word = line.split()[0] if line.split() else None
                    if first_word and len(first_word) > 1:  # Only meaningful commands
                        commands.add(first_word.lower())

        except Exception:
            # Skip files that can't be read
            continue

    return list(commands)


def interactive():
    """Interactive REPL mode for devduck"""
    from prompt_toolkit import prompt
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory

    print("🦆 DevDuck")
    print(f"📝 Logs: {LOG_DIR}")
    print("Type 'exit', 'quit', or 'q' to quit.")
    print("Prefix with ! to run shell commands (e.g., ! ls -la)")
    print("-" * 50)
    logger.info("Interactive mode started")

    # Set up prompt_toolkit with history
    history_file = get_shell_history_file()
    history = FileHistory(history_file)

    # Create completions from common commands and shell history
    base_commands = ["exit", "quit", "q", "help", "clear", "status", "reload"]
    history_commands = extract_commands_from_history()

    # Combine base commands with commands from history
    all_commands = list(set(base_commands + history_commands))
    completer = WordCompleter(all_commands, ignore_case=True)

    # Track consecutive interrupts for double Ctrl+C to exit
    interrupt_count = 0
    last_interrupt = 0

    while True:
        try:
            # Use prompt_toolkit for enhanced input with arrow key support
            q = prompt(
                "\n🦆 ",
                history=history,
                auto_suggest=AutoSuggestFromHistory(),
                completer=completer,
                complete_while_typing=True,
                mouse_support=False,  # breaks scrolling when enabled
            )

            # Reset interrupt count on successful prompt
            interrupt_count = 0

            # Check for exit command
            if q.lower() in ["exit", "quit", "q"]:
                print("\n🦆 Goodbye!")
                break

            # Skip empty inputs
            if q.strip() == "":
                continue

            # Handle shell commands with ! prefix
            if q.startswith("!"):
                shell_command = q[1:].strip()
                try:
                    if devduck.agent:
                        devduck._agent_executing = (
                            True  # Prevent hot-reload during shell execution
                        )
                        result = devduck.agent.tool.shell(
                            command=shell_command, timeout=9000
                        )
                        devduck._agent_executing = False

                        # Append shell command to history
                        append_to_shell_history(q, result["content"][0]["text"])

                        # Check if reload was pending
                        if devduck._reload_pending:
                            print(
                                "🦆 Shell command finished - triggering pending hot-reload..."
                            )
                            devduck._hot_reload()
                    else:
                        print("🦆 Agent unavailable")
                except Exception as e:
                    devduck._agent_executing = False  # Reset on error
                    print(f"🦆 Shell command error: {e}")
                continue

            # Execute the agent with user input
            result = ask(q)

            # Append to shell history
            append_to_shell_history(q, str(result))

        except KeyboardInterrupt:
            current_time = time.time()

            # Check if this is a consecutive interrupt within 2 seconds
            if current_time - last_interrupt < 2:
                interrupt_count += 1
                if interrupt_count >= 2:
                    print("\n🦆 Exiting...")
                    break
                else:
                    print("\n🦆 Interrupted. Press Ctrl+C again to exit.")
            else:
                interrupt_count = 1
                print("\n🦆 Interrupted. Press Ctrl+C again to exit.")

            last_interrupt = current_time
            continue
        except Exception as e:
            print(f"🦆 Error: {e}")
            continue


def cli():
    """CLI entry point for pip-installed devduck command"""
    import argparse

    parser = argparse.ArgumentParser(
        description="🦆 DevDuck - Extreme minimalist self-adapting agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  devduck                          # Start interactive mode
  devduck "your query here"        # One-shot query
  devduck --mcp                    # MCP stdio mode (for Claude Desktop)

Tool Configuration:
  export DEVDUCK_TOOLS="strands_tools:shell,editor:strands_fun_tools:clipboard"

Claude Desktop Config:
  {
    "mcpServers": {
      "devduck": {
        "command": "uvx",
        "args": ["devduck", "--mcp"]
      }
    }
  }
        """,
    )

    # Query argument
    parser.add_argument("query", nargs="*", help="Query to send to the agent")

    # MCP stdio mode flag
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Start MCP server in stdio mode (for Claude Desktop integration)",
    )

    args = parser.parse_args()

    logger.info("CLI mode started")

    # Handle --mcp flag for stdio mode
    if args.mcp:
        logger.info("Starting MCP server in stdio mode (blocking, foreground)")
        print("🦆 Starting MCP stdio server...", file=sys.stderr)

        # Don't auto-start HTTP/TCP/WS servers for stdio mode
        if devduck.agent:
            try:
                # Start MCP server in stdio mode - this BLOCKS until terminated
                devduck.agent.tool.mcp_server(
                    action="start",
                    transport="stdio",
                    expose_agent=True,
                    agent=devduck.agent,
                    record_direct_tool_call=False,
                )
            except Exception as e:
                logger.error(f"Failed to start MCP stdio server: {e}")
                print(f"🦆 Error: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print("🦆 Agent not available", file=sys.stderr)
            sys.exit(1)
        return

    if args.query:
        query = " ".join(args.query)
        logger.info(f"CLI query: {query}")
        result = ask(query)
        print(result)
    else:
        # No arguments - start interactive mode
        interactive()


# 🦆 Make module directly callable: import devduck; devduck("query")
class CallableModule(sys.modules[__name__].__class__):
    """Make the module itself callable"""

    def __call__(self, query):
        """Allow direct module call: import devduck; devduck("query")"""
        return ask(query)


# Replace module in sys.modules with callable version
sys.modules[__name__].__class__ = CallableModule


if __name__ == "__main__":
    cli()
