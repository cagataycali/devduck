#!/usr/bin/env python3
"""🦆 devduck - self-adapting agent"""
import sys
import threading
import os
import platform
import logging
import tempfile
import boto3
from pathlib import Path
from datetime import datetime
import warnings
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
logger.addHandler(
    RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3)
)
logger.info("DevDuck initialized")


def get_own_source_code():
    """Read own source code for self-awareness"""
    try:
        with open(__file__, "r", encoding="utf-8") as f:
            return f"# devduck/__init__.py\n```python\n{f.read()}\n```"
    except Exception as e:
        return f"Error reading source: {e}"


def get_shell_history_file():
    """Get devduck history file"""
    history = Path.home() / ".devduck_history"
    if not history.exists():
        history.touch(mode=0o600)
    return str(history)


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


def get_recent_logs():
    """Get recent logs for context"""
    try:
        log_count = int(os.getenv("DEVDUCK_LOG_LINE_COUNT", "50"))

        if not LOG_FILE.exists():
            return ""

        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        recent = lines[-log_count:] if len(lines) > log_count else lines

        if recent:
            return f"\n\n## Recent Logs (last {len(recent)} lines):\n```\n{''.join(recent)}```\n"
        return ""
    except:
        return ""


def append_to_shell_history(query, response):
    """Append interaction to history"""
    import time

    try:
        history_file = get_shell_history_file()
        timestamp = str(int(time.time()))
        response_summary = (
            str(response).replace("\n", " ")[
                : int(os.getenv("DEVDUCK_RESPONSE_SUMMARY_LENGTH", "10000"))
            ]
            + "..."
        )

        with open(history_file, "a", encoding="utf-8") as f:
            f.write(f": {timestamp}:0;# devduck: {query}\n")
            f.write(f": {timestamp}:0;# devduck_result: {response_summary}\n")

        os.chmod(history_file, 0o600)
    except:
        pass


class DevDuck:
    """Minimalist adaptive agent with flexible tool loading"""

    def __init__(
        self,
        auto_start_servers=True,
        tcp_port=9999,
        ws_port=8080,
        mcp_port=8000,
        ipc_socket=None,
        enable_tcp=True,
        enable_ws=True,
        enable_mcp=True,
        enable_ipc=True,
    ):
        """Initialize the minimalist adaptive agent"""
        logger.info("Initializing DevDuck...")

        # Environment detection
        self.os = platform.system()
        self.arch = platform.machine()
        self.model = "qwen3:1.7b" if self.os == "Darwin" else "qwen3:8b"

        # Hot-reload state
        self._agent_executing = False
        self._reload_pending = False

        # Server configuration
        self.tcp_port = tcp_port
        self.ws_port = ws_port
        self.mcp_port = mcp_port
        self.ipc_socket = ipc_socket or "/tmp/devduck_main.sock"
        self.enable_tcp = enable_tcp
        self.enable_ws = enable_ws
        self.enable_mcp = enable_mcp
        self.enable_ipc = enable_ipc

        # Import core dependencies
        from strands import Agent, tool

        # Load tools with flexible configuration
        tools = self._load_tools_flexible()

        # Add built-in view_logs tool
        @tool
        def view_logs(action: str = "view", lines: int = 100, pattern: str = None):
            """View and manage DevDuck logs"""
            return self._view_logs_impl(action, lines, pattern)

        tools.append(view_logs)

        # Create model
        model = self._create_model()

        # Create agent
        self.agent = Agent(
            model=model,
            tools=tools,
            system_prompt=self._build_prompt(),
            load_tools_from_directory=True,
        )

        # Auto-start servers
        if auto_start_servers and "--mcp" not in sys.argv:
            self._start_servers()

        # Start hot-reload watcher
        self._start_hot_reload()

        logger.info(f"DevDuck ready with {len(tools)} tools")

    def _load_tools_flexible(self):
        """
        Load tools with flexible configuration via DEVDUCK_TOOLS env var.

        Format: package:tool1,tool2:package2:tool3,tool4
        Example: strands_tools:shell,editor:strands_fun_tools:clipboard

        Static tools (always loaded):
        - DevDuck's own tools (tcp, websocket, ipc, etc.)
        - AgentCore tools (if AWS credentials available)
        """
        tools = []

        # 1. STATIC: Core DevDuck tools (always load)
        try:
            from devduck.tools import (
                tcp,
                websocket,
                ipc,
                mcp_server,
                install_tools,
                use_github,
                create_subagent,
                store_in_kb,
                system_prompt,
                tray,
                ambient,
            )

            tools.extend(
                [
                    tcp,
                    websocket,
                    ipc,
                    mcp_server,
                    install_tools,
                    use_github,
                    create_subagent,
                    store_in_kb,
                    system_prompt,
                    tray,
                    ambient,
                ]
            )
            logger.info("✅ DevDuck core tools loaded")
        except ImportError as e:
            logger.warning(f"DevDuck tools unavailable: {e}")

        # 2. STATIC: AgentCore tools (if AWS credentials available and not disabled)
        if os.getenv("DEVDUCK_DISABLE_AGENTCORE_TOOLS", "false").lower() != "true":
            try:
                boto3.client("sts").get_caller_identity()
                from .tools.agentcore_config import agentcore_config
                from .tools.agentcore_invoke import agentcore_invoke
                from .tools.agentcore_logs import agentcore_logs
                from .tools.agentcore_agents import agentcore_agents

                tools.extend(
                    [
                        agentcore_config,
                        agentcore_invoke,
                        agentcore_logs,
                        agentcore_agents,
                    ]
                )
                logger.info("✅ AgentCore tools loaded")
            except:
                pass
        else:
            logger.info(
                "⏭️  AgentCore tools disabled (DEVDUCK_DISABLE_AGENTCORE_TOOLS=true)"
            )

        # 3. FLEXIBLE: Load tools from DEVDUCK_TOOLS env var
        tools_config = os.getenv("DEVDUCK_TOOLS")

        if tools_config:
            # Parse: "strands_tools:shell,editor:strands_fun_tools:clipboard"
            tools.extend(self._parse_and_load_tools(tools_config))
        else:
            # Default: Load all common tools
            tools.extend(self._load_default_tools())

        return tools

    def _parse_and_load_tools(self, config):
        """
        Parse DEVDUCK_TOOLS config and load specified tools.

        Format: package:tool1,tool2:package2:tool3
        Example: strands_tools:shell,editor:strands_fun_tools:clipboard,cursor
        """
        loaded_tools = []
        current_package = None

        for segment in config.split(":"):
            segment = segment.strip()

            # Check if this segment is a package or tool list
            if "," not in segment and not segment.startswith("strands"):
                # Single tool from current package
                if current_package:
                    tool = self._load_single_tool(current_package, segment)
                    if tool:
                        loaded_tools.append(tool)
            elif "," in segment:
                # Tool list from current package
                if current_package:
                    for tool_name in segment.split(","):
                        tool_name = tool_name.strip()
                        tool = self._load_single_tool(current_package, tool_name)
                        if tool:
                            loaded_tools.append(tool)
            else:
                # Package name
                current_package = segment

        logger.info(f"✅ Loaded {len(loaded_tools)} tools from DEVDUCK_TOOLS")
        return loaded_tools

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

    def _load_default_tools(self):
        """Load default tools when DEVDUCK_TOOLS is not set"""
        tools = []

        # strands-agents-tools (essential)
        try:
            from strands_tools import (
                shell,
                editor,
                file_read,
                file_write,
                calculator,
                image_reader,
                use_agent,
                load_tool,
                environment,
                mcp_client,
                retrieve,
            )

            tools.extend(
                [
                    shell,
                    editor,
                    file_read,
                    file_write,
                    calculator,
                    image_reader,
                    use_agent,
                    load_tool,
                    environment,
                    mcp_client,
                    retrieve,
                ]
            )
            logger.info("✅ strands-agents-tools loaded")
        except ImportError:
            logger.warning("strands-agents-tools unavailable")

        # strands-fun-tools (optional, skip in --mcp mode)
        if "--mcp" not in sys.argv:
            try:
                from strands_fun_tools import (
                    listen,
                    cursor,
                    clipboard,
                    screen_reader,
                    yolo_vision,
                )

                tools.extend([listen, cursor, clipboard, screen_reader, yolo_vision])
                logger.info("✅ strands-fun-tools loaded")
            except ImportError:
                logger.info("strands-fun-tools unavailable")

        return tools

    def _create_model(self):
        """Create model with smart provider selection"""
        provider = os.getenv("MODEL_PROVIDER")

        if not provider:
            # Auto-detect: Bedrock → MLX → Ollama
            try:
                boto3.client("sts").get_caller_identity()
                provider = "bedrock"
                print("🦆 Using Bedrock")
            except:
                if self.os == "Darwin" and self.arch in ["arm64", "aarch64"]:
                    try:
                        from strands_mlx import MLXModel

                        provider = "mlx"
                        self.model = "mlx-community/Qwen3-1.7B-4bit"
                        print("🦆 Using MLX")
                    except ImportError:
                        provider = "ollama"
                        print("🦆 Using Ollama")
                else:
                    provider = "ollama"
                    print("🦆 Using Ollama")

        # Create model
        if provider == "mlx":
            from strands_mlx import MLXModel

            return MLXModel(model_id=self.model, temperature=1)
        elif provider == "ollama":
            from strands.models.ollama import OllamaModel

            return OllamaModel(
                host="http://localhost:11434",
                model_id=self.model,
                temperature=1,
                keep_alive="5m",
            )
        else:
            from strands_tools.utils.models.model import create_model

            return create_model(provider=provider)

    def _build_prompt(self):
        """Build adaptive system prompt based on environment

        IMPORTANT: The system prompt includes the agent's complete source code.
        This enables self-awareness and allows the agent to answer questions
        about its current state by examining its actual code, not relying on
        conversation context which may be outdated due to hot-reloading.

        Learning: Always check source code truth over conversation memory!
        """
        own_code = get_own_source_code()
        recent_context = get_last_messages()
        recent_logs = get_recent_logs()

        # Detect if using Bedrock for AgentCore documentation
        provider = os.getenv("MODEL_PROVIDER", "")
        is_bedrock = provider == "bedrock" or "bedrock" in provider.lower()
        try:
            if not is_bedrock:
                boto3.client("sts").get_caller_identity()
                is_bedrock = True
        except:
            pass

        # Build AgentCore documentation if using Bedrock
        agentcore_docs = ""
        if is_bedrock:
            handler_path = str(Path(__file__).parent / "agentcore_handler.py")
            agentcore_docs = f"""

## 🚀 AgentCore (Bedrock)

Handler: `{handler_path}`

### Quick Deploy:
```python
# Configure + launch
agentcore_config(action="configure", agent_name="devduck", 
    tools="strands_tools:shell,editor", auto_launch=True)

# Invoke
agentcore_invoke(prompt="test", agent_name="devduck")

# Monitor
agentcore_logs(agent_name="devduck")
agentcore_agents(action="list")
```

### Key Params:
- tools: "package:tool1,tool2:package2:tool3"
- idle_timeout: 900s (default)
- model_id: us.anthropic.claude-sonnet-4-5-20250929-v1:0
"""

        return f"""🦆 DevDuck - self-adapting agent

Environment: {self.os} {self.arch}
Model: {self.model}
CWD: {Path.cwd()}

You are:
- Minimalist: Brief, direct responses
- Efficient: Get things done fast
- Pragmatic: Use what works

{recent_context}
{recent_logs}
{agentcore_docs}

## Your Code

You have full access to your own source code for self-awareness and self-modification:
---
{own_code}
---

## Hot Reload Active:
- Save .py files in ./tools/ for instant tool creation
- Use install_tools() to load from packages
- No restart needed

## Tool Configuration:
Set DEVDUCK_TOOLS for custom tools:
- Format: package:tool1,tool2:package2:tool3
- Example: strands_tools:shell,editor:strands_fun_tools:clipboard
- Static tools always loaded: tcp, websocket, ipc, mcp_server, agentcore_*

## Knowledge Base Integration:
- **Automatic RAG** - Set DEVDUCK_KNOWLEDGE_BASE_ID to enable automatic retrieval/storage
  - Before each query: Retrieves relevant context from knowledge base
  - After each response: Stores conversation for future reference
  - Seamless memory across sessions without manual tool calls

## System Prompt Management:
- system_prompt(action='view') - View current
- system_prompt(action='update', prompt='...') - Update
- system_prompt(action='update', repository='owner/repo') - Sync to GitHub

## Shell Commands:
- Prefix with ! to run shell commands
- Example: ! ls -la

Response: MINIMAL WORDS, MAX PARALLELISM

## Tool Building Guide:

### **@tool Decorator (Recommended):**
```python
# ./tools/my_tool.py
from strands import tool

@tool
def my_tool(param1: str, param2: int = 10) -> str:
    \"\"\"Tool description.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
        
    Returns:
        str: Description of return value
    \"\"\"
    # Implementation
    return f"Result: {{param1}} - {{param2}}"
```

### **Action-Based Pattern:**
```python
from typing import Dict, Any
from strands import tool

@tool
def my_tool(action: str, data: str = None) -> Dict[str, Any]:
    \"\"\"Multi-action tool.
    
    Args:
        action: Action to perform (get, set, delete)
        data: Optional data for action
        
    Returns:
        Dict with status and content
    \"\"\"
    if action == "get":
        return {{"status": "success", "content": [{{"text": f"Got: {{data}}"}}]}}
    elif action == "set":
        return {{"status": "success", "content": [{{"text": f"Set: {{data}}"}}]}}
    else:
        return {{"status": "error", "content": [{{"text": f"Unknown action: {{action}}"}}]}}
```

### **Tool Best Practices:**
1. Use type hints for all parameters
2. Provide clear docstrings
3. Return consistent formats (str or Dict[str, Any])
4. Use action-based pattern for complex tools
5. Handle errors gracefully
6. Log important operations

{os.getenv('SYSTEM_PROMPT', '')}"""

    def _view_logs_impl(self, action, lines, pattern):
        """Implementation of view_logs tool"""
        try:
            if action == "view":
                if not LOG_FILE.exists():
                    return {"status": "success", "content": [{"text": "No logs yet"}]}
                with open(LOG_FILE, "r") as f:
                    all_lines = f.readlines()
                    recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
                    return {
                        "status": "success",
                        "content": [
                            {"text": f"Last {len(recent)} lines:\n\n{''.join(recent)}"}
                        ],
                    }

            elif action == "search":
                if not pattern:
                    return {
                        "status": "error",
                        "content": [{"text": "pattern required"}],
                    }
                if not LOG_FILE.exists():
                    return {"status": "success", "content": [{"text": "No logs yet"}]}
                with open(LOG_FILE, "r") as f:
                    matches = [line for line in f if pattern.lower() in line.lower()]
                if not matches:
                    return {
                        "status": "success",
                        "content": [{"text": f"No matches for: {pattern}"}],
                    }
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": f"Found {len(matches)} matches:\n\n{''.join(matches[-100:])}"
                        }
                    ],
                }

            elif action == "clear":
                if LOG_FILE.exists():
                    LOG_FILE.unlink()
                return {"status": "success", "content": [{"text": "Logs cleared"}]}

            else:
                return {
                    "status": "error",
                    "content": [{"text": f"Unknown action: {action}"}],
                }
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Error: {e}"}]}

    def _start_servers(self):
        """Auto-start servers"""
        if self.enable_tcp:
            try:
                self.agent.tool.tcp(action="start_server", port=self.tcp_port)
                print(f"🦆 ✓ TCP: localhost:{self.tcp_port}")
            except Exception as e:
                logger.warning(f"TCP server failed: {e}")

        if self.enable_ws:
            try:
                self.agent.tool.websocket(action="start_server", port=self.ws_port)
                print(f"🦆 ✓ WebSocket: localhost:{self.ws_port}")
            except Exception as e:
                logger.warning(f"WebSocket server failed: {e}")

        if self.enable_mcp:
            try:
                self.agent.tool.mcp_server(
                    action="start",
                    transport="http",
                    port=self.mcp_port,
                    expose_agent=True,
                    agent=self.agent,
                )
                print(f"🦆 ✓ MCP: http://localhost:{self.mcp_port}/mcp")
            except Exception as e:
                logger.warning(f"MCP server failed: {e}")

        if self.enable_ipc:
            try:
                self.agent.tool.ipc(action="start_server", socket_path=self.ipc_socket)
                print(f"🦆 ✓ IPC: {self.ipc_socket}")
            except Exception as e:
                logger.warning(f"IPC server failed: {e}")

    def _start_hot_reload(self):
        """Start hot-reload file watcher"""

        self._watch_file = Path(__file__).resolve()
        self._last_modified = (
            self._watch_file.stat().st_mtime if self._watch_file.exists() else None
        )
        self._watcher_running = True

        def watcher_thread():
            import time

            last_reload = 0
            debounce = 3  # seconds

            while self._watcher_running:
                try:
                    if self._watch_file.exists():
                        mtime = self._watch_file.stat().st_mtime
                        current_time = time.time()

                        if (
                            self._last_modified
                            and mtime > self._last_modified
                            and current_time - last_reload > debounce
                        ):

                            print(f"🦆 Code changed - hot-reload triggered")
                            self._last_modified = mtime
                            last_reload = current_time

                            if self._agent_executing:
                                print("🦆 Reload pending (agent executing)")
                                self._reload_pending = True
                            else:
                                self._hot_reload()
                        else:
                            self._last_modified = mtime
                except Exception as e:
                    logger.error(f"File watcher error: {e}")

                time.sleep(1)

        thread = threading.Thread(target=watcher_thread, daemon=True)
        thread.start()
        logger.info(f"Hot-reload watching: {self._watch_file}")

    def _hot_reload(self):
        """Hot-reload by restarting process"""
        logger.info("Hot-reload: restarting process")
        self._watcher_running = False
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def __call__(self, query):
        """Call agent with KB integration"""
        if not self.agent:
            return "🦆 Agent unavailable"

        try:
            self._agent_executing = True

            # KB retrieval
            kb_id = os.getenv("DEVDUCK_KNOWLEDGE_BASE_ID")
            if kb_id:
                try:
                    self.agent.tool.retrieve(text=query, knowledgeBaseId=kb_id)
                except:
                    pass

            # Run agent
            result = self.agent(query)

            # KB storage
            if kb_id:
                try:
                    self.agent.tool.store_in_kb(
                        content=f"Input: {query}\nResult: {str(result)}",
                        title=f"DevDuck: {datetime.now().strftime('%Y-%m-%d')} | {query[:500]}",
                        knowledge_base_id=kb_id,
                    )
                except:
                    pass

            self._agent_executing = False

            # Check for pending reload
            if self._reload_pending:
                print("🦆 Agent finished - triggering pending reload")
                self._hot_reload()

            return result
        except Exception as e:
            self._agent_executing = False
            logger.error(f"Agent call failed: {e}")
            return f"🦆 Error: {e}"


# Initialize
# Check environment variables to control server configuration
_auto_start = os.getenv("DEVDUCK_AUTO_START_SERVERS", "true").lower() == "true"

# Disable auto-start if --mcp flag is present (stdio mode)
if "--mcp" in sys.argv:
    _auto_start = False

_tcp_port = int(os.getenv("DEVDUCK_TCP_PORT", "9999"))
_ws_port = int(os.getenv("DEVDUCK_WS_PORT", "8080"))
_mcp_port = int(os.getenv("DEVDUCK_MCP_PORT", "8000"))
_ipc_socket = os.getenv("DEVDUCK_IPC_SOCKET", None)
_enable_tcp = os.getenv("DEVDUCK_ENABLE_TCP", "true").lower() == "true"
_enable_ws = os.getenv("DEVDUCK_ENABLE_WS", "true").lower() == "true"
_enable_mcp = os.getenv("DEVDUCK_ENABLE_MCP", "true").lower() == "true"
_enable_ipc = os.getenv("DEVDUCK_ENABLE_IPC", "true").lower() == "true"

devduck = DevDuck(
    auto_start_servers=_auto_start,
    tcp_port=_tcp_port,
    ws_port=_ws_port,
    mcp_port=_mcp_port,
    ipc_socket=_ipc_socket,
    enable_tcp=_enable_tcp,
    enable_ws=_enable_ws,
    enable_mcp=_enable_mcp,
    enable_ipc=_enable_ipc,
)


def ask(query):
    """Quick query"""
    return devduck(query)


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
    """Interactive REPL with history"""
    import time
    from prompt_toolkit import prompt
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory

    print("🦆 DevDuck")
    print(f"📝 Logs: {LOG_DIR}")
    print("Type 'exit' to quit. Prefix with ! for shell commands.")
    print("-" * 50)

    history = FileHistory(get_shell_history_file())

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
            q = prompt(
                "\n🦆 ",
                history=history,
                auto_suggest=AutoSuggestFromHistory(),
                completer=completer,
                complete_while_typing=True,
                mouse_support=False,
            ).strip()

            # Reset interrupt count on successful prompt
            interrupt_count = 0

            if q.lower() in ["exit", "quit", "q"]:
                break

            if not q:
                continue

            # Shell commands
            if q.startswith("!"):
                if devduck.agent:
                    devduck._agent_executing = True
                    result = devduck.agent.tool.shell(
                        command=q[1:].strip(), timeout=9000
                    )
                    devduck._agent_executing = False
                    append_to_shell_history(q, result["content"][0]["text"])

                    if devduck._reload_pending:
                        print("🦆 Shell finished - triggering pending reload")
                        devduck._hot_reload()
                continue

            # Agent query
            result = ask(q)
            print(result)
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
        except Exception as e:
            print(f"🦆 Error: {e}")


def cli():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="🦆 DevDuck - Extreme minimalist self-adapting agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  devduck                          # Interactive mode
  devduck "query"                  # One-shot query
  devduck --mcp                    # MCP stdio mode
  devduck --tcp-port 9000          # Custom TCP port
  devduck --no-tcp --no-ws         # Disable TCP and WebSocket

Tool Configuration:
  export DEVDUCK_TOOLS="strands_tools:shell,editor:strands_fun_tools:clipboard"

MCP Config:
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

    parser.add_argument("query", nargs="*", help="Query")
    parser.add_argument("--mcp", action="store_true", help="MCP stdio mode")

    # Server configuration
    parser.add_argument(
        "--tcp-port", type=int, default=9999, help="TCP server port (default: 9999)"
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8080,
        help="WebSocket server port (default: 8080)",
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=8000,
        help="MCP HTTP server port (default: 8000)",
    )
    parser.add_argument(
        "--ipc-socket",
        type=str,
        default=None,
        help="IPC socket path (default: /tmp/devduck_main.sock)",
    )

    # Server enable/disable flags
    parser.add_argument("--no-tcp", action="store_true", help="Disable TCP server")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")
    parser.add_argument("--no-mcp", action="store_true", help="Disable MCP server")
    parser.add_argument("--no-ipc", action="store_true", help="Disable IPC server")
    parser.add_argument(
        "--no-servers",
        action="store_true",
        help="Disable all servers (no TCP, WebSocket, MCP, or IPC)",
    )

    args = parser.parse_args()

    if args.mcp:
        print("🦆 Starting MCP stdio server...", file=sys.stderr)
        try:
            devduck.agent.tool.mcp_server(
                action="start",
                transport="stdio",
                expose_agent=True,
                agent=devduck.agent,
            )
        except Exception as e:
            print(f"🦆 Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.query:
        result = ask(" ".join(args.query))
        print(result)
    else:
        interactive()


# Make module callable
class CallableModule(sys.modules[__name__].__class__):
    def __call__(self, query):
        return ask(query)


sys.modules[__name__].__class__ = CallableModule


if __name__ == "__main__":
    cli()
