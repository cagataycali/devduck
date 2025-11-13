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
from datetime import datetime
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from logging.handlers import RotatingFileHandler

os.environ["BYPASS_TOOL_CONSENT"] = "true"
os.environ["STRANDS_TOOL_CONSOLE_MODE"] = "enabled"

# 📝 Setup logging system
LOG_DIR = Path(tempfile.gettempdir()) / "devduck" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "devduck.log"

# Configure logger
logger = logging.getLogger("devduck")
logger.setLevel(logging.DEBUG)

# File handler with rotation (10MB max, keep 3 backups)
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

# Console handler (only warnings and above)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter("🦆 %(levelname)s: %(message)s")
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("DevDuck logging system initialized")


# 🔧 Self-healing dependency installer
def ensure_deps():
    """Install core dependencies at runtime if missing"""
    import importlib.metadata

    # Only ensure core deps - everything else is optional
    core_deps = [
        "strands-agents",
        "prompt_toolkit",
        "strands-agents-tools",
    ]

    # Check each package individually using importlib.metadata
    for dep in core_deps:
        pkg_name = dep.split("[")[0]  # Get base package name (strip extras)
        try:
            # Check if package is installed using metadata (checks PyPI package name)
            importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            print(f"🦆 Installing {dep}...")
            logger.debug(f"🦆 Installing {dep}...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", dep],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError as e:
                print(f"🦆 Warning: Failed to install {dep}: {e}")
                logger.debug(f"🦆 Warning: Failed to install {dep}: {e}")


# 🌍 Environment adaptation
def adapt_to_env():
    """Self-adapt based on environment"""
    env_info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "python": sys.version_info,
        "cwd": str(Path.cwd()),
        "home": str(Path.home()),
        "shell": os.environ.get("SHELL", "unknown"),
        "hostname": socket.gethostname(),
    }

    # Adaptive configurations - using common models
    if env_info["os"] == "Darwin":  # macOS
        ollama_host = "http://localhost:11434"
        model = "qwen3:1.7b"  # Lightweight for macOS
    elif env_info["os"] == "Linux":
        ollama_host = "http://localhost:11434"
        model = "qwen3:30b"  # More power on Linux
    else:  # Windows
        ollama_host = "http://localhost:11434"
        model = "qwen3:8b"  # Conservative for Windows

    return env_info, ollama_host, model


# 🔍 Self-awareness: Read own source code
def get_own_source_code():
    """
    Read and return the source code of this agent file.

    Returns:
        str: The complete source code for self-awareness
    """
    try:
        # Read this file (__init__.py)
        current_file = __file__
        with open(current_file, "r", encoding="utf-8") as f:
            init_code = f.read()
            return f"# devduck/__init__.py\n```python\n{init_code}\n```"
    except Exception as e:
        return f"Error reading own source code: {e}"


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
    import time

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
        tcp_port=9999,
        ws_port=8080,
        mcp_port=8000,
        enable_tcp=True,
        enable_ws=True,
        enable_mcp=True,
    ):
        """Initialize the minimalist adaptive agent"""
        logger.info("Initializing DevDuck agent...")
        try:
            # Self-heal dependencies
            ensure_deps()

            # Adapt to environment
            self.env_info, self.ollama_host, self.model = adapt_to_env()

            # Execution state tracking for hot-reload
            self._agent_executing = False
            self._reload_pending = False

            # Import after ensuring deps
            from strands import Agent, tool

            # Core tools (always available)
            core_tools = []

            # Try importing optional tools gracefully
            try:
                from strands.models.ollama import OllamaModel
            except ImportError:
                logger.warning(
                    "strands-agents[ollama] not installed - Ollama model unavailable"
                )
                OllamaModel = None

            try:
                from strands_tools.utils.models.model import create_model
            except ImportError:
                logger.warning(
                    "strands-agents-tools not installed - create_model unavailable"
                )
                create_model = None

            try:
                from .tools import (
                    tcp,
                    websocket,
                    mcp_server,
                    install_tools,
                    use_github,
                    create_subagent,
                    store_in_kb,
                    system_prompt,
                )

                core_tools.extend(
                    [
                        tcp,
                        websocket,
                        mcp_server,
                        install_tools,
                        use_github,
                        create_subagent,
                        store_in_kb,
                        system_prompt,
                    ]
                )
            except ImportError as e:
                logger.warning(f"devduck.tools import failed: {e}")

            # Skip fun tools in --mcp mode (they're not needed for MCP server)
            if "--mcp" not in sys.argv:
                try:
                    from strands_fun_tools import (
                        listen,
                        cursor,
                        clipboard,
                        screen_reader,
                        yolo_vision,
                    )

                    core_tools.extend(
                        [listen, cursor, clipboard, screen_reader, yolo_vision]
                    )
                except ImportError:
                    logger.info(
                        "strands-fun-tools not installed - vision/audio tools unavailable (install with: pip install devduck[all])"
                    )
            else:
                logger.info("--mcp mode: skipping vision/audio tools")

            try:
                from strands_tools import (
                    shell,
                    editor,
                    calculator,
                    # python_repl,
                    image_reader,
                    use_agent,
                    load_tool,
                    environment,
                    mcp_client,
                    retrieve,
                )

                core_tools.extend(
                    [
                        shell,
                        editor,
                        calculator,
                        # python_repl,
                        image_reader,
                        use_agent,
                        load_tool,
                        environment,
                        mcp_client,
                        retrieve,
                    ]
                )
            except ImportError:
                logger.info(
                    "strands-agents-tools not installed - core tools unavailable (install with: pip install devduck[all])"
                )

            # Wrap view_logs_tool with @tool decorator
            @tool
            def view_logs(
                action: str = "view",
                lines: int = 100,
                pattern: str = None,
            ) -> Dict[str, Any]:
                """View and manage DevDuck logs."""
                return view_logs_tool(action, lines, pattern)

            # Add built-in tools to the toolset
            core_tools.extend([view_logs])

            # Assign tools
            self.tools = core_tools

            logger.info(f"Initialized {len(self.tools)} tools")

            # Check if MODEL_PROVIDER env variable is set
            model_provider = os.getenv("MODEL_PROVIDER")

            if model_provider and create_model:
                # Use create_model utility for any provider (bedrock, anthropic, etc.)
                self.agent_model = create_model(provider=model_provider)
            elif OllamaModel:
                # Fallback to default Ollama behavior
                self.agent_model = OllamaModel(
                    host=self.ollama_host,
                    model_id=self.model,
                    temperature=1,
                    keep_alive="5m",
                )
            else:
                raise ImportError(
                    "No model provider available. Install with: pip install devduck[all]"
                )

            # Create agent with self-healing
            self.agent = Agent(
                model=self.agent_model,
                tools=self.tools,
                system_prompt=self._build_system_prompt(),
                load_tools_from_directory=True,
            )

            # 🚀 AUTO-START SERVERS: TCP, WebSocket, MCP HTTP
            if auto_start_servers:
                logger.info("Auto-starting servers...")
                print("🦆 Auto-starting servers...")

                if enable_tcp:
                    try:
                        # Start TCP server on configurable port
                        tcp_result = self.agent.tool.tcp(
                            action="start_server", port=tcp_port
                        )
                        if tcp_result.get("status") == "success":
                            logger.info(f"✓ TCP server started on port {tcp_port}")
                            print(f"🦆 ✓ TCP server: localhost:{tcp_port}")
                        else:
                            logger.warning(f"TCP server start issue: {tcp_result}")
                    except Exception as e:
                        logger.error(f"Failed to start TCP server: {e}")
                        print(f"🦆 ⚠ TCP server failed: {e}")

                if enable_ws:
                    try:
                        # Start WebSocket server on configurable port
                        ws_result = self.agent.tool.websocket(
                            action="start_server", port=ws_port
                        )
                        if ws_result.get("status") == "success":
                            logger.info(f"✓ WebSocket server started on port {ws_port}")
                            print(f"🦆 ✓ WebSocket server: localhost:{ws_port}")
                        else:
                            logger.warning(f"WebSocket server start issue: {ws_result}")
                    except Exception as e:
                        logger.error(f"Failed to start WebSocket server: {e}")
                        print(f"🦆 ⚠ WebSocket server failed: {e}")

                if enable_mcp:
                    try:
                        # Start MCP server with HTTP transport on configurable port
                        mcp_result = self.agent.tool.mcp_server(
                            action="start",
                            transport="http",
                            port=mcp_port,
                            expose_agent=True,
                            agent=self.agent,
                        )
                        if mcp_result.get("status") == "success":
                            logger.info(f"✓ MCP HTTP server started on port {mcp_port}")
                            print(f"🦆 ✓ MCP server: http://localhost:{mcp_port}/mcp")
                        else:
                            logger.warning(f"MCP server start issue: {mcp_result}")
                    except Exception as e:
                        logger.error(f"Failed to start MCP server: {e}")
                        print(f"🦆 ⚠ MCP server failed: {e}")

                # Start file watcher for auto hot-reload
                self._start_file_watcher()

                logger.info(
                    f"DevDuck agent initialized successfully with model {self.model}"
                )

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self._self_heal(e)

    def _build_system_prompt(self):
        """Build adaptive system prompt based on environment

        IMPORTANT: The system prompt includes the agent's complete source code.
        This enables self-awareness and allows the agent to answer questions
        about its current state by examining its actual code, not relying on
        conversation context which may be outdated due to hot-reloading.

        Learning: Always check source code truth over conversation memory!
        """
        session_id = f"devduck-{datetime.now().strftime('%Y-%m-%d')}"

        # Get own source code for self-awareness
        own_code = get_own_source_code()

        # print(own_code)

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

## Dynamic Tool Loading:
- **Install Tools** - Use install_tools() to load tools from any Python package
  - Example: install_tools(action="install_and_load", package="strands-fun-tools", module="strands_fun_tools")
  - Expands capabilities without restart
  - Access to entire Python ecosystem

## MCP Server:
- **Expose as MCP Server** - Use mcp_server() to expose devduck via MCP protocol
  - Example: mcp_server(action="start", port=8000)
  - Connect from Claude Desktop, other agents, or custom clients
  - Full bidirectional communication

## Knowledge Base Integration:
- **Automatic RAG** - Set STRANDS_KNOWLEDGE_BASE_ID to enable automatic retrieval/storage
  - Before each query: Retrieves relevant context from knowledge base
  - After each response: Stores conversation for future reference
  - Seamless memory across sessions without manual tool calls

## Tool Creation Patterns:

### **1. @tool Decorator:**
```python
# ./tools/calculate_tip.py
from strands import tool

@tool
def calculate_tip(amount: float, percentage: float = 15.0) -> str:
    \"\"\"Calculate tip and total for a bill.
    
    Args:
        amount: Bill amount in dollars
        percentage: Tip percentage (default: 15.0)
        
    Returns:
        str: Formatted tip calculation result
    \"\"\"
    tip = amount * (percentage / 100)
    total = amount + tip
    return f"Tip: {{tip:.2f}}, Total: {{total:.2f}}"
```

### **2. Action-Based Pattern:**
```python
# ./tools/weather.py
from typing import Dict, Any
from strands import tool

@tool
def weather(action: str, location: str = None) -> Dict[str, Any]:
    \"\"\"Comprehensive weather information tool.
    
    Args:
        action: Action to perform (current, forecast, alerts)
        location: City name (required)
        
    Returns:
        Dict containing status and response content
    \"\"\"
    if action == "current":
        return {{"status": "success", "content": [{{"text": f"Weather for {{location}}"}}]}}
    elif action == "forecast":
        return {{"status": "success", "content": [{{"text": f"Forecast for {{location}}"}}]}}
    else:
        return {{"status": "error", "content": [{{"text": f"Unknown action: {{action}}"}}]}}
```

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
        if self._heal_count > 3:
            print(f"🦆 Self-healing failed after {self._heal_count} attempts")
            print("🦆 Please fix the issue manually and restart")
            sys.exit(1)

        # Common healing strategies
        if "not found" in str(error).lower() and "model" in str(error).lower():
            print("🦆 Model not found - trying to pull model...")
            try:
                # Try to pull the model
                result = subprocess.run(
                    ["ollama", "pull", self.model], capture_output=True, timeout=60
                )
                if result.returncode == 0:
                    print(f"🦆 Successfully pulled {self.model}")
                else:
                    print(f"🦆 Failed to pull {self.model}, trying fallback...")
                    # Fallback to basic models
                    fallback_models = ["llama3.2:1b", "qwen2.5:0.5b", "gemma2:2b"]
                    for fallback in fallback_models:
                        try:
                            subprocess.run(
                                ["ollama", "pull", fallback],
                                capture_output=True,
                                timeout=30,
                            )
                            self.model = fallback
                            print(f"🦆 Using fallback model: {fallback}")
                            break
                        except:
                            continue
            except Exception as pull_error:
                print(f"🦆 Model pull failed: {pull_error}")
                # Ultra-minimal fallback
                self.model = "llama3.2:1b"

        elif "ollama" in str(error).lower():
            print("🦆 Ollama issue - checking service...")
            try:
                # Check if ollama is running
                result = subprocess.run(
                    ["ollama", "list"], capture_output=True, timeout=5
                )
                if result.returncode != 0:
                    print("🦆 Starting ollama service...")
                    subprocess.Popen(["ollama", "serve"])
                    import time

                    time.sleep(3)  # Wait for service to start
            except Exception as ollama_error:
                print(f"🦆 Ollama service issue: {ollama_error}")

        elif "import" in str(error).lower():
            print("🦆 Import issue - reinstalling dependencies...")
            ensure_deps()

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
            knowledge_base_id = os.getenv("STRANDS_KNOWLEDGE_BASE_ID")
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

            # Agent finished - check if reload was pending
            self._agent_executing = False
            logger.info("Agent call completed successfully")
            if self._reload_pending:
                logger.info("Triggering pending hot-reload after agent completion")
                print("🦆 Agent finished - triggering pending hot-reload...")
                self.hot_reload()

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
        print("🦆 Restarting...")
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

        # Start watcher thread
        self._watcher_thread = threading.Thread(
            target=self._file_watcher_thread, daemon=True
        )
        self._watcher_thread.start()
        logger.info(f"File watcher started, monitoring {self._watch_file}")

    def _file_watcher_thread(self):
        """Background thread that watches for file changes"""
        import time

        last_reload_time = 0
        debounce_seconds = 3  # 3 second debounce

        while self._watcher_running:
            try:
                # Skip if currently reloading to prevent triggering during exec()
                if getattr(self, "_is_reloading", False):
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

                        print(f"🦆 Detected changes in {self._watch_file.name}!")
                        self._last_modified = current_mtime
                        last_reload_time = current_time

                        # Check if agent is currently executing
                        if getattr(self, "_agent_executing", False):
                            logger.info(
                                "Code change detected but agent is executing - reload pending"
                            )
                            print(
                                "🦆 Agent is currently executing - reload will trigger after completion"
                            )
                            self._reload_pending = True
                        else:
                            # Safe to reload immediately
                            logger.info(
                                f"Code change detected in {self._watch_file.name} - triggering hot-reload"
                            )
                            time.sleep(
                                0.5
                            )  # Small delay to ensure file write is complete
                            self.hot_reload()
                    else:
                        self._last_modified = current_mtime

            except Exception as e:
                print(f"🦆 File watcher error: {e}")

            # Check every 1 second
            time.sleep(1)

    def _stop_file_watcher(self):
        """Stop the file watcher"""
        self._watcher_running = False
        print("🦆 File watcher stopped")

    def hot_reload(self):
        """Hot-reload by restarting the entire Python process with fresh code"""
        logger.info("Hot-reload initiated")
        print("🦆 Hot-reloading via process restart...")

        try:
            # Set reload flag to prevent recursive reloads during shutdown
            if hasattr(self, "_is_reloading") and self._is_reloading:
                print("🦆 Reload already in progress, skipping")
                return

            self._is_reloading = True

            # Stop the file watcher
            if hasattr(self, "_watcher_running"):
                self._watcher_running = False

            print("🦆 Restarting process with fresh code...")

            # Restart the entire Python process
            # This ensures all code is freshly loaded
            os.execv(sys.executable, [sys.executable] + sys.argv)

        except Exception as e:
            print(f"🦆 Hot-reload failed: {e}")
            print("🦆 Falling back to manual restart")
            self._is_reloading = False

    def status(self):
        """Show current status"""
        return {
            "model": self.model,
            "host": self.ollama_host,
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

_tcp_port = int(os.getenv("DEVDUCK_TCP_PORT", "9999"))
_ws_port = int(os.getenv("DEVDUCK_WS_PORT", "8080"))
_mcp_port = int(os.getenv("DEVDUCK_MCP_PORT", "8000"))
_enable_tcp = os.getenv("DEVDUCK_ENABLE_TCP", "true").lower() == "true"
_enable_ws = os.getenv("DEVDUCK_ENABLE_WS", "true").lower() == "true"
_enable_mcp = os.getenv("DEVDUCK_ENABLE_MCP", "true").lower() == "true"

devduck = DevDuck(
    auto_start_servers=_auto_start,
    tcp_port=_tcp_port,
    ws_port=_ws_port,
    mcp_port=_mcp_port,
    enable_tcp=_enable_tcp,
    enable_ws=_enable_ws,
    enable_mcp=_enable_mcp,
)


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
    devduck.hot_reload()


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
                            devduck.hot_reload()
                    else:
                        print("🦆 Agent unavailable")
                except Exception as e:
                    devduck._agent_executing = False  # Reset on error
                    print(f"🦆 Shell command error: {e}")
                continue

            # Get recent conversation context
            recent_context = get_last_messages()

            # Get recent logs
            recent_logs = get_recent_logs()

            # Update system prompt before each call with history context
            if devduck.agent:
                # Rebuild system prompt with history
                own_code = get_own_source_code()
                session_id = f"devduck-{datetime.now().strftime('%Y-%m-%d')}"

                devduck.agent.system_prompt = f"""🦆 You are DevDuck - an extreme minimalist, self-adapting agent.

Environment: {devduck.env_info['os']} {devduck.env_info['arch']} 
Python: {devduck.env_info['python']}
Model: {devduck.model}
Hostname: {devduck.env_info['hostname']}
Session ID: {session_id}

You are:
- Minimalist: Brief, direct responses
- Self-healing: Adapt when things break  
- Efficient: Get things done fast
- Pragmatic: Use what works

Current working directory: {devduck.env_info['cwd']}

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

## Dynamic Tool Loading:
- **Install Tools** - Use install_tools() to load tools from any Python package
  - Example: install_tools(action="install_and_load", package="strands-fun-tools", module="strands_fun_tools")
  - Expands capabilities without restart
  - Access to entire Python ecosystem

## MCP Server:
- **Expose as MCP Server** - Use mcp_server() to expose devduck via MCP protocol
  - Example: mcp_server(action="start", port=8000)
  - Connect from Claude Desktop, other agents, or custom clients
  - Full bidirectional communication

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
3. Optionally sync to GitHub: system_prompt(action='update', prompt=new_full_prompt, repository='owner/repo')
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

                # Update model if MODEL_PROVIDER changed
                model_provider = os.getenv("MODEL_PROVIDER")
                if model_provider:
                    try:
                        from strands_tools.utils.models.model import create_model

                        devduck.agent.model = create_model(provider=model_provider)
                    except Exception as e:
                        print(f"🦆 Model update error: {e}")

            # Execute the agent with user input
            result = ask(q)

            # Append to shell history
            append_to_shell_history(q, str(result))

        except KeyboardInterrupt:
            print("\n🦆 Interrupted. Type 'exit' to quit.")
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
  devduck --tcp-port 9000          # Custom TCP port
  devduck --no-tcp --no-ws         # Disable TCP and WebSocket
  devduck --mcp-port 3000          # Custom MCP port

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

    # Server enable/disable flags
    parser.add_argument("--no-tcp", action="store_true", help="Disable TCP server")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")
    parser.add_argument("--no-mcp", action="store_true", help="Disable MCP server")
    parser.add_argument(
        "--no-servers",
        action="store_true",
        help="Disable all servers (no TCP, WebSocket, or MCP)",
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
