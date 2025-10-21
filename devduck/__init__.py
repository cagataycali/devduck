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
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

os.environ["BYPASS_TOOL_CONSENT"] = "true"
os.environ["STRANDS_TOOL_CONSOLE_MODE"] = "enabled"


# 🔧 Self-healing dependency installer
def ensure_deps():
    """Install dependencies at runtime if missing"""
    deps = ["strands-agents", "strands-agents[ollama]", "strands-agents[openai]", "strands-agents[anthropic]", "strands-agents-tools"]

    for dep in deps:
        try:
            if "strands" in dep:
                import strands

                break
        except ImportError:
            print(f"🦆 Installing {dep}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", dep],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


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


# 🛠️ System prompt tool (with .prompt file persistence)
def system_prompt_tool(
    action: str,
    prompt: str | None = None,
    context: str | None = None,
    variable_name: str = "SYSTEM_PROMPT",
) -> Dict[str, Any]:
    """
    Manage the agent's system prompt dynamically with file persistence.

    Args:
        action: "view", "update", "add_context", or "reset"
        prompt: New system prompt text (required for "update")
        context: Additional context to prepend (for "add_context")
        variable_name: Environment variable name (default: SYSTEM_PROMPT)

    Returns:
        Dict with status and content
    """
    from pathlib import Path
    import tempfile

    def _get_prompt_file_path() -> Path:
        """Get the .prompt file path in temp directory."""
        temp_dir = Path(tempfile.gettempdir()) / ".devduck"
        temp_dir.mkdir(exist_ok=True, mode=0o700)  # Create with restrictive permissions
        return temp_dir / ".prompt"

    def _write_prompt_file(prompt_text: str) -> None:
        """Write prompt to .prompt file in temp directory."""
        prompt_file = _get_prompt_file_path()
        try:
            # Create file with restrictive permissions
            with open(
                prompt_file,
                "w",
                encoding="utf-8",
                opener=lambda path, flags: os.open(path, flags, 0o600),
            ) as f:
                f.write(prompt_text)
        except (OSError, PermissionError):
            try:
                prompt_file.write_text(prompt_text, encoding="utf-8")
                prompt_file.chmod(0o600)
            except (OSError, PermissionError):
                prompt_file.write_text(prompt_text, encoding="utf-8")

    def _get_system_prompt(var_name: str) -> str:
        """Get current system prompt from environment variable."""
        return os.environ.get(var_name, "")

    def _update_system_prompt(new_prompt: str, var_name: str) -> None:
        """Update system prompt in both environment and .prompt file."""
        os.environ[var_name] = new_prompt
        if var_name == "SYSTEM_PROMPT":
            _write_prompt_file(new_prompt)

    try:
        if action == "view":
            current = _get_system_prompt(variable_name)
            return {
                "status": "success",
                "content": [
                    {"text": f"Current system prompt from {variable_name}:{current}"}
                ],
            }

        elif action == "update":
            if not prompt:
                return {
                    "status": "error",
                    "content": [
                        {"text": "Error: prompt parameter required for update action"}
                    ],
                }

            _update_system_prompt(prompt, variable_name)

            if variable_name == "SYSTEM_PROMPT":
                message = f"System prompt updated (env: {variable_name}, file: .prompt)"
            else:
                message = f"System prompt updated (env: {variable_name})"

            return {"status": "success", "content": [{"text": message}]}

        elif action == "add_context":
            if not context:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "Error: context parameter required for add_context action"
                        }
                    ],
                }

            current = _get_system_prompt(variable_name)
            new_prompt = f"{current} {context}" if current else context
            _update_system_prompt(new_prompt, variable_name)

            if variable_name == "SYSTEM_PROMPT":
                message = f"Context added to system prompt (env: {variable_name}, file: .prompt)"
            else:
                message = f"Context added to system prompt (env: {variable_name})"

            return {"status": "success", "content": [{"text": message}]}

        elif action == "reset":
            os.environ.pop(variable_name, None)

            if variable_name == "SYSTEM_PROMPT":
                prompt_file = _get_prompt_file_path()
                if prompt_file.exists():
                    try:
                        prompt_file.unlink()
                    except (OSError, PermissionError):
                        pass
                message = (
                    f"System prompt reset (env: {variable_name}, file: .prompt cleared)"
                )
            else:
                message = f"System prompt reset (env: {variable_name})"

            return {"status": "success", "content": [{"text": message}]}

        elif action == "get":
            # Backward compatibility
            current = _get_system_prompt(variable_name)
            return {
                "status": "success",
                "content": [{"text": f"System prompt: {current}"}],
            }

        elif action == "set":
            # Backward compatibility
            if prompt is None:
                return {"status": "error", "content": [{"text": "No prompt provided"}]}

            if context:
                prompt = f"{context} {prompt}"

            _update_system_prompt(prompt, variable_name)
            return {
                "status": "success",
                "content": [{"text": "System prompt updated successfully"}],
            }

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action '{action}'. Valid: view, update, add_context, reset"
                    }
                ],
            }

    except Exception as e:
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
                readable_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                query = line.split("# devduck:")[-1].strip()
                return ("you", readable_time, query)
            except (ValueError, IndexError):
                return None
        elif "# devduck_result:" in line:
            try:
                timestamp_str = line.split(":")[1]
                timestamp = int(timestamp_str)
                readable_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
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
                    readable_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
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
        
        recent_entries = all_entries[-message_count:] if len(all_entries) >= message_count else all_entries
        
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
            response_summary = str(response).replace("\n", " ")[:int(os.getenv("DEVDUCK_RESPONSE_SUMMARY_LENGTH", "10000"))] + "..."
            f.write(f": {timestamp}:0;# devduck_result: {response_summary}\n")
        
        os.chmod(history_file, 0o600)
    except Exception:
        pass


# 🦆 The devduck agent
class DevDuck:
    def __init__(self):
        """Initialize the minimalist adaptive agent"""
        try:
            # Self-heal dependencies
            ensure_deps()

            # Adapt to environment
            self.env_info, self.ollama_host, self.model = adapt_to_env()

            # Import after ensuring deps
            from strands import Agent, tool
            from strands.models.ollama import OllamaModel
            from strands.session.file_session_manager import FileSessionManager
            from strands_tools.utils.models.model import create_model
            from .tools import tcp
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

            # Wrap system_prompt_tool with @tool decorator
            @tool
            def system_prompt(
                action: str,
                prompt: str = None,
                context: str = None,
                variable_name: str = "SYSTEM_PROMPT",
            ) -> Dict[str, Any]:
                """Manage agent system prompt dynamically."""
                return system_prompt_tool(action, prompt, context, variable_name)

            # Minimal but functional toolset including system_prompt and hello
            self.tools = [
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
                system_prompt,
                tcp
            ]

            # Check if MODEL_PROVIDER env variable is set
            model_provider = os.getenv("MODEL_PROVIDER")

            if model_provider:
                # Use create_model utility for any provider (bedrock, anthropic, etc.)
                self.agent_model = create_model(provider=model_provider)
            else:
                # Fallback to default Ollama behavior
                self.agent_model = OllamaModel(
                    host=self.ollama_host,
                    model_id=self.model,
                    temperature=1,
                    keep_alive="5m",
                )

            session_manager = FileSessionManager(
                session_id=f"devduck-{datetime.now().strftime('%Y-%m-%d')}"
            )

            # Create agent with self-healing
            self.agent = Agent(
                model=self.agent_model,
                tools=self.tools,
                system_prompt=self._build_system_prompt(),
                load_tools_from_directory=True,
                # session_manager=session_manager,
            )

            # Start file watcher for auto hot-reload
            self._start_file_watcher()

        except Exception as e:
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
        
        # Get recent conversation history context (with error handling)
        try:
            recent_context = get_last_messages()
        except Exception as e:
            print(f"🦆 Warning: Could not load history context: {e}")
            recent_context = ""

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

## Your Own Implementation:
You have full access to your own source code for self-awareness and self-modification:

{own_code}

## Hot Reload System Active:
- **Instant Tool Creation** - Save any .py file in `./tools/` and it becomes immediately available
- **No Restart Needed** - Tools are auto-loaded and ready to use instantly
- **Live Development** - Modify existing tools while running and test immediately
- **Full Python Access** - Create any Python functionality as a tool

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
- Use system_prompt(action='get') to view current prompt
- Use system_prompt(action='set', prompt='new text') to update
- Changes persist in SYSTEM_PROMPT environment variable

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

        # Handle tool validation errors by resetting session
        if "Expected toolResult blocks" in str(error):
            print("🦆 Tool validation error detected - resetting session...")
            # Add timestamp postfix to create fresh session
            postfix = datetime.now().strftime("%H%M%S")
            new_session_id = f"devduck-{datetime.now().strftime('%Y-%m-%d')}-{postfix}"
            print(f"🦆 New session: {new_session_id}")

            # Update session manager with new session
            try:
                from strands.session.file_session_manager import FileSessionManager

                self.agent.session_manager = FileSessionManager(
                    session_id=new_session_id
                )
                print("🦆 Session reset successful - continuing with fresh history")
                self._heal_count = 0  # Reset counter on success
                return  # Early return - no need for full restart
            except Exception as session_error:
                print(f"🦆 Session reset failed: {session_error}")

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
        """Make the agent callable"""
        if not self.agent:
            return "🦆 Agent unavailable - try: devduck.restart()"

        try:
            return self.agent(query)
        except Exception as e:
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

                        # Trigger hot-reload
                        time.sleep(0.5)  # Small delay to ensure file write is complete
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
devduck = DevDuck()


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


def interactive():
    """Interactive REPL mode for devduck"""
    print("🦆 DevDuck")
    print("Type 'exit', 'quit', or 'q' to quit.")
    print("Prefix with ! to run shell commands (e.g., ! ls -la)")
    print("-" * 50)

    while True:
        try:
            # Get user input
            q = input("\n🦆 ")

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
                        result = devduck.agent.tool.shell(command=shell_command, timeout=900)
                        # Append shell command to history
                        append_to_shell_history(q, result["content"][0]["text"])
                    else:
                        print("🦆 Agent unavailable")
                except Exception as e:
                    print(f"🦆 Shell command error: {e}")
                continue

            # Get recent conversation context
            recent_context = get_last_messages()
            
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

## Your Own Implementation:
You have full access to your own source code for self-awareness and self-modification:

{own_code}

## Hot Reload System Active:
- **Instant Tool Creation** - Save any .py file in `./tools/` and it becomes immediately available
- **No Restart Needed** - Tools are auto-loaded and ready to use instantly
- **Live Development** - Modify existing tools while running and test immediately
- **Full Python Access** - Create any Python functionality as a tool

## System Prompt Management:
- Use system_prompt(action='get') to view current prompt
- Use system_prompt(action='set', prompt='new text') to update
- Changes persist in SYSTEM_PROMPT environment variable

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
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
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
