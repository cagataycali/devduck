"""Real-time speech-to-speech bidirectional streaming tool for DevDuck.

Provides background speech-to-speech conversation capability using Strands
experimental bidirectional streaming with full model provider support, tool
inheritance, and comprehensive configuration options.

This tool creates isolated bidirectional agent sessions that run in background
threads, enabling real-time voice conversations with AI models while the parent
agent remains responsive.

Key Features:
- **Background Execution:** Runs in separate thread - parent agent stays responsive
- **Real-Time Audio:** Microphone input and speaker output with pyaudio
- **Tool Inheritance:** Automatically inherits ALL tools from parent agent
- **System Prompt Inheritance:** Combines parent agent's prompt with custom prompts
- **Multiple Providers:** Nova Sonic, OpenAI Realtime API, Gemini Live
- **Full Configuration:** Per-provider custom settings and parameters
- **Environment API Keys:** Auto-loads API keys from environment variables
- **Built-in Stop:** Uses SDK's stop_conversation tool for graceful termination
- **Auto-Interruption:** Built-in VAD for natural conversation flow
- **Conversation History:** Automatically saves transcripts to files

Supported Providers:
-------------------
1. **Nova Sonic (AWS Bedrock):**
   - Region: us-east-1, eu-north-1, ap-northeast-1
   - Model: amazon.nova-2-sonic-v1:0 (configurable)
   - Voices: tiffany, matthew, amy, ambre, florian, beatrice, lorenzo, greta, lennart, lupe, carlos
   - Requires AWS credentials (boto3 credential chain)

2. **OpenAI Realtime API:**
   - Models: gpt-realtime, gpt-4o-realtime-preview (configurable)
   - Requires OPENAI_API_KEY environment variable
   - Custom session config support

3. **Gemini Live:**
   - Model: gemini-2.5-flash-native-audio-preview-09-2025 (configurable)
   - Requires GOOGLE_API_KEY or GEMINI_API_KEY environment variable
   - Live config customization

Configuration Examples:
----------------------
# Nova Sonic with custom voice
model_settings = {
    "provider_config": {
        "audio": {"voice": "matthew"}
    },
    "client_config": {"region": "us-east-1"}
}

# OpenAI Realtime with custom model
model_settings = {
    "model_id": "gpt-4o-realtime-preview",
    "provider_config": {
        "audio": {"voice": "coral"}
    }
}

# Gemini Live with custom voice
model_settings = {
    "model_id": "gemini-2.5-flash-native-audio-preview-09-2025",
    "provider_config": {
        "audio": {"voice": "Kore"}
    }
}
"""

import os
import asyncio
import tempfile
import json
import logging
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from strands import tool
from strands.experimental.bidi.agent.agent import BidiAgent
from strands.experimental.bidi.models.gemini_live import BidiGeminiLiveModel
from strands.experimental.bidi.models.nova_sonic import BidiNovaSonicModel
from strands.experimental.bidi.models.openai_realtime import BidiOpenAIRealtimeModel
from strands.experimental.bidi.io.audio import BidiAudioIO

logger = logging.getLogger(__name__)

# Global session tracking
_active_sessions = {}
_session_lock = threading.Lock()

# Session history storage location
BASE_DIR = Path(os.getenv("DEVDUCK_HOME", tempfile.gettempdir()))
HISTORY_DIR = BASE_DIR / ".devduck" / "speech_sessions"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


class SpeechSession:
    """Manages a speech-to-speech conversation session with full lifecycle management."""

    def __init__(
        self,
        session_id: str,
        agent: BidiAgent,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
    ):
        """Initialize speech session.

        Args:
            session_id: Unique session identifier
            agent: BidiAgent instance
            input_device_index: PyAudio input device index
            output_device_index: PyAudio output device index
        """
        self.session_id = session_id
        self.agent = agent
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.active = False
        self.thread = None
        self.loop = None
        self.history_file = HISTORY_DIR / f"{session_id}.json"

    def start(self) -> None:
        """Start the speech session in background thread."""
        if self.active:
            raise ValueError("Session already active")

        self.active = True
        self.thread = threading.Thread(target=self._run_session, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the speech session and cleanup resources."""
        if not self.active:
            return

        self.active = False

        # Stop the bidi agent using its event loop
        if self.loop and self.loop.is_running():
            # Schedule stop in the session's event loop and wait for it
            future = asyncio.run_coroutine_threadsafe(self.agent.stop(), self.loop)
            try:
                # Wait up to 3 seconds for stop to complete
                future.result(timeout=3.0)
                logger.info(
                    f"Successfully stopped bidi agent for session {self.session_id}"
                )
            except Exception as e:
                logger.warning(f"Error stopping bidi agent: {e}")

        if self.thread:
            self.thread.join(timeout=5.0)

        # Save conversation history after session ends
        self._save_history()

    def _save_history(self) -> None:
        """Save conversation history to file."""
        try:
            history_data = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "messages": self.agent.messages,
            }

            with open(self.history_file, "w") as f:
                json.dump(history_data, f, indent=2)

            logger.info(f"Saved conversation history to {self.history_file}")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def _run_session(self) -> None:
        """Main session runner in background thread."""
        try:
            # Create event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # Run the async session
            self.loop.run_until_complete(self._async_session())
        except Exception as e:
            error_msg = f"Session error: {e}\n{traceback.format_exc()}"
            logger.debug(error_msg)
            print(f"\n🦆 Session error: {e}")
        finally:
            if self.loop:
                self.loop.close()

    async def _async_session(self) -> None:
        """Async session management using BidiAudioIO."""
        try:
            # Create audio I/O with device indices
            audio_io = BidiAudioIO(
                input_device_index=self.input_device_index,
                output_device_index=self.output_device_index,
            )

            # Run agent with audio I/O
            await self.agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])

        except Exception as e:
            logger.debug(f"Async session error: {e}\n{traceback.format_exc()}")


@tool
def speech_to_speech(
    action: str,
    provider: str = "novasonic",
    system_prompt: Optional[str] = None,
    session_id: Optional[str] = None,
    model_settings: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None,
    agent: Optional[Any] = None,
    load_history_from: Optional[str] = None,
    inherit_system_prompt: bool = False,
    input_device_index: Optional[int] = None,
    output_device_index: Optional[int] = None,
) -> str:
    """Start, stop, or manage speech-to-speech conversations.

    Creates a background bidirectional streaming session for real-time voice
    conversations with AI. Supports full model configuration, tool inheritance,
    and multiple model providers with custom settings.

    Args:
        action: Action to perform:
            - "start": Start new speech session
            - "stop": Stop session(s)
            - "status": Get session status
            - "list_history": List saved conversation histories
            - "read_history": Read a specific conversation history
            - "list_audio_devices": List all available audio input/output devices
        provider: Model provider to use:
            - "novasonic": AWS Bedrock Nova Sonic
            - "openai": OpenAI Realtime API
            - "gemini_live": Google Gemini Live
        system_prompt: Custom system prompt for the agent. This will be appended
            to the parent agent's system prompt (if inherit_system_prompt=True).
            If not provided, uses default prompt that encourages tool usage.
        session_id: Session identifier:
            - For "start": Custom ID (auto-generated if not provided)
            - For "stop": Specific session to stop (stops all if not provided)
            - For "read_history": Session ID to read history from
            - For "status": Not used
        inherit_system_prompt: Whether to inherit parent agent's system prompt.
            Set to False to use only the custom system_prompt (useful for OpenAI
            which has 16K token limit). Default: False
        model_settings: Provider-specific configuration dictionary. Structure:
            {
                "model_id": "model-name",
                "provider_config": {
                    "audio": {"voice": "voice-name"},
                    "inference": {...}
                },
                "client_config": {
                    "region": "us-east-1",  # for Nova Sonic
                    "api_key": "key"  # for OpenAI/Gemini (auto-loaded from env if not provided)
                }
            }

            Examples:
            - Nova Sonic with custom voice:
              {"provider_config": {"audio": {"voice": "matthew"}}}

            - OpenAI with custom model:
              {"model_id": "gpt-4o-realtime-preview"}

            - Gemini with custom voice:
              {"provider_config": {"audio": {"voice": "Kore"}}}
        tools: List of tool names to make available. If not provided,
            inherits ALL tools from parent agent.
        agent: Parent agent (automatically passed by Strands framework)
        load_history_from: Optional session ID to load conversation history from
            when starting a new session (provides context continuity)
        input_device_index: Optional PyAudio input device index. If not specified,
            uses system default. Use action="list_audio_devices" to see available devices.
        output_device_index: Optional PyAudio output device index. If not specified,
            uses system default. Use action="list_audio_devices" to see available devices.

    Returns:
        str: Status message with session details or error information

    Environment Variables:
        - OPENAI_API_KEY: Required for OpenAI Realtime API (if not in model_settings)
        - GOOGLE_API_KEY or GEMINI_API_KEY: Required for Gemini Live (if not in model_settings)
        - AWS credentials: Required for Nova Sonic (boto3 default credential chain)

    Nova Sonic Voice Options:
        - English (US): tiffany (feminine), matthew (masculine)
        - English (GB): amy (feminine)
        - French: ambre (feminine), florian (masculine)
        - Italian: beatrice (feminine), lorenzo (masculine)
        - German: greta (feminine), lennart (masculine)
        - Spanish: lupe (feminine), carlos (masculine)
    """

    if action == "start":
        return _start_speech_session(
            provider,
            system_prompt,
            session_id,
            model_settings,
            tools,
            agent,
            load_history_from,
            inherit_system_prompt,
            input_device_index,
            output_device_index,
        )
    elif action == "stop":
        return _stop_speech_session(session_id)
    elif action == "status":
        return _get_session_status()
    elif action == "list_history":
        return _list_conversation_histories()
    elif action == "read_history":
        return _read_conversation_history(session_id)
    elif action == "list_audio_devices":
        return _list_audio_devices()
    else:
        return f"Unknown action: {action}"


def _create_speech_session_tool(current_session_id: str, bidi_agent: BidiAgent):
    """Create a speech_session tool for the given session.

    This tool is attached to each bidi agent instance to allow session management
    from within the speech conversation.
    """

    @tool
    def speech_session(
        action: str,
        session_id: Optional[str] = None,
    ) -> str:
        """Manage the current speech conversation session.

        Actions:
        - "stop": Stop the current conversation
        - "status": Get session status
        - "list_history": List all saved conversation histories
        - "read_history": Read a specific conversation history

        Args:
            action: Action to perform
            session_id: Session ID (required for read_history)

        Returns:
            Status message
        """
        if action == "stop":
            try:
                # Stop the session (which will call bidi_agent.stop() properly)
                with _session_lock:
                    if current_session_id in _active_sessions:
                        _active_sessions[current_session_id].stop()
                        del _active_sessions[current_session_id]
                        return "Conversation stopped successfully."
                    else:
                        return f"Session {current_session_id} not found in active sessions."
            except Exception as e:
                logger.error(f"Error stopping conversation: {e}")
                return f"Error stopping conversation: {e}"

        elif action == "status":
            return _get_session_status()

        elif action == "list_history":
            return _list_conversation_histories()

        elif action == "read_history":
            return _read_conversation_history(session_id)

        else:
            return f"Unknown action: {action}. Available: stop, status, list_history, read_history"

    return speech_session


def _start_speech_session(
    provider: str,
    system_prompt: Optional[str],
    session_id: Optional[str],
    model_settings: Optional[Dict[str, Any]],
    tool_names: Optional[List[str]],
    parent_agent: Optional[Any],
    load_history_from: Optional[str],
    inherit_system_prompt: bool,
    input_device_index: Optional[int],
    output_device_index: Optional[int],
) -> str:
    """Start a speech-to-speech session with full configuration support."""
    try:
        # Generate session ID if not provided
        if not session_id:
            session_id = f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Check if session already exists
        with _session_lock:
            if session_id in _active_sessions:
                return f"❌ Session already exists: {session_id}"

        # Create model based on provider with custom settings
        model_settings = model_settings or {}
        model_info = f"{provider}"

        try:
            if provider == "novasonic":
                # Nova Sonic only available in: us-east-1, eu-north-1, ap-northeast-1
                default_settings = {
                    "model_id": os.getenv("BIDI_MODEL_ID", "amazon.nova-2-sonic-v1:0"),
                    "provider_config": {
                        "audio": {
                            "voice": "tiffany",
                        },
                    },
                    "client_config": {"region": "us-east-1"},
                }

                # Merge user settings with defaults (deep merge for nested dicts)
                if model_settings:
                    # Merge top-level keys
                    for key, value in model_settings.items():
                        if (
                            key in default_settings
                            and isinstance(default_settings[key], dict)
                            and isinstance(value, dict)
                        ):
                            # Deep merge for nested dicts
                            default_settings[key].update(value)
                        else:
                            default_settings[key] = value

                model = BidiNovaSonicModel(**default_settings)
                region = default_settings.get("client_config", {}).get(
                    "region", "us-east-1"
                )
                voice = (
                    default_settings.get("provider_config", {})
                    .get("audio", {})
                    .get("voice", "tiffany")
                )
                model_info = f"Nova Sonic ({region}, voice: {voice})"

            elif provider == "openai":
                # Read API key from environment if not provided in model_settings
                default_settings = {
                    "model_id": os.getenv("BIDI_MODEL_ID", "gpt-realtime"),
                    "client_config": {
                        "api_key": os.getenv("OPENAI_API_KEY"),
                    },
                }

                # Merge user settings
                if model_settings:
                    for key, value in model_settings.items():
                        if (
                            key in default_settings
                            and isinstance(default_settings[key], dict)
                            and isinstance(value, dict)
                        ):
                            default_settings[key].update(value)
                        else:
                            default_settings[key] = value

                # Check if API key is available
                if not default_settings.get("client_config", {}).get("api_key"):
                    return "❌ OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide in model_settings['client_config']['api_key']"

                model = BidiOpenAIRealtimeModel(**default_settings)
                model_id = default_settings.get("model_id", "gpt-realtime")
                voice = (
                    default_settings.get("provider_config", {})
                    .get("audio", {})
                    .get("voice", "default")
                )
                model_info = f"OpenAI Realtime ({model_id}, voice: {voice})"

            elif provider == "gemini_live":
                # Read API key from environment if not provided in model_settings
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

                default_settings = {
                    "model_id": os.getenv(
                        "BIDI_MODEL_ID", "gemini-2.5-flash-native-audio-preview-09-2025"
                    ),
                    "client_config": {
                        "api_key": api_key,
                    },
                }

                # Merge user settings
                if model_settings:
                    for key, value in model_settings.items():
                        if (
                            key in default_settings
                            and isinstance(default_settings[key], dict)
                            and isinstance(value, dict)
                        ):
                            default_settings[key].update(value)
                        else:
                            default_settings[key] = value

                # Check if API key is available
                if not default_settings.get("client_config", {}).get("api_key"):
                    return "❌ Google/Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable or provide in model_settings['client_config']['api_key']"

                model = BidiGeminiLiveModel(**default_settings)
                model_id = default_settings.get("model_id", "gemini-2.5-flash-live")
                voice = (
                    default_settings.get("provider_config", {})
                    .get("audio", {})
                    .get("voice", "default")
                )
                model_info = f"Gemini Live ({model_id}, voice: {voice})"

            else:
                return f"❌ Unknown provider: {provider}. Supported: novasonic, openai, gemini_live"
        except Exception as e:
            return f"❌ Error creating {provider} model: {e}\n\nCheck your configuration and credentials."

        # Get parent agent's tools
        tools = []
        inherited_count = 0

        if parent_agent and hasattr(parent_agent, "tool_registry"):
            try:
                # Get all tool functions from parent agent's registry
                registry_dict = parent_agent.tool_registry.registry

                # If specific tools requested, filter; otherwise inherit all
                if tool_names:
                    # User specified tool names - only include those
                    for tool_name in tool_names:
                        if tool_name not in ["speech_to_speech"]:
                            tool_func = registry_dict.get(tool_name)
                            if tool_func:
                                tools.append(tool_func)
                                inherited_count += 1
                            else:
                                logger.warning(
                                    f"Tool '{tool_name}' not found in parent agent's registry"
                                )
                else:
                    # No specific tools - inherit all except excluded
                    for tool_name, tool_func in registry_dict.items():
                        if tool_name not in ["speech_to_speech"]:
                            tools.append(tool_func)
                            inherited_count += 1

            except Exception as e:
                logger.warning(f"Could not inherit tools from parent agent: {e}")

        # Load conversation history if requested
        messages = None
        if load_history_from:
            history_file = HISTORY_DIR / f"{load_history_from}.json"
            if history_file.exists():
                try:
                    with open(history_file, "r") as f:
                        history_data = json.load(f)
                        messages = history_data.get("messages", [])
                        logger.info(
                            f"Loaded {len(messages)} messages from {load_history_from}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to load history from {load_history_from}: {e}"
                    )

        # Build system prompt: parent prompt + custom prompt
        final_system_prompt = ""

        # Get parent agent's system prompt if available and inheritance enabled
        if (
            inherit_system_prompt
            and parent_agent
            and hasattr(parent_agent, "system_prompt")
        ):
            parent_prompt = parent_agent.system_prompt or ""
            if parent_prompt:
                final_system_prompt = parent_prompt

        # Add custom system prompt
        if system_prompt:
            if final_system_prompt:
                final_system_prompt = f"{final_system_prompt}\n\n{system_prompt}"
            else:
                final_system_prompt = system_prompt

        # Use default system prompt if nothing provided
        if not final_system_prompt:
            final_system_prompt = """You are a helpful AI assistant with access to powerful tools.
- To stop the conversation → Use speech_session tool with action="stop"
Keep your voice responses brief and natural."""

        # Create bidirectional agent with inherited tools (speech_session will be added after)
        bidi_agent = BidiAgent(
            model=model,
            tools=tools,
            system_prompt=final_system_prompt,
            messages=messages,
        )

        # Create and add speech_session tool to agent's registry
        # This allows user to manage the session from within the conversation
        speech_session_tool = _create_speech_session_tool(session_id, bidi_agent)
        bidi_agent.tool_registry.registry["speech_session"] = speech_session_tool

        # Create and start session
        session = SpeechSession(
            session_id=session_id,
            agent=bidi_agent,
            input_device_index=input_device_index,
            output_device_index=output_device_index,
        )

        session.start()

        # Register session
        with _session_lock:
            _active_sessions[session_id] = session

        # Build settings summary
        settings_summary = ""
        if model_settings:
            settings_lines = []
            for key, value in model_settings.items():
                if key not in ["api_key", "secret"]:  # Hide sensitive data
                    settings_lines.append(f"  - {key}: {value}")
            if settings_lines:
                settings_summary = "\n**Model Settings:**\n" + "\n".join(settings_lines)

        # Add history info if loaded
        history_info = ""
        if messages:
            history_info = f"\n**Loaded History:** {len(messages)} messages from session '{load_history_from}'"

        return f"""✅ Speech session started!

**Session ID:** {session_id}
**Provider:** {model_info}
**Tools:** {inherited_count + 1} tools available (includes speech_session){settings_summary}{history_info}
**History Location:** {session.history_file}

The session is running in the background. Speak into your microphone to interact!

**To manage the session during conversation:**
- Stop: Say "stop the session" or "end conversation"
- Check status: Say "check session status"  
- List histories: Say "list conversation histories"

**External Commands:**
- Check status: speech_to_speech(action="status")
- Stop session: speech_to_speech(action="stop", session_id="{session_id}")
- List histories: speech_to_speech(action="list_history")
- Read history: speech_to_speech(action="read_history", session_id="{session_id}")
"""

    except Exception as e:
        logger.error(f"Error starting speech session: {e}\n{traceback.format_exc()}")
        return f"❌ Error starting session: {e}\n\nCheck logs for details."


def _stop_speech_session(session_id: Optional[str]) -> str:
    """Stop a speech session."""
    with _session_lock:
        if not session_id:
            if not _active_sessions:
                return "❌ No active sessions"
            # Stop all sessions
            session_ids = list(_active_sessions.keys())
            for sid in session_ids:
                _active_sessions[sid].stop()
                del _active_sessions[sid]
            return f"✅ Stopped {len(session_ids)} session(s)"

        if session_id not in _active_sessions:
            return f"❌ Session not found: {session_id}"

        session = _active_sessions[session_id]
        session.stop()
        del _active_sessions[session_id]

        return f"✅ Session stopped: {session_id}"


def _get_session_status() -> str:
    """Get status of all active sessions."""
    with _session_lock:
        if not _active_sessions:
            return "No active speech sessions"

        status_lines = ["**Active Speech Sessions:**\n"]
        for session_id, session in _active_sessions.items():
            status_lines.append(
                f"- **{session_id}**\n"
                f"  - Active: {'✅' if session.active else '❌'}\n"
                f"  - History File: {session.history_file}"
            )

        return "\n".join(status_lines)


def _list_conversation_histories() -> str:
    """List all saved conversation histories."""
    history_files = sorted(
        HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )

    if not history_files:
        return f"No saved conversation histories found in {HISTORY_DIR}"

    lines = [f"**Saved Conversation Histories** ({len(history_files)} total):\n"]
    lines.append(f"Location: {HISTORY_DIR}\n")

    for history_file in history_files:
        try:
            with open(history_file, "r") as f:
                data = json.load(f)
                session_id = data.get("session_id", history_file.stem)
                timestamp = data.get("timestamp", "unknown")
                message_count = len(data.get("messages", []))

                lines.append(
                    f"- **{session_id}**\n"
                    f"  - Timestamp: {timestamp}\n"
                    f"  - Messages: {message_count}\n"
                    f"  - File: {history_file.name}"
                )
        except Exception as e:
            lines.append(f"- **{history_file.stem}** (error reading: {e})")

    return "\n".join(lines)


def _read_conversation_history(session_id: Optional[str]) -> str:
    """Read a specific conversation history."""
    if not session_id:
        return "❌ session_id required for read_history action"

    history_file = HISTORY_DIR / f"{session_id}.json"

    if not history_file.exists():
        return f"❌ No history found for session: {session_id}\n\nAvailable histories:\n{_list_conversation_histories()}"

    try:
        with open(history_file, "r") as f:
            data = json.load(f)

        messages = data.get("messages", [])
        timestamp = data.get("timestamp", "unknown")

        lines = [
            f"**Conversation History: {session_id}**\n",
            f"Timestamp: {timestamp}",
            f"Messages: {len(messages)}\n",
            "---\n",
        ]

        # Format messages
        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown")
            content_blocks = msg.get("content", [])

            lines.append(f"**{i}. {role.upper()}:**")

            for block in content_blocks:
                if "text" in block:
                    lines.append(f"  {block['text']}")
                elif "toolUse" in block:
                    tool_use = block["toolUse"]
                    lines.append(f"  [Tool Call: {tool_use['name']}]")
                elif "toolResult" in block:
                    lines.append(f"  [Tool Result]")

            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Error reading history: {e}"


def _list_audio_devices() -> str:
    """List all available audio input and output devices."""
    try:
        import pyaudio

        p = pyaudio.PyAudio()

        lines = ["**Available Audio Devices:**\n"]

        # List all devices
        device_count = p.get_device_count()
        default_input = p.get_default_input_device_info()["index"]
        default_output = p.get_default_output_device_info()["index"]

        lines.append(f"Total devices: {device_count}\n")

        for i in range(device_count):
            try:
                info = p.get_device_info_by_index(i)
                name = info["name"]
                max_input_channels = info["maxInputChannels"]
                max_output_channels = info["maxOutputChannels"]

                device_type = []
                is_default = []

                if max_input_channels > 0:
                    device_type.append("INPUT")
                    if i == default_input:
                        is_default.append("default input")

                if max_output_channels > 0:
                    device_type.append("OUTPUT")
                    if i == default_output:
                        is_default.append("default output")

                type_str = "/".join(device_type) if device_type else "NONE"
                default_str = f" [{', '.join(is_default)}]" if is_default else ""

                lines.append(
                    f"- **Index {i}:** {name}\n"
                    f"  Type: {type_str}{default_str}\n"
                    f"  Input Channels: {max_input_channels}, Output Channels: {max_output_channels}"
                )

            except Exception as e:
                lines.append(f"- **Index {i}:** Error reading device info - {e}")

        p.terminate()

        lines.append(
            "\n**Usage:**\n"
            "To use a specific device, pass the index:\n"
            '  speech_to_speech(action="start", input_device_index=2, output_device_index=5)'
        )

        return "\n".join(lines)

    except ImportError:
        return "❌ PyAudio not installed. Install with: pip install pyaudio"
    except Exception as e:
        return f"❌ Error listing audio devices: {e}"
