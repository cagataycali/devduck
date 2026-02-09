"""WebSocket tool for DevDuck agents with real-time streaming and voice support.

This module provides WebSocket server functionality for DevDuck agents,
supporting both text chat and bidirectional voice communication.

Key Features:
1. Text Chat: Traditional text-based agent interactions with streaming
2. Voice Chat: Real-time bidirectional audio using BidiAgent
3. Unified Context: Shared conversation history between text and voice
4. Browser Config: API keys can be configured from browser (persisted server-side)
5. Multi-Provider: Support for Nova Sonic, Gemini Live, OpenAI Realtime
6. Mesh Integration: Zenoh peer updates + ring context pushed to browser

Message Types (Client -> Server):
- text: {"type": "text", "text": "..."}
- audio_start: {"type": "audio_start", "provider": "novasonic", "voice": "tiffany"}
- audio_chunk: {"type": "audio_chunk", "audio": "base64...", "sample_rate": 16000}
- audio_stop: {"type": "audio_stop"}
- config: {"type": "config", "provider": "...", "anthropicKey": "...", ...}
- clear_history: {"type": "clear_history"}

Message Types (Server -> Client):
- connected: {"type": "connected", "data": "...", "history": [...]}
- turn_start: {"type": "turn_start", "turn_id": "...", "data": "...", "source": "text|voice"}
- chunk: {"type": "chunk", "turn_id": "...", "data": "..."}
- tool_start/tool_end: Tool execution notifications
- turn_end: {"type": "turn_end", "turn_id": "..."}
- audio_started: {"type": "audio_started", "provider": "..."}
- audio_stopped: {"type": "audio_stopped", "reason": "..."}
- audio_chunk: {"type": "audio_chunk", "audio": "base64...", "sample_rate": 16000}
- transcript: {"type": "transcript", "text": "...", "role": "user|assistant", "is_final": true}
- zenoh_peers_update: Zenoh peer discovery pushed to browser
- config_updated: {"type": "config_updated", ...}
- error: {"type": "error", "data": "..."}
"""

import logging
import threading
import time
import os
import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from strands import Agent, tool

logger = logging.getLogger(__name__)

# Global registry to store server threads
WS_SERVER_THREADS: dict[int, dict[str, Any]] = {}

# Shared thread pool for agent execution
_executor = ThreadPoolExecutor(max_workers=4)

# Parent agent reference — set when websocket() is called with agent param
_PARENT_AGENT_REF: Dict[str, Any] = {}


def _get_parent_agent():
    """Get parent agent from stored ref or fallback to devduck import."""
    if "agent" in _PARENT_AGENT_REF and _PARENT_AGENT_REF["agent"] is not None:
        return _PARENT_AGENT_REF["agent"]
    try:
        from devduck import devduck as dd

        if dd.agent:
            return dd.agent
    except Exception:
        pass
    return None


def _get_parent_tools(
    exclude=("websocket", "speech_to_speech", "mcp_server", "ipc", "tcp")
):
    """Get tools from parent agent, excluding server tools."""
    parent = _get_parent_agent()
    if not parent or not hasattr(parent, "tools"):
        return []
    tools = []
    for t in parent.tools:
        name = getattr(t, "__name__", str(t))
        if name not in exclude:
            tools.append(t)
    return tools


def _get_parent_system_prompt():
    """Get system prompt from parent agent."""
    parent = _get_parent_agent()
    if parent and hasattr(parent, "system_prompt") and parent.system_prompt:
        return parent.system_prompt
    return ""


def _get_ring_context_str(max_entries: int = 10) -> str:
    """Get unified ring context as a string for injection."""
    try:
        from devduck.tools.unified_mesh import get_ring_context

        entries = get_ring_context(max_entries=max_entries)
        if not entries:
            return ""
        lines = ["\n## Ring Context (recent agent activity):"]
        for entry in entries[-max_entries:]:
            agent_id = entry.get("agent_id", "?")
            text = entry.get("text", "")[:150]
            lines.append(f"- {agent_id}: {text}")
        return "\n".join(lines)
    except Exception:
        return ""


# ============================================================================
# Shared Conversation Store — Uses parent devduck agent's messages
# ============================================================================


def get_shared_messages() -> List[Dict]:
    """Get messages from the parent devduck agent."""
    try:
        from devduck import devduck as dd

        if dd.agent and hasattr(dd.agent, "messages"):
            return dd.agent.messages
    except Exception as e:
        logger.debug(f"Could not get shared messages: {e}")
    return []


def add_shared_message(role: str, content: str, source: str = "text") -> None:
    """Add a message to the parent devduck agent's history."""
    try:
        from devduck import devduck as dd

        if dd.agent and hasattr(dd.agent, "messages"):
            message = {"role": role, "content": [{"text": f"[{source}] {content}"}]}
            dd.agent.messages.append(message)
            logger.debug(f"Added message to shared history: {role} [{source}]")
    except Exception as e:
        logger.debug(f"Could not add shared message: {e}")


def get_messages_for_client(limit: int = 50) -> List[Dict]:
    """Get messages formatted for WebSocket client (simplified format)."""
    messages = get_shared_messages()
    result = []
    for msg in messages[-limit:]:
        content_text = ""
        source = "text"
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and "text" in block:
                    text = block["text"]
                    if text.startswith("[voice]"):
                        source = "voice"
                        text = text[7:].strip()
                    elif text.startswith("[text]"):
                        source = "text"
                        text = text[6:].strip()
                    content_text += text
        elif isinstance(msg.get("content"), str):
            content_text = msg["content"]
        if content_text:
            result.append(
                {
                    "role": msg.get("role", "user"),
                    "content": content_text,
                    "source": source,
                    "timestamp": datetime.now().isoformat(),
                }
            )
    return result


def get_context_prompt_from_shared(max_recent: int = 20) -> str:
    """Build context prompt from shared message history."""
    messages = get_shared_messages()
    if not messages:
        return ""
    recent = messages[-max_recent:]
    lines = ["\n## Recent Conversation Context:"]
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content_text = ""
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and "text" in block:
                    content_text += block["text"][:500]
        elif isinstance(msg.get("content"), str):
            content_text = msg["content"][:500]
        if content_text:
            lines.append(f"{role}: {content_text}")
    return "\n".join(lines)


# ============================================================================
# Client Session
# ============================================================================


@dataclass
class ClientSession:
    """Tracks state for a connected client."""

    websocket: Any
    client_id: str
    text_agent: Optional[Agent] = None
    bidi_agent: Any = None
    bidi_task: Optional[asyncio.Task] = None
    audio_active: bool = False
    config: Dict = field(default_factory=dict)


# ============================================================================
# Streaming Callback Handler
# ============================================================================


class WebSocketStreamingCallbackHandler:
    """Callback handler that streams agent responses over WebSocket."""

    def __init__(self, websocket, loop, turn_id: str, source: str = "text"):
        self.websocket = websocket
        self.loop = loop
        self.turn_id = turn_id
        self.source = source
        self.tool_count = 0
        self.previous_tool_use = None
        self.accumulated_text = ""
        self._closed = False

    async def _send(self, msg_type: str, data: str = "", **kwargs) -> None:
        if self._closed:
            return
        try:
            message = {
                "type": msg_type,
                "turn_id": self.turn_id,
                "data": data,
                "timestamp": time.time(),
                **kwargs,
            }
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to send message: {e}")
            self._closed = True

    def _schedule(self, msg_type: str, data: str = "", **kwargs) -> None:
        if self._closed:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._send(msg_type, data, **kwargs), self.loop
            )
        except RuntimeError:
            self._closed = True

    def __call__(self, **kwargs: Any) -> None:
        if self._closed:
            return
        data = kwargs.get("data", "")
        reasoning_text = kwargs.get("reasoningText", "")
        current_tool_use = kwargs.get("current_tool_use", {})
        message = kwargs.get("message", {})

        if reasoning_text:
            self._schedule("chunk", reasoning_text, reasoning=True)
        if data:
            self.accumulated_text += data
            self._schedule("chunk", data)
        if isinstance(current_tool_use, dict) and current_tool_use.get("name"):
            if self.previous_tool_use != current_tool_use:
                self.previous_tool_use = current_tool_use
                self.tool_count += 1
                self._schedule(
                    "tool_start", current_tool_use["name"], tool_number=self.tool_count
                )
        if isinstance(message, dict) and message.get("role") == "user":
            for content in message.get("content", []):
                if isinstance(content, dict) and content.get("toolResult"):
                    status = content["toolResult"].get("status", "unknown")
                    self._schedule("tool_end", status, success=status == "success")


# ============================================================================
# Agent Factory
# ============================================================================


def create_text_agent(session: ClientSession, base_system_prompt: str) -> Agent:
    """Create a text agent with parent tools, ring context, and appropriate model."""
    config = session.config
    provider = config.get("provider", os.getenv("MODEL_PROVIDER", ""))

    if not provider:
        try:
            if os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
                provider = "bedrock"
            else:
                import boto3

                boto3.client("sts").get_caller_identity()
                provider = "bedrock"
        except Exception:
            if config.get("anthropicKey") or os.getenv("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif config.get("openaiKey") or os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            else:
                provider = "ollama"

    model_id = config.get("modelId") or os.getenv("STRANDS_MODEL_ID", "")

    if provider == "ollama":
        from strands.models.ollama import OllamaModel

        model = OllamaModel(
            host=config.get("ollamaHost")
            or os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            model_id=model_id or "qwen3:1.7b",
            temperature=1,
            keep_alive="5m",
        )
    elif provider == "anthropic":
        from strands.models.anthropic import AnthropicModel

        api_key = config.get("anthropicKey") or os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        model = AnthropicModel(
            model_id=model_id or "global.anthropic.claude-opus-4-6-v1"
        )
    elif provider == "openai":
        from strands.models.openai import OpenAIModel

        api_key = config.get("openaiKey") or os.getenv("OPENAI_API_KEY")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        model = OpenAIModel(model_id=model_id or "gpt-5.2-2025-12-11")
    else:
        try:
            from strands.models.bedrock import BedrockModel
            import boto3

            region = config.get("bedrockRegion") or os.getenv("AWS_REGION", "us-east-1")
            boto_session = boto3.Session(region_name=region)
            model = BedrockModel(
                model_id=model_id
                or os.getenv("STRANDS_MODEL_ID", "global.anthropic.claude-opus-4-6-v1"),
                max_tokens=8192,
                boto_session=boto_session,
            )
        except Exception:
            from strands.models.ollama import OllamaModel

            model = OllamaModel(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                model_id="qwen3:1.7b",
                temperature=1,
            )

    # Build system prompt: parent prompt + ring context + custom
    parent_prompt = _get_parent_system_prompt()
    ring_context = _get_ring_context_str(max_entries=10)
    custom_prompt = config.get("systemPrompt", "")
    context = get_context_prompt_from_shared()

    full_prompt = parent_prompt if parent_prompt else base_system_prompt
    if context:
        full_prompt += f"\n{context}"
    if ring_context:
        full_prompt += f"\n{ring_context}"
    if custom_prompt:
        full_prompt += f"\n\n## Custom Instructions:\n{custom_prompt}"

    # Get tools from parent agent
    tools = _get_parent_tools()
    shared_messages = get_shared_messages()

    return Agent(
        model=model,
        system_prompt=full_prompt,
        tools=tools,
        messages=shared_messages,
    )


def create_bidi_agent(
    session: ClientSession, base_system_prompt: str, client_messages: list = None
):
    """Create a bidirectional audio agent with parent agent tools + ring context.

    Inherits tools and system prompt from the parent DevDuck agent for full
    capability. Adds ring context for mesh awareness. Instructs concise,
    direct tool execution — time is critical in voice interactions.

    Args:
        session: Client session
        base_system_prompt: Base system prompt for the agent
        client_messages: Optional list of messages from client for conversation context

    Returns:
        BidiAgent instance ready to start()
    """
    config = session.config
    voice_provider = config.get("voiceProvider", "novasonic")
    voice = config.get("voice")

    try:
        from strands.experimental.bidi import BidiAgent
        from strands.experimental.bidi import stop_conversation
    except ImportError:
        raise ImportError(
            "strands-agents[bidi-all] required for voice support. "
            "Install with: pip install 'strands-agents[bidi-all]'"
        )

    # === SYSTEM PROMPT: parent prompt + ring context + voice directives ===
    parent_prompt = _get_parent_system_prompt()
    ring_context = _get_ring_context_str(max_entries=10)
    custom_prompt = config.get("systemPrompt", "")
    context = get_context_prompt_from_shared()

    # Use parent system prompt as base if available, otherwise fallback
    if parent_prompt:
        full_prompt = parent_prompt
    else:
        full_prompt = base_system_prompt

    # Append conversation context
    if context:
        full_prompt += f"\n{context}"

    # Append ring context for mesh awareness
    if ring_context:
        full_prompt += f"\n{ring_context}"

    # Append custom instructions from browser config
    if custom_prompt:
        full_prompt += f"\n\n## Custom Instructions:\n{custom_prompt}"

    # Voice-specific directives — concise, direct, action-oriented
    full_prompt += """

## Voice Mode Directives:
- You are in LIVE VOICE mode. Time is critical.
- Speak SHORT and CONCISE. Max 1-2 sentences per response.
- Execute tools DIRECTLY when asked. Don't describe what you'll do — just do it.
- No verbose explanations. Action first, brief summary after.
- If a tool call is needed, call it immediately without asking for confirmation.
- Use parallel tool calls when possible for speed.
"""

    # === TOOLS: inherit ALL from parent agent + stop_conversation ===
    tools = [stop_conversation]
    parent_tools = _get_parent_tools()
    if parent_tools:
        tools.extend(parent_tools)
        inherited_names = [getattr(t, "__name__", "?") for t in parent_tools]
        logger.info(
            f"Bidi agent inheriting {len(parent_tools)} tools from parent: {inherited_names[:10]}..."
        )

    # Create model based on voice provider
    if voice_provider == "novasonic":
        from strands.experimental.bidi.models.nova_sonic import BidiNovaSonicModel

        provider_config = {"audio": {"voice": voice or "tiffany"}}
        model = BidiNovaSonicModel(
            model_id=os.getenv("NOVA_SONIC_MODEL_ID", "amazon.nova-2-sonic-v1:0"),
            provider_config=provider_config,
            client_config={
                "region": config.get("bedrockRegion")
                or os.getenv("AWS_REGION", "us-east-1"),
            },
        )
    elif voice_provider == "gemini_live":
        from strands.experimental.bidi.models.gemini_live import BidiGeminiLiveModel

        api_key = (
            config.get("geminiKey")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY required for Gemini Live"
            )
        model_kwargs = {
            "model_id": os.getenv(
                "GEMINI_MODEL_ID", "gemini-2.5-flash-native-audio-preview-09-2025"
            ),
            "client_config": {"api_key": api_key},
        }
        if voice:
            model_kwargs["provider_config"] = {"audio": {"voice": voice}}
        model = BidiGeminiLiveModel(**model_kwargs)
    elif voice_provider == "openai":
        from strands.experimental.bidi.models.openai_realtime import (
            BidiOpenAIRealtimeModel,
        )

        api_key = config.get("openaiKey") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI Realtime")
        model = BidiOpenAIRealtimeModel(
            model_id=os.getenv("OPENAI_MODEL_ID", "gpt-4o-realtime-preview-2025-06-03"),
            provider_config={"audio": {"voice": voice or "alloy"}},
            client_config={"api_key": api_key},
        )
    else:
        raise ValueError(f"Unknown voice provider: {voice_provider}")

    shared_messages = get_shared_messages()
    if client_messages:
        for msg in client_messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    formatted_msg = {
                        "role": role,
                        "content": [{"text": f"[context] {content}"}],
                    }
                    if formatted_msg not in shared_messages:
                        shared_messages.append(formatted_msg)
        logger.info(f"Merged {len(client_messages)} client messages into context")

    logger.info(
        f"Creating BidiAgent with {len(shared_messages)} messages, provider={voice_provider}"
    )

    bidi_agent = BidiAgent(
        model=model,
        system_prompt=full_prompt,
        tools=[t for t in tools if hasattr(t, "__name__")],
        messages=shared_messages,
    )

    return bidi_agent


# ============================================================================
# Message Handlers
# ============================================================================


async def handle_text_message(
    session: ClientSession, text: str, base_system_prompt: str
) -> None:
    """Process a text message with streaming response."""
    turn_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()

    add_shared_message("user", text, source="text")

    await session.websocket.send(
        json.dumps(
            {
                "type": "turn_start",
                "turn_id": turn_id,
                "data": text,
                "source": "text",
                "timestamp": time.time(),
            }
        )
    )

    session.text_agent = create_text_agent(session, base_system_prompt)
    handler = WebSocketStreamingCallbackHandler(
        session.websocket, loop, turn_id, "text"
    )
    session.text_agent.callback_handler = handler

    try:
        await loop.run_in_executor(_executor, session.text_agent, text)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        await session.websocket.send(
            json.dumps(
                {
                    "type": "error",
                    "turn_id": turn_id,
                    "data": str(e),
                    "timestamp": time.time(),
                }
            )
        )

    if handler.accumulated_text:
        add_shared_message("assistant", handler.accumulated_text, source="text")

    # Push to ring context
    try:
        from devduck.tools.unified_mesh import add_to_ring

        preview = (handler.accumulated_text or "")[:150]
        add_to_ring(
            "ws:client",
            "browser",
            f"Q: {text[:80]} → {preview}",
            {"source": "websocket"},
        )
    except Exception:
        pass

    await session.websocket.send(
        json.dumps(
            {
                "type": "turn_end",
                "turn_id": turn_id,
                "timestamp": time.time(),
            }
        )
    )


async def handle_audio_start(
    session: ClientSession,
    provider: str,
    voice: str,
    base_system_prompt: str,
    client_messages: list = None,
) -> None:
    """Start bidirectional audio session."""
    if session.audio_active:
        await handle_audio_stop(session)

    try:
        session.config["voiceProvider"] = provider
        if voice:
            session.config["voice"] = voice

        logger.info(
            f"Starting audio session for {session.client_id} with provider={provider}, voice={voice or 'default'}"
        )

        bidi_agent = create_bidi_agent(
            session, base_system_prompt, client_messages or []
        )
        session.bidi_agent = bidi_agent

        logger.info("Starting BidiAgent")
        await session.bidi_agent.start()
        session.audio_active = True

        session.bidi_task = asyncio.create_task(forward_bidi_events(session))

        await session.websocket.send(
            json.dumps(
                {
                    "type": "audio_started",
                    "provider": provider,
                    "timestamp": time.time(),
                }
            )
        )

        # Push to ring context
        try:
            from devduck.tools.unified_mesh import add_to_ring

            add_to_ring(
                "ws:voice",
                "browser",
                f"🎤 Voice session started ({provider})",
                {"source": "websocket"},
            )
        except Exception:
            pass

        # Register speech agent in mesh registry so browser sees it
        try:
            from devduck.tools.mesh_registry import registry

            speech_agent_id = f"speech:{session.client_id}"
            session.config["_speech_agent_id"] = speech_agent_id
            registry.register(
                agent_id=speech_agent_id,
                agent_type="speech",
                metadata={
                    "provider": provider,
                    "voice": voice or "default",
                    "client_id": session.client_id,
                    "name": f"🎙️ Voice ({provider})",
                },
            )
            logger.info(f"Registered speech agent in mesh: {speech_agent_id}")
        except Exception as e:
            logger.debug(f"Could not register speech agent in mesh: {e}")

        logger.info(f"Audio session started for {session.client_id} with {provider}")

    except Exception as e:
        logger.error(f"Failed to start audio with {provider}: {e}", exc_info=True)
        session.audio_active = False
        session.bidi_agent = None
        await session.websocket.send(
            json.dumps(
                {
                    "type": "error",
                    "data": f"Failed to start audio: {e}",
                    "timestamp": time.time(),
                }
            )
        )


async def forward_bidi_events(session: ClientSession) -> None:
    """Forward events from BidiAgent to WebSocket."""
    try:
        from strands.experimental.bidi import (
            BidiAudioStreamEvent,
            BidiTranscriptStreamEvent,
            BidiInterruptionEvent,
            BidiConnectionStartEvent,
            BidiConnectionCloseEvent,
            BidiResponseStartEvent,
            BidiResponseCompleteEvent,
            BidiUsageEvent,
            BidiErrorEvent,
        )

        async for event in session.bidi_agent.receive():
            msg = None
            ts = time.time()

            if isinstance(event, BidiAudioStreamEvent):
                msg = {
                    "type": "audio_chunk",
                    "audio": event.audio,
                    "format": event.format,
                    "sample_rate": event.sample_rate,
                    "channels": event.channels,
                    "timestamp": ts,
                }
            elif isinstance(event, BidiTranscriptStreamEvent):
                msg = {
                    "type": "transcript",
                    "text": event.text,
                    "role": event.role,
                    "is_final": event.is_final,
                    "source": "voice",
                    "timestamp": ts,
                }
                if event.is_final:
                    add_shared_message(event.role, event.text, source="voice")
                    # Push final transcripts to ring
                    try:
                        from devduck.tools.unified_mesh import add_to_ring

                        emoji = "🎤" if event.role == "user" else "🔊"
                        add_to_ring(
                            "ws:voice",
                            "browser",
                            f"{emoji} {event.text[:150]}",
                            {"source": "voice"},
                        )
                    except Exception:
                        pass
            elif isinstance(event, BidiInterruptionEvent):
                msg = {"type": "interruption", "reason": event.reason, "timestamp": ts}
            elif isinstance(event, BidiConnectionStartEvent):
                msg = {
                    "type": "bidi_connection_start",
                    "connection_id": event.connection_id,
                    "model": event.model,
                    "timestamp": ts,
                }
            elif isinstance(event, BidiResponseStartEvent):
                msg = {
                    "type": "response_start",
                    "response_id": event.response_id,
                    "timestamp": ts,
                }
            elif isinstance(event, BidiResponseCompleteEvent):
                msg = {
                    "type": "response_complete",
                    "response_id": event.response_id,
                    "stop_reason": event.stop_reason,
                    "timestamp": ts,
                }
            elif isinstance(event, BidiUsageEvent):
                msg = {
                    "type": "usage",
                    "input_tokens": event.input_tokens,
                    "output_tokens": event.output_tokens,
                    "total_tokens": event.total_tokens,
                    "timestamp": ts,
                }
            elif isinstance(event, BidiErrorEvent):
                logger.error(f"Bidi error event: {event.message}")
                msg = {
                    "type": "error",
                    "data": event.message,
                    "code": event.code,
                    "timestamp": ts,
                }
            elif isinstance(event, BidiConnectionCloseEvent):
                msg = {"type": "audio_stopped", "reason": event.reason, "timestamp": ts}
                break

            if msg:
                try:
                    await session.websocket.send(json.dumps(msg))
                except Exception as e:
                    logger.debug(f"Failed to send bidi event: {e}")
                    break

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Error forwarding bidi events: {e}")
        try:
            await asyncio.wait_for(
                session.websocket.send(
                    json.dumps(
                        {
                            "type": "error",
                            "data": f"Voice session error: {e}",
                            "timestamp": time.time(),
                        }
                    )
                ),
                timeout=1.0,
            )
            await asyncio.wait_for(
                session.websocket.send(
                    json.dumps(
                        {
                            "type": "audio_stopped",
                            "reason": "error",
                            "timestamp": time.time(),
                        }
                    )
                ),
                timeout=1.0,
            )
        except Exception:
            pass
    finally:
        session.audio_active = False


async def handle_audio_chunk(
    session: ClientSession, audio_b64: str, sample_rate: int = 16000
) -> None:
    """Forward audio chunk to bidi agent."""
    if not session.audio_active or not session.bidi_agent:
        return
    try:
        from strands.experimental.bidi import BidiAudioInputEvent

        event = BidiAudioInputEvent(
            audio=audio_b64, format="pcm", sample_rate=sample_rate, channels=1
        )
        await session.bidi_agent.send(event)
    except Exception as e:
        logger.error(f"Error sending audio chunk: {e}")


async def handle_audio_stop(session: ClientSession) -> None:
    """Stop bidirectional audio session."""
    if not session.audio_active and not session.bidi_agent:
        return

    session.audio_active = False

    try:
        if session.bidi_task:
            session.bidi_task.cancel()
            try:
                await asyncio.wait_for(session.bidi_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.debug(f"Error waiting for bidi task: {e}")
            session.bidi_task = None

        if session.bidi_agent:
            try:
                await asyncio.wait_for(session.bidi_agent.stop(), timeout=3.0)
            except asyncio.TimeoutError:
                logger.warning("Timeout stopping bidi agent")
            except Exception as e:
                error_str = str(e)
                if (
                    "1011" in error_str
                    or "1001" in error_str
                    or "going away" in error_str.lower()
                ):
                    logger.debug(f"Bidi agent closed by provider: {e}")
                else:
                    logger.warning(f"Error stopping bidi agent: {e}")
            session.bidi_agent = None

        try:
            await session.websocket.send(
                json.dumps(
                    {
                        "type": "audio_stopped",
                        "reason": "user_request",
                        "timestamp": time.time(),
                    }
                )
            )
        except Exception:
            pass

        # Push to ring context
        try:
            from devduck.tools.unified_mesh import add_to_ring

            add_to_ring(
                "ws:voice", "browser", "🎤 Voice session ended", {"source": "websocket"}
            )
        except Exception:
            pass

        # Unregister speech agent from mesh registry
        try:
            speech_agent_id = session.config.get("_speech_agent_id")
            if speech_agent_id:
                from devduck.tools.mesh_registry import registry

                registry.unregister(speech_agent_id)
                logger.info(f"Unregistered speech agent from mesh: {speech_agent_id}")
                del session.config["_speech_agent_id"]
        except Exception as e:
            logger.debug(f"Could not unregister speech agent from mesh: {e}")

        logger.info(f"Audio session stopped for {session.client_id}")

    except Exception as e:
        logger.error(f"Error stopping audio: {e}")


# ============================================================================
# Zenoh Push (bridges P2P network to browser)
# ============================================================================


async def push_zenoh_updates(websocket, loop):
    """Background task that pushes Zenoh peer updates to the browser."""
    last_peers = set()
    first_run = True

    while True:
        try:
            if not first_run:
                await asyncio.sleep(2)
            first_run = False

            try:
                from devduck.tools.zenoh_peer import ZENOH_STATE, get_instance_id

                if not ZENOH_STATE.get("running"):
                    continue

                current_peers = set(ZENOH_STATE.get("peers", {}).keys())
                new_peers = current_peers - last_peers
                lost_peers = last_peers - current_peers
                force_send = last_peers == set() and current_peers

                if new_peers or lost_peers or force_send:
                    update = {
                        "type": "zenoh_peers_update",
                        "instance_id": get_instance_id(),
                        "peers": [
                            {
                                "id": pid,
                                "hostname": ZENOH_STATE["peers"][pid].get(
                                    "hostname", "unknown"
                                ),
                                "model": ZENOH_STATE["peers"][pid].get(
                                    "model", "unknown"
                                ),
                                "last_seen": ZENOH_STATE["peers"][pid].get(
                                    "last_seen", 0
                                ),
                            }
                            for pid in current_peers
                        ],
                        "new_peers": list(new_peers),
                        "lost_peers": list(lost_peers),
                        "timestamp": time.time(),
                    }
                    await websocket.send(json.dumps(update))
                    logger.debug(
                        f"Pushed Zenoh update: +{len(new_peers)} -{len(lost_peers)} peers (total: {len(current_peers)})"
                    )
                    last_peers = current_peers

            except ImportError:
                pass

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug(f"Zenoh push error: {e}")
            await asyncio.sleep(5)


# ============================================================================
# Main WebSocket Handler
# ============================================================================


async def handle_websocket_client(websocket, system_prompt: str):
    """Handle a WebSocket client connection with text + voice support."""
    client_address = websocket.remote_address
    logger.info(f"WebSocket connection established with {client_address}")

    client_id = str(uuid.uuid4())
    session = ClientSession(websocket=websocket, client_id=client_id)

    # Register in mesh
    ws_id = str(uuid.uuid4())[:8]
    try:
        from devduck.tools.unified_mesh import MESH_STATE

        MESH_STATE["ws_clients"][ws_id] = websocket
    except ImportError:
        pass

    loop = asyncio.get_running_loop()
    active_tasks = set()

    try:
        # Build welcome message with shared history + zenoh info
        history = get_messages_for_client(20)
        welcome = {
            "type": "connected",
            "client_id": client_id,
            "ws_id": ws_id,
            "data": "🦆 DevDuck Voice+Text Ready!",
            "history": history,
            "timestamp": time.time(),
        }
        try:
            from devduck.tools.zenoh_peer import ZENOH_STATE, get_instance_id

            if ZENOH_STATE.get("running"):
                peers_dict = ZENOH_STATE.get("peers", {})
                welcome["zenoh"] = {
                    "instance_id": get_instance_id(),
                    "peers": [
                        {
                            "id": pid,
                            "hostname": peers_dict[pid].get("hostname", "unknown"),
                            "model": peers_dict[pid].get("model", "unknown"),
                            "last_seen": peers_dict[pid].get("last_seen", 0),
                        }
                        for pid in peers_dict.keys()
                    ],
                    "peer_count": len(peers_dict),
                }
        except ImportError:
            pass

        await websocket.send(json.dumps(welcome))

        # Start zenoh push background task
        zenoh_task = asyncio.create_task(push_zenoh_updates(websocket, loop))
        active_tasks.add(zenoh_task)
        zenoh_task.add_done_callback(active_tasks.discard)

        async for raw_message in websocket:
            raw_message = (raw_message or "").strip()
            if not raw_message:
                continue

            # JSON protocol
            if raw_message.startswith("{"):
                try:
                    msg = json.loads(raw_message)
                    msg_type = msg.get("type")

                    if msg_type == "text":
                        task = asyncio.create_task(
                            handle_text_message(
                                session, msg.get("text", ""), system_prompt
                            )
                        )
                        active_tasks.add(task)
                        task.add_done_callback(active_tasks.discard)

                    elif msg_type == "audio_start":
                        provider = msg.get("provider", "novasonic")
                        voice = msg.get("voice")
                        client_messages = msg.get("messages", [])
                        # Update session config with any keys sent in audio_start
                        for key in ("geminiKey", "openaiKey", "bedrockRegion"):
                            if key in msg and msg[key]:
                                session.config[key] = msg[key]
                        await handle_audio_start(
                            session, provider, voice, system_prompt, client_messages
                        )

                    elif msg_type == "audio_chunk":
                        await handle_audio_chunk(
                            session, msg.get("audio", ""), msg.get("sample_rate", 16000)
                        )

                    elif msg_type == "audio_stop":
                        await handle_audio_stop(session)

                    elif msg_type == "config":
                        for key in [
                            "provider",
                            "anthropicKey",
                            "openaiKey",
                            "bedrockRegion",
                            "modelId",
                            "systemPrompt",
                            "voiceProvider",
                            "voice",
                            "ollamaHost",
                            "geminiKey",
                        ]:
                            if key in msg:
                                session.config[key] = msg[key]
                        await websocket.send(
                            json.dumps(
                                {"type": "config_updated", "timestamp": time.time()}
                            )
                        )

                    elif msg_type == "clear_history":
                        try:
                            from devduck import devduck as dd

                            if dd.agent and hasattr(dd.agent, "messages"):
                                dd.agent.messages.clear()
                                logger.info("Cleared shared message history")
                        except Exception as e:
                            logger.debug(f"Could not clear history: {e}")
                        await websocket.send(
                            json.dumps(
                                {"type": "history_cleared", "timestamp": time.time()}
                            )
                        )

                    elif msg_type == "get_history":
                        history = get_messages_for_client(50)
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "history",
                                    "messages": history,
                                    "timestamp": time.time(),
                                }
                            )
                        )

                    elif msg_type == "ping":
                        await websocket.send(
                            json.dumps({"type": "pong", "timestamp": time.time()})
                        )

                    else:
                        # Ignore relay protocol messages (meant for agentcore_proxy)
                        if msg_type in (
                            "configure",
                            "list_agents",
                            "get_ring",
                            "trigger_github",
                            "invoke",
                            "broadcast",
                            "list_peers",
                            "get_status",
                        ):
                            logger.info(f"Ignoring relay protocol message: {msg_type}")
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "data": f"This is a DevDuck WS server, not a relay. '{msg_type}' not supported here.",
                                        "timestamp": time.time(),
                                    }
                                )
                            )
                        else:
                            logger.debug(f"Unknown message type: {msg_type}")

                except json.JSONDecodeError:
                    # Treat as plain text
                    task = asyncio.create_task(
                        handle_text_message(session, raw_message, system_prompt)
                    )
                    active_tasks.add(task)
                    task.add_done_callback(active_tasks.discard)
            else:
                # Plain text message
                if raw_message.lower() == "exit":
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "disconnected",
                                "data": "Goodbye!",
                                "timestamp": time.time(),
                            }
                        )
                    )
                    break

                task = asyncio.create_task(
                    handle_text_message(session, raw_message, system_prompt)
                )
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)

        # Wait for active tasks
        if active_tasks:
            logger.info(f"Waiting for {len(active_tasks)} active tasks to complete...")
            await asyncio.gather(*active_tasks, return_exceptions=True)

    except Exception as e:
        logger.error(f"Error handling client {client_address}: {e}", exc_info=True)
    finally:
        await handle_audio_stop(session)
        # Unregister from mesh
        try:
            from devduck.tools.unified_mesh import MESH_STATE

            if ws_id in MESH_STATE["ws_clients"]:
                del MESH_STATE["ws_clients"][ws_id]
        except Exception:
            pass
        logger.info(f"WebSocket connection with {client_address} closed")


# ============================================================================
# Server Runner
# ============================================================================


def run_websocket_server(host: str, port: int, system_prompt: str) -> None:
    """Run WebSocket server with voice + text support."""
    import websockets

    WS_SERVER_THREADS[port]["running"] = True
    WS_SERVER_THREADS[port]["connections"] = 0
    WS_SERVER_THREADS[port]["start_time"] = time.time()

    async def server_handler(websocket):
        WS_SERVER_THREADS[port]["connections"] += 1
        await handle_websocket_client(websocket, system_prompt)

    async def start_server():
        stop_future = asyncio.Future()
        WS_SERVER_THREADS[port]["stop_future"] = stop_future

        server = await websockets.serve(server_handler, host, port)
        logger.info(f"WebSocket Server (Voice+Text) listening on {host}:{port}")

        await stop_future

        server.close()
        await server.wait_closed()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        WS_SERVER_THREADS[port]["loop"] = loop
        loop.run_until_complete(start_server())
    except OSError as e:
        if "Address already in use" in str(e) or "address already in use" in str(e):
            logger.debug(f"Port {port} unavailable")
        else:
            logger.error(f"WebSocket server error: {e}")
    except Exception as e:
        logger.error(f"WebSocket server error: {e}")
    finally:
        logger.info(f"WebSocket Server on {host}:{port} stopped")
        WS_SERVER_THREADS[port]["running"] = False


# ============================================================================
# Tool Interface
# ============================================================================


@tool
def websocket(
    action: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    system_prompt: str = "You are a helpful WebSocket server assistant.",
    agent=None,
) -> dict:
    """Create and manage WebSocket servers with real-time streaming.

    Args:
        action: Action to perform (start_server, stop_server, get_status)
        host: Host address for server
        port: Port number for server
        system_prompt: System prompt for the server DevDuck instances
        agent: Parent agent (automatically passed by Strands framework)

    Returns:
        Dictionary containing status and response content
    """
    # Store parent agent reference globally so bidi/text agents can inherit
    if agent is not None:
        _PARENT_AGENT_REF["agent"] = agent
    if action == "start_server":
        if port in WS_SERVER_THREADS and WS_SERVER_THREADS[port].get("running", False):
            return {
                "status": "error",
                "content": [
                    {"text": f"❌ WebSocket Server already running on port {port}"}
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
                    {"text": f"❌ Failed to start WebSocket Server on {host}:{port}"}
                ],
            }

        return {
            "status": "success",
            "content": [
                {"text": f"✅ WebSocket Server started on {host}:{port}"},
                {"text": "🎤 Voice support: Nova Sonic, Gemini Live, OpenAI Realtime"},
                {"text": "💬 Text chat with streaming responses"},
                {"text": "🔗 Zenoh peer updates pushed to browser"},
                {"text": f"📝 Connect: ws://localhost:{port}"},
            ],
        }

    elif action == "stop_server":
        if port not in WS_SERVER_THREADS or not WS_SERVER_THREADS[port].get(
            "running", False
        ):
            return {
                "status": "error",
                "content": [{"text": f"❌ No WebSocket Server running on port {port}"}],
            }

        WS_SERVER_THREADS[port]["running"] = False
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
                {"text": f"✅ WebSocket Server on port {port} stopped"},
                {"text": f"Stats: {connections} connections, uptime {uptime:.2f}s"},
            ],
        }

    elif action == "get_status":
        if not WS_SERVER_THREADS:
            return {
                "status": "success",
                "content": [{"text": "No WebSocket Servers running"}],
            }

        status_info = []
        for p, data in WS_SERVER_THREADS.items():
            if data.get("running", False):
                uptime = time.time() - data.get("start_time", time.time())
                connections = data.get("connections", 0)
                status_info.append(
                    f"Port {p}: Running - {connections} connections, {uptime:.2f}s uptime"
                )

        return {
            "status": "success",
            "content": [
                {"text": "WebSocket Server Status:"},
                {"text": "\n".join(status_info) or "No active servers"},
            ],
        }

    else:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"Unknown action: {action}. Valid: start_server, stop_server, get_status"
                }
            ],
        }
