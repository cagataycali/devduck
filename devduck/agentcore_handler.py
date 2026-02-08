"""
🦆 DevDuck AgentCore Handler - Full DevDuck on AgentCore

This handler deploys a FULL DevDuck instance to Amazon Bedrock AgentCore,
not just a raw strands Agent - you get the complete DevDuck experience:
- Self-healing capabilities
- Hot-reload awareness
- Session recording
- Ambient mode support
- All DevDuck tools
- System prompt management

Deploy with:
    devduck deploy                    # Uses defaults
    devduck deploy --name my-agent    # Custom name
    devduck deploy --tools "strands_tools:shell,editor"
    devduck deploy --launch           # Auto-launch after configure

Or manually:
    agentcore configure -e devduck/agentcore_handler.py -n devduck
    agentcore launch -a devduck
"""

import json
import logging
import os
import platform
import socket
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# =============================================================================
# AgentCore Environment Setup - BEFORE importing devduck
# =============================================================================

# Disable auto-start servers (we're running in AgentCore)
os.environ["DEVDUCK_AUTO_START_SERVERS"] = "false"
os.environ["DEVDUCK_LOAD_TOOLS_FROM_DIR"] = "false"
os.environ["MODEL_PROVIDER"] = "bedrock"
os.environ["BYPASS_TOOL_CONSENT"] = "true"
os.environ["DEVDUCK_ENABLE_WS"] = "false"
os.environ["DEVDUCK_ENABLE_TCP"] = "false"
os.environ["DEVDUCK_ENABLE_MCP"] = "false"
os.environ["DEVDUCK_ENABLE_IPC"] = "false"
os.environ["DEVDUCK_ENABLE_ZENOH"] = "false"
os.environ["DEVDUCK_ENABLE_AGENTCORE_PROXY"] = "false"

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("devduck.agentcore")

# =============================================================================
# AgentCore App
# =============================================================================

from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

# =============================================================================
# Session & State Management
# =============================================================================

# Session storage: session_id -> {devduck, config, created, last_used}
sessions: Dict[str, Dict[str, Any]] = {}

# Default configuration
DEFAULT_TOOLS = os.getenv(
    "DEVDUCK_TOOLS",
    "devduck.tools:system_prompt,store_in_kb,scraper;strands_tools:retrieve,shell,file_read,file_write,editor,use_agent",
)

DEFAULT_MODEL = os.getenv(
    "STRANDS_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0"
)

DEFAULT_MAX_TOKENS = int(os.getenv("STRANDS_MAX_TOKENS", "60000"))
DEFAULT_TEMPERATURE = float(os.getenv("STRANDS_TEMPERATURE", "1.0"))

# =============================================================================
# DevDuck Factory - Create configured DevDuck instances
# =============================================================================


def create_devduck(
    system_prompt: Optional[str] = None,
    tools_config: Optional[str] = None,
    model_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    messages: Optional[List[Dict]] = None,
):
    """
    Create a configured DevDuck instance for AgentCore.

    This creates a REAL DevDuck with all its capabilities,
    not just a raw strands Agent.
    """
    # Set environment for this instance
    if tools_config:
        os.environ["DEVDUCK_TOOLS"] = tools_config
    else:
        os.environ["DEVDUCK_TOOLS"] = DEFAULT_TOOLS

    if model_id:
        os.environ["STRANDS_MODEL_ID"] = model_id

    if max_tokens:
        os.environ["STRANDS_MAX_TOKENS"] = str(max_tokens)

    if temperature is not None:
        os.environ["STRANDS_TEMPERATURE"] = str(temperature)

    if system_prompt:
        os.environ["SYSTEM_PROMPT"] = system_prompt

    # Import DevDuck AFTER setting environment
    # This ensures DevDuck picks up our configuration
    from devduck import DevDuck

    # Create DevDuck instance with servers disabled
    duck = DevDuck(auto_start_servers=False)

    # Restore messages if provided (ring attention)
    if messages and duck.agent:
        duck.agent.messages = messages
        logger.info(f"Restored {len(messages)} messages for ring attention")

    logger.info(f"Created DevDuck instance with model: {duck.model}")
    return duck


# =============================================================================
# Session Management
# =============================================================================


def get_or_create_session(
    session_id: str,
    system_prompt: Optional[str] = None,
    tools_config: Optional[str] = None,
    model_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    messages: Optional[List[Dict]] = None,
    force_new: bool = False,
) -> Dict[str, Any]:
    """
    Get existing session or create new one with full DevDuck.
    """
    current_config = {
        "system_prompt": system_prompt,
        "tools_config": tools_config,
        "model_id": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if session_id not in sessions or force_new:
        duck = create_devduck(
            system_prompt=system_prompt,
            tools_config=tools_config,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )

        sessions[session_id] = {
            "devduck": duck,
            "config": current_config,
            "created": time.time(),
            "last_used": time.time(),
            "request_count": 0,
        }
        logger.info(f"Created new DevDuck session: {session_id}")
    else:
        session = sessions[session_id]
        session["last_used"] = time.time()
        session["request_count"] += 1

        # Inject messages for ring attention
        if messages and session["devduck"].agent:
            session["devduck"].agent.messages = messages

        # Check if config changed
        old_config = session.get("config", {})
        config_changed = any(
            current_config.get(k) and current_config.get(k) != old_config.get(k)
            for k in current_config.keys()
        )

        if config_changed:
            logger.info(f"Config changed for session {session_id}, recreating DevDuck")

            existing_messages = messages or (
                session["devduck"].agent.messages if session["devduck"].agent else []
            )

            duck = create_devduck(
                system_prompt=system_prompt or old_config.get("system_prompt"),
                tools_config=tools_config or old_config.get("tools_config"),
                model_id=model_id or old_config.get("model_id"),
                max_tokens=max_tokens or old_config.get("max_tokens"),
                temperature=(
                    temperature
                    if temperature is not None
                    else old_config.get("temperature")
                ),
                messages=existing_messages,
            )

            session["devduck"] = duck
            session["config"] = {
                k: current_config.get(k) or old_config.get(k)
                for k in current_config.keys()
            }

    return sessions[session_id]


# =============================================================================
# AgentCore Entrypoint - Using DevDuck
# =============================================================================


@app.entrypoint
async def invoke(payload: Dict[str, Any], context):
    """
    🦆 DevDuck AgentCore Entrypoint

    This runs a REAL DevDuck instance, not just a raw Agent.

    Supports three modes:
    - sync: Blocking call, returns full response
    - streaming: SSE stream with chunks
    - async: Fire-and-forget with task_id

    Payload format:
    {
        "prompt": "your question",
        "mode": "streaming|sync|async",
        "session_id": "optional-session-id",

        # Configuration:
        "system_prompt": "Custom system prompt",
        "tools": "strands_tools:shell,editor",
        "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "max_tokens": 60000,
        "temperature": 1.0,

        # Ring attention:
        "messages": [{"role": "user", "content": [...]}],

        # Control:
        "force_new_session": false,
        "return_messages": false,
    }
    """
    # Handle health check
    if payload.get("type") == "health":
        yield get_health_response()
        return

    # Extract parameters
    mode = payload.get("mode", "streaming")
    query = payload.get("prompt", payload.get("text", ""))
    session_id = payload.get("session_id", str(uuid.uuid4()))

    # Ensure session_id meets minimum length (AgentCore requirement)
    if len(session_id) < 33:
        session_id = f"{session_id}-{str(uuid.uuid4())[:33 - len(session_id) - 1]}"

    # Configuration
    system_prompt = payload.get("system_prompt")
    tools_config = payload.get("tools")
    model_id = payload.get("model")
    max_tokens = payload.get("max_tokens")
    temperature = payload.get("temperature")
    messages = payload.get("messages")
    force_new = payload.get("force_new_session", False)
    return_messages = payload.get("return_messages", False)

    if not query:
        yield {"type": "error", "error": "No prompt provided"}
        return

    logger.info(
        f"Request: mode={mode}, query={query[:50]}..., session={session_id[:16]}..."
    )

    # Get or create DevDuck session
    try:
        session = get_or_create_session(
            session_id=session_id,
            system_prompt=system_prompt,
            tools_config=tools_config,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            force_new=force_new,
        )
        duck = session["devduck"]
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        yield {"type": "error", "error": f"Session creation failed: {e}"}
        return

    # ==========================================================================
    # SYNC MODE
    # ==========================================================================
    if mode == "sync":
        try:
            logger.info("Running DevDuck in sync mode")
            result = duck(query)  # Use DevDuck's __call__

            response = {
                "type": "response",
                "status": "success",
                "session_id": session_id,
                "response": str(result),
                "model": duck.model,
            }

            if return_messages and duck.agent:
                response["messages"] = duck.agent.messages
                response["messages_count"] = len(duck.agent.messages)

            yield response

        except Exception as e:
            logger.error(f"Sync mode error: {e}")
            yield {"type": "error", "error": str(e)}

    # ==========================================================================
    # ASYNC MODE
    # ==========================================================================
    elif mode == "async":
        task_id = str(uuid.uuid4())[:16]
        logger.info(f"Running DevDuck in async mode, task_id={task_id}")

        def run_async():
            try:
                result = duck(query)
                logger.info(f"Async task {task_id} completed: {str(result)[:100]}...")
            except Exception as e:
                logger.error(f"Async task {task_id} failed: {e}")

        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()

        yield {
            "type": "async_started",
            "task_id": task_id,
            "session_id": session_id,
            "model": duck.model,
            "status": "processing",
        }

    # ==========================================================================
    # STREAMING MODE (default)
    # ==========================================================================
    else:
        turn_id = str(uuid.uuid4())[:8]
        full_response = ""

        logger.info(f"Running DevDuck in streaming mode, turn_id={turn_id}")

        yield {
            "type": "turn_start",
            "turn_id": turn_id,
            "session_id": session_id,
            "model": duck.model,
            "config": {
                "system_prompt": bool(system_prompt),
                "tools": bool(tools_config),
                "model": model_id or duck.model,
                "messages_injected": len(messages) if messages else 0,
            },
            "timestamp": time.time(),
        }

        try:
            # Use the underlying agent's streaming
            if not duck.agent:
                yield {"type": "error", "error": "DevDuck agent not available"}
                return

            stream = duck.agent.stream_async(query)

            async for event in stream:
                # Extract text chunks
                if "data" in event:
                    chunk_data = event.get("data")
                    if chunk_data:
                        chunk_str = str(chunk_data)
                        full_response += chunk_str
                        yield {
                            "type": "chunk",
                            "turn_id": turn_id,
                            "data": chunk_str,
                            "timestamp": time.time(),
                        }

                # Tool events
                if "current_tool_use" in event:
                    tool_use = event.get("current_tool_use")
                    if tool_use and tool_use.get("name"):
                        yield {
                            "type": "tool_start",
                            "turn_id": turn_id,
                            "tool_name": tool_use["name"],
                            "tool_id": tool_use.get("toolUseId", ""),
                            "timestamp": time.time(),
                        }

                if "tool_result" in event:
                    yield {
                        "type": "tool_result",
                        "turn_id": turn_id,
                        "result": str(event["tool_result"])[:1000],
                        "timestamp": time.time(),
                    }

            # Turn end
            turn_end = {
                "type": "turn_end",
                "turn_id": turn_id,
                "session_id": session_id,
                "full_response": full_response,
                "model": duck.model,
                "timestamp": time.time(),
            }

            if return_messages and duck.agent:
                turn_end["messages"] = duck.agent.messages
                turn_end["messages_count"] = len(duck.agent.messages)

            yield turn_end
            logger.info(
                f"Turn {turn_id} completed, response length: {len(full_response)}"
            )

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {
                "type": "error",
                "turn_id": turn_id,
                "error": str(e),
                "timestamp": time.time(),
            }


# =============================================================================
# Health Check
# =============================================================================


def get_health_response():
    """Generate health check response."""
    return {
        "type": "health",
        "status": "healthy",
        "service": "devduck",
        "version": "2.0.0",
        "sessions": len(sessions),
        "default_model": DEFAULT_MODEL,
        "default_tools": DEFAULT_TOOLS,
        "features": [
            "self-healing",
            "session-recording",
            "ambient-mode",
            "hot-reload",
            "ring-attention",
        ],
        "timestamp": time.time(),
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    logger.info("🦆 Starting DevDuck AgentCore Handler v2...")
    logger.info(f"Model: {DEFAULT_MODEL}")
    logger.info(f"Tools: {DEFAULT_TOOLS}")
    app.run()
