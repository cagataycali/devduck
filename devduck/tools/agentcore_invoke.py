"""AgentCore Invoke Tool - Invoke deployed DevDuck instances on AgentCore."""

import json
import uuid
from typing import Any, Dict, Optional
from strands import tool


@tool
def agentcore_invoke(
    prompt: str,
    agent_name: Optional[str] = None,
    agent_id: Optional[str] = None,
    agent_arn: Optional[str] = None,
    session_id: Optional[str] = None,
    mode: str = "streaming",
    tools: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    region: str = "us-west-2",
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """Invoke a deployed DevDuck instance on AgentCore.

    **Quick Start:**
    ```python
    # Use agent_id directly (RECOMMENDED - no config needed):
    agentcore_invoke(agent_id="devduck-abc123", prompt="hello")

    # Or by name (requires local config):
    agentcore_invoke(agent_name="my-agent", prompt="hello")
    ```

    **Priority:** agent_arn > agent_id > agent_name (config lookup)

    **Note:** Agent names with hyphens are auto-converted to underscores for config lookup.
    For example: "test-agent-1" becomes "test_agent_1" in the config file.

    Args:
        prompt: Query/prompt to send to the agent
        agent_name: Name of deployed agent (default: "devduck")
            - Hyphens auto-converted to underscores for config lookup
            - Example: "test-agent" → looks for "test_agent" in config
        agent_id: Direct agent ID (e.g., "devduck-UrvQvkH6H7") - RECOMMENDED
            - Bypasses config lookup entirely
            - Get from agentcore_agents() or deployment output
        agent_arn: Direct agent ARN - bypasses config lookup
        session_id: Session ID for continuity (auto-generated if not provided)
        mode: Invocation mode (streaming, sync, async) - default: streaming
        tools: Comma-separated list of tools to enable (optional)
            Example: "file_read,calculator,shell"
        model: Model ID override (optional)
        system_prompt: System prompt override (optional)
        region: AWS region (default: us-west-2)
        agent: Parent agent for streaming callbacks

    Returns:
        Dict with status and response

    Examples:
        # Use agent ID (RECOMMENDED - fastest, no config needed)
        agentcore_invoke(
            prompt="analyze this code",
            agent_id="devduck-UrvQvkH6H7"
        )

        # Use agent name (requires config file)
        agentcore_invoke(
            prompt="analyze this code",
            agent_name="my-agent"  # Auto-converts to "my_agent" for lookup
        )

        # With session continuity
        agentcore_invoke(
            prompt="continue our discussion",
            agent_id="devduck-UrvQvkH6H7",
            session_id="previous-session-123"
        )

        # Custom configuration
        agentcore_invoke(
            prompt="analyze data",
            agent_id="devduck-UrvQvkH6H7",
            model="us.anthropic.claude-sonnet-4-20250514-v1:0",
            tools="file_read,calculator,python_repl",
            system_prompt="You are a data analyst"
        )
    """
    try:
        import boto3
        import yaml
        from botocore.config import Config
        from pathlib import Path

        # Determine agent ARN - priority: agent_arn > agent_id > agent_name (config lookup)
        final_agent_arn = None

        if agent_arn:
            # Direct ARN provided - use it
            final_agent_arn = agent_arn
        elif agent_id:
            # Direct agent ID provided - construct ARN
            # Get account ID from STS
            sts = boto3.client("sts", region_name=region)
            account_id = sts.get_caller_identity()["Account"]
            final_agent_arn = (
                f"arn:aws:bedrock-agentcore:{region}:{account_id}:runtime/{agent_id}"
            )
        else:
            # Fall back to config lookup by agent_name
            if not agent_name:
                agent_name = "devduck"  # Default

            # Normalize agent name: hyphens → underscores (matches agentcore_config behavior)
            agent_name = agent_name.replace("-", "_")

            # Try to find config file
            import devduck as devduck_module

            devduck_module_path = Path(devduck_module.__file__).parent

            # Check if we're in development mode (has parent .git folder)
            dev_mode = (devduck_module_path.parent / ".git").exists()

            if dev_mode:
                # Development mode: use parent directory
                devduck_dir = devduck_module_path.parent
            else:
                # Installed mode: use module directory directly
                devduck_dir = devduck_module_path

            config_path = devduck_dir / ".bedrock_agentcore.yaml"

            if not config_path.exists():
                # No config file - try to list available agents
                try:
                    import sys
                    from pathlib import Path

                    # Add tools directory to path if not already there
                    tools_dir = Path(__file__).parent
                    if str(tools_dir) not in sys.path:
                        sys.path.insert(0, str(tools_dir))

                    from agentcore_agents import agentcore_agents

                    agents_result = agentcore_agents(action="list", region=region)

                    if agents_result.get("status") == "success":
                        # Extract all text content from the result
                        agents_list = "\n".join(
                            item.get("text", "")
                            for item in agents_result.get("content", [])
                            if "text" in item
                        )
                        return {
                            "status": "error",
                            "content": [
                                {
                                    "text": "❌ No agent specified. Provide agent_id or agent_arn directly."
                                },
                                {
                                    "text": "\n**💡 If you just launched an agent:**\n"
                                    "Extract `agent_id` from the previous agentcore_config() result and use it directly.\n"
                                    "Example: If result had agent_id='cagatay_test_8-JMYhdpEgu9', then:\n"
                                    "agentcore_invoke(agent_id='cagatay_test_8-JMYhdpEgu9', prompt='hello')"
                                },
                                {
                                    "text": "\n**Available agents in your account:**\n"
                                    + agents_list
                                },
                            ],
                        }
                except Exception as e:
                    # Debug: print why listing failed
                    import traceback

                    error_detail = traceback.format_exc()

                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "❌ No agent specified. Provide agent_id, agent_arn, or agent_name with valid config."
                        },
                        {
                            "text": "\n**💡 If you just launched an agent:**\n"
                            "Extract `agent_id` from the previous agentcore_config() result.\n"
                            "Example: agentcore_invoke(agent_id='agent-xyz123', prompt='hello')"
                        },
                        {
                            "text": "\n**To see all agents:** agentcore_agents(action='list')"
                        },
                    ],
                }

            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Get agent ARN from config
            if "agents" not in config or agent_name not in config["agents"]:
                available_agents = list(config.get("agents", {}).keys())
                if available_agents:
                    return {
                        "status": "error",
                        "content": [
                            {"text": f"Agent '{agent_name}' not found in config."},
                            {
                                "text": f"Available agents in config: {', '.join(available_agents)}"
                            },
                        ],
                    }
                else:
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": f"Agent '{agent_name}' not found in config (no agents configured)"
                            }
                        ],
                    }

            final_agent_arn = (
                config["agents"][agent_name]
                .get("bedrock_agentcore", {})
                .get("agent_arn")
            )

            if not final_agent_arn:
                # Agent in config but not deployed - try to list all agents to see if it exists elsewhere
                try:
                    import sys
                    from pathlib import Path

                    # Add tools directory to path if not already there
                    tools_dir = Path(__file__).parent
                    if str(tools_dir) not in sys.path:
                        sys.path.insert(0, str(tools_dir))

                    from agentcore_agents import agentcore_agents

                    agents_result = agentcore_agents(action="list", region=region)

                    if agents_result.get("status") == "success":
                        # Extract all text content from the result
                        agents_list = "\n".join(
                            item.get("text", "")
                            for item in agents_result.get("content", [])
                            if "text" in item
                        )
                        return {
                            "status": "error",
                            "content": [
                                {
                                    "text": f"❌ Agent '{agent_name}' in config but not deployed."
                                },
                                {
                                    "text": f"\n**💡 Deploy it:** agentcore_launch(agent_name='{agent_name}')"
                                },
                                {
                                    "text": "\n**Or invoke existing agents by ID:**\n"
                                    + agents_list
                                },
                            ],
                        }
                except Exception:
                    pass  # Fall back to simple error

                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"❌ Agent '{agent_name}' not deployed. Run agentcore_launch()."
                        }
                    ],
                }

        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Configure boto3 client
        boto_config = Config(
            read_timeout=900,
            connect_timeout=60,
            retries={"max_attempts": 3},
        )

        client = boto3.client(
            "bedrock-agentcore", region_name=region, config=boto_config
        )

        # Build payload with optional parameters
        payload_data = {"prompt": prompt, "mode": mode}

        if tools:
            payload_data["tools"] = tools
        if model:
            payload_data["model"] = model
        if system_prompt:
            payload_data["system_prompt"] = system_prompt

        payload_json = json.dumps(payload_data)

        # Invoke agent
        response = client.invoke_agent_runtime(
            agentRuntimeArn=final_agent_arn,
            qualifier="DEFAULT",
            runtimeSessionId=session_id,
            payload=payload_json,
        )

        # Process response
        events = []
        content_type = response.get("contentType", "")

        if "text/event-stream" in content_type:
            # Streaming response - process SSE events
            for chunk in response.get("response", []):
                # Decode bytes to string with error handling
                if isinstance(chunk, bytes):
                    try:
                        chunk = chunk.decode("utf-8")
                    except UnicodeDecodeError:
                        # Skip malformed chunks
                        continue

                # Split SSE stream by delimiter to get individual events
                if isinstance(chunk, str):
                    # Split by "\n\ndata: " to separate events
                    parts = chunk.split("\n\ndata: ")
                    # First part may have "data: " prefix
                    if parts and parts[0].startswith("data: "):
                        parts[0] = parts[0][6:]  # Remove "data: " prefix

                    # Process each event
                    for event_str in parts:
                        if not event_str.strip():
                            continue

                        try:
                            # Parse JSON event
                            event = json.loads(event_str)

                            # Stream to callback handler if available
                            if (
                                agent
                                and hasattr(agent, "callback_handler")
                                and agent.callback_handler
                            ):
                                if isinstance(event, dict):
                                    # Extract text for display from any event type
                                    text_to_display = None

                                    # Check if this is a wrapped AgentCore event
                                    if "event" in event and isinstance(
                                        event["event"], dict
                                    ):
                                        inner = event["event"]
                                        # Extract text from contentBlockDelta
                                        if "contentBlockDelta" in inner:
                                            text_to_display = (
                                                inner["contentBlockDelta"]
                                                .get("delta", {})
                                                .get("text", "")
                                            )
                                        # Pass the inner event
                                        if text_to_display:
                                            agent.callback_handler(
                                                data=text_to_display, **inner
                                            )
                                        else:
                                            agent.callback_handler(**inner)
                                    # Check if this is a local agent event with 'data' field
                                    elif "data" in event:
                                        text_to_display = event.get("data", "")
                                        # Copy event and remove only non-serializable object references
                                        filtered = event.copy()
                                        for key in [
                                            "agent",
                                            "event_loop_cycle_trace",
                                            "event_loop_cycle_span",
                                        ]:
                                            filtered.pop(key, None)
                                        agent.callback_handler(**filtered)
                                    else:
                                        # Pass other events as-is
                                        agent.callback_handler(**event)

                            # Collect for response
                            events.append(event)
                        except (json.JSONDecodeError, ValueError):
                            # Skip non-JSON content
                            continue
        else:
            # Non-streaming response
            for event in response.get("response", []):
                if isinstance(event, bytes):
                    try:
                        events.append(event.decode("utf-8"))
                    except UnicodeDecodeError:
                        events.append(str(event))
                else:
                    events.append(event)

        # Format response
        response_text = (
            "\n".join(str(e) for e in events) if events else "No response content"
        )
        print("\n")

        return {
            "status": "success",
            "content": [
                {"text": f"**Agent ARN:** {final_agent_arn}"},
                {"text": f"**Agent Response:**\n{response_text}"},
                {"text": f"**Session ID:** {session_id}"},
                {"text": f"**Mode:** {mode}"},
            ],
        }

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}
