"""AgentCore Configuration and Launch Tool"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any
from strands import tool


@tool
def agentcore_config(
    action: str,
    agent_name: str = "devduck",
    handler_path: str = None,
    region: str = "us-west-2",
    model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    max_tokens: int = 60000,
    tools: str = None,
    idle_timeout: int = 900,
    max_lifetime: int = 28800,
    auto_launch: bool = False,
) -> Dict[str, Any]:
    """
    Configure and launch DevDuck agents on Bedrock AgentCore.

    **Hardcoded for DevDuck:**
    - deployment_type: "direct_code_deploy" (fast, no Docker)
    - protocol: "HTTP" (standard API)
    - runtime: "PYTHON_3_13" (latest)

    **Quick Start:**
    ```python
    # Configure and launch in one step
    agentcore_config(
        action="configure",
        agent_name="my-agent",
        auto_launch=True
    )
    ```

    Args:
        action: Action to perform (generate, configure, launch, status)
        agent_name: Name for the agent (default: devduck)
            - Hyphens auto-converted to underscores
            - Example: "test-agent" → "test_agent"
        handler_path: Path to handler.py (default: auto-copy from devduck)
        region: AWS region (default: us-west-2)
        model_id: Bedrock model ID (default: Claude Sonnet 4.5)
        max_tokens: Max tokens to generate (default: 60000)
        tools: Tool configuration (package:tool1,tool2)
            Example: "strands_tools:shell,editor,file_read"
        idle_timeout: Idle timeout in seconds (default: 900)
        max_lifetime: Max lifetime in seconds (default: 28800)
        auto_launch: Auto launch after configure (default: False)

    Returns:
        Dict with status and instructions

    Examples:
        # Configure and launch
        agentcore_config(
            action="configure",
            agent_name="my-agent",
            tools="strands_tools:shell,editor,file_read",
            auto_launch=True
        )

        # Check status
        agentcore_config(action="status", agent_name="my-agent")

        # Generate commands only
        agentcore_config(action="generate", agent_name="my-agent")
    """

    # Hardcoded values for DevDuck deployment
    protocol = "HTTP"
    deployment_type = "direct_code_deploy"
    runtime = "PYTHON_3_13"

    # Validate agent name
    import re

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]{0,47}$", agent_name):
        # Auto-fix: replace hyphens with underscores
        fixed_name = agent_name.replace("-", "_")
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]{0,47}$", fixed_name):
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Invalid agent name: {agent_name}\nMust start with letter, contain only letters/numbers/underscores, 1-48 chars"
                    }
                ],
            }
        agent_name = fixed_name

    # Auto-determine handler path
    if not handler_path:
        try:
            import devduck

            devduck_dir = Path(devduck.__file__).parent
            source_handler = devduck_dir / "agentcore_handler.py"

            if source_handler.exists():
                # Copy to current directory with agent-specific name
                local_handler = Path.cwd() / "agentcore_handler.py"
                import shutil

                shutil.copy(str(source_handler), str(local_handler))
                handler_path = "./agentcore_handler.py"

                # Create requirements.txt with devduck if it doesn't exist
                requirements_path = Path.cwd() / "requirements.txt"
                if not requirements_path.exists():
                    with open(requirements_path, "w") as f:
                        f.write("devduck\n")
            else:
                return {
                    "status": "error",
                    "content": [{"text": f"Handler not found: {source_handler}"}],
                }
        except Exception as e:
            return {
                "status": "error",
                "content": [{"text": f"Failed to locate handler: {e}"}],
            }
    else:
        # Ensure handler_path is local
        handler_file = Path(handler_path)
        if not handler_file.is_absolute():
            # Already relative, make sure it starts with ./
            if not handler_path.startswith("./"):
                handler_path = f"./{handler_path}"
        else:
            # Absolute path - copy to current directory
            local_handler = Path.cwd() / handler_file.name
            import shutil

            shutil.copy(str(handler_file), str(local_handler))
            handler_path = f"./{handler_file.name}"

    if action == "generate":
        # Generate configuration commands
        commands = f"""
# DevDuck AgentCore Configuration for: {agent_name}

# Step 1: Configure
agentcore configure \\
  -e {handler_path} \\
  -n {agent_name} \\
  -r {region} \\
  -p {protocol} \\
  -dt {deployment_type} \\
  -rt {runtime} \\
  --idle-timeout {idle_timeout} \\
  --max-lifetime {max_lifetime}

# Step 2: Launch
agentcore launch -a {agent_name} --auto-update-on-conflict

# Step 3: Check Status
agentcore status -a {agent_name}

# Step 4: Invoke
agentcore invoke {agent_name} "What are your capabilities?"

# Python Invocation:
from devduck import devduck
devduck.agent.tool.agentcore_invoke(
    prompt="test query",
    agent_name="{agent_name}"
)

# Environment Variables for Handler:
export MODEL_PROVIDER=bedrock
export STRANDS_PROVIDER=bedrock
export STRANDS_MODEL_ID={model_id}
export STRANDS_MAX_TOKENS={max_tokens}
export AWS_REGION={region}
export DEVDUCK_AUTO_START_SERVERS=false
"""
        if tools:
            commands += f"export DEVDUCK_TOOLS={tools}\n"
        else:
            commands += "# export DEVDUCK_TOOLS=strands_tools:shell,editor,file_read  # Limit tools for sub-agents\n"

        return {
            "status": "success",
            "content": [{"text": f"Generated configuration:\n{commands}"}],
        }

    elif action == "configure":
        # Run agentcore configure command with automated inputs
        try:
            # Set environment variables for handler
            env = os.environ.copy()
            env["DEVDUCK_AUTO_START_SERVERS"] = "false"
            if tools:
                env["DEVDUCK_TOOLS"] = tools

            cmd = [
                "agentcore",
                "configure",
                "-e",
                handler_path,
                "-n",
                agent_name,
                "-r",
                region,
                "-p",
                protocol,
                "-dt",
                deployment_type,
                "-rt",
                runtime,
                "--idle-timeout",
                str(idle_timeout),
                "--max-lifetime",
                str(max_lifetime),
            ]

            # Automated inputs for interactive prompts:
            # 1. Requirements file path (Enter)
            # 2. Execution role (Enter for auto-create)
            # 3. S3 bucket (Enter for auto-create)
            # 4. OAuth config (no/Enter)
            # 5. Request headers (no/Enter)
            # 6. Memory selection (s to skip)
            automated_inputs = "\n\n\n\n\ns\n"

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            stdout, stderr = process.communicate(input=automated_inputs, timeout=300)

            if process.returncode != 0:
                return {
                    "status": "error",
                    "content": [{"text": f"Configure failed:\n{stderr}\n{stdout}"}],
                }

            output = f"✅ Configured: {agent_name}\n{stdout}"

            # Auto-launch if requested
            agent_arn = None
            agent_id = None

            if auto_launch:
                launch_cmd = [
                    "agentcore",
                    "launch",
                    "-a",
                    agent_name,
                    "--auto-update-on-conflict",
                ]

                output += f"\n\n🚀 Launching {agent_name}...\n"
                output += "=" * 50 + "\n"

                # Stream output to BOTH terminal AND agent
                import sys
                import re

                launch_process = subprocess.Popen(
                    launch_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                # Send automated inputs
                launch_inputs = "\n\n\n"
                launch_process.stdin.write(launch_inputs)
                launch_process.stdin.close()

                # Stream output line by line
                launch_output = []
                try:
                    for line in launch_process.stdout:
                        print(line, end="", flush=True)  # Terminal
                        launch_output.append(line)  # Agent

                    launch_process.wait(timeout=600)

                    full_output = "".join(launch_output)
                    if launch_process.returncode != 0:
                        output += f"\n❌ Launch failed with code {launch_process.returncode}\n{full_output}"
                    else:
                        output += f"\n✅ Launched: {agent_name}\n{full_output}"

                        # Extract ARN after streaming is complete
                        arn_match = re.search(
                            r"arn:aws:bedrock-agentcore:[^:]+:[^:]+:runtime/([^\s\n]+)",
                            full_output,
                        )
                        if arn_match:
                            agent_arn = arn_match.group(0)
                            agent_id = arn_match.group(1)
                except subprocess.TimeoutExpired:
                    launch_process.kill()
                    output += f"\n❌ Launch timed out"

            # Return structured response with agent_id for direct invocation
            result = {"status": "success", "content": [{"text": output}]}

            # Add structured metadata in content for easy access
            if agent_arn and agent_id:
                result["content"].append({"text": f"\n📋 **Agent ARN:** {agent_arn}"})
                result["content"].append({"text": f"🆔 **Agent ID:** {agent_id}"})
                result["content"].append(
                    {
                        "text": f"\n💡 **Invoke directly:** agentcore_invoke(agent_id='{agent_id}', prompt='your query')"
                    }
                )

            return result

        except FileNotFoundError:
            return {
                "status": "error",
                "content": [
                    {
                        "text": "agentcore CLI not found. Install: pip install bedrock-agentcore-starter-toolkit"
                    }
                ],
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "content": [{"text": f"Configuration timed out for {agent_name}"}],
            }
        except Exception as e:
            return {
                "status": "error",
                "content": [{"text": f"Configuration error: {e}"}],
            }

    elif action == "launch":
        # Run agentcore launch command with streaming output
        try:
            import sys
            import re

            cmd = ["agentcore", "launch", "-a", agent_name, "--auto-update-on-conflict"]

            print(f"🚀 Launching {agent_name}...")
            print("=" * 50)

            # Stream to BOTH terminal AND agent
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Capture output
            output_lines = []
            for line in process.stdout:
                print(line, end="", flush=True)  # Terminal
                output_lines.append(line)  # Agent

            process.wait()
            full_output = "".join(output_lines)

            # Extract ARN after streaming is complete
            agent_arn = None
            agent_id = None
            arn_match = re.search(
                r"arn:aws:bedrock-agentcore:[^:]+:[^:]+:runtime/([^\s\n]+)", full_output
            )
            if arn_match:
                agent_arn = arn_match.group(0)
                agent_id = arn_match.group(1)

            if process.returncode != 0:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"❌ Launch failed with code {process.returncode}\n{full_output}"
                        }
                    ],
                }

            result = {
                "status": "success",
                "content": [{"text": f"✅ Launched: {agent_name}\n{full_output}"}],
            }

            # Add structured metadata in content for easy access
            if agent_arn and agent_id:
                result["content"].append({"text": f"\n📋 **Agent ARN:** {agent_arn}"})
                result["content"].append({"text": f"🆔 **Agent ID:** {agent_id}"})
                result["content"].append(
                    {
                        "text": f"\n💡 **Invoke directly:** agentcore_invoke(agent_id='{agent_id}', prompt='your query')"
                    }
                )

            return result

        except Exception as e:
            return {"status": "error", "content": [{"text": f"Launch error: {e}"}]}

    elif action == "status":
        # Check agent status
        try:
            cmd = ["agentcore", "status", "-a", agent_name]
            result = subprocess.run(cmd, capture_output=True, text=True)

            return {
                "status": "success",
                "content": [{"text": f"Status for {agent_name}:\n{result.stdout}"}],
            }

        except Exception as e:
            return {
                "status": "error",
                "content": [{"text": f"Status check error: {e}"}],
            }

    else:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"Unknown action: {action}. Use: generate, configure, launch, status"
                }
            ],
        }
