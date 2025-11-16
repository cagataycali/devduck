"""DevDuck State Manager - Time-travel for agent conversations"""

import os
import tempfile
import dill
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from strands import tool

base_dir = Path(os.getenv("DEVDUCK_HOME", tempfile.gettempdir()))
states_dir = base_dir / ".devduck" / "states"
states_dir.mkdir(parents=True, exist_ok=True)


@tool
def state_manager(
    action: str,
    state_file: str = None,
    query: str = None,
    metadata: dict = None,
    agent=None,  # Parent agent injection
) -> Dict[str, Any]:
    """Agent state management with time-travel capabilities.

    Inspired by cagataycali/research-agent state export pattern.

    Actions:
    - export: Save current agent state to pkl
    - load: Load and display state from pkl
    - list: List available saved states
    - resume: Load state and continue with new query (ephemeral)
    - modify: Update pkl file metadata
    - delete: Remove saved state

    Args:
        action: Operation to perform
        state_file: Path to pkl file (auto-generated for export)
        query: New query for resume action
        metadata: Additional metadata for export/modify
        agent: Parent agent (auto-injected by Strands)

    Returns:
        Dict with status and content

    Examples:
        # Save current state
        state_manager(action="export", metadata={"note": "before refactor"})

        # List saved states
        state_manager(action="list")

        # Resume from previous state (ephemeral, no mutation)
        state_manager(action="resume", state_file="~/.devduck/states/devduck_20250116_032000.pkl", query="continue analysis")

        # Modify state metadata
        state_manager(action="modify", state_file="path/to/state.pkl", metadata={"tags": ["important", "refactor"]})
    """
    try:
        if action == "export":
            # Capture current agent state
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            state_file = states_dir / f"devduck_{timestamp}.pkl"

            # Safe state extraction (avoid complex objects)
            state_data = {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "system_prompt": agent.system_prompt,
                "tools": list(agent.tool_names),
                "model": {
                    "model_id": getattr(agent.model, "model_id", "unknown"),
                    "temperature": getattr(agent.model, "temperature", None),
                    "provider": getattr(agent.model, "provider", "unknown"),
                },
                "metadata": metadata or {},
                "environment": {
                    "cwd": str(Path.cwd()),
                    "devduck_version": "0.6.0",
                },
            }

            # Try to capture conversation history if available
            if hasattr(agent, "conversation_history"):
                state_data["conversation_history"] = agent.conversation_history
            elif hasattr(agent, "messages"):
                state_data["conversation_history"] = agent.messages

            # Save with dill
            with open(state_file, "wb") as f:
                dill.dump(state_data, f)

            size = state_file.stat().st_size
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"✅ State exported: {state_file}\n"
                        f"📦 Size: {size} bytes\n"
                        f"🔧 Tools: {len(state_data['tools'])}\n"
                        f"📝 Metadata: {metadata or 'none'}"
                    }
                ],
            }

        elif action == "list":
            # List all saved states
            states = sorted(
                states_dir.glob("devduck_*.pkl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            if not states:
                return {
                    "status": "success",
                    "content": [{"text": "No saved states found"}],
                }

            output = f"📚 Found {len(states)} saved states:\n\n"
            for i, state_path in enumerate(states[:10], 1):  # Show last 10
                try:
                    with open(state_path, "rb") as f:
                        state_data = dill.load(f)

                    timestamp = state_data.get("timestamp", "unknown")
                    tools_count = len(state_data.get("tools", []))
                    meta = state_data.get("metadata", {})

                    output += f"{i}. {state_path.name}\n"
                    output += f"   📅 {timestamp}\n"
                    output += f"   🔧 {tools_count} tools\n"
                    if meta:
                        output += f"   📝 {meta}\n"
                    output += "\n"
                except:
                    output += f"{i}. {state_path.name} (corrupted)\n\n"

            return {"status": "success", "content": [{"text": output}]}

        elif action == "load":
            # Load and display state
            if not state_file:
                return {
                    "status": "error",
                    "content": [{"text": "state_file required for load"}],
                }

            state_path = Path(state_file).expanduser()
            if not state_path.exists():
                return {
                    "status": "error",
                    "content": [{"text": f"State file not found: {state_path}"}],
                }

            with open(state_path, "rb") as f:
                state_data = dill.load(f)

            # Pretty format
            output = f"📦 State: {state_path.name}\n\n"
            output += f"📅 Timestamp: {state_data.get('timestamp')}\n"
            output += f"🤖 Model: {state_data.get('model', {}).get('model_id')}\n"
            output += f"🔧 Tools ({len(state_data.get('tools', []))}): {', '.join(state_data.get('tools', []))}\n"
            output += f"📝 Metadata: {state_data.get('metadata', {})}\n"

            if "conversation_history" in state_data:
                history = state_data["conversation_history"]
                output += f"\n💬 Conversation: {len(history)} messages\n"

            return {"status": "success", "content": [{"text": output}]}

        elif action == "resume":
            # Time-travel: Load state and continue with ephemeral agent
            if not state_file or not query:
                return {
                    "status": "error",
                    "content": [{"text": "state_file and query required for resume"}],
                }

            state_path = Path(state_file).expanduser()
            if not state_path.exists():
                return {
                    "status": "error",
                    "content": [{"text": f"State file not found: {state_path}"}],
                }

            with open(state_path, "rb") as f:
                state_data = dill.load(f)

            # ✅ Create ephemeral DevDuck instance (no mutation!)
            try:
                from devduck import DevDuck

                ephemeral_duck = DevDuck(auto_start_servers=False)
                ephemeral_agent = ephemeral_duck.agent
            except Exception as e:
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to create ephemeral DevDuck: {e}"}],
                }

            # Load saved state into ephemeral agent
            ephemeral_agent.system_prompt = state_data["system_prompt"]

            # Restore conversation history
            if "conversation_history" in state_data:
                saved_history = state_data["conversation_history"]

                if hasattr(ephemeral_agent, "conversation_history"):
                    ephemeral_agent.conversation_history = saved_history
                elif hasattr(ephemeral_agent, "messages"):
                    ephemeral_agent.messages = saved_history

            # Build continuation prompt with context
            continuation_context = f"""
[Resumed from state: {state_path.name}]
[Original timestamp: {state_data.get('timestamp')}]

{query}
"""
            # Run ephemeral agent (parent agent unchanged!)
            result = ephemeral_agent(continuation_context)

            return {
                "status": "success",
                "content": [{"text": f"🔄 Resumed from {state_path.name}\n\n{result}"}],
            }

        elif action == "modify":
            # Modify state metadata
            if not state_file:
                return {
                    "status": "error",
                    "content": [{"text": "state_file required for modify"}],
                }

            state_path = Path(state_file).expanduser()
            if not state_path.exists():
                return {
                    "status": "error",
                    "content": [{"text": f"State file not found: {state_path}"}],
                }

            with open(state_path, "rb") as f:
                state_data = dill.load(f)

            # Update metadata
            if metadata:
                state_data["metadata"].update(metadata)

            # Save back
            with open(state_path, "wb") as f:
                dill.dump(state_data, f)

            return {
                "status": "success",
                "content": [
                    {
                        "text": f"✅ Modified {state_path.name}\n📝 New metadata: {state_data['metadata']}"
                    }
                ],
            }

        elif action == "delete":
            # Delete saved state
            if not state_file:
                return {
                    "status": "error",
                    "content": [{"text": "state_file required for delete"}],
                }

            state_path = Path(state_file).expanduser()
            if not state_path.exists():
                return {
                    "status": "error",
                    "content": [{"text": f"State file not found: {state_path}"}],
                }

            state_path.unlink()
            return {
                "status": "success",
                "content": [{"text": f"🗑️  Deleted {state_path.name}"}],
            }

        else:
            return {
                "status": "error",
                "content": [{"text": f"Unknown action: {action}"}],
            }

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}
