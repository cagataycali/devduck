"""Messages management tool for agent self-modification.

This tool allows the agent to inspect, modify, export, import, and manage its own conversation history.
Operates on "turns" which are complete user-assistant interaction cycles.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from strands import tool


def _parse_messages_into_turns(messages: List[Dict]) -> List[Tuple[int, int]]:
    """
    Parse messages into turns.

    A turn is a complete interaction cycle:
    - User message (input)
    - Assistant message (may have toolUse)
    - User message (toolResult) [optional]
    - Assistant message (final response) [optional]

    Returns:
        List of tuples (start_index, end_index) for each turn
    """
    turns = []
    i = 0

    while i < len(messages):
        # A turn starts with a user message
        if messages[i]["role"] == "user":
            turn_start = i
            i += 1

            # Look for assistant response
            if i < len(messages) and messages[i]["role"] == "assistant":
                i += 1

                # Check if there's a toolResult (user) followed by final assistant
                if i < len(messages) and messages[i]["role"] == "user":
                    # Check if this is a toolResult
                    if any("toolResult" in block for block in messages[i]["content"]):
                        i += 1

                        # Look for final assistant message
                        if i < len(messages) and messages[i]["role"] == "assistant":
                            i += 1

            turn_end = i
            turns.append((turn_start, turn_end))
        else:
            # Orphaned assistant message (shouldn't happen, but handle it)
            i += 1

    return turns


@tool
def messages(
    action: str,
    file_path: Optional[str] = None,
    turn_indices: Optional[str] = None,
    start_turn: Optional[int] = None,
    end_turn: Optional[int] = None,
    filter_role: Optional[str] = None,
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Manage the agent's conversation history by TURNS (not individual messages).

    A TURN is a complete interaction cycle:
    - User input
    - Assistant response (possibly with toolUse)
    - User toolResult (if tool was used)
    - Assistant final response

    This tool provides complete control over conversation history:
    - Export/import conversations
    - Drop complete turns (maintains message validity)
    - Clear all messages
    - List and inspect turns

    Args:
        action: Action to perform - "list", "stats", "export", "import", "drop", "clear"
        file_path: Path to JSON file (for export/import actions)
        turn_indices: Comma-separated turn indices to drop (e.g., "0,2,5")
        start_turn: Start turn index for range operations (inclusive)
        end_turn: End turn index for range operations (exclusive)
        filter_role: Filter messages by role when listing ("user", "assistant")
        agent: Agent instance (auto-injected)

    Returns:
        Dict with status and content

    Examples:
        # List all turns
        messages(action="list")

        # List only user messages
        messages(action="list", filter_role="user")

        # Export conversation to file
        messages(action="export", file_path="/tmp/conversation.json")

        # Import conversation from file
        messages(action="import", file_path="/tmp/conversation.json")

        # Drop specific turns (complete cycles)
        messages(action="drop", turn_indices="0,2")  # Drops turn 0 and turn 2

        # Drop range of turns
        messages(action="drop", start_turn=0, end_turn=3)  # Drops turns 0,1,2

        # Clear all messages
        messages(action="clear")

        # Get statistics
        messages(action="stats")
    """
    try:
        # Import devduck here to avoid circular imports
        from devduck import devduck

        if not agent and hasattr(devduck, "agent") and devduck.agent:
            agent = devduck.agent

        if not agent:
            return {"status": "error", "content": [{"text": "Agent not initialized"}]}

        agent_messages = agent.messages

        if action == "list":
            if not agent_messages:
                return {
                    "status": "success",
                    "content": [{"text": "No messages in conversation history"}],
                }

            # Parse into turns
            turns = _parse_messages_into_turns(agent_messages)

            # Filter by role if specified
            if filter_role:
                filtered_messages = [
                    m for m in agent_messages if m["role"] == filter_role
                ]
                text_parts = [
                    f"Total {filter_role} messages: {len(filtered_messages)}\n"
                ]
                for i, msg in enumerate(filtered_messages):
                    role = msg["role"]
                    content_summary = _summarize_content(msg["content"])
                    text_parts.append(f"{i}. [{role}] {content_summary}")
            else:
                # Show by turns
                text_parts = [f"Total turns: {len(turns)}\n"]
                for turn_idx, (start, end) in enumerate(turns):
                    turn_messages = agent_messages[start:end]
                    text_parts.append(
                        f"\n--- Turn {turn_idx} (messages {start}-{end-1}) ---"
                    )
                    for msg_idx, msg in enumerate(turn_messages):
                        role = msg["role"]
                        content_summary = _summarize_content(msg["content"])
                        text_parts.append(
                            f"  {start + msg_idx}. [{role}] {content_summary}"
                        )

            return {"status": "success", "content": [{"text": "\n".join(text_parts)}]}

        elif action == "stats":
            if not agent_messages:
                return {
                    "status": "success",
                    "content": [{"text": "No messages in conversation"}],
                }

            turns = _parse_messages_into_turns(agent_messages)
            total = len(agent_messages)
            user_count = sum(1 for m in agent_messages if m["role"] == "user")
            assistant_count = sum(1 for m in agent_messages if m["role"] == "assistant")

            # Count content types
            text_blocks = 0
            tool_uses = 0
            tool_results = 0
            images = 0

            for msg in agent_messages:
                for block in msg["content"]:
                    if "text" in block:
                        text_blocks += 1
                    elif "toolUse" in block:
                        tool_uses += 1
                    elif "toolResult" in block:
                        tool_results += 1
                    elif "image" in block:
                        images += 1

            stats_text = f"""Conversation Statistics:
Total turns: {len(turns)}
Total messages: {total}
User messages: {user_count}
Assistant messages: {assistant_count}

Content blocks:
- Text: {text_blocks}
- Tool uses: {tool_uses}
- Tool results: {tool_results}
- Images: {images}
"""
            return {"status": "success", "content": [{"text": stats_text}]}

        elif action == "export":
            if not file_path:
                return {
                    "status": "error",
                    "content": [{"text": "'file_path' required for export action"}],
                }

            export_path = Path(file_path).expanduser()
            export_path.parent.mkdir(parents=True, exist_ok=True)

            # Export messages to JSON
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(agent_messages, f, indent=2, default=str)

            turns = _parse_messages_into_turns(agent_messages)
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"✅ Exported {len(turns)} turns ({len(agent_messages)} messages) to {export_path}"
                    }
                ],
            }

        elif action == "import":
            if not file_path:
                return {
                    "status": "error",
                    "content": [{"text": "'file_path' required for import action"}],
                }

            import_path = Path(file_path).expanduser()
            if not import_path.exists():
                return {
                    "status": "error",
                    "content": [{"text": f"File not found: {import_path}"}],
                }

            # Import messages from JSON
            with open(import_path, "r", encoding="utf-8") as f:
                imported_messages = json.load(f)

            # Validate message structure
            if not isinstance(imported_messages, list):
                return {
                    "status": "error",
                    "content": [{"text": "Invalid message format: expected list"}],
                }

            # Replace agent messages
            agent.messages.clear()
            agent.messages.extend(imported_messages)

            # ✅ Sync to agent.state so main loop detects the change
            agent.state.set("_messages_modified", True)
            agent.state.set("_messages_snapshot", list(imported_messages))

            turns = _parse_messages_into_turns(imported_messages)
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"✅ Imported {len(turns)} turns ({len(imported_messages)} messages) from {import_path}"
                    }
                ],
            }

        elif action == "drop":
            if not agent_messages:
                return {
                    "status": "success",
                    "content": [{"text": "No messages to drop"}],
                }

            # Parse messages into turns
            turns = _parse_messages_into_turns(agent_messages)

            if not turns:
                return {
                    "status": "success",
                    "content": [{"text": "No complete turns found"}],
                }

            turn_indices_to_drop = set()

            # Parse turn_indices parameter (e.g., "0,2,5")
            if turn_indices:
                for part in turn_indices.split(","):
                    part = part.strip()
                    try:
                        turn_indices_to_drop.add(int(part))
                    except ValueError:
                        return {
                            "status": "error",
                            "content": [{"text": f"Invalid turn index: {part}"}],
                        }

            # Parse range parameters
            if start_turn is not None and end_turn is not None:
                turn_indices_to_drop.update(range(start_turn, end_turn))
            elif start_turn is not None:
                turn_indices_to_drop.update(range(start_turn, len(turns)))
            elif end_turn is not None:
                turn_indices_to_drop.update(range(0, end_turn))

            if not turn_indices_to_drop:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "No turn indices specified. Use 'turn_indices' or 'start_turn'/'end_turn'"
                        }
                    ],
                }

            # Collect message indices to drop (all messages in specified turns)
            message_indices_to_drop = set()
            for turn_idx in turn_indices_to_drop:
                if 0 <= turn_idx < len(turns):
                    start, end = turns[turn_idx]
                    message_indices_to_drop.update(range(start, end))

            # Drop messages
            original_count = len(agent_messages)
            original_turn_count = len(turns)

            kept_messages = [
                msg
                for i, msg in enumerate(agent_messages)
                if i not in message_indices_to_drop
            ]

            agent.messages.clear()
            agent.messages.extend(kept_messages)

            # ✅ Sync to agent.state so main loop detects the change
            agent.state.set("_messages_modified", True)
            agent.state.set("_messages_snapshot", list(kept_messages))

            dropped_message_count = original_count - len(kept_messages)
            dropped_turn_count = len(turn_indices_to_drop)
            new_turns = _parse_messages_into_turns(kept_messages)

            return {
                "status": "success",
                "content": [
                    {
                        "text": f"✅ Dropped {dropped_turn_count} turns ({dropped_message_count} messages)\n"
                        f"Remaining: {len(new_turns)} turns ({len(kept_messages)} messages)"
                    }
                ],
            }

        elif action == "clear":
            count = len(agent_messages)
            turns = _parse_messages_into_turns(agent_messages)
            turn_count = len(turns)

            # Defer clearing until after current turn completes to avoid incomplete tool cycles
            agent.state.set("_clear_pending", True)
            agent.state.set("_clear_message_count", count)
            agent.state.set("_clear_turn_count", turn_count)

            return {
                "status": "success",
                "content": [
                    {"text": f"✅ Clear scheduled ({turn_count} turns, {count} messages) - will execute after turn completes"}
                ],
            }

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Valid: list, stats, export, import, drop, clear"
                    }
                ],
            }

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}


def _validate_message_integrity(messages: List[Dict]) -> List[Dict]:
    """
    Validate and fix message integrity after modifications.
    
    Ensures no incomplete tool cycles (toolUse without toolResult).
    If the last message contains toolUse blocks without matching toolResults,
    adds a synthetic toolResult message to maintain Bedrock API compliance.
    
    Args:
        messages: List of messages to validate
        
    Returns:
        Validated and fixed message list
    """
    if not messages:
        return messages
    
    # Check if last message has incomplete toolUse blocks
    last_msg = messages[-1]
    
    if last_msg["role"] == "assistant":
        # Find all toolUse blocks in last assistant message
        tool_use_ids = [
            content["toolUse"]["toolUseId"]
            for content in last_msg["content"]
            if "toolUse" in content
        ]
        
        if tool_use_ids:
            # Check if there's a following user message with toolResults
            # Since this is the last message, there are no toolResults
            # We need to add synthetic toolResult blocks
            
            synthetic_results = []
            for tool_use_id in tool_use_ids:
                synthetic_results.append({
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "status": "success",
                        "content": [{"text": "[Message history modified - synthetic result]"}]
                    }
                })
            
            # Add synthetic user message with toolResults
            messages.append({
                "role": "user",
                "content": synthetic_results
            })
    
    return messages


def _summarize_content(content: list) -> str:
    """Summarize content blocks for display."""
    summary_parts = []

    for block in content[:3]:  # Show first 3 blocks
        if "text" in block:
            text = block["text"][:100]  # First 100 chars
            if len(block["text"]) > 100:
                text += "..."
            summary_parts.append(f"text: {text}")
        elif "toolUse" in block:
            tool_name = block["toolUse"]["name"]
            summary_parts.append(f"toolUse: {tool_name}")
        elif "toolResult" in block:
            tool_id = block["toolResult"]["toolUseId"]
            summary_parts.append(f"toolResult: {tool_id}")
        elif "image" in block:
            summary_parts.append("image")

    if len(content) > 3:
        summary_parts.append(f"... +{len(content) - 3} more blocks")

    return " | ".join(summary_parts) if summary_parts else "empty"
