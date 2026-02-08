"""
Unified Agent Mesh — Single source of truth via file-based registry.

Before:  Each component maintained its own in-memory dict, cross-thread hacks everywhere.
After:   registry.json IS the mesh. Any process reads it. Done.

Ring context stays in-memory (high-frequency, real-time broadcast to browsers).
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from strands import tool

logger = logging.getLogger("devduck.tools.unified_mesh")

# ── In-memory state (ring context + WS handles for broadcast) ──
MESH_STATE: Dict[str, Any] = {
    "running": False,
    "ring_context": [],
    "ws_clients": {},  # ws_id → websocket (for real-time ring broadcast)
}

RING_MAX_ENTRIES = 100


# ============================================================================
# Peer Discovery — ALL reads go through registry
# ============================================================================


def get_all_peers(region: str = "us-west-2") -> List[Dict[str, Any]]:
    """Get unified list of ALL live peers: registry + AgentCore API.

    Registry gives us zenoh, browser, and locally-registered agents.
    AgentCore API gives us cloud-deployed agents (not in registry).
    We merge both, deduplicating by ID.
    """
    result = []
    seen_ids = set()

    # 1. Read from file registry (zenoh, browser, local agents)
    try:
        from devduck.tools.mesh_registry import registry

        agents = registry.get_all()
        for aid, info in agents.items():
            meta = info.get("metadata", {})
            seen_ids.add(aid)
            result.append(
                {
                    "id": aid,
                    "type": info.get("type", "unknown"),
                    "layer": meta.get("layer", "local"),
                    "name": meta.get("name") or meta.get("hostname") or aid,
                    "hostname": meta.get("hostname", ""),
                    "model": meta.get("model", ""),
                    "status": "active",
                    "is_self": meta.get("is_self", False),
                    "last_seen": round(info.get("last_seen", 0), 2),
                    "tools": meta.get("tools", []),
                    "tool_count": meta.get("tool_count", 0),
                    "system_prompt": meta.get("system_prompt", ""),
                    "cwd": meta.get("cwd", ""),
                    "platform": meta.get("platform", ""),
                    "description": meta.get("description", ""),
                    # Type-specific fields
                    "repo": meta.get("repo", ""),
                    "workflow": meta.get("workflow", ""),
                    "region": meta.get("region", ""),
                    "arn": meta.get("arn", ""),
                    "ws_id": meta.get("ws_id", ""),
                }
            )
    except Exception as e:
        logger.debug(f"Registry read error: {e}")

    # 2. Also fetch AgentCore agents from API (they're not in the file registry)
    try:
        import boto3

        client = boto3.client("bedrock-agentcore-control", region_name=region)
        response = client.list_agent_runtimes(maxResults=100)
        for agent in response.get("agentRuntimes", []):
            aid = agent.get("agentRuntimeId")
            if aid and aid not in seen_ids:
                status_raw = agent.get("status", "unknown")
                status = (
                    status_raw.lower() if isinstance(status_raw, str) else "unknown"
                )
                seen_ids.add(aid)
                result.append(
                    {
                        "id": aid,
                        "type": "agentcore",
                        "layer": "cloud",
                        "name": agent.get("agentRuntimeName", "unknown"),
                        "hostname": "",
                        "model": "",
                        "status": status,
                        "is_self": False,
                        "last_seen": time.time(),
                        "tools": [],
                        "tool_count": 0,
                        "system_prompt": "",
                        "cwd": "",
                        "platform": "",
                        "description": f"AgentCore ({region})",
                        "repo": "",
                        "workflow": "",
                        "region": region,
                        "arn": agent.get("agentRuntimeArn", ""),
                        "ws_id": "",
                    }
                )
    except Exception as e:
        logger.debug(f"AgentCore API error (non-fatal): {e}")

    return result


def get_peers_by_type(agent_type: str) -> List[Dict[str, Any]]:
    """Get live peers of a specific type."""
    return [p for p in get_all_peers() if p["type"] == agent_type]


def get_zenoh_peers() -> List[Dict[str, Any]]:
    return get_peers_by_type("zenoh")


def get_browser_peers() -> List[Dict[str, Any]]:
    return get_peers_by_type("browser")


def get_agentcore_agents(region: str = "us-west-2") -> List[Dict[str, Any]]:
    """Get AgentCore agents. Reads from registry first, falls back to API."""
    registry_agents = get_peers_by_type("agentcore")
    if registry_agents:
        return registry_agents
    # Fallback: live API call (for agents not yet in registry)
    try:
        import boto3

        client = boto3.client("bedrock-agentcore-control", region_name=region)
        response = client.list_agent_runtimes(maxResults=100)
        agents = []
        for agent in response.get("agentRuntimes", []):
            agents.append(
                {
                    "id": agent.get("agentRuntimeId"),
                    "type": "agentcore",
                    "layer": "cloud",
                    "name": agent.get("agentRuntimeName", "unknown"),
                    "arn": agent.get("agentRuntimeArn"),
                    "status": agent.get("status", "unknown").lower(),
                    "region": region,
                }
            )
        return agents
    except Exception as e:
        logger.debug(f"AgentCore API error: {e}")
        return []


def get_websocket_clients() -> List[Dict[str, Any]]:
    return get_peers_by_type("browser")


# ============================================================================
# Ring Context — In-memory + real-time broadcast
# ============================================================================


def add_to_ring(agent_id: str, agent_type: str, text: str, metadata: Dict = None):
    """Add entry to shared ring context."""
    entry = {
        "id": str(uuid.uuid4())[:8],
        "agent_id": agent_id,
        "agent_type": agent_type,
        "text": text[:2000],
        "timestamp": time.time(),
        "metadata": metadata or {},
    }
    MESH_STATE["ring_context"].append(entry)
    if len(MESH_STATE["ring_context"]) > RING_MAX_ENTRIES:
        MESH_STATE["ring_context"] = MESH_STATE["ring_context"][-RING_MAX_ENTRIES:]
    _broadcast_to_browsers({"type": "ring_update", "entry": entry})


def get_ring_context(max_entries: int = 20) -> List[Dict[str, Any]]:
    return MESH_STATE["ring_context"][-max_entries:]


def _broadcast_to_browsers(message: Dict):
    """Broadcast to all connected browser clients via any available event loop."""
    try:
        ws_clients = MESH_STATE.get("ws_clients", {})
        if not ws_clients:
            return
        msg_json = json.dumps(message)
        # Find an event loop
        loop = None
        try:
            from devduck.tools.agentcore_proxy import _GATEWAY_STATE

            if _GATEWAY_STATE.get("running") and _GATEWAY_STATE.get("loop"):
                loop = _GATEWAY_STATE["loop"]
        except ImportError:
            pass
        if not loop:
            return
        for ws_id, websocket in list(ws_clients.items()):
            try:
                asyncio.run_coroutine_threadsafe(websocket.send(msg_json), loop)
            except Exception as e:
                logger.debug(f"Broadcast to {ws_id} failed: {e}")
    except Exception as e:
        logger.debug(f"Broadcast error: {e}")


# ============================================================================
# Routing helpers
# ============================================================================


async def route_to_zenoh(peer_id: str, message: str, wait_time: float = 60.0):
    try:
        from devduck.tools.zenoh_peer import send_to_peer

        return send_to_peer(peer_id, message, wait_time)
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Zenoh error: {e}"}]}


async def route_to_agentcore(
    agent_id: str, prompt: str, region: str = "us-west-2", **kwargs
):
    try:
        import boto3
        from botocore.config import Config

        sts = boto3.client("sts", region_name=region)
        account_id = sts.get_caller_identity()["Account"]
        agent_arn = (
            f"arn:aws:bedrock-agentcore:{region}:{account_id}:runtime/{agent_id}"
        )
        session_id = kwargs.get("session_id") or str(uuid.uuid4())
        if len(session_id) < 33:
            session_id = (
                session_id + "-" + str(uuid.uuid4())[: 33 - len(session_id) - 1]
            )
        payload = {"prompt": prompt, "mode": "sync"}
        for k in ("system_prompt", "tools", "model"):
            if kwargs.get(k):
                payload[k] = kwargs[k]
        boto_config = Config(read_timeout=900, connect_timeout=60)
        client = boto3.client(
            "bedrock-agentcore", region_name=region, config=boto_config
        )
        response = client.invoke_agent_runtime(
            agentRuntimeArn=agent_arn,
            qualifier="DEFAULT",
            runtimeSessionId=session_id,
            payload=json.dumps(payload),
        )
        full = ""
        for chunk in response.get("response", []):
            if isinstance(chunk, bytes):
                full += chunk.decode("utf-8", errors="ignore")
            elif isinstance(chunk, str):
                full += chunk
        return {
            "status": "success",
            "content": [{"text": full}],
            "session_id": session_id,
        }
    except Exception as e:
        return {"status": "error", "content": [{"text": f"AgentCore error: {e}"}]}


# ============================================================================
# Main Tool
# ============================================================================


@tool
def unified_mesh(
    action: str,
    target: str = "",
    message: str = "",
    text: str = "",
    agent_id: str = "",
    agent_type: str = "local",
    max_entries: int = 20,
    region: str = "us-west-2",
) -> Dict[str, Any]:
    """Unified Agent Mesh - Bridge ALL agent types into a single network.

    Connects GitHub agents, browser agents, local DevDuck terminals (Zenoh), and AgentCore
    deployed agents into one unified mesh with shared ring context.

    Args:
        action: Action to perform:
            - "start": Start the mesh service
            - "stop": Stop the mesh service
            - "status": Get mesh status
            - "list_all": List all agents across all layers
            - "list_github": List GitHub agents only
            - "list_browser": List browser agents only
            - "list_zenoh": List Zenoh peers only
            - "list_agentcore": List AgentCore agents only
            - "route": Route message to specific agent
            - "broadcast_all": Broadcast to ALL agents
            - "get_ring": Get shared ring context
            - "add_ring": Add to shared ring context
            - "sync": Force sync with Zenoh peers
        target: Target agent ID for routing
        message: Message content for routing/broadcast
        text: Text content for ring entries
        agent_id: Agent ID for ring entries
        agent_type: Agent type for ring entries
        max_entries: Max ring entries to return
        region: AWS region for AgentCore

    Returns:
        Dict with status and results

    Examples:
        # Start the mesh
        unified_mesh(action="start")

        # List all agents
        unified_mesh(action="list_all")

        # Route to specific Zenoh peer
        unified_mesh(action="route", target="hostname-abc123", message="hello")

        # Broadcast to everyone
        unified_mesh(action="broadcast_all", message="sync now!")

        # Add to ring context
        unified_mesh(action="add_ring", agent_id="devduck-main", text="Working on file analysis...")

        # Get ring context
        unified_mesh(action="get_ring", max_entries=10)
    """
    try:
        if action == "start":
            MESH_STATE["running"] = True
            peers = get_all_peers(region)
            by_type = {}
            for p in peers:
                t = p.get("type", "unknown")
                by_type[t] = by_type.get(t, 0) + 1
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"✅ Unified Mesh active\nAgents: {len(peers)} — {by_type}\nRing: {len(MESH_STATE['ring_context'])} entries"
                    }
                ],
            }

        elif action == "stop":
            MESH_STATE["running"] = False
            return {"status": "success", "content": [{"text": "✅ Mesh stopped"}]}

        elif action == "status":
            peers = get_all_peers(region)
            by_type = {}
            for p in peers:
                by_type.setdefault(p["type"], []).append(p)
            lines = [f"🌐 Mesh Status — {len(peers)} agents"]
            for t, lst in sorted(by_type.items()):
                lines.append(f"  {t}: {len(lst)}")
                for p in lst[:5]:
                    lines.append(
                        f"    • {p['id']} ({p.get('name','')}) — {p.get('model','')}"
                    )
            lines.append(f"Ring: {len(MESH_STATE['ring_context'])} entries")
            # Show registry file path
            try:
                from devduck.tools.mesh_registry import REGISTRY_PATH

                lines.append(f"Registry: {REGISTRY_PATH}")
            except:
                pass
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "list_all":
            peers = get_all_peers(region)
            return {
                "status": "success",
                "content": [{"text": f"All agents ({len(peers)})"}],
                "peers": peers,
            }

        elif action == "list_zenoh":
            return {
                "status": "success",
                "peers": get_zenoh_peers(),
                "content": [{"text": f"Zenoh: {len(get_zenoh_peers())}"}],
            }
        elif action == "list_browser":
            return {
                "status": "success",
                "peers": get_browser_peers(),
                "content": [{"text": f"Browser: {len(get_browser_peers())}"}],
            }
        elif action == "list_agentcore":
            agents = get_agentcore_agents(region)
            return {
                "status": "success",
                "peers": agents,
                "content": [{"text": f"AgentCore: {len(agents)}"}],
            }
        elif action == "list_github":
            return {
                "status": "success",
                "peers": get_peers_by_type("github"),
                "content": [{"text": f"GitHub: {len(get_peers_by_type('github'))}"}],
            }

        elif action == "route":
            if not target or not message:
                return {
                    "status": "error",
                    "content": [{"text": "target and message required"}],
                }
            peers = {p["id"]: p for p in get_all_peers(region)}
            p = peers.get(target)
            if not p:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"'{target}' not found. Available: {list(peers.keys())[:10]}"
                        }
                    ],
                }
            if p["type"] == "zenoh" or p["type"] == "browser":
                import asyncio

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(route_to_zenoh(target, message))
                loop.close()
                add_to_ring(target, p["type"], f"Q: {message[:100]}")
                return result
            elif p["type"] == "agentcore":
                import asyncio

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    route_to_agentcore(target, message, region)
                )
                loop.close()
                add_to_ring(target, "agentcore", f"Q: {message[:100]}")
                return result
            return {
                "status": "error",
                "content": [{"text": f"Routing to {p['type']} not supported"}],
            }

        elif action == "broadcast_all":
            if not message:
                return {"status": "error", "content": [{"text": "No message"}]}
            results = []
            try:
                from devduck.tools.zenoh_peer import broadcast_message

                results.append(broadcast_message(message, wait_time=30.0))
            except Exception as e:
                results.append({"error": str(e)})
            add_to_ring("broadcast", "mesh", f"Broadcast: {message[:100]}")
            return {
                "status": "success",
                "content": [{"text": f"Broadcast sent"}],
                "results": results,
            }

        elif action == "get_ring":
            entries = get_ring_context(max_entries)
            return {
                "status": "success",
                "entries": entries,
                "content": [{"text": f"Ring: {len(entries)} entries"}],
            }

        elif action == "add_ring":
            if not text:
                return {"status": "error", "content": [{"text": "No text"}]}
            add_to_ring(agent_id or "unknown", agent_type, text)
            return {"status": "success", "content": [{"text": "Added to ring"}]}

        elif action == "sync":
            peers = get_all_peers(region)
            _broadcast_to_browsers(
                {
                    "type": "mesh_sync",
                    "peers": peers,
                    "ring": get_ring_context(20),
                    "timestamp": time.time(),
                }
            )
            return {
                "status": "success",
                "content": [{"text": f"Synced {len(peers)} agents"}],
            }

        else:
            return {
                "status": "error",
                "content": [{"text": f"Unknown action: {action}"}],
            }
    except Exception as e:
        logger.error(f"Mesh error: {e}")
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}
