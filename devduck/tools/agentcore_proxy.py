"""
AgentCore Proxy - Universal Agent Relay for Unified Mesh.

This tool provides a WebSocket server that exposes ALL agent types:
- GitHub Actions agents (from configured repos)
- AgentCore deployed agents (AWS cloud)
- Zenoh peers (local DevDuck terminals)
- Browser agents (via one.html — treated as first-class mesh peers)

Auto-starts on port 10000 (well-known relay port) for browser auto-discovery.

KEY ARCHITECTURE:
- Browser connects via WS → gets a ws_id
- Browser can register as a "browser peer" with register_browser_peer
- Zenoh peers see browser peers and can send messages to them
- Ring context is shared: CLI writes → browser gets ring_update, browser writes → CLI gets it via unified_mesh
- Browser agents appear in zenoh peer list as type "browser"
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
from strands import tool

logger = logging.getLogger("devduck.tools.agentcore_proxy")

# ============================================================================
# Global State
# ============================================================================

_GATEWAY_STATE: Dict[str, Any] = {
    "server": None,
    "port": None,
    "running": False,
    "region": "us-west-2",
    "github_repos": [],
    "github_token": None,
    "loop": None,
    # Browser peers: browser tabs that registered as mesh participants
    # {ws_id: {name, model, tools, system_prompt, registered_at, ws}}
    "browser_peers": {},
}


# ============================================================================
# GitHub Agent Discovery
# ============================================================================


def get_github_agents(
    repos: List[str] = None, token: str = None
) -> List[Dict[str, Any]]:
    """Discover GitHub Actions agents from configured repositories."""
    token = token or os.getenv("GITHUB_TOKEN") or os.getenv("PAT_TOKEN")
    if not token:
        return []

    repos = repos or _GATEWAY_STATE.get("github_repos", [])
    if not repos:
        repos = os.getenv("GITHUB_AGENT_REPOS", "").split(",")
        repos = [r.strip() for r in repos if r.strip()]

    agents = []
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    for repo in repos:
        try:
            url = f"https://api.github.com/repos/{repo}/contents/.github/workflows"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
            workflows = resp.json()
            workflow_names = [w.get("name", "") for w in workflows]
            agent_workflows = [
                w
                for w in workflow_names
                if "agent" in w.lower() or w in ["agent.yml", "agent.yaml"]
            ]
            if not agent_workflows:
                continue
            repo_url = f"https://api.github.com/repos/{repo}"
            repo_resp = requests.get(repo_url, headers=headers, timeout=10)
            repo_info = repo_resp.json() if repo_resp.status_code == 200 else {}
            for workflow in agent_workflows:
                runs_url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs"
                runs_resp = requests.get(
                    runs_url, headers=headers, params={"per_page": 5}, timeout=10
                )
                runs = (
                    runs_resp.json().get("workflow_runs", [])
                    if runs_resp.status_code == 200
                    else []
                )
                status = "ready"
                last_run = None
                if runs:
                    last_run = runs[0]
                    if last_run.get("status") == "in_progress":
                        status = "running"
                    elif last_run.get("conclusion") == "failure":
                        status = "failed"
                agents.append(
                    {
                        "id": f"github:{repo}:{workflow}",
                        "type": "github",
                        "layer": "github",
                        "name": repo_info.get("name", repo.split("/")[-1]),
                        "repo": repo,
                        "workflow": workflow,
                        "description": repo_info.get("description", ""),
                        "status": status,
                        "url": f"https://github.com/{repo}/actions/workflows/{workflow}",
                        "last_run": (
                            {
                                "id": last_run.get("id"),
                                "status": last_run.get("status"),
                                "conclusion": last_run.get("conclusion"),
                                "created_at": last_run.get("created_at"),
                            }
                            if last_run
                            else None
                        ),
                        "stars": repo_info.get("stargazers_count", 0),
                        "owner": repo.split("/")[0],
                    }
                )
        except Exception as e:
            logger.debug(f"Error scanning {repo}: {e}")
            continue
    return agents


def trigger_github_agent(
    repo: str,
    workflow: str,
    prompt: str,
    token: str = None,
    system_prompt: str = None,
    tools: str = None,
    model: str = None,
    max_tokens: int = None,
) -> Dict[str, Any]:
    """Trigger a GitHub Actions agent workflow."""
    token = token or os.getenv("GITHUB_TOKEN") or os.getenv("PAT_TOKEN")
    if not token:
        return {"status": "error", "error": "No GitHub token"}
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    inputs = {"prompt": prompt}
    if system_prompt:
        inputs["system_prompt"] = system_prompt
    if tools:
        inputs["tools"] = tools
    if model:
        inputs["model"] = model
    if max_tokens:
        inputs["max_tokens"] = str(max_tokens)
    data = {"ref": "main", "inputs": inputs}
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        if resp.status_code == 204:
            return {
                "status": "success",
                "message": f"Workflow {workflow} triggered on {repo}",
                "url": f"https://github.com/{repo}/actions/workflows/{workflow}",
            }
        else:
            return {
                "status": "error",
                "error": f"Failed: {resp.status_code} - {resp.text}",
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ============================================================================
# Browser Peer Management
# ============================================================================


def get_browser_peers() -> List[Dict[str, Any]]:
    """Get all registered browser peers as mesh-compatible agent dicts."""
    peers = []
    for ws_id, info in _GATEWAY_STATE.get("browser_peers", {}).items():
        peers.append(
            {
                "id": f"browser:{ws_id}",
                "type": "browser",
                "layer": "browser",
                "name": info.get("name", f"browser-{ws_id}"),
                "hostname": "browser",
                "model": info.get("model", "browser-agent"),
                "status": "active",
                "tools": info.get("tools", []),
                "tool_count": info.get("tool_count", 0),
                "system_prompt": info.get("system_prompt", ""),
                "registered_at": info.get("registered_at", 0),
                "ws_id": ws_id,
            }
        )
    return peers


# ============================================================================
# WebSocket Gateway
# ============================================================================


class MeshGateway:
    """WebSocket gateway exposing unified mesh to browsers with auto-discovery.

    Browser peers are first-class citizens: they appear in zenoh peer lists,
    can be targeted by zenoh nodes, and share ring context bidirectionally.
    """

    def __init__(self, port: int = 8765, region: str = "us-west-2"):
        self.port = port
        self.region = region
        self.server = None
        self._running = False
        self.clients: Dict[str, Any] = {}  # ws_id -> websocket

    async def handle_client(self, websocket):
        """Handle WebSocket client connection."""
        ws_id = str(uuid.uuid4())[:8]
        self.clients[ws_id] = websocket

        # Register with unified_mesh
        try:
            from devduck.tools.unified_mesh import MESH_STATE

            MESH_STATE["ws_clients"][ws_id] = websocket
        except ImportError:
            pass

        logger.info(f"Client connected: {ws_id}")

        # Background task to push Zenoh peer updates to this client
        zenoh_push_task = asyncio.ensure_future(self._push_zenoh_updates(websocket))

        try:
            # Send welcome with FULL mesh state (all agent types)
            all_agents = await self._get_all_agents()
            ring = self._get_ring_context(50)

            await websocket.send(
                json.dumps(
                    {
                        "type": "connected",
                        "ws_id": ws_id,
                        "agents": all_agents,
                        "ring_context": ring,
                        "timestamp": time.time(),
                        "capabilities": {
                            "github": bool(
                                os.getenv("GITHUB_TOKEN") or os.getenv("PAT_TOKEN")
                            ),
                            "agentcore": True,
                            "zenoh": True,
                            "browser_peer": True,  # Browser can register as peer
                        },
                    }
                )
            )

            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(ws_id, websocket, data)
                except json.JSONDecodeError:
                    # Plain text message — treat as a prompt to the local agent
                    if message.strip():
                        await self._handle_message(
                            ws_id,
                            websocket,
                            {
                                "type": "invoke",
                                "agent_id": "local:devduck",
                                "agent_type": "local",
                                "prompt": message.strip(),
                            },
                        )
                except Exception as e:
                    logger.error(f"Message error: {e}")
                    await websocket.send(json.dumps({"type": "error", "error": str(e)}))

        except Exception as e:
            logger.debug(f"WebSocket error: {e}")
        finally:
            zenoh_push_task.cancel()
            if ws_id in self.clients:
                del self.clients[ws_id]
            # Unregister browser peer if it was registered
            if ws_id in _GATEWAY_STATE.get("browser_peers", {}):
                del _GATEWAY_STATE["browser_peers"][ws_id]
                logger.info(f"Browser peer unregistered: {ws_id}")
                # Remove from file registry
                try:
                    from devduck.tools.mesh_registry import registry

                    registry.unregister(f"browser:{ws_id}")
                except Exception:
                    pass
                self._notify_browser_peers_changed()
            try:
                from devduck.tools.unified_mesh import MESH_STATE

                if ws_id in MESH_STATE["ws_clients"]:
                    del MESH_STATE["ws_clients"][ws_id]
            except:
                pass
            logger.info(f"Client disconnected: {ws_id}")

    def _notify_browser_peers_changed(self):
        """Push updated browser peer list to all zenoh peers via presence."""
        # Push to all connected WS clients so they update their sidebar
        try:
            loop = _GATEWAY_STATE.get("loop")
            if not loop:
                return
            browser_peers = get_browser_peers()
            msg = json.dumps(
                {
                    "type": "browser_peers_update",
                    "peers": browser_peers,
                    "timestamp": time.time(),
                }
            )
            for ws_id, ws in list(self.clients.items()):
                try:
                    asyncio.run_coroutine_threadsafe(ws.send(msg), loop)
                except:
                    pass
        except Exception as e:
            logger.debug(f"Browser peer notification error: {e}")

    async def _push_zenoh_updates(self, websocket):
        """Registry poller — reads /tmp/devduck/mesh_registry.json every 3s.

        Replaces the old sys.modules hack. The registry file IS the truth.
        Zenoh heartbeat writes to it, proxy reads from it, browser sees everything.
        """
        last_hash = ""
        await asyncio.sleep(1)

        while True:
            try:
                await asyncio.sleep(3)

                try:
                    from devduck.tools.mesh_registry import registry

                    # Heartbeat browser peers in the file registry so they don't expire
                    for bp_ws_id in list(
                        _GATEWAY_STATE.get("browser_peers", {}).keys()
                    ):
                        try:
                            registry.heartbeat(f"browser:{bp_ws_id}")
                        except Exception:
                            pass

                    agents = registry.get_all()

                    # Build hash to detect changes
                    current_hash = f"{len(agents)}:{','.join(sorted(agents.keys()))}"

                    if current_hash != last_hash:
                        last_hash = current_hash

                        # Build peer list for browser
                        peer_list = []
                        my_instance_id = None

                        for aid, info in agents.items():
                            meta = info.get("metadata", {})
                            is_zenoh = info.get("type") == "zenoh"
                            is_self = meta.get("is_self", False)

                            if is_self and is_zenoh:
                                my_instance_id = aid

                            peer_list.append(
                                {
                                    "id": aid,
                                    "type": info.get("type", "unknown"),
                                    "hostname": meta.get("hostname", ""),
                                    "model": meta.get("model", ""),
                                    "last_seen": info.get("last_seen", 0),
                                    "is_self": is_self,
                                    "tools": meta.get("tools", []),
                                    "tool_count": meta.get("tool_count", 0),
                                    "system_prompt": meta.get("system_prompt", ""),
                                    "cwd": meta.get("cwd", ""),
                                    "platform": meta.get("platform", ""),
                                    "name": meta.get("name")
                                    or meta.get("hostname")
                                    or aid,
                                    "description": meta.get("description", ""),
                                    "status": "active",
                                }
                            )

                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "zenoh_peers_update",
                                    "instance_id": my_instance_id,
                                    "peers": peer_list,
                                    "timestamp": time.time(),
                                }
                            )
                        )
                        logger.debug(f"Proxy: Registry push: {len(peer_list)} agents")

                except Exception as e:
                    logger.debug(f"Registry poll error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Registry watch error: {e}")
                await asyncio.sleep(5)

    async def _get_all_agents(self) -> List[Dict[str, Any]]:
        """Get all agents. Registry is the source of truth — one read, done."""
        try:
            from devduck.tools.unified_mesh import get_all_peers

            return get_all_peers(self.region)
        except Exception as e:
            logger.debug(f"Registry read error: {e}")
            return []

    def _get_ring_context(self, max_entries: int = 20) -> List[Dict]:
        try:
            from devduck.tools.unified_mesh import get_ring_context

            return get_ring_context(max_entries)
        except:
            return []

    async def _handle_message(self, ws_id: str, ws, data: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        msg_type = data.get("type", "")

        if msg_type == "list_agents" or msg_type == "list_peers":
            agents = await self._get_all_agents()
            await ws.send(
                json.dumps(
                    {
                        "type": "agents_list",
                        "agents": agents,
                        "count": len(agents),
                        "by_type": {
                            "github": len([a for a in agents if a["type"] == "github"]),
                            "agentcore": len(
                                [a for a in agents if a["type"] == "agentcore"]
                            ),
                            "zenoh": len([a for a in agents if a["type"] == "zenoh"]),
                            "browser": len(
                                [a for a in agents if a["type"] == "browser"]
                            ),
                        },
                    }
                )
            )

        elif msg_type == "register_browser_peer":
            # Browser registers as a first-class mesh peer → writes to REGISTRY FILE
            peer_id = f"browser:{ws_id}"
            peer_meta = {
                "name": data.get("name", f"browser-{ws_id}"),
                "model": data.get("model", "browser-agent"),
                "tools": data.get("tools", []),
                "tool_count": data.get("tool_count", 0),
                "system_prompt": data.get("system_prompt", ""),
                "hostname": "browser",
                "layer": "browser",
                "ws_id": ws_id,
                "is_self": False,
            }
            # Write to in-memory (for WS routing)
            _GATEWAY_STATE["browser_peers"][ws_id] = {
                **peer_meta,
                "registered_at": time.time(),
                "ws": ws,
            }
            # Write to file registry (visible to ALL processes)
            try:
                from devduck.tools.mesh_registry import registry

                registry.register(peer_id, "browser", peer_meta)
            except Exception as e:
                logger.warning(f"Registry write failed for browser peer: {e}")

            logger.info(f"Browser peer registered: {ws_id} as '{peer_meta['name']}'")
            try:
                from devduck.tools.unified_mesh import add_to_ring

                add_to_ring(
                    peer_id,
                    "browser",
                    f"Browser peer '{peer_meta['name']}' joined the mesh",
                    {"source": "browser_register"},
                )
            except:
                pass
            self._notify_browser_peers_changed()
            await ws.send(
                json.dumps(
                    {
                        "type": "browser_peer_registered",
                        "ws_id": ws_id,
                        "peer_id": peer_id,
                        "success": True,
                    }
                )
            )

        elif msg_type == "update_browser_peer":
            # Update browser peer metadata (e.g., after agent change)
            if ws_id in _GATEWAY_STATE.get("browser_peers", {}):
                peer = _GATEWAY_STATE["browser_peers"][ws_id]
                if "name" in data:
                    peer["name"] = data["name"]
                if "model" in data:
                    peer["model"] = data["model"]
                if "tools" in data:
                    peer["tools"] = data["tools"]
                if "tool_count" in data:
                    peer["tool_count"] = data["tool_count"]
                if "system_prompt" in data:
                    peer["system_prompt"] = data["system_prompt"]
                self._notify_browser_peers_changed()
                await ws.send(
                    json.dumps({"type": "browser_peer_updated", "success": True})
                )
            else:
                await ws.send(
                    json.dumps(
                        {"type": "error", "error": "Not registered as browser peer"}
                    )
                )

        elif msg_type == "browser_invoke_response":
            # Browser agent completed a task sent by zenoh/other peer
            # Forward the response back to the requester
            turn_id = data.get("turn_id")
            response_text = data.get("response", "")
            requester_ws_id = data.get("requester_ws_id")

            # If requester is a WS client, forward directly
            if requester_ws_id and requester_ws_id in self.clients:
                target_ws = self.clients[requester_ws_id]
                await target_ws.send(
                    json.dumps(
                        {
                            "type": "turn_end",
                            "turn_id": turn_id,
                            "agent_id": f"browser:{ws_id}",
                            "agent_type": "browser",
                            "response": response_text,
                            "timestamp": time.time(),
                        }
                    )
                )

            # Also publish to zenoh response channel if the requester was a zenoh peer
            zenoh_requester = data.get("zenoh_requester_id")
            if zenoh_requester and turn_id:
                try:
                    from devduck.tools.zenoh_peer import publish_message

                    publish_message(
                        f"devduck/response/{zenoh_requester}/{turn_id}",
                        {
                            "type": "turn_end",
                            "responder_id": f"browser:{ws_id}",
                            "turn_id": turn_id,
                            "result": response_text,
                            "chunks_sent": 0,
                            "timestamp": time.time(),
                        },
                    )
                except Exception as e:
                    logger.debug(f"Failed to publish browser response to zenoh: {e}")

            # Add to ring
            try:
                from devduck.tools.unified_mesh import add_to_ring

                add_to_ring(
                    f"browser:{ws_id}",
                    "browser",
                    f"A: {response_text[:150]}",
                    {"source": "browser_response"},
                )
            except:
                pass

        elif msg_type == "browser_stream_chunk":
            # Browser agent streaming chunks — forward to zenoh requester
            turn_id = data.get("turn_id")
            zenoh_requester = data.get("zenoh_requester_id")
            chunk_text = data.get("data", "")
            if zenoh_requester and turn_id and chunk_text:
                try:
                    from devduck.tools.zenoh_peer import publish_message

                    publish_message(
                        f"devduck/response/{zenoh_requester}/{turn_id}",
                        {
                            "type": "stream",
                            "chunk_type": "text",
                            "responder_id": f"browser:{ws_id}",
                            "turn_id": turn_id,
                            "data": chunk_text,
                            "timestamp": time.time(),
                        },
                    )
                except:
                    pass

        elif msg_type == "list_github":
            token = data.get("token") or os.getenv("GITHUB_TOKEN")
            repos = data.get("repos", [])
            agents = get_github_agents(repos, token)
            await ws.send(json.dumps({"type": "github_list", "agents": agents}))

        elif msg_type == "list_agentcore":
            try:
                from devduck.tools.unified_mesh import get_agentcore_agents

                agents = get_agentcore_agents(self.region)
                await ws.send(json.dumps({"type": "agentcore_list", "agents": agents}))
            except Exception as e:
                await ws.send(json.dumps({"type": "error", "error": str(e)}))

        elif msg_type == "list_zenoh":
            try:
                from devduck.tools.unified_mesh import get_zenoh_peers

                peers = get_zenoh_peers()
                await ws.send(json.dumps({"type": "zenoh_list", "peers": peers}))
            except Exception as e:
                await ws.send(json.dumps({"type": "error", "error": str(e)}))

        elif msg_type == "configure":
            if "github_token" in data:
                _GATEWAY_STATE["github_token"] = data["github_token"]
            if "github_repos" in data:
                _GATEWAY_STATE["github_repos"] = data["github_repos"]
            if "region" in data:
                self.region = data["region"]
                _GATEWAY_STATE["region"] = data["region"]
            await ws.send(
                json.dumps(
                    {
                        "type": "configured",
                        "success": True,
                        "repos": _GATEWAY_STATE.get("github_repos", []),
                        "region": self.region,
                    }
                )
            )

        elif msg_type == "invoke":
            asyncio.create_task(self._handle_invoke(ws_id, ws, data))

        elif msg_type == "trigger_github":
            asyncio.create_task(self._handle_github_trigger(ws, data))

        elif msg_type == "poll_github_logs":
            asyncio.create_task(self._poll_github_run_logs(ws, data))

        elif msg_type == "broadcast":
            asyncio.create_task(self._handle_broadcast(ws, data))

        elif msg_type == "get_ring":
            entries = self._get_ring_context(data.get("max_entries", 50))
            await ws.send(json.dumps({"type": "ring_context", "entries": entries}))

        elif msg_type == "add_ring":
            try:
                from devduck.tools.unified_mesh import add_to_ring

                add_to_ring(
                    data.get("agent_id", f"browser:{ws_id}"),
                    data.get("agent_type", "browser"),
                    data.get("text", ""),
                    data.get("metadata", {"source": "browser"}),
                )
                await ws.send(json.dumps({"type": "ring_added", "success": True}))
            except Exception as e:
                await ws.send(json.dumps({"type": "error", "error": str(e)}))

        elif msg_type == "ping":
            await ws.send(json.dumps({"type": "pong", "timestamp": time.time()}))

        else:
            await ws.send(
                json.dumps({"type": "error", "error": f"Unknown type: {msg_type}"})
            )

    def _build_ring_block(self, max_entries: int = 15) -> str:
        """Build a human-readable ring context block."""
        ring_entries = self._get_ring_context(max_entries)
        if not ring_entries:
            return ""
        from datetime import datetime as _dt

        lines = ["## 🔗 Mesh Ring Context (recent agent activity):"]
        for entry in ring_entries:
            aid = entry.get("agent_id", "?")
            atype = entry.get("agent_type", "?")
            txt = entry.get("text", "")[:300]
            ts = entry.get("timestamp", 0)
            time_str = _dt.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "?"
            lines.append(f"- [{time_str}] {aid} ({atype}): {txt}")
        return "\n".join(lines)

    def _enrich_prompt_with_ring(self, prompt: str) -> str:
        """Prepend ring context to a prompt string."""
        ring_block = self._build_ring_block()
        if not ring_block:
            return prompt
        return f"[Ring Context]\n{ring_block}\n\n[User Query]\n{prompt}"

    async def _handle_invoke(self, sender_ws_id: str, ws, data: Dict[str, Any]):
        """Handle invoke - route to specific agent with streaming."""
        agent_id = data.get("agent_id") or data.get("peer_id")
        agent_type = data.get("agent_type") or data.get("peer_type")
        prompt = data.get("prompt", data.get("message", ""))

        if not agent_id or not prompt:
            await ws.send(
                json.dumps({"type": "error", "error": "agent_id and prompt required"})
            )
            return

        turn_id = data.get("turn_id") or str(uuid.uuid4())[:8]

        await ws.send(
            json.dumps(
                {
                    "type": "turn_start",
                    "turn_id": turn_id,
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "timestamp": time.time(),
                }
            )
        )

        # Auto-detect agent type
        if not agent_type:
            if agent_id.startswith("github:"):
                agent_type = "github"
            elif agent_id.startswith("local:"):
                agent_type = "local"
            elif agent_id.startswith("browser:"):
                agent_type = "browser"
            elif agent_id in self.clients:
                agent_type = "browser"
            else:
                is_zenoh = False
                try:
                    from devduck.tools.zenoh_peer import ZENOH_STATE

                    if agent_id in ZENOH_STATE.get("peers", {}):
                        is_zenoh = True
                except ImportError:
                    pass
                if is_zenoh:
                    agent_type = "zenoh"
                elif "-" in agent_id and len(agent_id.split("-")[0]) > 5:
                    agent_type = "agentcore"
                else:
                    agent_type = "zenoh"

        # Route based on type
        if agent_type == "github":
            parts = agent_id.split(":")
            if len(parts) >= 3:
                enriched_prompt = self._enrich_prompt_with_ring(prompt)
                result = trigger_github_agent(
                    repo=parts[1],
                    workflow=parts[2],
                    prompt=enriched_prompt,
                    system_prompt=data.get("system_prompt"),
                    tools=data.get("tools"),
                    model=data.get("model"),
                )
                await ws.send(
                    json.dumps(
                        {
                            "type": "turn_end",
                            "turn_id": turn_id,
                            "agent_id": agent_id,
                            "agent_type": "github",
                            "response": result.get("message", result.get("error", "")),
                            "url": result.get("url"),
                            "timestamp": time.time(),
                        }
                    )
                )
            else:
                await ws.send(
                    json.dumps(
                        {
                            "type": "error",
                            "turn_id": turn_id,
                            "error": f"Invalid GitHub agent ID: {agent_id}",
                        }
                    )
                )

        elif agent_type == "browser":
            # Route to browser peer
            # Extract ws_id from browser:ws_id format
            target_ws_id = (
                agent_id.replace("browser:", "")
                if agent_id.startswith("browser:")
                else agent_id
            )
            target_ws = None

            # Find the target browser peer's websocket
            if target_ws_id in _GATEWAY_STATE.get("browser_peers", {}):
                peer_info = _GATEWAY_STATE["browser_peers"][target_ws_id]
                target_ws = peer_info.get("ws")
            elif target_ws_id in self.clients:
                target_ws = self.clients[target_ws_id]

            if target_ws and target_ws != ws:
                enriched_prompt = self._enrich_prompt_with_ring(prompt)
                # Send invoke to browser peer — browser will process with its agent
                await target_ws.send(
                    json.dumps(
                        {
                            "type": "browser_invoke",
                            "turn_id": turn_id,
                            "from_ws_id": sender_ws_id,
                            "prompt": enriched_prompt,
                            "ring_context": self._get_ring_context(10),
                            "timestamp": time.time(),
                        }
                    )
                )
                # Don't send turn_end yet — browser will send browser_invoke_response
                # which we'll forward
            elif target_ws == ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "error",
                            "turn_id": turn_id,
                            "error": "Cannot send to yourself",
                        }
                    )
                )
            else:
                await ws.send(
                    json.dumps(
                        {
                            "type": "error",
                            "turn_id": turn_id,
                            "error": f"Browser {agent_id} not connected",
                        }
                    )
                )

        elif agent_type == "zenoh":
            try:
                # Check if target is self (this DevDuck instance)
                # If so, invoke agent directly — avoids ZENOH_STATE dependency
                is_self = False
                try:
                    from devduck.tools.zenoh_peer import get_instance_id, ZENOH_STATE

                    my_id = (
                        get_instance_id() if ZENOH_STATE.get("instance_id") else None
                    )
                    if my_id and agent_id == my_id:
                        is_self = True
                except:
                    pass

                # Also check registry for is_self flag
                if not is_self:
                    try:
                        from devduck.tools.mesh_registry import registry

                        agent_info = registry.get_agent(agent_id)
                        if agent_info and agent_info.get("metadata", {}).get(
                            "is_self", False
                        ):
                            is_self = True
                    except:
                        pass

                if is_self:
                    # Self-invocation: use agent directly, no zenoh pubsub needed
                    logger.info(
                        f"Proxy: Self-invoke detected for {agent_id}, using agent directly"
                    )
                    enriched_prompt = self._enrich_prompt_with_ring(prompt)
                    loop = asyncio.get_running_loop()

                    def _self_invoke():
                        try:
                            # Get agent from zenoh state or devduck global
                            agent = None
                            try:
                                from devduck.tools.zenoh_peer import ZENOH_STATE

                                agent = ZENOH_STATE.get("agent")
                            except:
                                pass
                            if not agent:
                                try:
                                    import devduck as _dd

                                    agent = _dd.devduck.agent
                                except:
                                    pass
                            if agent:
                                result = agent(enriched_prompt)
                                return str(result)
                            else:
                                return "❌ No agent available for self-processing"
                        except Exception as e:
                            return f"❌ Error: {e}"

                    response = await loop.run_in_executor(None, _self_invoke)
                else:
                    # Remote peer: use zenoh send_to_peer directly
                    # First, verify the peer exists in the mesh registry (file-based, cross-process)
                    peer_verified = False
                    try:
                        from devduck.tools.mesh_registry import registry

                        agent_info = registry.get_agent(agent_id)
                        if agent_info and agent_info.get("type") == "zenoh":
                            peer_verified = True
                            logger.info(
                                f"Proxy: Peer {agent_id} verified in mesh registry"
                            )
                    except Exception as e:
                        logger.debug(f"Proxy: Registry check failed: {e}")

                    if not peer_verified:
                        response = f"❌ Peer '{agent_id}' not found in mesh registry. Check mesh_registry(action='list') for available agents."
                    else:
                        # Use send_to_peer from zenoh — it publishes to devduck/cmd/{peer_id}
                        from devduck.tools.zenoh_peer import (
                            send_to_peer,
                            ZENOH_STATE as _ZS,
                        )

                        # Ensure the peer is in ZENOH_STATE["peers"] so send_to_peer doesn't reject it
                        # The registry file is the truth — sync it into in-memory state if missing
                        if agent_id not in _ZS.get("peers", {}):
                            meta = agent_info.get("metadata", {})
                            _ZS.setdefault("peers", {})[agent_id] = {
                                "last_seen": agent_info.get("last_seen", time.time()),
                                "hostname": meta.get("hostname", "unknown"),
                                "model": meta.get("model", "unknown"),
                                "tools": meta.get("tools", []),
                                "tool_count": meta.get("tool_count", 0),
                                "system_prompt": meta.get("system_prompt", ""),
                                "cwd": meta.get("cwd", ""),
                                "platform": meta.get("platform", ""),
                            }
                            logger.info(
                                f"Proxy: Synced peer {agent_id} from registry into ZENOH_STATE"
                            )

                        enriched_prompt = self._enrich_prompt_with_ring(prompt)
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(
                            None,
                            lambda: send_to_peer(
                                agent_id, enriched_prompt, wait_time=120.0
                            ),
                        )
                        content_entries = result.get("content", [])
                        response_parts = []
                        for entry in content_entries:
                            text = entry.get("text", "")
                            if text.startswith("📥") or text.startswith("\n📥"):
                                lines = text.strip().split("\n", 1)
                                if len(lines) > 1:
                                    response_parts.append(lines[1].strip())
                                else:
                                    response_parts.append(text.strip())
                            elif text.startswith("❌") or text.startswith("\n❌"):
                                response_parts.append(text.strip())
                            elif text.startswith("⏱️") or text.startswith("\n⏱️"):
                                response_parts.append(text.strip())
                        response = (
                            "\n".join(response_parts)
                            if response_parts
                            else "No response from peer"
                        )
            except Exception as e:
                response = f"Error: {e}"

            try:
                from devduck.tools.unified_mesh import add_to_ring

                add_to_ring(agent_id, "zenoh", f"Q: {prompt[:50]} A: {response[:100]}")
            except:
                pass

            await ws.send(
                json.dumps(
                    {
                        "type": "turn_end",
                        "turn_id": turn_id,
                        "agent_id": agent_id,
                        "agent_type": "zenoh",
                        "response": response,
                        "timestamp": time.time(),
                    }
                )
            )

        elif agent_type == "agentcore":
            enriched_data = dict(data)
            ring_block = self._build_ring_block(20)
            ring_entries = self._get_ring_context(20)
            if ring_block:
                base_sp = data.get("system_prompt") or ""
                enriched_data["system_prompt"] = (
                    f"{base_sp}\n\n{ring_block}\n\nUse this to maintain continuity."
                )
            if ring_entries:
                ring_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": f"[mesh:{e.get('agent_id','mesh')}] {e.get('text','')}"
                            }
                        ],
                    }
                    for e in ring_entries
                ]
                enriched_data["messages"] = ring_messages + (data.get("messages") or [])
            enriched_data["return_messages"] = data.get("return_messages", False)

            result = await self._invoke_agentcore_streaming(
                ws, turn_id, agent_id, prompt, enriched_data, self.region
            )
            response = result.get("content", [{}])[0].get("text", "")
            try:
                from devduck.tools.unified_mesh import add_to_ring

                add_to_ring(
                    agent_id, "agentcore", f"Q: {prompt[:50]} A: {response[:100]}"
                )
            except:
                pass
            await ws.send(
                json.dumps(
                    {
                        "type": "turn_end",
                        "turn_id": turn_id,
                        "agent_id": agent_id,
                        "agent_type": "agentcore",
                        "response": response,
                        "session_id": result.get("session_id"),
                        "timestamp": time.time(),
                    }
                )
            )
        elif agent_type == "local":
            # Local DevDuck agent — invoke directly
            try:
                enriched_prompt = self._enrich_prompt_with_ring(prompt)
                loop = asyncio.get_running_loop()

                def _local_invoke():
                    try:
                        agent = None
                        try:
                            from devduck.tools.zenoh_peer import ZENOH_STATE

                            agent = ZENOH_STATE.get("agent")
                        except:
                            pass
                        if not agent:
                            try:
                                import devduck as _dd

                                agent = _dd.devduck.agent
                            except:
                                pass
                        if agent:
                            return str(agent(enriched_prompt))
                        else:
                            return "❌ No local agent available"
                    except Exception as e:
                        return f"❌ Error: {e}"

                response = await loop.run_in_executor(None, _local_invoke)
            except Exception as e:
                response = f"Error: {e}"

            try:
                from devduck.tools.unified_mesh import add_to_ring

                add_to_ring(agent_id, "local", f"Q: {prompt[:50]} A: {response[:100]}")
            except:
                pass

            await ws.send(
                json.dumps(
                    {
                        "type": "turn_end",
                        "turn_id": turn_id,
                        "agent_id": agent_id,
                        "agent_type": "local",
                        "response": response,
                        "timestamp": time.time(),
                    }
                )
            )

        else:
            await ws.send(
                json.dumps(
                    {
                        "type": "error",
                        "turn_id": turn_id,
                        "error": f"Unknown agent type: {agent_type}",
                    }
                )
            )

    async def _invoke_agentcore_streaming(
        self, ws, turn_id, agent_id, prompt, data, region
    ):
        """Invoke AgentCore agent with streaming to WebSocket.

        IMPORTANT: boto3 is synchronous — all blocking I/O runs in a thread pool
        via run_in_executor to avoid blocking the event loop. This allows multiple
        AgentCore invocations to run concurrently (user can chat with agent A
        while agent B is still responding).
        """
        try:
            import boto3
            from botocore.config import Config

            loop = asyncio.get_running_loop()

            # Build session and payload (lightweight, can stay on event loop)
            session_id = data.get("session_id") or str(uuid.uuid4())
            if len(session_id) < 33:
                session_id = (
                    session_id + "-" + str(uuid.uuid4())[: 33 - len(session_id) - 1]
                )

            payload = {
                "prompt": prompt,
                "mode": data.get("mode", "streaming"),
                "session_id": session_id,
            }
            for key in (
                "system_prompt",
                "tools",
                "model",
                "max_tokens",
                "temperature",
                "messages",
                "return_messages",
                "force_new_session",
            ):
                if data.get(key) is not None:
                    payload[key] = data[key]

            # Run ALL blocking boto3 calls in a thread pool executor
            # This is the critical fix: without this, the event loop blocks
            # and no other WebSocket messages can be processed until this finishes.
            def _blocking_invoke():
                """Runs in thread pool — safe to block here."""
                sts = boto3.client("sts", region_name=region)
                account_id = sts.get_caller_identity()["Account"]
                agent_arn = f"arn:aws:bedrock-agentcore:{region}:{account_id}:runtime/{agent_id}"

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
                return response

            response = await loop.run_in_executor(None, _blocking_invoke)

            # Stream chunks — iterate in thread, send WS from event loop
            # We use a queue-based approach: thread reads chunks, event loop sends them
            import queue

            chunk_queue = queue.Queue()
            _done_sentinel = object()

            def _read_chunks():
                """Read all chunks from boto3 response in thread pool."""
                try:
                    _line_buffer = ""
                    for chunk in response.get("response", []):
                        chunk_str = ""
                        if isinstance(chunk, bytes):
                            chunk_str = chunk.decode("utf-8", errors="ignore")
                        elif isinstance(chunk, str):
                            chunk_str = chunk
                        if not chunk_str:
                            continue
                        _line_buffer += chunk_str
                        while "\n" in _line_buffer:
                            line, _line_buffer = _line_buffer.split("\n", 1)
                            parsed = self._parse_sse_line(line)
                            if parsed is not None:
                                chunk_queue.put(parsed)
                    # Process remaining buffer
                    if _line_buffer.strip():
                        parsed = self._parse_sse_line(_line_buffer)
                        if parsed is not None:
                            chunk_queue.put(parsed)
                except Exception as e:
                    chunk_queue.put({"type": "error", "data": str(e)})
                finally:
                    chunk_queue.put(_done_sentinel)

            # Start chunk reader in thread
            reader_future = loop.run_in_executor(None, _read_chunks)

            full_response = ""
            # Drain queue from event loop — non-blocking
            while True:
                try:
                    # Non-blocking get with short sleep to yield to event loop
                    parsed = await loop.run_in_executor(
                        None, lambda: chunk_queue.get(timeout=0.1)
                    )
                except queue.Empty:
                    # Check if reader is done
                    if reader_future.done():
                        # Drain remaining items
                        while not chunk_queue.empty():
                            parsed = chunk_queue.get_nowait()
                            if parsed is _done_sentinel:
                                break
                            if parsed["type"] == "text":
                                text = parsed["data"]
                                full_response += text
                                await ws.send(
                                    json.dumps(
                                        {
                                            "type": "chunk",
                                            "turn_id": turn_id,
                                            "agent_id": agent_id,
                                            "data": text,
                                            "timestamp": time.time(),
                                        }
                                    )
                                )
                            elif parsed["type"] in ("tool_start", "tool_end"):
                                await ws.send(
                                    json.dumps(
                                        {
                                            "type": parsed["type"],
                                            "turn_id": turn_id,
                                            "agent_id": agent_id,
                                            "tool_name": parsed.get("tool_name", ""),
                                            "tool_id": parsed.get("tool_id", ""),
                                            "timestamp": time.time(),
                                        }
                                    )
                                )
                        break
                    continue

                if parsed is _done_sentinel:
                    break

                if parsed["type"] == "text":
                    text = parsed["data"]
                    full_response += text
                    await ws.send(
                        json.dumps(
                            {
                                "type": "chunk",
                                "turn_id": turn_id,
                                "agent_id": agent_id,
                                "data": text,
                                "timestamp": time.time(),
                            }
                        )
                    )
                elif parsed["type"] in ("tool_start", "tool_end"):
                    await ws.send(
                        json.dumps(
                            {
                                "type": parsed["type"],
                                "turn_id": turn_id,
                                "agent_id": agent_id,
                                "tool_name": parsed.get("tool_name", ""),
                                "tool_id": parsed.get("tool_id", ""),
                                "timestamp": time.time(),
                            }
                        )
                    )
                elif parsed["type"] == "error":
                    logger.error(f"AgentCore chunk reader error: {parsed['data']}")

            return {
                "status": "success",
                "content": [{"text": full_response}],
                "session_id": session_id,
            }
        except Exception as e:
            logger.error(f"AgentCore streaming error: {e}")
            return {"status": "error", "content": [{"text": str(e)}]}

    _GARBAGE_PATTERNS = (
        "SpanContext",
        "trace_id=0x",
        "span_id=0x",
        "trace_flags=",
        "is_remote=",
        "object at 0x",
        "<strands.",
        "AgentResult(",
        "EventLoopMetrics(",
        "'agent':",
        "'current_tool':",
        "'model':",
        "record_direct_tool_call",
    )

    def _parse_sse_line(self, raw_line: str) -> Optional[Dict]:
        line = raw_line.strip()
        if not line or line.startswith(":"):
            return None
        if line.startswith("data: "):
            line = line[6:]
        elif line.startswith("data:"):
            line = line[5:]
        if not line:
            return None
        for pattern in self._GARBAGE_PATTERNS:
            if pattern in line:
                return None
        if not line.startswith("{") and not line.startswith("["):
            return None
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None
        event_str = str(event)
        for pattern in self._GARBAGE_PATTERNS:
            if pattern in event_str:
                if "contentBlockDelta" in event or "contentBlockStart" in event:
                    break
                return None
        return self._extract_text(event)

    def _extract_text(self, event: Dict) -> Optional[Dict]:
        if not isinstance(event, dict):
            return None
        if "contentBlockStart" in event:
            cbs = event["contentBlockStart"]
            if isinstance(cbs, dict) and "start" in cbs:
                start = cbs["start"]
                if isinstance(start, dict) and "toolUse" in start:
                    tu = start["toolUse"]
                    return {
                        "type": "tool_start",
                        "tool_name": tu.get("name", "unknown"),
                        "tool_id": tu.get("toolUseId", ""),
                    }
        if "tool_name" in event and "tool_id" in event:
            return {
                "type": "tool_start",
                "tool_name": event["tool_name"],
                "tool_id": event["tool_id"],
            }
        if "contentBlockDelta" in event:
            cbd = event["contentBlockDelta"]
            if isinstance(cbd, dict):
                delta = cbd.get("delta", {})
                if isinstance(delta, dict):
                    text = delta.get("text", "")
                    if text:
                        return {"type": "text", "data": text}
            return None
        if "event" in event and isinstance(event["event"], dict):
            return self._extract_text(event["event"])
        if "delta" in event and isinstance(event["delta"], dict):
            text = event["delta"].get("text", "")
            if isinstance(text, str) and text:
                return {"type": "text", "data": text}
            return None
        if "message" in event and isinstance(event["message"], dict):
            return None
        if "messageStop" in event or "metadata" in event:
            return None
        if "data" in event:
            data = event["data"]
            if isinstance(data, str):
                if (
                    "<strands." in data
                    or "object at 0x" in data
                    or "SpanContext" in data
                ):
                    return None
                if data:
                    return {"type": "text", "data": data}
            if isinstance(data, dict):
                return self._extract_text(data)
            return None
        if "text" in event and isinstance(event["text"], str):
            text = event["text"]
            if "<strands." in text or "object at 0x" in text:
                return None
            if text:
                return {"type": "text", "data": text}
        if "model" in event and "response" not in event:
            return None
        return None

    async def _handle_github_trigger(self, ws, data):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: trigger_github_agent(
                repo=data.get("repo"),
                workflow=data.get("workflow", "agent.yml"),
                prompt=data.get("prompt", ""),
                token=data.get("token"),
                system_prompt=data.get("system_prompt"),
                tools=data.get("tools"),
                model=data.get("model"),
                max_tokens=data.get("max_tokens"),
            ),
        )
        await ws.send(json.dumps({"type": "github_triggered", **result}))
        if result.get("status") == "success":
            asyncio.create_task(
                self._poll_github_run_logs(
                    ws,
                    {
                        "repo": data.get("repo"),
                        "workflow": data.get("workflow", "agent.yml"),
                        "token": data.get("token"),
                        "agent_id": f"github:{data.get('repo')}:{data.get('workflow', 'agent.yml')}",
                    },
                )
            )

    async def _poll_github_run_logs(self, ws, data):
        import io
        import zipfile as _zipfile

        repo = data.get("repo")
        workflow = data.get("workflow", "agent.yml")
        run_id = data.get("run_id")
        agent_id = data.get("agent_id", f"github:{repo}:{workflow}")
        token = (
            data.get("token")
            or _GATEWAY_STATE.get("github_token")
            or os.getenv("GITHUB_TOKEN")
            or os.getenv("PAT_TOKEN")
        )
        if not token or not repo:
            await ws.send(
                json.dumps({"type": "error", "error": "repo and token required"})
            )
            return
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        loop = asyncio.get_running_loop()
        if not run_id:
            for attempt in range(15):
                await asyncio.sleep(2)
                try:
                    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs"
                    resp = await loop.run_in_executor(
                        None,
                        lambda: requests.get(
                            url, headers=headers, params={"per_page": 1}, timeout=10
                        ),
                    )
                    if resp.status_code == 200:
                        runs = resp.json().get("workflow_runs", [])
                        if runs and runs[0].get("status") in ("queued", "in_progress"):
                            run_id = runs[0]["id"]
                            break
                        elif runs:
                            run_id = runs[0]["id"]
                            break
                except Exception as e:
                    logger.debug(f"Run discovery attempt {attempt}: {e}")
        if not run_id:
            await ws.send(
                json.dumps(
                    {
                        "type": "github_status",
                        "agent_id": agent_id,
                        "repo": repo,
                        "status": "not_found",
                        "message": "Could not find workflow run",
                    }
                )
            )
            return
        await ws.send(
            json.dumps(
                {
                    "type": "github_status",
                    "agent_id": agent_id,
                    "repo": repo,
                    "run_id": run_id,
                    "status": "polling",
                    "message": f"Tracking run {run_id}",
                    "url": f"https://github.com/{repo}/actions/runs/{run_id}",
                    "timestamp": time.time(),
                }
            )
        )
        last_log_length = 0
        prev_status = None
        for poll_num in range(180):
            try:
                run_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}"
                run_resp = await loop.run_in_executor(
                    None, lambda: requests.get(run_url, headers=headers, timeout=10)
                )
                if run_resp.status_code != 200:
                    await asyncio.sleep(5)
                    continue
                run_data = run_resp.json()
                status = run_data.get("status", "unknown")
                conclusion = run_data.get("conclusion")
                if status != prev_status:
                    prev_status = status
                    await ws.send(
                        json.dumps(
                            {
                                "type": "github_status",
                                "agent_id": agent_id,
                                "repo": repo,
                                "run_id": run_id,
                                "status": status,
                                "conclusion": conclusion,
                                "url": f"https://github.com/{repo}/actions/runs/{run_id}",
                                "timestamp": time.time(),
                            }
                        )
                    )
                if status in ("in_progress", "completed"):
                    try:
                        logs_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/logs"
                        logs_resp = await loop.run_in_executor(
                            None,
                            lambda: requests.get(logs_url, headers=headers, timeout=30),
                        )
                        if logs_resp.status_code == 200 and logs_resp.content:
                            try:
                                with _zipfile.ZipFile(
                                    io.BytesIO(logs_resp.content)
                                ) as zf:
                                    txt_files = [
                                        n for n in zf.namelist() if n.endswith(".txt")
                                    ]
                                    if txt_files:
                                        biggest = max(
                                            txt_files,
                                            key=lambda n: zf.getinfo(n).file_size,
                                        )
                                        log_content = zf.read(biggest).decode(
                                            "utf-8", errors="ignore"
                                        )
                                        if len(log_content) > last_log_length:
                                            new_chunk = log_content[last_log_length:]
                                            last_log_length = len(log_content)
                                            await ws.send(
                                                json.dumps(
                                                    {
                                                        "type": "github_logs",
                                                        "agent_id": agent_id,
                                                        "repo": repo,
                                                        "run_id": run_id,
                                                        "log_chunk": new_chunk,
                                                        "total_size": last_log_length,
                                                        "status": status,
                                                        "conclusion": conclusion,
                                                        "timestamp": time.time(),
                                                    }
                                                )
                                            )
                            except _zipfile.BadZipFile:
                                pass
                    except Exception as e:
                        logger.debug(f"Log fetch error: {e}")
                if status == "completed":
                    try:
                        from devduck.tools.unified_mesh import add_to_ring

                        add_to_ring(
                            agent_id,
                            "github",
                            f"Run {run_id} {conclusion or 'completed'} — {repo}",
                        )
                    except:
                        pass
                    await ws.send(
                        json.dumps(
                            {
                                "type": "github_status",
                                "agent_id": agent_id,
                                "repo": repo,
                                "run_id": run_id,
                                "status": "completed",
                                "conclusion": conclusion,
                                "url": f"https://github.com/{repo}/actions/runs/{run_id}",
                                "timestamp": time.time(),
                            }
                        )
                    )
                    break
            except Exception as e:
                logger.debug(f"Poll error: {e}")
            await asyncio.sleep(5)

    async def _handle_broadcast(self, ws, data):
        message = data.get("message", "")
        if not message:
            await ws.send(json.dumps({"type": "error", "error": "No message"}))
            return
        try:
            from devduck.tools.zenoh_peer import broadcast_message

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: broadcast_message(message, wait_time=30.0)
            )
            await ws.send(
                json.dumps(
                    {
                        "type": "broadcast_result",
                        "result": result,
                        "timestamp": time.time(),
                    }
                )
            )
        except Exception as e:
            await ws.send(json.dumps({"type": "error", "error": str(e)}))

    async def start(self):
        try:
            import websockets
        except ImportError:
            logger.error("websockets not installed")
            return False
        try:
            self.server = await websockets.serve(
                self.handle_client, "0.0.0.0", self.port
            )
            self._running = True
            logger.info(f"MeshGateway started on ws://0.0.0.0:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start: {e}")
            return False

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self._running = False


_gateway: Optional[MeshGateway] = None


def _start_gateway(port: int, region: str) -> Dict[str, Any]:
    global _gateway, _GATEWAY_STATE
    if _GATEWAY_STATE["running"]:
        return {
            "status": "success",
            "content": [
                {"text": f"Gateway already running on port {_GATEWAY_STATE['port']}"}
            ],
        }
    _gateway = MeshGateway(port=port, region=region)

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _GATEWAY_STATE["loop"] = loop
        try:
            loop.run_until_complete(_gateway.start())
            _GATEWAY_STATE["running"] = True
            _GATEWAY_STATE["port"] = port
            _GATEWAY_STATE["region"] = region
            loop.run_forever()
        finally:
            loop.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    time.sleep(1)
    agents = {"github": 0, "agentcore": 0, "zenoh": 0}
    try:
        github_agents = get_github_agents()
        agents["github"] = len(github_agents)
    except:
        pass
    try:
        from devduck.tools.unified_mesh import get_agentcore_agents, get_zenoh_peers

        agents["agentcore"] = len(get_agentcore_agents(region))
        agents["zenoh"] = len(get_zenoh_peers())
    except:
        pass
    return {
        "status": "success",
        "content": [
            {
                "text": f"✅ Agent Relay started on ws://0.0.0.0:{port}\n\n"
                f"Agents discovered:\n  • GitHub: {agents['github']}\n  • AgentCore: {agents['agentcore']}\n  • Zenoh: {agents['zenoh']}\n\n"
                f"Browser auto-discovery: ws://localhost:{port}\nOpen one.html to see all agents!"
            }
        ],
    }


def _stop_gateway() -> Dict[str, Any]:
    global _gateway, _GATEWAY_STATE
    if not _GATEWAY_STATE["running"]:
        return {"status": "success", "content": [{"text": "Gateway not running"}]}
    port = _GATEWAY_STATE["port"]
    if _gateway:
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_gateway.stop())
            loop.close()
        except:
            pass
    _GATEWAY_STATE["running"] = False
    _GATEWAY_STATE["port"] = None
    _GATEWAY_STATE["loop"] = None
    _GATEWAY_STATE["browser_peers"] = {}
    _gateway = None
    return {
        "status": "success",
        "content": [{"text": f"✅ Gateway stopped (was on port {port})"}],
    }


def send_to_browser_peer(
    ws_id: str, prompt: str, turn_id: str = None, zenoh_requester_id: str = None
) -> Dict[str, Any]:
    """Send a message to a browser peer from CLI/zenoh side.

    This is the bridge: zenoh peers can call this to invoke browser agents.
    The browser will process the prompt with its local agent and respond.

    Args:
        ws_id: The browser peer's ws_id (from browser:{ws_id} format)
        prompt: The prompt to send
        turn_id: Optional turn ID for tracking
        zenoh_requester_id: The zenoh instance ID of the requester

    Returns:
        Status dict
    """
    browser_peers = _GATEWAY_STATE.get("browser_peers", {})
    if ws_id not in browser_peers:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"Browser peer {ws_id} not found. Available: {list(browser_peers.keys())}"
                }
            ],
        }

    peer_info = browser_peers[ws_id]
    ws = peer_info.get("ws")
    if not ws:
        return {
            "status": "error",
            "content": [{"text": f"Browser peer {ws_id} has no websocket"}],
        }

    loop = _GATEWAY_STATE.get("loop")
    if not loop:
        return {
            "status": "error",
            "content": [{"text": "Gateway event loop not running"}],
        }

    if not turn_id:
        turn_id = str(uuid.uuid4())[:8]

    msg = json.dumps(
        {
            "type": "browser_invoke",
            "turn_id": turn_id,
            "from_ws_id": "zenoh",
            "zenoh_requester_id": zenoh_requester_id or "",
            "prompt": prompt,
            "ring_context": [],
            "timestamp": time.time(),
        }
    )

    try:
        asyncio.run_coroutine_threadsafe(ws.send(msg), loop)
        return {
            "status": "success",
            "content": [
                {"text": f"Message sent to browser peer {ws_id}, turn_id={turn_id}"}
            ],
        }
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Failed to send: {e}"}]}


@tool
def agentcore_proxy(
    action: str,
    port: int = 8090,
    region: str = "us-west-2",
) -> Dict[str, Any]:
    """
    AgentCore Proxy - WebSocket gateway unifying Zenoh peers + AgentCore agents.

    Provides a single WebSocket interface where both local DevDuck instances
    (via Zenoh) and deployed AgentCore agents appear as peers in the same
    ring context.

    Args:
        action: Action to perform:
            - "start": Start the gateway
            - "stop": Stop the gateway
            - "status": Check status and list peers
            - "restart": Restart the gateway
        port: WebSocket port (default: 8090)
        region: AWS region for AgentCore (default: us-west-2)

    Returns:
        Dict with status and gateway information

    Examples:
        # Start gateway
        agentcore_proxy(action="start", port=8090)

        # Check status
        agentcore_proxy(action="status")

        # Stop
        agentcore_proxy(action="stop")

    WebSocket Protocol:
        # List all peers (Zenoh + AgentCore)
        {"type": "list_peers"}

        # Invoke specific peer
        {
            "type": "invoke",
            "peer_id": "devduck-xyz123",
            "peer_type": "agentcore",  # or "zenoh"
            "prompt": "hello"
        }

        # Broadcast to all Zenoh peers
        {"type": "broadcast", "message": "sync now"}

        # Get ring context
        {"type": "get_ring", "max_entries": 20}

        # Add to ring context
        {"type": "add_ring", "agent_id": "browser", "text": "user note"}
    """
    try:
        if action == "start":
            return _start_gateway(port, region)
        elif action == "stop":
            return _stop_gateway()
        elif action == "status":
            if _GATEWAY_STATE["running"]:
                agents = {"github": 0, "agentcore": 0, "zenoh": 0, "browser": 0}
                try:
                    agents["github"] = len(get_github_agents())
                except:
                    pass
                try:
                    from devduck.tools.unified_mesh import (
                        get_agentcore_agents,
                        get_zenoh_peers,
                        MESH_STATE,
                    )

                    agents["agentcore"] = len(
                        get_agentcore_agents(_GATEWAY_STATE["region"])
                    )
                    agents["zenoh"] = len(get_zenoh_peers())
                    agents["browser"] = len(_GATEWAY_STATE.get("browser_peers", {}))
                except:
                    pass
                total = sum(agents.values())
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": f"✅ Gateway running on port {_GATEWAY_STATE['port']}\nRegion: {_GATEWAY_STATE['region']}\n\nAgents ({total}):\n  • GitHub: {agents['github']}\n  • AgentCore: {agents['agentcore']}\n  • Zenoh: {agents['zenoh']}\n  • Browser: {agents['browser']}"
                        }
                    ],
                }
            else:
                return {
                    "status": "success",
                    "content": [{"text": "❌ Gateway not running"}],
                }
        elif action == "restart":
            _stop_gateway()
            time.sleep(1)
            return _start_gateway(port, region)
        else:
            return {
                "status": "error",
                "content": [{"text": f"Unknown action: {action}"}],
            }
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}
