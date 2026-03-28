"""
🪪 Identity — SQLite-backed DevDuck persona & configuration manager.

Each identity is a complete DevDuck configuration profile stored in SQLite:
system prompt, model, tools, communication channels (Telegram/Slack/WhatsApp),
server ports, ambient mode, knowledge base, env vars, and cloud sync.

Database location (priority order):
    1. db_path parameter on each tool call
    2. DEVDUCK_IDENTITY_DB env var
    3. ~/.devduck/identities.db (default)

For Lambda / edge / containers, set:
    DEVDUCK_IDENTITY_DB=/tmp/identities.db

Examples:
    # Create a code reviewer identity
    identity(action="create", name="code-reviewer",
             system_prompt="You are a senior code reviewer...",
             model_provider="bedrock", model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
             tools_config="strands_tools:shell,file_read;devduck:use_github,lsp",
             tags="dev,security,python")

    # Use custom db path (Lambda / edge)
    identity(action="list", db_path="/tmp")

    # Search
    identity(action="search", query="telegram")

    # Activate — sets all env vars, restart to apply
    identity(action="activate", name="code-reviewer")

    # Compare two identities
    identity(action="diff", name="code-reviewer", description="devops-bot")

    # Clone and customize
    identity(action="clone", name="code-reviewer", description="strict-reviewer")

    # Sync to tiny.technology cloud
    identity(action="sync", name="code-reviewer")
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from strands import tool

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

_DEFAULT_DB_DIR = Path.home() / ".devduck"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "identities.db"
_last_resolved_path: Path = _DEFAULT_DB_PATH


def _resolve_db_path(db_path: str = "") -> Path:
    """Resolve database path from: param > env var > default.

    Priority:
        1. Explicit db_path parameter
        2. DEVDUCK_IDENTITY_DB environment variable
        3. ~/.devduck/identities.db
    """
    if db_path:
        p = Path(db_path)
    else:
        env_path = os.getenv("DEVDUCK_IDENTITY_DB", "")
        if env_path:
            p = Path(env_path)
        else:
            p = _DEFAULT_DB_PATH

    # If path is a directory, append the db filename
    if p.is_dir():
        p = p / "identities.db"

    # Ensure parent directory exists
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _get_db(db_path: str = "") -> sqlite3.Connection:
    """Get SQLite connection with WAL mode.

    Args:
        db_path: Override path. Falls back to DEVDUCK_IDENTITY_DB env or ~/.devduck/identities.db
    """
    global _last_resolved_path
    resolved = _resolve_db_path(db_path)
    _last_resolved_path = resolved
    conn = sqlite3.connect(str(resolved), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(conn)
    return conn


def _db_display(conn: sqlite3.Connection) -> str:
    """Get the display path from the last resolved path."""
    return str(_last_resolved_path)


def _ensure_schema(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS identities (
            name TEXT PRIMARY KEY,
            description TEXT DEFAULT '',

            -- Core AI persona
            system_prompt TEXT DEFAULT '',
            system_knowledge TEXT DEFAULT '',
            model_provider TEXT DEFAULT '',
            model_id TEXT DEFAULT '',
            max_tokens INTEGER DEFAULT 60000,
            temperature REAL DEFAULT 1.0,

            -- Tool configuration
            tools_config TEXT DEFAULT '',
            mcp_servers TEXT DEFAULT '',
            load_tools_from_dir INTEGER DEFAULT 1,

            -- Communication channels
            telegram_token TEXT DEFAULT '',
            telegram_chat_id TEXT DEFAULT '',
            slack_token TEXT DEFAULT '',
            slack_channel TEXT DEFAULT '',
            whatsapp_number TEXT DEFAULT '',

            -- Server config
            enable_ws INTEGER DEFAULT 1,
            ws_port INTEGER DEFAULT 10001,
            enable_tcp INTEGER DEFAULT 0,
            tcp_port INTEGER DEFAULT 10002,
            enable_mcp INTEGER DEFAULT 0,
            mcp_port INTEGER DEFAULT 10003,
            enable_zenoh INTEGER DEFAULT 1,
            enable_ipc INTEGER DEFAULT 0,
            ipc_socket TEXT DEFAULT '',

            -- Mesh / proxy
            enable_agentcore_proxy INTEGER DEFAULT 1,
            agentcore_proxy_port INTEGER DEFAULT 10000,

            -- Ambient mode
            ambient_mode INTEGER DEFAULT 0,
            ambient_idle_seconds REAL DEFAULT 30.0,
            ambient_max_iterations INTEGER DEFAULT 15,
            ambient_cooldown REAL DEFAULT 60.0,
            autonomous_max_iterations INTEGER DEFAULT 100,
            autonomous_cooldown REAL DEFAULT 10.0,

            -- Knowledge base
            knowledge_base_id TEXT DEFAULT '',

            -- Cloud sync (tiny.technology)
            tiny_name TEXT DEFAULT '',
            tiny_synced INTEGER DEFAULT 0,
            tiny_url TEXT DEFAULT '',

            -- Custom env vars (JSON object)
            env_vars TEXT DEFAULT '{}',

            -- Metadata
            tags TEXT DEFAULT '[]',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            last_activated_at TEXT DEFAULT '',
            activation_count INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_identities_tags ON identities(tags);
        CREATE INDEX IF NOT EXISTS idx_identities_model ON identities(model_provider);
        CREATE INDEX IF NOT EXISTS idx_identities_updated ON identities(updated_at);

        CREATE TABLE IF NOT EXISTS identity_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identity_name TEXT NOT NULL,
            action TEXT NOT NULL,
            changes TEXT DEFAULT '{}',
            timestamp TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (identity_name) REFERENCES identities(name) ON DELETE CASCADE
        );
    """)
    conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert sqlite3.Row to dict, parsing JSON fields."""
    if row is None:
        return None
    d = dict(row)
    for field in ("env_vars", "tags"):
        if field in d and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


def _record_history(conn: sqlite3.Connection, name: str, action: str, changes: dict = None):
    """Record identity change history."""
    conn.execute(
        "INSERT INTO identity_history (identity_name, action, changes) VALUES (?, ?, ?)",
        (name, action, json.dumps(changes or {})),
    )


# ---------------------------------------------------------------------------
# Updatable fields
# ---------------------------------------------------------------------------

_TEXT_FIELDS = [
    "description", "system_prompt", "system_knowledge", "model_provider",
    "model_id", "tools_config", "mcp_servers", "telegram_token",
    "telegram_chat_id", "slack_token", "slack_channel", "whatsapp_number",
    "ipc_socket", "knowledge_base_id", "tiny_name",
]
_INT_FIELDS = [
    "max_tokens", "enable_ws", "ws_port", "enable_tcp", "tcp_port",
    "enable_mcp", "mcp_port", "enable_zenoh", "enable_ipc",
    "enable_agentcore_proxy", "agentcore_proxy_port", "ambient_mode",
    "ambient_max_iterations", "autonomous_max_iterations",
    "load_tools_from_dir",
]
_FLOAT_FIELDS = [
    "temperature", "ambient_idle_seconds", "ambient_cooldown",
    "autonomous_cooldown",
]


def _collect_updates(kwargs: dict) -> dict:
    """Collect only explicitly-provided field updates from kwargs."""
    updates: dict = {}
    for f in _TEXT_FIELDS:
        v = kwargs.get(f, "")
        if v:
            updates[f] = v
    for f in _INT_FIELDS:
        v = kwargs.get(f, -1)
        if v >= 0:
            updates[f] = v
    for f in _FLOAT_FIELDS:
        v = kwargs.get(f, -1.0)
        if v >= 0:
            updates[f] = v
    # Special: tags & env_vars
    tags = kwargs.get("tags", "")
    if tags:
        updates["tags"] = json.dumps([t.strip() for t in tags.split(",") if t.strip()])
    env_vars = kwargs.get("env_vars", "")
    if env_vars:
        updates["env_vars"] = env_vars
    return updates


# ---------------------------------------------------------------------------
# Activation env-var mapping
# ---------------------------------------------------------------------------

_ENV_MAP = {
    "system_prompt":             "SYSTEM_PROMPT",
    "model_provider":            "MODEL_PROVIDER",
    "model_id":                  "STRANDS_MODEL_ID",
    "max_tokens":                "STRANDS_MAX_TOKENS",
    "temperature":               "STRANDS_TEMPERATURE",
    "tools_config":              "DEVDUCK_TOOLS",
    "mcp_servers":               "MCP_SERVERS",
    "telegram_token":            "TELEGRAM_BOT_TOKEN",
    "telegram_chat_id":          "TELEGRAM_CHAT_ID",
    "slack_token":               "SLACK_BOT_TOKEN",
    "slack_channel":             "SLACK_CHANNEL",
    "whatsapp_number":           "WHATSAPP_NUMBER",
    "enable_ws":                 "DEVDUCK_ENABLE_WS",
    "ws_port":                   "DEVDUCK_WS_PORT",
    "enable_tcp":                "DEVDUCK_ENABLE_TCP",
    "tcp_port":                  "DEVDUCK_TCP_PORT",
    "enable_mcp":                "DEVDUCK_ENABLE_MCP",
    "mcp_port":                  "DEVDUCK_MCP_PORT",
    "enable_zenoh":              "DEVDUCK_ENABLE_ZENOH",
    "enable_ipc":                "DEVDUCK_ENABLE_IPC",
    "ipc_socket":                "DEVDUCK_IPC_SOCKET",
    "enable_agentcore_proxy":    "DEVDUCK_ENABLE_AGENTCORE_PROXY",
    "agentcore_proxy_port":      "DEVDUCK_AGENTCORE_PROXY_PORT",
    "ambient_mode":              "DEVDUCK_AMBIENT_MODE",
    "ambient_idle_seconds":      "DEVDUCK_AMBIENT_IDLE_SECONDS",
    "ambient_max_iterations":    "DEVDUCK_AMBIENT_MAX_ITERATIONS",
    "ambient_cooldown":          "DEVDUCK_AMBIENT_COOLDOWN",
    "autonomous_max_iterations": "DEVDUCK_AUTONOMOUS_MAX_ITERATIONS",
    "autonomous_cooldown":       "DEVDUCK_AUTONOMOUS_COOLDOWN",
    "knowledge_base_id":         "DEVDUCK_KNOWLEDGE_BASE_ID",
    "load_tools_from_dir":       "DEVDUCK_LOAD_TOOLS_FROM_DIR",
}

# Boolean-style integer fields rendered as "true"/"false"
_BOOL_FIELDS = {
    "enable_ws", "enable_tcp", "enable_mcp", "enable_zenoh", "enable_ipc",
    "enable_agentcore_proxy", "ambient_mode", "load_tools_from_dir",
}


# ---------------------------------------------------------------------------
# Zenoh publish helper — publishes identity activity to the mesh
# ---------------------------------------------------------------------------

def _publish_to_zenoh(identity_name: str, action: str, task: str, result: str, status: str = "success"):
    """Publish identity activity to Zenoh mesh so all peers see it.

    Publishes to: devduck/identity/{identity_name}
    Safe to call even if Zenoh is not running — silently no-ops.

    Args:
        identity_name: Name of the identity that acted
        action: "talk" or "fan_out"
        task: The task/query that was given
        result: The result (truncated)
        status: "success" or "error"
    """
    try:
        import sys as _sys
        _zp_mod = _sys.modules.get("devduck.tools.zenoh_peer")
        if not _zp_mod:
            from devduck.tools import zenoh_peer as _zp_mod

        if not _zp_mod.ZENOH_STATE.get("running"):
            return

        import time
        _zp_mod.publish_message(
            f"devduck/identity/{identity_name}",
            {
                "type": "identity_activity",
                "identity": identity_name,
                "action": action,
                "task": task[:500],
                "result_preview": result[:1000],
                "status": status,
                "timestamp": time.time(),
                "source_instance": _zp_mod.ZENOH_STATE.get("instance_id", "unknown"),
            },
        )
    except Exception:
        pass  # Zenoh not available — that's fine



# ---------------------------------------------------------------------------
# Agent builder — creates a fresh Strands Agent from identity config
# ---------------------------------------------------------------------------

def _build_agent_from_identity(data: dict, additional_prompt: str = ""):
    """Build a Strands Agent from an identity's stored configuration.

    Args:
        data: Identity dict from _row_to_dict()
        additional_prompt: Extra context appended to system prompt

    Returns:
        Agent instance, or error string if build fails
    """
    try:
        from strands import Agent

        # Build system prompt
        sys_prompt = data.get("system_prompt", "") or ""
        sys_knowledge = data.get("system_knowledge", "") or ""
        if sys_knowledge:
            sys_prompt += f"\n\n## Knowledge:\n{sys_knowledge}"
        if additional_prompt:
            sys_prompt += f"\n\n## Additional Context:\n{additional_prompt}"

        if not sys_prompt.strip():
            sys_prompt = f"You are {data.get('name', 'an AI assistant')}."

        # Load tools from tools_config
        tools = []
        tools_config = data.get("tools_config", "") or ""
        if tools_config:
            try:
                from devduck.tools.manage_tools import _load_tools_from_spec
                loaded = _load_tools_from_spec(tools_config)
                if loaded:
                    tools = loaded
            except Exception as e:
                # Fallback: try the basic loading approach
                for group in tools_config.split(";"):
                    group = group.strip()
                    if not group or ":" not in group:
                        continue
                    package, tool_names_str = group.split(":", 1)
                    for tool_name in tool_names_str.split(","):
                        tool_name = tool_name.strip()
                        if not tool_name:
                            continue
                        try:
                            mod = __import__(package.strip(), fromlist=[tool_name])
                            tools.append(getattr(mod, tool_name))
                        except Exception:
                            pass

        # Select model
        model_provider = data.get("model_provider", "") or ""
        model_id_val = data.get("model_id", "") or ""
        max_tokens = data.get("max_tokens", 60000) or 60000
        temperature = data.get("temperature", 1.0)
        if temperature < 0:
            temperature = 1.0

        model = None
        if model_provider or model_id_val:
            try:
                # Use the same model selection logic as DevDuck
                if model_provider == "ollama":
                    from strands.models.ollama import OllamaModel
                    model = OllamaModel(
                        model_id=model_id_val or "qwen3:1.7b",
                        temperature=temperature,
                        num_predict=max_tokens,
                    )
                elif model_provider == "gemini":
                    from strands.models.gemini import GeminiModel
                    model = GeminiModel(
                        model_id=model_id_val or "gemini-2.5-flash",
                    )
                elif model_provider == "mlx":
                    from strands_mlx import MLXModel
                    model = MLXModel(
                        model_id=model_id_val or "mlx-community/Qwen3-1.7B-4bit",
                    )
                else:
                    # Bedrock, Anthropic, OpenAI, etc — use create_model
                    from strands_tools.utils.models.model import create_model
                    # Temporarily set env vars for model creation
                    old_provider = os.environ.get("MODEL_PROVIDER", "")
                    old_model = os.environ.get("STRANDS_MODEL_ID", "")
                    old_max = os.environ.get("STRANDS_MAX_TOKENS", "")
                    old_temp = os.environ.get("STRANDS_TEMPERATURE", "")
                    try:
                        if model_provider:
                            os.environ["MODEL_PROVIDER"] = model_provider
                        if model_id_val:
                            os.environ["STRANDS_MODEL_ID"] = model_id_val
                        os.environ["STRANDS_MAX_TOKENS"] = str(max_tokens)
                        os.environ["STRANDS_TEMPERATURE"] = str(temperature)
                        model = create_model(provider=model_provider or None)
                    finally:
                        # Restore env vars
                        for k, v in [("MODEL_PROVIDER", old_provider),
                                     ("STRANDS_MODEL_ID", old_model),
                                     ("STRANDS_MAX_TOKENS", old_max),
                                     ("STRANDS_TEMPERATURE", old_temp)]:
                            if v:
                                os.environ[k] = v
                            elif k in os.environ:
                                del os.environ[k]
            except Exception as e:
                # Fall back to no explicit model (uses default)
                model = None

        # Build the agent — no servers, no file watcher, just tools + prompt
        agent_kwargs = {
            "system_prompt": sys_prompt,
            "load_tools_from_directory": False,
        }
        if tools:
            agent_kwargs["tools"] = tools
        if model:
            agent_kwargs["model"] = model

        return Agent(**agent_kwargs)

    except Exception as e:
        return f"Failed to build agent: {e}"


def _load_tools_from_spec(tools_config: str) -> list:
    """Load tools from a DEVDUCK_TOOLS-style config string.

    Format: package1:tool1,tool2;package2:tool3
    """
    tools = []
    for group in tools_config.split(";"):
        group = group.strip()
        if not group or ":" not in group:
            continue
        package, tool_names_str = group.split(":", 1)
        for tool_name in tool_names_str.split(","):
            tool_name = tool_name.strip()
            if not tool_name:
                continue
            try:
                mod = __import__(package.strip(), fromlist=[tool_name])
                tools.append(getattr(mod, tool_name))
            except Exception:
                pass
    return tools




# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool
def identity(
    action: str,
    name: str = "",
    description: str = "",
    system_prompt: str = "",
    system_knowledge: str = "",
    model_provider: str = "",
    model_id: str = "",
    max_tokens: int = -1,
    temperature: float = -1.0,
    tools_config: str = "",
    mcp_servers: str = "",
    telegram_token: str = "",
    telegram_chat_id: str = "",
    slack_token: str = "",
    slack_channel: str = "",
    whatsapp_number: str = "",
    enable_ws: int = -1,
    ws_port: int = -1,
    enable_tcp: int = -1,
    tcp_port: int = -1,
    enable_mcp: int = -1,
    mcp_port: int = -1,
    enable_zenoh: int = -1,
    enable_ipc: int = -1,
    ipc_socket: str = "",
    enable_agentcore_proxy: int = -1,
    agentcore_proxy_port: int = -1,
    ambient_mode: int = -1,
    ambient_idle_seconds: float = -1.0,
    ambient_max_iterations: int = -1,
    ambient_cooldown: float = -1.0,
    autonomous_max_iterations: int = -1,
    autonomous_cooldown: float = -1.0,
    load_tools_from_dir: int = -1,
    knowledge_base_id: str = "",
    env_vars: str = "",
    tags: str = "",
    query: str = "",
    tiny_name: str = "",
    db_path: str = "",
) -> Dict[str, Any]:
    """🪪 Manage DevDuck identities — portable AI persona configurations in SQLite.

    Each identity stores ALL DevDuck configuration: system prompt, model, tools,
    communication channels (Telegram/Slack/WhatsApp), server ports, ambient mode,
    knowledge base, env vars, and cloud sync to tiny.technology.

    Database location (priority):
        1. db_path parameter — pass on every call
        2. DEVDUCK_IDENTITY_DB env var — set once for the process
        3. ~/.devduck/identities.db — default for desktop

    For Lambda: identity(action="list", db_path="/tmp")
    For edge:   export DEVDUCK_IDENTITY_DB=/data/identities.db

    Args:
        action: Action to perform:
            - "create": Create new identity
            - "get": Get identity by name
            - "update": Update identity fields (only provided fields change)
            - "delete": Delete identity
            - "list": List all identities
            - "search": Full-text search across name, description, prompt, knowledge, tags
            - "activate": Switch DevDuck to this identity (sets all env vars)
            - "export": Export identity as JSON
            - "import": Import from JSON (pass JSON string in system_knowledge)
            - "clone": Clone identity (name=source, description=new_name)
            - "history": Show change history
            - "diff": Compare two identities (name=first, description=second)
            - "talk": Send a query to an identity (spawns a fresh agent with its config)
            - "fan_out": Run multiple identities in parallel with independent tasks.
                         Pass JSON array in system_knowledge:
                         [{"identity": "name", "task": "prompt", "context": "optional extra"}]
            - "sync": Sync to tiny.technology cloud
            - "stats": Database statistics
        name: Identity name (unique identifier)
        query: Search query (for search action)
        tags: Comma-separated tags (e.g., "dev,python,reviewer")
        env_vars: JSON string of custom env vars (e.g., '{"MY_KEY": "value"}')
        db_path: Custom database path or directory. Overrides DEVDUCK_IDENTITY_DB env var.
                 If a directory is given, 'identities.db' is appended automatically.
                 Examples: "/tmp", "/tmp/identities.db", "/data/myagent"
        ... all other params map 1:1 to DevDuck __init__.py configuration
    """
    conn = None
    try:
        conn = _get_db(db_path)
        db_display = _db_display(conn)

        # ── CREATE ──────────────────────────────────────────────
        if action == "create":
            if not name:
                return {"status": "error", "content": [{"text": "Name required."}]}
            if conn.execute("SELECT 1 FROM identities WHERE name=?", (name,)).fetchone():
                return {"status": "error", "content": [{"text": f"'{name}' exists. Use action='update'."}]}

            fields = {"name": name}
            fields.update(_collect_updates(locals()))
            if "description" not in fields or not fields["description"]:
                fields["description"] = f"DevDuck identity: {name}"
            if "tiny_name" not in fields or not fields["tiny_name"]:
                fields["tiny_name"] = name

            cols = ", ".join(fields.keys())
            phs = ", ".join(["?"] * len(fields))
            conn.execute(f"INSERT INTO identities ({cols}) VALUES ({phs})", list(fields.values()))
            _record_history(conn, name, "created", {k: str(v)[:100] for k, v in fields.items()})
            conn.commit()

            return {"status": "success", "content": [{"text":
                f"✅ Identity **{name}** created!\n"
                f"📁 DB: `{db_display}`\n"
                f"🎯 `identity(action='activate', name='{name}')`"}]}

        # ── GET ─────────────────────────────────────────────────
        elif action == "get":
            if not name:
                return {"status": "error", "content": [{"text": "Name required."}]}
            row = conn.execute("SELECT * FROM identities WHERE name=?", (name,)).fetchone()
            if not row:
                return {"status": "error", "content": [{"text": f"'{name}' not found."}]}
            data = _row_to_dict(row)
            display = dict(data)
            for secret in ("telegram_token", "slack_token"):
                if display.get(secret):
                    v = display[secret]
                    display[secret] = v[:8] + "..." + v[-4:] if len(v) > 12 else "***"
            return {"status": "success", "content": [{"text":
                f"🪪 **{name}**\n```json\n{json.dumps(display, indent=2, default=str)}\n```"}]}

        # ── UPDATE ──────────────────────────────────────────────
        elif action == "update":
            if not name:
                return {"status": "error", "content": [{"text": "Name required."}]}
            if not conn.execute("SELECT 1 FROM identities WHERE name=?", (name,)).fetchone():
                return {"status": "error", "content": [{"text": f"'{name}' not found. Use create."}]}

            updates = _collect_updates(locals())
            if not updates:
                return {"status": "error", "content": [{"text": "No fields to update."}]}

            updates["updated_at"] = datetime.now().isoformat()
            set_clause = ", ".join(f"{k}=?" for k in updates)
            conn.execute(f"UPDATE identities SET {set_clause} WHERE name=?",
                         list(updates.values()) + [name])
            _record_history(conn, name, "updated", {k: str(v)[:100] for k, v in updates.items()})
            conn.commit()
            return {"status": "success", "content": [{"text": f"✅ **{name}** updated ({len(updates)} fields)"}]}

        # ── DELETE ──────────────────────────────────────────────
        elif action == "delete":
            if not name:
                return {"status": "error", "content": [{"text": "Name required."}]}
            cur = conn.execute("DELETE FROM identities WHERE name=?", (name,))
            conn.commit()
            if cur.rowcount:
                return {"status": "success", "content": [{"text": f"🗑️ **{name}** deleted"}]}
            return {"status": "error", "content": [{"text": f"'{name}' not found."}]}

        # ── LIST ────────────────────────────────────────────────
        elif action == "list":
            rows = conn.execute("""
                SELECT name, description, model_provider, model_id, tools_config,
                       telegram_token, slack_token, whatsapp_number,
                       ambient_mode, tiny_synced, activation_count, tags, updated_at
                FROM identities ORDER BY updated_at DESC
            """).fetchall()

            if not rows:
                return {"status": "success", "content": [{"text":
                    "No identities yet.\n`identity(action='create', name='my-bot', system_prompt='...')`"}]}

            lines = [f"🪪 **{len(rows)} Identities** (db: `{db_display}`)\n"]
            lines.append("| Name | Description | Model | Channels | Tags | Used |")
            lines.append("|------|-------------|-------|----------|------|------|")
            for r in rows:
                ch = []
                if r["telegram_token"]: ch.append("📱TG")
                if r["slack_token"]: ch.append("💬Slack")
                if r["whatsapp_number"]: ch.append("📞WA")
                if r["ambient_mode"]: ch.append("🌙Amb")
                channels = " ".join(ch) or "—"

                model = r["model_id"] or r["model_provider"] or "default"
                if len(model) > 25:
                    model = model[:22] + "..."

                tags_list = json.loads(r["tags"]) if r["tags"] else []
                tags_str = ", ".join(tags_list[:3]) or "—"
                desc = (r["description"] or "")[:35]

                lines.append(
                    f"| **{r['name']}** | {desc} | {model} | {channels} | {tags_str} | {r['activation_count']}× |"
                )
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        # ── SEARCH ──────────────────────────────────────────────
        elif action == "search":
            if not query:
                return {"status": "error", "content": [{"text": "query param required."}]}
            q = f"%{query}%"
            rows = conn.execute("""
                SELECT name, description, model_id, tags, system_prompt
                FROM identities
                WHERE name LIKE ? OR description LIKE ? OR system_prompt LIKE ?
                      OR system_knowledge LIKE ? OR tags LIKE ? OR model_id LIKE ?
                      OR tools_config LIKE ? OR env_vars LIKE ?
                ORDER BY updated_at DESC LIMIT 20
            """, (q, q, q, q, q, q, q, q)).fetchall()

            if not rows:
                return {"status": "success", "content": [{"text": f"No results for '{query}'"}]}

            lines = [f"🔍 **{len(rows)}** results for '{query}':\n"]
            for r in rows:
                tags_list = json.loads(r["tags"]) if r["tags"] else []
                tag_str = f" [{', '.join(tags_list)}]" if tags_list else ""
                lines.append(f"- **{r['name']}**: {(r['description'] or '')[:50]}{tag_str}")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        # ── ACTIVATE ────────────────────────────────────────────
        elif action == "activate":
            if not name:
                return {"status": "error", "content": [{"text": "Name required."}]}
            row = conn.execute("SELECT * FROM identities WHERE name=?", (name,)).fetchone()
            if not row:
                return {"status": "error", "content": [{"text": f"'{name}' not found."}]}

            data = _row_to_dict(row)
            activated = [f"🪪 Activating **{name}**...\n"]
            env_set = {}

            for col, env_key in _ENV_MAP.items():
                val = data.get(col)
                if val is None or val == "" or val == 0:
                    continue
                if col in _BOOL_FIELDS:
                    env_set[env_key] = "true" if val else "false"
                else:
                    env_set[env_key] = str(val)

            # Custom env vars
            custom_env = data.get("env_vars", {})
            if isinstance(custom_env, dict):
                for k, v in custom_env.items():
                    env_set[k] = str(v)

            # Apply all env vars
            for k, v in env_set.items():
                os.environ[k] = v

            # Summary
            sections = {
                "📝 Persona": ["SYSTEM_PROMPT", "MODEL_PROVIDER", "STRANDS_MODEL_ID",
                               "STRANDS_MAX_TOKENS", "STRANDS_TEMPERATURE"],
                "🔧 Tools": ["DEVDUCK_TOOLS", "MCP_SERVERS", "DEVDUCK_LOAD_TOOLS_FROM_DIR"],
                "📱 Channels": ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "SLACK_BOT_TOKEN",
                                "SLACK_CHANNEL", "WHATSAPP_NUMBER"],
                "🌐 Servers": ["DEVDUCK_ENABLE_WS", "DEVDUCK_WS_PORT", "DEVDUCK_ENABLE_TCP",
                               "DEVDUCK_ENABLE_ZENOH", "DEVDUCK_ENABLE_AGENTCORE_PROXY"],
                "🌙 Ambient": ["DEVDUCK_AMBIENT_MODE", "DEVDUCK_AMBIENT_IDLE_SECONDS",
                               "DEVDUCK_AMBIENT_MAX_ITERATIONS"],
                "📚 Knowledge": ["DEVDUCK_KNOWLEDGE_BASE_ID"],
            }
            for label, keys in sections.items():
                set_keys = [k for k in keys if k in env_set]
                if set_keys:
                    activated.append(f"{label}: {', '.join(set_keys)}")

            if custom_env:
                activated.append(f"🔑 Custom: {len(custom_env)} env vars")

            # Update activation stats
            conn.execute("""
                UPDATE identities
                SET last_activated_at=?, activation_count=activation_count+1
                WHERE name=?
            """, (datetime.now().isoformat(), name))
            _record_history(conn, name, "activated")
            conn.commit()

            activated.append(f"\n✅ Set **{len(env_set)} env vars**. Restart DevDuck to fully apply.")
            return {"status": "success", "content": [{"text": "\n".join(activated)}]}

        # ── EXPORT ──────────────────────────────────────────────
        elif action == "export":
            if not name:
                return {"status": "error", "content": [{"text": "Name required."}]}
            row = conn.execute("SELECT * FROM identities WHERE name=?", (name,)).fetchone()
            if not row:
                return {"status": "error", "content": [{"text": f"'{name}' not found."}]}
            return {"status": "success", "content": [{"text": json.dumps(_row_to_dict(row), indent=2, default=str)}]}

        # ── IMPORT ──────────────────────────────────────────────
        elif action == "import":
            if not system_knowledge:
                return {"status": "error", "content": [{"text": "Pass JSON in system_knowledge param."}]}
            try:
                data = json.loads(system_knowledge)
            except json.JSONDecodeError:
                return {"status": "error", "content": [{"text": "Invalid JSON."}]}

            import_name = name or data.get("name", "")
            if not import_name:
                return {"status": "error", "content": [{"text": "Name required."}]}

            data["name"] = import_name
            data["updated_at"] = datetime.now().isoformat()

            for field in ("env_vars", "tags"):
                if field in data and not isinstance(data[field], str):
                    data[field] = json.dumps(data[field])

            cols_info = conn.execute("PRAGMA table_info(identities)").fetchall()
            valid_cols = {c["name"] for c in cols_info}
            filtered = {k: v for k, v in data.items() if k in valid_cols}

            cols = ", ".join(filtered.keys())
            phs = ", ".join(["?"] * len(filtered))
            conflict = ", ".join(f"{k}=excluded.{k}" for k in filtered if k != "name")

            conn.execute(
                f"INSERT INTO identities ({cols}) VALUES ({phs}) ON CONFLICT(name) DO UPDATE SET {conflict}",
                list(filtered.values()),
            )
            _record_history(conn, import_name, "imported")
            conn.commit()
            return {"status": "success", "content": [{"text": f"✅ Imported **{import_name}**"}]}

        # ── CLONE ───────────────────────────────────────────────
        elif action == "clone":
            if not name or not description:
                return {"status": "error", "content": [{"text": "name=source, description=new_name"}]}

            row = conn.execute("SELECT * FROM identities WHERE name=?", (name,)).fetchone()
            if not row:
                return {"status": "error", "content": [{"text": f"Source '{name}' not found."}]}

            data = _row_to_dict(row)
            new_name = description
            data["name"] = new_name
            data["description"] = f"Cloned from {name}"
            data["created_at"] = datetime.now().isoformat()
            data["updated_at"] = datetime.now().isoformat()
            data["activation_count"] = 0
            data["tiny_synced"] = 0
            data["last_activated_at"] = ""

            for field in ("env_vars", "tags"):
                if field in data and not isinstance(data[field], str):
                    data[field] = json.dumps(data[field])

            cols_info = conn.execute("PRAGMA table_info(identities)").fetchall()
            valid_cols = {c["name"] for c in cols_info}
            filtered = {k: v for k, v in data.items() if k in valid_cols}

            cols = ", ".join(filtered.keys())
            phs = ", ".join(["?"] * len(filtered))
            conn.execute(f"INSERT INTO identities ({cols}) VALUES ({phs})", list(filtered.values()))
            _record_history(conn, new_name, "cloned", {"source": name})
            conn.commit()
            return {"status": "success", "content": [{"text": f"✅ Cloned **{name}** → **{new_name}**"}]}

        # ── HISTORY ─────────────────────────────────────────────
        elif action == "history":
            target = name or "%"
            rows = conn.execute("""
                SELECT identity_name, action, changes, timestamp
                FROM identity_history
                WHERE identity_name LIKE ?
                ORDER BY timestamp DESC LIMIT 30
            """, (target,)).fetchall()

            if not rows:
                return {"status": "success", "content": [{"text": f"No history{' for ' + name if name else ''}."}]}

            header = f"for **{name}**" if name else "(all)"
            lines = [f"📜 History {header} ({len(rows)} entries):\n"]
            for r in rows:
                changes = json.loads(r["changes"]) if r["changes"] else {}
                detail = ", ".join(list(changes.keys())[:5]) if changes else ""
                lines.append(f"- `{r['timestamp']}` **{r['identity_name']}** {r['action']} {detail}")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        # ── DIFF ────────────────────────────────────────────────
        elif action == "diff":
            if not name or not description:
                return {"status": "error", "content": [{"text": "name=first, description=second"}]}

            r1 = conn.execute("SELECT * FROM identities WHERE name=?", (name,)).fetchone()
            r2 = conn.execute("SELECT * FROM identities WHERE name=?", (description,)).fetchone()
            missing = name if not r1 else (description if not r2 else None)
            if missing:
                return {"status": "error", "content": [{"text": f"'{missing}' not found."}]}

            d1, d2 = _row_to_dict(r1), _row_to_dict(r2)
            skip = {"created_at", "updated_at", "last_activated_at", "activation_count"}

            lines = [f"🔀 **{name}** vs **{description}**\n"]
            lines.append(f"| Field | {name} | {description} |")
            lines.append(f"|-------|{'─' * len(name)}|{'─' * len(description)}|")

            diffs = 0
            for k in d1:
                if k in skip:
                    continue
                v1, v2 = str(d1.get(k, "")), str(d2.get(k, ""))
                if v1 != v2:
                    diffs += 1
                    s1 = (v1[:30] + "…") if len(v1) > 30 else (v1 or "—")
                    s2 = (v2[:30] + "…") if len(v2) > 30 else (v2 or "—")
                    lines.append(f"| **{k}** | {s1} | {s2} |")

            if diffs == 0:
                lines.append("| — | *identical* | *identical* |")
            lines.append(f"\n**{diffs} differences**")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        # ── SYNC ────────────────────────────────────────────────
        elif action == "sync":
            if not name:
                return {"status": "error", "content": [{"text": "Name required."}]}
            row = conn.execute("SELECT * FROM identities WHERE name=?", (name,)).fetchone()
            if not row:
                return {"status": "error", "content": [{"text": f"'{name}' not found."}]}

            data = _row_to_dict(row)
            import urllib.request

            tiny = tiny_name or data.get("tiny_name") or name
            payload = json.dumps({
                "name": tiny,
                "systemPrompt": data.get("system_prompt", ""),
                "systemKnowledge": (
                    data.get("system_knowledge", "")
                    + f"\n\nModel: {data.get('model_id', 'default')}"
                    + f"\nTools: {data.get('tools_config', '')}"
                ),
            }).encode()

            req = urllib.request.Request(
                "https://api.tiny.technology/upsert",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    result = json.loads(resp.read())
                conn.execute(
                    "UPDATE identities SET tiny_synced=1, tiny_name=?, tiny_url=? WHERE name=?",
                    (tiny, f"https://tiny.technology/{tiny}", name),
                )
                _record_history(conn, name, "synced", {"tiny_name": tiny})
                conn.commit()
                return {"status": "success", "content": [{"text":
                    f"☁️ Synced **{name}** → tiny.technology/{tiny}\n{result.get('response', '')}"}]}
            except Exception as e:
                return {"status": "error", "content": [{"text": f"Sync failed: {e}"}]}

        # ── STATS ───────────────────────────────────────────────
        elif action == "stats":
            resolved = _resolve_db_path(db_path)
            total = conn.execute("SELECT COUNT(*) as c FROM identities").fetchone()["c"]
            with_tg = conn.execute(
                "SELECT COUNT(*) as c FROM identities WHERE telegram_token != ''"
            ).fetchone()["c"]
            with_slack = conn.execute(
                "SELECT COUNT(*) as c FROM identities WHERE slack_token != ''"
            ).fetchone()["c"]
            with_wa = conn.execute(
                "SELECT COUNT(*) as c FROM identities WHERE whatsapp_number != ''"
            ).fetchone()["c"]
            with_ambient = conn.execute(
                "SELECT COUNT(*) as c FROM identities WHERE ambient_mode = 1"
            ).fetchone()["c"]
            history_count = conn.execute("SELECT COUNT(*) as c FROM identity_history").fetchone()["c"]
            most_used = conn.execute(
                "SELECT name, activation_count FROM identities ORDER BY activation_count DESC LIMIT 5"
            ).fetchall()
            db_size = resolved.stat().st_size / 1024 if resolved.exists() else 0

            lines = [
                "📊 **Identity Database Stats**\n",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Total identities | {total} |",
                f"| With Telegram | {with_tg} |",
                f"| With Slack | {with_slack} |",
                f"| With WhatsApp | {with_wa} |",
                f"| Ambient mode | {with_ambient} |",
                f"| History entries | {history_count} |",
                f"| DB size | {db_size:.1f} KB |",
                f"| DB path | `{db_display}` |",
            ]
            if most_used:
                active = [r for r in most_used if r["activation_count"] > 0]
                if active:
                    lines.append("\n**Most activated:**")
                    for r in active:
                        lines.append(f"- {r['name']}: {r['activation_count']}×")

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}


        # ── TALK ────────────────────────────────────────────────
        elif action == "talk":
            if not name:
                return {"status": "error", "content": [{"text": "Name required — which identity to talk to."}]}
            if not query:
                return {"status": "error", "content": [{"text": "query required — what to say."}]}

            row = conn.execute("SELECT * FROM identities WHERE name=?", (name,)).fetchone()
            if not row:
                return {"status": "error", "content": [{"text": f"'{name}' not found."}]}
            data = _row_to_dict(row)

            # Build agent from identity
            agent = _build_agent_from_identity(data)
            if isinstance(agent, str):
                return {"status": "error", "content": [{"text": agent}]}

            try:
                result = agent(query)
                result_str = str(result)

                # Record activation
                conn.execute("""
                    UPDATE identities
                    SET last_activated_at=?, activation_count=activation_count+1
                    WHERE name=?
                """, (datetime.now().isoformat(), name))
                _record_history(conn, name, "talked", {"query": query[:200], "result": result_str[:200]})
                conn.commit()

                # Publish to Zenoh mesh
                _publish_to_zenoh(name, "talk", query, result_str)

                return {"status": "success", "content": [{"text":
                    f"🪪 **{name}** responded:\n\n{result_str}"}]}
            except Exception as e:
                _publish_to_zenoh(name, "talk", query, str(e), status="error")
                return {"status": "error", "content": [{"text": f"Agent error for '{name}': {e}"}]}

        # ── FAN_OUT ─────────────────────────────────────────────
        elif action == "fan_out":
            if not system_knowledge:
                return {"status": "error", "content": [{"text":
                    "Pass task list in system_knowledge as JSON:\n"
                    '```json\n[\n  {"identity": "code-reviewer", "task": "Review this PR"},\n'
                    '  {"identity": "devops-bot", "task": "Check deployment readiness"}\n]\n```'}]}

            try:
                tasks = json.loads(system_knowledge)
            except json.JSONDecodeError:
                return {"status": "error", "content": [{"text": "Invalid JSON in system_knowledge."}]}

            if not isinstance(tasks, list) or not tasks:
                return {"status": "error", "content": [{"text": "Expected a JSON array of task objects."}]}

            # Validate all identities exist before starting
            agents = {}
            for i, task in enumerate(tasks):
                identity_name = task.get("identity") or task.get("name", "")
                task_prompt = task.get("task") or task.get("prompt") or task.get("query", "")
                if not identity_name:
                    return {"status": "error", "content": [{"text": f"Task #{i}: missing 'identity' field."}]}
                if not task_prompt:
                    return {"status": "error", "content": [{"text": f"Task #{i} ({identity_name}): missing 'task' field."}]}

                row = conn.execute("SELECT * FROM identities WHERE name=?", (identity_name,)).fetchone()
                if not row:
                    return {"status": "error", "content": [{"text": f"Identity '{identity_name}' not found."}]}

                data = _row_to_dict(row)
                agent = _build_agent_from_identity(data, additional_prompt=task.get("context", ""))
                if isinstance(agent, str):
                    return {"status": "error", "content": [{"text": f"Can't build agent for '{identity_name}': {agent}"}]}

                agents[i] = {"name": identity_name, "agent": agent, "task": task_prompt}

            # Run all tasks in parallel
            import concurrent.futures
            results = {}
            errors = {}

            def _run_task(idx, info):
                try:
                    result = info["agent"](info["task"])
                    result_str = str(result)
                    # Each identity publishes its own result to Zenoh
                    _publish_to_zenoh(info["name"], "fan_out", info["task"], result_str)
                    return idx, result_str, None
                except Exception as e:
                    _publish_to_zenoh(info["name"], "fan_out", info["task"], str(e), status="error")
                    return idx, None, str(e)

            max_workers = min(len(agents), int(os.getenv("DEVDUCK_FAN_OUT_MAX_WORKERS", "5")))

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_run_task, idx, info): idx
                    for idx, info in agents.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    idx, result, error = future.result()
                    if error:
                        errors[idx] = error
                    else:
                        results[idx] = result

            # Record history for each identity
            for idx, info in agents.items():
                conn.execute("""
                    UPDATE identities
                    SET last_activated_at=?, activation_count=activation_count+1
                    WHERE name=?
                """, (datetime.now().isoformat(), info["name"]))
                status = "completed" if idx in results else "failed"
                _record_history(conn, info["name"], f"fan_out:{status}",
                                {"task": info["task"][:200]})
            conn.commit()

            # Build merged output
            lines = [f"🪪 **Fan-out complete** — {len(results)} succeeded, {len(errors)} failed\n"]

            for idx in sorted(agents.keys()):
                info = agents[idx]
                lines.append(f"---\n### 🪪 {info['name']}")
                lines.append(f"**Task:** {info['task'][:200]}\n")
                if idx in results:
                    lines.append(results[idx])
                else:
                    lines.append(f"❌ Error: {errors.get(idx, 'unknown')}")
                lines.append("")

            merged_text = "\n".join(lines)

            # Publish fan_out summary to Zenoh
            _publish_to_zenoh(
                "fan_out",
                "fan_out_summary",
                f"{len(agents)} identities",
                f"{len(results)} succeeded, {len(errors)} failed",
                status="success" if not errors else "partial",
            )

            return {"status": "success", "content": [{"text": merged_text}]}


        # ── UNKNOWN ─────────────────────────────────────────────
        else:
            return {"status": "error", "content": [{"text":
                f"Unknown action: {action}. Valid: create, get, update, delete, list, search, "
                f"activate, export, import, clone, history, diff, sync, stats"}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
