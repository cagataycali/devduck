# Power User Setup

A real-world `.zshrc` configuration for daily driving DevDuck with all features enabled.

---

## Full Configuration

```bash
# Model — Claude Opus via Bedrock bearer token (fastest auth, no STS calls)
export AWS_BEARER_TOKEN_BEDROCK="ABSK..."
export STRANDS_MODEL_ID="global.anthropic.claude-opus-4-6-v1"
export STRANDS_MAX_TOKENS="64000"

# Tools — curated toolset (loads faster than all 60+)
export DEVDUCK_TOOLS="devduck.tools:use_github,editor,system_prompt,store_in_kb,manage_tools,websocket,zenoh_peer,agentcore_proxy,manage_messages,sqlite_memory,dialog,listen,use_computer,tasks,scheduler,telegram;strands_tools:retrieve,shell,file_read,file_write,use_agent"

# Knowledge Base — automatic RAG (stores & retrieves every conversation)
export STRANDS_KNOWLEDGE_BASE_ID="YOUR_KB_ID"

# MCP — auto-load Strands docs server
export MCP_SERVERS='{"mcpServers":{"strands-docs":{"command":"uvx","args":["strands-agents-mcp-server"]}}}'

# Messaging — Telegram & Slack bots
export TELEGRAM_BOT_TOKEN="your-telegram-bot-token"
export SLACK_BOT_TOKEN="xoxb-your-slack-bot-token"
export SLACK_APP_TOKEN="xapp-your-slack-app-token"

# Spotify control
export SPOTIFY_CLIENT_ID="your-client-id"
export SPOTIFY_CLIENT_SECRET="your-client-secret"
export SPOTIFY_REDIRECT_URI="http://127.0.0.1:8888/callback"

# Gemini as fallback/sub-agent model
export GEMINI_API_KEY="your-gemini-key"
```

---

## What This Gives You

```mermaid
graph TB
    subgraph Config["⚙️ Power User Config"]
        Model["🧠 Claude Opus<br/>Bedrock Bearer Token"]
        KB["📚 Auto-RAG<br/>Knowledge Base"]
        MCP["📖 Strands Docs<br/>MCP Server"]
        Msg["📱 Telegram + Slack + WhatsApp"]
        Music["🎵 Spotify"]
        Mesh["🔗 Zenoh P2P + Mesh"]
        Tools["💬 26 Tools Loaded"]
    end

    Model --> DevDuck["🦆 DevDuck"]
    KB --> DevDuck
    MCP --> DevDuck
    Msg --> DevDuck
    Music --> DevDuck
    Mesh --> DevDuck
    Tools --> DevDuck

    DevDuck --> Expand["🔧 Expandable to 60+<br/>via manage_tools"]

    style Config fill:#1a1a2e,stroke:#4a90d9,color:#fff
    style DevDuck fill:#f5a623,stroke:#333,color:#000
    style Model fill:#9b59b6,stroke:#333,color:#fff
    style KB fill:#e74c3c,stroke:#333,color:#fff
    style MCP fill:#3498db,stroke:#333,color:#fff
    style Msg fill:#2ecc71,stroke:#333,color:#fff
    style Mesh fill:#7ed321,stroke:#333,color:#000
```

| Feature | Benefit |
|---------|---------|
| 🧠 **Opus on Bedrock** | Bearer token auth — zero-latency, no STS calls |
| 📚 **Auto-RAG** | Every conversation stored in KB, context retrieved before each query |
| 📖 **Strands docs** | SDK documentation available as MCP tools |
| 📱 **Telegram + Slack + WhatsApp** | Three messaging channels ready to go |
| 🎵 **Spotify** | Music control via `use_spotify` |
| 🔗 **Zenoh + Mesh** | Auto-enabled multi-terminal awareness |
| 💬 **26 tools** | Curated set on startup, expandable to 60+ on demand |

---

## Curated Tool Breakdown

The `DEVDUCK_TOOLS` string above loads exactly these tools:

### From `devduck.tools` (16 tools)

| Tool | Purpose |
|------|---------|
| `use_github` | GitHub GraphQL API |
| `editor` | File create/replace/insert/undo |
| `system_prompt` | Self-improvement via prompt management |
| `store_in_kb` | Store to Bedrock Knowledge Base |
| `manage_tools` | Runtime tool add/remove/create |
| `websocket` | WebSocket server |
| `zenoh_peer` | P2P auto-discovery |
| `agentcore_proxy` | Unified mesh relay |
| `manage_messages` | Conversation history management |
| `sqlite_memory` | Persistent memory with FTS |
| `dialog` | Interactive terminal dialogs |
| `listen` | Background Whisper transcription |
| `use_computer` | Mouse, keyboard, screenshots |
| `tasks` | Background parallel agents |
| `scheduler` | Cron + one-time jobs |
| `telegram` | Telegram bot |

### From `strands_tools` (5 tools)

| Tool | Purpose |
|------|---------|
| `retrieve` | Bedrock KB retrieval |
| `shell` | Interactive PTY shell |
| `file_read` | File reading with search |
| `file_write` | File writing |
| `use_agent` | Spawn sub-agents with different models |

### Load more on demand

```python
# Add Apple tools
manage_tools(action="add", tools="devduck.tools.apple_vision")
manage_tools(action="add", tools="devduck.tools.apple_nlp")

# Add Slack
manage_tools(action="add", tools="devduck.tools.slack")

# Add from external packages
manage_tools(action="add", tools="strands_fun_tools.clipboard", install=True)
```

---

## Tips

!!! tip "Bearer Token Auth"
    `AWS_BEARER_TOKEN_BEDROCK` is faster than STS-based auth. No credential chain resolution needed. Get it from Bedrock console → "Get API Key".

!!! tip "Multi-Model Workflows"
    Set `GEMINI_API_KEY` to use Gemini as a sub-agent model:
    ```python
    use_agent(
        prompt="summarize this",
        system_prompt="You are a summarizer",
        model_provider="gemini"
    )
    ```

!!! tip "AGENTS.md"
    Drop an `AGENTS.md` in your project directory. DevDuck auto-loads it into the system prompt — project-specific instructions without any config.

!!! tip "Ambient Mode"
    Enable background thinking while you're idle:
    ```bash
    export DEVDUCK_AMBIENT_MODE=true
    ```
    Or type `ambient` in the REPL. Type `auto` for fully autonomous mode.
