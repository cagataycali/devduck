# 🦆 DevDuck

**Self-healing agent. One file. Zero config.**

Minimalist AI that adapts to your environment and fixes itself when things break.

## Install

```bash
pipx install "devduck[all]"  # Full install (recommended)
```

Requires: Python 3.10+, Ollama (or set `MODEL_PROVIDER`)

## Quick Start

```bash
devduck                      # Interactive mode (auto-starts servers)
devduck "analyze this code"  # CLI mode
```

```python
import devduck
devduck("calculate 2+2")  # Python API
```

## Core Features

- **🔧 Self-healing** - Auto-fixes dependencies, models, errors
- **🔥 Hot-reload** - Save `.py` files in `./tools/`, use instantly (no restart)
- **🧠 RAG memory** - Set `STRANDS_KNOWLEDGE_BASE_ID` for automatic context retrieval/storage
- **🌍 Multi-protocol** - TCP (9999), WebSocket (8080), MCP (8000), CLI, Python
- **📚 19+ tools** - shell, editor, calculator, python, GitHub, subagents, and more
- **🎯 Adaptive** - Auto-selects model by OS (macOS: 1.7b, Linux: 30b)

## Auto-Started Servers

When you run `devduck`, 3 servers start automatically:

| Server | Endpoint | Usage |
|--------|----------|-------|
| 🌐 **Web UI** | [cagataycali.github.io/devduck](http://cagataycali.github.io/devduck) | Browser interface |
| 🔌 **TCP** | `localhost:9999` | `nc localhost 9999` |
| 🌊 **WebSocket** | `ws://localhost:8080` | Structured JSON messages |
| 🔗 **MCP** | `http://localhost:8000/mcp` | Model Context Protocol |

**Customize ports:**
```bash
devduck --tcp-port 9000 --ws-port 8001 --mcp-port 3000
devduck --no-tcp --no-ws  # Disable specific servers
```

## Connect Options

### MCP Client (Claude Desktop)

**Simple stdio mode (recommended):**
```json
{
  "mcpServers": {
    "devduck": {
      "command": "uvx",
      "args": ["devduck", "--mcp"]
    }
  }
}
```

**Proxy mode (if devduck already running):**
```json
{
  "mcpServers": {
    "devduck": {
      "command": "uvx",
      "args": ["strands-mcp-server", "--upstream-url", "http://localhost:8000/mcp/"]
    }
  }
}
```

### Terminal (TCP)
```bash
nc localhost 9999
> analyze logs
```

### Shell Prefix
```bash
devduck
🦆 ! git status  # Run shell commands with !
```

## Hot-Reload Tool Creation

Create tools instantly—no restart needed:

```python
# ./tools/tip_calc.py
from strands import tool

@tool
def calculate_tip(amount: float, percent: float = 15.0) -> str:
    """Calculate restaurant tip."""
    tip = amount * (percent / 100)
    return f"Tip: ${tip:.2f} | Total: ${amount + tip:.2f}"
```

Save → Available instantly → Use: `devduck "calculate tip for $42"`

## Built-in Tools (19+)

| Category | Tools |
|----------|-------|
| **Development** | shell, editor, python_repl, load_tool, environment |
| **GitHub** | use_github, create_subagent, gist, add_comment, list_issues |
| **Network** | tcp, websocket, mcp_server, mcp_client, http_request |
| **AI** | use_agent, install_tools, retrieve, store_in_kb |
| **Utilities** | calculator, image_reader, scraper, system_prompt, view_logs |

## Multi-Model Support

Switch models via environment variables:

```bash
# Bedrock (Claude)
export MODEL_PROVIDER="bedrock"
export STRANDS_MODEL_ID="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
export STRANDS_MAX_TOKENS="64000"

# Anthropic API
export MODEL_PROVIDER="anthropic"
export STRANDS_MODEL_ID="claude-sonnet-4-20250514"

# Ollama (default)
export MODEL_PROVIDER="ollama"
export OLLAMA_HOST="http://localhost:11434"
```

## Knowledge Base (RAG)

Enable automatic memory across sessions:

```bash
export STRANDS_KNOWLEDGE_BASE_ID="your-kb-id"
devduck  # Auto-retrieves context before queries, stores after responses
```

Works with AWS Bedrock Knowledge Bases.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PROVIDER` | `ollama` | Model provider (bedrock, anthropic, ollama) |
| `STRANDS_KNOWLEDGE_BASE_ID` | - | Enable auto-RAG with KB ID |
| `DEVDUCK_TCP_PORT` | `9999` | TCP server port |
| `DEVDUCK_WS_PORT` | `8080` | WebSocket port |
| `DEVDUCK_MCP_PORT` | `8000` | MCP server port |
| `DEVDUCK_ENABLE_TCP` | `true` | Enable TCP server |
| `DEVDUCK_ENABLE_WS` | `true` | Enable WebSocket |
| `DEVDUCK_ENABLE_MCP` | `true` | Enable MCP server |
| `DEVDUCK_LOG_LINE_COUNT` | `50` | Log lines in context |
| `SYSTEM_PROMPT` | - | Custom system prompt |

## Dynamic Tool Loading

Load tools from any Python package at runtime:

```python
devduck
🦆 install_tools(action="install_and_load", 
              package="strands-fun-tools", 
              module="strands_fun_tools")
```

No restart required—tools available immediately.

## System Prompt Management

Modify agent behavior dynamically:

```bash
devduck
🦆 system_prompt(action="update", prompt="You are a senior Python expert.")
🦆 system_prompt(action="view")  # See current prompt
```

## Logs

```bash
devduck
🦆 view_logs(action="view", lines=100)
🦆 view_logs(action="search", pattern="error")
🦆 view_logs(action="stats")
```

Log location: `/tmp/devduck/logs/devduck.log`

## GitHub Actions

Run DevDuck in CI/CD:

```yaml
- name: DevDuck Analysis
  uses: cagataycali/devduck@main
  with:
    query: "analyze test coverage"
    model: "us.anthropic.claude-sonnet-4-20250514-v1:0"
```

---

**One file. 19+ tools. Self-healing. Hot-reload. RAG memory.**

*Built with [Strands Agents SDK](https://github.com/strands-agents/sdk-python)*
