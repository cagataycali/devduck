# Architecture

DevDuck is a single `DevDuck` class in `__init__.py` that auto-initializes on import.

---

## High-Level Architecture

```mermaid
graph LR
    User([👤 User]) --> Interface
    subgraph Interface[" "]
        CLI["CLI / REPL"]
        TUI["TUI"]
        WS["WebSocket"]
        TCP["TCP"]
        MCP["MCP"]
    end
    Interface --> Core["🦆 DevDuck Core"]
    Core --> Tools["🔧 60+ Tools"]
    Core <--> Zenoh["🔗 Zenoh P2P"]
    Core <--> KB["📚 Knowledge Base"]
    Core <--> Mesh["🌐 Unified Mesh"]
    Mesh --> Browser["🖥️ Browser"]
    Mesh --> Cloud["☁️ AgentCore"]

    style Core fill:#f5a623,stroke:#333,color:#000
    style Mesh fill:#4a90d9,stroke:#333,color:#fff
    style Zenoh fill:#7ed321,stroke:#333,color:#000
    style KB fill:#9b59b6,stroke:#333,color:#fff
```

**Ports:** 10000 (mesh relay) · 10001 (WebSocket) · 10002 (TCP) · 10003 (MCP) · multicast (Zenoh)

---

## File Structure

```
devduck/
├── __init__.py              # Core: DevDuck class, REPL, CLI, session recording, ambient mode
├── tui.py                   # Multi-conversation Textual TUI
├── landing.py               # Rich landing screen for REPL
├── callback_handler.py      # Streaming callback handler for CLI
├── asciinema_callback_handler.py  # .cast file recording
├── agentcore_handler.py     # HTTP handler for AgentCore deployment
├── tools/                   # 60+ built-in tools
│   ├── system_prompt.py     # Self-improvement via prompt management
│   ├── manage_tools.py      # Runtime tool add/remove/create/fetch
│   ├── manage_messages.py   # Conversation history management
│   ├── websocket.py         # WebSocket server
│   ├── zenoh_peer.py        # P2P auto-discovery networking (1602 lines)
│   ├── agentcore_proxy.py   # Unified mesh relay (1964 lines)
│   ├── unified_mesh.py      # Ring context shared memory (522 lines)
│   ├── mesh_registry.py     # File-based agent discovery (401 lines)
│   ├── tasks.py             # Background parallel agent tasks
│   ├── scheduler.py         # Cron and one-time job scheduling
│   ├── telegram.py          # Telegram bot integration
│   ├── slack.py             # Slack integration
│   ├── whatsapp.py          # WhatsApp via wacli
│   ├── speech_to_speech.py  # Real-time voice
│   ├── lsp.py               # Language Server Protocol
│   ├── use_mac.py           # Unified macOS control
│   ├── apple_vision.py      # On-device OCR, barcode, face detection
│   ├── apple_nlp.py         # On-device NLP
│   └── ...                  # 40+ more tools
└── tools/ (hot-reload)      # ./tools/*.py auto-loaded at runtime
```

---

## Core Design Patterns

### Self-Awareness

The system prompt includes the agent's **complete source code** via `get_own_source_code()`. This means the agent can inspect its own implementation to answer questions accurately — source code is truth, not conversation memory.

### Self-Healing

```mermaid
flowchart TD
    E["Error occurs"] --> Check{"Error type?"}
    Check -->|Context overflow| Clear["Clear message history"]
    Clear --> Retry["Retry with fresh context"]
    Check -->|Connection error| Fix["Check service (ollama serve)"]
    Fix --> Init["Retry __init__()"]
    Check -->|Other| Heal["_self_heal()"]
    Heal --> Init
    Init --> Ready{"Success?"}
    Ready -->|Yes| Done["✅ Recovered"]
    Ready -->|No, attempt < 3| Init
    Ready -->|No, attempt ≥ 3| Fail["❌ Exit"]

    style Clear fill:#f5a623,stroke:#333,color:#000
    style Done fill:#7ed321,stroke:#333,color:#000
    style Fail fill:#e74c3c,stroke:#333,color:#fff
```

### Hot-Reload

A background `_file_watcher_thread` monitors `__init__.py` for changes. On detection, `os.execv()` restarts the process. If the agent is executing, reload is deferred until completion (`_reload_pending`).

### Dynamic Tool Loading

Tools are configured via `DEVDUCK_TOOLS` env var. Additional tools can be loaded at runtime via `manage_tools()` or by dropping `.py` files in `./tools/`.

---

## Initialization Flow

```mermaid
flowchart TD
    A["import devduck"] --> B["DevDuck.__init__()"]
    B --> C["Load tools from DEVDUCK_TOOLS"]
    C --> D["Load MCP servers"]
    D --> E["Select model<br/>(auto-detect provider)"]
    E --> F["Create Strands Agent"]
    F --> G["Start servers<br/>(WS, Zenoh, Proxy)"]
    G --> H["Start file watcher"]
    H --> I{"Ambient mode?"}
    I -->|Yes| J["Start background thread"]
    I -->|No| K["Ready ✅"]
    J --> K

    style A fill:#4a90d9,stroke:#333,color:#fff
    style F fill:#f5a623,stroke:#333,color:#000
    style K fill:#7ed321,stroke:#333,color:#000
```

---

## Query Flow

```mermaid
flowchart TD
    A["devduck(query)"] --> B{"Recording?"}
    B -->|Yes| C["Record user message + snapshot"]
    B -->|No| D["Check ambient results"]
    C --> D
    D --> E{"KB configured?"}
    E -->|Yes| F["Retrieve from Knowledge Base"]
    E -->|No| G["Inject dynamic context"]
    F --> G
    G --> G2["Zenoh peers + Ring context +<br/>Ambient status"]
    G2 --> H["Run LLM agent"]
    H --> I{"Recording?"}
    I -->|Yes| J["Record response + snapshot"]
    I -->|No| K["Store to KB"]
    J --> K
    K --> L["Push to mesh ring"]
    L --> M["Return result ✅"]

    style A fill:#4a90d9,stroke:#333,color:#fff
    style H fill:#f5a623,stroke:#333,color:#000
    style M fill:#7ed321,stroke:#333,color:#000
    style G2 fill:#3498db,stroke:#333,color:#fff
```

---

## Mesh Architecture

See [Unified Mesh](guide/mesh.md) for the full deep dive. Summary:

```mermaid
graph TB
    subgraph Mesh["🌐 Unified Mesh"]
        T["🖥️ Terminal<br/>(Zenoh)"]
        B["🌍 Browser<br/>(WS)"]
        C["☁️ Cloud<br/>(AgentCore)"]
        Ring[("🔄 Ring Context")]

        T <--> Ring
        B <--> Ring
        C <--> Ring
    end

    style Mesh fill:#1a1a2e,stroke:#4a90d9,color:#fff
    style Ring fill:#f5a623,stroke:#333,color:#000
    style T fill:#7ed321,stroke:#333,color:#000
    style B fill:#4a90d9,stroke:#333,color:#fff
    style C fill:#9b59b6,stroke:#333,color:#fff
```

Four components: **Registry** (file-based peer discovery) → **Ring Context** (shared activity buffer) → **Relay** (WebSocket bridge on :10000) → **Zenoh** (P2P terminal-to-terminal).

---

## Module Callable Pattern

The module itself is callable — no need to access the `devduck` instance:

```python
import devduck
devduck("query")  # Works because of CallableModule metaclass
```

This is achieved by replacing `sys.modules[__name__].__class__` with a custom `CallableModule` that defines `__call__`.

---

## State Management

| Data | Location | Persistence |
|------|----------|-------------|
| Conversation history | `agent.messages` | In-memory (cleared on restart) |
| Shell history | `~/.devduck_history` | Persistent |
| Logs | `/tmp/devduck/logs/devduck.log` | Rotating (10MB × 3) |
| Session recordings | `/tmp/devduck/recordings/` | Persistent ZIP files |
| Mesh registry | `/tmp/devduck/mesh_registry.json` | File-based with TTL |
| Ring context | In-memory (`unified_mesh.py`) | Volatile (100 entries) |
| Scheduler jobs | Disk-persisted | Persistent |
| SQLite memory | Default path | Persistent |
| Knowledge Base | AWS Bedrock | Cloud-persistent |

---

## Framework

Built on [Strands Agents SDK](https://strandsagents.com) — a model-agnostic agent framework with tool use, streaming, and multi-provider support.
