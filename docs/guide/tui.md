# TUI Mode

Multi-conversation terminal UI with concurrent panels, streaming markdown, and shared awareness.

---

## Launch

```bash
devduck --tui
```

!!! note "Requires `textual`"
    ```bash
    pip install textual
    ```

---

## Concurrency Model

```mermaid
graph TB
    subgraph SharedMessages["📋 SharedMessages (thread-safe)"]
        msgs["msg1, msg2, msg3, msg4, ..."]
    end

    SharedMessages --> A1
    SharedMessages --> A2
    SharedMessages --> A3

    subgraph A1["🟦 Agent #1"]
        cb1["callback → panel #1"]
    end
    subgraph A2["🟩 Agent #2"]
        cb2["callback → panel #2"]
    end
    subgraph A3["🟨 Agent #3"]
        cb3["callback → panel #3"]
    end

    style SharedMessages fill:#e74c3c,stroke:#333,color:#fff
    style A1 fill:#3498db,stroke:#333,color:#fff
    style A2 fill:#2ecc71,stroke:#333,color:#fff
    style A3 fill:#f1c40f,stroke:#333,color:#000
```

Each conversation creates a **fresh Agent** instance, but all agents point their `.messages` at a single `SharedMessages` instance — a thread-safe list subclass that serializes all reads and writes via a lock.

### What this gives you

- **True concurrency** — separate Agent instances with separate callback handlers, no conflicts
- **Real-time shared awareness** — when Agent #1 appends a message, Agent #2 sees it immediately
- **Correct ordering** — the lock ensures messages are appended in order
- **Isolated rendering** — each agent's callback routes streaming output to its own color-coded panel

### Message cap

Shared history is capped at 100 messages (configurable via `DEVDUCK_TUI_MAX_SHARED_MESSAGES`) and auto-clears on context window overflow.

---

## Features

| Feature | Description |
|---------|-------------|
| **Multiple conversations** | Run several conversations concurrently in separate panels |
| **Streaming markdown** | Rich formatted output as the agent responds |
| **Interleaved execution** | Conversations run in parallel, not blocking each other |
| **Full tool access** | Each conversation has the complete tool set |
| **Shared context** | All conversations share awareness via `SharedMessages` |
| **Mesh integration** | Each TUI conversation pushes to the unified ring context |

---

## Comparison Across Interfaces

```mermaid
graph LR
    subgraph CLI["CLI · single thread"]
        C1["🦆 One Agent<br/>sequential"]
    end

    subgraph TUI_["TUI · concurrent"]
        T1["🟦 Agent 1"]
        T2["🟩 Agent 2"]
        T3["🟨 Agent 3"]
        SM["📋 SharedMessages"]
        T1 <--> SM
        T2 <--> SM
        T3 <--> SM
    end

    subgraph External["TCP / Telegram / WS · isolated"]
        E1["🦆 Fresh DevDuck"]
        E2["🦆 Fresh DevDuck"]
    end

    style CLI fill:#f5a623,stroke:#333,color:#000
    style TUI_ fill:#3498db,stroke:#333,color:#fff
    style External fill:#9b59b6,stroke:#333,color:#fff
```

| Interface | Agent per request | Shared messages | Use case |
|-----------|:-:|:-:|---|
| **CLI** | No (reuse one) | N/A (single-threaded) | Sequential interactive REPL |
| **TUI** | Yes (fresh Agent) | Yes (`SharedMessages`) | Concurrent conversations with shared context |
| **TCP** | Yes (fresh DevDuck) | No (fully isolated) | External network clients |
| **Telegram** | Yes (fresh DevDuck) | No (fully isolated) | Chat bot, each user isolated |
| **WebSocket** | Yes (fresh DevDuck) | No (fully isolated) | Browser clients |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVDUCK_TUI_MAX_SHARED_MESSAGES` | `100` | Max shared message history |
