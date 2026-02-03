# 🌙 Ambient Mode - Background Thinking for DevDuck

> **TL;DR**: DevDuck continues working in the background while you're idle, then injects findings into your next query. Like a colleague who keeps researching while you grab coffee.

## Overview

Ambient Mode transforms DevDuck from a reactive assistant into a proactive thinking partner. Instead of waiting for your next command, it continues exploring, researching, and building on the current topic.

**Two Modes Available:**

| Mode | Trigger | Behavior | Use Case |
|------|---------|----------|----------|
| **Standard** | `ambient` | Runs when idle (30s), max 3 iterations | Light exploration, validation |
| **Autonomous** | `auto` | Runs continuously until done or stopped | Deep research, building, complex tasks |

---

## Quick Start

```bash
# Start devduck with ambient mode enabled
DEVDUCK_AMBIENT_MODE=true devduck

# Or toggle in the REPL
🦆 ambient    # Standard mode (idle-triggered)
🦆 auto       # Autonomous mode (continuous)
```

### Using the `ambient_mode` Tool

```python
# Start/stop programmatically
ambient_mode(action="start")                    # Standard mode
ambient_mode(action="start", autonomous=True)   # Autonomous mode
ambient_mode(action="stop")                     # Stop

# Check status
ambient_mode(action="status")

# Configure on the fly
ambient_mode(action="configure", idle_threshold=60, max_iterations=5, cooldown=30)
```

---

## Standard Ambient Mode

### How It Works

```
You: "analyze the security of this codebase"
Agent: [responds with initial analysis]
[You go idle for 30 seconds...]

🌙 [ambient] Thinking... (iteration 1/3)
──────────────────────────────────────────────────────
[Agent explores deeper - checks for vulnerabilities, reviews dependencies...]
──────────────────────────────────────────────────────
🌙 [ambient] Work stored. Will be injected into next query.

[60 seconds later, if still idle...]

🌙 [ambient] Thinking... (iteration 2/3)
...

You: "what did you find?"
🌙 [ambient] Injecting background work into context...
Agent: [responds with enriched context from background work]
```

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DEVDUCK_AMBIENT_MODE` | `false` | Enable ambient mode on startup |
| `DEVDUCK_AMBIENT_IDLE_SECONDS` | `30` | Seconds of idle before triggering |
| `DEVDUCK_AMBIENT_MAX_ITERATIONS` | `3` | Maximum background runs per topic |
| `DEVDUCK_AMBIENT_COOLDOWN` | `60` | Seconds between background runs |

### Best For
- Exploring edge cases while you think
- Validating assumptions
- Finding related information
- Light research tasks

---

## 🚀 Autonomous Mode

### How It Works

Autonomous mode is **fully self-directed**. The agent keeps working continuously until:
1. It signals completion with `[AMBIENT_DONE]`
2. You stop it manually (type `auto` again or `Ctrl+C`)
3. It hits the maximum iteration limit (default: 50)

```
🦆 auto
🌙 Ambient mode started (AUTONOMOUS - runs until stopped or [AMBIENT_DONE])

🦆 build a complete REST API for user management

[Agent starts working immediately...]

🌙 [AUTONOMOUS] Thinking... (iteration 1/50)
──────────────────────────────────────────────────────
[Creates project structure, sets up dependencies...]
──────────────────────────────────────────────────────
🌙 [AUTONOMOUS] Iteration complete. Continuing... (1 stored)

🌙 [AUTONOMOUS] Thinking... (iteration 2/50)
──────────────────────────────────────────────────────
[Implements user model, database schema...]
──────────────────────────────────────────────────────
🌙 [AUTONOMOUS] Iteration complete. Continuing... (2 stored)

... [keeps going] ...

🌙 [AUTONOMOUS] Thinking... (iteration 8/50)
──────────────────────────────────────────────────────
All endpoints implemented and tested. Documentation complete.
[AMBIENT_DONE]
──────────────────────────────────────────────────────
🌙 [AUTONOMOUS] Agent signaled completion. Stopping.
```

### Real-World Test: Counting to 50

Autonomous mode was tested counting iterations - successfully ran **39 iterations** before user stopped it:

```
🌙 [AUTONOMOUS] Thinking... (iteration 1/50)
**1**
🌙 [AUTONOMOUS] Thinking... (iteration 2/50)
**2**
...
🌙 [AUTONOMOUS] Thinking... (iteration 25/50)
**25** 🎯 Halfway!
...
🌙 [AUTONOMOUS] Thinking... (iteration 39/50)
**39**

User: "i think it's enough"
[AMBIENT_DONE]
```

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DEVDUCK_AUTONOMOUS_MAX_ITERATIONS` | `50` | Maximum iterations before auto-stop |
| `DEVDUCK_AUTONOMOUS_COOLDOWN` | `10` | Seconds between iterations (fast!) |

### Completion Signals

The agent can stop autonomous mode by including any of these phrases:

```
[AMBIENT_DONE]
[TASK_COMPLETE]
[NOTHING_MORE_TO_DO]
"I've completed my exploration"
"Nothing more to explore"
```

### Best For
- Building complete features
- Deep research tasks
- Multi-step workflows
- Tasks you can walk away from

---

## Commands Reference

### REPL Commands

| Command | Action |
|---------|--------|
| `ambient` | Toggle standard ambient mode |
| `auto` or `autonomous` | Toggle autonomous mode |
| `status` | Check current mode and iteration count |
| `Ctrl+C` | Interrupt current work (double to exit) |

### `ambient_mode` Tool

| Action | Description | Example |
|--------|-------------|---------|
| `start` | Start ambient/autonomous mode | `ambient_mode(action="start", autonomous=True)` |
| `stop` | Stop ambient mode | `ambient_mode(action="stop")` |
| `status` | Get current status | `ambient_mode(action="status")` |
| `configure` | Update settings | `ambient_mode(action="configure", idle_threshold=60)` |

---

## Programmatic Access

### Via `ambient_mode` Tool (Recommended)

```python
# Start autonomous mode
result = agent.tool.ambient_mode(action="start", autonomous=True)

# Check status
status = agent.tool.ambient_mode(action="status")
# Returns iteration count, mode, stored results, etc.

# Configure runtime settings
agent.tool.ambient_mode(
    action="configure",
    idle_threshold=45,
    max_iterations=10,
    cooldown=30
)

# Stop
agent.tool.ambient_mode(action="stop")
```

### Via Python API

```python
import devduck

# Check ambient status
status = devduck.status()
print(status["ambient_mode"])
# {
#     "enabled": True,
#     "autonomous": True,
#     "idle_threshold": 30.0,
#     "max_iterations": 50,
#     "current_iterations": 39,
#     "has_pending_result": True,
#     "results_stored": 39,
#     "last_query": "count to 50..."
# }

# Direct control
devduck.devduck.ambient.start()                 # Start standard
devduck.devduck.ambient.start(autonomous=True)  # Start autonomous
devduck.devduck.ambient.stop()                  # Stop
```

---

## Dynamic Context Injection

Ambient mode status is automatically injected into the system context:

```
## 🌙 Ambient Mode Status:
- **Enabled**: True
- **Mode**: AUTONOMOUS
- **Iterations**: 38/50
- **Pending Results**: 0 stored
- **Last Query**: so enable the ambient mode to autonomous...
```

This helps the agent understand its current state and accumulated work.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DevDuck REPL                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   User Input ──► Agent Execution ──► Response               │
│        │                                    │               │
│        ▼                                    ▼               │
│   ┌─────────┐                        ┌─────────────┐        │
│   │ Record  │                        │   Inject    │        │
│   │ Query   │                        │   Results   │        │
│   └────┬────┘                        └──────▲──────┘        │
│        │                                    │               │
│        ▼                                    │               │
│   ┌─────────────────────────────────────────┴─────┐         │
│   │              AmbientMode Thread               │         │
│   │                                               │         │
│   │   Standard:                                   │         │
│   │   [Idle 30s] ──► [Prompt] ──► [Run] ──► [Store]        │
│   │                                               │         │
│   │   Autonomous:                                 │         │
│   │   [10s cooldown] ──► [Prompt] ──► [Run] ──► [Store]    │
│   │         ▲                              │      │         │
│   │         └──────── Loop until done ─────┘      │         │
│   │                                               │         │
│   └───────────────────────────────────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Flow

1. **User sends query** → Agent responds → Query recorded in AmbientMode
2. **Idle detected** → Standard waits 30s; Autonomous uses 10s cooldown
3. **Background run** → Agent called with continuation prompt
4. **Result stored** → Accumulated in `ambient_results_history`
5. **User returns** → Stored results injected into next query context
6. **Completion check** → Agent can signal `[AMBIENT_DONE]` to stop

---

## Example Workflows

### 1. Code Review with Background Analysis

```bash
🦆 review the authentication module in src/auth/

# [Agent gives initial review]
# [You start reading the code yourself...]

# Meanwhile, ambient mode:
🌙 [ambient] Checking for common vulnerabilities...
🌙 [ambient] Analyzing dependency security...

🦆 any security concerns?
# [Response includes background findings]
```

### 2. Autonomous Feature Building

```bash
🦆 auto
🦆 generate complete API documentation for this project

# [Walk away, grab coffee...]
# [Agent iterates through all endpoints]
# [Generates OpenAPI spec]
# [Creates markdown docs]
# [Tests examples]

🌙 [AUTONOMOUS] Agent signaled completion. Stopping.

🦆 show me the generated docs
```

### 3. Programmatic Autonomous Task

```python
# Use tool to start autonomous mode for a long task
agent.tool.ambient_mode(action="start", autonomous=True)
result = agent("refactor all deprecated functions in this codebase")
# Agent will continue working autonomously until [AMBIENT_DONE]
```

---

## Cost Considerations

⚠️ **Ambient mode increases API usage** - each iteration is a full agent call.

| Mode | Typical Calls | Use Case |
|------|---------------|----------|
| Standard | 1-3 extra calls per topic | Low-cost exploration |
| Autonomous | 5-50 calls per task | High-value complex tasks |

**Tips to manage costs:**
- Use standard mode for exploration
- Reserve autonomous for tasks worth the investment
- Set lower `MAX_ITERATIONS` if concerned
- Agent can self-terminate with `[AMBIENT_DONE]`
- Use `ambient_mode(action="configure", max_iterations=10)` to limit

---

## Troubleshooting

### Ambient mode not triggering?

1. Check it's enabled: `ambient_mode(action="status")`
2. Verify idle threshold: default is 30s
3. Ensure agent isn't executing: waits for completion
4. Check cooldown: 60s between runs (standard), 10s (autonomous)

### Too many iterations?

```bash
# Reduce max iterations via environment
export DEVDUCK_AMBIENT_MAX_ITERATIONS=2
export DEVDUCK_AUTONOMOUS_MAX_ITERATIONS=10

# Or at runtime
ambient_mode(action="configure", max_iterations=5)
```

### Want faster autonomous cycling?

```bash
# Reduce cooldown (careful with costs!)
export DEVDUCK_AUTONOMOUS_COOLDOWN=5

# Or at runtime
ambient_mode(action="configure", cooldown=5)
```

---

## FAQ

**Q: Does ambient mode work in non-interactive mode?**
A: No, designed for REPL. For batch processing, use regular agent calls.

**Q: Can I see what ambient mode is thinking?**
A: Yes! All output streams to terminal with 🌙 prefix.

**Q: Does typing interrupt ambient work?**
A: Yes, gracefully. Current iteration completes but results may be discarded.

**Q: Is the background work saved if I exit?**
A: No, results are in-memory. They're injected into next query or lost on exit.

**Q: Can I control ambient mode from another tool?**
A: Yes! Use `ambient_mode(action="start", autonomous=True)` from any tool.

**Q: What's the difference between `ambient` tool and `ambient_mode` tool?**
A: `ambient` is the GUI overlay for input. `ambient_mode` controls background thinking.

---

## Related Features

- **Zenoh P2P**: Multiple DevDucks can discover and collaborate
- **Hot Reload**: Code changes apply without losing ambient state
- **Knowledge Base**: Ambient findings can be persisted to Bedrock KB
- **`ambient_mode` Tool**: Programmatic control over ambient behavior

---

*🦆 DevDuck - Your proactive AI pair programmer*
