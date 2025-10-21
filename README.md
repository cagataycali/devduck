# 🦆 DevDuck

**One file. Self-healing. Adaptive.**

Minimalist AI agent that fixes itself when things break.

## Install

```bash
pipx install devduck
```

Requires: Python 3.10+, Ollama running

## Use

```bash
# CLI
devduck "what's the time?"

# Interactive
devduck

# Python
import devduck

devduck("calculate 2+2")
```

## Features

- **Self-healing** - Auto-fixes deps, models, errors
- **Hot-reload** - Create tools in `./tools/*.py`, use instantly
- **Adaptive** - Picks model based on OS (macOS: 1.7b, Linux: 30b)
- **14 tools** - shell, editor, files, python, calculator, tcp, etc.
- **History aware** - Remembers shell/conversation context

## Create Tool

```python
# ./tools/greet.py
from strands import tool

@tool
def greet(name: str) -> str:
    return f"Hello {name}!"
```

Save. Done. Use immediately.

## Multi-Model

```bash
export MODEL_PROVIDER="bedrock"
export STRANDS_MODEL_ID="us.anthropic.claude-sonnet-4-5-20250929-v1:0"
export STRANDS_ADDITIONAL_REQUEST_FIELDS='{"anthropic_beta": ["interleaved-thinking-2025-05-14", "context-1m-2025-08-07"], "thinking": {"type": "enabled", "budget_tokens": 2048}}'
export STRANDS_MAX_TOKENS="64000"

devduck "analyze data"
```

---

**Quack.** 🦆

*Built with [Strands Agents SDK](https://github.com/strands-agents/sdk-python)*
