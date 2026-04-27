"""🦆 Welcome - Manage DevDuck's dynamic welcome screen.

The welcome text is NOT static — it lives in `$CWD/.welcome` and can be
edited by the agent itself, persisted across sessions, and used as a
shared scratchpad between concurrent DevDuck instances.

Supports `.hushlogin` (Unix convention):
    - If `$CWD/.hushlogin` or `$HOME/.hushlogin` exists → suppress welcome
    - Or set env var `DEVDUCK_HUSHLOGIN=true`

Usage by the agent:
    welcome(action="view")                  # read current welcome
    welcome(action="edit", content="...")   # persist new welcome
    welcome(action="reset")                 # back to default
    welcome(action="hush")                  # create .hushlogin in cwd
    welcome(action="unhush")                # remove .hushlogin in cwd
    welcome(action="status")                # show hushlogin / welcome state
"""

import os
from pathlib import Path
from typing import Any, Dict

from strands import tool


# ─── Default welcome (used when no .welcome file exists) ─────────
DEFAULT_WELCOME_TEXT = """# 🦆 DevDuck — Self-Modifying AI Agent

**One file. Self-healing. Runtime dependencies. Adaptive.**

## Quick Start
- Type natural language → agent executes
- Prefix `!` for shell commands (`! ls -la`)
- `ambient` / `auto` — background / autonomous thinking
- `record` — toggle session recording
- `exit` / `q` — quit

## Self-Awareness
I have full access to my own source code at `/Users/cagatay/devduck/devduck/__init__.py`.
I can modify myself — edits hot-reload instantly via `os.execv()`.

## Extending Me
- Drop a `.py` file into `./tools/` → instantly available as a tool
- `manage_tools(action="fetch", url=...)` → load tools from GitHub
- `system_prompt(action="add_context", context="...")` → teach me something new

## Customize This Welcome
Edit `$CWD/.welcome` or call `welcome(action="edit", content="...")`.
Create `.hushlogin` to silence the welcome screen entirely.
"""


def _welcome_path() -> Path:
    return Path.cwd() / ".welcome"


def _hushlogin_paths() -> list:
    """Return locations where .hushlogin is checked (in priority order)."""
    return [
        Path.cwd() / ".hushlogin",
        Path.home() / ".hushlogin",
    ]


def is_hushed() -> bool:
    """True if welcome should be suppressed.

    Honored signals (any of these):
        - Env var DEVDUCK_HUSHLOGIN=true/1/yes
        - $CWD/.hushlogin exists
        - $HOME/.hushlogin exists
    """
    env_val = os.getenv("DEVDUCK_HUSHLOGIN", "").strip().lower()
    if env_val in ("true", "1", "yes", "on"):
        return True
    for p in _hushlogin_paths():
        if p.exists():
            return True
    return False


def get_welcome_text() -> str:
    """Get the current welcome text (from .welcome file or default).

    Called by landing.py / tui.py to inject dynamic content into the UI.
    """
    path = _welcome_path()
    try:
        if path.exists() and path.is_file():
            content = path.read_text(encoding="utf-8", errors="ignore").strip()
            if content:
                return content
    except Exception:
        pass
    return DEFAULT_WELCOME_TEXT


def has_custom_welcome() -> bool:
    """True if a custom .welcome file exists (non-empty)."""
    path = _welcome_path()
    try:
        return path.exists() and path.is_file() and bool(path.read_text(encoding="utf-8", errors="ignore").strip())
    except Exception:
        return False


@tool
def welcome(action: str = "view", content: str = None) -> Dict[str, Any]:
    """🦆 Manage DevDuck's dynamic welcome screen and hushlogin state.

    The welcome screen is NOT static — it's sourced from `$CWD/.welcome` and
    can be modified by me at any time. Use `.hushlogin` to disable it
    (Unix convention, honored on cwd or home dir).

    Also works as a shared scratchpad for coordination between concurrent
    DevDuck instances running in the same directory.

    Args:
        action: One of:
            - "view":   Show current welcome text
            - "edit":   Persist new welcome text to $CWD/.welcome (requires content)
            - "reset":  Delete $CWD/.welcome → revert to default
            - "hush":   Create $CWD/.hushlogin to silence welcome on next start
            - "unhush": Remove $CWD/.hushlogin
            - "status": Show hushlogin state, welcome file location, preview
        content: New welcome content (required for edit)

    Returns:
        Dict with status + content
    """
    try:
        path = _welcome_path()

        if action == "view":
            text = get_welcome_text()
            source = "custom (.welcome)" if has_custom_welcome() else "default"
            return {
                "status": "success",
                "content": [{"text": f"📜 Welcome source: {source}\nPath: {path}\n\n{text}"}],
            }

        elif action == "edit":
            if not content:
                return {
                    "status": "error",
                    "content": [{"text": "content is required for edit action"}],
                }
            path.write_text(content, encoding="utf-8")
            return {
                "status": "success",
                "content": [{"text": f"✅ Welcome updated: {path} ({len(content)} chars)"}],
            }

        elif action == "reset":
            if path.exists():
                path.unlink()
                return {
                    "status": "success",
                    "content": [{"text": f"✅ Deleted {path} — reverted to default welcome"}],
                }
            return {
                "status": "success",
                "content": [{"text": "No custom welcome — already using default"}],
            }

        elif action == "hush":
            hush_path = Path.cwd() / ".hushlogin"
            hush_path.touch(exist_ok=True)
            return {
                "status": "success",
                "content": [{"text": f"🤫 Created {hush_path} — welcome will be silent in this dir"}],
            }

        elif action == "unhush":
            removed = []
            for p in _hushlogin_paths():
                if p.exists():
                    try:
                        p.unlink()
                        removed.append(str(p))
                    except Exception as e:
                        return {
                            "status": "error",
                            "content": [{"text": f"Could not remove {p}: {e}"}],
                        }
            if removed:
                return {
                    "status": "success",
                    "content": [{"text": f"🔊 Removed: {', '.join(removed)}"}],
                }
            return {
                "status": "success",
                "content": [{"text": "No .hushlogin files to remove"}],
            }

        elif action == "status":
            hushed = is_hushed()
            hush_hits = [str(p) for p in _hushlogin_paths() if p.exists()]
            env_hush = os.getenv("DEVDUCK_HUSHLOGIN", "")
            custom = has_custom_welcome()
            preview = get_welcome_text()[:200]

            lines = [
                f"🤫 Hushed: {hushed}",
                f"   .hushlogin locations: {hush_hits or 'none'}",
                f"   DEVDUCK_HUSHLOGIN env: {env_hush or '(unset)'}",
                "",
                f"📜 Welcome source: {'custom (.welcome)' if custom else 'default'}",
                f"   Path: {path}",
                f"   Exists: {path.exists()}",
                "",
                "Preview:",
                preview + ("..." if len(get_welcome_text()) > 200 else ""),
            ]
            return {
                "status": "success",
                "content": [{"text": "\n".join(lines)}],
            }

        else:
            return {
                "status": "error",
                "content": [{"text": f"Unknown action: {action}. Valid: view, edit, reset, hush, unhush, status"}],
            }

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"Error: {e}"}],
        }
