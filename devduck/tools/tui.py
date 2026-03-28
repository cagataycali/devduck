"""
🖥️ TUI Tool — Let the agent dynamically push content to the TUI.

The agent can write markdown panels, tables, notifications, images,
and custom Rich renderables into the running TUI from any conversation.
"""

import sys
import time
from typing import Any, Dict, List, Optional

from strands import tool


# ─── Global TUI app reference (set by tui.py on startup) ────────
_tui_app = None


def set_tui_app(app):
    """Register the running TUI app instance for tool access."""
    global _tui_app
    _tui_app = app


def get_tui_app():
    """Get the current TUI app if running."""
    return _tui_app


@tool
def tui(
    action: str,
    content: str = "",
    title: str = "",
    style: str = "cyan",
    markdown: bool = True,
    conv_id: int = 0,
) -> Dict[str, Any]:
    """
    🖥️ Push dynamic content to the DevDuck TUI.

    Use this to render rich panels, markdown, notifications, and status
    updates directly in the TUI interface. Works from any conversation.

    Args:
        action: Action to perform:
            - "panel": Render a Rich panel with content (markdown or plain)
            - "notify": Show a transient notification toast
            - "markdown": Render raw markdown into the conversation log
            - "image": Render an image file as halfblock pixels in the TUI
            - "status": Update the status bar text
            - "clear_done": Clear completed conversation panels
            - "info": Get current TUI state (active conversations, peers, etc.)
        content: Text content (supports markdown when markdown=True).
            For "image" action: file path to image (png, jpg, gif, bmp, webp) or URL.
        title: Panel title (for "panel" and "image" actions)
        style: Border/accent style color (e.g. "cyan", "green", "red", "#61afef")
        markdown: Whether to render content as Markdown (default: True)
        conv_id: Target conversation ID (0 = create a new standalone panel)

    Returns:
        Dict with status and result info

    Examples:
        # Render a markdown report panel
        tui(action="panel", content="## Results\\n- Item 1\\n- Item 2", title="Report")

        # Show a notification toast
        tui(action="notify", content="Build complete! ✅")

        # Render an image in the TUI
        tui(action="image", content="/path/to/screenshot.png", title="Screenshot")

        # Render image into a specific conversation
        tui(action="image", content="./chart.png", conv_id=3)

        # Push markdown into the current conversation
        tui(action="markdown", content="| Col1 | Col2 |\\n|------|------|\\n| a | b |", conv_id=3)

        # Get TUI state
        tui(action="info")
    """
    try:
        app = get_tui_app()

        if app is None:
            return {
                "status": "error",
                "content": [{"text": "TUI is not running. Use `devduck --tui` to start TUI mode."}],
            }

        if action == "panel":
            return _action_panel(app, content, title, style, markdown)
        elif action == "notify":
            return _action_notify(app, content, title, style)
        elif action == "markdown":
            return _action_markdown(app, content, conv_id)
        elif action == "image":
            return _action_image(app, content, title, style, conv_id)
        elif action == "status":
            return _action_status(app, content)
        elif action == "clear_done":
            app.action_clear_done()
            return {
                "status": "success",
                "content": [{"text": "Cleared completed conversations."}],
            }
        elif action == "info":
            return _action_info(app)
        else:
            return {
                "status": "error",
                "content": [{"text": f"Unknown action: {action}. Valid: panel, notify, markdown, image, status, clear_done, info"}],
            }

    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"TUI tool error: {e}"}],
        }


# ─── Action implementations ─────────────────────────────────────

def _action_panel(app, content: str, title: str, style: str, use_markdown: bool) -> Dict:
    """Render a standalone panel in the TUI."""
    from textual.widgets import Static
    from textual.containers import ScrollableContainer
    from textual.css.query import NoMatches
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.text import Text

    try:
        scroll = app.query_one("#conversations-scroll", ScrollableContainer)
    except NoMatches:
        return {"status": "error", "content": [{"text": "TUI scroll area not found"}]}

    if use_markdown and content:
        renderable = Markdown(content)
    else:
        renderable = Text(content)

    panel_widget = Static(
        Panel(
            renderable,
            title=title or "🦆 Agent Output",
            border_style=style,
            padding=(1, 2),
        )
    )

    app.call_from_thread(scroll.mount, panel_widget)
    app.call_from_thread(scroll.scroll_end, animate=False)

    return {
        "status": "success",
        "content": [{"text": f"Panel rendered: {title or 'Agent Output'} ({len(content)} chars)"}],
    }


def _action_notify(app, content: str, title: str, style: str) -> Dict:
    """Show a notification toast."""
    severity = "information"
    if style in ("red", "error"):
        severity = "error"
    elif style in ("yellow", "warning"):
        severity = "warning"

    app.call_from_thread(app.notify, content, title=title or "🦆 DevDuck", severity=severity)

    return {
        "status": "success",
        "content": [{"text": f"Notification shown: {content[:100]}"}],
    }


def _action_markdown(app, content: str, conv_id: int) -> Dict:
    """Render markdown into a specific conversation or as standalone."""
    from textual.widgets import Static, RichLog
    from textual.containers import ScrollableContainer
    from textual.css.query import NoMatches
    from rich.markdown import Markdown

    if conv_id > 0:
        # Push into existing conversation
        panel = app._active_conversations.get(conv_id)
        if not panel:
            return {
                "status": "error",
                "content": [{"text": f"Conversation #{conv_id} not found. Active: {list(app._active_conversations.keys())}"}],
            }

        try:
            log = panel.query_one(f"#log-{conv_id}", RichLog)
            app.call_from_thread(log.write, Markdown(content))
            return {
                "status": "success",
                "content": [{"text": f"Markdown rendered in conversation #{conv_id}"}],
            }
        except NoMatches:
            return {"status": "error", "content": [{"text": f"Log widget not found for #{conv_id}"}]}
    else:
        # Standalone markdown panel
        return _action_panel(app, content, "", "dim", True)


def _action_status(app, content: str) -> Dict:
    """Update the TUI status bar."""
    from textual.widgets import Static
    from textual.css.query import NoMatches
    from rich.text import Text

    try:
        bar = app.query_one("#status-bar", Static)
        t = Text()
        t.append(" 🦆 ", style="bold bright_yellow")
        t.append(content, style="bold")
        app.call_from_thread(bar.update, t)
        return {
            "status": "success",
            "content": [{"text": f"Status bar updated: {content[:100]}"}],
        }
    except NoMatches:
        return {"status": "error", "content": [{"text": "Status bar not found"}]}


def _action_info(app) -> Dict:
    """Get current TUI state."""
    active = []
    done = []
    for cid, panel in app._active_conversations.items():
        entry = {"id": cid, "query": panel.query[:80], "color": panel.color}
        if panel.is_done:
            done.append(entry)
        else:
            active.append(entry)

    peer_count = 0
    try:
        _zp_mod = sys.modules.get("devduck.tools.zenoh_peer")
        if _zp_mod:
            peer_count = len(_zp_mod.ZENOH_STATE.get("peers", {}))
    except Exception:
        pass

    info = {
        "tui_running": True,
        "total_queries": app._total_queries,
        "active_conversations": active,
        "done_conversations": len(done),
        "zenoh_peers": peer_count,
    }

    import json
    return {
        "status": "success",
        "content": [{"text": json.dumps(info, indent=2)}],
    }


# ─── Image rendering ────────────────────────────────────────────

def _load_image_bytes(path_or_url: str) -> tuple:
    """Load image bytes from a file path or URL.

    Returns:
        (image_bytes, source_label, error_message)
    """
    import os

    path_or_url = path_or_url.strip()

    # URL
    if path_or_url.startswith(("http://", "https://")):
        try:
            import urllib.request
            with urllib.request.urlopen(path_or_url, timeout=15) as resp:
                data = resp.read()
            return data, path_or_url.split("/")[-1].split("?")[0][:40], None
        except Exception as e:
            return None, None, f"Failed to fetch URL: {e}"

    # File path
    expanded = os.path.expanduser(path_or_url)
    if not os.path.isabs(expanded):
        expanded = os.path.abspath(expanded)

    if not os.path.exists(expanded):
        return None, None, f"File not found: {expanded}"

    try:
        with open(expanded, "rb") as f:
            data = f.read()
        label = os.path.basename(expanded)
        return data, label, None
    except Exception as e:
        return None, None, f"Failed to read file: {e}"


def _action_image(app, content: str, title: str, style: str, conv_id: int) -> Dict:
    """Render an image as halfblock pixels in the TUI.

    Supports: PNG, JPEG, GIF, BMP, WebP, TIFF — files and URLs.
    Uses HalfblockRenderer from the browse tool, with a standalone fallback.
    """
    from textual.widgets import Static, RichLog
    from textual.containers import ScrollableContainer
    from textual.css.query import NoMatches
    from rich.panel import Panel
    from rich.text import Text

    if not content:
        return {"status": "error", "content": [{"text": "No image path provided. Use content='/path/to/image.png'"}]}

    # Load image bytes
    image_bytes, source_label, error = _load_image_bytes(content)
    if error:
        return {"status": "error", "content": [{"text": error}]}

    # Determine render size from TUI dimensions
    try:
        size = app.size
        render_width = min(size.width - 6, 200)
        # Each halfblock row = 2 pixel rows, leave room for panel chrome
        render_height = min((size.height - 6) * 2, 120)
    except Exception:
        render_width = 120
        render_height = 60

    # Ensure even height
    if render_height % 2 != 0:
        render_height += 1

    # Render via HalfblockRenderer (try browse module first, then fallback)
    try:
        from devduck.tools.browse import HalfblockRenderer
        rich_text = HalfblockRenderer.render_to_rich_text(image_bytes, render_width, render_height)
    except ImportError:
        rich_text = _fallback_render_image(image_bytes, render_width, render_height)
        if rich_text is None:
            return {"status": "error", "content": [{"text": "Pillow not installed. pip install Pillow"}]}

    # Get image metadata for the title
    img_info = ""
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        img_info = f" ({img.size[0]}×{img.size[1]} {img.mode})"
    except Exception:
        pass

    panel_title = title or f"🖼️ {source_label}{img_info}"

    if conv_id > 0:
        # Render into existing conversation
        panel = app._active_conversations.get(conv_id)
        if not panel:
            return {
                "status": "error",
                "content": [{"text": f"Conversation #{conv_id} not found."}],
            }
        try:
            log = panel.query_one(f"#log-{conv_id}", RichLog)
            header = Text()
            header.append(f"\n  {panel_title}\n", style=f"bold {style}")
            app.call_from_thread(log.write, header)
            app.call_from_thread(log.write, rich_text)
            return {
                "status": "success",
                "content": [{"text": f"Image rendered in conversation #{conv_id}: {source_label}{img_info}"}],
            }
        except NoMatches:
            return {"status": "error", "content": [{"text": f"Log widget not found for #{conv_id}"}]}
    else:
        # Standalone image panel
        try:
            scroll = app.query_one("#conversations-scroll", ScrollableContainer)
        except NoMatches:
            return {"status": "error", "content": [{"text": "TUI scroll area not found"}]}

        image_panel = Static(
            Panel(
                rich_text,
                title=panel_title,
                border_style=style,
                padding=(0, 1),
            )
        )
        app.call_from_thread(scroll.mount, image_panel)
        app.call_from_thread(scroll.scroll_end, animate=False)

        return {
            "status": "success",
            "content": [{"text": f"Image rendered: {source_label}{img_info} ({render_width}×{render_height // 2} cells)"}],
        }


def _fallback_render_image(image_bytes: bytes, width: int, height: int):
    """Standalone halfblock renderer — used when browse tool is not available.

    Converts a PIL image to Rich Text using ▀ (upper half block) characters
    with foreground = top pixel color, background = bottom pixel color.
    This gives 2 vertical pixels per terminal row.
    """
    try:
        from PIL import Image
        from rich.text import Text
        from rich.style import Style
        from rich.color import Color
        import io
    except ImportError:
        return None

    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    if height % 2 != 0:
        height += 1
    img = img.resize((width, height), Image.LANCZOS)
    pixels = img.load()

    _style_cache = {}
    text = Text()

    for y in range(0, height, 2):
        prev_key = None
        run = 0

        for x in range(width):
            top = pixels[x, y]
            bottom = pixels[x, y + 1] if y + 1 < height else (0, 0, 0)
            key = (top[0], top[1], top[2], bottom[0], bottom[1], bottom[2])

            if key == prev_key:
                run += 1
            else:
                if run > 0 and prev_key is not None:
                    s = _style_cache.get(prev_key)
                    if s is None:
                        fg = Color.from_rgb(prev_key[0], prev_key[1], prev_key[2])
                        bg = Color.from_rgb(prev_key[3], prev_key[4], prev_key[5])
                        s = Style(color=fg, bgcolor=bg)
                        _style_cache[prev_key] = s
                    text.append("▀" * run, style=s)
                prev_key = key
                run = 1

        # Flush last run on the line
        if run > 0 and prev_key is not None:
            s = _style_cache.get(prev_key)
            if s is None:
                fg = Color.from_rgb(prev_key[0], prev_key[1], prev_key[2])
                bg = Color.from_rgb(prev_key[3], prev_key[4], prev_key[5])
                s = Style(color=fg, bgcolor=bg)
                _style_cache[prev_key] = s
            text.append("▀" * run, style=s)

        text.append("\n")

    return text
