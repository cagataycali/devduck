"""
🦆 DevDuck Landing UI - Beautiful terminal dashboard
"""

import os
import sys
import time
import socket
import platform
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich.align import Align
from rich.markdown import Markdown
from rich import box

# Dynamic welcome — NOT static! Honors $CWD/.welcome + .hushlogin
try:
    from devduck.tools.welcome import get_welcome_text, is_hushed, has_custom_welcome
except Exception:
    def get_welcome_text() -> str:
        return ""
    def is_hushed() -> bool:
        return False
    def has_custom_welcome() -> bool:
        return False


console = Console()

DUCK_ART = r"""
     __
   <(o )___
    ( ._> /
     `---'
"""

DUCK_ART_BIG = r"""
        .__
     __/  .\
   <(o  )___\
    ( ._>   /
     `----'`
"""


def _get_gradient_duck():
    """Create an animated-looking gradient duck."""
    lines = [
        "        .__       ",
        "     __/  .\\     ",
        "   <(o  )___\\   ",
        "    ( ._>   /    ",
        "     `----'`     ",
    ]
    colors = ["bright_yellow", "yellow", "bright_yellow", "yellow", "bright_yellow"]
    text = Text()
    for i, line in enumerate(lines):
        text.append(line + "\n", style=colors[i])
    return text


def _status_dot(ok: bool) -> str:
    return "[green]●[/green]" if ok else "[dim]○[/dim]"


def render_landing(devduck_instance):
    """Render the full landing dashboard.

    Honors `.hushlogin` (Unix convention): if `$CWD/.hushlogin` or
    `$HOME/.hushlogin` exists, or `DEVDUCK_HUSHLOGIN=true`, the landing
    screen is suppressed entirely — only a single-line status is printed.

    The welcome content itself is dynamic — sourced from `$CWD/.welcome`
    via `devduck.tools.welcome.get_welcome_text()`. There is no static
    welcome message anywhere.
    """
    from devduck import LOG_DIR, get_session_recorder

    # 🤫 Hushlogin → minimal output, skip the big dashboard
    if is_hushed():
        console.print(
            f"[dim]🦆 DevDuck ready · [/][bright_yellow]{getattr(devduck_instance, 'model', '?')}[/]"
            f"[dim] · {len(getattr(devduck_instance, 'tools', []))} tools · type 'exit' to quit[/]"
        )
        return

    console.clear()

    # ── Header ──────────────────────────────────────────────────
    duck_text = _get_gradient_duck()
    title_text = Text()
    title_text.append("D", style="bold bright_yellow")
    title_text.append("ev", style="bold white")
    title_text.append("D", style="bold bright_yellow")
    title_text.append("uck", style="bold white")
    title_text.append("  ", style="")
    title_text.append("v", style="dim")
    try:
        from importlib.metadata import version as pkg_version
        ver = pkg_version("devduck")
    except Exception:
        ver = "dev"
    title_text.append(ver, style="dim")
    title_text.append("\n", style="")
    title_text.append("Self-modifying AI agent", style="dim italic")

    header_table = Table.grid(padding=(0, 2))
    header_table.add_row(duck_text, title_text)
    header_panel = Panel(
        Align.center(header_table),
        border_style="bright_yellow",
        box=box.DOUBLE_EDGE,
        padding=(0, 1),
    )
    console.print(header_panel)

    # ── Info Cards Row ──────────────────────────────────────────
    cards = []

    # Model card
    model_name = getattr(devduck_instance, "model", "unknown")
    # Shorten long model names
    model_display = str(model_name)
    if len(model_display) > 35:
        model_display = "…" + model_display[-34:]
    model_card = Panel(
        f"[bold bright_cyan]{model_display}[/]",
        title="[bold]🧠 Model[/]",
        border_style="cyan",
        box=box.ROUNDED,
        expand=True,
    )
    cards.append(model_card)

    # Environment card
    env = getattr(devduck_instance, "env_info", {})
    os_str = f"{env.get('os', '?')} {env.get('arch', '')}"
    py_info = env.get("python", sys.version_info)
    py_str = f"{py_info.major}.{py_info.minor}.{py_info.micro}" if hasattr(py_info, "major") else str(py_info)
    env_card = Panel(
        f"[bold]{os_str}[/]\nPython {py_str}",
        title="[bold]💻 Environment[/]",
        border_style="green",
        box=box.ROUNDED,
        expand=True,
    )
    cards.append(env_card)

    # Tools card
    tool_count = len(devduck_instance.tools) if hasattr(devduck_instance, "tools") else 0
    tools_card = Panel(
        f"[bold bright_green]{tool_count}[/] [dim]tools loaded[/]",
        title="[bold]🛠️  Tools[/]",
        border_style="bright_green",
        box=box.ROUNDED,
        expand=True,
    )
    cards.append(tools_card)

    console.print(Columns(cards, equal=True, expand=True))

    # ── Services Status ─────────────────────────────────────────
    svc_table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold",
        expand=True,
        padding=(0, 1),
    )
    svc_table.add_column("Service", style="bold", min_width=12)
    svc_table.add_column("Status", justify="center", min_width=6)
    svc_table.add_column("Endpoint", style="dim")
    svc_table.add_column("Info", style="dim italic")

    servers = getattr(devduck_instance, "servers", {})

    # Zenoh
    zenoh_enabled = servers.get("zenoh_peer", {}).get("enabled", False)
    zenoh_id = ""
    zenoh_peers = 0
    if zenoh_enabled:
        try:
            import sys as _sys
            _zp_mod = _sys.modules.get("devduck.tools.zenoh_peer")
            if _zp_mod:
                ZENOH_STATE = _zp_mod.ZENOH_STATE
                zenoh_id = ZENOH_STATE.get("instance_id", "")
                zenoh_peers = len(ZENOH_STATE.get("peers", {}))
        except Exception:
            pass
    svc_table.add_row(
        "Zenoh P2P",
        _status_dot(zenoh_enabled),
        f"[bright_magenta]{zenoh_id}[/]" if zenoh_id else "—",
        f"{zenoh_peers} peer(s)" if zenoh_enabled else "",
    )

    # Mesh Relay / AgentCore Proxy
    ac_cfg = servers.get("agentcore_proxy", {})
    ac_enabled = ac_cfg.get("enabled", False)
    ac_port = ac_cfg.get("port", 10000)
    svc_table.add_row(
        "Mesh Relay",
        _status_dot(ac_enabled),
        f"ws://localhost:{ac_port}" if ac_enabled else "—",
        "browser + cloud agents",
    )

    # WebSocket
    ws_cfg = servers.get("ws", {})
    ws_enabled = ws_cfg.get("enabled", False)
    ws_port = ws_cfg.get("port", 10001)
    svc_table.add_row(
        "WebSocket",
        _status_dot(ws_enabled),
        f"ws://localhost:{ws_port}" if ws_enabled else "—",
        "per-message streaming",
    )

    # TCP
    tcp_cfg = servers.get("tcp", {})
    tcp_enabled = tcp_cfg.get("enabled", False)
    tcp_port = tcp_cfg.get("port", 10002)
    svc_table.add_row(
        "TCP",
        _status_dot(tcp_enabled),
        f"localhost:{tcp_port}" if tcp_enabled else "—",
        "",
    )

    # MCP
    mcp_cfg = servers.get("mcp", {})
    mcp_enabled = mcp_cfg.get("enabled", False)
    mcp_port = mcp_cfg.get("port", 10003)
    svc_table.add_row(
        "MCP",
        _status_dot(mcp_enabled),
        f"http://localhost:{mcp_port}/mcp" if mcp_enabled else "—",
        "",
    )

    # IPC
    ipc_cfg = servers.get("ipc", {})
    ipc_enabled = ipc_cfg.get("enabled", False)
    ipc_path = ipc_cfg.get("socket_path", "/tmp/devduck_main.sock")
    svc_table.add_row(
        "IPC",
        _status_dot(ipc_enabled),
        ipc_path if ipc_enabled else "—",
        "",
    )

    svc_panel = Panel(
        svc_table,
        title="[bold]⚡ Services[/]",
        border_style="bright_blue",
        box=box.ROUNDED,
    )
    console.print(svc_panel)

    # ── Features Row ────────────────────────────────────────────
    # Ambient mode
    ambient = getattr(devduck_instance, "ambient", None)
    ambient_on = ambient and ambient.running
    ambient_mode = "AUTONOMOUS" if (ambient and ambient.autonomous) else "standard"

    # Recording
    recorder = get_session_recorder()
    recording = recorder and recorder.recording

    feat_parts = []

    # Ambient
    if ambient_on:
        feat_parts.append(f"[bright_yellow]🌙 Ambient[/]: [green]ON[/] ({ambient_mode})")
    else:
        feat_parts.append("[dim]🌙 Ambient: OFF[/]  [dim italic]type 'ambient' to enable[/]")

    # Recording
    if recording:
        feat_parts.append(f"[red]🎬 Recording[/]: [green]ON[/] ({recorder.session_id})")
    else:
        feat_parts.append("[dim]🎬 Recording: OFF[/]  [dim italic]type 'record' to start[/]")

    # Hot-reload
    watcher = hasattr(devduck_instance, "_watcher_running") and devduck_instance._watcher_running
    if watcher:
        feat_parts.append("[green]🔥 Hot-Reload[/]: [green]watching[/]")
    else:
        feat_parts.append("[dim]🔥 Hot-Reload: OFF[/]")

    feat_table = Table.grid(padding=(0, 3))
    feat_table.add_row(*feat_parts)
    console.print(Panel(
        Align.center(feat_table),
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
    ))

    # ── Quick Reference ─────────────────────────────────────────
    help_table = Table(
        box=None,
        show_header=False,
        padding=(0, 2),
        expand=True,
    )
    help_table.add_column("Key", style="bold bright_yellow", min_width=18)
    help_table.add_column("Description", style="dim")

    help_table.add_row("  [bold]ask anything[/]", "natural language → agent executes")
    help_table.add_row("  [bold bright_cyan]![/]command", "run shell command directly")
    help_table.add_row("  [bold bright_cyan]ambient[/]", "toggle background thinking")
    help_table.add_row("  [bold bright_cyan]auto[/]", "autonomous mode (runs until done)")
    help_table.add_row("  [bold bright_cyan]record[/]", "toggle session recording")
    help_table.add_row("  [bold bright_cyan]exit[/] / [bold bright_cyan]q[/]", "quit  ·  [dim]Ctrl+C twice to force[/]")

    console.print(Panel(
        help_table,
        title="[bold]⌨️  Commands[/]",
        border_style="dim yellow",
        box=box.ROUNDED,
        padding=(0, 0),
    ))

    # ── Dynamic Welcome (from $CWD/.welcome or default) ────────
    try:
        welcome_text = get_welcome_text()
        if welcome_text:
            try:
                welcome_body = Markdown(welcome_text)
            except Exception:
                welcome_body = Text(welcome_text)

            subtitle = "[dim italic]custom · $CWD/.welcome[/]" if has_custom_welcome() else "[dim italic]default · welcome(action='edit', content=...)[/]"
            console.print(Panel(
                welcome_body,
                title="[bold]📜 Welcome[/]",
                subtitle=subtitle,
                border_style="bright_magenta",
                box=box.ROUNDED,
                padding=(1, 2),
            ))
    except Exception:
        pass

    # ── Footer ──────────────────────────────────────────────────
    now = datetime.now()
    footer = Text()
    footer.append(f"📝 Logs: {LOG_DIR}", style="dim")
    footer.append("  ·  ", style="dim")
    footer.append(f"📂 {os.getcwd()}", style="dim")
    footer.append("  ·  ", style="dim")
    footer.append(now.strftime("%H:%M"), style="dim bold")
    console.print(Align.center(footer))
    console.print()
