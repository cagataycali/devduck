"""
DevDuck Tools Package

This module exports all available tools for devduck.
"""

from .agentcore_agents import agentcore_agents
from .agentcore_config import agentcore_config
from .agentcore_invoke import agentcore_invoke
from .agentcore_logs import agentcore_logs
from .agentcore_proxy import agentcore_proxy
from .ambient import ambient
from .create_subagent import create_subagent
from .fetch_github_tool import fetch_github_tool
from .install_tools import install_tools
from .ipc import ipc
from .lsp import lsp
from .mcp_server import mcp_server
from .scraper import scraper
from .state_manager import state_manager
from .store_in_kb import store_in_kb
from .system_prompt import system_prompt
from .tcp import tcp
from .tray import tray
from .use_github import use_github
from .websocket import websocket
from .zenoh_peer import zenoh_peer
from .ambient_mode import ambient_mode
from .listen import listen
from .tasks import tasks
from .use_computer import use_computer
from .dialog import dialog
from .sqlite_memory import sqlite_memory
from .scheduler import scheduler
from .dialog import dialog
from .manage_messages import manage_messages
from .manage_tools import manage_tools
from .rich_interface import rich_interface
from .editor import editor
from .use_flipper import use_flipper

# Optional Tools
try:
    from .speech_to_speech import speech_to_speech

    __all__ = [
        "agentcore_agents",
        "agentcore_config",
        "agentcore_invoke",
        "agentcore_logs",
        "agentcore_proxy",
        "ambient",
        "create_subagent",
        "fetch_github_tool",
        "install_tools",
        "ipc",
        "lsp",
        "mcp_server",
        "scraper",
        "speech_to_speech",
        "state_manager",
        "store_in_kb",
        "system_prompt",
        "tcp",
        "tray",
        "use_github",
        "websocket",
        "zenoh_peer",
        "ambient_mode",
        "listen",
        "tasks",
        "use_computer",
        "dialog",
        "sqlite_memory",
        "scheduler",
        "dialog",
        "manage_messages",
        "manage_tools",
        "rich_interface",
        "editor",
        "use_flipper",
    ]
except ImportError:
    __all__ = [
        "agentcore_agents",
        "agentcore_config",
        "agentcore_invoke",
        "agentcore_logs",
        "agentcore_proxy",
        "ambient",
        "create_subagent",
        "fetch_github_tool",
        "install_tools",
        "ipc",
        "lsp",
        "mcp_server",
        "scraper",
        "state_manager",
        "store_in_kb",
        "system_prompt",
        "tcp",
        "tray",
        "use_github",
        "websocket",
        "zenoh_peer",
        "ambient_mode",
        "listen",
        "tasks",
        "use_computer",
        "dialog",
        "sqlite_memory",
        "scheduler",
        "dialog",
        "manage_messages",
        "manage_tools",
        "rich_interface",
        "editor",
        "use_flipper",
    ]
