"""DevDuck tools package."""

from .tcp import tcp
from .mcp_server import mcp_server
from .install_tools import install_tools
from .tray import tray
from .ambient import ambient
from .websocket import websocket
from .ipc import ipc
from .use_github import use_github
from .create_subagent import create_subagent
from .store_in_kb import store_in_kb
from .system_prompt import system_prompt
from .state_manager import state_manager

# AgentCore tools (conditionally available)
try:
    from .agentcore_config import agentcore_config
    from .agentcore_invoke import agentcore_invoke
    from .agentcore_logs import agentcore_logs
    from .agentcore_agents import agentcore_agents

    __all__ = [
        "tcp",
        "websocket",
        "ipc",
        "mcp_server",
        "install_tools",
        "use_github",
        "create_subagent",
        "store_in_kb",
        "system_prompt",
        "state_manager",
        "tray",
        "ambient",
        "agentcore_config",
        "agentcore_invoke",
        "agentcore_logs",
        "agentcore_agents",
    ]
except ImportError:
    __all__ = [
        "tcp",
        "websocket",
        "ipc",
        "mcp_server",
        "install_tools",
        "use_github",
        "create_subagent",
        "store_in_kb",
        "system_prompt",
        "state_manager",
        "tray",
        "ambient",
    ]
