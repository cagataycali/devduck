"""Dynamic Tool Installation for DevDuck.

Install and load tools from any Python package at runtime, expanding DevDuck's
capabilities on-the-fly without requiring restarts.
"""

import importlib
import logging
import subprocess
import sys
from typing import Any, Dict, List, Optional

from strands import tool

logger = logging.getLogger(__name__)


@tool
def install_tools(
    action: str,
    package: Optional[str] = None,
    module: Optional[str] = None,
    tool_names: Optional[List[str]] = None,
    agent: Any = None,
) -> Dict[str, Any]:
    """Install and load tools from Python packages dynamically.

    This tool allows DevDuck to expand its capabilities by installing Python packages
    and loading their tools into the agent's registry at runtime.

    Args:
        action: Action to perform - "install", "load", "install_and_load", "list_loaded", "list_available"
        package: Python package to install (e.g., "strands-agents-tools", "strands-fun-tools")
        module: Module to import tools from (e.g., "strands_tools", "strands_fun_tools")
        tool_names: Optional list of specific tools to load. If None, loads all available tools
        agent: Parent agent instance (auto-injected by Strands framework)

    Returns:
        Result dictionary with status and content

    Examples:
        # List available tools in a package (without loading)
        install_tools(
            action="list_available",
            package="strands-fun-tools",
            module="strands_fun_tools"
        )

        # Install and load all tools from strands-agents-tools
        install_tools(
            action="install_and_load",
            package="strands-agents-tools",
            module="strands_tools"
        )

        # Install and load specific tools
        install_tools(
            action="install_and_load",
            package="strands-fun-tools",
            module="strands_fun_tools",
            tool_names=["clipboard", "cursor", "bluetooth"]
        )

        # Load tools from already installed package
        install_tools(
            action="load",
            module="strands_tools",
            tool_names=["shell", "calculator"]
        )

        # List currently loaded tools
        install_tools(action="list_loaded")
    """
    try:
        if action == "install":
            return _install_package(package)
        elif action == "load":
            return _load_tools_from_module(module, tool_names, agent)
        elif action == "install_and_load":
            # Install first
            install_result = _install_package(package)
            if install_result["status"] == "error":
                return install_result

            # Then load
            return _load_tools_from_module(module, tool_names, agent)
        elif action == "list_loaded":
            return _list_loaded_tools(agent)
        elif action == "list_available":
            return _list_available_tools(package, module)
        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"❌ Unknown action: {action}\n\n"
                        f"Valid actions: install, load, install_and_load, list_loaded, list_available"
                    }
                ],
            }

    except Exception as e:
        logger.exception("Error in install_tools")
        return {"status": "error", "content": [{"text": f"❌ Error: {str(e)}"}]}


def _install_package(package: str) -> Dict[str, Any]:
    """Install a Python package using pip."""
    if not package:
        return {
            "status": "error",
            "content": [
                {"text": "❌ package parameter is required for install action"}
            ],
        }

    try:
        logger.info(f"Installing package: {package}")

        # Use subprocess to install the package
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            return {
                "status": "error",
                "content": [
                    {"text": f"❌ Failed to install {package}:\n{result.stderr}"}
                ],
            }

        logger.info(f"Successfully installed: {package}")

        return {
            "status": "success",
            "content": [{"text": f"✅ Successfully installed package: {package}"}],
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "content": [
                {"text": f"❌ Installation of {package} timed out (>5 minutes)"}
            ],
        }
    except Exception as e:
        logger.exception(f"Error installing package {package}")
        return {
            "status": "error",
            "content": [{"text": f"❌ Failed to install {package}: {str(e)}"}],
        }


def _load_tools_from_module(
    module: str, tool_names: Optional[List[str]], agent: Any
) -> Dict[str, Any]:
    """Load tools from a Python module into the agent's registry."""
    if not module:
        return {
            "status": "error",
            "content": [{"text": "❌ module parameter is required for load action"}],
        }

    if not agent:
        return {
            "status": "error",
            "content": [{"text": "❌ agent instance is required for load action"}],
        }

    if not hasattr(agent, "tool_registry") or not hasattr(
        agent.tool_registry, "register_tool"
    ):
        return {
            "status": "error",
            "content": [{"text": "❌ Agent does not have a tool registry"}],
        }

    try:
        # Import the module
        logger.info(f"Importing module: {module}")
        imported_module = importlib.import_module(module)

        # Get all tool objects from the module
        available_tools = {}
        for attr_name in dir(imported_module):
            attr = getattr(imported_module, attr_name)
            # Check if it's a tool (has tool_name and tool_spec attributes)
            if hasattr(attr, "tool_name") and hasattr(attr, "tool_spec"):
                available_tools[attr.tool_name] = attr

        if not available_tools:
            return {
                "status": "error",
                "content": [{"text": f"❌ No tools found in module: {module}"}],
            }

        # Filter tools if specific ones requested
        if tool_names:
            tools_to_load = {
                name: tool
                for name, tool in available_tools.items()
                if name in tool_names
            }
            missing_tools = set(tool_names) - set(tools_to_load.keys())
            if missing_tools:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"❌ Requested tools not found: {', '.join(missing_tools)}\n\n"
                            f"Available tools: {', '.join(available_tools.keys())}"
                        }
                    ],
                }
        else:
            tools_to_load = available_tools

        # Load tools into agent registry
        loaded_tools = []
        skipped_tools = []

        for tool_name, tool_obj in tools_to_load.items():
            try:
                # Check if tool already exists
                existing_tools = agent.tool_registry.get_all_tools_config()
                if tool_name in existing_tools:
                    skipped_tools.append(f"{tool_name} (already loaded)")
                    continue

                # Register the tool
                agent.tool_registry.register_tool(tool_obj)
                loaded_tools.append(tool_name)
                logger.info(f"Loaded tool: {tool_name}")

            except Exception as e:
                skipped_tools.append(f"{tool_name} (error: {str(e)})")
                logger.error(f"Failed to load tool {tool_name}: {e}")

        # Build result message
        result_lines = [f"✅ Loaded {len(loaded_tools)} tools from {module}"]

        if loaded_tools:
            result_lines.append(f"\n📦 Loaded tools:")
            for tool_name in loaded_tools:
                result_lines.append(f"  • {tool_name}")

        if skipped_tools:
            result_lines.append(f"\n⚠️  Skipped tools:")
            for skip_msg in skipped_tools:
                result_lines.append(f"  • {skip_msg}")

        result_lines.append(
            f"\n🔧 Total available tools: {len(existing_tools) + len(loaded_tools)}"
        )

        return {"status": "success", "content": [{"text": "\n".join(result_lines)}]}

    except ImportError as e:
        logger.exception(f"Failed to import module {module}")
        return {
            "status": "error",
            "content": [
                {
                    "text": f"❌ Failed to import module {module}: {str(e)}\n\n"
                    f"Make sure the package is installed first using action='install'"
                }
            ],
        }
    except Exception as e:
        logger.exception(f"Error loading tools from {module}")
        return {
            "status": "error",
            "content": [{"text": f"❌ Failed to load tools: {str(e)}"}],
        }


def _list_loaded_tools(agent: Any) -> Dict[str, Any]:
    """List all currently loaded tools in the agent."""
    if not agent:
        return {
            "status": "error",
            "content": [{"text": "❌ agent instance is required"}],
        }

    if not hasattr(agent, "tool_registry"):
        return {
            "status": "error",
            "content": [{"text": "❌ Agent does not have a tool registry"}],
        }

    try:
        all_tools = agent.tool_registry.get_all_tools_config()

        result_lines = [f"🔧 **Loaded Tools ({len(all_tools)})**\n"]

        # Group tools by category (if available)
        for tool_name, tool_spec in sorted(all_tools.items()):
            description = tool_spec.get("description", "No description available")
            # Truncate long descriptions
            if len(description) > 100:
                description = description[:97] + "..."

            result_lines.append(f"**{tool_name}**")
            result_lines.append(f"  {description}\n")

        return {"status": "success", "content": [{"text": "\n".join(result_lines)}]}

    except Exception as e:
        logger.exception("Error listing loaded tools")
        return {
            "status": "error",
            "content": [{"text": f"❌ Failed to list tools: {str(e)}"}],
        }


def _list_available_tools(package: Optional[str], module: str) -> Dict[str, Any]:
    """List available tools in a package without loading them."""
    if not module:
        return {
            "status": "error",
            "content": [
                {"text": "❌ module parameter is required for list_available action"}
            ],
        }

    try:
        # Try to import the module
        try:
            imported_module = importlib.import_module(module)
            logger.info(f"Module {module} already installed")
        except ImportError:
            # Module not installed - try to install package first
            if not package:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"❌ Module {module} not found and no package specified to install.\n\n"
                            f"Please provide the 'package' parameter to install first."
                        }
                    ],
                }

            logger.info(f"Module {module} not found, installing package {package}")
            install_result = _install_package(package)
            if install_result["status"] == "error":
                return install_result

            # Try importing again after installation
            try:
                imported_module = importlib.import_module(module)
            except ImportError as e:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"❌ Failed to import {module} even after installing {package}: {str(e)}"
                        }
                    ],
                }

        # Discover tools in the module
        available_tools = {}
        for attr_name in dir(imported_module):
            attr = getattr(imported_module, attr_name)
            # Check if it's a tool (has tool_name and tool_spec attributes)
            if hasattr(attr, "tool_name") and hasattr(attr, "tool_spec"):
                tool_spec = attr.tool_spec
                description = tool_spec.get("description", "No description available")
                available_tools[attr.tool_name] = description

        if not available_tools:
            return {
                "status": "success",
                "content": [{"text": f"⚠️  No tools found in module: {module}"}],
            }

        # Build result message
        result_lines = [
            f"📦 **Available Tools in {module} ({len(available_tools)})**\n"
        ]

        for tool_name, description in sorted(available_tools.items()):
            # Truncate long descriptions
            if len(description) > 100:
                description = description[:97] + "..."

            result_lines.append(f"**{tool_name}**")
            result_lines.append(f"  {description}\n")

        result_lines.append(f"\n💡 To load these tools, use:")
        result_lines.append(f"   install_tools(action='load', module='{module}')")
        result_lines.append(f"   # Or load specific tools:")
        result_lines.append(
            f"   install_tools(action='load', module='{module}', tool_names=['tool1', 'tool2'])"
        )

        return {"status": "success", "content": [{"text": "\n".join(result_lines)}]}

    except Exception as e:
        logger.exception(f"Error listing available tools from {module}")
        return {
            "status": "error",
            "content": [{"text": f"❌ Failed to list available tools: {str(e)}"}],
        }
