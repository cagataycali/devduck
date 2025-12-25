"""System prompt management tool for Strands Agents.

This module provides a tool to view and modify system prompts used by the agent.
It helps with dynamic adaptation of the agent's behavior and capabilities,
and can persist changes by updating GitHub repository variables and local .prompt files.

Key Features:
1. View current system prompt from any environment variable
2. Update system prompt (in-memory, .prompt file, and GitHub repository variable)
3. Add context information to system prompt
4. Reset system prompt to default
5. Support for custom variable names (SYSTEM_PROMPT, TOOL_BUILDER_SYSTEM_PROMPT, etc.)
6. Local file persistence via .prompt files with predictable fallback locations

Usage Examples:
```python
from strands import Agent
from protocol_tools import system_prompt

agent = Agent(tools=[system_prompt])

# View current system prompt (default SYSTEM_PROMPT variable)
result = agent.tool.system_prompt(action="view")

# Update system prompt for tool builder
result = agent.tool.system_prompt(
    action="update",
    prompt="You are a specialized tool builder agent...",
    repository="owner/repo",
    variable_name="TOOL_BUILDER_SYSTEM_PROMPT",
)

# Work with any custom variable name
result = agent.tool.system_prompt(
    action="view", variable_name="MY_CUSTOM_PROMPT"
)
```
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import requests
from strands import tool


def _get_github_token() -> str:
    """Get GitHub token from environment variable."""
    return os.environ.get("PAT_TOKEN", os.environ.get("GITHUB_TOKEN", ""))


def _get_prompt_file_path(variable_name: str = "SYSTEM_PROMPT") -> Path:
    """Get the path to the .prompt file for a given variable name with fallback strategy.

    Tries locations in this order:
    1. CWD (current working directory)
    2. /tmp/devduck/prompts
    3. tempdir/devduck/prompts

    Args:
        variable_name: Name of the variable (used to generate filename)

    Returns:
        Path to the .prompt file (first writable location)
    """
    # Convert variable name to lowercase filename
    # SYSTEM_PROMPT -> system_prompt.prompt
    # MY_CUSTOM_PROMPT -> my_custom_prompt.prompt
    filename = f"{variable_name.lower()}.prompt"

    # Try 1: CWD
    try:
        cwd_path = Path.cwd() / filename
        # Test if we can write to CWD
        cwd_path.touch(exist_ok=True)
        return cwd_path
    except (OSError, PermissionError):
        pass

    # Try 2: /tmp/devduck/prompts
    try:
        tmp_dir = Path("/tmp/devduck/prompts")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / filename
        # Test if we can write to /tmp/devduck
        tmp_path.touch(exist_ok=True)
        return tmp_path
    except (OSError, PermissionError):
        pass

    # Try 3: tempdir/devduck/prompts (system temp directory)
    temp_dir = Path(tempfile.gettempdir()) / "devduck" / "prompts"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / filename


def _read_prompt_file(variable_name: str = "SYSTEM_PROMPT") -> str:
    """Read prompt from .prompt file across all possible locations.

    Checks all locations in priority order and returns the first found.

    Args:
        variable_name: Name of the variable

    Returns:
        Content of the .prompt file or empty string if not found
    """
    filename = f"{variable_name.lower()}.prompt"

    # Check all possible locations in priority order
    possible_paths = [
        Path.cwd() / filename,  # CWD
        Path("/tmp/devduck/prompts") / filename,  # /tmp/devduck
        Path(tempfile.gettempdir()) / "devduck" / "prompts" / filename,  # tempdir
    ]

    for prompt_file in possible_paths:
        try:
            if prompt_file.exists():
                return prompt_file.read_text(encoding="utf-8")
        except Exception:
            continue

    return ""


def _write_prompt_file(
    prompt: str, variable_name: str = "SYSTEM_PROMPT"
) -> tuple[bool, str]:
    """Write prompt to .prompt file with fallback strategy.

    Args:
        prompt: The prompt content to write
        variable_name: Name of the variable

    Returns:
        Tuple of (success, path) where success is True if write succeeded
    """
    prompt_file = _get_prompt_file_path(variable_name)
    try:
        prompt_file.write_text(prompt, encoding="utf-8")
        return True, str(prompt_file)
    except Exception:
        return False, str(prompt_file)


def _delete_prompt_file(variable_name: str = "SYSTEM_PROMPT") -> tuple[bool, str]:
    """Delete .prompt file from all possible locations.

    Args:
        variable_name: Name of the variable

    Returns:
        Tuple of (success, path) - success if any file was deleted
    """
    filename = f"{variable_name.lower()}.prompt"
    deleted = False
    deleted_path = ""

    # Try to delete from all possible locations
    possible_paths = [
        Path.cwd() / filename,  # CWD
        Path("/tmp/devduck/prompts") / filename,  # /tmp/devduck
        Path(tempfile.gettempdir()) / "devduck" / "prompts" / filename,  # tempdir
    ]

    for prompt_file in possible_paths:
        try:
            if prompt_file.exists():
                prompt_file.unlink()
                deleted = True
                deleted_path = str(prompt_file)
        except Exception:
            continue

    return deleted, deleted_path


def _get_github_repository_variable(
    repository: str, name: str, token: str
) -> dict[str, Any]:
    """Fetch a GitHub repository variable.

    Args:
        repository: The repository in format "owner/repo"
        name: The variable name
        token: GitHub token

    Returns:
        Dictionary with success status, message, and value if successful
    """
    # GitHub API endpoint for repository variables
    url = f"https://api.github.com/repos/{repository}/actions/variables/{name}"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "message": f"Variable {name} fetched successfully",
                "value": data.get("value", ""),
            }
        else:
            error_message = (
                f"Failed to fetch variable: {response.status_code} - {response.text}"
            )
            return {"success": False, "message": error_message, "value": ""}
    except Exception as e:
        return {
            "success": False,
            "message": f"Error fetching GitHub variable: {e!s}",
            "value": "",
        }


def _get_system_prompt(
    repository: str | None = None, variable_name: str = "SYSTEM_PROMPT"
) -> str:
    """Get the current system prompt.

    Priority order:
    1. Local environment variable
    2. Local .prompt file (CWD → /tmp/devduck → tempdir)
    3. GitHub repository variable (if repository specified)

    Args:
        repository: Optional GitHub repository in format "owner/repo"
        variable_name: Name of the environment/repository variable to use

    Returns:
        The system prompt string
    """
    # First check local environment
    local_prompt = os.environ.get(variable_name, "")
    if local_prompt:
        return local_prompt

    # Second, check .prompt file across all locations
    file_prompt = _read_prompt_file(variable_name)
    if file_prompt:
        # Load into environment for caching
        os.environ[variable_name] = file_prompt
        return file_prompt

    # Third, if repository is provided, try GitHub
    if repository:
        token = _get_github_token()
        if token:
            result = _get_github_repository_variable(
                repository=repository, name=variable_name, token=token
            )

            if result["success"] and result["value"]:
                # Store in local environment and file for future use
                os.environ[variable_name] = result["value"]
                _write_prompt_file(result["value"], variable_name)
                return str(result["value"])

    # Default to empty string if nothing found
    return ""


def _update_system_prompt(
    new_prompt: str, variable_name: str = "SYSTEM_PROMPT"
) -> dict[str, Any]:
    """Update the system prompt in environment variable and .prompt file.

    Args:
        new_prompt: The new prompt content
        variable_name: Name of the variable

    Returns:
        Dictionary with success status and messages
    """
    # Update environment variable
    os.environ[variable_name] = new_prompt

    # Update .prompt file with fallback strategy
    file_success, file_path = _write_prompt_file(new_prompt, variable_name)

    return {"env_updated": True, "file_updated": file_success, "file_path": file_path}


def _get_github_event_context() -> str:
    """Get GitHub event context information from environment variables."""
    event_context = []

    # GitHub repository information
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if repo:
        event_context.append(f"Repository: {repo}")

    # Event type
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    if event_name:
        event_context.append(f"Event Type: {event_name}")

    # Actor
    actor = os.environ.get("GITHUB_ACTOR", "")
    if actor:
        event_context.append(f"Actor: {actor}")

    # Add more GitHub context variables as needed
    return "\n".join(event_context)


def _update_github_repository_variable(
    repository: str, name: str, value: str, token: str
) -> dict[str, Any]:
    """Update a GitHub repository variable.

    Args:
        repository: The repository in format "owner/repo"
        name: The variable name
        value: The variable value
        token: GitHub token

    Returns:
        Dictionary with status and message
    """
    # GitHub API endpoint for repository variables
    url = f"https://api.github.com/repos/{repository}/actions/variables/{name}"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    data = {"name": name, "value": value}

    response = requests.patch(url, headers=headers, json=data, timeout=30)

    if response.status_code == 204:
        return {"success": True, "message": f"Variable {name} updated successfully"}
    else:
        error_message = (
            f"Failed to update variable: {response.status_code} - {response.text}"
        )
        return {"success": False, "message": error_message}


@tool
def system_prompt(
    action: str,
    prompt: str | None = None,
    context: str | None = None,
    repository: str | None = None,
    variable_name: str = "SYSTEM_PROMPT",
) -> dict[str, str | list[dict[str, str]]]:
    """Manage the agent's system prompt.

    This tool allows viewing and modifying the system prompt used by the agent.
    It can be used to adapt the agent's behavior dynamically during runtime
    and can update GitHub repository variables and local .prompt files to persist changes.

    Args:
        action: The action to perform on the system prompt. One of:
            - "view": View the current system prompt
            - "update": Replace the current system prompt
            - "add_context": Add additional context to the system prompt
            - "reset": Reset to default (empty or environment-defined)
            - "get_github_context": Get GitHub event context
        prompt: New system prompt when using the "update" action
        context: Additional context to add when using the "add_context" action
        repository: GitHub repository in format "owner/repo" to update repository
                   variable (e.g., "cagataycali/report-agent")
        variable_name: Name of the environment/repository variable to use
                      (default: "SYSTEM_PROMPT")

    Returns:
        A dictionary with the operation status and current system prompt

    Example:
        ```python
        # View current system prompt
        result = system_prompt(action="view")

        # Update system prompt (saves to CWD → /tmp/devduck → tempdir)
        result = system_prompt(
            action="update", prompt="You are a specialized agent for task X..."
        )

        # Update GitHub repository variable (+ env var + .prompt file)
        result = system_prompt(
            action="update",
            prompt="You are a specialized agent for task X...",
            repository="owner/repo",
        )

        # Work with custom variable name
        result = system_prompt(
            action="update",
            prompt="You are a tool builder...",
            repository="owner/repo",
            variable_name="TOOL_BUILDER_SYSTEM_PROMPT",
        )
        ```
    """
    try:
        if action == "view":
            current_prompt = _get_system_prompt(repository, variable_name)

            # Determine source
            source_parts = []
            if os.environ.get(variable_name):
                source_parts.append("environment variable")

            # Check all possible file locations
            filename = f"{variable_name.lower()}.prompt"
            file_locations = [
                (Path.cwd() / filename, "CWD"),
                (Path("/tmp/devduck/prompts") / filename, "/tmp/devduck"),
                (
                    Path(tempfile.gettempdir()) / "devduck" / "prompts" / filename,
                    "tempdir",
                ),
            ]

            for file_path, location in file_locations:
                if file_path.exists():
                    source_parts.append(f"file ({location}: {file_path})")
                    break

            if repository:
                source_parts.append(f"GitHub ({repository})")

            source = " → ".join(source_parts) if source_parts else "not found"

            return {
                "status": "success",
                "content": [
                    {
                        "text": f"Current system prompt from {variable_name}:\nSource: {source}\n\n{current_prompt}"
                    }
                ],
            }

        elif action == "update":
            if not prompt:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "Error: prompt parameter is required for the update action"
                        }
                    ],
                }

            # Update in-memory environment variable and .prompt file
            update_result = _update_system_prompt(prompt, variable_name)

            messages = []
            messages.append(f"✓ Environment variable updated ({variable_name})")

            if update_result["file_updated"]:
                messages.append(f"✓ File saved ({update_result['file_path']})")
            else:
                messages.append(f"⚠ File save failed ({update_result['file_path']})")

            # If repository is specified, also update GitHub repository variable
            if repository:
                token = _get_github_token()
                if not token:
                    messages.append(
                        "⚠ GitHub token not available - skipped repository update"
                    )
                else:
                    result = _update_github_repository_variable(
                        repository=repository,
                        name=variable_name,
                        value=prompt,
                        token=token,
                    )

                    if result["success"]:
                        messages.append(
                            f"✓ GitHub repository variable updated ({repository})"
                        )
                    else:
                        messages.append(f"⚠ GitHub update failed: {result['message']}")

            return {
                "status": "success",
                "content": [{"text": "\n".join(messages)}],
            }

        elif action == "add_context":
            if not context:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "Error: context parameter is required for the add_context action"
                        }
                    ],
                }

            current_prompt = _get_system_prompt(repository, variable_name)
            new_prompt = f"{current_prompt}\n\n{context}" if current_prompt else context

            # Update in-memory environment variable and .prompt file
            update_result = _update_system_prompt(new_prompt, variable_name)

            messages = []
            messages.append(f"✓ Context added to {variable_name}")
            messages.append(f"✓ Environment variable updated")

            if update_result["file_updated"]:
                messages.append(f"✓ File saved ({update_result['file_path']})")
            else:
                messages.append(f"⚠ File save failed ({update_result['file_path']})")

            # If repository is specified, also update GitHub repository variable
            if repository:
                token = _get_github_token()
                if not token:
                    messages.append(
                        "⚠ GitHub token not available - skipped repository update"
                    )
                else:
                    result = _update_github_repository_variable(
                        repository=repository,
                        name=variable_name,
                        value=new_prompt,
                        token=token,
                    )

                    if result["success"]:
                        messages.append(
                            f"✓ GitHub repository variable updated ({repository})"
                        )
                    else:
                        messages.append(f"⚠ GitHub update failed: {result['message']}")

            return {
                "status": "success",
                "content": [{"text": "\n".join(messages)}],
            }

        elif action == "reset":
            # Reset environment variable
            os.environ.pop(variable_name, None)

            # Delete .prompt file from all locations
            file_deleted, deleted_path = _delete_prompt_file(variable_name)

            messages = []
            messages.append(f"✓ Environment variable reset ({variable_name})")

            if file_deleted:
                messages.append(f"✓ File deleted ({deleted_path})")
            else:
                messages.append("⚠ File deletion failed or file doesn't exist")

            # If repository is specified, reset GitHub repository variable
            if repository:
                token = _get_github_token()
                if not token:
                    messages.append(
                        "⚠ GitHub token not available - skipped repository reset"
                    )
                else:
                    result = _update_github_repository_variable(
                        repository=repository, name=variable_name, value="", token=token
                    )

                    if result["success"]:
                        messages.append(
                            f"✓ GitHub repository variable reset ({repository})"
                        )
                    else:
                        messages.append(f"⚠ GitHub reset failed: {result['message']}")

            return {
                "status": "success",
                "content": [{"text": "\n".join(messages)}],
            }

        elif action == "get_github_context":
            github_context = _get_github_event_context()
            return {
                "status": "success",
                "content": [{"text": f"GitHub Event Context:\n\n{github_context}"}],
            }

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Error: Unknown action '{action}'. Valid actions are view, update, add_context, reset, get_github_context"
                    }
                ],
            }

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {e!s}"}]}
