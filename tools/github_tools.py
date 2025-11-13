import os

import requests
from strands import tool


@tool
def list_pull_requests(state: str = "open", repo: str | None = None) -> str:
    """Lists pull requests from the specified GitHub repository or GITHUB_REPOSITORY environment variable.

    Args:
        state: Filter PRs by state: "open", "closed", or "all" (default: "open")
        repo: GitHub repository in the format "owner/repo" (optional; falls back to env var)

    Returns:
        String representation of the pull requests
    """
    if repo is None:
        repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        return "Error: GITHUB_REPOSITORY environment variable not found and no repo provided"

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return "Error: GITHUB_TOKEN environment variable not found"

    url = f"https://api.github.com/repos/{repo}/pulls?state={state}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        pulls = response.json()
        if not pulls:
            return f"No {state} pull requests found in {repo}"

        result = f"Pull Requests ({state}) in {repo}:\n"
        for pr in pulls:
            result += f"#{pr['number']} - {pr['title']} by {pr['user']['login']} - {pr['html_url']}\n"

        return result

    except Exception as e:
        return f"Error fetching pull requests: {e!s}"


@tool
def list_issues(state: str = "open", repo: str | None = None) -> str:
    """Lists issues from the specified GitHub repository or GITHUB_REPOSITORY environment variable.

    Args:
        state: Filter issues by state: "open", "closed", or "all" (default: "open")
        repo: GitHub repository in the format "owner/repo" (optional; falls back to env var)

    Returns:
        String representation of the issues
    """
    if repo is None:
        repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        return "Error: GITHUB_REPOSITORY environment variable not found and no repo provided"

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return "Error: GITHUB_TOKEN environment variable not found"

    url = f"https://api.github.com/repos/{repo}/issues?state={state}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        issues = response.json()
        # Filter out pull requests from issues list
        issues = [issue for issue in issues if "pull_request" not in issue]

        if not issues:
            return f"No {state} issues found in {repo}"

        result = f"Issues ({state}) in {repo}:\n"
        for issue in issues:
            result += f"#{issue['number']} - {issue['title']} by {issue['user']['login']} - {issue['html_url']}\n"

        return result

    except Exception as e:
        return f"Error fetching issues: {e!s}"


@tool
def add_comment(issue_number: int, comment_text: str, repo: str | None = None) -> str:
    """Adds a comment to an issue or pull request in the specified repository or GITHUB_REPOSITORY environment variable.

    Args:
        issue_number: The issue or PR number to comment on
        comment_text: The comment text
        repo: GitHub repository in the format "owner/repo" (optional; falls back to env var)

    Returns:
        Result of the operation
    """
    if repo is None:
        repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        return "Error: GITHUB_REPOSITORY environment variable not found and no repo provided"

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return "Error: GITHUB_TOKEN environment variable not found"

    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {"body": comment_text}

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()

        comment = response.json()
        return f"Comment added successfully: {comment['html_url']}"

    except Exception as e:
        return f"Error adding comment: {e!s}"
