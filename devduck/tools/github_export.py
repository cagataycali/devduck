#!/usr/bin/env python3
"""
🦆 GitHub Export Tool - Export GitHub activities for knowledge base ingestion.

Exports PRs, issues, discussions, commits, and reviews from GitHub repositories.
Tracks last export for incremental/diff-based updates.
"""

import os
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from strands import tool

# State file for tracking exports
STATE_FILE = Path.home() / ".devduck" / "github_export_state.json"


def _ensure_state_dir():
    """Ensure state directory exists."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_state() -> Dict[str, Any]:
    """Load export state from file."""
    _ensure_state_dir()
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"repositories": {}, "last_global_export": None}


def _save_state(state: Dict[str, Any]):
    """Save export state to file."""
    _ensure_state_dir()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _clean_json_output(output: str) -> str:
    """Clean ANSI escape codes and find JSON in output."""
    import re
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])|\]11;[^\x07]*\x07?|\[\d+n')
    cleaned = ansi_escape.sub('', output)
    
    # Find JSON array or object
    json_match = re.search(r'(\[.*\]|\{.*\})', cleaned, re.DOTALL)
    if json_match:
        return json_match.group(1)
    return cleaned.strip()


def _run_gh(args: List[str], repo: Optional[str] = None) -> tuple[bool, str]:
    """Run GitHub CLI command."""
    cmd = ["gh"]
    if repo:
        cmd.extend(["-R", repo])
    cmd.extend(args)
    
    # Set environment to disable paging and color
    env = os.environ.copy()
    env["GH_PAGER"] = ""
    env["NO_COLOR"] = "1"
    env["GH_NO_UPDATE_NOTIFIER"] = "1"
    env["TERM"] = "dumb"
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env
        )
        if result.returncode == 0:
            # Clean output of ANSI codes
            return True, _clean_json_output(result.stdout)
        return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except FileNotFoundError:
        return False, "GitHub CLI (gh) not installed. Install: brew install gh"
    except Exception as e:
        return False, str(e)


def _format_pr_for_kb(pr: Dict, repo: str) -> str:
    """Format a PR for knowledge base storage."""
    labels = ", ".join([l.get("name", "") for l in pr.get("labels", [])]) or "none"
    review_requests = ", ".join([r.get("login", r.get("name", "")) for r in pr.get("reviewRequests", [])]) or "none"
    
    content = f"""## Pull Request: {pr.get('title', 'Untitled')}
Repository: {repo}
Number: #{pr.get('number')}
State: {pr.get('state', 'unknown')}
Author: {pr.get('author', {}).get('login', 'unknown')}
Created: {pr.get('createdAt', 'unknown')}
Updated: {pr.get('updatedAt', 'unknown')}
Merged: {pr.get('mergedAt', 'not merged')}
Labels: {labels}
Review Requests: {review_requests}
URL: {pr.get('url', '')}

### Description:
{pr.get('body', 'No description provided.')}

### Files Changed: {pr.get('changedFiles', 0)}
Additions: +{pr.get('additions', 0)} | Deletions: -{pr.get('deletions', 0)}
"""
    return content


def _format_issue_for_kb(issue: Dict, repo: str) -> str:
    """Format an issue for knowledge base storage."""
    labels = ", ".join([l.get("name", "") for l in issue.get("labels", [])]) or "none"
    assignees = ", ".join([a.get("login", "") for a in issue.get("assignees", [])]) or "unassigned"
    
    content = f"""## Issue: {issue.get('title', 'Untitled')}
Repository: {repo}
Number: #{issue.get('number')}
State: {issue.get('state', 'unknown')}
Author: {issue.get('author', {}).get('login', 'unknown')}
Created: {issue.get('createdAt', 'unknown')}
Updated: {issue.get('updatedAt', 'unknown')}
Closed: {issue.get('closedAt', 'open')}
Labels: {labels}
Assignees: {assignees}
URL: {issue.get('url', '')}

### Description:
{issue.get('body', 'No description provided.')}
"""
    return content


def _format_comment_for_kb(comment: Dict, context: str) -> str:
    """Format a comment for knowledge base storage."""
    return f"""### Comment on {context}
Author: {comment.get('author', {}).get('login', 'unknown')}
Created: {comment.get('createdAt', 'unknown')}

{comment.get('body', '')}
"""


def _format_review_for_kb(review: Dict, pr_number: int) -> str:
    """Format a PR review for knowledge base storage."""
    return f"""### Review on PR #{pr_number}
Reviewer: {review.get('author', {}).get('login', 'unknown')}
State: {review.get('state', 'unknown')}
Submitted: {review.get('submittedAt', 'unknown')}

{review.get('body', 'No review comment.')}
"""


@tool
def github_export(
    action: str,
    repository: str = None,
    since: str = None,
    output_format: str = "markdown",
    include_comments: bool = True,
    include_reviews: bool = True,
    limit: int = 100,
    state: str = "all",
    output_dir: str = None,
) -> Dict[str, Any]:
    """
    Export GitHub activities (PRs, issues, commits) for knowledge base ingestion.
    
    Tracks export state for incremental updates - only exports new/changed items.
    
    Args:
        action: Action to perform:
            - "export_prs": Export pull requests
            - "export_issues": Export issues  
            - "export_all": Export PRs + issues + discussions
            - "export_commits": Export recent commits
            - "export_releases": Export releases
            - "list_repos": List accessible repositories
            - "status": Show export status/state
            - "reset": Reset export state for fresh export
        repository: GitHub repository (owner/repo format, e.g., "strands-agents/sdk-python")
        since: Export items since date (ISO format: 2024-01-01) or "last" for incremental
        output_format: Output format - "markdown", "json", or "kb" (knowledge base ready)
        include_comments: Include comments on PRs/issues (default: True)
        include_reviews: Include PR reviews (default: True)
        limit: Maximum items to export per type (default: 100)
        state: Filter by state - "all", "open", "closed", "merged" (default: "all")
        output_dir: Directory to save exports (default: ./github_exports)
    
    Returns:
        Dict with export results and file paths
    """
    try:
        # Verify gh CLI is authenticated
        success, output = _run_gh(["auth", "status"])
        if not success:
            return {
                "status": "error",
                "content": [{"text": f"GitHub CLI not authenticated. Run: gh auth login\n{output}"}]
            }
        
        export_state = _load_state()
        
        # Handle actions
        if action == "status":
            repos_exported = list(export_state.get("repositories", {}).keys())
            last_export = export_state.get("last_global_export", "never")
            
            status_text = f"""GitHub Export Status:
━━━━━━━━━━━━━━━━━━━━━
Last Global Export: {last_export}
Repositories Tracked: {len(repos_exported)}

"""
            for repo, repo_state in export_state.get("repositories", {}).items():
                status_text += f"""
📦 {repo}:
   PRs exported: {repo_state.get('prs_count', 0)} (last: {repo_state.get('prs_last', 'never')})
   Issues exported: {repo_state.get('issues_count', 0)} (last: {repo_state.get('issues_last', 'never')})
"""
            
            return {"status": "success", "content": [{"text": status_text}]}
        
        if action == "reset":
            if repository:
                if repository in export_state.get("repositories", {}):
                    del export_state["repositories"][repository]
                    _save_state(export_state)
                    return {"status": "success", "content": [{"text": f"Reset state for {repository}"}]}
                return {"status": "success", "content": [{"text": f"No state found for {repository}"}]}
            else:
                _save_state({"repositories": {}, "last_global_export": None})
                return {"status": "success", "content": [{"text": "Reset all export state"}]}
        
        if action == "list_repos":
            success, output = _run_gh(["repo", "list", "--json", "nameWithOwner,description,updatedAt", "--limit", str(limit)])
            if success:
                repos = json.loads(output)
                text = "Accessible Repositories:\n━━━━━━━━━━━━━━━━━━━━━\n\n"
                for repo in repos:
                    text += f"📦 {repo['nameWithOwner']}\n   {repo.get('description', 'No description')[:80]}\n   Updated: {repo.get('updatedAt', 'unknown')}\n\n"
                return {"status": "success", "content": [{"text": text}]}
            return {"status": "error", "content": [{"text": f"Failed to list repos: {output}"}]}
        
        # Actions requiring repository
        if not repository:
            return {"status": "error", "content": [{"text": "repository parameter required for this action"}]}
        
        # Setup output directory
        out_dir = Path(output_dir) if output_dir else Path("./github_exports")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine since date
        if since == "last":
            repo_state = export_state.get("repositories", {}).get(repository, {})
            if action == "export_prs":
                since = repo_state.get("prs_last")
            elif action == "export_issues":
                since = repo_state.get("issues_last")
            else:
                since = repo_state.get("last_export")
        
        results = {"exported": [], "files": [], "counts": {}}
        now = datetime.now(timezone.utc).isoformat()
        
        # Export PRs
        if action in ["export_prs", "export_all"]:
            gh_args = [
                "pr", "list",
                "--state", state if state != "merged" else "all",
                "--json", "number,title,state,author,createdAt,updatedAt,mergedAt,body,labels,reviewRequests,url,additions,deletions,changedFiles",
                "--limit", str(limit)
            ]
            
            success, output = _run_gh(gh_args, repository)
            if not success:
                return {"status": "error", "content": [{"text": f"Failed to fetch PRs: {output}"}]}
            
            try:
                prs = json.loads(output) if output.strip() else []
            except json.JSONDecodeError as e:
                return {"status": "error", "content": [{"text": f"JSON parse error: {e}\nOutput: {output[:500]}"}]}
            
            if not prs:
                return {"status": "success", "content": [{"text": "No PRs found."}]}
            
            # Filter by since date if provided
            if since and prs:
                prs = [p for p in prs if p.get("updatedAt", "") > since]
            
            # Filter merged if requested
            if state == "merged":
                prs = [p for p in prs if p.get("mergedAt")]
            
            if prs:
                pr_content = f"# Pull Requests Export - {repository}\nExported: {now}\nCount: {len(prs)}\n\n"
                
                for pr in prs:
                    pr_content += _format_pr_for_kb(pr, repository)
                    pr_content += "\n---\n\n"
                    
                    # Get comments if requested
                    if include_comments:
                        success_c, comments_out = _run_gh(
                            ["pr", "view", str(pr["number"]), "--json", "comments"],
                            repository
                        )
                        if success_c:
                            try:
                                comments_data = json.loads(comments_out)
                                for comment in comments_data.get("comments", []):
                                    pr_content += _format_comment_for_kb(comment, f"PR #{pr['number']}")
                                    pr_content += "\n"
                            except json.JSONDecodeError:
                                pass
                    
                    # Get reviews if requested
                    if include_reviews:
                        success_r, reviews_out = _run_gh(
                            ["pr", "view", str(pr["number"]), "--json", "reviews"],
                            repository
                        )
                        if success_r:
                            try:
                                reviews_data = json.loads(reviews_out)
                                for review in reviews_data.get("reviews", []):
                                    if review.get("body"):  # Only include reviews with content
                                        pr_content += _format_review_for_kb(review, pr["number"])
                                        pr_content += "\n"
                            except json.JSONDecodeError:
                                pass
                
                # Save to file
                safe_repo = repository.replace("/", "_")
                pr_file = out_dir / f"{safe_repo}_prs.md"
                try:
                    with open(pr_file, "w") as f:
                        f.write(pr_content)
                    results["files"].append(str(pr_file))
                except Exception as write_err:
                    return {"status": "error", "content": [{"text": f"Failed to write file: {write_err}"}]}
                
                results["counts"]["prs"] = len(prs)
                results["exported"].extend([f"PR #{p['number']}: {p['title']}" for p in prs[:10]])
            else:
                # No PRs after filtering - still update state
                pass
            
            # Update state
            if repository not in export_state["repositories"]:
                export_state["repositories"][repository] = {}
            export_state["repositories"][repository]["prs_last"] = now
            export_state["repositories"][repository]["prs_count"] = len(prs) if prs else 0
        
        # Export Issues
        if action in ["export_issues", "export_all"]:
            gh_args = [
                "issue", "list",
                "--state", state if state in ["all", "open", "closed"] else "all",
                "--json", "number,title,state,author,createdAt,updatedAt,closedAt,body,labels,assignees,url",
                "--limit", str(limit)
            ]
            
            success, output = _run_gh(gh_args, repository)
            if success:
                issues = json.loads(output)
                
                # Filter by since date if provided
                if since:
                    issues = [i for i in issues if i.get("updatedAt", "") > since]
                
                issue_content = f"# Issues Export - {repository}\nExported: {now}\nCount: {len(issues)}\n\n"
                
                for issue in issues:
                    issue_content += _format_issue_for_kb(issue, repository)
                    issue_content += "\n---\n\n"
                    
                    # Get comments if requested
                    if include_comments:
                        success_c, comments_out = _run_gh(
                            ["issue", "view", str(issue["number"]), "--json", "comments"],
                            repository
                        )
                        if success_c:
                            comments_data = json.loads(comments_out)
                            for comment in comments_data.get("comments", []):
                                issue_content += _format_comment_for_kb(comment, f"Issue #{issue['number']}")
                                issue_content += "\n"
                
                # Save to file
                safe_repo = repository.replace("/", "_")
                issue_file = out_dir / f"{safe_repo}_issues.md"
                with open(issue_file, "w") as f:
                    f.write(issue_content)
                
                results["files"].append(str(issue_file))
                results["counts"]["issues"] = len(issues)
                results["exported"].extend([f"Issue #{i['number']}: {i['title']}" for i in issues[:10]])
                
                # Update state
                if repository not in export_state["repositories"]:
                    export_state["repositories"][repository] = {}
                export_state["repositories"][repository]["issues_last"] = now
                export_state["repositories"][repository]["issues_count"] = len(issues)
        
        # Export Commits
        if action == "export_commits":
            gh_args = [
                "api", f"repos/{repository}/commits",
                "--jq", f".[:{limit}]"
            ]
            
            if since:
                gh_args = ["api", f"repos/{repository}/commits?since={since}", "--jq", f".[:{limit}]"]
            
            success, output = _run_gh(gh_args)
            if success:
                commits = json.loads(output)
                
                commit_content = f"# Commits Export - {repository}\nExported: {now}\nCount: {len(commits)}\n\n"
                
                for commit in commits:
                    commit_data = commit.get("commit", {})
                    commit_content += f"""## Commit: {commit.get('sha', 'unknown')[:8]}
Author: {commit_data.get('author', {}).get('name', 'unknown')} <{commit_data.get('author', {}).get('email', '')}>
Date: {commit_data.get('author', {}).get('date', 'unknown')}
URL: {commit.get('html_url', '')}

{commit_data.get('message', 'No message')}

---

"""
                
                safe_repo = repository.replace("/", "_")
                commit_file = out_dir / f"{safe_repo}_commits.md"
                with open(commit_file, "w") as f:
                    f.write(commit_content)
                
                results["files"].append(str(commit_file))
                results["counts"]["commits"] = len(commits)
        
        # Export Releases
        if action == "export_releases":
            success, output = _run_gh(
                ["release", "list", "--json", "name,tagName,publishedAt,body,url", "--limit", str(limit)],
                repository
            )
            if success:
                releases = json.loads(output)
                
                release_content = f"# Releases Export - {repository}\nExported: {now}\nCount: {len(releases)}\n\n"
                
                for release in releases:
                    release_content += f"""## Release: {release.get('name', release.get('tagName', 'Unnamed'))}
Tag: {release.get('tagName', 'unknown')}
Published: {release.get('publishedAt', 'unknown')}
URL: {release.get('url', '')}

{release.get('body', 'No release notes.')}

---

"""
                
                safe_repo = repository.replace("/", "_")
                release_file = out_dir / f"{safe_repo}_releases.md"
                with open(release_file, "w") as f:
                    f.write(release_content)
                
                results["files"].append(str(release_file))
                results["counts"]["releases"] = len(releases)
        
        # Save updated state
        export_state["last_global_export"] = now
        if repository not in export_state["repositories"]:
            export_state["repositories"][repository] = {}
        export_state["repositories"][repository]["last_export"] = now
        _save_state(export_state)
        
        # Build summary
        summary = f"""GitHub Export Complete - {repository}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exported at: {now}
"""
        for item_type, count in results["counts"].items():
            summary += f"  {item_type.title()}: {count}\n"
        
        summary += f"\nFiles created:\n"
        for f in results["files"]:
            summary += f"  📄 {f}\n"
        
        if results["exported"]:
            summary += f"\nSample items exported:\n"
            for item in results["exported"][:5]:
                summary += f"  • {item}\n"
        
        summary += f"\n💡 Use since='last' for incremental exports next time"
        
        return {"status": "success", "content": [{"text": summary}], "files": results["files"]}
        
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Export failed: {str(e)}"}]}
