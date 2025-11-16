"""AgentCore Logs Tool - View CloudWatch logs from deployed DevDuck instances."""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from strands import tool


@tool
def agentcore_logs(
    action: str = "recent",
    agent_name: str = "devduck",
    limit: int = 50,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    filter_pattern: Optional[str] = None,
    log_stream_name: Optional[str] = None,
    endpoint: str = "DEFAULT",
    region: str = "us-west-2",
) -> Dict[str, Any]:
    """View CloudWatch logs from deployed DevDuck instances on AgentCore.

    Args:
        action: Log operation (recent, streams, search, tail)
        agent_name: Name of deployed agent (default: devduck)
        limit: Max events/streams to return (default: 50)
        start_time: Start time in ISO format (default: last hour)
        end_time: End time in ISO format (default: now)
        filter_pattern: CloudWatch filter pattern for search
        log_stream_name: Specific stream to tail
        endpoint: Endpoint qualifier (default: DEFAULT)
        region: AWS region (default: us-west-2)

    Returns:
        Dict with status and log content

    Examples:
        # Recent logs
        agentcore_logs(agent_name="devduck")

        # Search for errors
        agentcore_logs(
            action="search",
            filter_pattern="ERROR"
        )

        # List streams
        agentcore_logs(action="streams")

        # Tail specific stream
        agentcore_logs(
            action="tail",
            log_stream_name="2025/11/16/[runtime-logs]session-abc"
        )
    """
    try:
        import boto3
        import yaml
        from pathlib import Path
        from botocore.exceptions import ClientError

        # Load config to get agent ID
        devduck_dir = Path(__file__).parent.parent
        config_path = devduck_dir / ".bedrock_agentcore.yaml"

        if not config_path.exists():
            return {
                "status": "error",
                "content": [
                    {"text": "Agent not configured. Run agentcore_launch() first."}
                ],
            }

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Get agent ID
        if "agents" not in config or agent_name not in config["agents"]:
            return {
                "status": "error",
                "content": [{"text": f"Agent '{agent_name}' not found in config"}],
            }

        agent_id = (
            config["agents"][agent_name].get("bedrock_agentcore", {}).get("agent_id")
        )

        if not agent_id:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Agent '{agent_name}' not deployed. Run agentcore_launch()."
                    }
                ],
            }

        # CloudWatch client
        logs_client = boto3.client("logs", region_name=region)

        # Build log group name
        log_group_name = f"/aws/bedrock-agentcore/runtimes/{agent_id}-{endpoint}"

        # Route to appropriate handler
        if action == "recent":
            return _get_recent_logs(
                logs_client, log_group_name, limit, start_time, end_time, filter_pattern
            )
        elif action == "streams":
            return _list_log_streams(logs_client, log_group_name, limit)
        elif action == "search":
            if not filter_pattern:
                return {
                    "status": "error",
                    "content": [{"text": "filter_pattern required for search"}],
                }
            return _search_logs(
                logs_client, log_group_name, filter_pattern, limit, start_time, end_time
            )
        elif action == "tail":
            if not log_stream_name:
                log_stream_name = _get_latest_stream(logs_client, log_group_name)
                if not log_stream_name:
                    return {
                        "status": "error",
                        "content": [{"text": "No log streams found"}],
                    }
            return _tail_logs(logs_client, log_group_name, log_stream_name, limit)
        else:
            return {
                "status": "error",
                "content": [{"text": f"Unknown action: {action}"}],
            }

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "ResourceNotFoundException":
            return {
                "status": "error",
                "content": [
                    {"text": f"Log group not found: {log_group_name}"},
                    {"text": "Agent may not be deployed or hasn't logged yet"},
                ],
            }
        return {"status": "error", "content": [{"text": f"AWS Error: {str(e)}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}


def _get_recent_logs(
    client,
    log_group_name: str,
    limit: int,
    start_time: Optional[str],
    end_time: Optional[str],
    filter_pattern: Optional[str],
) -> Dict[str, Any]:
    """Get recent log events."""
    params = {
        "logGroupName": log_group_name,
        "limit": limit,
        "interleaved": True,
    }

    if start_time:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        params["startTime"] = int(start_dt.timestamp() * 1000)
    else:
        # Default: last 5 minutes for truly recent logs
        params["startTime"] = int(
            (datetime.now() - timedelta(minutes=5)).timestamp() * 1000
        )

    if end_time:
        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        params["endTime"] = int(end_dt.timestamp() * 1000)

    if filter_pattern:
        params["filterPattern"] = filter_pattern

    response = client.filter_log_events(**params)
    events = response.get("events", [])

    if not events:
        return {
            "status": "success",
            "content": [{"text": "No log events found in last 5 minutes"}],
        }

    # Sort by timestamp descending (most recent first)
    events.sort(key=lambda e: e["timestamp"], reverse=True)

    # Format logs
    log_lines = []
    for event in events:
        timestamp = datetime.fromtimestamp(event["timestamp"] / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        message = event["message"].rstrip()
        log_lines.append(f"[{timestamp}] {message}")

    return {
        "status": "success",
        "content": [
            {"text": f"Found {len(events)} log events:\n"},
            {"text": "\n".join(log_lines)},
        ],
    }


def _list_log_streams(client, log_group_name: str, limit: int) -> Dict[str, Any]:
    """List log streams."""
    response = client.describe_log_streams(
        logGroupName=log_group_name,
        orderBy="LastEventTime",
        descending=True,
        limit=limit,
    )

    streams = response.get("logStreams", [])

    if not streams:
        return {"status": "success", "content": [{"text": "No log streams found"}]}

    stream_lines = [f"Found {len(streams)} log streams:\n"]
    for stream in streams:
        stream_name = stream["logStreamName"]
        last_event = datetime.fromtimestamp(
            stream.get("lastEventTimestamp", 0) / 1000
        ).strftime("%Y-%m-%d %H:%M:%S")
        stream_lines.append(f"• {stream_name} (last: {last_event})")

    return {"status": "success", "content": [{"text": "\n".join(stream_lines)}]}


def _search_logs(
    client,
    log_group_name: str,
    filter_pattern: str,
    limit: int,
    start_time: Optional[str],
    end_time: Optional[str],
) -> Dict[str, Any]:
    """Search logs with pattern."""
    params = {
        "logGroupName": log_group_name,
        "filterPattern": filter_pattern,
        "limit": limit,
        "interleaved": True,
    }

    if start_time:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        params["startTime"] = int(start_dt.timestamp() * 1000)
    else:
        params["startTime"] = int(
            (datetime.now() - timedelta(hours=24)).timestamp() * 1000
        )

    if end_time:
        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        params["endTime"] = int(end_dt.timestamp() * 1000)

    response = client.filter_log_events(**params)
    events = response.get("events", [])

    if not events:
        return {
            "status": "success",
            "content": [{"text": f"No matches for pattern: {filter_pattern}"}],
        }

    log_lines = [f"Found {len(events)} matches:\n"]
    for event in events:
        timestamp = datetime.fromtimestamp(event["timestamp"] / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        message = event["message"].rstrip()
        log_lines.append(f"[{timestamp}] {message}")

    return {"status": "success", "content": [{"text": "\n".join(log_lines)}]}


def _tail_logs(
    client, log_group_name: str, log_stream_name: str, limit: int
) -> Dict[str, Any]:
    """Tail specific log stream."""
    response = client.get_log_events(
        logGroupName=log_group_name,
        logStreamName=log_stream_name,
        limit=limit,
        startFromHead=False,
    )

    events = response.get("events", [])

    if not events:
        return {
            "status": "success",
            "content": [{"text": f"No events in stream: {log_stream_name}"}],
        }

    log_lines = [f"Latest {len(events)} events:\n"]
    for event in events:
        timestamp = datetime.fromtimestamp(event["timestamp"] / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        message = event["message"].rstrip()
        log_lines.append(f"[{timestamp}] {message}")

    return {"status": "success", "content": [{"text": "\n".join(log_lines)}]}


def _get_latest_stream(client, log_group_name: str) -> Optional[str]:
    """Get latest log stream."""
    response = client.describe_log_streams(
        logGroupName=log_group_name, orderBy="LastEventTime", descending=True, limit=1
    )

    streams = response.get("logStreams", [])
    return streams[0]["logStreamName"] if streams else None
