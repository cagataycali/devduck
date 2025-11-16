"""AgentCore Agents Tool - List and manage deployed DevDuck agents."""

import json
from typing import Any, Dict, Optional
from strands import tool


@tool
def agentcore_agents(
    action: str = "list",
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    max_results: int = 100,
    region: str = "us-west-2",
) -> Dict[str, Any]:
    """Manage and discover Bedrock AgentCore agent runtimes.

    Args:
        action: Agent operation (list, get, find_by_name)
        agent_id: Agent runtime ID (for "get" action)
        agent_name: Agent name to search (for "find_by_name")
        max_results: Max results for list (default: 100)
        region: AWS region (default: us-west-2)

    Returns:
        Dict with status and agent information

    Examples:
        # List all agents
        agentcore_agents(action="list")

        # Get specific agent
        agentcore_agents(action="get", agent_id="devduck-UrvQvkH6H7")

        # Find by name
        agentcore_agents(action="find_by_name", agent_name="devduck")
    """
    try:
        import boto3
        from botocore.exceptions import ClientError

        client = boto3.client("bedrock-agentcore-control", region_name=region)

        if action == "list":
            all_agents = []
            next_token = None

            while True:
                params = {"maxResults": min(max_results - len(all_agents), 100)}
                if next_token:
                    params["nextToken"] = next_token

                response = client.list_agent_runtimes(**params)
                agents = response.get("agentRuntimes", [])
                all_agents.extend(agents)

                if len(all_agents) >= max_results:
                    all_agents = all_agents[:max_results]
                    break

                next_token = response.get("nextToken")
                if not next_token:
                    break

            # Format output
            agent_list = []
            for agent in all_agents:
                agent_list.append(
                    {
                        "name": agent.get("agentRuntimeName"),
                        "id": agent.get("agentRuntimeId"),
                        "arn": agent.get("agentRuntimeArn"),
                        "created": str(agent.get("createdAt")),
                        "updated": str(agent.get("lastUpdatedAt")),
                    }
                )

            return {
                "status": "success",
                "content": [
                    {"text": f"**Found {len(agent_list)} agents:**\n"},
                    {"text": json.dumps(agent_list, indent=2)},
                ],
            }

        elif action == "get":
            if not agent_id:
                return {
                    "status": "error",
                    "content": [{"text": "agent_id required for get action"}],
                }

            response = client.get_agent_runtime(agentRuntimeId=agent_id)

            agent_info = {
                "id": response.get("agentRuntimeId"),
                "arn": response.get("agentRuntimeArn"),
                "name": response.get("agentRuntimeName"),
                "status": response.get("status"),
                "roleArn": response.get("roleArn"),
                "created": str(response.get("createdAt")),
                "updated": str(response.get("lastUpdatedAt")),
            }

            # Add container info
            if "agentRuntimeArtifact" in response:
                container = response["agentRuntimeArtifact"].get(
                    "containerConfiguration", {}
                )
                if container:
                    agent_info["containerUri"] = container.get("containerUri")

            # Add network info
            if "networkConfiguration" in response:
                agent_info["networkMode"] = response["networkConfiguration"].get(
                    "networkMode"
                )

            return {
                "status": "success",
                "content": [
                    {"text": "**Agent Details:**\n"},
                    {"text": json.dumps(agent_info, indent=2)},
                ],
            }

        elif action == "find_by_name":
            if not agent_name:
                return {
                    "status": "error",
                    "content": [
                        {"text": "agent_name required for find_by_name action"}
                    ],
                }

            # List and search
            all_agents = []
            next_token = None

            while True:
                params = {"maxResults": 100}
                if next_token:
                    params["nextToken"] = next_token

                response = client.list_agent_runtimes(**params)
                agents = response.get("agentRuntimes", [])
                all_agents.extend(agents)

                next_token = response.get("nextToken")
                if not next_token:
                    break

            # Find match
            matching = None
            for agent in all_agents:
                if agent.get("agentRuntimeName") == agent_name:
                    matching = agent
                    break

            if matching:
                return {
                    "status": "success",
                    "content": [
                        {"text": f"✅ **Found: {agent_name}**\n"},
                        {"text": json.dumps(matching, indent=2, default=str)},
                    ],
                }
            else:
                return {
                    "status": "error",
                    "content": [{"text": f"Agent not found: {agent_name}"}],
                }

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Valid: list, get, find_by_name"
                    }
                ],
            }

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_message = e.response.get("Error", {}).get("Message", str(e))

        return {
            "status": "error",
            "content": [
                {"text": f"**AWS Error ({error_code}):** {error_message}"},
                {"text": f"**Region:** {region}"},
            ],
        }

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}
