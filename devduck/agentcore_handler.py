#!/usr/bin/env python3
"""DevDuck AgentCore Handler"""
import json
import os
import threading
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Configure for AgentCore deployment
os.environ["DEVDUCK_AUTO_START_SERVERS"] = "false"
os.environ["MODEL_PROVIDER"] = "bedrock"

from devduck import devduck

app = BedrockAgentCoreApp()


@app.entrypoint
async def invoke(payload, context):
    """AgentCore entrypoint - streaming by default with async generator"""
    mode = payload.get("mode", "streaming")  # streaming (default), sync, async

    query = payload.get("prompt", payload.get("text", ""))
    if not query:
        yield {"error": "No query provided"}
        return

    print(f"Mode: {mode}, Query: {query}")

    agent = devduck.agent

    if mode == "sync":
        # Sync mode - return result directly (blocking)
        try:
            result = agent(query)
            yield {"statusCode": 200, "response": str(result)}
        except Exception as e:
            print(f"Error in sync: {str(e)}")
            yield {"statusCode": 500, "error": str(e)}

    elif mode == "async":
        # Async mode - fire and forget in background thread
        task_id = app.add_async_task("devduck_processing", payload)
        thread = threading.Thread(
            target=lambda: _run_in_thread(agent, query, task_id), daemon=True
        )
        thread.start()
        yield {"statusCode": 200, "task_id": task_id}

    else:
        # Streaming mode (default) - stream events as they happen
        try:
            stream = agent.stream_async(query)
            async for event in stream:
                print(event)
                yield event
        except Exception as e:
            print(f"Error in streaming: {str(e)}")
            yield {"error": str(e)}


def _run_in_thread(agent, query, task_id):
    """Run agent in background thread for async mode"""
    try:
        result = agent(query)
        print(f"DevDuck result: {result}")
        app.complete_async_task(task_id)
    except Exception as e:
        print(f"Error in async thread: {str(e)}")
        try:
            app.complete_async_task(task_id)
        except:
            pass


if __name__ == "__main__":
    app.run()
