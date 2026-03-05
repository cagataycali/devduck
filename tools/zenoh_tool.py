"""Zenoh tool for DevDuck agents with real-time pub/sub/query support + message history."""

import logging
import threading
import time
import re
from typing import Any, Optional
from collections import deque

from strands import Agent, tool

logger = logging.getLogger(__name__)

# Global registry to store Zenoh sessions
ZENOH_SESSIONS: dict[str, dict[str, Any]] = {}


class ZenohSubscriberCallbackHandler:
    """Callback handler with background storage and conditional triggering."""

    def __init__(
        self,
        session_id: str,
        key: str,
        system_prompt: str,
        trigger_pattern: Optional[str] = None,
        store_history: bool = True,
        max_history: int = 1000,
    ):
        """Initialize the handler.

        Args:
            session_id: Session identifier
            key: Key expression being subscribed to
            system_prompt: System prompt for DevDuck instance
            trigger_pattern: Regex pattern to trigger agent (None = no trigger)
            store_history: Whether to store messages in history buffer
            max_history: Maximum messages to store (FIFO buffer)
        """
        self.session_id = session_id
        self.key = key
        self.system_prompt = system_prompt
        self.trigger_pattern = re.compile(trigger_pattern) if trigger_pattern else None
        self.store_history = store_history
        self.message_count = 0
        self.trigger_count = 0
        
        # Message history buffer (thread-safe deque)
        self.message_buffer = deque(maxlen=max_history) if store_history else None
        self.buffer_lock = threading.Lock()
        
        # DevDuck instance (lazy init on first trigger)
        self.devduck_agent = None

    def _init_devduck(self):
        """Lazy initialize DevDuck agent (only when needed for triggering)."""
        if self.devduck_agent:
            return
            
        try:
            from devduck import DevDuck
            connection_devduck = DevDuck(auto_start_servers=False)
            if connection_devduck.agent and self.system_prompt:
                connection_devduck.agent.system_prompt += (
                    "\n\nZenoh Subscriber Context:\n"
                    f"- Session: {self.session_id}\n"
                    f"- Subscribed to: {self.key}\n"
                    f"- Trigger pattern: {self.trigger_pattern.pattern if self.trigger_pattern else 'none'}\n"
                    f"- System prompt: {self.system_prompt}"
                )
            self.devduck_agent = connection_devduck.agent
            logger.info(f"DevDuck agent initialized for subscriber on '{self.key}'")
        except Exception as e:
            logger.error(f"Failed to create DevDuck for subscriber: {e}")

    def _process_with_agent(self, sample_data: dict):
        """Process sample with DevDuck agent (called in background thread)."""
        try:
            if not self.devduck_agent:
                self._init_devduck()
            
            if self.devduck_agent:
                query = (
                    f"Received Zenoh message #{sample_data['msg_num']} (triggered):\n"
                    f"Key: {sample_data['key']}\n"
                    f"Payload: {sample_data['payload']}\n"
                    f"Kind: {sample_data['kind']}\n\n"
                    f"Please process this message."
                )
                self.devduck_agent(query)
                logger.info(f"Agent processed trigger #{self.trigger_count} for key '{sample_data['key']}'")
        except Exception as e:
            logger.error(f"Error in agent processing: {e}")

    def __call__(self, sample):
        """Handle incoming Zenoh sample - fast storage, optional async trigger."""
        self.message_count += 1
        
        try:
            payload_str = sample.payload.to_string()
            key_expr = str(sample.key_expr)
            kind_str = str(sample.kind)
            timestamp = time.time()
            
            # 1. Fast background storage (non-blocking)
            if self.store_history:
                with self.buffer_lock:
                    self.message_buffer.append({
                        'timestamp': timestamp,
                        'msg_num': self.message_count,
                        'key': key_expr,
                        'payload': payload_str,
                        'kind': kind_str,
                    })
            
            # 2. Check trigger pattern (optional)
            should_trigger = False
            if self.trigger_pattern:
                should_trigger = self.trigger_pattern.search(key_expr) is not None
            
            if should_trigger:
                self.trigger_count += 1
                # Async processing in background thread (non-blocking)
                sample_data = {
                    'timestamp': timestamp,
                    'msg_num': self.message_count,
                    'key': key_expr,
                    'payload': payload_str,
                    'kind': kind_str,
                }
                threading.Thread(
                    target=self._process_with_agent,
                    args=(sample_data,),
                    daemon=True
                ).start()
                logger.info(f"Triggered agent processing for '{key_expr}' (trigger #{self.trigger_count})")
            
            # Log every 100 messages to show it's working
            if self.message_count % 100 == 0:
                logger.info(f"Subscriber on '{self.key}': {self.message_count} messages received, {self.trigger_count} triggers")
                
        except Exception as e:
            logger.error(f"Error in Zenoh callback: {e}")

    def get_history(self, limit: Optional[int] = None) -> list[dict]:
        """Get message history (last N messages).
        
        Args:
            limit: Number of messages to return (None = all)
            
        Returns:
            List of message dicts with timestamp, key, payload, kind
        """
        if not self.store_history or not self.message_buffer:
            return []
        
        with self.buffer_lock:
            if limit is None:
                return list(self.message_buffer)
            else:
                # Get last N messages
                buffer_list = list(self.message_buffer)
                return buffer_list[-limit:] if len(buffer_list) > limit else buffer_list

    def clear_history(self):
        """Clear message history buffer."""
        if self.message_buffer:
            with self.buffer_lock:
                self.message_buffer.clear()
            logger.info(f"Cleared history buffer for subscriber on '{self.key}'")


class ZenohQueryableCallbackHandler:
    """Callback handler for Zenoh queryables."""

    def __init__(self, session_id: str, key: str, system_prompt: str, default_payload: str):
        self.session_id = session_id
        self.key = key
        self.system_prompt = system_prompt
        self.default_payload = default_payload
        self.query_count = 0
        self.devduck_agent = None

        try:
            from devduck import DevDuck
            connection_devduck = DevDuck(auto_start_servers=False)
            if connection_devduck.agent and system_prompt:
                connection_devduck.agent.system_prompt += (
                    "\n\nZenoh Queryable Context:\n"
                    f"- Session: {session_id}\n"
                    f"- Queryable key: {key}\n"
                    f"- System prompt: {system_prompt}"
                )
            self.devduck_agent = connection_devduck.agent
            logger.info(f"Created DevDuck instance for queryable on '{key}'")
        except Exception as e:
            logger.error(f"Failed to create DevDuck for queryable: {e}")

    def __call__(self, query):
        self.query_count += 1
        try:
            selector = str(query.selector)
            query_payload = None
            
            if query.payload:
                query_payload = query.payload.to_string()

            logger.info(f"Queryable on '{self.key}' received query: '{selector}'")

            reply_payload = self.default_payload

            if self.devduck_agent and query_payload:
                devduck_query = (
                    f"Received Zenoh query #{self.query_count}:\n"
                    f"Selector: {selector}\n"
                    f"Query payload: {query_payload}\n\n"
                    f"Please generate a response."
                )
                response = self.devduck_agent(devduck_query)
                if response:
                    reply_payload = str(response)

            query.reply(self.key, reply_payload)

        except Exception as e:
            logger.error(f"Error processing Zenoh query: {e}")


def run_zenoh_session(session_id: str, config: Optional[dict], mode: str):
    """Run a Zenoh session in background thread."""
    try:
        import zenoh

        zenoh.init_log_from_env_or("error")

        if config:
            zenoh_config = zenoh.Config(config)
        else:
            zenoh_config = zenoh.Config()

        if mode == "router":
            zenoh_config.insert_json5("mode", '"router"')
        else:
            zenoh_config.insert_json5("mode", '"peer"')

        logger.info(f"Opening Zenoh session '{session_id}' in {mode} mode...")
        session = zenoh.open(zenoh_config)

        ZENOH_SESSIONS[session_id]["session"] = session
        ZENOH_SESSIONS[session_id]["running"] = True
        ZENOH_SESSIONS[session_id]["start_time"] = time.time()

        logger.info(f"Zenoh session '{session_id}' opened successfully")

        while ZENOH_SESSIONS[session_id]["running"]:
            time.sleep(1.0)

        session.close()
        logger.info(f"Zenoh session '{session_id}' closed")

    except Exception as e:
        logger.error(f"Error in Zenoh session '{session_id}': {e}")
        ZENOH_SESSIONS[session_id]["running"] = False


@tool
def zenoh(
    action: str,
    session_id: str = "main",
    mode: str = "peer",
    config: Optional[dict] = None,
    key: str = "",
    payload: str = "",
    selector: str = "",
    timeout: float = 10.0,
    system_prompt: str = "You are a helpful Zenoh assistant.",
    complete: bool = False,
    trigger_pattern: Optional[str] = None,
    store_history: bool = True,
    max_history: int = 1000,
    limit: int = 100,
) -> dict:
    """Zenoh protocol tool with background storage and conditional triggering.

    **NEW Features:**
    - **Background Storage**: Messages stored fast in memory (non-blocking)
    - **Conditional Triggering**: Agent only processes on pattern match
    - **History Query**: Read last N messages anytime
    - **High-Frequency Friendly**: Handle 100+ msg/sec streams (e.g., Reachy Mini)

    **Actions:**
    - start_session: Create Zenoh session
    - stop_session: Stop session
    - publish: Publish message
    - subscribe: Subscribe with storage + optional trigger
    - query: Query for data
    - queryable: Register queryable
    - get_status: Session status
    - list_sessions: List all sessions
    - **get_history**: Retrieve stored messages (NEW)
    - **clear_history**: Clear message buffer (NEW)

    **Subscribe with Trigger Pattern:**
    ```python
    # Store all reachy_mini messages, only trigger agent for daemon_status
    zenoh(
        action="subscribe",
        key="reachy_mini/**",
        trigger_pattern="daemon_status",  # regex pattern
        store_history=True,
        max_history=5000
    )
    ```

    **Query History:**
    ```python
    # Get last 100 messages from specific topic
    zenoh(
        action="get_history",
        session_id="main",
        key="reachy_mini/joint_positions",
        limit=100
    )
    ```

    Args:
        action: Action to perform
        session_id: Session identifier
        mode: Session mode ("peer" or "router")
        config: Zenoh config dict
        key: Key expression for pub/sub
        payload: Payload data
        selector: Selector for queries
        timeout: Query timeout in seconds
        system_prompt: System prompt for DevDuck instances
        complete: Mark queryable as complete
        trigger_pattern: Regex pattern to trigger agent (None = no trigger)
        store_history: Store messages in buffer (default: True)
        max_history: Max messages to store (default: 1000)
        limit: Number of messages to return for get_history (default: 100)

    Returns:
        Dict with status and content
    """
    if action == "start_session":
        if session_id in ZENOH_SESSIONS and ZENOH_SESSIONS[session_id].get("running", False):
            return {
                "status": "error",
                "content": [{"text": f"❌ Error: Zenoh session '{session_id}' already running"}],
            }

        ZENOH_SESSIONS[session_id] = {
            "running": False,
            "publishers": {},
            "subscribers": {},
            "queryables": {},
        }

        session_thread = threading.Thread(
            target=run_zenoh_session,
            args=(session_id, config, mode),
        )
        session_thread.daemon = True
        session_thread.start()

        time.sleep(1.0)

        if not ZENOH_SESSIONS[session_id].get("running", False):
            return {
                "status": "error",
                "content": [{"text": f"❌ Error: Failed to start Zenoh session '{session_id}'"}],
            }

        return {
            "status": "success",
            "content": [
                {"text": f"✅ Zenoh session '{session_id}' started successfully in {mode} mode"},
                {"text": "🌐 Auto-discovery enabled for local network peers"},
                {"text": f"📝 Use session_id='{session_id}' for operations"},
            ],
        }

    elif action == "stop_session":
        if session_id not in ZENOH_SESSIONS or not ZENOH_SESSIONS[session_id].get("running", False):
            return {
                "status": "error",
                "content": [{"text": f"❌ Error: No Zenoh session '{session_id}' running"}],
            }

        ZENOH_SESSIONS[session_id]["running"] = False
        time.sleep(1.0)

        uptime = time.time() - ZENOH_SESSIONS[session_id].get("start_time", time.time())
        pub_count = len(ZENOH_SESSIONS[session_id].get("publishers", {}))
        sub_count = len(ZENOH_SESSIONS[session_id].get("subscribers", {}))
        queryable_count = len(ZENOH_SESSIONS[session_id].get("queryables", {}))

        del ZENOH_SESSIONS[session_id]

        return {
            "status": "success",
            "content": [
                {"text": f"✅ Zenoh session '{session_id}' stopped successfully"},
                {
                    "text": f"Statistics: {pub_count} publishers, {sub_count} subscribers, "
                    f"{queryable_count} queryables, uptime {uptime:.2f}s"
                },
            ],
        }

    elif action == "publish":
        if session_id not in ZENOH_SESSIONS or not ZENOH_SESSIONS[session_id].get("running", False):
            return {
                "status": "error",
                "content": [{"text": f"❌ Error: No Zenoh session '{session_id}' running"}],
            }

        if not key:
            return {
                "status": "error",
                "content": [{"text": "❌ Error: 'key' parameter required for publish"}],
            }

        try:
            session = ZENOH_SESSIONS[session_id]["session"]
            session.put(key, payload)
            
            ZENOH_SESSIONS[session_id]["publishers"][key] = {
                "last_publish": time.time(),
                "payload": payload,
            }

            logger.info(f"Published to '{key}': '{payload}'")

            return {
                "status": "success",
                "content": [
                    {"text": f"✅ Published to key '{key}'"},
                    {"text": f"Payload: {payload}"},
                ],
            }
        except Exception as e:
            logger.error(f"Error publishing: {e}")
            return {
                "status": "error",
                "content": [{"text": f"❌ Error publishing: {str(e)}"}],
            }

    elif action == "subscribe":
        if session_id not in ZENOH_SESSIONS or not ZENOH_SESSIONS[session_id].get("running", False):
            return {
                "status": "error",
                "content": [{"text": f"❌ Error: No Zenoh session '{session_id}' running"}],
            }

        if not key:
            return {
                "status": "error",
                "content": [{"text": "❌ Error: 'key' parameter required for subscribe"}],
            }

        if key in ZENOH_SESSIONS[session_id]["subscribers"]:
            return {
                "status": "error",
                "content": [{"text": f"❌ Error: Already subscribed to '{key}'"}],
            }

        try:
            session = ZENOH_SESSIONS[session_id]["session"]

            callback_handler = ZenohSubscriberCallbackHandler(
                session_id=session_id,
                key=key,
                system_prompt=system_prompt,
                trigger_pattern=trigger_pattern,
                store_history=store_history,
                max_history=max_history,
            )

            subscriber = session.declare_subscriber(key, callback_handler)

            ZENOH_SESSIONS[session_id]["subscribers"][key] = {
                "subscriber": subscriber,
                "handler": callback_handler,
                "created_at": time.time(),
                "trigger_pattern": trigger_pattern,
                "store_history": store_history,
                "max_history": max_history,
            }

            logger.info(f"Subscribed to '{key}' with trigger_pattern={trigger_pattern}, store_history={store_history}")

            trigger_info = f"Trigger: {trigger_pattern}" if trigger_pattern else "Trigger: none (storage only)"
            storage_info = f"History: {max_history} messages" if store_history else "History: disabled"

            return {
                "status": "success",
                "content": [
                    {"text": f"✅ Subscribed to key '{key}'"},
                    {"text": trigger_info},
                    {"text": storage_info},
                    {"text": "Messages are being stored in background (non-blocking)"},
                ],
            }
        except Exception as e:
            logger.error(f"Error subscribing: {e}")
            return {
                "status": "error",
                "content": [{"text": f"❌ Error subscribing: {str(e)}"}],
            }

    elif action == "get_history":
        if session_id not in ZENOH_SESSIONS or not ZENOH_SESSIONS[session_id].get("running", False):
            return {
                "status": "error",
                "content": [{"text": f"❌ Error: No Zenoh session '{session_id}' running"}],
            }

        if not key:
            return {
                "status": "error",
                "content": [{"text": "❌ Error: 'key' parameter required for get_history"}],
            }

        if key not in ZENOH_SESSIONS[session_id]["subscribers"]:
            return {
                "status": "error",
                "content": [{"text": f"❌ Error: Not subscribed to '{key}'"}],
            }

        try:
            handler = ZENOH_SESSIONS[session_id]["subscribers"][key]["handler"]
            history = handler.get_history(limit=limit)

            if not history:
                return {
                    "status": "success",
                    "content": [{"text": f"No history available for '{key}'"}],
                }

            # Format history nicely
            history_text = f"History for '{key}' (last {len(history)} messages):\n\n"
            for msg in history:
                timestamp_str = time.strftime("%H:%M:%S", time.localtime(msg['timestamp']))
                history_text += f"[{timestamp_str}] #{msg['msg_num']} {msg['key']}\n"
                history_text += f"  {msg['payload'][:200]}{'...' if len(msg['payload']) > 200 else ''}\n\n"

            return {
                "status": "success",
                "content": [
                    {"text": history_text},
                    {"text": f"Total: {handler.message_count} received, {handler.trigger_count} triggered"},
                ],
            }
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return {
                "status": "error",
                "content": [{"text": f"❌ Error getting history: {str(e)}"}],
            }

    elif action == "clear_history":
        if session_id not in ZENOH_SESSIONS or not ZENOH_SESSIONS[session_id].get("running", False):
            return {
                "status": "error",
                "content": [{"text": f"❌ Error: No Zenoh session '{session_id}' running"}],
            }

        if not key:
            return {
                "status": "error",
                "content": [{"text": "❌ Error: 'key' parameter required for clear_history"}],
            }

        if key not in ZENOH_SESSIONS[session_id]["subscribers"]:
            return {
                "status": "error",
                "content": [{"text": f"❌ Error: Not subscribed to '{key}'"}],
            }

        try:
            handler = ZENOH_SESSIONS[session_id]["subscribers"][key]["handler"]
            handler.clear_history()

            return {
                "status": "success",
                "content": [{"text": f"✅ Cleared history for '{key}'"}],
            }
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return {
                "status": "error",
                "content": [{"text": f"❌ Error clearing history: {str(e)}"}],
            }

    elif action == "query":
        if session_id not in ZENOH_SESSIONS or not ZENOH_SESSIONS[session_id].get("running", False):
            return {
                "status": "error",
                "content": [{"text": f"❌ Error: No Zenoh session '{session_id}' running"}],
            }

        if not selector:
            return {
                "status": "error",
                "content": [{"text": "❌ Error: 'selector' parameter required for query"}],
            }

        try:
            session = ZENOH_SESSIONS[session_id]["session"]
            replies = session.get(selector, timeout=timeout)

            results = []
            for reply in replies:
                if reply.is_ok:
                    sample = reply.ok
                    results.append({
                        "key": str(sample.key_expr),
                        "payload": sample.payload.to_string(),
                    })

            logger.info(f"Query '{selector}' returned {len(results)} results")

            if not results:
                return {
                    "status": "success",
                    "content": [{"text": f"Query '{selector}' returned no results"}],
                }

            results_text = f"Query '{selector}' results:\n\n"
            for r in results:
                results_text += f"Key: {r['key']}\nPayload: {r['payload']}\n\n"

            return {
                "status": "success",
                "content": [{"text": results_text}],
            }
        except Exception as e:
            logger.error(f"Error querying: {e}")
            return {
                "status": "error",
                "content": [{"text": f"❌ Error querying: {str(e)}"}],
            }

    elif action == "queryable":
        if session_id not in ZENOH_SESSIONS or not ZENOH_SESSIONS[session_id].get("running", False):
            return {
                "status": "error",
                "content": [{"text": f"❌ Error: No Zenoh session '{session_id}' running"}],
            }

        if not key:
            return {
                "status": "error",
                "content": [{"text": "❌ Error: 'key' parameter required for queryable"}],
            }

        if key in ZENOH_SESSIONS[session_id]["queryables"]:
            return {
                "status": "error",
                "content": [{"text": f"❌ Error: Queryable already registered for '{key}'"}],
            }

        try:
            session = ZENOH_SESSIONS[session_id]["session"]

            callback_handler = ZenohQueryableCallbackHandler(
                session_id=session_id,
                key=key,
                system_prompt=system_prompt,
                default_payload=payload,
            )

            queryable = session.declare_queryable(key, callback_handler, complete=complete)

            ZENOH_SESSIONS[session_id]["queryables"][key] = {
                "queryable": queryable,
                "handler": callback_handler,
                "created_at": time.time(),
            }

            logger.info(f"Registered queryable on '{key}'")

            return {
                "status": "success",
                "content": [
                    {"text": f"✅ Queryable registered on key '{key}'"},
                    {"text": f"Default payload: {payload}"},
                ],
            }
        except Exception as e:
            logger.error(f"Error creating queryable: {e}")
            return {
                "status": "error",
                "content": [{"text": f"❌ Error creating queryable: {str(e)}"}],
            }

    elif action == "get_status":
        if session_id not in ZENOH_SESSIONS or not ZENOH_SESSIONS[session_id].get("running", False):
            return {
                "status": "error",
                "content": [{"text": f"❌ Error: No Zenoh session '{session_id}' running"}],
            }

        session_data = ZENOH_SESSIONS[session_id]
        uptime = time.time() - session_data.get("start_time", time.time())

        status_text = f"Zenoh session '{session_id}' status:\n\n"
        status_text += f"Uptime: {uptime:.2f}s\n"
        status_text += f"Publishers: {len(session_data.get('publishers', {}))}\n"
        status_text += f"Subscribers: {len(session_data.get('subscribers', {}))}\n"
        status_text += f"Queryables: {len(session_data.get('queryables', {}))}\n\n"

        # Subscriber details with message counts
        if session_data.get("subscribers"):
            status_text += "Subscriber details:\n"
            for key, sub_data in session_data["subscribers"].items():
                handler = sub_data["handler"]
                status_text += f"  • {key}\n"
                status_text += f"    - Messages: {handler.message_count}\n"
                status_text += f"    - Triggers: {handler.trigger_count}\n"
                status_text += f"    - History: {len(handler.message_buffer) if handler.message_buffer else 0} stored\n"

        return {
            "status": "success",
            "content": [{"text": status_text}],
        }

    elif action == "list_sessions":
        if not ZENOH_SESSIONS:
            return {
                "status": "success",
                "content": [{"text": "No active Zenoh sessions"}],
            }

        sessions_text = f"Active Zenoh sessions ({len(ZENOH_SESSIONS)}):\n\n"
        for sid, data in ZENOH_SESSIONS.items():
            running = data.get("running", False)
            status = "✅ running" if running else "❌ stopped"
            uptime = time.time() - data.get("start_time", time.time()) if running else 0
            sessions_text += f"  • {sid} - {status} (uptime: {uptime:.2f}s)\n"

        return {
            "status": "success",
            "content": [{"text": sessions_text}],
        }

    else:
        return {
            "status": "error",
            "content": [{"text": f"❌ Unknown action: {action}"}],
        }
