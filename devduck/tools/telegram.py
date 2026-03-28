"""Telegram tool for DevDuck agents - listen & send messages.

Follows the TCP tool pattern: each incoming Telegram message spawns a fresh
DevDuck instance that processes it with full tool access. Responses are sent
back to the originating chat automatically (when auto_reply=true).

Features:
- Background long-polling listener (one DevDuck per message)
- Send messages, photos, documents, polls, locations, etc.
- Event history stored as JSONL
- Auto-reply toggle via STRANDS_TELEGRAM_AUTO_REPLY env var
- Listen-only tag filtering via STRANDS_TELEGRAM_LISTEN_ONLY_TAG

Environment Variables:
    TELEGRAM_BOT_TOKEN: Required bot token from @BotFather
    STRANDS_TELEGRAM_AUTO_REPLY: "true" to auto-reply (default: "false")
    STRANDS_TELEGRAM_LISTEN_ONLY_TAG: Only process messages containing this tag
    TELEGRAM_ALLOWED_USERS: Comma-separated allowlist of user IDs or usernames
        Example: "149632499,cagataycali" or "149632499" or "cagataycali,other_user"
        If not set, all users are allowed (open access).
    TELEGRAM_DEFAULT_EVENT_COUNT: Context event count (default: 20)

Usage:
    # Start listening
    telegram(action="start_listener")

    # Send a message
    telegram(action="send_message", chat_id="123456", text="Hello!")

    # Send a photo
    telegram(action="send_photo", chat_id="123456", file_path="/tmp/img.png")

    # Get recent events
    telegram(action="get_recent_events", count=10)

    # Stop listener
    telegram(action="stop_listener")
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from strands import Agent, tool

logger = logging.getLogger(__name__)

# Event storage
EVENTS_DIR = Path.cwd() / "telegram_events"
EVENTS_FILE = EVENTS_DIR / "events.jsonl"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

# Global listener state
_LISTENER_STATE: Dict[str, Any] = {
    "running": False,
    "thread": None,
    "bot_info": None,
    "last_update_id": 0,
    "connections": 0,
    "start_time": None,
    "parent_agent": None,
}


def _get_bot_token() -> Optional[str]:
    return os.environ.get("TELEGRAM_BOT_TOKEN")


def _get_bot_info(bot_token: str) -> Dict:
    """Fetch bot info from Telegram API (cached)."""
    if _LISTENER_STATE["bot_info"]:
        return _LISTENER_STATE["bot_info"]
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{bot_token}/getMe", timeout=10
        )
        if resp.status_code == 200:
            _LISTENER_STATE["bot_info"] = resp.json().get("result", {})
        else:
            _LISTENER_STATE["bot_info"] = {}
    except Exception as e:
        logger.error(f"Error getting bot info: {e}")
        _LISTENER_STATE["bot_info"] = {}
    return _LISTENER_STATE["bot_info"]


def _store_event(event_data: Dict):
    """Append event to JSONL storage."""
    try:
        EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "event_type": "telegram_update",
            "payload": event_data,
            "timestamp": time.time(),
            "update_id": event_data.get("update_id"),
        }
        with open(EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Error storing telegram event: {e}")


def _get_recent_events(count: int = 20) -> List[Dict]:
    """Read last N events from JSONL."""
    if not EVENTS_FILE.exists():
        return []
    try:
        with open(EVENTS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()[-count:]
        events = []
        for line in lines:
            try:
                events.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
        return events
    except Exception as e:
        logger.error(f"Error reading telegram events: {e}")
        return []


def _parse_allowed_users() -> Optional[Dict[str, set]]:
    """Parse TELEGRAM_ALLOWED_USERS env var into sets of allowed IDs and usernames.

    Returns None if no allowlist is configured (open access).
    Returns {"ids": set of int, "usernames": set of lowercase str} otherwise.
    """
    raw = os.environ.get("TELEGRAM_ALLOWED_USERS", "").strip()
    if not raw:
        return None

    allowed_ids = set()
    allowed_usernames = set()

    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        # Try as numeric user ID first
        try:
            allowed_ids.add(int(entry))
        except ValueError:
            # Treat as username (strip leading @ if present)
            allowed_usernames.add(entry.lstrip("@").lower())

    return {"ids": allowed_ids, "usernames": allowed_usernames}


def _is_user_allowed(user: Dict) -> bool:
    """Check if a user is in the allowlist.

    If TELEGRAM_ALLOWED_USERS is not set, all users are allowed.
    Otherwise, the user's ID or username must match an entry.
    """
    allowlist = _parse_allowed_users()
    if allowlist is None:
        return True  # No allowlist = open access

    user_id = user.get("id")
    username = (user.get("username") or "").lower()

    # Check ID match
    if user_id and user_id in allowlist["ids"]:
        return True

    # Check username match
    if username and username in allowlist["usernames"]:
        return True

    logger.info(
        f"Telegram: blocked message from user_id={user_id} username=@{username} "
        f"(not in TELEGRAM_ALLOWED_USERS)"
    )
    return False


def _should_process(message: Dict, bot_token: str) -> bool:
    """Check if message should be processed."""
    if not message:
        return False
    # Skip bot messages
    if message.get("from", {}).get("is_bot"):
        return False
    # Skip own bot messages
    bot_info = _get_bot_info(bot_token)
    if message.get("from", {}).get("id") == bot_info.get("id"):
        return False
    # User allowlist filter
    if not _is_user_allowed(message.get("from", {})):
        return False
    # Tag filter
    tag = os.environ.get("STRANDS_TELEGRAM_LISTEN_ONLY_TAG")
    if tag and tag not in message.get("text", ""):
        return False
    return True


def _send_telegram_message(bot_token: str, chat_id, text: str, reply_to: int = None):
    """Send a text message via Telegram API."""
    try:
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        if reply_to:
            payload["reply_to_message_id"] = reply_to
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json=payload,
            timeout=30,
        )
        if resp.status_code != 200:
            logger.error(f"Telegram send error: {resp.text}")
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        logger.error(f"Error sending telegram message: {e}")
        return None


def _emit_event(event_type: str, summary: str, detail: str = "", metadata: dict = None):
    """Push event to the unified event bus (if available)."""
    try:
        from devduck.tools.event_bus import emit
        emit(event_type, "telegram", summary, detail, metadata)
    except ImportError:
        pass


def _process_message(message: Dict, bot_token: str):
    """Process a single message with a fresh DevDuck instance (TCP pattern)."""
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")
    user = message.get("from", {})
    message_id = message.get("message_id")

    user_name = user.get("first_name", "User")
    username = user.get("username", "")
    user_display = f"{user_name} (@{username})" if username else user_name

    _LISTENER_STATE["connections"] += 1
    logger.info(f"Telegram: processing message from {user_display} in chat {chat_id}")

    # 🔔 Emit incoming message event
    _emit_event(
        "telegram.in",
        f"{user_display}: {text[:60]}",
        text,
        {"chat_id": chat_id, "user": user_display, "message_id": message_id},
    )

    try:
        from devduck import DevDuck

        # Fresh DevDuck per message (like TCP tool)
        connection_devduck = DevDuck(auto_start_servers=False)
        agent = connection_devduck.agent

        if not agent:
            logger.error("Failed to create DevDuck instance for telegram message")
            return

        # Build context with recent events
        event_count = int(os.getenv("TELEGRAM_DEFAULT_EVENT_COUNT", "20"))
        recent = _get_recent_events(event_count)
        event_ctx = (
            f"\nRecent Telegram Events:\n{json.dumps(recent[-5:], indent=2)}"
            if recent
            else ""
        )

        # Inject telegram context into system prompt
        agent.system_prompt += f"""

## Telegram Integration Active:
- You are responding to a Telegram message
- Chat ID: {chat_id}
- User: {user_display}
- Use telegram tool to send messages back: telegram(action="send_message", chat_id="{chat_id}", text="your reply")
- You can also send photos, documents, polls, etc.
{event_ctx}
"""

        prompt = f"[Telegram Chat {chat_id}] {user_display} says: {text}"
        result = agent(prompt)

        # Auto-reply if enabled
        auto_reply = os.getenv("STRANDS_TELEGRAM_AUTO_REPLY", "false").lower() == "true"
        if auto_reply and result and str(result).strip():
            response_text = str(result).strip()
            # Truncate to Telegram limit
            if len(response_text) > 4096:
                response_text = response_text[:4093] + "..."
            _send_telegram_message(bot_token, chat_id, response_text, message_id)

    except Exception as e:
        logger.error(f"Error processing telegram message: {e}", exc_info=True)


def _polling_loop(bot_token: str):
    """Long-polling loop for Telegram updates."""
    logger.info("Telegram polling loop started")

    while _LISTENER_STATE["running"]:
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{bot_token}/getUpdates",
                params={
                    "offset": _LISTENER_STATE["last_update_id"] + 1,
                    "timeout": 30,
                    "limit": 100,
                },
                timeout=35,
            )

            if resp.status_code == 200:
                updates = resp.json().get("result", [])
                for update in updates:
                    _store_event(update)
                    _LISTENER_STATE["last_update_id"] = max(
                        _LISTENER_STATE["last_update_id"],
                        update.get("update_id", 0),
                    )
                    if "message" in update:
                        msg = update["message"]
                        if _should_process(msg, bot_token):
                            # Process in a thread (like TCP handle_client)
                            t = threading.Thread(
                                target=_process_message,
                                args=(msg, bot_token),
                                daemon=True,
                            )
                            t.start()
            else:
                logger.error(f"Telegram getUpdates error: {resp.text}")
                time.sleep(5)

        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            logger.error(f"Telegram polling error: {e}")
            time.sleep(5)

    logger.info("Telegram polling loop stopped")


def _telegram_api_call(
    bot_token: str,
    api_method: str,
    params: Dict = None,
    files: Dict = None,
) -> Dict:
    """Generic Telegram Bot API call."""
    url = f"https://api.telegram.org/bot{bot_token}/{api_method}"
    # Remove None values
    if params:
        params = {k: v for k, v in params.items() if v is not None}

    try:
        if files:
            resp = requests.post(url, data=params, files=files, timeout=30)
        else:
            resp = requests.post(url, json=params if params else None, timeout=30)

        if resp.status_code == 200:
            result = resp.json()
            if result.get("ok"):
                return {
                    "status": "success",
                    "content": [
                        {"text": f"✅ {api_method} successful"},
                        {"json": result.get("result", {})},
                    ],
                }
            else:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"❌ API error: {result.get('description', 'Unknown')}"
                        }
                    ],
                }
        else:
            return {
                "status": "error",
                "content": [{"text": f"❌ HTTP {resp.status_code}: {resp.text}"}],
            }
    except Exception as e:
        return {"status": "error", "content": [{"text": f"❌ Error: {e}"}]}
    finally:
        if files:
            for fh in files.values():
                if hasattr(fh, "close"):
                    fh.close()


@tool
def telegram(
    action: str,
    chat_id: Optional[Union[str, int]] = None,
    text: Optional[str] = None,
    message_id: Optional[int] = None,
    user_id: Optional[int] = None,
    file_path: Optional[str] = None,
    file_url: Optional[str] = None,
    inline_keyboard: Optional[List[List[Dict]]] = None,
    reply_markup: Optional[Dict] = None,
    parse_mode: Optional[str] = "HTML",
    disable_notification: bool = False,
    reply_to_message_id: Optional[int] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    phone_number: Optional[str] = None,
    first_name: Optional[str] = None,
    question: Optional[str] = None,
    options: Optional[List[str]] = None,
    emoji: Optional[str] = None,
    from_chat_id: Optional[Union[str, int]] = None,
    count: int = 20,
    custom_params: Optional[Dict] = None,
    method: Optional[str] = None,
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """Telegram integration for DevDuck - listen for messages and send responses.

    Each incoming message spawns a fresh DevDuck instance with full tool access.
    Also provides direct access to all major Telegram Bot API methods.

    Args:
        action: Action to perform:
            Listener:
            - "start_listener": Start background message polling
            - "stop_listener": Stop the listener
            - "listener_status": Get listener status
            - "get_recent_events": Get stored events

            Messages:
            - "send_message": Send text message
            - "send_photo": Send photo
            - "send_document": Send document
            - "send_video": Send video
            - "send_audio": Send audio
            - "send_voice": Send voice message
            - "send_location": Send location
            - "send_poll": Send poll
            - "send_dice": Send dice

            Management:
            - "edit_message": Edit message text
            - "delete_message": Delete a message
            - "forward_message": Forward a message
            - "pin_message": Pin a message
            - "get_me": Get bot info
            - "get_chat": Get chat info
            - "custom": Custom API method

        chat_id: Target chat ID
        text: Message text
        message_id: Message ID for edit/delete/pin
        file_path: Local file path for uploads
        file_url: URL of file to send
        inline_keyboard: Inline keyboard markup
        reply_markup: Full reply markup object
        parse_mode: Parse mode (HTML, Markdown, MarkdownV2)
        disable_notification: Send silently
        reply_to_message_id: Reply to specific message
        latitude: Latitude for location
        longitude: Longitude for location
        phone_number: Phone for contact
        first_name: Name for contact
        question: Poll question
        options: Poll options
        emoji: Dice emoji
        from_chat_id: Source chat for forwarding
        count: Number of recent events to retrieve
        custom_params: Params for custom API calls
        method: Custom API method name
        agent: Parent agent (auto-provided)

    Returns:
        Dict with status and content

    Environment Variables:
        TELEGRAM_BOT_TOKEN: Bot token from @BotFather
        STRANDS_TELEGRAM_AUTO_REPLY: "true" to enable auto-reply
        STRANDS_TELEGRAM_LISTEN_ONLY_TAG: Only process tagged messages
        TELEGRAM_ALLOWED_USERS: Comma-separated user IDs/usernames allowlist
            Example: "149632499,cagataycali" - if unset, all users allowed
    """
    bot_token = _get_bot_token()

    # --- Listener actions (no token needed for get_recent_events) ---
    if action == "get_recent_events":
        events = _get_recent_events(count)
        if events:
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"Recent {len(events)} Telegram events:\n{json.dumps(events, indent=2, ensure_ascii=False)}"
                    }
                ],
            }
        return {"status": "success", "content": [{"text": "No events found"}]}

    if action == "listener_status":
        allowlist = _parse_allowed_users()
        status = {
            "running": _LISTENER_STATE["running"],
            "bot_info": _LISTENER_STATE["bot_info"],
            "last_update_id": _LISTENER_STATE["last_update_id"],
            "messages_processed": _LISTENER_STATE["connections"],
            "events_file": str(EVENTS_FILE),
            "auto_reply": os.getenv("STRANDS_TELEGRAM_AUTO_REPLY", "false"),
            "listen_only_tag": os.getenv("STRANDS_TELEGRAM_LISTEN_ONLY_TAG", ""),
            "allowed_users": (
                {
                    "ids": sorted(allowlist["ids"]),
                    "usernames": sorted(allowlist["usernames"]),
                }
                if allowlist
                else "all (no restriction)"
            ),
            "uptime": (
                time.time() - _LISTENER_STATE["start_time"]
                if _LISTENER_STATE["start_time"]
                else 0
            ),
        }
        return {
            "status": "success",
            "content": [{"text": json.dumps(status, indent=2)}],
        }

    # All other actions need a token
    if not bot_token:
        return {
            "status": "error",
            "content": [
                {"text": "❌ TELEGRAM_BOT_TOKEN not set. Get one from @BotFather."}
            ],
        }

    # --- Listener management ---
    if action == "start_listener":
        if _LISTENER_STATE["running"]:
            return {
                "status": "success",
                "content": [{"text": "Telegram listener already running"}],
            }

        _LISTENER_STATE["running"] = True
        _LISTENER_STATE["connections"] = 0
        _LISTENER_STATE["start_time"] = time.time()
        _LISTENER_STATE["parent_agent"] = agent

        t = threading.Thread(target=_polling_loop, args=(bot_token,), daemon=True)
        t.start()
        _LISTENER_STATE["thread"] = t

        bot_info = _get_bot_info(bot_token)
        bot_name = bot_info.get("username", "unknown")

        return {
            "status": "success",
            "content": [
                {"text": f"✅ Telegram listener started (@{bot_name})"},
                {"text": "🦆 Each message spawns a fresh DevDuck instance"},
                {
                    "text": f"Auto-reply: {os.getenv('STRANDS_TELEGRAM_AUTO_REPLY', 'false')}"
                },
                {"text": f"Events stored: {EVENTS_FILE}"},
            ],
        }

    if action == "stop_listener":
        if not _LISTENER_STATE["running"]:
            return {
                "status": "success",
                "content": [{"text": "Telegram listener not running"}],
            }

        _LISTENER_STATE["running"] = False
        processed = _LISTENER_STATE["connections"]
        uptime = time.time() - (_LISTENER_STATE["start_time"] or time.time())

        return {
            "status": "success",
            "content": [
                {"text": "✅ Telegram listener stopped"},
                {
                    "text": f"Stats: {processed} messages processed, uptime {uptime:.0f}s"
                },
            ],
        }

    # --- Telegram API actions ---
    if action == "send_message":
        params = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
            "reply_to_message_id": reply_to_message_id,
        }
        if reply_markup:
            params["reply_markup"] = json.dumps(reply_markup)
        elif inline_keyboard:
            params["reply_markup"] = json.dumps({"inline_keyboard": inline_keyboard})
        result = _telegram_api_call(bot_token, "sendMessage", params)
        # 🔔 Emit outgoing message event
        if result.get("status") == "success":
            _emit_event("telegram.out", f"→ {chat_id}: {(text or '')[:60]}", text or "",
                        {"chat_id": chat_id})
        return result

    elif action == "send_photo":
        params = {
            "chat_id": chat_id,
            "caption": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
            "reply_to_message_id": reply_to_message_id,
        }
        files = {}
        if file_path:
            files["photo"] = open(file_path, "rb")
        elif file_url:
            params["photo"] = file_url
        return _telegram_api_call(bot_token, "sendPhoto", params, files or None)

    elif action == "send_document":
        params = {
            "chat_id": chat_id,
            "caption": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
            "reply_to_message_id": reply_to_message_id,
        }
        files = {}
        if file_path:
            files["document"] = open(file_path, "rb")
        elif file_url:
            params["document"] = file_url
        return _telegram_api_call(bot_token, "sendDocument", params, files or None)

    elif action == "send_video":
        params = {
            "chat_id": chat_id,
            "caption": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
            "reply_to_message_id": reply_to_message_id,
        }
        files = {}
        if file_path:
            files["video"] = open(file_path, "rb")
        elif file_url:
            params["video"] = file_url
        return _telegram_api_call(bot_token, "sendVideo", params, files or None)

    elif action == "send_audio":
        params = {
            "chat_id": chat_id,
            "caption": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
            "reply_to_message_id": reply_to_message_id,
        }
        files = {}
        if file_path:
            files["audio"] = open(file_path, "rb")
        elif file_url:
            params["audio"] = file_url
        return _telegram_api_call(bot_token, "sendAudio", params, files or None)

    elif action == "send_voice":
        params = {
            "chat_id": chat_id,
            "caption": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
            "reply_to_message_id": reply_to_message_id,
        }
        files = {}
        if file_path:
            files["voice"] = open(file_path, "rb")
        elif file_url:
            params["voice"] = file_url
        return _telegram_api_call(bot_token, "sendVoice", params, files or None)

    elif action == "send_location":
        params = {
            "chat_id": chat_id,
            "latitude": latitude,
            "longitude": longitude,
            "disable_notification": disable_notification,
            "reply_to_message_id": reply_to_message_id,
        }
        return _telegram_api_call(bot_token, "sendLocation", params)

    elif action == "send_poll":
        params = {
            "chat_id": chat_id,
            "question": question,
            "options": json.dumps(options or []),
            "disable_notification": disable_notification,
            "reply_to_message_id": reply_to_message_id,
        }
        return _telegram_api_call(bot_token, "sendPoll", params)

    elif action == "send_dice":
        params = {
            "chat_id": chat_id,
            "emoji": emoji or "🎲",
            "disable_notification": disable_notification,
        }
        return _telegram_api_call(bot_token, "sendDice", params)

    elif action == "edit_message":
        params = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_markup:
            params["reply_markup"] = json.dumps(reply_markup)
        elif inline_keyboard:
            params["reply_markup"] = json.dumps({"inline_keyboard": inline_keyboard})
        return _telegram_api_call(bot_token, "editMessageText", params)

    elif action == "delete_message":
        return _telegram_api_call(
            bot_token, "deleteMessage", {"chat_id": chat_id, "message_id": message_id}
        )

    elif action == "forward_message":
        return _telegram_api_call(
            bot_token,
            "forwardMessage",
            {
                "chat_id": chat_id,
                "from_chat_id": from_chat_id,
                "message_id": message_id,
                "disable_notification": disable_notification,
            },
        )

    elif action == "pin_message":
        return _telegram_api_call(
            bot_token,
            "pinChatMessage",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "disable_notification": disable_notification,
            },
        )

    elif action == "get_me":
        return _telegram_api_call(bot_token, "getMe")

    elif action == "get_chat":
        return _telegram_api_call(bot_token, "getChat", {"chat_id": chat_id})

    elif action == "custom":
        api_method = method or (custom_params.get("method") if custom_params else None)
        if not api_method:
            return {
                "status": "error",
                "content": [{"text": "❌ 'method' required for custom action"}],
            }
        return _telegram_api_call(bot_token, api_method, custom_params or {})

    else:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"❌ Unknown action: {action}. Use: start_listener, stop_listener, "
                    "listener_status, get_recent_events, send_message, send_photo, "
                    "send_document, send_video, send_audio, send_voice, send_location, "
                    "send_poll, send_dice, edit_message, delete_message, forward_message, "
                    "pin_message, get_me, get_chat, custom"
                }
            ],
        }
