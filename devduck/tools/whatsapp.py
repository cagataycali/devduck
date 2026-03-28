"""WhatsApp tool for DevDuck agents via wacli CLI.

Wraps steipete/wacli (https://github.com/steipete/wacli) — a Go-based WhatsApp
CLI using the WhatsApp Web protocol (whatsmeow). No Cloud API / Business account
needed. Authenticate once with a QR code, then sync, search, send from the CLI.

Features:
- Send text & file messages
- List/search chats, messages, contacts, groups
- Background sync with --follow (continuous capture)
- History backfill for older messages
- Media download
- Full offline search (FTS5)
- Listener mode: poll for new messages → spawn DevDuck per message

Prerequisites:
    brew install steipete/tap/wacli
    wacli auth   # scan QR code once

Environment Variables:
    WACLI_STORE: Override store directory (default: ~/.wacli)
    WACLI_BINARY: Override wacli binary path (default: "wacli")
    STRANDS_WHATSAPP_AUTO_REPLY: "true" to auto-reply to incoming messages
    WHATSAPP_ALLOWED_SENDERS: Comma-separated phone/JID allowlist
        Example: "+14155551234,1234567890@s.whatsapp.net"
        If unset, all senders allowed (open access).
    WHATSAPP_POLL_INTERVAL: Seconds between poll cycles (default: 5)
    WHATSAPP_POLL_LIMIT: Messages per poll cycle (default: 20)

Usage:
    # Send a text message
    whatsapp(action="send_text", to="1234567890", text="Hello!")

    # Send a file
    whatsapp(action="send_file", to="1234567890", file_path="./pic.jpg", caption="Check this")

    # List chats
    whatsapp(action="chats_list")

    # Search messages
    whatsapp(action="messages_search", query="meeting", limit=20)

    # Start listener (polls for new messages, spawns DevDuck per message)
    whatsapp(action="start_listener")
"""

import json
import logging
import os
import subprocess
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from strands import tool

logger = logging.getLogger(__name__)

# Event storage
EVENTS_DIR = Path.cwd() / "whatsapp_events"
EVENTS_FILE = EVENTS_DIR / "events.jsonl"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

# Global listener state
_LISTENER_STATE: Dict[str, Any] = {
    "running": False,
    "thread": None,
    "sync_process": None,
    "messages_processed": 0,
    "start_time": None,
    "last_seen_id": None,
    "last_seen_ts": None,
}

# Serialize ALL wacli calls — the store uses a file lock, so only one
# wacli process can run at a time. Without this, concurrent sync/list/send
# calls hit "store is locked" errors.
_WACLI_LOCK = threading.Lock()


def _get_wacli_bin() -> str:
    """Get wacli binary path."""
    custom = os.environ.get("WACLI_BINARY")
    if custom and os.path.isfile(custom):
        return custom
    found = shutil.which("wacli")
    if found:
        return found
    return "wacli"


def _get_store_args() -> List[str]:
    """Get --store flag if WACLI_STORE is set."""
    store = os.environ.get("WACLI_STORE")
    if store:
        return ["--store", store]
    return []


def _run_wacli(
    args: List[str], timeout: int = 60, json_output: bool = True
) -> Dict[str, Any]:
    """Run a wacli command and return parsed output.

    All calls are serialized through _WACLI_LOCK because wacli uses a file-level
    store lock — concurrent processes hit "store is locked" errors.

    Args:
        args: Command arguments (e.g., ["chats", "list"])
        timeout: Command timeout in seconds
        json_output: Whether to request --json output

    Returns:
        Dict with status, output (parsed JSON or raw text), and exit_code
    """
    binary = _get_wacli_bin()
    cmd = [binary] + _get_store_args()
    if json_output:
        cmd.append("--json")
    cmd.extend(args)

    logger.debug(f"Running: {' '.join(cmd)}")

    with _WACLI_LOCK:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode != 0:
                error_msg = stderr or output or f"Exit code {result.returncode}"
                return {
                    "status": "error",
                    "output": error_msg,
                    "exit_code": result.returncode,
                }

            # Try to parse JSON
            if json_output and output:
                try:
                    parsed = json.loads(output)
                    return {"status": "success", "output": parsed, "exit_code": 0}
                except json.JSONDecodeError:
                    pass

            return {"status": "success", "output": output, "exit_code": 0}

        except FileNotFoundError:
            return {
                "status": "error",
                "output": "wacli not found. Install: brew install steipete/tap/wacli",
                "exit_code": -1,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "output": f"Command timed out after {timeout}s",
                "exit_code": -1,
            }
        except Exception as e:
            return {"status": "error", "output": str(e), "exit_code": -1}


def _store_event(event_data: Dict):
    """Append event to JSONL storage."""
    try:
        EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "event_type": "whatsapp_message",
            "payload": event_data,
            "timestamp": time.time(),
        }
        with open(EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Error storing whatsapp event: {e}")


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
        logger.error(f"Error reading whatsapp events: {e}")
        return []


def _parse_allowed_senders() -> Optional[set]:
    """Parse WHATSAPP_ALLOWED_SENDERS env var."""
    raw = os.environ.get("WHATSAPP_ALLOWED_SENDERS", "").strip()
    if not raw:
        return None
    return {s.strip().lstrip("+") for s in raw.split(",") if s.strip()}


def _is_sender_allowed(sender_jid: str) -> bool:
    """Check if sender is in the allowlist."""
    allowlist = _parse_allowed_senders()
    if allowlist is None:
        return True

    if not sender_jid:
        return False

    # Extract phone number from JID (e.g., "1234567890@s.whatsapp.net" → "1234567890")
    phone = sender_jid.split("@")[0] if "@" in sender_jid else sender_jid
    phone = phone.lstrip("+")

    return phone in allowlist or sender_jid in allowlist


def _emit_event(event_type: str, summary: str, detail: str = "", metadata: dict = None):
    """Push event to the unified event bus (if available)."""
    try:
        from devduck.tools.event_bus import emit
        emit(event_type, "whatsapp", summary, detail, metadata)
    except ImportError:
        pass


def _process_whatsapp_message(message: Dict):
    """Process a single WhatsApp message with a fresh DevDuck instance."""
    chat_jid = message.get("ChatJID", message.get("chat_jid", ""))
    sender_jid = message.get("SenderJID", message.get("sender_jid", ""))
    text = message.get("Text", message.get("Body", message.get("body", "")))
    if not text:
        text = message.get("DisplayText", "")
    msg_id = message.get("ID", message.get("id", ""))
    timestamp = message.get("Timestamp", message.get("timestamp", ""))

    if not text or not text.strip():
        return

    _LISTENER_STATE["messages_processed"] += 1
    logger.info(f"WhatsApp: processing message from {sender_jid} in {chat_jid}")

    # 🔔 Emit incoming message event
    _emit_event(
        "whatsapp.in",
        f"{sender_jid.split('@')[0]}: {text[:60]}",
        text,
        {"chat_jid": chat_jid, "sender_jid": sender_jid, "msg_id": msg_id},
    )

    try:
        from devduck import DevDuck

        connection_devduck = DevDuck(auto_start_servers=False)
        agent = connection_devduck.agent

        if not agent:
            logger.error("Failed to create DevDuck instance for whatsapp message")
            return

        # Build context
        recent = _get_recent_events(10)
        event_ctx = (
            f"\nRecent WhatsApp Events:\n{json.dumps(recent[-5:], indent=2)}"
            if recent
            else ""
        )

        agent.system_prompt += f"""

## WhatsApp Integration Active (via wacli):
- You are responding to a WhatsApp message
- Chat JID: {chat_jid}
- Sender JID: {sender_jid}
- Message ID: {msg_id}
- To reply: whatsapp(action="send_text", to="{chat_jid}", text="your reply")
- To send file: whatsapp(action="send_file", to="{chat_jid}", file_path="path", caption="text")
- To search history: whatsapp(action="messages_search", query="term", chat="{chat_jid}")
{event_ctx}
"""

        prompt = f"[WhatsApp {chat_jid}] {sender_jid} says: {text}"
        result = agent(prompt)

        # Always auto-reply with 🦆 prefix (so bot ignores its own messages)
        if result and str(result).strip():
            reply_text = "🦆 " + str(result).strip()
            _run_wacli(
                ["send", "text", "--to", chat_jid, "--message", reply_text],
                json_output=False,
            )

    except Exception as e:
        logger.error(f"Error processing whatsapp message: {e}", exc_info=True)


def _extract_messages(raw_output) -> List[Dict]:
    """Safely extract message list from wacli JSON output.

    wacli returns varying structures:
    - A list of messages directly: [{"ID": ...}, ...]
    - A dict with "data" key: {"success": true, "data": {"messages": [...]}}
    - A dict with "messages" key: {"messages": [...]}
    - A dict that IS a single message: {"ID": ..., "Text": ...}

    Returns:
        List of message dicts (never None)
    """
    if raw_output is None:
        return []
    if isinstance(raw_output, list):
        return [m for m in raw_output if isinstance(m, dict)]
    if isinstance(raw_output, dict):
        # Check for nested data structures
        data = raw_output.get("data", raw_output)
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict)]
        if isinstance(data, dict):
            msgs = data.get("messages")
            if isinstance(msgs, list):
                return [m for m in msgs if isinstance(m, dict)]
            # Could be a single message dict with ID/Text keys
            if data.get("ID") or data.get("id"):
                return [data]
        return []
    return []


def _listener_loop():
    """Background poll loop: fetch new messages and process them.

    Sync is done in a separate background thread on a slower cadence to avoid
    blocking the fast poll loop. wacli uses a file lock so sync + list can't
    run concurrently — but by syncing less frequently we keep the poll responsive.
    """
    poll_interval = int(os.environ.get("WHATSAPP_POLL_INTERVAL", "5"))
    poll_limit = int(os.environ.get("WHATSAPP_POLL_LIMIT", "20"))
    sync_interval = int(
        os.environ.get("WHATSAPP_SYNC_INTERVAL", "60")
    )  # sync every 60s

    # Track what we've seen — use set to avoid re-processing
    seen_ids: set = set()
    last_ts = _LISTENER_STATE.get("last_seen_ts") or time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )

    last_sync_time = 0  # Force first sync

    logger.info(
        f"WhatsApp listener started, polling every {poll_interval}s from {last_ts}"
    )

    while _LISTENER_STATE["running"]:
        try:
            now = time.time()

            # Sync on a slower cadence (non-blocking relative to poll)
            # First sync uses longer timeout for WA Web reconnection
            if now - last_sync_time > sync_interval:
                is_first = last_sync_time == 0
                sync_timeout = 120 if is_first else 45
                logger.debug(
                    f"WhatsApp: running periodic sync (first={is_first}, timeout={sync_timeout}s)"
                )
                sync_result = _run_wacli(
                    ["sync", "--once"],
                    timeout=sync_timeout,
                    json_output=False,
                )
                if sync_result["status"] == "success":
                    logger.info("WhatsApp sync completed successfully")
                else:
                    logger.warning(
                        f"WhatsApp sync failed: {sync_result.get('output', 'unknown')}"
                    )
                last_sync_time = time.time()

            # Fetch recent messages after our last timestamp
            result = _run_wacli(
                ["messages", "list", "--after", last_ts, "--limit", str(poll_limit)],
                timeout=30,
            )

            if result["status"] == "success":
                raw_output = result.get("output")
                # Safely extract messages using helper (never returns None)
                messages = _extract_messages(raw_output)

                logger.debug(
                    f"WhatsApp poll: got {len(messages)} message(s) after {last_ts}"
                )

                # Also handle the MsgID key variant — wacli uses "MsgID" not "ID"
                new_count = 0
                for msg in messages:
                    msg_id = msg.get("MsgID", msg.get("ID", msg.get("id", "")))
                    if not msg_id:
                        logger.debug(
                            f"WhatsApp: skipping message with no ID: {list(msg.keys())}"
                        )
                        continue

                    msg_ts = msg.get("Timestamp", msg.get("timestamp", ""))
                    is_from_me = msg.get(
                        "FromMe", msg.get("IsFromMe", msg.get("is_from_me", False))
                    )

                    msg_text = (
                        msg.get("Text", msg.get("Body", msg.get("DisplayText", "")))
                        or ""
                    )

                    # ONLY process messages from ME (bot is my personal assistant)
                    if not is_from_me:
                        # Still track non-me messages to advance watermark
                        seen_ids.add(msg_id)
                        continue

                    # Skip bot's own replies (they start with 🦆)
                    if msg_text.startswith("🦆"):
                        seen_ids.add(msg_id)
                        continue

                    # Skip empty messages
                    if not msg_text.strip():
                        seen_ids.add(msg_id)
                        continue

                    # Skip already-seen messages
                    if msg_id in seen_ids:
                        continue

                    # Mark as seen BEFORE spawning thread (prevent double-processing)
                    seen_ids.add(msg_id)
                    new_count += 1

                    logger.info(
                        f"WhatsApp: NEW message [{msg_id}] in {msg.get('ChatJID', '?')}: {msg_text[:80]}"
                    )

                    # Store event
                    _store_event(msg)

                    # Process in thread
                    t = threading.Thread(
                        target=_process_whatsapp_message,
                        args=(msg,),
                        daemon=True,
                    )
                    t.start()

                # Update watermark to latest message timestamp (regardless of from_me)
                # This ensures we don't re-fetch old messages
                if messages:
                    latest_ts = max(
                        (m.get("Timestamp", m.get("timestamp", "")) for m in messages),
                        default="",
                    )
                    if latest_ts and latest_ts > last_ts:
                        last_ts = latest_ts
                        _LISTENER_STATE["last_seen_ts"] = latest_ts

                if new_count > 0:
                    logger.info(
                        f"WhatsApp: processed {new_count} new message(s) this cycle"
                    )

            else:
                logger.debug(
                    f"WhatsApp poll returned: {result.get('output', 'no output')[:200]}"
                )

            # Prune seen_ids if it gets too large
            if len(seen_ids) > 2000:
                logger.info(f"Pruning seen_ids from {len(seen_ids)} entries")
                seen_ids.clear()

        except Exception as e:
            logger.error(f"WhatsApp listener poll error: {e}", exc_info=True)

        time.sleep(poll_interval)

    logger.info("WhatsApp listener stopped")


@tool
def whatsapp(
    action: str,
    # Send params
    to: Optional[str] = None,
    text: Optional[str] = None,
    file_path: Optional[str] = None,
    caption: Optional[str] = None,
    filename: Optional[str] = None,
    mime: Optional[str] = None,
    # Query params
    query: Optional[str] = None,
    chat: Optional[str] = None,
    sender: Optional[str] = None,
    message_id: Optional[str] = None,
    limit: int = 50,
    after: Optional[str] = None,
    before: Optional[str] = None,
    media_type: Optional[str] = None,
    # Context params
    context_before: int = 5,
    context_after: int = 5,
    # Media params
    output: Optional[str] = None,
    # Group params
    jid: Optional[str] = None,
    name: Optional[str] = None,
    # General
    count: int = 20,
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """WhatsApp integration via wacli CLI (steipete/wacli).

    No Cloud API needed — uses WhatsApp Web protocol directly.
    Requires: brew install steipete/tap/wacli && wacli auth

    Args:
        action: Action to perform:
            Auth & Sync:
            - "auth": Authenticate with QR code (interactive)
            - "sync": Start syncing messages (one-shot)
            - "doctor": Run diagnostics

            Listener:
            - "start_listener": Start polling for new messages
            - "stop_listener": Stop the listener
            - "listener_status": Get listener status
            - "get_recent_events": Get stored events

            Messages:
            - "send_text": Send text message (requires to + text)
            - "send_file": Send file (requires to + file_path)
            - "messages_list": List messages (optional: chat, after, before, limit)
            - "messages_search": Search messages (requires query)
            - "messages_context": Show context around a message (requires chat + message_id)
            - "messages_show": Show one message (requires chat + message_id)

            Chats:
            - "chats_list": List chats (optional: query, limit)
            - "chats_show": Show one chat (requires chat)

            Contacts:
            - "contacts_search": Search contacts (requires query)
            - "contacts_show": Show one contact (requires jid)
            - "contacts_refresh": Refresh contacts from session store

            Groups:
            - "groups_list": List groups (optional: query, limit)
            - "groups_info": Get group info (requires jid)
            - "groups_rename": Rename group (requires jid + name)

            Media:
            - "media_download": Download media (requires chat + message_id)

            History:
            - "history_backfill": Backfill older messages (requires chat)

        to: Recipient phone number or JID
        text: Message text
        file_path: Path to file to send
        caption: Caption for file messages
        filename: Override display name for file
        mime: Override MIME type for file
        query: Search query
        chat: Chat JID for filtering
        sender: Sender JID for filtering
        message_id: Message ID
        limit: Result limit (default: 50)
        after: Only messages after time (RFC3339 or YYYY-MM-DD)
        before: Only messages before time (RFC3339 or YYYY-MM-DD)
        media_type: Media type filter (image|video|audio|document)
        context_before: Messages before for context (default: 5)
        context_after: Messages after for context (default: 5)
        output: Output path for media download
        jid: JID for contact/group operations
        name: New name for group rename
        count: Number of recent events to retrieve
        agent: Parent agent (auto-provided)

    Returns:
        Dict with status and content

    Environment Variables:
        WACLI_STORE: Override store directory (default: ~/.wacli)
        WACLI_BINARY: Override wacli binary path
        STRANDS_WHATSAPP_AUTO_REPLY: "true" to auto-reply
        WHATSAPP_ALLOWED_SENDERS: Comma-separated phone/JID allowlist
        WHATSAPP_POLL_INTERVAL: Seconds between poll cycles (default: 5)
        WHATSAPP_POLL_LIMIT: Messages per poll cycle (default: 20)
    """

    def _result(status: str, text: str) -> Dict[str, Any]:
        return {"status": status, "content": [{"text": text}]}

    def _success(output) -> Dict[str, Any]:
        if isinstance(output, (dict, list)):
            return _result("success", json.dumps(output, indent=2, ensure_ascii=False))
        return _result("success", str(output))

    # Check wacli is available (except for event retrieval)
    if action not in ("get_recent_events", "listener_status"):
        binary = _get_wacli_bin()
        if not shutil.which(binary) and not os.path.isfile(binary):
            return _result(
                "error",
                "❌ wacli not found. Install: brew install steipete/tap/wacli\n"
                "Then authenticate: wacli auth",
            )

    # ─── Event retrieval (no wacli needed) ───
    if action == "get_recent_events":
        events = _get_recent_events(count)
        if events:
            return _success(events)
        return _result("success", "No events found")

    if action == "listener_status":
        allowlist = _parse_allowed_senders()
        status = {
            "running": _LISTENER_STATE["running"],
            "messages_processed": _LISTENER_STATE["messages_processed"],
            "events_file": str(EVENTS_FILE),
            "auto_reply": os.getenv("STRANDS_WHATSAPP_AUTO_REPLY", "false"),
            "allowed_senders": (
                sorted(allowlist) if allowlist else "all (no restriction)"
            ),
            "poll_interval": os.getenv("WHATSAPP_POLL_INTERVAL", "5"),
            "uptime": (
                time.time() - _LISTENER_STATE["start_time"]
                if _LISTENER_STATE["start_time"]
                else 0
            ),
        }
        return _success(status)

    # ─── Listener management ───
    if action == "start_listener":
        if _LISTENER_STATE["running"]:
            return _result("success", "WhatsApp listener already running")

        _LISTENER_STATE["running"] = True
        _LISTENER_STATE["messages_processed"] = 0
        _LISTENER_STATE["start_time"] = time.time()
        _LISTENER_STATE["last_seen_ts"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )

        t = threading.Thread(target=_listener_loop, daemon=True)
        t.start()
        _LISTENER_STATE["thread"] = t

        auto_reply = os.getenv("STRANDS_WHATSAPP_AUTO_REPLY", "false")
        return {
            "status": "success",
            "content": [
                {"text": "✅ WhatsApp listener started (polling via wacli)"},
                {"text": "🦆 Each message spawns a fresh DevDuck instance"},
                {"text": f"Auto-reply: {auto_reply}"},
                {"text": f"Events stored: {EVENTS_FILE}"},
                {"text": f"Poll interval: {os.getenv('WHATSAPP_POLL_INTERVAL', '5')}s"},
            ],
        }

    if action == "stop_listener":
        if not _LISTENER_STATE["running"]:
            return _result("success", "WhatsApp listener not running")

        _LISTENER_STATE["running"] = False
        processed = _LISTENER_STATE["messages_processed"]
        uptime = time.time() - (_LISTENER_STATE["start_time"] or time.time())

        return {
            "status": "success",
            "content": [
                {"text": "✅ WhatsApp listener stopped"},
                {
                    "text": f"Stats: {processed} messages processed, uptime {uptime:.0f}s"
                },
            ],
        }

    # ─── Auth & Sync ───
    if action == "auth":
        # Auth is interactive (QR code) — run in foreground
        return _result(
            "success",
            "🔐 Run `wacli auth` in your terminal to scan the QR code.\n"
            "This is interactive and cannot be run from within the agent.\n"
            "After auth, use whatsapp(action='sync') or whatsapp(action='doctor').",
        )

    if action == "sync":
        # One-shot sync (--once)
        result = _run_wacli(
            ["sync", "--once", "--refresh-contacts", "--refresh-groups"],
            timeout=120,
            json_output=False,
        )
        if result["status"] == "success":
            return _result("success", f"✅ Sync complete:\n{result['output']}")
        return _result("error", f"❌ Sync failed: {result['output']}")

    if action == "doctor":
        result = _run_wacli(["doctor"], timeout=30)
        if result["status"] == "success":
            return _success(result["output"])
        return _result("error", f"❌ Doctor failed: {result['output']}")

    # ─── Send messages ───
    if action == "send_text":
        if not to or not text:
            return _result("error", "❌ 'to' and 'text' required for send_text")
        result = _run_wacli(
            ["send", "text", "--to", to, "--message", text],
            json_output=False,
        )
        if result["status"] == "success":
            # 🔔 Emit outgoing message event
            _emit_event("whatsapp.out", f"→ {to}: {text[:60]}", text, {"to": to})
            return _result("success", f"✅ Message sent to {to}")
        return _result("error", f"❌ Send failed: {result['output']}")

    if action == "send_file":
        if not to or not file_path:
            return _result("error", "❌ 'to' and 'file_path' required for send_file")
        args = ["send", "file", "--to", to, "--file", file_path]
        if caption:
            args.extend(["--caption", caption])
        if filename:
            args.extend(["--filename", filename])
        if mime:
            args.extend(["--mime", mime])
        result = _run_wacli(args, json_output=False)
        if result["status"] == "success":
            return _result("success", f"✅ File sent to {to}")
        return _result("error", f"❌ Send file failed: {result['output']}")

    # ─── Messages ───
    if action == "messages_list":
        args = ["messages", "list", "--limit", str(limit)]
        if chat:
            args.extend(["--chat", chat])
        if after:
            args.extend(["--after", after])
        if before:
            args.extend(["--before", before])
        result = _run_wacli(args)
        if result["status"] == "success":
            return _success(result["output"])
        return _result("error", f"❌ {result['output']}")

    if action == "messages_search":
        if not query:
            return _result("error", "❌ 'query' required for messages_search")
        args = ["messages", "search", query, "--limit", str(limit)]
        if chat:
            args.extend(["--chat", chat])
        if sender:
            args.extend(["--from", sender])
        if after:
            args.extend(["--after", after])
        if before:
            args.extend(["--before", before])
        if media_type:
            args.extend(["--type", media_type])
        result = _run_wacli(args)
        if result["status"] == "success":
            return _success(result["output"])
        return _result("error", f"❌ {result['output']}")

    if action == "messages_context":
        if not chat or not message_id:
            return _result(
                "error", "❌ 'chat' and 'message_id' required for messages_context"
            )
        args = [
            "messages",
            "context",
            "--chat",
            chat,
            "--id",
            message_id,
            "--before",
            str(context_before),
            "--after",
            str(context_after),
        ]
        result = _run_wacli(args)
        if result["status"] == "success":
            return _success(result["output"])
        return _result("error", f"❌ {result['output']}")

    if action == "messages_show":
        if not chat or not message_id:
            return _result(
                "error", "❌ 'chat' and 'message_id' required for messages_show"
            )
        result = _run_wacli(["messages", "show", "--chat", chat, "--id", message_id])
        if result["status"] == "success":
            return _success(result["output"])
        return _result("error", f"❌ {result['output']}")

    # ─── Chats ───
    if action == "chats_list":
        args = ["chats", "list", "--limit", str(limit)]
        if query:
            args.extend(["--query", query])
        result = _run_wacli(args)
        if result["status"] == "success":
            return _success(result["output"])
        return _result("error", f"❌ {result['output']}")

    if action == "chats_show":
        if not chat:
            return _result("error", "❌ 'chat' required for chats_show")
        result = _run_wacli(["chats", "show", chat])
        if result["status"] == "success":
            return _success(result["output"])
        return _result("error", f"❌ {result['output']}")

    # ─── Contacts ───
    if action == "contacts_search":
        if not query:
            return _result("error", "❌ 'query' required for contacts_search")
        args = ["contacts", "search", query, "--limit", str(limit)]
        result = _run_wacli(args)
        if result["status"] == "success":
            return _success(result["output"])
        return _result("error", f"❌ {result['output']}")

    if action == "contacts_show":
        if not jid:
            return _result("error", "❌ 'jid' required for contacts_show")
        result = _run_wacli(["contacts", "show", jid])
        if result["status"] == "success":
            return _success(result["output"])
        return _result("error", f"❌ {result['output']}")

    if action == "contacts_refresh":
        result = _run_wacli(["contacts", "refresh"])
        if result["status"] == "success":
            return _result(
                "success",
                f"✅ Contacts refreshed:\n{json.dumps(result['output'], indent=2)}",
            )
        return _result("error", f"❌ {result['output']}")

    # ─── Groups ───
    if action == "groups_list":
        args = ["groups", "list", "--limit", str(limit)]
        if query:
            args.extend(["--query", query])
        result = _run_wacli(args)
        if result["status"] == "success":
            return _success(result["output"])
        return _result("error", f"❌ {result['output']}")

    if action == "groups_info":
        if not jid:
            return _result("error", "❌ 'jid' required for groups_info")
        result = _run_wacli(["groups", "info", "--jid", jid])
        if result["status"] == "success":
            return _success(result["output"])
        return _result("error", f"❌ {result['output']}")

    if action == "groups_rename":
        if not jid or not name:
            return _result("error", "❌ 'jid' and 'name' required for groups_rename")
        result = _run_wacli(
            ["groups", "rename", "--jid", jid, "--name", name],
            json_output=False,
        )
        if result["status"] == "success":
            return _result("success", f"✅ Group renamed to '{name}'")
        return _result("error", f"❌ {result['output']}")

    # ─── Media ───
    if action == "media_download":
        if not chat or not message_id:
            return _result(
                "error", "❌ 'chat' and 'message_id' required for media_download"
            )
        args = ["media", "download", "--chat", chat, "--id", message_id]
        if output:
            args.extend(["--output", output])
        result = _run_wacli(args, json_output=False)
        if result["status"] == "success":
            return _result("success", f"✅ Media downloaded:\n{result['output']}")
        return _result("error", f"❌ {result['output']}")

    # ─── History ───
    if action == "history_backfill":
        if not chat:
            return _result("error", "❌ 'chat' required for history_backfill")
        args = [
            "history",
            "backfill",
            "--chat",
            chat,
            "--requests",
            "10",
            "--count",
            "50",
        ]
        result = _run_wacli(args, timeout=120, json_output=False)
        if result["status"] == "success":
            return _result(
                "success", f"✅ Backfill requested for {chat}:\n{result['output']}"
            )
        return _result("error", f"❌ {result['output']}")

    # ─── Unknown action ───
    return _result(
        "error",
        f"❌ Unknown action: {action}. Valid actions:\n"
        "  Auth: auth, sync, doctor\n"
        "  Listener: start_listener, stop_listener, listener_status, get_recent_events\n"
        "  Send: send_text, send_file\n"
        "  Messages: messages_list, messages_search, messages_context, messages_show\n"
        "  Chats: chats_list, chats_show\n"
        "  Contacts: contacts_search, contacts_show, contacts_refresh\n"
        "  Groups: groups_list, groups_info, groups_rename\n"
        "  Media: media_download\n"
        "  History: history_backfill",
    )
