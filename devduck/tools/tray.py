"""
Tray app control tool - integrated with devduck
"""

from strands import tool
from typing import Dict, Any, List
import subprocess
import socket
import json
import tempfile
import os
import sys
import time
import signal
from pathlib import Path

# Global state
_tray_process = None


def _send_ipc_command(socket_path: str, command: Dict) -> Dict:
    """Send IPC command to tray app"""
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(10.0)
        client.connect(socket_path)

        # Send command
        message = json.dumps(command).encode("utf-8")
        client.sendall(message)

        # Receive response
        response_data = b""
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            response_data += chunk
            # Check if we have complete JSON
            try:
                json.loads(response_data.decode("utf-8"))
                break
            except:
                continue

        client.close()

        if not response_data:
            return {"status": "error", "message": "Empty response"}

        return json.loads(response_data.decode("utf-8"))
    except socket.timeout:
        return {"status": "error", "message": "Timeout"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool
def tray(
    action: str,
    items: List[Dict[str, Any]] = None,
    title: str = None,
    message: Dict[str, Any] = None,
    text: str = None,
) -> Dict[str, Any]:
    """Control system tray app with devduck integration.

    Returns:
        Dict with status and content
    """
    global _tray_process

    socket_path = os.path.join(tempfile.gettempdir(), "devduck_tray.sock")

    if action == "start":
        if _tray_process and _tray_process.poll() is None:
            return {"status": "success", "content": [{"text": "✓ Already running"}]}

        # Get tray script path
        tools_dir = Path(__file__).parent
        tray_script = tools_dir / "_tray_app.py"

        if not tray_script.exists():
            return {
                "status": "error",
                "content": [{"text": f"❌ Tray app not found: {tray_script}"}],
            }

        _tray_process = subprocess.Popen(
            [sys.executable, str(tray_script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        time.sleep(1.5)

        return {
            "status": "success",
            "content": [{"text": f"✓ Tray app started (PID: {_tray_process.pid})"}],
        }

    elif action == "stop":
        if _tray_process:
            try:
                os.kill(_tray_process.pid, signal.SIGTERM)
                _tray_process.wait(timeout=3)
            except:
                pass
            _tray_process = None

        return {"status": "success", "content": [{"text": "✓ Stopped"}]}

    elif action == "status":
        is_running = _tray_process and _tray_process.poll() is None
        return {"status": "success", "content": [{"text": f"Running: {is_running}"}]}

    elif action == "update_menu":
        if not items:
            return {
                "status": "error",
                "content": [{"text": "items parameter required"}],
            }

        result = _send_ipc_command(
            socket_path, {"action": "update_menu", "items": items}
        )

        if result.get("status") == "success":
            return {
                "status": "success",
                "content": [{"text": f"✓ Menu updated ({len(items)} items)"}],
            }
        else:
            return {
                "status": "error",
                "content": [
                    {"text": f"Failed: {result.get('message', 'Unknown error')}"}
                ],
            }

    elif action == "update_title":
        if not title:
            return {
                "status": "error",
                "content": [{"text": "title parameter required"}],
            }

        result = _send_ipc_command(
            socket_path, {"action": "update_title", "title": title}
        )

        if result.get("status") == "success":
            return {"status": "success", "content": [{"text": f"✓ Title: {title}"}]}
        else:
            return {
                "status": "error",
                "content": [{"text": f"Failed: {result.get('message')}"}],
            }

    elif action == "set_progress":
        """Set progress indicator: idle, thinking, processing, complete, error"""
        if not text:
            return {
                "status": "error",
                "content": [
                    {
                        "text": "text parameter required (idle/thinking/processing/complete/error)"
                    }
                ],
            }

        result = _send_ipc_command(
            socket_path, {"action": "set_progress", "progress": text}
        )

        if result.get("status") == "success":
            return {"status": "success", "content": [{"text": f"✓ Progress: {text}"}]}
        else:
            return {
                "status": "error",
                "content": [{"text": f"Failed: {result.get('message')}"}],
            }

    elif action == "notify":
        if not message:
            return {
                "status": "error",
                "content": [{"text": "message parameter required"}],
            }

        result = _send_ipc_command(
            socket_path, {"action": "notify", "message": message}
        )

        if result.get("status") == "success":
            return {"status": "success", "content": [{"text": "✓ Notification sent"}]}
        else:
            return {
                "status": "error",
                "content": [{"text": f"Failed: {result.get('message')}"}],
            }

    elif action == "show_input":
        result = _send_ipc_command(socket_path, {"action": "show_input"})

        if result.get("status") == "success":
            return {"status": "success", "content": [{"text": "✓ Input shown"}]}
        else:
            return {
                "status": "error",
                "content": [{"text": f"Failed: {result.get('message')}"}],
            }

    elif action == "stream_text":
        if not text:
            return {"status": "error", "content": [{"text": "text parameter required"}]}

        result = _send_ipc_command(socket_path, {"action": "stream_text", "text": text})

        if result.get("status") == "success":
            return {"status": "success", "content": [{"text": "✓ Text streamed"}]}
        else:
            return {
                "status": "error",
                "content": [{"text": f"Failed: {result.get('message')}"}],
            }

    elif action in ["toggle_tcp", "toggle_ws", "toggle_mcp"]:
        result = _send_ipc_command(socket_path, {"action": action})

        if result.get("status") == "success":
            return {"status": "success", "content": [{"text": f"✓ {action} executed"}]}
        else:
            return {
                "status": "error",
                "content": [{"text": f"Failed: {result.get('message')}"}],
            }

    else:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"Unknown action: {action}. Available: start, stop, status, update_menu, update_title, set_progress, notify, show_input, stream_text, toggle_tcp, toggle_ws, toggle_mcp"
                }
            ],
        }
