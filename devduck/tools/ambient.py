"""
Ambient input overlay control tool - integrated with devduck
"""

from strands import tool
from typing import Dict, Any
import subprocess
import socket
import json
import tempfile
import os
import time
import signal
import sys
from pathlib import Path

# Global state
_ambient_process = None


def _send_command(command: Dict) -> Dict:
    """Send command to ambient input overlay"""
    socket_path = os.path.join(tempfile.gettempdir(), "devduck_ambient.sock")

    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(2.0)
        client.connect(socket_path)
        client.sendall(json.dumps(command).encode("utf-8"))

        response_data = client.recv(4096)
        client.close()

        if not response_data:
            return {"status": "error", "message": "Empty response"}

        return json.loads(response_data.decode("utf-8"))
    except socket.timeout:
        return {"status": "error", "message": "Timeout"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool
def ambient(
    action: str,
    text: str = None,
) -> Dict[str, Any]:
    """Control ambient AI input overlay.

    Args:
        action: Action to perform
            - "start": Start ambient overlay
            - "stop": Stop overlay
            - "show": Show overlay
            - "hide": Hide overlay
            - "status": Check if running
            - "set_text": Pre-fill text
        text: Text to pre-fill (for set_text action)

    Returns:
        Dict with status and content

    Features:
        🎨 Modern glassmorphism UI
        ⚡ Blinking cursor with auto-focus
        🌊 Real-time IPC streaming from devduck
        📦 Structured message handling
        ⌨️ ESC to hide, Enter to send
    """
    global _ambient_process

    if action == "start":
        if _ambient_process and _ambient_process.poll() is None:
            return {"status": "success", "content": [{"text": "✓ Already running"}]}

        # Get ambient script path (in same directory as this file)
        tools_dir = Path(__file__).parent
        ambient_script = tools_dir / "_ambient_input.py"

        if not ambient_script.exists():
            return {
                "status": "error",
                "content": [{"text": f"❌ Ambient script not found: {ambient_script}"}],
            }

        _ambient_process = subprocess.Popen(
            [sys.executable, str(ambient_script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        time.sleep(1.5)

        return {
            "status": "success",
            "content": [
                {"text": f"✓ Ambient overlay started (PID: {_ambient_process.pid})"}
            ],
        }

    elif action == "stop":
        if _ambient_process:
            try:
                os.kill(_ambient_process.pid, signal.SIGTERM)
                _ambient_process.wait(timeout=3)
            except:
                pass
            _ambient_process = None

        return {"status": "success", "content": [{"text": "✓ Stopped"}]}

    elif action == "status":
        is_running = _ambient_process and _ambient_process.poll() is None
        return {"status": "success", "content": [{"text": f"Running: {is_running}"}]}

    elif action == "show":
        result = _send_command({"action": "show"})
        if result.get("status") == "success":
            return {"status": "success", "content": [{"text": "✓ Overlay shown"}]}
        else:
            return {
                "status": "error",
                "content": [
                    {"text": f"Failed: {result.get('message', 'Unknown error')}"}
                ],
            }

    elif action == "hide":
        result = _send_command({"action": "hide"})
        if result.get("status") == "success":
            return {"status": "success", "content": [{"text": "✓ Overlay hidden"}]}
        else:
            return {
                "status": "error",
                "content": [
                    {"text": f"Failed: {result.get('message', 'Unknown error')}"}
                ],
            }

    elif action == "set_text":
        if not text:
            return {"status": "error", "content": [{"text": "text parameter required"}]}

        result = _send_command({"action": "set_text", "text": text})
        if result.get("status") == "success":
            return {"status": "success", "content": [{"text": f"✓ Text set: {text}"}]}
        else:
            return {
                "status": "error",
                "content": [
                    {"text": f"Failed: {result.get('message', 'Unknown error')}"}
                ],
            }

    else:
        return {"status": "error", "content": [{"text": f"Unknown action: {action}"}]}
