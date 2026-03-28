"""
🦆 Chrome Native Messaging Host for DevDuck Bridge Extension.

This acts as the bridge between the Chrome extension and DevDuck.
Communication protocol: JSON messages over stdin/stdout with 4-byte length prefix.

Can also run as a standalone WebSocket relay for the extension.
"""

import json
import logging
import os
import platform
import struct
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from strands import tool

logger = logging.getLogger("devduck.tools.chrome_bridge")

# ═══════════════════════════════════════════════════════════════
# Native Messaging Protocol (Chrome ↔ Host)
# ═══════════════════════════════════════════════════════════════

def _read_native_message() -> Optional[Dict]:
    """Read a message from Chrome via stdin (native messaging protocol)."""
    raw_length = sys.stdin.buffer.read(4)
    if not raw_length:
        return None
    length = struct.unpack("=I", raw_length)[0]
    if length > 1024 * 1024:  # 1MB limit
        return None
    data = sys.stdin.buffer.read(length)
    return json.loads(data.decode("utf-8"))


def _send_native_message(msg: Dict):
    """Send a message to Chrome via stdout (native messaging protocol)."""
    data = json.dumps(msg).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("=I", len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


# ═══════════════════════════════════════════════════════════════
# Native Messaging Host Installer
# ═══════════════════════════════════════════════════════════════

def _get_native_host_manifest_path() -> Path:
    """Get the path where the native messaging host manifest should be installed."""
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Google" / "Chrome" / "NativeMessagingHosts" / "com.devduck.bridge.json"
    elif system == "Linux":
        return Path.home() / ".config" / "google-chrome" / "NativeMessagingHosts" / "com.devduck.bridge.json"
    else:
        # Windows
        return Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data" / "NativeMessagingHosts" / "com.devduck.bridge.json"


def _get_extension_id() -> str:
    """Get the extension ID. When loaded unpacked, this needs to be updated."""
    return os.getenv("DEVDUCK_CHROME_EXTENSION_ID", "*")


def _install_native_host(extension_path: str = None) -> Dict:
    """Install the native messaging host manifest.

    Args:
        extension_path: Path to the devduck-chrome-extension directory
            If not provided, will try to find it relative to this file.
    """
    # Find the host script path
    host_script = Path(__file__).resolve()

    # Create the host wrapper script
    wrapper_dir = Path.home() / ".devduck"
    wrapper_dir.mkdir(exist_ok=True)
    wrapper_path = wrapper_dir / "chrome_native_host.sh"

    # Find python path
    python_path = sys.executable

    wrapper_content = f"""#!/bin/bash
exec "{python_path}" "{host_script}" --native-host "$@"
"""
    wrapper_path.write_text(wrapper_content)
    wrapper_path.chmod(0o755)

    # Create manifest
    manifest = {
        "name": "com.devduck.bridge",
        "description": "DevDuck terminal agent bridge",
        "path": str(wrapper_path),
        "type": "stdio",
        "allowed_origins": [
            f"chrome-extension://{_get_extension_id()}/",
        ],
    }

    # Allow all extensions if ID is wildcard
    ext_id = _get_extension_id()
    if ext_id == "*":
        # For development, we'll update this after loading the extension
        manifest["allowed_origins"] = ["chrome-extension://*/"]

    manifest_path = _get_native_host_manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "manifest_path": str(manifest_path),
        "wrapper_path": str(wrapper_path),
        "host_script": str(host_script),
    }


def _uninstall_native_host() -> bool:
    """Remove the native messaging host manifest."""
    manifest_path = _get_native_host_manifest_path()
    if manifest_path.exists():
        manifest_path.unlink()
        return True
    return False


# ═══════════════════════════════════════════════════════════════
# Native Host Main Loop
# ═══════════════════════════════════════════════════════════════

def _run_native_host():
    """Run as native messaging host (called by Chrome)."""
    while True:
        msg = _read_native_message()
        if msg is None:
            break

        # Process the message
        response = _handle_native_message(msg)
        _send_native_message(response)


def _handle_native_message(msg: Dict) -> Dict:
    """Handle a message from the Chrome extension."""
    action = msg.get("action", msg.get("type", ""))

    if action == "ping":
        return {"status": "pong", "agent": "devduck"}

    if action == "query":
        # Forward to DevDuck agent
        try:
            from devduck import ask
            result = str(ask(msg.get("prompt", "")))
            return {"status": "success", "result": result[:5000]}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    if action == "screenshot_ready":
        # Extension captured a screenshot and is sending it to us
        # Store for browse tool to pick up
        _store_extension_frame(msg.get("data"), msg.get("tabId"), msg.get("url"))
        return {"status": "ok"}

    return {"status": "error", "message": f"Unknown action: {action}"}


# Frame storage for extension bridge
_extension_frames = {}
_extension_lock = threading.Lock()


def _store_extension_frame(data_url: str, tab_id: int = None, url: str = None):
    """Store a frame from the extension."""
    import base64
    # data:image/png;base64,... → bytes
    if data_url and ";base64," in data_url:
        b64 = data_url.split(";base64,")[1]
        frame_bytes = base64.b64decode(b64)
        with _extension_lock:
            _extension_frames["latest"] = {
                "data": frame_bytes,
                "tab_id": tab_id,
                "url": url,
                "timestamp": __import__("time").time(),
            }


def get_extension_frame() -> Optional[Dict]:
    """Get the latest frame from the Chrome extension."""
    with _extension_lock:
        return _extension_frames.get("latest")


# ═══════════════════════════════════════════════════════════════
# Tool Interface
# ═══════════════════════════════════════════════════════════════

@tool
def chrome_bridge(
    action: str = "status",
    extension_id: str = None,
) -> Dict[str, Any]:
    """
    🔌 Chrome Extension Bridge — install and manage the DevDuck Chrome extension connection.

    Actions:
        - "install": Install native messaging host for Chrome extension
        - "uninstall": Remove native messaging host
        - "status": Check bridge status
        - "extension_path": Show path to load the extension

    Args:
        action: Action to perform
        extension_id: Chrome extension ID (set after loading unpacked extension)

    Returns:
        Dict with installation status and paths

    Setup:
        1. chrome_bridge(action="install")
        2. Open chrome://extensions
        3. Enable Developer Mode
        4. Load unpacked → select devduck-chrome-extension/
        5. Copy extension ID
        6. chrome_bridge(action="install", extension_id="your_id_here")
    """
    try:
        if action == "install":
            if extension_id:
                os.environ["DEVDUCK_CHROME_EXTENSION_ID"] = extension_id

            result = _install_native_host()

            ext_dir = Path(__file__).parent.parent.parent / "devduck-chrome-extension"
            if not ext_dir.exists():
                ext_dir = Path.cwd() / "devduck-chrome-extension"

            return {
                "status": "success",
                "content": [{
                    "text": f"🔌 Native messaging host installed!\n\n"
                            f"  Manifest: {result['manifest_path']}\n"
                            f"  Wrapper: {result['wrapper_path']}\n"
                            f"  Host: {result['host_script']}\n\n"
                            f"Next steps:\n"
                            f"  1. Open chrome://extensions\n"
                            f"  2. Enable 'Developer mode' (top right)\n"
                            f"  3. Click 'Load unpacked'\n"
                            f"  4. Select: {ext_dir}\n"
                            f"  5. Note the extension ID\n"
                            f"  6. Run: chrome_bridge(action='install', extension_id='YOUR_ID')\n\n"
                            f"The extension auto-connects to DevDuck mesh on ws://localhost:10000"
                }],
            }

        elif action == "uninstall":
            removed = _uninstall_native_host()
            return {
                "status": "success",
                "content": [{
                    "text": f"🔌 Native messaging host {'removed' if removed else 'not found'}"
                }],
            }

        elif action == "status":
            manifest_path = _get_native_host_manifest_path()
            installed = manifest_path.exists()

            ext_dir = Path(__file__).parent.parent.parent / "devduck-chrome-extension"
            ext_exists = ext_dir.exists()

            frame = get_extension_frame()
            has_frame = frame is not None

            return {
                "status": "success",
                "content": [{
                    "text": f"🔌 Chrome Bridge Status:\n"
                            f"  Native host installed: {installed}\n"
                            f"  Extension directory: {ext_dir} ({'exists' if ext_exists else 'NOT FOUND'})\n"
                            f"  Extension frame available: {has_frame}\n"
                            f"  Manifest path: {manifest_path}\n"
                            f"  Extension ID env: {os.getenv('DEVDUCK_CHROME_EXTENSION_ID', 'not set')}"
                }],
            }

        elif action == "extension_path":
            ext_dir = Path(__file__).parent.parent.parent / "devduck-chrome-extension"
            return {
                "status": "success",
                "content": [{
                    "text": f"🔌 Load this directory as unpacked extension in Chrome:\n"
                            f"  {ext_dir}\n\n"
                            f"  chrome://extensions → Developer mode → Load unpacked"
                }],
            }

        else:
            return {
                "status": "error",
                "content": [{"text": f"Unknown action: {action}. Valid: install, uninstall, status, extension_path"}],
            }

    except Exception as e:
        logger.error(f"Chrome bridge error: {e}")
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}


# ═══════════════════════════════════════════════════════════════
# Entry point for native messaging host
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--native-host" in sys.argv:
        _run_native_host()
    elif "--install" in sys.argv:
        result = _install_native_host()
        print(json.dumps(result, indent=2))
    else:
        print("Usage:")
        print("  python chrome_bridge.py --install    Install native messaging host")
        print("  python chrome_bridge.py --native-host  Run as native messaging host (called by Chrome)")
