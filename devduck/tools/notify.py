"""
🔔 Native notification tool for DevDuck.

Sends notifications via:
  1. macOS Notification Center (osascript / terminal-notifier)
  2. TUI toast (if running in TUI mode)
  3. Terminal bell
  4. Speech (macOS `say`)
  5. Sound (macOS `afplay`)

Works everywhere — auto-detects the best method for the current environment.
"""

import os
import sys
import shutil
import subprocess
import platform
import logging
from typing import Any, Dict, Optional

from strands import tool

logger = logging.getLogger(__name__)


def _notify_macos(title: str, message: str, sound: str = "default",
                  subtitle: str = "", url: str = "", group: str = "devduck") -> bool:
    """Send macOS native notification. Returns True on success."""
    # Prefer terminal-notifier (clickable, more features)
    tn = shutil.which("terminal-notifier")
    if tn:
        cmd = [tn, "-title", title, "-message", message, "-group", group]
        if subtitle:
            cmd.extend(["-subtitle", subtitle])
        if sound and sound != "none":
            cmd.extend(["-sound", sound])
        if url:
            cmd.extend(["-open", url])
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            logger.debug(f"terminal-notifier failed: {e}")

    # Fallback to osascript
    try:
        script = f'display notification "{message}" with title "{title}"'
        if subtitle:
            script += f' subtitle "{subtitle}"'
        if sound and sound != "none":
            script += f' sound name "{sound}"'
        subprocess.Popen(
            ["osascript", "-e", script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except Exception as e:
        logger.debug(f"osascript notification failed: {e}")
        return False


def _notify_tui(title: str, message: str, severity: str = "information") -> bool:
    """Send notification via TUI toast if running."""
    try:
        from devduck.tools.tui import get_tui_app
        app = get_tui_app()
        if app:
            app.call_from_thread(app.notify, message, title=title, severity=severity)
            return True
    except ImportError:
        pass
    return False


def _notify_bell() -> bool:
    """Ring the terminal bell."""
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
        return True
    except Exception:
        return False


def _speak(text: str, voice: str = "", rate: int = 200) -> bool:
    """Speak text using macOS `say` command."""
    if platform.system() != "Darwin":
        return False
    say = shutil.which("say")
    if not say:
        return False
    try:
        cmd = [say, "-r", str(rate)]
        if voice:
            cmd.extend(["-v", voice])
        cmd.append(text)
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        logger.debug(f"say failed: {e}")
        return False


def _play_sound(sound: str = "Ping") -> bool:
    """Play a macOS system sound."""
    if platform.system() != "Darwin":
        return False
    # System sounds live in /System/Library/Sounds/
    sound_path = f"/System/Library/Sounds/{sound}.aiff"
    if not os.path.exists(sound_path):
        # Try common names
        for s in ["Ping", "Glass", "Basso", "Hero", "Pop", "Purr", "Submarine", "Tink"]:
            p = f"/System/Library/Sounds/{s}.aiff"
            if os.path.exists(p):
                sound_path = p
                break
        else:
            return False
    try:
        subprocess.Popen(
            ["afplay", sound_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        return False


@tool
def notify(
    message: str,
    title: str = "🦆 DevDuck",
    method: str = "auto",
    sound: str = "default",
    subtitle: str = "",
    url: str = "",
    voice: str = "",
    severity: str = "information",
) -> Dict[str, Any]:
    """
    🔔 Send a notification — native macOS, TUI toast, speech, or sound.

    Automatically picks the best method based on the environment.
    Can send to multiple channels simultaneously.

    Args:
        message: The notification message text
        title: Notification title (default: "🦆 DevDuck")
        method: Delivery method:
            - "auto": Try all available (native + TUI + bell)
            - "native": macOS Notification Center only
            - "tui": TUI toast only (if TUI is running)
            - "bell": Terminal bell only
            - "speak": Text-to-speech (macOS `say`)
            - "sound": Play system sound
            - "all": Send via ALL channels simultaneously
        sound: Sound name for native notification:
            - "default": Default notification sound
            - "none": Silent notification
            - "Ping", "Glass", "Basso", "Hero", "Pop", "Purr", etc.
        subtitle: Optional subtitle (macOS notification)
        url: URL to open when notification is clicked (requires terminal-notifier)
        voice: Voice for speech (e.g. "Samantha", "Alex", "Daniel")
        severity: TUI notification severity ("information", "warning", "error")

    Returns:
        Dict with status and delivery results

    Examples:
        # Auto notification (best available method)
        notify(message="Build complete!")

        # Native macOS notification with sound
        notify(message="Tests passed ✅", sound="Hero")

        # Speak the message
        notify(message="Hey, the deployment is done", method="speak")

        # Silent notification
        notify(message="Background task complete", sound="none")

        # Everything at once
        notify(message="CRITICAL: Server down!", method="all", severity="error")

        # Clickable notification with URL
        notify(message="PR merged!", url="https://github.com/...", subtitle="repo-name")
    """
    results = {}

    try:
        if method == "auto":
            # Try TUI first, then native, then bell
            tui_ok = _notify_tui(title, message, severity)
            results["tui"] = tui_ok

            if platform.system() == "Darwin":
                native_ok = _notify_macos(title, message, sound, subtitle, url)
                results["native"] = native_ok
            else:
                results["native"] = False

            if not tui_ok and not results.get("native"):
                bell_ok = _notify_bell()
                results["bell"] = bell_ok

        elif method == "native":
            if platform.system() == "Darwin":
                results["native"] = _notify_macos(title, message, sound, subtitle, url)
            else:
                results["native"] = False
                results["note"] = "Native notifications only available on macOS"

        elif method == "tui":
            results["tui"] = _notify_tui(title, message, severity)

        elif method == "bell":
            results["bell"] = _notify_bell()

        elif method == "speak":
            results["speak"] = _speak(message, voice)

        elif method == "sound":
            sound_name = sound if sound not in ("default", "none") else "Ping"
            results["sound"] = _play_sound(sound_name)

        elif method == "all":
            results["tui"] = _notify_tui(title, message, severity)
            if platform.system() == "Darwin":
                results["native"] = _notify_macos(title, message, sound, subtitle, url)
                results["speak"] = _speak(message, voice)
            results["bell"] = _notify_bell()

        else:
            return {
                "status": "error",
                "content": [{"text": f"Unknown method: {method}. Valid: auto, native, tui, bell, speak, sound, all"}],
            }

        # Summarize
        delivered = [k for k, v in results.items() if v]
        failed = [k for k, v in results.items() if not v]

        summary = f"🔔 Notification sent via: {', '.join(delivered) if delivered else 'none'}"
        if failed:
            summary += f" (failed: {', '.join(failed)})"

        # 🔔 Push to unified event bus
        if delivered:
            try:
                from devduck.tools.event_bus import emit
                emit("notify", "notify", f"{title}: {message[:60]}", message,
                     {"method": method, "delivered": delivered})
            except ImportError:
                pass

        return {
            "status": "success" if delivered else "error",
            "content": [{"text": summary}],
        }

    except Exception as e:
        logger.error(f"Notification error: {e}")
        return {
            "status": "error",
            "content": [{"text": f"Notification error: {e}"}],
        }
