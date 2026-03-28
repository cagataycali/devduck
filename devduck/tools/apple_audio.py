"""🔊 Apple Audio device management via CoreAudio."""

from typing import Dict, Any
from strands import tool


@tool
def apple_audio(
    action: str = "devices",
    device_id: int = None,
    volume: float = None,
    mute: bool = None,
) -> Dict[str, Any]:
    """🔊 Audio device management — list devices, get/set volume, mute.

    Args:
        action: Action to perform:
            - "devices": List all audio devices
            - "volume": Get current volume (or set if volume param provided)
            - "mute": Get mute state (or toggle if mute param provided)
            - "default": Show default input/output devices
        device_id: Target device ID (optional, uses default if omitted)
        volume: Volume level 0.0-1.0 (for setting volume)
        mute: True/False to set mute state

    Returns:
        Dict with audio data
    """
    try:
        import CoreAudio
    except ImportError:
        return {"status": "error", "content": [{"text": "Install: pip install pyobjc-framework-CoreAudio"}]}

    if action == "devices":
        return _list_devices()
    elif action == "volume":
        if volume is not None:
            return _set_volume(volume, device_id)
        return _get_volume(device_id)
    elif action == "mute":
        if mute is not None:
            return _set_mute(mute, device_id)
        return _get_mute(device_id)
    elif action == "default":
        return _get_defaults()
    else:
        return {"status": "error", "content": [{"text": f"Unknown action: {action}. Use: devices, volume, mute, default"}]}


def _list_devices():
    """List all audio devices using system_profiler."""
    try:
        import subprocess, json

        r = subprocess.run(
            ["system_profiler", "SPAudioDataType", "-json"],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(r.stdout)
        audio = data.get("SPAudioDataType", [{}])[0]

        lines = ["🔊 Audio Devices:\n"]

        for section_key in ["_items", "spaudiovideo_output_device", "spaudiovideo_input_device"]:
            items = audio.get(section_key, [])
            if not items:
                continue

            for dev in items:
                if isinstance(dev, dict):
                    name = dev.get("_name", "Unknown")
                    manufacturer = dev.get("coreaudio_device_manufacturer", "")
                    transport = dev.get("coreaudio_device_transport", "")
                    sample_rate = dev.get("coreaudio_device_srate", "")
                    is_output = dev.get("coreaudio_output_source", "")
                    is_input = dev.get("coreaudio_input_source", "")

                    direction = "🔈" if is_output else "🎤" if is_input else "🔊"
                    lines.append(f"  {direction} {name}")
                    if manufacturer:
                        lines.append(f"     Manufacturer: {manufacturer}")
                    if sample_rate:
                        lines.append(f"     Sample Rate: {sample_rate}")
                    if transport:
                        lines.append(f"     Transport: {transport}")

        # Also get current volume via osascript
        r2 = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True, text=True, timeout=5
        )
        if r2.returncode == 0:
            vol = r2.stdout.strip()
            lines.append(f"\n  🔊 Current Output Volume: {vol}%")

        r3 = subprocess.run(
            ["osascript", "-e", "input volume of (get volume settings)"],
            capture_output=True, text=True, timeout=5
        )
        if r3.returncode == 0:
            vol = r3.stdout.strip()
            lines.append(f"  🎤 Current Input Volume: {vol}%")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Devices error: {e}"}]}


def _get_volume(device_id=None):
    """Get current output volume."""
    try:
        import subprocess
        r = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True, text=True, timeout=5
        )
        vol = r.stdout.strip()
        return {"status": "success", "content": [{"text": f"🔊 Volume: {vol}%"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Volume error: {e}"}]}


def _set_volume(volume, device_id=None):
    """Set output volume (0.0-1.0)."""
    try:
        import subprocess
        vol_pct = int(volume * 100) if volume <= 1.0 else int(volume)
        vol_pct = max(0, min(100, vol_pct))

        r = subprocess.run(
            ["osascript", "-e", f"set volume output volume {vol_pct}"],
            capture_output=True, text=True, timeout=5
        )
        return {"status": "success", "content": [{"text": f"🔊 Volume set to {vol_pct}%"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Set volume error: {e}"}]}


def _get_mute(device_id=None):
    """Get mute state."""
    try:
        import subprocess
        r = subprocess.run(
            ["osascript", "-e", "output muted of (get volume settings)"],
            capture_output=True, text=True, timeout=5
        )
        muted = r.stdout.strip()
        emoji = "🔇" if muted == "true" else "🔊"
        return {"status": "success", "content": [{"text": f"{emoji} Muted: {muted}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Mute error: {e}"}]}


def _set_mute(mute, device_id=None):
    """Set mute state."""
    try:
        import subprocess
        state = "true" if mute else "false"
        r = subprocess.run(
            ["osascript", "-e", f"set volume output muted {state}"],
            capture_output=True, text=True, timeout=5
        )
        emoji = "🔇" if mute else "🔊"
        return {"status": "success", "content": [{"text": f"{emoji} Mute {'enabled' if mute else 'disabled'}"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Set mute error: {e}"}]}


def _get_defaults():
    """Get default input/output devices."""
    try:
        import subprocess
        lines = ["🔊 Default Audio Devices:\n"]

        # Output
        r = subprocess.run(
            ["osascript", "-e", 'get name of current application'],
            capture_output=True, text=True, timeout=5
        )

        # Get volume settings
        r2 = subprocess.run(
            ["osascript", "-e", "get volume settings"],
            capture_output=True, text=True, timeout=5
        )
        if r2.returncode == 0:
            lines.append(f"  Settings: {r2.stdout.strip()}")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Defaults error: {e}"}]}
