"""
🐬 Flipper Zero tool for DevDuck — serial CLI interface.

Communicates with Flipper Zero over USB serial (CDC) using the built-in CLI.
No protobuf needed — uses the text-based CLI protocol directly.

Capabilities:
  - Auto-detect connected Flipper
  - Device info, power info, date/time
  - Storage: list, read, write, delete, mkdir, stat, md5
  - File transfer: send/receive files to/from Flipper
  - LED control, vibro, speaker
  - IR/SubGHz/NFC/RFID signal replay
  - Run CLI commands directly
  - App management

Requirements: pyserial (auto-installed if missing)

Environment:
  FLIPPER_PORT: Override auto-detection (e.g., /dev/cu.usbmodemflip_XXXXXXX1)
"""

import os
import sys
import time
import math
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional

from strands import tool

logger = logging.getLogger("devduck.tools.use_flipper")

# ─── Serial helpers ──────────────────────────────────────────────────────────

def _ensure_pyserial():
    """Ensure pyserial is available, install if missing."""
    try:
        import serial  # noqa: F401
        return True
    except ImportError:
        logger.info("Installing pyserial...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pyserial", "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True


def _find_flipper_port() -> Optional[str]:
    """Auto-detect connected Flipper Zero."""
    env_port = os.environ.get("FLIPPER_PORT")
    if env_port:
        return env_port

    _ensure_pyserial()
    import serial.tools.list_ports as list_ports

    flippers = list(list_ports.grep("flip_"))
    if len(flippers) == 1:
        logger.info(f"Found Flipper: {flippers[0].serial_number} on {flippers[0].device}")
        return flippers[0].device
    elif len(flippers) > 1:
        # Return first, but log warning
        logger.warning(f"Multiple Flippers found ({len(flippers)}), using first: {flippers[0].device}")
        return flippers[0].device
    return None


# ─── Buffered serial reader ─────────────────────────────────────────────────

class _BufferedRead:
    """Buffered serial reader that reads until a delimiter."""

    def __init__(self, stream):
        self.buffer = bytearray()
        self.stream = stream

    def until(self, eol: str = "\n", cut_eol: bool = True, timeout: float = 10.0) -> bytes:
        eol_bytes = eol.encode("ascii")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            i = self.buffer.find(eol_bytes)
            if i >= 0:
                if cut_eol:
                    read = self.buffer[:i]
                else:
                    read = self.buffer[:i + len(eol_bytes)]
                self.buffer = self.buffer[i + len(eol_bytes):]
                return read
            # Read available data
            n = max(1, self.stream.in_waiting)
            data = self.stream.read(n)
            if data:
                self.buffer.extend(data)
            else:
                time.sleep(0.01)
        raise TimeoutError(f"Timeout waiting for delimiter: {repr(eol)}")


# ─── Flipper connection ─────────────────────────────────────────────────────

CLI_PROMPT = ">: "
CLI_EOL = "\r\n"

# Connection pool (reuse across calls)
_connections: Dict[str, Any] = {}
_conn_lock = threading.Lock()


def _get_connection(port: str = None):
    """Get or create a serial connection to Flipper."""
    _ensure_pyserial()
    import serial

    if port is None:
        port = _find_flipper_port()
    if port is None:
        raise ConnectionError(
            "No Flipper Zero found. Connect via USB or set FLIPPER_PORT env var."
        )

    with _conn_lock:
        if port in _connections:
            conn = _connections[port]
            if conn["serial"].is_open:
                return conn
            else:
                del _connections[port]

        # Open new connection
        ser = serial.Serial()
        ser.port = port
        ser.timeout = 2
        ser.baudrate = 230400
        ser.open()
        time.sleep(0.5)

        reader = _BufferedRead(ser)

        # Flush and sync with CLI prompt
        ser.reset_input_buffer()
        ser.write(b"\r")
        try:
            reader.until(CLI_PROMPT, timeout=5)
        except TimeoutError:
            # Try again
            ser.write(b"\r")
            reader.until(CLI_PROMPT, timeout=5)

        conn = {"serial": ser, "reader": reader, "port": port}
        _connections[port] = conn
        logger.info(f"Connected to Flipper on {port}")
        return conn


def _send_cmd(conn, cmd: str, timeout: float = 15.0) -> str:
    """Send a CLI command and return the response text."""
    ser = conn["serial"]
    reader = conn["reader"]

    # Clear any pending data
    ser.reset_input_buffer()
    reader.buffer.clear()

    # Send command
    ser.write(f"{cmd}\r".encode("ascii"))

    # Read echo of command
    try:
        reader.until(CLI_EOL, timeout=timeout)
    except TimeoutError:
        pass

    # Read until prompt
    response = reader.until(CLI_PROMPT, timeout=timeout)
    return response.decode("utf-8", errors="replace").strip()


def _close_connection(port: str = None):
    """Close a Flipper connection."""
    with _conn_lock:
        if port and port in _connections:
            try:
                _connections[port]["serial"].close()
            except Exception:
                pass
            del _connections[port]
        elif port is None:
            for p, c in list(_connections.items()):
                try:
                    c["serial"].close()
                except Exception:
                    pass
            _connections.clear()


# ─── Storage helpers ─────────────────────────────────────────────────────────

def _storage_list(conn, path: str) -> list:
    """List files/dirs at path."""
    response = _send_cmd(conn, f'storage list "{path}"')
    entries = []
    for line in response.split("\r\n"):
        line = line.strip()
        if not line or line == "Empty" or "Storage error:" in line:
            continue
        if line.startswith("[D]"):
            name = line[4:].strip()
            entries.append({"type": "dir", "name": name})
        elif line.startswith("[F]"):
            info = line[4:].strip()
            # Last token is size
            parts = info.rsplit(" ", 1)
            if len(parts) == 2:
                entries.append({"type": "file", "name": parts[0], "size": parts[1]})
            else:
                entries.append({"type": "file", "name": info, "size": "?"})
    return entries


def _storage_read_file(conn, flipper_path: str) -> bytes:
    """Read a file from Flipper and return bytes."""
    ser = conn["serial"]
    reader = conn["reader"]
    chunk_size = 8192

    ser.reset_input_buffer()
    reader.buffer.clear()

    ser.write(f'storage read_chunks "{flipper_path}" {chunk_size}\r'.encode("ascii"))
    reader.until(CLI_EOL, timeout=10)
    answer = reader.until(CLI_EOL, timeout=10)

    if b"Storage error:" in answer:
        error = answer.decode("utf-8", errors="replace")
        reader.until(CLI_PROMPT, timeout=5)
        raise FileNotFoundError(f"Flipper: {error}")

    # Parse size
    size = int(answer.split(b": ")[1])
    filedata = bytearray()
    read_size = 0

    while read_size < size:
        reader.until("Ready?" + CLI_EOL, timeout=15)
        ser.write(b"y")
        to_read = min(size - read_size, chunk_size)
        filedata.extend(ser.read(to_read))
        read_size += to_read

    reader.until(CLI_PROMPT, timeout=5)
    return bytes(filedata)


def _storage_write_file(conn, flipper_path: str, data: bytes):
    """Write data to a file on Flipper."""
    ser = conn["serial"]
    reader = conn["reader"]
    chunk_size = 8192

    offset = 0
    while offset < len(data):
        chunk = data[offset:offset + chunk_size]
        size = len(chunk)

        ser.reset_input_buffer()
        reader.buffer.clear()

        ser.write(f'storage write_chunk "{flipper_path}" {size}\r'.encode("ascii"))
        reader.until(CLI_EOL, timeout=10)
        answer = reader.until(CLI_EOL, timeout=10)

        if b"Storage error:" in answer:
            error = answer.decode("utf-8", errors="replace")
            reader.until(CLI_PROMPT, timeout=5)
            raise IOError(f"Flipper write error: {error}")

        ser.write(chunk)
        reader.until(CLI_PROMPT, timeout=15)
        offset += size


# ─── The tool ────────────────────────────────────────────────────────────────

@tool
def use_flipper(
    action: str,
    path: str = None,
    local_path: str = None,
    data: str = None,
    command: str = None,
    port: str = None,
    frequency: float = None,
    duration: float = None,
) -> Dict[str, Any]:
    """
    🐬 Flipper Zero serial CLI interface.

    Auto-detects connected Flipper Zero via USB. Provides full access to
    the Flipper's CLI including storage, device info, LED/vibro/speaker,
    IR/SubGHz signal replay, and raw CLI commands.

    Args:
        action: Action to perform:
            Connection:
            - "detect": Find connected Flipper Zero devices
            - "connect": Connect to Flipper (auto or specify port)
            - "disconnect": Close connection

            Device Info:
            - "info": Get device info (firmware, hardware, etc.)
            - "power_info": Battery and power status
            - "datetime": Get current date/time
            - "uptime": Device uptime

            Storage:
            - "ls": List files/dirs (path required)
            - "tree": Recursive directory listing
            - "read": Read file content from Flipper (path required)
            - "write": Write data to Flipper file (path + data required)
            - "send": Send local file to Flipper (local_path + path required)
            - "receive": Download file from Flipper (path + local_path required)
            - "mkdir": Create directory (path required)
            - "rm": Remove file/directory (path required)
            - "stat": File/dir info (path required)
            - "md5": Get file MD5 hash (path required)
            - "df": Storage space info (path: /int or /ext)

            Hardware:
            - "led": Control LED (data: "r/g/b/bl" + "0-255", e.g. "r 255")
            - "vibro": Vibrate (data: "1" on, "0" off)
            - "speaker": Play tone (frequency in Hz, duration in seconds)
            - "alert": Play audiovisual alert

            Signals:
            - "ir_tx": Transmit IR signal file (path to .ir file on Flipper)
            - "subghz_tx": Transmit SubGHz signal (path to .sub file)
            - "nfc_detect": Detect NFC tags

            Apps:
            - "app_list": List installed apps
            - "app_start": Start an app (command = app path)

            Raw:
            - "cli": Send raw CLI command (command required)

        path: Flipper path (e.g., "/ext/subghz/garage.sub")
        local_path: Local file path for send/receive
        data: Data for write/led/vibro actions
        command: Raw CLI command or app path
        port: Serial port override (default: auto-detect)
        frequency: Frequency in Hz for speaker
        duration: Duration in seconds for speaker

    Returns:
        Dict with status and content

    Examples:
        use_flipper(action="detect")
        use_flipper(action="info")
        use_flipper(action="ls", path="/ext")
        use_flipper(action="read", path="/ext/nfc/my_card.nfc")
        use_flipper(action="send", local_path="./payload.sub", path="/ext/subghz/payload.sub")
        use_flipper(action="led", data="r 255")
        use_flipper(action="vibro", data="1")
        use_flipper(action="speaker", frequency=440, duration=0.5)
        use_flipper(action="cli", command="bt info")
        use_flipper(action="subghz_tx", path="/ext/subghz/gate.sub")
    """
    try:
        # ── Detection (no connection needed) ──
        if action == "detect":
            _ensure_pyserial()
            import serial.tools.list_ports as list_ports

            flippers = list(list_ports.grep("flip_"))
            all_ports = list(list_ports.comports())

            if flippers:
                flipper_info = []
                for f in flippers:
                    flipper_info.append(
                        f"  🐬 {f.device} (SN: {f.serial_number}, {f.description})"
                    )
                text = f"Found {len(flippers)} Flipper(s):\n" + "\n".join(flipper_info)
            else:
                usb_ports = [p for p in all_ports if "usb" in p.device.lower() or "usbmodem" in p.device.lower()]
                if usb_ports:
                    text = "No Flipper found. USB devices detected:\n" + "\n".join(
                        f"  {p.device} ({p.description})" for p in usb_ports
                    )
                else:
                    text = "No Flipper Zero detected. Is it connected via USB and not in DFU mode?"

            return {"status": "success", "content": [{"text": text}]}

        # ── Disconnect ──
        if action == "disconnect":
            _close_connection(port)
            return {"status": "success", "content": [{"text": "🐬 Disconnected"}]}

        # ── All other actions need a connection ──
        conn = _get_connection(port)

        # ── Device Info ──
        if action == "info":
            response = _send_cmd(conn, "device_info")
            lines = response.replace("\r", "").split("\n")
            info = {}
            for line in lines:
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k.strip()] = v.strip()
            # Format nicely
            text = "🐬 Flipper Zero Device Info:\n"
            for k, v in info.items():
                text += f"  {k}: {v}\n"
            return {"status": "success", "content": [{"text": text}]}

        elif action == "power_info":
            response = _send_cmd(conn, "power info")
            text = f"🐬 Power Info:\n{response}"
            return {"status": "success", "content": [{"text": text}]}

        elif action == "datetime":
            response = _send_cmd(conn, "date")
            text = f"🐬 Date/Time: {response}"
            return {"status": "success", "content": [{"text": text}]}

        elif action == "uptime":
            response = _send_cmd(conn, "uptime")
            text = f"🐬 Uptime: {response}"
            return {"status": "success", "content": [{"text": text}]}

        # ── Storage ──
        elif action == "ls":
            if not path:
                path = "/ext"
            entries = _storage_list(conn, path)
            if not entries:
                text = f"📁 {path}: (empty)"
            else:
                lines = [f"📁 {path}:"]
                for e in entries:
                    if e["type"] == "dir":
                        lines.append(f"  📂 {e['name']}/")
                    else:
                        lines.append(f"  📄 {e['name']} ({e.get('size', '?')})")
                text = "\n".join(lines)
            return {"status": "success", "content": [{"text": text}]}

        elif action == "tree":
            if not path:
                path = "/ext"
            response = _send_cmd(conn, f'storage list "{path}"', timeout=30)
            text = f"🌲 {path}:\n{response}"
            return {"status": "success", "content": [{"text": text}]}

        elif action == "read":
            if not path:
                return {"status": "error", "content": [{"text": "path required for read"}]}
            file_data = _storage_read_file(conn, path)
            # Try text first
            try:
                text_content = file_data.decode("utf-8")
                text = f"📄 {path} ({len(file_data)} bytes):\n{text_content}"
            except UnicodeDecodeError:
                import binascii
                text = f"📄 {path} ({len(file_data)} bytes, binary):\n{binascii.hexlify(file_data).decode()[:2000]}"
            return {"status": "success", "content": [{"text": text}]}

        elif action == "write":
            if not path or data is None:
                return {"status": "error", "content": [{"text": "path and data required for write"}]}
            _storage_write_file(conn, path, data.encode("utf-8"))
            text = f"✅ Written {len(data)} bytes to {path}"
            return {"status": "success", "content": [{"text": text}]}

        elif action == "send":
            if not path or not local_path:
                return {"status": "error", "content": [{"text": "path and local_path required for send"}]}
            local_file = Path(local_path).expanduser().resolve()
            if not local_file.exists():
                return {"status": "error", "content": [{"text": f"Local file not found: {local_file}"}]}
            file_data = local_file.read_bytes()
            _storage_write_file(conn, path, file_data)
            text = f"✅ Sent {local_file.name} ({len(file_data)} bytes) → {path}"
            return {"status": "success", "content": [{"text": text}]}

        elif action == "receive":
            if not path or not local_path:
                return {"status": "error", "content": [{"text": "path and local_path required for receive"}]}
            file_data = _storage_read_file(conn, path)
            local_file = Path(local_path).expanduser().resolve()
            local_file.parent.mkdir(parents=True, exist_ok=True)
            local_file.write_bytes(file_data)
            text = f"✅ Received {path} ({len(file_data)} bytes) → {local_file}"
            return {"status": "success", "content": [{"text": text}]}

        elif action == "mkdir":
            if not path:
                return {"status": "error", "content": [{"text": "path required"}]}
            response = _send_cmd(conn, f'storage mkdir "{path}"')
            if "Storage error:" in response:
                return {"status": "error", "content": [{"text": f"🐬 {response}"}]}
            return {"status": "success", "content": [{"text": f"✅ Created directory: {path}"}]}

        elif action == "rm":
            if not path:
                return {"status": "error", "content": [{"text": "path required"}]}
            response = _send_cmd(conn, f'storage remove "{path}"')
            if "Storage error:" in response:
                return {"status": "error", "content": [{"text": f"🐬 {response}"}]}
            return {"status": "success", "content": [{"text": f"✅ Removed: {path}"}]}

        elif action == "stat":
            if not path:
                return {"status": "error", "content": [{"text": "path required"}]}
            response = _send_cmd(conn, f'storage stat "{path}"')
            return {"status": "success", "content": [{"text": f"🐬 {path}: {response}"}]}

        elif action == "md5":
            if not path:
                return {"status": "error", "content": [{"text": "path required"}]}
            response = _send_cmd(conn, f'storage md5 "{path}"')
            return {"status": "success", "content": [{"text": f"🐬 MD5({path}): {response}"}]}

        elif action == "df":
            p = path or "/ext"
            response = _send_cmd(conn, f'storage info "{p}"')
            return {"status": "success", "content": [{"text": f"🐬 Storage info ({p}):\n{response}"}]}

        # ── Hardware ──
        elif action == "led":
            if not data:
                return {"status": "error", "content": [{"text": "data required: 'r 255', 'g 128', 'b 0', 'bl 255'"}]}
            response = _send_cmd(conn, f"led {data}")
            return {"status": "success", "content": [{"text": f"💡 LED: {data}"}]}

        elif action == "vibro":
            if data is None:
                data = "1"
            response = _send_cmd(conn, f"vibro {data}")
            state = "ON" if data.strip() != "0" else "OFF"
            return {"status": "success", "content": [{"text": f"📳 Vibro: {state}"}]}

        elif action == "speaker":
            freq = frequency or 440.0
            dur = duration or 0.5
            # Use the tone command
            response = _send_cmd(conn, f"tone {int(freq)} {int(dur * 1000)}")
            return {"status": "success", "content": [{"text": f"🔊 Tone: {freq}Hz for {dur}s"}]}

        elif action == "alert":
            response = _send_cmd(conn, "led r 255")
            time.sleep(0.1)
            _send_cmd(conn, "vibro 1")
            time.sleep(0.3)
            _send_cmd(conn, "vibro 0")
            _send_cmd(conn, "led r 0")
            return {"status": "success", "content": [{"text": "🚨 Alert sent! (LED + Vibro)"}]}

        # ── Signals ──
        elif action == "ir_tx":
            if not path:
                return {"status": "error", "content": [{"text": "path required (Flipper .ir file path)"}]}
            # Use the ir CLI command
            if command:
                # Specific signal name in the file
                response = _send_cmd(conn, f'ir tx "{path}" {command}', timeout=10)
            else:
                response = _send_cmd(conn, f'ir tx "{path}"', timeout=10)
            return {"status": "success", "content": [{"text": f"📡 IR TX: {path}\n{response}"}]}

        elif action == "subghz_tx":
            if not path:
                return {"status": "error", "content": [{"text": "path required (Flipper .sub file path)"}]}
            response = _send_cmd(conn, f'subghz tx "{path}"', timeout=15)
            return {"status": "success", "content": [{"text": f"📡 SubGHz TX: {path}\n{response}"}]}

        elif action == "nfc_detect":
            response = _send_cmd(conn, "nfc detect", timeout=10)
            return {"status": "success", "content": [{"text": f"📱 NFC: {response}"}]}

        # ── Apps ──
        elif action == "app_list":
            response = _send_cmd(conn, "loader list", timeout=10)
            return {"status": "success", "content": [{"text": f"🐬 Installed Apps:\n{response}"}]}

        elif action == "app_start":
            if not command:
                return {"status": "error", "content": [{"text": "command required (app path)"}]}
            response = _send_cmd(conn, f'loader open "{command}"', timeout=10)
            return {"status": "success", "content": [{"text": f"🚀 App started: {command}\n{response}"}]}

        # ── Bluetooth ──
        elif action == "bt_info":
            response = _send_cmd(conn, "bt info", timeout=5)
            return {"status": "success", "content": [{"text": f"🔵 Bluetooth:\n{response}"}]}

        # ── Raw CLI ──
        elif action == "cli":
            if not command:
                return {"status": "error", "content": [{"text": "command required for cli action"}]}
            response = _send_cmd(conn, command, timeout=30)
            return {"status": "success", "content": [{"text": f"🐬> {command}\n{response}"}]}

        # ── Connect (explicit) ──
        elif action == "connect":
            text = f"🐬 Connected to Flipper on {conn['port']}"
            return {"status": "success", "content": [{"text": text}]}

        else:
            return {
                "status": "error",
                "content": [{"text": f"Unknown action: {action}. Valid: detect, info, ls, read, write, send, receive, mkdir, rm, stat, md5, df, led, vibro, speaker, alert, ir_tx, subghz_tx, nfc_detect, app_list, app_start, bt_info, cli, connect, disconnect"}],
            }

    except ConnectionError as e:
        return {"status": "error", "content": [{"text": f"🐬 Connection error: {e}"}]}
    except TimeoutError as e:
        return {"status": "error", "content": [{"text": f"🐬 Timeout: {e}"}]}
    except FileNotFoundError as e:
        return {"status": "error", "content": [{"text": f"🐬 Not found: {e}"}]}
    except Exception as e:
        logger.error(f"Flipper tool error: {e}", exc_info=True)
        return {"status": "error", "content": [{"text": f"🐬 Error: {e}"}]}
