"""
🌐 Browse tool — inline browser in terminal via Chrome DevTools Protocol.

Features:
  - Headless or visible Chrome with YOUR profile (cookies, extensions, logins)
  - Screenshot → halfblock terminal rendering (works in any 24-bit color terminal)
  - Full interaction: click, type, scroll, navigate, DOM queries
  - CDP screencast for live streaming (5-15 fps)
  - Tab management (list, switch, close)
  - Connect to already-running Chrome via extension bridge
  - Export rendered frames as Rich Text for Textual TUI embedding

Usage:
    browse(action="open", url="https://github.com")
    browse(action="screenshot")
    browse(action="click", x=400, y=300)
    browse(action="type", text="hello world")
    browse(action="scroll", direction="down")
    browse(action="tabs")
    browse(action="attach")  # connect to running Chrome

Architecture:
    Chrome (headless/visible) ←CDP WebSocket→ BrowserSession ←→ browse() tool
                                                                    ↓
                                                          halfblock renderer
                                                                    ↓
                                                            Rich Text / TUI
"""

import asyncio
import base64
import io
import json
import logging
import os
import platform
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from strands import tool

logger = logging.getLogger("devduck.tools.browse")

# ═══════════════════════════════════════════════════════════════════
# Chrome Discovery
# ═══════════════════════════════════════════════════════════════════

def _find_chrome() -> Optional[str]:
    """Find Chrome binary on the system."""
    candidates = []
    system = platform.system()

    if system == "Darwin":
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            shutil.which("google-chrome"),
            shutil.which("chromium"),
        ]
    elif system == "Linux":
        candidates = [
            shutil.which("google-chrome"),
            shutil.which("google-chrome-stable"),
            shutil.which("chromium"),
            shutil.which("chromium-browser"),
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
        ]
    else:  # Windows
        candidates = [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
            shutil.which("chrome"),
        ]

    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _find_chrome_profile() -> Optional[str]:
    """Find default Chrome user data directory."""
    system = platform.system()
    home = Path.home()

    if system == "Darwin":
        path = home / "Library" / "Application Support" / "Google" / "Chrome"
    elif system == "Linux":
        path = home / ".config" / "google-chrome"
    else:
        path = Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"

    return str(path) if path.exists() else None


def _find_available_port(start: int = 9222, attempts: int = 20) -> int:
    """Find an available port."""
    for offset in range(attempts):
        port = start + offset
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", port))
            s.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports in range {start}-{start + attempts}")


# ═══════════════════════════════════════════════════════════════════
# Halfblock Terminal Renderer (High-Fidelity)
# ═══════════════════════════════════════════════════════════════════

class HalfblockRenderer:
    """Convert PIL Image to terminal halfblock characters with 24-bit color.

    Uses fast pixel array access and run-length style merging for optimal
    rendering quality and performance.
    """

    @staticmethod
    def _prepare_image(image_data: bytes, width: int, height: int):
        """Load, resize with high-quality downsampling, return pixel accessor.

        Uses LANCZOS for the best downscaling quality — preserves text edges
        and thin lines much better than bilinear.
        """
        from PIL import Image

        img = Image.open(io.BytesIO(image_data))

        # Convert to RGB if needed (handles RGBA, P, L modes)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Ensure even height for halfblock pairing
        if height % 2 != 0:
            height += 1

        # High-quality downscale
        img = img.resize((width, height), Image.LANCZOS)

        # Use load() for fast pixel access (returns PixelAccess, ~10x faster than getpixel)
        pixels = img.load()
        return pixels, width, height

    @staticmethod
    def render(
        image_data: bytes,
        width: int = 120,
        height: int = 60,
        format: str = "ansi",
    ) -> str:
        """Render image bytes to halfblock string.

        Args:
            image_data: PNG/JPEG image bytes
            width: Target width in terminal columns
            height: Target height in pixel rows (terminal rows = height/2)
            format: "ansi" for raw ANSI, "rich" for Rich Text markup

        Returns:
            Rendered string
        """
        try:
            from PIL import Image
        except ImportError:
            return "[PIL not installed - pip install Pillow]"

        pixels, width, height = HalfblockRenderer._prepare_image(image_data, width, height)

        if format == "rich":
            return HalfblockRenderer._render_rich(pixels, width, height)
        else:
            return HalfblockRenderer._render_ansi(pixels, width, height)

    @staticmethod
    def _render_ansi(pixels, width: int, height: int) -> str:
        """Render to ANSI escape sequences with run-length optimization."""
        lines = []
        for y in range(0, height, 2):
            parts = []
            prev_top = prev_bot = None
            run = 0

            for x in range(width):
                top = pixels[x, y]
                bottom = pixels[x, y + 1] if y + 1 < height else (0, 0, 0)

                if top == prev_top and bottom == prev_bot:
                    run += 1
                else:
                    if run > 0:
                        parts.append(
                            f"\033[38;2;{prev_top[0]};{prev_top[1]};{prev_top[2]}m"
                            f"\033[48;2;{prev_bot[0]};{prev_bot[1]};{prev_bot[2]}m"
                            f"{'▀' * run}"
                        )
                    prev_top, prev_bot = top, bottom
                    run = 1

            # Flush last run
            if run > 0:
                parts.append(
                    f"\033[38;2;{prev_top[0]};{prev_top[1]};{prev_top[2]}m"
                    f"\033[48;2;{prev_bot[0]};{prev_bot[1]};{prev_bot[2]}m"
                    f"{'▀' * run}"
                )
            lines.append("".join(parts) + "\033[0m")
        return "\n".join(lines)

    @staticmethod
    def _render_rich(pixels, width: int, height: int) -> str:
        """Render to Rich markup with run-length style merging."""
        lines = []
        for y in range(0, height, 2):
            parts = []
            prev_fg = prev_bg = None
            run = 0

            for x in range(width):
                top = pixels[x, y]
                bottom = pixels[x, y + 1] if y + 1 < height else (0, 0, 0)
                fg = f"#{top[0]:02x}{top[1]:02x}{top[2]:02x}"
                bg = f"#{bottom[0]:02x}{bottom[1]:02x}{bottom[2]:02x}"

                if fg == prev_fg and bg == prev_bg:
                    run += 1
                else:
                    if run > 0:
                        parts.append(f"[{prev_fg} on {prev_bg}]{'▀' * run}[/]")
                    prev_fg, prev_bg = fg, bg
                    run = 1

            if run > 0:
                parts.append(f"[{prev_fg} on {prev_bg}]{'▀' * run}[/]")
            lines.append("".join(parts))
        return "\n".join(lines)

    @staticmethod
    def render_to_rich_text(image_data: bytes, width: int = 120, height: int = 60):
        """Render to a Rich Text object for direct use in Textual.

        Uses run-length style merging: adjacent pixels with the same color
        share a single Style object → fewer allocations, faster rendering.

        Returns a rich.text.Text object with proper styling.
        """
        try:
            from PIL import Image
            from rich.text import Text
            from rich.style import Style
            from rich.color import Color
        except ImportError:
            from rich.text import Text
            return Text("[PIL not installed]")

        pixels, width, height = HalfblockRenderer._prepare_image(image_data, width, height)

        # Style cache — reuse Style objects for identical color pairs
        _style_cache = {}

        text = Text()
        for y in range(0, height, 2):
            # Run-length encode each scanline pair
            prev_key = None
            run = 0

            for x in range(width):
                top = pixels[x, y]
                bottom = pixels[x, y + 1] if y + 1 < height else (0, 0, 0)
                key = (top[0], top[1], top[2], bottom[0], bottom[1], bottom[2])

                if key == prev_key:
                    run += 1
                else:
                    # Flush previous run
                    if run > 0:
                        style = _style_cache.get(prev_key)
                        if style is None:
                            style = Style(
                                color=Color.from_rgb(prev_key[0], prev_key[1], prev_key[2]),
                                bgcolor=Color.from_rgb(prev_key[3], prev_key[4], prev_key[5]),
                            )
                            _style_cache[prev_key] = style
                        text.append("▀" * run, style=style)
                    prev_key = key
                    run = 1

            # Flush last run on this line
            if run > 0:
                style = _style_cache.get(prev_key)
                if style is None:
                    style = Style(
                        color=Color.from_rgb(prev_key[0], prev_key[1], prev_key[2]),
                        bgcolor=Color.from_rgb(prev_key[3], prev_key[4], prev_key[5]),
                    )
                    _style_cache[prev_key] = style
                text.append("▀" * run, style=style)

            if y + 2 < height:
                text.append("\n")

        return text


# ═══════════════════════════════════════════════════════════════════
# CDP (Chrome DevTools Protocol) Client
# ═══════════════════════════════════════════════════════════════════

class CDPClient:
    """Async Chrome DevTools Protocol client over WebSocket."""

    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self._ws = None
        self._msg_id = 0
        self._pending = {}
        self._events = {}
        self._recv_task = None
        self._lock = asyncio.Lock()

    async def connect(self):
        """Connect to CDP WebSocket."""
        import websockets
        self._ws = await websockets.connect(
            self.ws_url,
            max_size=50 * 1024 * 1024,  # 50MB for screenshots
            close_timeout=5,
        )
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info(f"CDP connected: {self.ws_url}")

    async def disconnect(self):
        """Disconnect from CDP."""
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def send(self, method: str, params: dict = None, timeout: float = 30) -> dict:
        """Send CDP command and wait for response."""
        async with self._lock:
            self._msg_id += 1
            msg_id = self._msg_id

        msg = {"id": msg_id, "method": method}
        if params:
            msg["params"] = params

        future = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = future

        await self._ws.send(json.dumps(msg))

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(msg_id, None)
            raise TimeoutError(f"CDP command timed out: {method}")

    def on_event(self, method: str, callback):
        """Register event handler."""
        self._events.setdefault(method, []).append(callback)

    async def _recv_loop(self):
        """Receive messages from CDP."""
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if "id" in msg:
                    future = self._pending.pop(msg["id"], None)
                    if future and not future.done():
                        if "error" in msg:
                            future.set_exception(
                                RuntimeError(f"CDP error: {msg['error'].get('message', msg['error'])}")
                            )
                        else:
                            future.set_result(msg.get("result", {}))
                elif "method" in msg:
                    for cb in self._events.get(msg["method"], []):
                        try:
                            cb(msg.get("params", {}))
                        except Exception as e:
                            logger.debug(f"Event handler error: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"CDP recv loop ended: {e}")


# ═══════════════════════════════════════════════════════════════════
# Browser Session Manager
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BrowserSession:
    """Manages a Chrome browser session."""
    cdp: Optional[CDPClient] = None
    process: Optional[subprocess.Popen] = None
    port: int = 9222
    ws_url: str = ""
    page_url: str = ""
    viewport_width: int = 1280
    viewport_height: int = 800
    use_profile: bool = False
    headless: bool = True
    # Screencast state
    _screencast_active: bool = False
    _last_frame: Optional[bytes] = None
    _frame_lock: threading.Lock = field(default_factory=threading.Lock)
    _screencast_callbacks: list = field(default_factory=list)


# Global session
_session: Optional[BrowserSession] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None


def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    """Get or create a dedicated event loop for CDP."""
    global _event_loop, _loop_thread

    if _event_loop and _event_loop.is_running():
        return _event_loop

    _event_loop = asyncio.new_event_loop()

    def _run_loop():
        asyncio.set_event_loop(_event_loop)
        _event_loop.run_forever()

    _loop_thread = threading.Thread(target=_run_loop, daemon=True)
    _loop_thread.start()
    time.sleep(0.1)  # Let it start
    return _event_loop


def _run_async(coro):
    """Run async coroutine from sync context."""
    loop = _get_or_create_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=60)


async def _launch_chrome(
    headless: bool = True,
    use_profile: bool = False,
    port: int = None,
    url: str = "about:blank",
    viewport_width: int = 1280,
    viewport_height: int = 800,
) -> BrowserSession:
    """Launch Chrome with CDP enabled."""
    chrome_path = _find_chrome()
    if not chrome_path:
        raise RuntimeError("Chrome not found. Install Google Chrome.")

    if port is None:
        port = _find_available_port()

    args = [
        chrome_path,
        f"--remote-debugging-port={port}",
        f"--window-size={viewport_width},{viewport_height}",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--no-first-run",
        "--no-default-browser-check",
    ]

    if headless:
        args.append("--headless=new")
        args.append("--disable-gpu")
        args.append("--hide-scrollbars")

    if use_profile:
        profile_dir = _find_chrome_profile()
        if profile_dir:
            args.append(f"--user-data-dir={profile_dir}")
            args.append("--profile-directory=Default")
            logger.info(f"Using Chrome profile: {profile_dir}")
        else:
            logger.warning("Chrome profile not found, using temp profile")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="devduck_chrome_")
        args.append(f"--user-data-dir={tmp_dir}")

    args.append(url)

    process = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid if platform.system() != "Windows" else None,
    )

    # Wait for CDP to be ready
    ws_url = None
    for attempt in range(50):
        await asyncio.sleep(0.2)
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2)
            data = json.loads(resp.read())
            ws_url = data.get("webSocketDebuggerUrl")
            if ws_url:
                break
        except Exception:
            continue

    if not ws_url:
        process.kill()
        raise RuntimeError(f"Chrome failed to start CDP on port {port}")

    # Get page target
    resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2)
    targets = json.loads(resp.read())
    page_target = None
    for t in targets:
        if t.get("type") == "page":
            page_target = t
            break

    if not page_target:
        process.kill()
        raise RuntimeError("No page target found")

    page_ws_url = page_target["webSocketDebuggerUrl"]

    # Connect CDP
    cdp = CDPClient(page_ws_url)
    await cdp.connect()

    # Enable required domains
    await cdp.send("Page.enable")
    await cdp.send("DOM.enable")
    await cdp.send("Runtime.enable")
    await cdp.send("Network.enable")

    # Set viewport
    await cdp.send("Emulation.setDeviceMetricsOverride", {
        "width": viewport_width,
        "height": viewport_height,
        "deviceScaleFactor": 2,
        "mobile": False,
    })

    session = BrowserSession(
        cdp=cdp,
        process=process,
        port=port,
        ws_url=page_ws_url,
        page_url=url,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        use_profile=use_profile,
        headless=headless,
    )

    logger.info(f"Chrome launched: port={port}, headless={headless}, profile={use_profile}")
    return session


async def _attach_to_chrome(port: int = 9222) -> BrowserSession:
    """Attach to an already-running Chrome with CDP enabled."""
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=3)
        version_data = json.loads(resp.read())
    except Exception as e:
        raise RuntimeError(
            f"Cannot connect to Chrome on port {port}. "
            f"Restart Chrome with: --remote-debugging-port={port}\n"
            f"Error: {e}"
        )

    # Get first page target
    resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=3)
    targets = json.loads(resp.read())
    page_target = None
    for t in targets:
        if t.get("type") == "page":
            page_target = t
            break

    if not page_target:
        raise RuntimeError("No page target found in running Chrome")

    page_ws_url = page_target["webSocketDebuggerUrl"]

    cdp = CDPClient(page_ws_url)
    await cdp.connect()

    await cdp.send("Page.enable")
    await cdp.send("DOM.enable")
    await cdp.send("Runtime.enable")

    session = BrowserSession(
        cdp=cdp,
        port=port,
        ws_url=page_ws_url,
        page_url=page_target.get("url", ""),
    )

    logger.info(f"Attached to Chrome on port {port}: {session.page_url}")
    return session


async def _screenshot(session: BrowserSession, quality: int = 80) -> bytes:
    """Capture screenshot as PNG bytes."""
    result = await session.cdp.send("Page.captureScreenshot", {
        "format": "png",
        "captureBeyondViewport": False,
        "optimizeForSpeed": False,
    })
    return base64.b64decode(result["data"])


async def _navigate(session: BrowserSession, url: str, wait_for_load: bool = True):
    """Navigate to URL."""
    await session.cdp.send("Page.navigate", {"url": url})
    session.page_url = url
    if wait_for_load:
        try:
            # Wait for load event
            await session.cdp.send("Page.loadEventFired", timeout=15)
        except Exception:
            pass
        # Extra settle time for JS rendering
        await asyncio.sleep(0.5)


async def _click(session: BrowserSession, x: int, y: int, button: str = "left", click_count: int = 1):
    """Click at coordinates."""
    btn = {"left": "left", "right": "right", "middle": "middle"}.get(button, "left")
    await session.cdp.send("Input.dispatchMouseEvent", {
        "type": "mousePressed",
        "x": x, "y": y,
        "button": btn,
        "clickCount": click_count,
    })
    await asyncio.sleep(0.05)
    await session.cdp.send("Input.dispatchMouseEvent", {
        "type": "mouseReleased",
        "x": x, "y": y,
        "button": btn,
        "clickCount": click_count,
    })


async def _type_text(session: BrowserSession, text: str):
    """Type text character by character."""
    for char in text:
        await session.cdp.send("Input.dispatchKeyEvent", {
            "type": "keyDown",
            "text": char,
            "key": char,
            "unmodifiedText": char,
        })
        await session.cdp.send("Input.dispatchKeyEvent", {
            "type": "keyUp",
            "key": char,
        })
        await asyncio.sleep(0.02)


async def _press_key(session: BrowserSession, key: str):
    """Press a special key (Enter, Tab, Escape, etc.)."""
    key_map = {
        "enter": {"key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13},
        "tab": {"key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9},
        "escape": {"key": "Escape", "code": "Escape", "windowsVirtualKeyCode": 27},
        "backspace": {"key": "Backspace", "code": "Backspace", "windowsVirtualKeyCode": 8},
        "delete": {"key": "Delete", "code": "Delete", "windowsVirtualKeyCode": 46},
        "arrowup": {"key": "ArrowUp", "code": "ArrowUp", "windowsVirtualKeyCode": 38},
        "arrowdown": {"key": "ArrowDown", "code": "ArrowDown", "windowsVirtualKeyCode": 40},
        "arrowleft": {"key": "ArrowLeft", "code": "ArrowLeft", "windowsVirtualKeyCode": 37},
        "arrowright": {"key": "ArrowRight", "code": "ArrowRight", "windowsVirtualKeyCode": 39},
        "space": {"key": " ", "code": "Space", "windowsVirtualKeyCode": 32},
    }
    kd = key_map.get(key.lower(), {"key": key, "code": key})
    await session.cdp.send("Input.dispatchKeyEvent", {"type": "rawKeyDown", **kd})
    await session.cdp.send("Input.dispatchKeyEvent", {"type": "keyUp", **kd})


async def _scroll(session: BrowserSession, direction: str = "down", amount: int = 300, x: int = 640, y: int = 400):
    """Scroll the page."""
    delta_y = -amount if direction == "up" else amount if direction == "down" else 0
    delta_x = -amount if direction == "left" else amount if direction == "right" else 0
    await session.cdp.send("Input.dispatchMouseEvent", {
        "type": "mouseWheel",
        "x": x, "y": y,
        "deltaX": delta_x,
        "deltaY": delta_y,
    })
    await asyncio.sleep(0.2)


async def _get_tabs(port: int = 9222) -> List[Dict]:
    """List all open tabs."""
    resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=3)
    targets = json.loads(resp.read())
    return [
        {
            "id": t.get("id"),
            "title": t.get("title", ""),
            "url": t.get("url", ""),
            "type": t.get("type", ""),
        }
        for t in targets
        if t.get("type") == "page"
    ]


async def _switch_tab(session: BrowserSession, tab_id: str):
    """Switch to a different tab by reconnecting CDP."""
    port = session.port
    resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=3)
    targets = json.loads(resp.read())

    target = None
    for t in targets:
        if t.get("id") == tab_id:
            target = t
            break

    if not target:
        raise RuntimeError(f"Tab {tab_id} not found")

    # Activate the tab
    urllib.request.urlopen(f"http://127.0.0.1:{port}/json/activate/{tab_id}", timeout=3)

    # Disconnect old CDP, connect to new target
    await session.cdp.disconnect()

    ws_url = target["webSocketDebuggerUrl"]
    session.cdp = CDPClient(ws_url)
    await session.cdp.connect()
    await session.cdp.send("Page.enable")
    session.ws_url = ws_url
    session.page_url = target.get("url", "")


async def _evaluate_js(session: BrowserSession, expression: str) -> Any:
    """Evaluate JavaScript in the page."""
    result = await session.cdp.send("Runtime.evaluate", {
        "expression": expression,
        "returnByValue": True,
        "awaitPromise": True,
    })
    if "exceptionDetails" in result:
        raise RuntimeError(f"JS error: {result['exceptionDetails']}")
    return result.get("result", {}).get("value")


async def _get_dom(session: BrowserSession, selector: str = None) -> str:
    """Get page DOM or specific element."""
    if selector:
        js = f"""
        (() => {{
            const el = document.querySelector('{selector}');
            return el ? el.outerHTML : null;
        }})()
        """
    else:
        js = "document.documentElement.outerHTML.substring(0, 50000)"
    return await _evaluate_js(session, js)


async def _get_page_info(session: BrowserSession) -> Dict:
    """Get current page info."""
    title = await _evaluate_js(session, "document.title")
    url = await _evaluate_js(session, "window.location.href")
    scroll_y = await _evaluate_js(session, "window.scrollY")
    scroll_height = await _evaluate_js(session, "document.body.scrollHeight")
    viewport_height = await _evaluate_js(session, "window.innerHeight")
    return {
        "title": title,
        "url": url,
        "scroll_y": scroll_y,
        "scroll_height": scroll_height,
        "viewport_height": viewport_height,
    }


async def _start_screencast(session: BrowserSession, quality: int = 60, fps: int = 5):
    """Start CDP screencast for live streaming."""
    if session._screencast_active:
        return

    def _on_frame(params):
        frame_data = base64.b64decode(params["data"])
        with session._frame_lock:
            session._last_frame = frame_data
        # Ack the frame
        _run_async(session.cdp.send("Page.screencastFrameAck", {
            "sessionId": params["sessionId"]
        }))
        # Notify callbacks
        for cb in session._screencast_callbacks:
            try:
                cb(frame_data)
            except Exception:
                pass

    session.cdp.on_event("Page.screencastFrame", _on_frame)
    await session.cdp.send("Page.startScreencast", {
        "format": "jpeg",
        "quality": quality,
        "maxWidth": session.viewport_width,
        "maxHeight": session.viewport_height,
        "everyNthFrame": max(1, 60 // fps),
    })
    session._screencast_active = True
    logger.info(f"Screencast started: quality={quality}, fps={fps}")


async def _stop_screencast(session: BrowserSession):
    """Stop CDP screencast."""
    if not session._screencast_active:
        return
    await session.cdp.send("Page.stopScreencast")
    session._screencast_active = False
    logger.info("Screencast stopped")


async def _close_session(session: BrowserSession):
    """Close browser session."""
    try:
        if session._screencast_active:
            await _stop_screencast(session)
        if session.cdp:
            await session.cdp.disconnect()
        if session.process:
            session.process.terminate()
            try:
                session.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                session.process.kill()
    except Exception as e:
        logger.debug(f"Error closing session: {e}")


# ═══════════════════════════════════════════════════════════════════
# Main Tool
# ═══════════════════════════════════════════════════════════════════

@tool
def browse(
    action: str = "screenshot",
    url: str = None,
    x: int = None,
    y: int = None,
    text: str = None,
    key: str = None,
    selector: str = None,
    expression: str = None,
    direction: str = "down",
    amount: int = 300,
    button: str = "left",
    tab_id: str = None,
    width: int = 120,
    height: int = 60,
    quality: int = 80,
    fps: int = 5,
    headless: bool = True,
    use_profile: bool = False,
    port: int = None,
    render: str = "ansi",
    click_count: int = 1,
) -> Dict[str, Any]:
    """
    🌐 Browse the web inline in terminal via Chrome DevTools Protocol.

    Renders web pages as halfblock characters in any terminal with 24-bit color.
    Can use YOUR Chrome profile (cookies, extensions, logins) or fresh profile.

    Actions:
        - "open": Launch browser and navigate to URL
        - "screenshot": Capture current page as terminal halfblocks
        - "navigate": Go to a new URL
        - "click": Click at (x, y) coordinates on the page
        - "type": Type text into focused element
        - "key": Press special key (enter, tab, escape, etc.)
        - "scroll": Scroll page (direction: up/down/left/right)
        - "tabs": List all open tabs
        - "switch_tab": Switch to tab by tab_id
        - "dom": Get DOM element by CSS selector
        - "eval": Evaluate JavaScript
        - "info": Get current page info (title, URL, scroll)
        - "attach": Connect to already-running Chrome (needs --remote-debugging-port)
        - "screencast_start": Start live frame streaming
        - "screencast_stop": Stop frame streaming
        - "screencast_frame": Get latest screencast frame
        - "close": Close browser session
        - "status": Show current session status

    Args:
        action: Action to perform
        url: URL to navigate to
        x: X coordinate for click
        y: Y coordinate for click
        text: Text to type
        key: Special key to press (enter, tab, escape, backspace, etc.)
        selector: CSS selector for DOM queries
        expression: JavaScript expression for eval
        direction: Scroll direction (up, down, left, right)
        amount: Scroll amount in pixels
        button: Mouse button (left, right, middle)
        tab_id: Tab ID for switch_tab
        width: Render width in terminal columns
        height: Render height in pixels (rows = height/2)
        quality: Screenshot quality (1-100)
        fps: Screencast frames per second
        headless: Run Chrome headless (True) or visible (False)
        use_profile: Use your Chrome profile (cookies, extensions, logins)
        port: Chrome debugging port
        render: Render format: "ansi", "rich", or "raw" (base64 PNG)
        click_count: Number of clicks (2 for double-click)

    Returns:
        Dict with status, rendered frame, and metadata

    Examples:
        browse(action="open", url="https://github.com", use_profile=True)
        browse(action="screenshot", width=160, height=80)
        browse(action="click", x=400, y=300)
        browse(action="type", text="hello world")
        browse(action="key", key="enter")
        browse(action="scroll", direction="down", amount=500)
        browse(action="navigate", url="https://example.com")
        browse(action="eval", expression="document.title")
        browse(action="attach", port=9222)
    """
    global _session

    try:
        # ── Status ──
        if action == "status":
            if _session and _session.cdp:
                try:
                    info = _run_async(_get_page_info(_session))
                except Exception:
                    info = {"url": _session.page_url}
                return {
                    "status": "success",
                    "content": [{
                        "text": f"🌐 Browser session active\n"
                                f"  URL: {info.get('url', '?')}\n"
                                f"  Title: {info.get('title', '?')}\n"
                                f"  Port: {_session.port}\n"
                                f"  Headless: {_session.headless}\n"
                                f"  Profile: {_session.use_profile}\n"
                                f"  Viewport: {_session.viewport_width}x{_session.viewport_height}\n"
                                f"  Screencast: {_session._screencast_active}"
                    }],
                }
            return {
                "status": "success",
                "content": [{"text": "🌐 No active browser session. Use action='open' to start."}],
            }

        # ── Open ──
        if action == "open":
            if _session and _session.cdp:
                _run_async(_close_session(_session))

            target_url = url or "about:blank"
            # Use 2:1 ratio for viewport → halfblock mapping.
            # Each terminal column = ~8px, each halfblock row = ~16px.
            # Higher viewport = sharper downscale. Cap at 1920x1080 for perf.
            vp_width = min(max(width * 8, 1280), 1920)
            vp_height = min(max(height * 8, 800), 1080)
            _session = _run_async(_launch_chrome(
                headless=headless,
                use_profile=use_profile,
                port=port,
                url=target_url,
                viewport_width=vp_width,
                viewport_height=vp_height,
            ))

            # Wait for page load
            time.sleep(2)

            # Take initial screenshot
            img_data = _run_async(_screenshot(_session, quality))
            rendered = HalfblockRenderer.render(img_data, width, height, render)

            try:
                info = _run_async(_get_page_info(_session))
            except Exception:
                info = {"title": "?", "url": target_url}

            return {
                "status": "success",
                "content": [{
                    "text": f"🌐 Browser opened: {info.get('title', '?')}\n"
                            f"   URL: {info.get('url', target_url)}\n"
                            f"   Port: {_session.port}\n\n"
                            f"{rendered}"
                }],
            }

        # ── Attach ──
        if action == "attach":
            if _session and _session.cdp:
                _run_async(_close_session(_session))

            attach_port = port or 9222
            _session = _run_async(_attach_to_chrome(attach_port))

            img_data = _run_async(_screenshot(_session, quality))
            rendered = HalfblockRenderer.render(img_data, width, height, render)

            return {
                "status": "success",
                "content": [{
                    "text": f"🌐 Attached to Chrome on port {attach_port}\n"
                            f"   URL: {_session.page_url}\n\n"
                            f"{rendered}"
                }],
            }

        # ── All remaining actions need an active session ──
        if not _session or not _session.cdp:
            return {
                "status": "error",
                "content": [{"text": "🌐 No active session. Use action='open' or action='attach' first."}],
            }

        # ── Screenshot ──
        if action == "screenshot":
            img_data = _run_async(_screenshot(_session, quality))

            if render == "raw":
                return {
                    "status": "success",
                    "content": [{"text": f"🌐 Screenshot captured ({len(img_data)} bytes)"}],
                    "image_data": base64.b64encode(img_data).decode(),
                }

            rendered = HalfblockRenderer.render(img_data, width, height, render)
            try:
                info = _run_async(_get_page_info(_session))
            except Exception:
                info = {}

            return {
                "status": "success",
                "content": [{
                    "text": f"🌐 {info.get('title', '?')} | {info.get('url', '?')}\n\n{rendered}"
                }],
            }

        # ── Navigate ──
        if action in ("navigate", "goto", "go"):
            if not url:
                return {"status": "error", "content": [{"text": "URL required for navigate"}]}

            _run_async(_navigate(_session, url))
            time.sleep(0.5)

            img_data = _run_async(_screenshot(_session, quality))
            rendered = HalfblockRenderer.render(img_data, width, height, render)
            info = _run_async(_get_page_info(_session))

            return {
                "status": "success",
                "content": [{
                    "text": f"🌐 Navigated to: {info.get('title', '?')}\n"
                            f"   URL: {info.get('url', url)}\n\n{rendered}"
                }],
            }

        # ── Click ──
        if action == "click":
            if x is None or y is None:
                return {"status": "error", "content": [{"text": "x and y coordinates required for click"}]}

            # Scale coordinates from terminal space to viewport space
            scale_x = _session.viewport_width / width if width > 0 else 1
            scale_y = _session.viewport_height / (height / 2) if height > 0 else 1
            real_x = int(x * scale_x)
            real_y = int(y * scale_y)

            _run_async(_click(_session, real_x, real_y, button, click_count))
            time.sleep(0.5)

            img_data = _run_async(_screenshot(_session, quality))
            rendered = HalfblockRenderer.render(img_data, width, height, render)

            return {
                "status": "success",
                "content": [{
                    "text": f"🌐 Clicked ({real_x}, {real_y})\n\n{rendered}"
                }],
            }

        # ── Type ──
        if action == "type":
            if not text:
                return {"status": "error", "content": [{"text": "text required for type action"}]}

            _run_async(_type_text(_session, text))
            time.sleep(0.3)

            img_data = _run_async(_screenshot(_session, quality))
            rendered = HalfblockRenderer.render(img_data, width, height, render)

            return {
                "status": "success",
                "content": [{"text": f"🌐 Typed: '{text}'\n\n{rendered}"}],
            }

        # ── Key Press ──
        if action == "key":
            if not key:
                return {"status": "error", "content": [{"text": "key required (enter, tab, escape, etc.)"}]}

            _run_async(_press_key(_session, key))
            time.sleep(0.3)

            img_data = _run_async(_screenshot(_session, quality))
            rendered = HalfblockRenderer.render(img_data, width, height, render)

            return {
                "status": "success",
                "content": [{"text": f"🌐 Pressed: {key}\n\n{rendered}"}],
            }

        # ── Scroll ──
        if action == "scroll":
            _run_async(_scroll(_session, direction, amount))

            img_data = _run_async(_screenshot(_session, quality))
            rendered = HalfblockRenderer.render(img_data, width, height, render)

            return {
                "status": "success",
                "content": [{"text": f"🌐 Scrolled {direction} {amount}px\n\n{rendered}"}],
            }

        # ── Tabs ──
        if action == "tabs":
            tabs = _run_async(_get_tabs(_session.port))
            tab_list = "\n".join(
                f"  {'→ ' if t['url'] == _session.page_url else '  '}"
                f"[{t['id'][:8]}] {t['title'][:50]} — {t['url'][:80]}"
                for t in tabs
            )
            return {
                "status": "success",
                "content": [{"text": f"🌐 Open tabs ({len(tabs)}):\n{tab_list}"}],
            }

        # ── Switch Tab ──
        if action == "switch_tab":
            if not tab_id:
                return {"status": "error", "content": [{"text": "tab_id required"}]}

            # Allow partial tab_id match
            tabs = _run_async(_get_tabs(_session.port))
            matched = None
            for t in tabs:
                if t["id"].startswith(tab_id):
                    matched = t["id"]
                    break
            if not matched:
                return {"status": "error", "content": [{"text": f"Tab '{tab_id}' not found"}]}

            _run_async(_switch_tab(_session, matched))
            time.sleep(0.5)

            img_data = _run_async(_screenshot(_session, quality))
            rendered = HalfblockRenderer.render(img_data, width, height, render)

            return {
                "status": "success",
                "content": [{"text": f"🌐 Switched to tab: {_session.page_url}\n\n{rendered}"}],
            }

        # ── DOM ──
        if action == "dom":
            result = _run_async(_get_dom(_session, selector))
            if result is None:
                return {"status": "error", "content": [{"text": f"Element not found: {selector}"}]}
            return {
                "status": "success",
                "content": [{"text": f"🌐 DOM{f' ({selector})' if selector else ''}:\n{str(result)[:5000]}"}],
            }

        # ── Eval JS ──
        if action == "eval":
            if not expression:
                return {"status": "error", "content": [{"text": "expression required for eval"}]}
            result = _run_async(_evaluate_js(_session, expression))
            return {
                "status": "success",
                "content": [{"text": f"🌐 JS result: {json.dumps(result, default=str)[:5000]}"}],
            }

        # ── Page Info ──
        if action == "info":
            info = _run_async(_get_page_info(_session))
            return {
                "status": "success",
                "content": [{
                    "text": f"🌐 Page Info:\n"
                            f"  Title: {info.get('title', '?')}\n"
                            f"  URL: {info.get('url', '?')}\n"
                            f"  Scroll: {info.get('scroll_y', 0)}px / {info.get('scroll_height', 0)}px\n"
                            f"  Viewport: {info.get('viewport_height', 0)}px"
                }],
            }

        # ── Screencast Start ──
        if action == "screencast_start":
            _run_async(_start_screencast(_session, quality, fps))
            return {
                "status": "success",
                "content": [{"text": f"🌐 Screencast started (quality={quality}, fps={fps})"}],
            }

        # ── Screencast Stop ──
        if action == "screencast_stop":
            _run_async(_stop_screencast(_session))
            return {
                "status": "success",
                "content": [{"text": "🌐 Screencast stopped"}],
            }

        # ── Screencast Frame ──
        if action == "screencast_frame":
            with _session._frame_lock:
                frame = _session._last_frame

            if not frame:
                return {"status": "error", "content": [{"text": "No screencast frame available"}]}

            rendered = HalfblockRenderer.render(frame, width, height, render)
            return {
                "status": "success",
                "content": [{"text": rendered}],
            }

        # ── Close ──
        if action == "close":
            _run_async(_close_session(_session))
            _session = None
            return {
                "status": "success",
                "content": [{"text": "🌐 Browser session closed"}],
            }

        return {
            "status": "error",
            "content": [{
                "text": f"Unknown action: {action}. Valid: open, screenshot, navigate, click, type, key, "
                        f"scroll, tabs, switch_tab, dom, eval, info, attach, screencast_start, "
                        f"screencast_stop, screencast_frame, close, status"
            }],
        }

    except Exception as e:
        logger.error(f"Browse error: {e}")
        return {"status": "error", "content": [{"text": f"🌐 Error: {str(e)}"}]}


# ═══════════════════════════════════════════════════════════════════
# Utility: Get session for TUI integration
# ═══════════════════════════════════════════════════════════════════

def get_browser_session() -> Optional[BrowserSession]:
    """Get the current browser session (for TUI widget integration)."""
    return _session


def get_renderer():
    """Get the HalfblockRenderer class (for external use)."""
    return HalfblockRenderer
