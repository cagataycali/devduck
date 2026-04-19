#!/usr/bin/env python3
"""
🌐 DevDuck tunnel — expose your agent to the public internet via Cloudflare Tunnel.

Think of it as Zenoh for the public internet:
  • Zero inbound ports, survives NAT
  • Quick mode: random *.trycloudflare.com URL (zero auth)
  • Named mode: stable hostname + DNS + Access policies

Typical workflow:
  tunnel(action="install")                             # brew/apt install cloudflared
  tunnel(action="start", name="ws", port=10001)        # quick tunnel to WS server
  tunnel(action="list")                                # see all active tunnels
  tunnel(action="stop", name="ws")

Named tunnels (persistent):
  tunnel(action="login")                               # opens browser for CF auth
  tunnel(action="create_named", name="my-agent",
         hostname="agent.example.com", port=10001)
  tunnel(action="start", name="my-agent")
"""

import os
import re
import sys
import json
import time
import signal
import shutil
import logging
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from strands import tool

logger = logging.getLogger(__name__)

# ─────────── state ───────────

TUNNEL_DIR = Path.home() / ".devduck" / "tunnels"
TUNNEL_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = TUNNEL_DIR / "tunnels.json"
LOG_DIR = TUNNEL_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# In-memory handle to running subprocesses { name -> Popen }
_PROCS: Dict[str, subprocess.Popen] = {}


def _load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _which_cloudflared() -> Optional[str]:
    return shutil.which("cloudflared")


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _err(msg: str) -> Dict[str, Any]:
    return {"status": "error", "content": [{"text": msg}]}


def _ok(msg: str) -> Dict[str, Any]:
    return {"status": "success", "content": [{"text": msg}]}


def _install_cloudflared() -> Dict[str, Any]:
    """Install cloudflared via brew (macOS) or apt (Linux)."""
    if _which_cloudflared():
        return _ok(f"cloudflared already installed: {_which_cloudflared()}")

    system = platform.system()
    try:
        if system == "Darwin":
            if not shutil.which("brew"):
                return _err("Homebrew not found. Install from https://brew.sh or download cloudflared manually.")
            subprocess.run(["brew", "install", "cloudflared"], check=True)
        elif system == "Linux":
            # Use cloudflare repo or direct .deb
            arch = platform.machine()
            arch_map = {"x86_64": "amd64", "aarch64": "arm64", "armv7l": "arm"}
            deb_arch = arch_map.get(arch, "amd64")
            url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{deb_arch}.deb"
            tmp = Path("/tmp/cloudflared.deb")
            subprocess.run(["curl", "-L", "-o", str(tmp), url], check=True)
            subprocess.run(["sudo", "dpkg", "-i", str(tmp)], check=True)
        else:
            return _err(f"Unsupported platform: {system}")
        return _ok(f"✅ cloudflared installed: {_which_cloudflared()}")
    except subprocess.CalledProcessError as e:
        return _err(f"Install failed: {e}")


def _extract_quick_url(log_path: Path, timeout: float = 20.0) -> Optional[str]:
    """Poll the log file for a *.trycloudflare.com URL."""
    pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if log_path.exists():
            try:
                content = log_path.read_text(errors="ignore")
                m = pattern.search(content)
                if m:
                    return m.group(0)
            except Exception:
                pass
        time.sleep(0.5)
    return None


def _start_quick_tunnel(name: str, port: int, host: str = "localhost",
                       protocol: str = "http") -> Dict[str, Any]:
    """Start an ephemeral tunnel to http://localhost:PORT."""
    cf = _which_cloudflared()
    if not cf:
        return _err("cloudflared not installed. Run tunnel(action='install') first.")

    state = _load_state()
    if name in state and _pid_alive(state[name].get("pid", 0)):
        return _err(f"Tunnel '{name}' already running (pid={state[name]['pid']})")

    url = f"{protocol}://{host}:{port}"
    log_path = LOG_DIR / f"{name}.log"
    # truncate
    log_path.write_text("")

    logf = open(log_path, "ab")
    try:
        proc = subprocess.Popen(
            [cf, "tunnel", "--no-autoupdate", "--url", url],
            stdout=logf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        return _err(f"Failed to spawn cloudflared: {e}")

    _PROCS[name] = proc

    # Wait for the public URL
    public_url = _extract_quick_url(log_path, timeout=25.0)
    if not public_url:
        # Kill on failure
        try:
            proc.terminate()
        except Exception:
            pass
        tail = log_path.read_text(errors="ignore")[-2000:]
        return _err(f"Timed out waiting for tunnel URL. Logs:\n{tail}")

    entry = {
        "name": name,
        "mode": "quick",
        "port": port,
        "host": host,
        "protocol": protocol,
        "local_url": url,
        "public_url": public_url,
        "pid": proc.pid,
        "log": str(log_path),
        "started_at": time.time(),
    }
    state[name] = entry
    _save_state(state)

    return _ok(
        f"✅ Tunnel '{name}' started\n"
        f"   Local:  {url}\n"
        f"   Public: {public_url}\n"
        f"   PID:    {proc.pid}\n"
        f"   Log:    {log_path}\n\n"
        f"💡 Note: trycloudflare.com URLs are ephemeral — use 'create_named' for stable hostnames."
    )


def _start_named_tunnel(name: str) -> Dict[str, Any]:
    """Start a pre-configured named tunnel by name."""
    cf = _which_cloudflared()
    if not cf:
        return _err("cloudflared not installed.")

    state = _load_state()
    entry = state.get(name)
    if not entry:
        return _err(f"No tunnel config for '{name}'. Use create_named first.")
    if entry.get("mode") != "named":
        return _err(f"Tunnel '{name}' is not a named tunnel. Use start with port= for quick tunnels.")
    if _pid_alive(entry.get("pid", 0)):
        return _err(f"Already running (pid={entry['pid']})")

    tunnel_id = entry.get("tunnel_id") or name
    log_path = LOG_DIR / f"{name}.log"
    log_path.write_text("")
    logf = open(log_path, "ab")

    try:
        proc = subprocess.Popen(
            [cf, "tunnel", "--no-autoupdate", "--config", str(Path.home() / ".cloudflared" / f"{name}.yml"), "run", tunnel_id],
            stdout=logf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        return _err(f"Spawn failed: {e}")

    _PROCS[name] = proc
    entry["pid"] = proc.pid
    entry["started_at"] = time.time()
    entry["log"] = str(log_path)
    state[name] = entry
    _save_state(state)

    return _ok(
        f"✅ Named tunnel '{name}' started\n"
        f"   Hostname: {entry.get('hostname', 'unknown')}\n"
        f"   Local:    {entry.get('local_url', 'unknown')}\n"
        f"   PID:      {proc.pid}\n"
        f"   Log:      {log_path}"
    )


def _stop_tunnel(name: str) -> Dict[str, Any]:
    state = _load_state()
    if name not in state:
        return _err(f"No tunnel '{name}' found")
    entry = state[name]
    pid = entry.get("pid", 0)
    if pid and _pid_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        # Wait briefly
        for _ in range(20):
            if not _pid_alive(pid):
                break
            time.sleep(0.1)
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
    entry["pid"] = 0
    entry["stopped_at"] = time.time()

    # For quick tunnels, remove entirely (they can't restart anyway)
    if entry.get("mode") == "quick":
        del state[name]
    else:
        state[name] = entry

    _save_state(state)
    _PROCS.pop(name, None)
    return _ok(f"🛑 Tunnel '{name}' stopped")


def _list_tunnels() -> Dict[str, Any]:
    state = _load_state()
    if not state:
        return _ok("No tunnels configured. Use tunnel(action='start', name=..., port=...) to create one.")

    lines = ["🌐 DevDuck Tunnels:\n"]
    for name, entry in state.items():
        pid = entry.get("pid", 0)
        alive = _pid_alive(pid)
        status = "🟢 running" if alive else "🔴 stopped"
        mode = entry.get("mode", "?")
        lines.append(f"  {status}  [{mode:5}] {name}")
        if entry.get("public_url"):
            lines.append(f"             public: {entry['public_url']}")
        if entry.get("hostname"):
            lines.append(f"             hostname: {entry['hostname']}")
        lines.append(f"             local:  {entry.get('local_url', '?')}")
        if pid:
            lines.append(f"             pid:    {pid}")
        lines.append("")
    return _ok("\n".join(lines))


def _status_tunnel(name: str) -> Dict[str, Any]:
    state = _load_state()
    if name not in state:
        return _err(f"No tunnel '{name}'")
    entry = state[name]
    pid = entry.get("pid", 0)
    alive = _pid_alive(pid)
    entry["alive"] = alive
    return _ok(json.dumps(entry, indent=2))


def _logs_tunnel(name: str, lines: int = 50) -> Dict[str, Any]:
    state = _load_state()
    entry = state.get(name)
    if not entry:
        return _err(f"No tunnel '{name}'")
    log_path = Path(entry.get("log", LOG_DIR / f"{name}.log"))
    if not log_path.exists():
        return _err(f"No log file at {log_path}")
    content = log_path.read_text(errors="ignore").splitlines()
    tail = "\n".join(content[-lines:])
    return _ok(f"📜 Last {lines} lines of {log_path}:\n\n{tail}")


def _login_cloudflare() -> Dict[str, Any]:
    """Run `cloudflared tunnel login` — opens browser for CF auth."""
    cf = _which_cloudflared()
    if not cf:
        return _err("cloudflared not installed.")
    try:
        # This opens a browser and waits for the user. Run interactively.
        print("🌐 Opening Cloudflare login in your browser...")
        print("   Select the zone you want to authorize.")
        result = subprocess.run([cf, "tunnel", "login"], check=False)
        if result.returncode == 0:
            cert = Path.home() / ".cloudflared" / "cert.pem"
            if cert.exists():
                return _ok(f"✅ Authorized. Cert saved to {cert}")
            return _ok("✅ Login command completed. Check ~/.cloudflared/cert.pem")
        return _err(f"Login failed with code {result.returncode}")
    except Exception as e:
        return _err(f"Login error: {e}")


def _create_named(name: str, hostname: str, port: int,
                  host: str = "localhost", protocol: str = "http") -> Dict[str, Any]:
    """Create a named tunnel + DNS route + config file."""
    cf = _which_cloudflared()
    if not cf:
        return _err("cloudflared not installed.")

    cert = Path.home() / ".cloudflared" / "cert.pem"
    if not cert.exists():
        return _err("Not authorized. Run tunnel(action='login') first.")

    # 1. Create the tunnel
    try:
        result = subprocess.run(
            [cf, "tunnel", "create", name],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0 and "already exists" not in result.stderr.lower():
            return _err(f"Create failed: {result.stderr or result.stdout}")
    except Exception as e:
        return _err(f"Create error: {e}")

    # 2. Get the tunnel ID
    try:
        list_res = subprocess.run(
            [cf, "tunnel", "list", "--output", "json"],
            capture_output=True, text=True, check=True,
        )
        tunnels = json.loads(list_res.stdout)
        tunnel_id = None
        for t in tunnels:
            if t.get("name") == name:
                tunnel_id = t.get("id")
                break
        if not tunnel_id:
            return _err(f"Tunnel '{name}' created but couldn't find ID")
    except Exception as e:
        return _err(f"Failed to fetch tunnel ID: {e}")

    # 3. Write config file
    config_dir = Path.home() / ".cloudflared"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / f"{name}.yml"
    creds_path = config_dir / f"{tunnel_id}.json"
    local_url = f"{protocol}://{host}:{port}"

    config_yaml = (
        f"tunnel: {tunnel_id}\n"
        f"credentials-file: {creds_path}\n"
        f"ingress:\n"
        f"  - hostname: {hostname}\n"
        f"    service: {local_url}\n"
        f"    originRequest:\n"
        f"      connectTimeout: 30s\n"
        f"      noTLSVerify: true\n"
        f"  - service: http_status:404\n"
    )
    config_path.write_text(config_yaml)

    # 4. Create DNS route
    try:
        dns_res = subprocess.run(
            [cf, "tunnel", "route", "dns", name, hostname],
            capture_output=True, text=True, check=False,
        )
        if dns_res.returncode != 0 and "already exists" not in (dns_res.stderr + dns_res.stdout).lower():
            return _err(f"DNS route failed: {dns_res.stderr or dns_res.stdout}")
    except Exception as e:
        return _err(f"DNS route error: {e}")

    # 5. Persist state
    state = _load_state()
    state[name] = {
        "name": name,
        "mode": "named",
        "tunnel_id": tunnel_id,
        "hostname": hostname,
        "port": port,
        "host": host,
        "protocol": protocol,
        "local_url": local_url,
        "public_url": f"https://{hostname}",
        "config": str(config_path),
        "credentials": str(creds_path),
        "pid": 0,
        "created_at": time.time(),
    }
    _save_state(state)

    return _ok(
        f"✅ Named tunnel '{name}' created\n"
        f"   Tunnel ID:  {tunnel_id}\n"
        f"   Hostname:   https://{hostname}\n"
        f"   Routes to:  {local_url}\n"
        f"   Config:     {config_path}\n\n"
        f"▶️  Start it with: tunnel(action='start', name='{name}')"
    )


def _delete_named(name: str) -> Dict[str, Any]:
    cf = _which_cloudflared()
    if not cf:
        return _err("cloudflared not installed.")
    state = _load_state()
    entry = state.get(name)
    if entry and _pid_alive(entry.get("pid", 0)):
        _stop_tunnel(name)
    try:
        subprocess.run([cf, "tunnel", "delete", "-f", name], check=False)
    except Exception:
        pass
    if name in state:
        del state[name]
        _save_state(state)
    return _ok(f"🗑  Tunnel '{name}' deleted")


# ─────────── main tool ───────────

@tool
def tunnel(
    action: str,
    name: str = "devduck",
    port: int = 0,
    host: str = "localhost",
    protocol: str = "http",
    hostname: str = "",
    lines: int = 50,
) -> Dict[str, Any]:
    """
    🌐 Expose DevDuck to the public internet via Cloudflare Tunnel.

    Actions:
      - "install":       Install cloudflared binary (brew/apt)
      - "login":         Authorize with Cloudflare (opens browser)
      - "start":         Start tunnel. If hostname is empty → quick tunnel
                         (random *.trycloudflare.com). Otherwise runs named tunnel.
      - "stop":          Stop a running tunnel
      - "list":          List all tunnels and their status
      - "status":        Detailed status for one tunnel
      - "logs":          Tail cloudflared logs for a tunnel
      - "create_named":  Create a persistent named tunnel + DNS route
                         (requires login first)
      - "delete_named":  Delete a named tunnel from Cloudflare

    Examples:
      # Quick tunnel to WebSocket server
      tunnel(action="start", name="ws", port=10001)

      # Expose MCP server
      tunnel(action="start", name="mcp", port=10003)

      # Named persistent tunnel
      tunnel(action="login")
      tunnel(action="create_named", name="agent",
             hostname="agent.example.com", port=10001)
      tunnel(action="start", name="agent")

      # List + stop
      tunnel(action="list")
      tunnel(action="stop", name="ws")

    Args:
        action:   Action to perform (see above)
        name:     Tunnel name (unique, default "devduck")
        port:     Local port to expose (required for start/create_named)
        host:     Local host (default "localhost")
        protocol: "http" or "https" (default "http")
        hostname: Public DNS hostname (for named tunnels only)
        lines:    Number of log lines (for logs action)

    Returns:
        Dict with status and content
    """
    try:
        if action == "install":
            return _install_cloudflared()
        if action == "login":
            return _login_cloudflare()
        if action == "list":
            return _list_tunnels()
        if action == "status":
            return _status_tunnel(name)
        if action == "logs":
            return _logs_tunnel(name, lines)
        if action == "stop":
            return _stop_tunnel(name)
        if action == "delete_named":
            return _delete_named(name)
        if action == "create_named":
            if not hostname:
                return _err("hostname required for create_named")
            if not port:
                return _err("port required for create_named")
            return _create_named(name, hostname, port, host, protocol)
        if action == "start":
            state = _load_state()
            existing = state.get(name)
            # If existing named tunnel config → run it
            if existing and existing.get("mode") == "named":
                return _start_named_tunnel(name)
            # Otherwise quick tunnel
            if not port:
                return _err("port required for quick tunnel start")
            return _start_quick_tunnel(name, port, host, protocol)

        return _err(f"Unknown action: {action}")
    except Exception as e:
        logger.exception("tunnel tool failed")
        return _err(f"Error: {e}")


# ─────────── CLI subcommand (`devduck tunnel ...`) ───────────

def register_parser(subparsers) -> None:
    """Attach the `tunnel` subcommand to a parent argparse subparsers object."""
    p = subparsers.add_parser(
        "tunnel",
        help="Expose DevDuck to the public internet via Cloudflare Tunnel",
        description=(
            "Manage Cloudflare tunnels for public exposure of DevDuck services.\n"
            "Quick tunnels (zero auth) or named tunnels (persistent + DNS)."
        ),
    )
    p_sub = p.add_subparsers(dest="tunnel_command", required=True)

    def _name_arg(sp, default="devduck"):
        sp.add_argument("--name", "-n", default=default, help=f"Tunnel name (default: {default})")

    # install
    p_sub.add_parser("install", help="Install cloudflared binary (brew/apt)")

    # login
    p_sub.add_parser("login", help="Authorize with Cloudflare (opens browser)")

    # start
    sp = p_sub.add_parser("start", help="Start a tunnel (quick or named)")
    _name_arg(sp)
    sp.add_argument("--port", "-p", type=int, default=0, help="Local port to expose")
    sp.add_argument("--host", default="localhost", help="Local host (default: localhost)")
    sp.add_argument("--protocol", default="http", choices=["http", "https"], help="Protocol")

    # stop / status / logs
    sp = p_sub.add_parser("stop", help="Stop a running tunnel")
    _name_arg(sp)
    sp = p_sub.add_parser("status", help="Show detailed status of a tunnel")
    _name_arg(sp)
    sp = p_sub.add_parser("logs", help="Tail cloudflared logs for a tunnel")
    _name_arg(sp)
    sp.add_argument("--lines", type=int, default=50, help="Number of lines")

    # list
    p_sub.add_parser("list", help="List all tunnels")

    # create_named
    sp = p_sub.add_parser("create-named", help="Create a persistent named tunnel + DNS")
    _name_arg(sp)
    sp.add_argument("--hostname", "-H", required=True, help="Public DNS hostname (e.g. agent.example.com)")
    sp.add_argument("--port", "-p", type=int, required=True, help="Local port to expose")
    sp.add_argument("--host", default="localhost", help="Local host (default: localhost)")
    sp.add_argument("--protocol", default="http", choices=["http", "https"], help="Protocol")

    # delete_named
    sp = p_sub.add_parser("delete-named", help="Delete a named tunnel from Cloudflare")
    _name_arg(sp)


def _print_result(result: Dict[str, Any]) -> int:
    """Print a tool result dict to stdout and return exit code."""
    for item in result.get("content", []):
        print(item.get("text", ""))
    return 0 if result.get("status") == "success" else 1


def dispatch(args) -> int:
    """Dispatch a parsed tunnel subcommand."""
    cmd = args.tunnel_command

    if cmd == "install":
        return _print_result(_install_cloudflared())
    if cmd == "login":
        return _print_result(_login_cloudflare())
    if cmd == "list":
        return _print_result(_list_tunnels())
    if cmd == "status":
        return _print_result(_status_tunnel(args.name))
    if cmd == "logs":
        return _print_result(_logs_tunnel(args.name, args.lines))
    if cmd == "stop":
        return _print_result(_stop_tunnel(args.name))
    if cmd == "delete-named":
        return _print_result(_delete_named(args.name))
    if cmd == "create-named":
        return _print_result(_create_named(
            args.name, args.hostname, args.port, args.host, args.protocol
        ))
    if cmd == "start":
        state = _load_state()
        existing = state.get(args.name)
        if existing and existing.get("mode") == "named":
            return _print_result(_start_named_tunnel(args.name))
        if not args.port:
            return _print_result(_err("--port required for quick tunnel start"))
        return _print_result(_start_quick_tunnel(args.name, args.port, args.host, args.protocol))

    print(f"unknown tunnel command: {cmd}")
    return 2


# ─────────── Auto-start hook ───────────

def auto_start_tunnels() -> None:
    """
    Auto-start tunnels based on env vars or saved state.

    Env var config (optional):
      DEVDUCK_TUNNEL_AUTO=true            → start tunnels from saved state
      DEVDUCK_TUNNEL_WS=true              → quick tunnel for WS port (10001)
      DEVDUCK_TUNNEL_MCP=true             → quick tunnel for MCP port (10003)
      DEVDUCK_TUNNEL_AGENTCORE=true       → quick tunnel for proxy port (10000)
      DEVDUCK_TUNNEL_PORTS="10001,10003"  → quick tunnels for arbitrary ports
    """
    if not _which_cloudflared():
        return  # silently skip if cloudflared not installed

    started_any = False

    # 1. Restart any saved named tunnels
    if os.getenv("DEVDUCK_TUNNEL_AUTO", "false").lower() == "true":
        state = _load_state()
        for name, entry in state.items():
            if entry.get("mode") == "named" and not _pid_alive(entry.get("pid", 0)):
                result = _start_named_tunnel(name)
                if result.get("status") == "success":
                    logger.info(f"✓ Auto-started named tunnel: {name}")
                    print(f"🌐 ✓ Tunnel (named): {entry.get('hostname')}")
                    started_any = True

    # 2. Shortcuts for common DevDuck ports
    shortcuts = [
        ("DEVDUCK_TUNNEL_WS", "ws-auto", int(os.getenv("DEVDUCK_WS_PORT", "10001"))),
        ("DEVDUCK_TUNNEL_MCP", "mcp-auto", int(os.getenv("DEVDUCK_MCP_PORT", "10003"))),
        ("DEVDUCK_TUNNEL_AGENTCORE", "agentcore-auto",
         int(os.getenv("DEVDUCK_AGENTCORE_PROXY_PORT", "10000"))),
    ]
    for env_key, name, port in shortcuts:
        if os.getenv(env_key, "false").lower() != "true":
            continue
        state = _load_state()
        if name in state and _pid_alive(state[name].get("pid", 0)):
            continue  # already running
        result = _start_quick_tunnel(name, port)
        if result.get("status") == "success":
            # Re-read to get public URL
            entry = _load_state().get(name, {})
            public = entry.get("public_url", "?")
            logger.info(f"✓ Auto-started {name} → {public}")
            print(f"🌐 ✓ Tunnel ({name}): {public}")
            started_any = True

    # 3. Arbitrary ports from comma-separated list
    ports_str = os.getenv("DEVDUCK_TUNNEL_PORTS", "").strip()
    if ports_str:
        for port_str in ports_str.split(","):
            port_str = port_str.strip()
            if not port_str.isdigit():
                continue
            port = int(port_str)
            name = f"port-{port}"
            state = _load_state()
            if name in state and _pid_alive(state[name].get("pid", 0)):
                continue
            result = _start_quick_tunnel(name, port)
            if result.get("status") == "success":
                entry = _load_state().get(name, {})
                public = entry.get("public_url", "?")
                logger.info(f"✓ Auto-started {name} → {public}")
                print(f"🌐 ✓ Tunnel (port {port}): {public}")
                started_any = True

    if not started_any:
        logger.debug("No tunnels to auto-start")
