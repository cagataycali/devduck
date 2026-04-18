#!/usr/bin/env python3
"""
🦆 devduck service — install devduck as a persistent system service.

Supports:
  • Linux  → systemd  (user-level default, --system for root)
  • macOS  → launchd  (LaunchAgent default, --system for LaunchDaemon)
  • Remote → SSH-in and install on any reachable Linux host

Fully covered config via CLI flags + env file:
  • model / tools / MCP servers / system prompt / startup prompt
  • arbitrary KEY=VALUE env injection
  • log paths, memory limits, restart policy
  • self-healing wrapper (pydantic-core fix, crash-recovery)
"""
from __future__ import annotations

import os
import sys
import shlex
import shutil
import getpass
import platform
import subprocess
import textwrap
from pathlib import Path
from typing import Optional, Sequence, Dict, List


# ─────────────────────────── helpers ──────────────────────────────

def _is_linux() -> bool:
    return platform.system() == "Linux"


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _run(cmd: Sequence[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a local command."""
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def _ssh_run(host: str, remote_cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command on a remote host via SSH."""
    return subprocess.run(
        ["ssh", host, "bash", "-s"],
        input=remote_cmd,
        check=check,
        text=True,
    )


def _ssh_write(host: str, remote_path: str, content: str, mode: str = "0644") -> None:
    """Write content to a file on a remote host via SSH."""
    # Encode as base64 to avoid any quoting hell
    import base64
    b64 = base64.b64encode(content.encode()).decode()
    cmd = f"""
mkdir -p {shlex.quote(str(Path(remote_path).parent))}
echo {shlex.quote(b64)} | base64 -d > {shlex.quote(remote_path)}
chmod {mode} {shlex.quote(remote_path)}
"""
    _ssh_run(host, cmd)


def _local_write(path: Path, content: str, mode: int = 0o644) -> None:
    """Write content to a local file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(mode)


# ─────────────────── install plan (pure data) ─────────────────────

class InstallPlan:
    """
    Computes all file paths & content for a devduck service install.

    Designed to be executed locally OR piped over SSH.
    """

    def __init__(
        self,
        name: str = "devduck",
        user: Optional[str] = None,
        home: Optional[str] = None,
        system: bool = False,
        model: Optional[str] = None,
        model_provider: Optional[str] = None,
        tools: Optional[str] = None,
        system_prompt: Optional[str] = None,
        startup_prompt: Optional[str] = None,
        mcp_servers: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        work_dir: Optional[str] = None,
        restart_sec: int = 15,
        memory_max: str = "8G",
        description: Optional[str] = None,
        platform_override: Optional[str] = None,  # "linux" | "macos"
    ) -> None:
        self.name = name
        self.system = system
        self.user = user or getpass.getuser()
        self.home = home or str(Path.home())
        self.model = model
        self.model_provider = model_provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.startup_prompt = startup_prompt or self._default_startup_prompt()
        self.mcp_servers = mcp_servers
        self.env_vars = env_vars or {}
        self.work_dir = work_dir or self.home
        self.restart_sec = restart_sec
        self.memory_max = memory_max
        self.description = description or f"🦆 DevDuck Service ({name})"
        self.platform = platform_override or ("linux" if _is_linux() else "macos" if _is_macos() else "linux")

    @property
    def service_name(self) -> str:
        return f"devduck-{self.name}" if self.name != "devduck" else "devduck"

    # ---------- paths ----------

    @property
    def env_file(self) -> str:
        if self.platform == "linux":
            return (f"/etc/devduck/{self.service_name}.env"
                    if self.system
                    else f"{self.home}/.config/devduck/{self.service_name}.env")
        # macos
        return (f"/Library/Application Support/devduck/{self.service_name}.env"
                if self.system
                else f"{self.home}/Library/Application Support/devduck/{self.service_name}.env")

    @property
    def wrapper_path(self) -> str:
        if self.platform == "linux":
            return (f"/usr/local/bin/{self.service_name}-agent"
                    if self.system
                    else f"{self.home}/.local/bin/{self.service_name}-agent")
        return (f"/usr/local/bin/{self.service_name}-agent"
                if self.system
                else f"{self.home}/.local/bin/{self.service_name}-agent")

    @property
    def unit_path(self) -> str:
        if self.platform == "linux":
            return (f"/etc/systemd/system/{self.service_name}.service"
                    if self.system
                    else f"{self.home}/.config/systemd/user/{self.service_name}.service")
        # macos launchd plist
        label = f"com.devduck.{self.name}"
        return (f"/Library/LaunchDaemons/{label}.plist"
                if self.system
                else f"{self.home}/Library/LaunchAgents/{label}.plist")

    @property
    def launchd_label(self) -> str:
        return f"com.devduck.{self.name}"

    @property
    def log_path(self) -> str:
        if self.platform == "linux":
            return (f"/var/log/{self.service_name}.log"
                    if self.system
                    else f"{self.home}/.cache/{self.service_name}.log")
        return (f"/var/log/{self.service_name}.log"
                if self.system
                else f"{self.home}/Library/Logs/{self.service_name}.log")

    # ---------- content ----------

    def _default_startup_prompt(self) -> str:
        return (
            "You are DevDuck running as a persistent system service. "
            "Initialize your enabled tools (e.g. telegram/whatsapp/slack listeners, scheduler). "
            "Announce readiness where appropriate. Then stay alive — do NOT exit. "
            "If you have nothing to do, sleep in long intervals."
        )

    def env_file_content(self) -> str:
        lines = [
            "# 🦆 devduck service environment",
            f"# Service: {self.service_name}",
            "# Managed by `devduck service install` — edit with `devduck service edit`",
            "",
            "BYPASS_TOOL_CONSENT=true",
            "DEVDUCK_AUTO_START_SERVERS=false",
        ]

        if self.model_provider:
            lines.append(f"MODEL_PROVIDER={self.model_provider}")
        if self.model:
            lines.append(f"STRANDS_MODEL_ID={self.model}")
        if self.tools:
            lines.append(f"DEVDUCK_TOOLS={self.tools}")
        if self.system_prompt:
            # Keep single-line; env files don't do multi-line well. Use \\n for newlines.
            escaped = self.system_prompt.replace("\n", "\\n").replace('"', '\\"')
            lines.append(f'SYSTEM_PROMPT="{escaped}"')
        if self.mcp_servers:
            # caller already passes valid JSON
            lines.append(f"MCP_SERVERS={self.mcp_servers}")

        lines.append("")
        lines.append("# --- User-provided env ---")
        for k, v in self.env_vars.items():
            if "\n" in v:
                raise ValueError(f"env var {k} contains newline")
            lines.append(f"{k}={v}")

        # ensure PATH so shell tool finds things on Linux hosts
        if self.platform == "linux" and "PATH" not in self.env_vars:
            lines.append(
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                f":{self.home}/.local/bin"
            )
        if "HOME" not in self.env_vars:
            lines.append(f"HOME={self.home}")

        return "\n".join(lines) + "\n"

    def wrapper_content(self) -> str:
        """
        Self-healing launcher script. Runs devduck with the startup prompt via
        the Python API to bypass argparse (so prompts with emojis/flags-like
        text don't confuse the CLI parser).
        """
        # The prompt is baked into the wrapper as a heredoc-safe Python triple-
        # string. We escape only triple-quotes.
        prompt_safe = self.startup_prompt.replace('"""', '\\"\\"\\"')

        # The wrapper is written as bash that execs `python - <<'PYEOF'`.
        # This ensures argparse is never invoked with the prompt.
        return textwrap.dedent(f"""\
            #!/usr/bin/env bash
            # 🦆 devduck-{self.name} service wrapper — auto-generated
            set -eu
            cd {shlex.quote(self.work_dir)}

            # Find devduck's python interpreter
            if command -v devduck >/dev/null 2>&1; then
                DEVDUCK_BIN="$(command -v devduck)"
                # Resolve symlinks to find the real pipx venv
                DEVDUCK_REAL="$(readlink -f "$DEVDUCK_BIN" 2>/dev/null || echo "$DEVDUCK_BIN")"
                DEVDUCK_PY="$(dirname "$DEVDUCK_REAL")/python"
            else
                echo "devduck not found in PATH" >&2
                exit 127
            fi

            # Self-heal common pydantic-core/strands dep issues
            "$DEVDUCK_PY" -c "import pydantic" 2>/dev/null || \\
                "$DEVDUCK_PY" -m pip install --upgrade "pydantic-core>=2.46.2" --quiet || true

            # Launch via Python API — bypasses argparse so arbitrary prompts work
            exec "$DEVDUCK_PY" - <<'PYEOF'
            import time, sys, os
            PROMPT = r\"\"\"{prompt_safe}\"\"\"
            print("[devduck-service] starting devduck...", flush=True)
            import devduck
            try:
                result = devduck.ask(PROMPT)
                preview = str(result)[:400].replace("\\n", " ")
                print(f"[devduck-service] startup ask() returned: {{preview}}", flush=True)
            except Exception as e:
                print(f"[devduck-service] startup ask() failed: {{e}}", flush=True)
                sys.exit(1)
            print("[devduck-service] startup complete; entering idle loop", flush=True)
            while True:
                time.sleep(3600)
            PYEOF
        """)

    def systemd_unit_content(self) -> str:
        install_target = "multi-user.target" if self.system else "default.target"
        user_line = f"User={self.user}\nGroup={self.user}" if self.system else ""
        return textwrap.dedent(f"""\
            [Unit]
            Description={self.description}
            After=network-online.target
            Wants=network-online.target

            [Service]
            Type=simple
            {user_line}
            WorkingDirectory={self.work_dir}
            EnvironmentFile={self.env_file}
            ExecStart={self.wrapper_path}
            Restart=always
            RestartSec={self.restart_sec}
            StandardOutput=append:{self.log_path}
            StandardError=append:{self.log_path}
            MemoryMax={self.memory_max}
            KillMode=mixed
            TimeoutStopSec=20

            [Install]
            WantedBy={install_target}
            """)

    def launchd_plist_content(self) -> str:
        """macOS LaunchAgent/LaunchDaemon plist."""
        # Read the env file and fold into EnvironmentVariables? Simpler: let the
        # wrapper source it via `set -a`.
        return textwrap.dedent(f"""\
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
            <plist version="1.0">
            <dict>
                <key>Label</key>
                <string>{self.launchd_label}</string>
                <key>ProgramArguments</key>
                <array>
                    <string>/bin/bash</string>
                    <string>-c</string>
                    <string>set -a; . {self.env_file}; set +a; exec {self.wrapper_path}</string>
                </array>
                <key>RunAtLoad</key><true/>
                <key>KeepAlive</key><true/>
                <key>WorkingDirectory</key>
                <string>{self.work_dir}</string>
                <key>StandardOutPath</key>
                <string>{self.log_path}</string>
                <key>StandardErrorPath</key>
                <string>{self.log_path}</string>
                <key>ProcessType</key>
                <string>Interactive</string>
                <key>ThrottleInterval</key>
                <integer>{self.restart_sec}</integer>
            </dict>
            </plist>
            """)


# ───────────────────── local executor ─────────────────────────────

def _ensure_devduck_available_local() -> None:
    if not _which("devduck"):
        raise RuntimeError(
            "devduck binary not found in PATH. Install with: pipx install devduck"
        )


def _systemctl(plan: InstallPlan, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["systemctl"]
    if not plan.system:
        cmd.append("--user")
    cmd.extend(args)
    return _run(cmd, check=check)


def _launchctl(plan: InstallPlan, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["launchctl", *args]
    if plan.system:
        cmd = ["sudo", *cmd]
    return _run(cmd, check=check)


def _install_local(plan: InstallPlan, start: bool = True) -> Dict[str, str]:
    """Install service on the local machine."""
    if plan.system:
        print("🦆 Installing system-wide service (requires sudo)...")
    else:
        print(f"🦆 Installing user-level service as {plan.user}...")

    _ensure_devduck_available_local()

    env_p = Path(plan.env_file)
    wrapper_p = Path(plan.wrapper_path)
    unit_p = Path(plan.unit_path)
    log_p = Path(plan.log_path)

    def _write_sudo(path: Path, content: str, mode: int) -> None:
        """Write a file that may require sudo."""
        tmp = Path(f"/tmp/.devduck-{os.getpid()}-{path.name}")
        tmp.write_text(content)
        tmp.chmod(mode)
        _run(["sudo", "mkdir", "-p", str(path.parent)])
        _run(["sudo", "cp", str(tmp), str(path)])
        _run(["sudo", "chmod", f"{mode:o}", str(path)])
        tmp.unlink(missing_ok=True)

    needs_sudo = plan.system
    writer = (
        (lambda p, c, m: _write_sudo(p, c, m))
        if needs_sudo
        else (lambda p, c, m: _local_write(p, c, m))
    )

    # 1. env file (600 to protect secrets)
    writer(env_p, plan.env_file_content(), 0o600)
    print(f"  ✓ env:     {env_p}")

    # 2. wrapper (755)
    writer(wrapper_p, plan.wrapper_content(), 0o755)
    print(f"  ✓ wrapper: {wrapper_p}")

    # 3. log file (touch, 644)
    if not log_p.exists():
        if needs_sudo:
            _run(["sudo", "mkdir", "-p", str(log_p.parent)])
            _run(["sudo", "touch", str(log_p)])
            _run(["sudo", "chown", f"{plan.user}:{plan.user}", str(log_p)])
        else:
            log_p.parent.mkdir(parents=True, exist_ok=True)
            log_p.touch()
    print(f"  ✓ log:     {log_p}")

    # 4. unit/plist
    if plan.platform == "linux":
        writer(unit_p, plan.systemd_unit_content(), 0o644)
        print(f"  ✓ unit:    {unit_p}")
        _systemctl(plan, "daemon-reload")
        _systemctl(plan, "enable", plan.service_name + ".service")
        if start:
            _systemctl(plan, "restart", plan.service_name + ".service")
    else:
        # macOS
        writer(unit_p, plan.launchd_plist_content(), 0o644)
        print(f"  ✓ plist:   {unit_p}")
        # Unload first (idempotent)
        _launchctl(plan, "unload", str(unit_p), check=False)
        if start:
            _launchctl(plan, "load", "-w", str(unit_p))

    return {
        "service_name": plan.service_name,
        "env_file": str(env_p),
        "wrapper": str(wrapper_p),
        "unit": str(unit_p),
        "log": str(log_p),
    }


def _uninstall_local(plan: InstallPlan) -> None:
    needs_sudo = plan.system
    rm = (
        (lambda p: _run(["sudo", "rm", "-f", p], check=False))
        if needs_sudo
        else (lambda p: Path(p).unlink(missing_ok=True))
    )

    if plan.platform == "linux":
        _systemctl(plan, "stop", plan.service_name + ".service", check=False)
        _systemctl(plan, "disable", plan.service_name + ".service", check=False)
        rm(plan.unit_path)
        _systemctl(plan, "daemon-reload", check=False)
    else:
        _launchctl(plan, "unload", plan.unit_path, check=False)
        rm(plan.unit_path)

    rm(plan.env_file)
    rm(plan.wrapper_path)
    print("🦆 Uninstalled. (log file kept at {})".format(plan.log_path))


# ───────────────────── remote executor (SSH) ────────────────────────

def _install_remote(host: str, plan: InstallPlan, start: bool = True) -> Dict[str, str]:
    """Install service on a remote Linux host via SSH."""
    if plan.platform != "linux":
        raise RuntimeError("Remote install only supports Linux hosts")

    print(f"🦆 Installing on {host} (system={plan.system})...")

    # Sanity check: devduck present on remote (use login shell so PATH includes ~/.local/bin)
    r = subprocess.run(
        ["ssh", host, "bash -lc 'command -v devduck >/dev/null 2>&1 && echo OK || echo MISSING'"],
        capture_output=True, text=True, check=False,
    )
    if "OK" not in (r.stdout or ""):
        raise RuntimeError(
            f"devduck not installed on {host}. "
            f"Run: ssh {host} 'pipx install devduck' first."
        )

    sudo = "sudo " if plan.system else ""

    # Write env file
    _ssh_write(host, plan.env_file, plan.env_file_content(), mode="0600")
    if plan.system:
        _ssh_run(host, f"sudo mkdir -p {shlex.quote(str(Path(plan.env_file).parent))}")
        _ssh_run(host, f"sudo mv {shlex.quote(plan.env_file)} {shlex.quote(plan.env_file)}.tmp && "
                       f"sudo cp {shlex.quote(plan.env_file)}.tmp {shlex.quote(plan.env_file)} && "
                       f"sudo chmod 600 {shlex.quote(plan.env_file)} && "
                       f"sudo rm {shlex.quote(plan.env_file)}.tmp", check=False)
    print(f"  ✓ env:     {host}:{plan.env_file}")

    # Write wrapper
    _ssh_write(host, plan.wrapper_path, plan.wrapper_content(), mode="0755")
    print(f"  ✓ wrapper: {host}:{plan.wrapper_path}")

    # Log file
    _ssh_run(host, f"{sudo}mkdir -p {shlex.quote(str(Path(plan.log_path).parent))} && "
                   f"{sudo}touch {shlex.quote(plan.log_path)} && "
                   f"{sudo}chown {plan.user}:{plan.user} {shlex.quote(plan.log_path)} 2>/dev/null || true")
    print(f"  ✓ log:     {host}:{plan.log_path}")

    # Write unit
    _ssh_write(host, plan.unit_path, plan.systemd_unit_content(), mode="0644")
    print(f"  ✓ unit:    {host}:{plan.unit_path}")

    # systemctl
    sctl = "sudo systemctl" if plan.system else "systemctl --user"
    _ssh_run(host, f"{sctl} daemon-reload")
    _ssh_run(host, f"{sctl} enable {plan.service_name}.service")
    if start:
        _ssh_run(host, f"{sctl} restart {plan.service_name}.service")

    # Brief status
    _ssh_run(host, f"sleep 2 && {sctl} is-active {plan.service_name}.service || true", check=False)

    return {
        "host": host,
        "service_name": plan.service_name,
        "env_file": plan.env_file,
        "wrapper": plan.wrapper_path,
        "unit": plan.unit_path,
        "log": plan.log_path,
    }


# ───────────────────── public entry points ──────────────────────────

def _resolve_remote_home(host: str) -> str:
    """Query remote $HOME via SSH login shell."""
    r = subprocess.run(
        ["ssh", host, "bash -lc 'echo $HOME'"],
        capture_output=True, text=True, check=True,
    )
    return r.stdout.strip() or "/root"


def _resolve_remote_user(host: str) -> str:
    r = subprocess.run(
        ["ssh", host, "bash -lc 'echo $USER'"],
        capture_output=True, text=True, check=True,
    )
    return r.stdout.strip() or "root"


def _parse_env_list(env_list: Optional[Sequence[str]]) -> Dict[str, str]:
    """Parse --env KEY=VALUE repeated flags into a dict."""
    out: Dict[str, str] = {}
    for item in env_list or []:
        if "=" not in item:
            raise ValueError(f"--env must be KEY=VALUE, got: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = v
    return out


def cmd_install(args) -> int:
    env_vars = _parse_env_list(getattr(args, "env_vars", None))

    # Read system prompt from file if requested
    system_prompt = args.system_prompt
    if system_prompt and system_prompt.startswith("@"):
        system_prompt = Path(system_prompt[1:]).expanduser().read_text()

    startup_prompt = args.startup_prompt
    if startup_prompt and startup_prompt.startswith("@"):
        startup_prompt = Path(startup_prompt[1:]).expanduser().read_text()

    # Remote host sets home/user from SSH target
    remote_host = getattr(args, "ssh", None)
    user = getattr(args, "service_user", None)
    home = None
    if remote_host:
        # Ask the remote for $HOME and $USER (login shell to get .zshrc/.bashrc env)
        r = subprocess.run(
            ["ssh", remote_host, "bash -lc 'echo $USER:$HOME'"],
            capture_output=True, text=True, check=True,
        )
        rem_user, _, rem_home = r.stdout.strip().partition(":")
        user = user or rem_user
        home = rem_home

    plan = InstallPlan(
        name=args.name,
        user=user,
        home=home,
        system=args.system,
        model=args.model,
        model_provider=args.model_provider,
        tools=args.tools,
        system_prompt=system_prompt,
        startup_prompt=startup_prompt,
        mcp_servers=args.mcp_servers,
        env_vars=env_vars,
        work_dir=args.work_dir,
        restart_sec=args.restart_sec,
        memory_max=args.memory_max,
        platform_override="linux" if remote_host else None,
    )

    if remote_host:
        result = _install_remote(remote_host, plan, start=not args.no_start)
    else:
        result = _install_local(plan, start=not args.no_start)

    print("\n🦆 Installed successfully.")
    for k, v in result.items():
        print(f"  {k:14} {v}")
    print(f"\n  Manage with: devduck service status --name {args.name}"
          + (f" --ssh {remote_host}" if remote_host else ""))
    return 0


def cmd_uninstall(args) -> int:
    remote_host = getattr(args, "ssh", None)
    home = _resolve_remote_home(remote_host) if remote_host else None
    user = _resolve_remote_user(remote_host) if remote_host else None
    plan = InstallPlan(name=args.name, system=args.system,
                       home=home, user=user,
                       platform_override="linux" if remote_host else None)

    if remote_host:
        sctl = "sudo systemctl" if plan.system else "systemctl --user"
        _ssh_run(remote_host, f"""
            {sctl} stop {plan.service_name}.service 2>/dev/null || true
            {sctl} disable {plan.service_name}.service 2>/dev/null || true
            {'sudo ' if plan.system else ''}rm -f {shlex.quote(plan.unit_path)} {shlex.quote(plan.env_file)} {shlex.quote(plan.wrapper_path)}
            {sctl} daemon-reload 2>/dev/null || true
        """, check=False)
        print(f"🦆 Uninstalled from {remote_host}.")
    else:
        _uninstall_local(plan)
    return 0


def _sctl_action(args, action: str) -> int:
    remote_host = getattr(args, "ssh", None)
    home = _resolve_remote_home(remote_host) if remote_host else None
    user = _resolve_remote_user(remote_host) if remote_host else None
    plan = InstallPlan(name=args.name, system=args.system,
                       home=home, user=user,
                       platform_override="linux" if remote_host else None)
    svc = f"{plan.service_name}.service"

    if plan.platform == "linux":
        sctl_base = "sudo systemctl" if plan.system else "systemctl --user"
        cmd = f"{sctl_base} {action} {svc}"
    else:
        # macOS
        if action in ("start",):
            cmd = f"launchctl load -w {shlex.quote(plan.unit_path)}"
        elif action == "stop":
            cmd = f"launchctl unload {shlex.quote(plan.unit_path)}"
        elif action == "restart":
            cmd = (f"launchctl unload {shlex.quote(plan.unit_path)} 2>/dev/null; "
                   f"launchctl load -w {shlex.quote(plan.unit_path)}")
        elif action == "status":
            cmd = f"launchctl list | grep {plan.launchd_label} || echo 'not loaded'"
        else:
            cmd = f"echo '{action} not supported on macOS'"

    if remote_host:
        _ssh_run(remote_host, cmd, check=False)
    else:
        subprocess.run(cmd, shell=True, check=False)
    return 0


def cmd_start(args):   return _sctl_action(args, "start")
def cmd_stop(args):    return _sctl_action(args, "stop")
def cmd_restart(args): return _sctl_action(args, "restart")
def cmd_status(args):  return _sctl_action(args, "status")
def cmd_enable(args):  return _sctl_action(args, "enable")
def cmd_disable(args): return _sctl_action(args, "disable")


def cmd_logs(args) -> int:
    remote_host = getattr(args, "ssh", None)
    home = _resolve_remote_home(remote_host) if remote_host else None
    user = _resolve_remote_user(remote_host) if remote_host else None
    plan = InstallPlan(name=args.name, system=args.system,
                       home=home, user=user,
                       platform_override="linux" if remote_host else None)

    follow = "-f" if args.follow else ""
    n = args.lines
    cmd = f"tail -n {n} {follow} {shlex.quote(plan.log_path)}"

    if remote_host:
        subprocess.run(["ssh", remote_host, cmd], check=False)
    else:
        subprocess.run(cmd, shell=True, check=False)
    return 0


def cmd_edit(args) -> int:
    """Open the env file in $EDITOR."""
    remote_host = getattr(args, "ssh", None)
    home = _resolve_remote_home(remote_host) if remote_host else None
    user = _resolve_remote_user(remote_host) if remote_host else None
    plan = InstallPlan(name=args.name, system=args.system,
                       home=home, user=user,
                       platform_override="linux" if remote_host else None)
    editor = os.environ.get("EDITOR", "nano")
    if remote_host:
        # Use ssh -t for a TTY-capable editor
        subprocess.run(["ssh", "-t", remote_host, f"{editor} {shlex.quote(plan.env_file)}"], check=False)
    else:
        subprocess.run([editor, plan.env_file], check=False)
    return 0


def cmd_show(args) -> int:
    """Preview the generated files without installing."""
    env_vars = _parse_env_list(getattr(args, "env_vars", None))
    plan = InstallPlan(
        name=args.name,
        system=args.system,
        model=args.model,
        tools=args.tools,
        system_prompt=args.system_prompt,
        startup_prompt=args.startup_prompt,
        env_vars=env_vars,
        platform_override="linux" if getattr(args, "ssh", None) else None,
    )
    print("=" * 60)
    print(f"Service name: {plan.service_name}")
    print(f"Platform:     {plan.platform}")
    print(f"System-wide:  {plan.system}")
    print(f"Env file:     {plan.env_file}")
    print(f"Wrapper:      {plan.wrapper_path}")
    print(f"Unit:         {plan.unit_path}")
    print(f"Log:          {plan.log_path}")
    print("=" * 60)
    print("\n--- env file content ---")
    print(plan.env_file_content())
    print("--- wrapper content ---")
    print(plan.wrapper_content())
    print("--- unit content ---")
    if plan.platform == "linux":
        print(plan.systemd_unit_content())
    else:
        print(plan.launchd_plist_content())
    return 0


# ──────────────────── argparse registration ────────────────────────

def register_parser(subparsers) -> None:
    """Attach the `service` subcommand to a parent argparse subparsers object."""
    p = subparsers.add_parser(
        "service",
        help="Install/manage devduck as a persistent OS service (systemd/launchd)",
        description=(
            "Manage devduck as a persistent system service.\n"
            "Supports local install (Linux systemd, macOS launchd) and remote install over SSH."
        ),
    )
    p_sub = p.add_subparsers(dest="service_command", required=True)

    # ---- shared args ----
    def _common(sp):
        sp.add_argument("--name", "-n", default="devduck",
                        help="Service name (default: devduck)")
        sp.add_argument("--system", action="store_true",
                        help="Install system-wide (needs sudo); default is user-level")
        sp.add_argument("--ssh", metavar="USER@HOST",
                        help="Install/manage on a remote Linux host over SSH")

    # ---- install ----
    ip = p_sub.add_parser("install", help="Install the service")
    _common(ip)
    ip.add_argument("--model", help="STRANDS_MODEL_ID (e.g. global.anthropic.claude-opus-4-7)")
    ip.add_argument("--model-provider", help="MODEL_PROVIDER (e.g. bedrock, anthropic, openai)")
    ip.add_argument("--tools", help="DEVDUCK_TOOLS config (e.g. 'devduck.tools:telegram,scheduler;strands_tools:shell')")
    ip.add_argument("--system-prompt", help="System prompt text, or @/path/to/file")
    ip.add_argument("--startup-prompt", help="Initial ask() prompt on service boot, or @/path/to/file")
    ip.add_argument("--mcp-servers", help="MCP_SERVERS JSON")
    ip.add_argument("--env", "-e", action="append", dest="env_vars",
                    metavar="KEY=VALUE",
                    help="Inject env var (repeatable). e.g. -e TELEGRAM_BOT_TOKEN=xxx")
    ip.add_argument("--work-dir", help="WorkingDirectory (default: $HOME)")
    ip.add_argument("--memory-max", default="8G", help="MemoryMax (systemd)")
    ip.add_argument("--restart-sec", type=int, default=15, help="RestartSec (systemd)")
    ip.add_argument("--service-user", help="User to run as (system mode only)")
    ip.add_argument("--no-start", action="store_true", help="Install but don't start")

    # ---- uninstall ----
    up = p_sub.add_parser("uninstall", help="Uninstall the service")
    _common(up)

    # ---- lifecycle ----
    for action in ("start", "stop", "restart", "status", "enable", "disable"):
        sp = p_sub.add_parser(action, help=f"{action} the service")
        _common(sp)

    # ---- logs ----
    lp = p_sub.add_parser("logs", help="Tail the service log")
    _common(lp)
    lp.add_argument("--lines", type=int, default=80, help="Number of lines to show")
    lp.add_argument("-f", "--follow", action="store_true")

    # ---- edit ----
    ep = p_sub.add_parser("edit", help="Edit the env file in $EDITOR")
    _common(ep)

    # ---- show ----
    shp = p_sub.add_parser("show", help="Preview generated files without installing")
    _common(shp)
    shp.add_argument("--model"); shp.add_argument("--tools")
    shp.add_argument("--system-prompt"); shp.add_argument("--startup-prompt")
    shp.add_argument("--env", "-e", action="append", dest="env_vars",
                    metavar="KEY=VALUE", help="Env var (repeatable)")


def dispatch(args) -> int:
    """Dispatch a parsed service subcommand."""
    table = {
        "install":   cmd_install,
        "uninstall": cmd_uninstall,
        "start":     cmd_start,
        "stop":      cmd_stop,
        "restart":   cmd_restart,
        "status":    cmd_status,
        "enable":    cmd_enable,
        "disable":   cmd_disable,
        "logs":      cmd_logs,
        "edit":      cmd_edit,
        "show":      cmd_show,
    }
    fn = table.get(args.service_command)
    if not fn:
        print(f"unknown service command: {args.service_command}")
        return 2
    return fn(args)
