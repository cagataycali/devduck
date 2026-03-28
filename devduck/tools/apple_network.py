"""🌐 Apple Network diagnostics via SystemConfiguration."""

from typing import Dict, Any
from strands import tool


@tool
def apple_network(
    action: str = "status",
    host: str = None,
) -> Dict[str, Any]:
    """🌐 Network diagnostics — interfaces, DNS, reachability, proxy.

    Args:
        action: Action to perform:
            - "status": Overall network status (interfaces, IP, DNS)
            - "interfaces": List all network interfaces
            - "dns": Current DNS configuration
            - "reachability": Check if a host is reachable
            - "proxy": Proxy settings
        host: Hostname for reachability check (default: apple.com)

    Returns:
        Dict with network data
    """
    try:
        import SystemConfiguration
    except ImportError:
        return {"status": "error", "content": [{"text": "Install: pip install pyobjc-framework-SystemConfiguration"}]}

    if action == "status":
        return _network_status()
    elif action == "interfaces":
        return _list_interfaces()
    elif action == "dns":
        return _get_dns()
    elif action == "reachability":
        return _check_reachability(host or "apple.com")
    elif action == "proxy":
        return _get_proxy()
    else:
        return {"status": "error", "content": [{"text": f"Unknown action: {action}. Use: status, interfaces, dns, reachability, proxy"}]}


def _network_status():
    """Get comprehensive network status."""
    try:
        import subprocess, json

        r = subprocess.run(
            ["system_profiler", "SPNetworkDataType", "-json"],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(r.stdout)

        lines = ["🌐 Network Status:\n"]

        # Get active interface via scutil
        r2 = subprocess.run(
            ["scutil", "--nwi"],
            capture_output=True, text=True, timeout=5
        )
        if r2.returncode == 0:
            for line in r2.stdout.split("\n"):
                line = line.strip()
                if line and not line.startswith("Network"):
                    lines.append(f"  {line}")

        # Get IP addresses
        r3 = subprocess.run(
            ["ifconfig"],
            capture_output=True, text=True, timeout=5
        )
        if r3.returncode == 0:
            current_iface = None
            lines.append("\n  Active Interfaces:")
            for line in r3.stdout.split("\n"):
                if not line.startswith("\t") and ":" in line:
                    current_iface = line.split(":")[0]
                elif "inet " in line and "127.0.0.1" not in line:
                    ip = line.strip().split()[1]
                    lines.append(f"    {current_iface}: {ip}")

        # DNS
        r4 = subprocess.run(
            ["scutil", "--dns"],
            capture_output=True, text=True, timeout=5
        )
        if r4.returncode == 0:
            lines.append("\n  DNS Servers:")
            for line in r4.stdout.split("\n"):
                if "nameserver" in line:
                    lines.append(f"    {line.strip()}")
                    break

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Network status error: {e}"}]}


def _list_interfaces():
    """List all network interfaces."""
    try:
        import subprocess

        r = subprocess.run(
            ["networksetup", "-listallhardwareports"],
            capture_output=True, text=True, timeout=10
        )

        lines = ["🌐 Network Interfaces:\n"]
        lines.append(r.stdout.strip())

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Interfaces error: {e}"}]}


def _get_dns():
    """Get DNS configuration."""
    try:
        import SystemConfiguration

        store = SystemConfiguration.SCDynamicStoreCreate(None, "devduck", None, None)
        dns_key = "State:/Network/Global/DNS"
        dns_dict = SystemConfiguration.SCDynamicStoreCopyValue(store, dns_key)

        if not dns_dict:
            return {"status": "success", "content": [{"text": "🌐 No DNS configuration found."}]}

        lines = ["🌐 DNS Configuration:\n"]

        servers = dns_dict.get("ServerAddresses", [])
        if servers:
            lines.append("  Servers:")
            for s in servers:
                lines.append(f"    • {s}")

        domain = dns_dict.get("DomainName", "")
        if domain:
            lines.append(f"  Domain: {domain}")

        search = dns_dict.get("SearchDomains", [])
        if search:
            lines.append(f"  Search Domains: {', '.join(search)}")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"DNS error: {e}"}]}


def _check_reachability(host):
    """Check host reachability via SystemConfiguration."""
    try:
        import SystemConfiguration

        target = SystemConfiguration.SCNetworkReachabilityCreateWithName(None, host.encode())
        flags = SystemConfiguration.SCNetworkReachabilityGetFlags(target, None)

        if isinstance(flags, tuple):
            ok, flags = flags
        else:
            ok = True

        reachable = bool(flags & SystemConfiguration.kSCNetworkFlagsReachable) if ok else False
        connection_required = bool(flags & SystemConfiguration.kSCNetworkFlagsConnectionRequired) if ok else True

        status = "✅ Reachable" if reachable and not connection_required else "❌ Not reachable"

        lines = [
            f"🌐 Reachability: {host}",
            f"  Status: {status}",
            f"  Flags: {flags if ok else 'unavailable'}",
        ]

        # Also do a quick ping
        import subprocess
        r = subprocess.run(
            ["ping", "-c", "3", "-t", "5", host],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            # Extract avg ping time
            for line in r.stdout.split("\n"):
                if "avg" in line:
                    lines.append(f"  Ping: {line.strip()}")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Reachability error: {e}"}]}


def _get_proxy():
    """Get proxy settings."""
    try:
        import SystemConfiguration

        proxies = SystemConfiguration.SCDynamicStoreCopyProxies(None)

        if not proxies:
            return {"status": "success", "content": [{"text": "🌐 No proxy configured."}]}

        lines = ["🌐 Proxy Settings:\n"]

        proxy_types = [
            ("HTTPEnable", "HTTPProxy", "HTTPPort", "HTTP"),
            ("HTTPSEnable", "HTTPSProxy", "HTTPSPort", "HTTPS"),
            ("SOCKSEnable", "SOCKSProxy", "SOCKSPort", "SOCKS"),
        ]

        for enable_key, host_key, port_key, label in proxy_types:
            if proxies.get(enable_key, 0):
                host = proxies.get(host_key, "?")
                port = proxies.get(port_key, "?")
                lines.append(f"  {label}: {host}:{port}")

        exceptions = proxies.get("ExceptionsList", [])
        if exceptions:
            lines.append(f"  Bypass: {', '.join(str(e) for e in exceptions)}")

        if len(lines) == 1:
            lines.append("  No proxy enabled.")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Proxy error: {e}"}]}
