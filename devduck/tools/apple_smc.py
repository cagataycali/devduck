"""🌡️ Apple SMC - System Management Controller thermal, power data via accessible APIs."""

from strands import tool
from typing import Dict, Any
import re
import subprocess


@tool
def apple_smc(
    action: str = "all",
    key: str = None,
) -> Dict[str, Any]:
    """🌡️ Apple SMC (System Management Controller) — read thermal, fan, and power data.

    Direct access to hardware sensors via SMC for real-time monitoring.

    Args:
        action: Action to perform:
            - "all": Read all temperatures, fans, and power
            - "temps": Read all temperature sensors
            - "fans": Read fan speeds
            - "power": Read power draw
            - "read": Read a specific SMC key (requires key param)
            - "keys": List all known sensor keys

    Returns:
        Dict with sensor data
    """
    try:
        if action == "all":
            sections = []

            # Temperatures
            temps = _read_temperatures()
            sections.append(f"🌡️ Temperatures:\n{temps}")

            # Fans
            fans = _read_fans()
            sections.append(f"🌀 Fans:\n{fans}")

            # Power
            power = _read_power()
            sections.append(f"⚡ Power:\n{power}")

            # Thermal state
            therm = _read_thermal_state()
            sections.append(f"🔥 Thermal State:\n{therm}")

            return {"status": "success", "content": [{"text": "\n\n".join(sections)}]}

        elif action == "temps":
            temps = _read_temperatures(detailed=True)
            return {"status": "success", "content": [{"text": f"🌡️ All Temperatures:\n{temps}"}]}

        elif action == "fans":
            fans = _read_fans()
            return {"status": "success", "content": [{"text": f"🌀 Fan Status:\n{fans}"}]}

        elif action == "power":
            power = _read_power(detailed=True)
            return {"status": "success", "content": [{"text": f"⚡ Power:\n{power}"}]}

        elif action == "read":
            if not key:
                return {"status": "error", "content": [{"text": "key parameter required (e.g., 'TC0P')"}]}
            result = _try_read_smc_key(key)
            return {"status": "success", "content": [{"text": f"SMC Key '{key}': {result}"}]}

        elif action == "keys":
            keys = _list_known_keys()
            return {"status": "success", "content": [{"text": f"📋 Known SMC Keys:\n{keys}"}]}

        else:
            return {"status": "error", "content": [{"text": f"Unknown action: {action}. Use: all, temps, fans, power, read, keys"}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}


def _read_temperatures(detailed=False):
    """Read temperature data from available sources."""
    lines = []

    # Battery temperature
    try:
        result = subprocess.run(
            ["ioreg", "-l", "-n", "AppleSmartBattery"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if '"Temperature"' in line:
                match = re.search(r'"Temperature"\s*=\s*(\d+)', line)
                if match:
                    temp = int(match.group(1)) / 100.0
                    lines.append(f"  Battery: {temp:.1f}°C")
    except:
        pass

    # NVMe/SSD temperature
    try:
        result = subprocess.run(
            ["smartctl", "-a", "/dev/disk0"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "Temperature" in line:
                match = re.search(r'(\d+)\s*Celsius', line)
                if match:
                    lines.append(f"  SSD: {match.group(1)}°C")
    except:
        pass

    # Try powermetrics (requires root, will fail gracefully)
    if detailed:
        try:
            result = subprocess.run(
                ["sudo", "-n", "powermetrics", "-n", "1", "-i", "100", "--samplers", "smc"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "die temperature" in line.lower() or "cpu" in line.lower():
                        lines.append(f"  {line.strip()}")
        except:
            pass

    # CPU thermal level
    try:
        result = subprocess.run(
            ["sysctl", "machdep.xcpm.cpu_thermal_level"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0 and result.stdout.strip():
            lines.append(f"  CPU Thermal Level: {result.stdout.strip().split(':')[-1].strip()}")
    except:
        pass

    if not lines:
        lines.append("  No temperature data accessible without root/entitlements")
        lines.append("  Run: sudo powermetrics -n 1 --samplers smc  for full data")

    return "\n".join(lines)


def _read_fans():
    """Read fan data. M3 Air has NO fan."""
    try:
        result = subprocess.run(
            ["ioreg", "-l"],
            capture_output=True, text=True, timeout=10
        )

        has_fan = any(x in result.stdout for x in ["FanType", "ActualSpeed", "TargetSpeed", "fan-count"])

        if not has_fan:
            return "  No fans (fanless design - M3 Air/MacBook)"

        # Extract fan data
        lines = []
        for line in result.stdout.split("\n"):
            for key in ["ActualSpeed", "TargetSpeed", "FanType"]:
                if f'"{key}"' in line:
                    lines.append(f"  {line.strip()}")

        return "\n".join(lines) if lines else "  Fan data not readable"
    except:
        return "  Error reading fan data"


def _read_power(detailed=False):
    """Read power consumption data."""
    lines = []

    # Battery power draw
    try:
        result = subprocess.run(
            ["ioreg", "-l", "-n", "AppleSmartBattery"],
            capture_output=True, text=True, timeout=5
        )

        voltage = amperage = 0
        for line in result.stdout.split("\n"):
            if '"Voltage"' in line:
                match = re.search(r'"Voltage"\s*=\s*(\d+)', line)
                if match:
                    voltage = int(match.group(1))
            if '"InstantAmperage"' in line:
                match = re.search(r'"InstantAmperage"\s*=\s*(-?\d+)', line)
                if match:
                    amperage = int(match.group(1))

        if voltage and amperage:
            watts = abs(voltage * amperage / 1_000_000)
            direction = "consuming" if amperage < 0 else "charging"
            lines.append(f"  Battery: {watts:.1f}W ({direction})")
            lines.append(f"  Voltage: {voltage}mV | Current: {amperage}mA")
    except:
        pass

    # pmset info
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "InternalBattery" in line:
                lines.append(f"  {line.strip()}")
            elif "drawing from" in line.lower():
                lines.append(f"  Source: {line.strip()}")
    except:
        pass

    if detailed:
        # AC adapter info
        try:
            result = subprocess.run(
                ["ioreg", "-l", "-n", "AppleSmartBattery"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split("\n"):
                for key in ["AdapterDetails", "ChargingCurrent", "ChargingVoltage", "ExternalConnected"]:
                    if f'"{key}"' in line:
                        lines.append(f"  {line.strip()}")
        except:
            pass

    return "\n".join(lines) if lines else "  No power data available"


def _read_thermal_state():
    """Read thermal pressure state."""
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout.strip()
        if "No thermal" in output:
            return "  ✅ Nominal - no thermal warnings"
        return f"  {output}"
    except:
        return "  Unknown"


def _try_read_smc_key(key):
    """Attempt to read a specific SMC key (usually requires root on Apple Silicon)."""
    return f"Direct SMC key reading requires root/entitlements on Apple Silicon. Try: sudo powermetrics -n 1 --samplers smc | grep {key}"


def _list_known_keys():
    """List well-known SMC sensor keys."""
    keys = {
        "TC0P": "CPU Proximity Temperature",
        "TC0D": "CPU Die Temperature",
        "TC0E": "CPU Efficiency Core Temp",
        "TC0F": "CPU Performance Core Temp",
        "TG0P": "GPU Proximity Temperature",
        "TG0D": "GPU Die Temperature",
        "Tm0P": "Memory Proximity Temperature",
        "Ts0P": "SSD Temperature",
        "TB0T": "Battery Temperature",
        "TW0P": "WiFi Module Temperature",
        "TPCD": "Thunderbolt Temperature",
        "TA0P": "Ambient Temperature",
        "TL0P": "LCD Temperature",
        "Tp01": "Power Supply Temperature",
        "F0Ac": "Fan 0 Actual Speed (RPM)",
        "F0Tg": "Fan 0 Target Speed (RPM)",
        "F0Mn": "Fan 0 Minimum Speed",
        "F0Mx": "Fan 0 Maximum Speed",
        "PSTR": "System Total Power (W)",
        "PCPC": "CPU Package Power (W)",
        "PCPG": "GPU Package Power (W)",
        "PDTR": "DC In Total Power (W)",
    }

    lines = []
    for key, desc in sorted(keys.items()):
        lines.append(f"  {key:6} - {desc}")

    lines.append("\n  ⚠️  On Apple Silicon, reading these keys requires root/entitlements")
    lines.append("  Use: sudo powermetrics -n 1 --samplers smc")
    return "\n".join(lines)
