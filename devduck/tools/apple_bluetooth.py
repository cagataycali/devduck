"""🔵 Apple Bluetooth scanner & device info via CoreBluetooth/IOBluetooth."""

from typing import Dict, Any
from strands import tool


@tool
def apple_bluetooth(
    action: str = "status",
    scan_duration: int = 5,
) -> Dict[str, Any]:
    """🔵 Bluetooth device discovery and status via Apple frameworks.

    Args:
        action: Action to perform:
            - "status": Current Bluetooth state and paired devices
            - "scan": Scan for nearby BLE devices
            - "paired": List paired/connected devices
        scan_duration: How many seconds to scan (default: 5)

    Returns:
        Dict with Bluetooth data
    """
    try:
        import objc
        from Foundation import NSObject, NSRunLoop, NSDate, NSDefaultRunLoopMode
    except ImportError:
        return {"status": "error", "content": [{"text": "Install: pip install pyobjc-framework-CoreBluetooth"}]}

    if action == "status":
        return _bt_status()
    elif action == "scan":
        return _bt_scan(scan_duration)
    elif action == "paired":
        return _bt_paired()
    else:
        return {"status": "error", "content": [{"text": f"Unknown action: {action}. Use: status, scan, paired"}]}


def _bt_status():
    """Get Bluetooth power state and basic info."""
    try:
        import subprocess
        # Use system_profiler for reliable Bluetooth info
        r = subprocess.run(
            ["system_profiler", "SPBluetoothDataType", "-json"],
            capture_output=True, text=True, timeout=10
        )
        import json
        data = json.loads(r.stdout)
        bt = data.get("SPBluetoothDataType", [{}])[0]

        controller = bt.get("controller_properties", bt.get("local_device_title", {}))
        state = controller.get("controller_state", controller.get("controller_power", "Unknown"))
        address = controller.get("controller_address", "Unknown")
        chipset = controller.get("controller_chipset", "Unknown")
        fw = controller.get("controller_firmwareVersion", "Unknown")

        lines = [
            f"🔵 Bluetooth Status:",
            f"  State: {state}",
            f"  Address: {address}",
            f"  Chipset: {chipset}",
            f"  Firmware: {fw}",
        ]

        # Connected devices
        connected = bt.get("device_connected", bt.get("device_title", []))
        if connected:
            lines.append(f"\n📱 Connected Devices ({len(connected)}):")
            if isinstance(connected, list):
                for dev in connected:
                    if isinstance(dev, dict):
                        for name, info in dev.items():
                            addr = info.get("device_address", "?")
                            minor = info.get("device_minorType", "?")
                            lines.append(f"  • {name} ({minor}) — {addr}")
            elif isinstance(connected, dict):
                for name, info in connected.items():
                    addr = info.get("device_address", "?") if isinstance(info, dict) else "?"
                    lines.append(f"  • {name} — {addr}")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Bluetooth status error: {e}"}]}


def _bt_scan(duration):
    """Scan for nearby BLE devices using CoreBluetooth."""
    try:
        import subprocess
        import time

        # Use CoreBluetooth via a quick Python subprocess with run loop
        scan_script = f"""
import objc
from Foundation import NSObject, NSRunLoop, NSDate, NSDefaultRunLoopMode
import CoreBluetooth
import json, time

class BLEScanner(NSObject):
    def init(self):
        self = objc.super(BLEScanner, self).init()
        self.devices = {{}}
        self.ready = False
        return self

    def centralManagerDidUpdateState_(self, central):
        if central.state() == CoreBluetooth.CBManagerStatePoweredOn:
            self.ready = True
            central.scanForPeripheralsWithServices_options_(None, None)

    def centralManager_didDiscoverPeripheral_advertisementData_RSSI_(self, central, peripheral, ad_data, rssi):
        uuid = str(peripheral.identifier())
        name = peripheral.name() or ad_data.get('kCBAdvDataLocalName', 'Unknown')
        self.devices[uuid] = {{'name': str(name), 'rssi': int(rssi), 'uuid': uuid}}

scanner = BLEScanner.alloc().init()
central = CoreBluetooth.CBCentralManager.alloc().initWithDelegate_queue_(scanner, None)

end_time = time.time() + {duration}
while time.time() < end_time:
    NSRunLoop.currentRunLoop().runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.5))

if scanner.ready:
    central.stopScan()

print(json.dumps(list(scanner.devices.values())))
"""
        r = subprocess.run(
            ["python3", "-c", scan_script],
            capture_output=True, text=True, timeout=duration + 10
        )

        if r.returncode != 0:
            return {"status": "error", "content": [{"text": f"BLE scan error: {r.stderr[:300]}"}]}

        import json
        devices = json.loads(r.stdout.strip())
        devices.sort(key=lambda d: d["rssi"], reverse=True)

        lines = [f"📡 BLE Scan ({duration}s) — Found {len(devices)} devices:\n"]
        lines.append(f"{'Name':<35} {'RSSI':>6}  UUID")
        lines.append("-" * 80)
        for d in devices[:30]:
            name = d["name"][:34]
            rssi = d["rssi"]
            # Signal bars
            if rssi > -50: bars = "████"
            elif rssi > -65: bars = "███░"
            elif rssi > -75: bars = "██░░"
            elif rssi > -85: bars = "█░░░"
            else: bars = "░░░░"
            lines.append(f"  {name:<33} {rssi:>4} {bars}  {d['uuid'][:18]}…")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"BLE scan error: {e}"}]}


def _bt_paired():
    """List paired Bluetooth devices via IOBluetooth."""
    try:
        import subprocess
        r = subprocess.run(
            ["system_profiler", "SPBluetoothDataType", "-json"],
            capture_output=True, text=True, timeout=10
        )
        import json
        data = json.loads(r.stdout)
        bt = data.get("SPBluetoothDataType", [{}])[0]

        lines = ["🔵 Paired Bluetooth Devices:\n"]

        for section_key in ["device_connected", "device_not_connected", "device_title"]:
            devices = bt.get(section_key, [])
            if not devices:
                continue

            connected = "connected" in section_key
            label = "Connected" if connected else "Not Connected"
            lines.append(f"  {'🟢' if connected else '⚪'} {label}:")

            if isinstance(devices, list):
                for dev in devices:
                    if isinstance(dev, dict):
                        for name, info in dev.items():
                            addr = info.get("device_address", "?") if isinstance(info, dict) else "?"
                            minor = info.get("device_minorType", "") if isinstance(info, dict) else ""
                            lines.append(f"    • {name} ({minor}) — {addr}")
            elif isinstance(devices, dict):
                for name, info in devices.items():
                    addr = info.get("device_address", "?") if isinstance(info, dict) else "?"
                    lines.append(f"    • {name} — {addr}")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Paired devices error: {e}"}]}
