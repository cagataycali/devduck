"""🍎 Apple Silicon Sensor Suite - battery, temp, keyboard backlight, SPU devices.

Direct hardware access via IOKit/CoreFoundation/CoreBrightness.
Works on M1/M2/M3/M4 Macs. No root for battery/keyboard/temp basics.
"""

import ctypes
import ctypes.util
import struct
import time
import platform
import re
import subprocess
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from strands import tool

# ============================================================================
# Framework bindings (lazy load)
# ============================================================================

_iokit = None
_cf = None
_objc = None

kCFStringEncodingUTF8 = 0x08000100
kCFNumberSInt64Type = 4
kCFNumberFloat64Type = 6
kIOMainPortDefault = 0


def _load_frameworks():
    global _iokit, _cf, _objc
    if _iokit is not None:
        return

    _iokit = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/IOKit.framework/IOKit")
    _cf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    _objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")

    # IOKit signatures
    _iokit.IOServiceMatching.restype = ctypes.c_void_p
    _iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
    _iokit.IOServiceGetMatchingServices.restype = ctypes.c_int
    _iokit.IOServiceGetMatchingServices.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    _iokit.IOIteratorNext.restype = ctypes.c_uint
    _iokit.IOIteratorNext.argtypes = [ctypes.c_uint]
    _iokit.IOObjectRelease.restype = ctypes.c_int
    _iokit.IOObjectRelease.argtypes = [ctypes.c_uint]
    _iokit.IORegistryEntryCreateCFProperty.restype = ctypes.c_void_p
    _iokit.IORegistryEntryCreateCFProperty.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    _iokit.IORegistryEntryGetName.restype = ctypes.c_int
    _iokit.IORegistryEntryGetName.argtypes = [ctypes.c_uint, ctypes.c_char_p]

    # CF signatures
    _cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    _cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
    _cf.CFNumberGetValue.restype = ctypes.c_bool
    _cf.CFNumberGetValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    _cf.CFRelease.restype = None
    _cf.CFRelease.argtypes = [ctypes.c_void_p]
    _cf.CFGetTypeID.restype = ctypes.c_ulong
    _cf.CFGetTypeID.argtypes = [ctypes.c_void_p]
    _cf.CFNumberGetTypeID.restype = ctypes.c_ulong
    _cf.CFStringGetTypeID.restype = ctypes.c_ulong
    _cf.CFBooleanGetTypeID.restype = ctypes.c_ulong
    _cf.CFBooleanGetValue.restype = ctypes.c_bool
    _cf.CFBooleanGetValue.argtypes = [ctypes.c_void_p]
    _cf.CFStringGetCStringPtr.restype = ctypes.c_char_p
    _cf.CFStringGetCStringPtr.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    _cf.CFStringGetCString.restype = ctypes.c_bool
    _cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint]


def _cf_string(s):
    return _cf.CFStringCreateWithCString(None, s.encode("utf-8"), kCFStringEncodingUTF8)


def _cf_to_python(ref):
    if ref is None or ref == 0:
        return None
    type_id = _cf.CFGetTypeID(ref)
    if type_id == _cf.CFNumberGetTypeID():
        val = ctypes.c_int64()
        if _cf.CFNumberGetValue(ref, kCFNumberSInt64Type, ctypes.byref(val)):
            v = val.value
            # Handle unsigned overflow (IOKit returns uint64 for negative values)
            if v > 2**63:
                v = v - 2**64
            return v
        fval = ctypes.c_double()
        if _cf.CFNumberGetValue(ref, kCFNumberFloat64Type, ctypes.byref(fval)):
            return fval.value
        return None
    elif type_id == _cf.CFStringGetTypeID():
        ptr = _cf.CFStringGetCStringPtr(ref, kCFStringEncodingUTF8)
        if ptr:
            return ptr.decode("utf-8")
        buf = ctypes.create_string_buffer(1024)
        if _cf.CFStringGetCString(ref, buf, 1024, kCFStringEncodingUTF8):
            return buf.value.decode("utf-8")
    elif type_id == _cf.CFBooleanGetTypeID():
        return bool(_cf.CFBooleanGetValue(ref))
    return None


def _get_property(service, key):
    cf_key = _cf_string(key)
    ref = _iokit.IORegistryEntryCreateCFProperty(service, cf_key, None, 0)
    if ref:
        val = _cf_to_python(ref)
        _cf.CFRelease(ref)
        return val
    return None


def _iter_services(class_name):
    matching = _iokit.IOServiceMatching(class_name.encode("utf-8"))
    iterator = ctypes.c_uint()
    kr = _iokit.IOServiceGetMatchingServices(kIOMainPortDefault, matching, ctypes.byref(iterator))
    if kr != 0:
        return
    while True:
        service = _iokit.IOIteratorNext(iterator.value)
        if service == 0:
            break
        yield service
        _iokit.IOObjectRelease(service)


def _service_name(service):
    name = ctypes.create_string_buffer(128)
    _iokit.IORegistryEntryGetName(service, name)
    return name.value.decode("utf-8", errors="ignore")


# ============================================================================
# Sensor reading functions
# ============================================================================

def _read_battery():
    """Read battery via IOKit AppleSmartBattery. Handles uint64 overflow."""
    _load_frameworks()
    for service in _iter_services("AppleSmartBattery"):
        current = _get_property(service, "CurrentCapacity") or 0
        max_cap = _get_property(service, "MaxCapacity") or 1
        design_cap = _get_property(service, "DesignCapacity") or 1
        is_charging = _get_property(service, "IsCharging") or False
        fully_charged = _get_property(service, "FullyCharged") or False
        external = _get_property(service, "ExternalConnected") or False
        temp_raw = _get_property(service, "Temperature") or 0
        cycle_count = _get_property(service, "CycleCount") or 0
        voltage = _get_property(service, "Voltage") or 0
        amperage = _get_property(service, "InstantAmperage") or 0
        time_empty = _get_property(service, "AvgTimeToEmpty") or 0
        time_full = _get_property(service, "AvgTimeToFull") or 0

        temp_c = temp_raw / 100.0
        health = round(max_cap / design_cap * 100, 1) if design_cap > 0 else 0
        power_w = round(abs(voltage * amperage) / 1_000_000, 2)
        pct = round(current / max_cap * 100, 1) if max_cap > 0 else 0

        state = "⚡ Charging" if is_charging else ("✅ Full" if fully_charged else "🔋 Discharging")
        if external and not is_charging and not fully_charged:
            state = "🔌 AC (not charging)"

        return {
            "percent": pct, "state": state, "health": health,
            "cycles": cycle_count, "temp_c": temp_c,
            "voltage_mv": voltage, "amperage_ma": amperage,
            "power_w": power_w,
            "current_mah": current, "max_mah": max_cap, "design_mah": design_cap,
            "time_empty_min": max(0, time_empty), "time_full_min": max(0, time_full),
            "external": external,
        }
    return None


def _read_thermal_sensors():
    """Read accessible temperature sensors."""
    _load_frameworks()
    sensors = []

    for cls_name, source in [
        ("AppleARMPMUTempSensor", "pmu"),
        ("AppleSCCTempSensor", "scc"),
        ("AppleEmbeddedNVMeTemperatureSensor", "nvme"),
    ]:
        for service in _iter_services(cls_name):
            name = _service_name(service)
            temp = _get_property(service, "Temperature")
            if temp is not None:
                temp_c = temp / 100.0 if temp > 1000 else temp
                sensors.append({"name": name, "temp_c": round(temp_c, 1), "source": source})

    return sensors


def _keyboard_brightness(set_val=None, fade_ms=200):
    """Get/set keyboard backlight via CoreBrightness private framework."""
    _load_frameworks()

    _objc.objc_getClass.restype = ctypes.c_void_p
    _objc.objc_getClass.argtypes = [ctypes.c_char_p]
    _objc.sel_registerName.restype = ctypes.c_void_p
    _objc.sel_registerName.argtypes = [ctypes.c_char_p]
    _objc.objc_msgSend.restype = ctypes.c_void_p
    _objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    def sel(n):
        return _objc.sel_registerName(n.encode("utf-8") if isinstance(n, str) else n)

    def cls(n):
        return _objc.objc_getClass(n.encode("utf-8") if isinstance(n, str) else n)

    # Load CoreBrightness
    ns_string = cls("NSString")
    _objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
    path = _objc.objc_msgSend(ns_string, sel("stringWithUTF8String:"),
                               b"/System/Library/PrivateFrameworks/CoreBrightness.framework")
    _objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    bundle = _objc.objc_msgSend(cls("NSBundle"), sel("bundleWithPath:"), path)
    if bundle:
        _objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _objc.objc_msgSend(bundle, sel("load"))

    kbc = cls("KeyboardBrightnessClient")
    if not kbc:
        return {"error": "KeyboardBrightnessClient unavailable"}

    _objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    instance = _objc.objc_msgSend(kbc, sel("alloc"))
    instance = _objc.objc_msgSend(instance, sel("init"))
    if not instance:
        return {"error": "Failed to init KeyboardBrightnessClient"}

    # Read current
    _objc.objc_msgSend.restype = ctypes.c_float
    _objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    current = float(_objc.objc_msgSend(instance, sel("brightnessForKeyboard:"), 1))
    result = {"brightness": round(current, 4)}

    # Set if requested
    if set_val is not None:
        set_val = max(0.0, min(1.0, set_val))
        _objc.objc_msgSend.restype = ctypes.c_void_p
        _objc.objc_msgSend.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_float, ctypes.c_int, ctypes.c_bool, ctypes.c_uint64
        ]
        _objc.objc_msgSend(instance, sel("setBrightness:fadeSpeed:commit:forKeyboard:"),
                           ctypes.c_float(set_val), ctypes.c_int(fade_ms), True, 1)
        result["set_to"] = set_val

    _objc.objc_msgSend.restype = ctypes.c_void_p
    _objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return result


def _spu_devices():
    """List SPU sensor devices."""
    _load_frameworks()
    usage_map = {
        (0xFF00, 3): "Accelerometer (BMI286)",
        (0xFF00, 9): "Gyroscope (BMI286)",
        (0xFF00, 4): "Ambient Light Sensor",
        (0x0020, 138): "Lid Angle Sensor",
    }
    devices = []
    for service in _iter_services("AppleSPUHIDDevice"):
        name = _service_name(service)
        page = _get_property(service, "PrimaryUsagePage") or 0
        usage = _get_property(service, "PrimaryUsage") or 0
        sensor_type = usage_map.get((page, usage), f"Unknown (0x{page:04X}:{usage})")
        devices.append({"name": name, "type": sensor_type, "root_required": page == 0xFF00})
    return devices


# ============================================================================
# Tool definition
# ============================================================================

@tool
def apple_sensors(
    action: str = "status",
    brightness: float = None,
    fade_ms: int = 200,
    duration: float = 1,
) -> Dict[str, Any]:
    """🍎 Apple Silicon sensor suite — read hardware sensors & control keyboard backlight.

    Access accelerometer, gyroscope, ambient light, lid angle, temperature sensors,
    battery data, and keyboard backlight on Apple Silicon Macs via IOKit.

    Args:
        action: Action to perform:
            - "status": Full system sensor status (temp, battery, keyboard, SPU devices)
            - "temperature": Read all temperature sensors
            - "battery": Read detailed battery info
            - "keyboard": Get keyboard backlight brightness
            - "set_keyboard": Set keyboard backlight (requires brightness param)
            - "devices": List all SPU sensor devices
            - "ambient_light": Read ambient light sensor (if accessible)
            - "lid": Read lid angle (if accessible)
            - "wake_spu": Wake up SPU drivers for sensor access (requires root)
        brightness: Keyboard brightness level 0.0-1.0 (for set_keyboard action)
        fade_ms: Keyboard brightness fade duration in ms (default: 200)
        duration: Duration in seconds for continuous readings (for future streaming)

    Returns:
        Dict with sensor data
    """
    if platform.system() != "Darwin":
        return {"status": "error", "content": [{"text": "macOS only"}]}

    try:
        if action == "status":
            parts = []

            # Battery
            bat = _read_battery()
            if bat:
                parts.append(f"🔋 Battery: {bat['percent']}% {bat['state']}")
                parts.append(f"  Health: {bat['health']}% | Cycles: {bat['cycles']} | Temp: {bat['temp_c']}°C")
                parts.append(f"  Power: {bat['power_w']}W ({bat['voltage_mv']}mV, {bat['amperage_ma']}mA)")
                parts.append(f"  Capacity: {bat['current_mah']}/{bat['max_mah']} mAh (design: {bat['design_mah']})")

            # Temperature
            temps = _read_thermal_sensors()
            if temps:
                parts.append(f"\n🌡️ Temperature ({len(temps)} sensors):")
                for t in temps[:10]:
                    parts.append(f"  {t['name']}: {t['temp_c']}°C ({t['source']})")

            # Keyboard
            try:
                kb = _keyboard_brightness()
                parts.append(f"\n⌨️ Keyboard: {kb.get('brightness', 'N/A')}")
            except:
                parts.append("\n⌨️ Keyboard: N/A")

            # SPU devices
            devices = _spu_devices()
            if devices:
                parts.append(f"\n📡 SPU Devices ({len(devices)}):")
                for d in devices:
                    tag = " 🔐" if d["root_required"] else " ✅"
                    parts.append(f"  {d['type']}{tag}")

            # Thermal pressure
            try:
                r = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=3)
                if "No thermal" in r.stdout:
                    parts.append("\n🔥 Thermal: ✅ Nominal")
                else:
                    parts.append(f"\n🔥 Thermal: {r.stdout.strip()}")
            except:
                pass

            # System
            try:
                cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2)
                mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2)
                gb = int(mem.stdout.strip()) / (1024**3) if mem.stdout.strip() else 0
                parts.append(f"\n💻 {cpu.stdout.strip()} | {gb:.0f}GB RAM")
            except:
                pass

            return {"status": "success", "content": [{"text": "\n".join(parts)}]}

        elif action == "battery":
            bat = _read_battery()
            if not bat:
                return {"status": "error", "content": [{"text": "No battery found"}]}
            lines = [
                f"🔋 Battery: {bat['percent']}% {bat['state']}",
                f"  Health: {bat['health']}% | Cycles: {bat['cycles']}",
                f"  Temperature: {bat['temp_c']}°C",
                f"  Power: {bat['power_w']}W ({bat['voltage_mv']}mV, {bat['amperage_ma']}mA)",
                f"  Capacity: {bat['current_mah']}/{bat['max_mah']} mAh (design: {bat['design_mah']})",
                f"  External: {'Yes' if bat['external'] else 'No'}",
            ]
            if bat['time_empty_min'] > 0:
                lines.append(f"  Time remaining: {bat['time_empty_min']} min")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "temperature":
            temps = _read_thermal_sensors()
            # Add battery temp as fallback
            bat = _read_battery()
            if bat:
                temps.append({"name": "Battery", "temp_c": bat["temp_c"], "source": "battery"})
            if not temps:
                return {"status": "success", "content": [{"text": "No temperature sensors accessible"}]}
            lines = [f"🌡️ Temperatures ({len(temps)}):\n"]
            for t in temps:
                lines.append(f"  {t['name']}: {t['temp_c']}°C ({t['source']})")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "keyboard":
            kb = _keyboard_brightness()
            if "error" in kb:
                return {"status": "error", "content": [{"text": kb["error"]}]}
            return {"status": "success", "content": [{"text": f"⌨️ Keyboard brightness: {kb['brightness']}"}]}

        elif action == "set_keyboard":
            if brightness is None:
                return {"status": "error", "content": [{"text": "brightness required (0.0-1.0)"}]}
            kb = _keyboard_brightness(set_val=brightness, fade_ms=fade_ms)
            if "error" in kb:
                return {"status": "error", "content": [{"text": kb["error"]}]}
            return {"status": "success", "content": [{"text": f"⌨️ Set keyboard to {brightness:.0%} (was {kb['brightness']:.0%})"}]}

        elif action == "devices":
            devices = _spu_devices()
            if not devices:
                return {"status": "success", "content": [{"text": "No SPU devices found"}]}
            lines = [f"📡 SPU Devices ({len(devices)}):"]
            for d in devices:
                tag = " 🔐 (root)" if d["root_required"] else " ✅"
                lines.append(f"  • {d['type']}{tag} ({d['name']})")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action in ("ambient_light", "lid"):
            return {"status": "success", "content": [{"text": f"🔐 {action} requires SPU wake (root). Use apple_sensors(action='wake_spu') with sudo."}]}

        elif action == "wake_spu":
            import os
            if os.geteuid() != 0:
                return {"status": "error", "content": [{"text": "🔐 Requires root. Run: sudo devduck"}]}
            # Wake SPU drivers
            _load_frameworks()
            for service in _iter_services("AppleSPUHIDDriver"):
                for key, val in [("SensorPropertyReportingState", 1), ("SensorPropertyPowerState", 1)]:
                    cf_key = _cf_string(key)
                    v = ctypes.c_int32(val)
                    cf_val = _cf.CFNumberCreate(None, 3, ctypes.byref(v))
                    _iokit.IORegistryEntrySetCFProperty(service, cf_key, cf_val)
            return {"status": "success", "content": [{"text": "✅ SPU drivers awakened"}]}

        else:
            return {"status": "error", "content": [{"text": f"Unknown: {action}. Use: status, battery, temperature, keyboard, set_keyboard, devices, ambient_light, lid, wake_spu"}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}
