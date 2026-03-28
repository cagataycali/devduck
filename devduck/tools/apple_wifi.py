"""🛜 Apple WiFi Intelligence - scan networks, signal mapping, device tracking."""

from strands import tool
from typing import Dict, Any


@tool
def apple_wifi(
    action: str = "status",
    ssid_filter: str = None,
    duration: int = 1,
) -> Dict[str, Any]:
    """🛜 WiFi intelligence via CoreWLAN - scan, signal mapping, diagnostics.

    Args:
        action: Action to perform:
            - "status": Current connection details (SSID, RSSI, noise, channel, speed)
            - "scan": Scan all nearby networks with signal strength
            - "signal": Signal quality analysis (SNR, channel congestion)
            - "neighbors": Find all networks on same/adjacent channels
            - "best_channel": Recommend least congested channel
            - "diagnostics": Full WiFi diagnostics report
        ssid_filter: Filter scan results by SSID substring
        duration: Number of scans to average (for signal action)

    Returns:
        Dict with WiFi data
    """
    try:
        from CoreWLAN import CWWiFiClient

        client = CWWiFiClient.sharedWiFiClient()
        iface = client.interface()

        if not iface:
            return {"status": "error", "content": [{"text": "No WiFi interface found"}]}

        if action == "status":
            ssid = iface.ssid()
            rssi = iface.rssiValue()
            noise = iface.noiseMeasurement()
            snr = rssi - noise if rssi and noise else 0
            ch = iface.wlanChannel()
            channel_num = ch.channelNumber() if ch else 0
            channel_band = ch.channelBand() if ch else 0
            channel_width = ch.channelWidth() if ch else 0

            # Quality rating
            if snr > 40:
                quality = "Excellent"
            elif snr > 25:
                quality = "Good"
            elif snr > 15:
                quality = "Fair"
            else:
                quality = "Poor"

            text = f"""📡 WiFi Status:
  SSID: {ssid or '[not connected]'}
  BSSID: {iface.bssid() or 'N/A'}
  RSSI: {rssi} dBm
  Noise: {noise} dBm
  SNR: {snr} dB ({quality})
  Channel: {channel_num} (band: {channel_band}, width: {channel_width})
  Tx Rate: {iface.transmitRate()} Mbps
  Security: {iface.security()}
  Interface: {iface.interfaceName()}
  MAC: {iface.hardwareAddress()}
  Country: {iface.countryCode() or 'N/A'}
  Power: {'ON' if iface.powerOn() else 'OFF'}"""

            return {"status": "success", "content": [{"text": text}]}

        elif action == "scan":
            networks, err = iface.scanForNetworksWithName_error_(None, None)
            if not networks:
                return {"status": "error", "content": [{"text": f"Scan failed: {err}"}]}

            sorted_nets = sorted(networks, key=lambda x: x.rssiValue(), reverse=True)

            if ssid_filter:
                sorted_nets = [n for n in sorted_nets if ssid_filter.lower() in (n.ssid() or "").lower()]

            lines = [f"📡 Found {len(sorted_nets)} networks:\n"]
            lines.append(f"{'SSID':32} {'RSSI':>6} {'Ch':>4} {'Band':>6} {'Security':>10}")
            lines.append("-" * 70)

            for n in sorted_nets[:30]:
                ssid_name = n.ssid() or "[hidden]"
                rssi = n.rssiValue()
                ch_num = n.wlanChannel().channelNumber() if n.wlanChannel() else 0

                # Signal bars
                if rssi > -50:
                    bars = "████"
                elif rssi > -60:
                    bars = "███░"
                elif rssi > -70:
                    bars = "██░░"
                elif rssi > -80:
                    bars = "█░░░"
                else:
                    bars = "░░░░"

                band = "5GHz" if ch_num > 14 else "2.4G"
                if ch_num > 100:
                    band = "5GHz+"

                lines.append(f"{ssid_name:32} {rssi:>4}dB {ch_num:>4} {band:>6} {bars}")

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        elif action == "signal":
            import time

            readings = []
            for _ in range(max(1, duration)):
                rssi = iface.rssiValue()
                noise = iface.noiseMeasurement()
                readings.append({"rssi": rssi, "noise": noise, "snr": rssi - noise})
                if duration > 1:
                    time.sleep(1)

            avg_rssi = sum(r["rssi"] for r in readings) / len(readings)
            avg_noise = sum(r["noise"] for r in readings) / len(readings)
            avg_snr = sum(r["snr"] for r in readings) / len(readings)
            min_rssi = min(r["rssi"] for r in readings)
            max_rssi = max(r["rssi"] for r in readings)

            text = f"""📊 Signal Analysis ({len(readings)} samples):
  Avg RSSI: {avg_rssi:.1f} dBm
  Min/Max: {min_rssi}/{max_rssi} dBm
  Jitter: {max_rssi - min_rssi} dB
  Avg Noise: {avg_noise:.1f} dBm
  Avg SNR: {avg_snr:.1f} dB
  Tx Rate: {iface.transmitRate()} Mbps"""

            return {"status": "success", "content": [{"text": text}]}

        elif action in ("neighbors", "best_channel"):
            networks, err = iface.scanForNetworksWithName_error_(None, None)
            if not networks:
                return {"status": "error", "content": [{"text": f"Scan failed: {err}"}]}

            # Count networks per channel
            channel_usage = {}
            for n in networks:
                ch = n.wlanChannel()
                if ch:
                    ch_num = ch.channelNumber()
                    if ch_num not in channel_usage:
                        channel_usage[ch_num] = []
                    channel_usage[ch_num].append({
                        "ssid": n.ssid() or "[hidden]",
                        "rssi": n.rssiValue(),
                    })

            if action == "best_channel":
                # Find least congested 2.4GHz and 5GHz channels
                channels_24 = {1: 0, 6: 0, 11: 0}
                channels_5 = {}
                for ch, nets in channel_usage.items():
                    if ch <= 14:
                        # Map to nearest non-overlapping channel
                        for base in [1, 6, 11]:
                            if abs(ch - base) <= 2:
                                channels_24[base] = channels_24.get(base, 0) + len(nets)
                    else:
                        channels_5[ch] = len(nets)

                best_24 = min(channels_24, key=channels_24.get)
                best_5 = min(channels_5, key=channels_5.get) if channels_5 else None

                text = f"""🎯 Best Channel Recommendation:
  2.4GHz: Channel {best_24} ({channels_24[best_24]} competing networks)
  5GHz: Channel {best_5 or 'N/A'} ({channels_5.get(best_5, 0) if best_5 else 0} competing)

Channel Load (2.4GHz):
  Ch 1: {'█' * channels_24.get(1, 0)} ({channels_24.get(1, 0)})
  Ch 6: {'█' * channels_24.get(6, 0)} ({channels_24.get(6, 0)})
  Ch 11: {'█' * channels_24.get(11, 0)} ({channels_24.get(11, 0)})

5GHz Channels: {dict(sorted(channels_5.items()))}"""
            else:
                my_ch = iface.wlanChannel().channelNumber() if iface.wlanChannel() else 0
                neighbors = channel_usage.get(my_ch, [])
                text = f"📡 Channel {my_ch} Neighbors ({len(neighbors)}):\n"
                for n in sorted(neighbors, key=lambda x: x["rssi"], reverse=True):
                    text += f"  {n['ssid']:32} {n['rssi']} dBm\n"

            return {"status": "success", "content": [{"text": text}]}

        elif action == "diagnostics":
            rssi = iface.rssiValue()
            noise = iface.noiseMeasurement()
            snr = rssi - noise

            networks, _ = iface.scanForNetworksWithName_error_(None, None)
            net_count = len(networks) if networks else 0

            ch = iface.wlanChannel()
            my_channel = ch.channelNumber() if ch else 0

            # Count same-channel networks
            same_ch = 0
            if networks:
                for n in networks:
                    nch = n.wlanChannel()
                    if nch and nch.channelNumber() == my_channel:
                        same_ch += 1

            issues = []
            if snr < 15:
                issues.append("⚠️  Low SNR - high interference")
            if rssi < -75:
                issues.append("⚠️  Weak signal - move closer to AP")
            if same_ch > 5:
                issues.append(f"⚠️  Channel congested ({same_ch} networks on ch {my_channel})")
            if iface.transmitRate() < 100:
                issues.append("⚠️  Low tx rate - possible legacy mode")

            text = f"""🔬 WiFi Diagnostics:
  Connection: {iface.ssid() or 'disconnected'}
  RSSI: {rssi} dBm | Noise: {noise} dBm | SNR: {snr} dB
  Channel: {my_channel} | Same-channel APs: {same_ch}
  Tx Rate: {iface.transmitRate()} Mbps
  Nearby Networks: {net_count}

{'Issues Found:' if issues else '✅ No issues detected'}
{'chr(10)'.join(issues) if issues else ''}"""

            return {"status": "success", "content": [{"text": text}]}

        else:
            return {"status": "error", "content": [{"text": f"Unknown action: {action}. Use: status, scan, signal, neighbors, best_channel, diagnostics"}]}

    except ImportError:
        return {"status": "error", "content": [{"text": "Install: pip install pyobjc-framework-CoreWLAN"}]}
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {e}"}]}
