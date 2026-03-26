# macOS Native Tools

Deep macOS integration using native frameworks — Calendar, Mail, Safari, Keychain, Notes, Spotify, and Apple Silicon sensors. All processing on-device.

---

## use_mac

Unified macOS control via AppleScript and native APIs.

```python
# Calendar
use_mac(action="calendar.events", days=7)
use_mac(action="calendar.create", title="Standup", date="2026-03-27", time="09:00")

# Mail
use_mac(action="mail.send", to="team@company.com", subject="Deploy", body="Shipped v2.0")
use_mac(action="mail.read", count=5)

# Safari
use_mac(action="safari.read")       # Read current tab content
use_mac(action="safari.url")        # Get current URL
use_mac(action="safari.tabs")       # List all tabs

# System
use_mac(action="system.screenshot", path="/tmp/screen.png")
use_mac(action="system.dark_mode", enable=True)
use_mac(action="system.volume", level=50)
use_mac(action="system.notification", title="Done", message="Build complete")

# Keychain
use_mac(action="keychain.get", service="MyApp", account="me")
```

---

## Apple Notes

```python
apple_notes(action="list")
apple_notes(action="search", query="project ideas")
apple_notes(action="create", title="Meeting Notes", body="...")
apple_notes(action="read", note_id="...")
```

---

## Spotify

```python
use_spotify(action="now_playing")
use_spotify(action="play")
use_spotify(action="pause")
use_spotify(action="next")
use_spotify(action="previous")
use_spotify(action="search", query="lofi beats")
```

Requires Spotify API credentials:

```bash
export SPOTIFY_CLIENT_ID="your-client-id"
export SPOTIFY_CLIENT_SECRET="your-client-secret"
export SPOTIFY_REDIRECT_URI="http://127.0.0.1:8888/callback"
```

---

## Apple Vision (Neural Engine)

On-device image analysis — zero cloud calls.

```python
apple_vision(action="ocr", image_path="/tmp/screenshot.png")
apple_vision(action="ocr_screen")           # OCR current screen
apple_vision(action="barcode", image_path="qr.png")
apple_vision(action="faces", image_path="photo.jpg")
apple_vision(action="rectangles", image_path="doc.jpg")  # Document detection
apple_vision(action="saliency", image_path="photo.jpg")  # Attention regions
apple_vision(action="languages")             # Supported OCR languages
```

---

## Apple NLP (Neural Engine)

On-device natural language processing.

```python
apple_nlp(action="detect", text="Bonjour le monde")     # → French
apple_nlp(action="sentiment", text="This is amazing!")   # → positive
apple_nlp(action="entities", text="Apple is in Cupertino")  # → NER
apple_nlp(action="pos", text="The cat sat on the mat")  # → POS tagging
apple_nlp(action="embed", word="hello")                  # → 300-dim vector
apple_nlp(action="similar", word="happy")                # → nearest neighbors
apple_nlp(action="distance", word="king", word2="queen") # → semantic distance
```

---

## Hardware Sensors

### Temperature, Battery, Fans

```python
apple_sensors(action="status")        # Full system status
apple_sensors(action="temperature")   # All temp sensors
apple_sensors(action="battery")       # Detailed battery info
apple_sensors(action="keyboard")      # Keyboard backlight level
apple_sensors(action="set_keyboard", brightness=0.5)  # Set backlight

apple_smc(action="all")              # All temps + fans + power
apple_smc(action="temps")            # Temperature sensors
apple_smc(action="fans")             # Fan speeds
apple_smc(action="power")            # Power draw
```

### WiFi

```python
apple_wifi(action="status")           # Current connection details
apple_wifi(action="scan")             # Scan nearby networks
apple_wifi(action="signal")           # Signal quality analysis
apple_wifi(action="best_channel")     # Recommend least congested channel
apple_wifi(action="diagnostics")      # Full WiFi diagnostics
```

---

## Computer Control

```python
use_computer(action="screenshot")                    # Full screen
use_computer(action="screenshot", region=[0,0,800,600])  # Region
use_computer(action="click", x=500, y=300)           # Click
use_computer(action="type", text="Hello")             # Type text
use_computer(action="hotkey", keys=["cmd", "c"])     # Keyboard shortcut
use_computer(action="scroll", direction="down", clicks=5)
use_computer(action="switch_app", app_name="Terminal")
```
