# Messaging

Three messaging channels — Telegram, Slack, WhatsApp. Each incoming message spawns a fresh DevDuck with full tool access.

---

## Overview

```mermaid
graph LR
    TG["📱 Telegram"] --> Listener["Message Listener"]
    SL["💬 Slack"] --> Listener
    WA["📞 WhatsApp"] --> Listener

    Listener --> Spawn["Spawn Fresh DevDuck"]
    Spawn --> Tools["🔧 Full Tool Access"]
    Spawn --> Response["📨 Reply to User"]

    style TG fill:#0088cc,stroke:#333,color:#fff
    style SL fill:#4a154b,stroke:#333,color:#fff
    style WA fill:#25d366,stroke:#333,color:#fff
    style Listener fill:#f5a623,stroke:#333,color:#000
```

---

## Telegram

### Setup

1. Create a bot via [@BotFather](https://t.me/BotFather)
2. Set the token:

```bash
export TELEGRAM_BOT_TOKEN="your-bot-token"
```

### Start Listener

```python
telegram(action="start_listener")
```

Or enable auto-reply:

```bash
export STRANDS_TELEGRAM_AUTO_REPLY=true
```

### Features

| Action | Description |
|--------|-------------|
| `start_listener` | Start background message polling |
| `stop_listener` | Stop the listener |
| `send_message` | Send text message |
| `send_photo` | Send photo |
| `send_document` | Send document |
| `send_poll` | Send poll |
| `send_location` | Send location |
| `edit_message` | Edit sent message |
| `delete_message` | Delete a message |
| `forward_message` | Forward a message |

### Security

Restrict access to specific users:

```bash
export TELEGRAM_ALLOWED_USERS="149632499,cagataycali"
```

---

## Slack

### Setup

1. Create a Slack App at [api.slack.com](https://api.slack.com/apps)
2. Enable Socket Mode and Events API
3. Subscribe to `message.im` and `app_mention` events
4. Set tokens:

```bash
export SLACK_BOT_TOKEN="xoxb-your-bot-token"
export SLACK_APP_TOKEN="xapp-your-app-token"
```

### Start Listener

```python
slack(action="start_listener")
```

### Features

- Listens for direct messages and @mentions
- Responds in-thread
- Full tool access per message
- Each message spawns a fresh DevDuck instance

---

## WhatsApp

Uses [wacli](https://github.com/nickhub-dev/wacli) for local WhatsApp access — **no Cloud API needed**.

### Setup

```bash
# Install wacli (Go required)
go install github.com/nickhub-dev/wacli@latest
```

### Start Listener

```python
whatsapp(action="start_listener")
```

First run will show a QR code — scan with your WhatsApp mobile app to pair.

### Features

- No API tokens needed
- Local pairing via QR code
- Each message spawns a fresh DevDuck
- Full tool access

---

## Architecture

All three channels follow the same pattern:

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant Channel as 📱 Channel
    participant Listener as 🔄 Listener
    participant Duck as 🦆 Fresh DevDuck

    User->>Channel: "deploy to prod"
    Channel->>Listener: Message event
    Listener->>Duck: Spawn new instance
    Duck->>Duck: Execute with full tools
    Duck-->>Channel: Send response
    Channel-->>User: Reply
```

Each message gets a **fresh, isolated DevDuck** — no cross-user state leakage, full tool access, automatic cleanup.
