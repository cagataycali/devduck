# Privacy Policy for DevDuck Browser Bridge

**Last Updated: December 11, 2025**

## Overview
DevDuck Browser Bridge is a Chrome extension that enables local communication between your browser and the DevDuck AI agent running on your computer.

## Data Collection
**We do NOT collect, store, or transmit any user data.**

### What We Access
The extension requests various permissions to enable browser automation features:
- **Tabs**: Read and manage browser tabs
- **Active Tab**: Interact with the current tab
- **Scripting**: Execute automation scripts
- **Storage**: Store connection settings locally
- **Clipboard**: Read/write clipboard for automation
- **Geolocation**: Access location when requested by DevDuck
- **Notifications**: Display status notifications

### How Data is Used
All data accessed by the extension:
- Stays on your local machine
- Is only transmitted to your local DevDuck agent via WebSocket (localhost:9223)
- Never sent to external servers
- Never collected or stored by the extension developer

## Connection & Communication
- **Local Only**: WebSocket connection to `ws://localhost:9223`
- **No External Servers**: All communication is between browser and local agent
- **No Tracking**: No analytics, telemetry, or usage tracking
- **No Third Parties**: No data shared with any third party

## Permissions Justification
- **activeTab, tabs**: Required for tab management and navigation
- **scripting**: Enables page interaction and content extraction
- **storage**: Stores connection settings locally
- **clipboardRead, clipboardWrite**: Enables clipboard automation
- **geolocation**: Provides location data when requested
- **notifications**: Shows connection status
- **host_permissions (<all_urls>)**: Required for automation on any website

## Data Storage
- Connection settings stored locally using Chrome's storage API
- No cookies, no external storage, no cloud synchronization

## Security
- Open source code available on GitHub
- Local-only operation
- Requires explicit DevDuck agent to function
- No external dependencies loaded at runtime

## Updates
This privacy policy may be updated. Changes will be reflected in the extension listing.

## Contact
For questions or concerns: https://github.com/cagataycali/devduck

## Consent
By installing DevDuck Browser Bridge, you consent to this privacy policy.
