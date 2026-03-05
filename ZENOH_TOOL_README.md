# strands-zenoh: Zenoh Protocol Tool for DevDuck

Zero-overhead pub/sub/query tool for DevDuck agents, enabling real-time IoT/robotics communication.

## Architecture

Based on `tcp.py` and `websocket.py` patterns:

1. **Background Threads** - Zenoh sessions run non-blocking
2. **DevDuck per Callback** - Each subscriber/queryable gets isolated DevDuck instance
3. **Global Registry** - `ZENOH_SESSIONS` tracks all sessions
4. **Real-time Streaming** - Callbacks process data as it arrives

## Features

### Session Management
- **Peer Mode**: Auto-discovery on local network
- **Router Mode**: Central routing infrastructure
- **Multiple Sessions**: Separate session_id for isolation

### Pub/Sub
- **Publishers**: Send data to key expressions
- **Subscribers**: DevDuck processes incoming messages
- **Wildcards**: `demo/**` (any depth), `demo/*/temp` (single level)

### Query/Queryable
- **Query**: Request data with selectors + timeout
- **Queryable**: Register handlers (DevDuck generates replies)
- **Aggregation**: Collect responses from multiple queryables

## Installation

```bash
# Install zenoh-python
pip install eclipse-zenoh

# Copy tool to DevDuck
cp zenoh_tool.py ~/.local/pipx/venvs/devduck/lib/python3.13/site-packages/devduck/tools/

# Or add to DEVDUCK_TOOLS env var
export DEVDUCK_TOOLS="devduck.tools:zenoh_tool,system_prompt;strands_tools:shell"
```

## Usage Examples

### Example 1: Basic Pub/Sub

```python
from devduck import devduck

# Terminal 1: Start session and subscribe
devduck("start a zenoh session")
devduck("subscribe to zenoh key 'demo/hello'")

# Terminal 2: Publish messages
devduck("publish 'Hello World!' to zenoh key 'demo/hello'")
devduck("publish 'Another message' to zenoh key 'demo/hello'")

# Subscriber DevDuck processes each message
```

### Example 2: Wildcard Subscriptions

```python
# Subscribe to all sensors
devduck("subscribe to 'home/sensors/**' in zenoh")

# Publish to different sensors
devduck("publish 'temperature:25' to 'home/sensors/living_room/temp'")
devduck("publish 'humidity:60' to 'home/sensors/bedroom/humidity'")

# Single subscriber receives both
```

### Example 3: Query/Queryable

```python
# Terminal 1: Register queryable
devduck("create zenoh queryable on 'robot/status' that replies with system info")

# Terminal 2: Query
result = devduck("query zenoh selector 'robot/status'")
# DevDuck generates reply based on query
```

### Example 4: IoT Sensor Network

```python
# Router node
devduck("start zenoh session in router mode")
devduck("subscribe to 'sensors/**' with prompt 'Analyze sensor data and alert on anomalies'")

# Sensor 1
devduck("start zenoh peer session named 'sensor1'")
devduck("publish 'temp:45' to 'sensors/room1/temp'")

# Sensor 2  
devduck("start zenoh peer session named 'sensor2'")
devduck("publish 'temp:22' to 'sensors/room2/temp'")

# Router DevDuck analyzes all sensor data
```

### Example 5: Robotics Coordination

```python
# Control station
devduck("start zenoh session")
devduck("publish 'move_forward:10' to 'robot/commands'")

# Robot
devduck("subscribe to 'robot/commands' with prompt 'Execute robot commands safely'")

# Robot processes commands with full DevDuck capabilities
```

## Testing Plan

### Test 1: Session Lifecycle
```bash
# Start session
devduck("start zenoh session named 'test1'")
devduck("check zenoh status")
devduck("stop zenoh session 'test1'")
```

### Test 2: Pub/Sub
```bash
# Terminal 1
devduck("start zenoh session")
devduck("subscribe to 'test/topic'")

# Terminal 2
devduck("publish 'test message' to 'test/topic'")
# Verify subscriber receives message
```

### Test 3: Query/Queryable
```bash
# Terminal 1
devduck("start zenoh session")
devduck("create queryable on 'test/query' replying with 'Hello'")

# Terminal 2
devduck("query zenoh 'test/query'")
# Verify receives "Hello"
```

### Test 4: DevDuck Integration
```bash
# Subscriber with custom prompt
devduck("subscribe to 'sensors/temp' with prompt 'Convert Celsius to Fahrenheit'")
devduck("publish '25' to 'sensors/temp'")
# Verify DevDuck processes and converts
```

### Test 5: Wildcards
```bash
devduck("subscribe to 'demo/**'")
devduck("publish 'a' to 'demo/x'")
devduck("publish 'b' to 'demo/y/z'")
# Both received
```

## Packaging as strands-zenoh

### Directory Structure
```
strands-zenoh/
├── pyproject.toml
├── README.md
├── LICENSE
├── strands_zenoh/
│   ├── __init__.py
│   └── zenoh_tool.py
└── examples/
    ├── pub_sub_example.py
    ├── query_example.py
    └── iot_sensor_network.py
```

### pyproject.toml
```toml
[project]
name = "strands-zenoh"
version = "0.1.0"
description = "Zenoh protocol tool for Strands agents"
dependencies = [
    "strands-agents>=1.0.0",
    "eclipse-zenoh>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest", "devduck"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Installation
```bash
# From PyPI (after publishing)
pip install strands-zenoh

# From source
git clone https://github.com/yourusername/strands-zenoh
cd strands-zenoh
pip install -e .
```

### Usage After Installation
```python
# Auto-discovered by DevDuck if in DEVDUCK_TOOLS
from devduck import devduck
devduck("start zenoh session")

# Or manual import
from strands_zenoh import zenoh
result = zenoh(action="start_session", session_id="main")
```

## Comparison with TCP/WebSocket Tools

| Feature | TCP | WebSocket | Zenoh |
|---------|-----|-----------|-------|
| Discovery | Manual | Manual | **Auto** |
| Pub/Sub | ❌ | ❌ | **✅** |
| Query | ❌ | ❌ | **✅** |
| Overhead | Medium | Medium | **Zero** |
| Zero-Copy | ❌ | ❌ | **✅** (shared memory) |
| Wildcards | ❌ | ❌ | **✅** |
| Routing | ❌ | ❌ | **✅** |

## Next Steps

1. **Test the tool** with examples above
2. **Create package structure** with pyproject.toml
3. **Add examples/** directory with sample scripts
4. **Write tests** (pytest + zenoh sessions)
5. **Publish to PyPI** as `strands-zenoh`
6. **Add to devduck** default tools (optional)

## Advanced Features (Future)

- **Shared Memory**: Zero-copy for local communication
- **Storage**: Query historical data
- **Liveliness**: Track peer presence
- **REST API**: HTTP gateway for Zenoh
- **DDS Bridge**: Connect to ROS2/DDS systems
- **Time Series**: Built-in time-series support

## License

Same as DevDuck (MIT/Apache-2.0)

## Credits

- **Eclipse Zenoh**: https://zenoh.io
- **DevDuck**: https://github.com/cagataycali/devduck
- **Pattern inspiration**: tcp.py, websocket.py
