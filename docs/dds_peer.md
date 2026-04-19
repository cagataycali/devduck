# CycloneDDS peer — ROS2-native interop without rclpy

The `dds_peer` tool lets a DevDuck agent join a DDS domain as a
first-class participant. Because ROS2 uses DDS as its transport layer,
this means a single DevDuck instance can talk to ROS2 nodes **without
installing rclpy or any ROS2 Python bindings**.

## Why this works

ROS2's middleware abstraction (RMW) defaults to FastDDS, but it
ships with a CycloneDDS RMW too:

```
rmw_cyclonedds_cpp   ← what ROS2 uses when RMW_IMPLEMENTATION is set
```

When ROS2 is configured with `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`
and both sides share the same `ROS_DOMAIN_ID` / `DEVDUCK_DDS_DOMAIN`,
the wire protocol is identical. A DevDuck agent running the
`dds_peer` tool on that domain is indistinguishable (to ROS2) from
any other DDS participant — it can:

- Be discovered by `ros2 node list` (as a plain DDS participant)
- Show up in `ros2 topic list` if it creates matching topics
- Publish to topics ROS2 subscribes to
- Subscribe to any topic ROS2 publishes

## Setup

### On the ROS2 side (robot, simulator, etc.)

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
# Then run your ROS2 nodes as usual.
```

### On the DevDuck side

```bash
pip install cyclonedds
export DEVDUCK_ENABLE_DDS=true
export DEVDUCK_DDS_DOMAIN=0   # must match ROS_DOMAIN_ID
devduck
```

DevDuck will auto-start the peer and inject a live DDS network
summary into every agent turn's context, so the model knows what
robots/topics are currently on the network.

## Tool usage

```python
# List everything ROS2 is advertising
dds_peer(action="list_participants")
dds_peer(action="list_topics")

# Subscribe to a ROS2 string topic (std_msgs/String compatible)
dds_peer(action="subscribe", topic="/chatter", wait_time=2.0)

# Publish to it
dds_peer(action="publish", topic="/chatter", message="hello from DevDuck")

# DevDuck↔DevDuck direct addressing
dds_peer(action="send_to_peer", peer_id="dds-robot-1234", message="...")
```

## Limitations (phase 1)

- The built-in default type is `std_msgs/msg/String`-compatible only.
  Arbitrary ROS2 message types require registering an `IdlStruct`
  mirror of the `.msg` definition (future work, easy extension).
- Discovery metadata is intentionally minimal — key + type name —
  enough for an LLM to reason about the topology, but QoS policies
  are only surfaced as a `repr()` blob for now.

## Phase 2 roadmap

- Generate IDL structs for common ROS2 message packages
  (`geometry_msgs/Twist`, `sensor_msgs/Image`, `nav_msgs/Odometry`).
- Hook `dds_peer` into the event bus so incoming samples surface
  as ambient context the way Telegram/WhatsApp events already do.
- Optional QoS tuning per topic (reliability, history depth) for
  robotics-grade workloads.
