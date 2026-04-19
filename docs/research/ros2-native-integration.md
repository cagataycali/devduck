# ROS2 Native Integration Research

**Branch:** `feat/dds-ros2-native-integration`
**Goal:** Make DevDuck natively interoperable with any ROS2 device/robot fleet
without requiring ROS2 runtime (`rclpy`, `ros2 cli`) on the DevDuck host.
**Target platform:** NVIDIA Jetson AGX Thor + existing ROS2 Humble/Jazzy fleets.

---

## 1. Why DDS (not just Zenoh)?

DevDuck already has two P2P transports:
- `zenoh_peer` — Zenoh multicast + unicast, DevDuck-internal framing
- `zcm_peer`   — ZCM UDP multicast, robotics-style flat channels

Neither speaks **ROS2** natively. ROS2 runs on top of **DDS (Data Distribution
Service)**: `CycloneDDS` (default since Humble), `FastDDS`, `RTI Connext`. If
DevDuck speaks DDS with the ROS2 wire format, it can:

- **Discover** every running ROS2 node on the same LAN with zero config
  (automatic via SPDP/SEDP multicast participant discovery)
- **Subscribe** to any topic (`/cmd_vel`, `/camera/image_raw`, `/tf`, `/scan`)
- **Publish** control commands (Twist, JointState, custom messages)
- **Call services** (via DDS request/reply pattern)
- Do all of this **without** installing ROS2 — just the `cyclonedds` Python
  package (already present on Thor).

This is the canonical "bridge without installing ROS" play, used by
`ros2cli-dds`, `zenoh-plugin-dds`, and `foxglove-bridge`. We build it into
DevDuck as a first-class tool.

---

## 2. Existing DevDuck primitives we compose with

| Primitive               | Role in DDS/ROS2 integration                                  |
|-------------------------|---------------------------------------------------------------|
| `zenoh_peer`            | P2P agent-to-agent mesh (unchanged)                           |
| `zcm_peer`              | Robotics-flavored pub/sub (unchanged)                         |
| **`dds_peer`** (new)    | DDS participant: raw DDS topic pub/sub + ROS2 framing         |
| **`use_ros`** (new)     | Agent-facing, opinionated ROS2 wrapper over `dds_peer`        |
| `event_bus`             | Emit `dds.peer.join`, `dds.topic.new`, `ros.message`          |
| `unified_mesh`          | Ring-context hints for other DevDuck instances                |
| `scheduler`             | Periodic topic polls, health checks                           |
| `use_aws`, `use_google` | Vision / ASR / LLM routing for camera + audio topics          |
| `use_agents`            | Route structured ROS data to specialized subagents            |
| `tasks`                 | Long-running ROS subscriber loops as named background tasks   |

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          DevDuck Agent                           │
│                                                                  │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────┐  │
│  │  use_ros   │  │   use_aws   │  │  use_google  │  │use_agts│  │
│  │  (tool)    │  │  (Rek/Bed)  │  │  (Vision)    │  │(router)│  │
│  └─────┬──────┘  └──────┬──────┘  └───────┬──────┘  └────┬───┘  │
│        │                │                 │              │      │
│        └────────┬───────┴─────────┬───────┘              │      │
│                 │                 │                      │      │
│           ┌─────▼─────┐      ┌────▼────────┐       ┌─────▼────┐ │
│           │ dds_peer  │      │ event_bus   │       │  tasks   │ │
│           │  (tool)   │──────│  (stream)   │───────│ (bg subs)│ │
│           └─────┬─────┘      └─────────────┘       └──────────┘ │
│                 │                                                │
│    ┌────────────┼───────────────────────────────┐                │
│    │            │                               │                │
│    ▼            ▼                               ▼                │
│ CycloneDDS  ROS2 type     SPDP/SEDP discovery (automatic)        │
│ participant registry      over UDP multicast 239.255.0.1:7400    │
└──────────────────────────────────────────────────────────────────┘
                 │
    ═════════════╪═════════════════════════════════════════
                 │    DDS wire (UDP multicast + unicast)
    ═════════════╪═════════════════════════════════════════
                 ▼
   ┌──────────────────────────────────────────────────┐
   │ ROS2 fleet on LAN (Humble / Jazzy / Rolling)     │
   │                                                  │
   │   /camera/image_raw   sensor_msgs/Image          │
   │   /cmd_vel            geometry_msgs/Twist        │
   │   /scan               sensor_msgs/LaserScan      │
   │   /tf, /tf_static     tf2_msgs/TFMessage         │
   │   /joint_states       sensor_msgs/JointState     │
   │   /odom               nav_msgs/Odometry          │
   │   …any custom topic…                             │
   └──────────────────────────────────────────────────┘
```

---

## 4. ROS2-on-DDS wire format (what we need to replicate)

A ROS2 topic `/cmd_vel` with type `geometry_msgs/msg/Twist` maps to DDS as:

| ROS2 concept         | DDS concept                                          |
|----------------------|------------------------------------------------------|
| Topic name           | `rt/cmd_vel` (prefix `rt/` for topics)               |
| Service req / resp   | `rq/<svc>Request`, `rr/<svc>Reply`                   |
| Action goal/result   | `rq/<ac>/_action/send_goalRequest`, etc.             |
| Message type         | Fully-qualified DDS type `geometry_msgs::msg::dds_::Twist_` |
| QoS                  | Reliable / BestEffort, Keep Last N, Durability       |
| Serialization        | CDR (Common Data Representation) big-/little-endian  |

For each ROS2 message type we want to handle, we need the **IDL** (or an
equivalent `IdlStruct` Python class). Options:

1. **Static library of stock ROS2 types** — ship IDL stubs for the top ~40 most
   common message types (`std_msgs`, `geometry_msgs`, `sensor_msgs`,
   `nav_msgs`, `tf2_msgs`, `trajectory_msgs`, `diagnostic_msgs`,
   `visualization_msgs`). Covers >95% of fleets. Generated once, checked into
   `devduck/tools/_ros_msgs/`.
2. **Dynamic introspection** — use DDS built-in topics
   (`DCPSPublication`, `DCPSSubscription`) to discover topic types at runtime,
   then fetch IDL via `rosidl_dynamic_typesupport` when ROS2 IS available,
   else fall back to opaque byte pass-through (still useful — we can relay
   raw CDR to another DevDuck that DOES have the type).
3. **Hybrid** — static lib for speed, dynamic fallback for unknown.

We ship **option 3**.

---

## 5. Discovery flow (SPDP / SEDP)

CycloneDDS does this for us automatically once a `DomainParticipant` is alive:

1. Participant sends multicast SPDP announcements on port 7400 (domain 0).
2. Every other DDS participant on the LAN (including ROS2 nodes) replies.
3. SEDP exchanges publisher/subscriber metadata per topic.
4. DevDuck reads `DCPSParticipant`, `DCPSPublication`, `DCPSSubscription`
   built-in topics → builds a live registry of:
   - Every ROS2 node on the network
   - Every topic + type + QoS
   - Every service + action

We expose this registry through:
```python
dds_peer(action="list_participants")
dds_peer(action="list_topics")
use_ros(action="list_topics")
use_ros(action="list_nodes")
```

---

## 6. Tool API sketch (agent-facing)

### `dds_peer` — low-level DDS transport
```python
dds_peer(action="start", domain_id=0)
dds_peer(action="list_topics")
dds_peer(action="subscribe", topic="rt/chatter", type="std_msgs::msg::dds_::String_")
dds_peer(action="publish",   topic="rt/chatter", type="...", data={"data": "hi"})
dds_peer(action="stop")
```

### `use_ros` — opinionated ROS2 wrapper
```python
use_ros(action="list_nodes")                    # every ROS2 node on LAN
use_ros(action="list_topics")                   # "/cmd_vel", "/scan", ...
use_ros(action="echo", topic="/scan", count=1)  # one-shot read
use_ros(action="tail", topic="/odom", seconds=5)# stream for N sec
use_ros(action="pub",  topic="/cmd_vel", msg={"linear":{"x":0.2}})
use_ros(action="call", service="/add_two_ints", req={"a":1,"b":2})
use_ros(action="bag_record", topics=["/scan","/tf"], duration=30)
```

Every received ROS2 message is ALSO pushed into `event_bus` as
`ros.<topic_name>` so the agent gets it in its dynamic context if it's a
"hot" topic.

---

## 7. Vision / perception pipeline — `use_aws`, `use_google`, `use_agents`

Raw sensor topics are huge and useless to an LLM. We pipe them through
specialist tools:

| Topic                         | Router                            | Downstream           |
|-------------------------------|-----------------------------------|----------------------|
| `/camera/image_raw`           | `use_aws(Rekognition, Bedrock)`   | labels, caption, VQA |
| `/camera/image_raw`           | `use_google(Vision API)`          | OCR, faces, products |
| `/audio/audio`                | `use_aws(Transcribe)`             | live transcript      |
| `/diagnostics`                | `use_agents(route=diagnostic)`    | sub-agent triage     |
| `/tf`, `/odom`, `/joint_states`| `use_agents(route=spatial)`      | pose → text summary  |
| `/scan` (LaserScan)           | local reducer → `use_agents`      | obstacle summary     |

`use_agents` picks the right subagent based on topic content and routes.

Critical design: **framerate gating**. Nobody wants a 30 Hz image stream in
the LLM context. Each vision/audio bridge has:
- `trigger`: `on_change`, `on_event`, `every(seconds)`, `manual`
- `summarize`: keep only the latest delta / caption
- Emits one `event_bus` entry at most every N seconds.

---

## 8. Thor testing plan

Thor box already:
- Has `cyclonedds` Python package ✅
- Is on the same LAN as our ROS2 fleet (Unitree G1, SO-100 arms, iRobot) ✅
- Has DevDuck installed ✅
- `enP2p1s0` is the LAN interface used for multicast ✅

Tests (per commit):
1. `dds_peer start` → verify participant appears in `ros2 topic list` from a
   peer machine.
2. Echo `/chatter` while an external `ros2 run demo_nodes_cpp talker` runs.
3. Publish `/cmd_vel` to a running rosbot, observe motion.
4. Subscribe `/camera/image_raw`, pipe to `use_aws(Rekognition)`, print labels.
5. Full loop: voice → `use_ros(pub, /cmd_vel)` → robot moves → camera →
   vision → natural language reply. **This is the demo.**

---

## 9. Commit roadmap

1. `docs(research): add ROS2 native integration research doc` ✅ **shipped**
2. `feat(dds_peer): scaffold CycloneDDS participant + lifecycle` ✅ **shipped**
3. `feat(dds_peer): SPDP/SEDP discovery loop + ROS2 topic introspection` ✅ **shipped**
4. `feat(_ros_msgs): bundle 28 common ROS2 message IDL stubs` ✅ **shipped**
5. `feat(use_ros): agent-facing ROS2 tool (list/echo/pub + JSON)` ✅ **shipped**
6. `feat(use_ros): tail bridge — stream ROS2 topics into event_bus` ✅ **shipped**
7. `feat(devduck): register dds_peer + use_ros in default tool config` ✅ **shipped** (this commit)
8. `feat(use_ros,use_aws): vision pipeline for Image topics` — next
9. `feat(use_ros,use_google): OCR/vision pipeline via Google` — next
10. `feat(use_ros,use_agents): spatial + diagnostics routing` — next
11. `test(thor): live integration test log & README with Thor run` — next

Each commit is small, testable, and reversible.

---

## 10. Non-goals (explicitly out of scope)

- Becoming a full `rclpy` replacement.
- Re-implementing the ROS2 parameter server, lifecycle, or executors.
- Supporting FastDDS or RTI Connext on the DevDuck side (we stick with
  CycloneDDS — the default since Humble).
- Rewriting `zenoh_peer` or `zcm_peer`. They're orthogonal transports for
  *agent-to-agent* chat, not *robot-to-agent* pub/sub.

---

## 11. Open questions / follow-ups

- **QoS mapping**: ROS2 SensorDataQoS (BestEffort, KeepLast 5) vs default
  reliable — we need a small QoS profile mapper.
- **Security**: DDS-Security is off by default on ROS2 fleets; if the target
  fleet enables it, we'll need to load certs. Deferred.
- **Large payloads**: Point clouds on DDS hit IP fragmentation. First
  implementation just streams preview / downsample; full handling deferred.
- **Zenoh-DDS bridge coexistence**: If a ROS2 fleet already runs
  `zenoh-plugin-dds`, we can ALSO reach it via `zenoh_peer` — future
  convergence.
