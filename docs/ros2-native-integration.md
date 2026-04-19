# DevDuck ↔ ROS2: Native Integration

> TL;DR: Install DevDuck on any host with `cyclonedds`, and it can
> discover, subscribe, and publish to **every ROS2 node on the LAN** —
> no ROS2 install, no `rclpy`, no code generation. Works in under 30
> lines of agent prompting.

## What this gives you

A DevDuck agent that, out-of-the-box, can:

- See every ROS2 node, topic, and type on the LAN (`use_ros list_nodes`,
  `list_topics`)
- Read one sample or stream live samples of any known ROS2 topic
  (`use_ros echo`, `use_ros tail`)
- Publish typed commands back to the fleet — `Twist`, `PoseStamped`,
  `JointState`, etc. (`use_ros pub`)
- Pipe high-bandwidth sensor topics into the agent context, rate-limited,
  via the DevDuck event bus (`use_ros tail` → `ros.<topic>` events)

Supported out of the box: 28 canonical ROS2 message types covering
`std_msgs`, `geometry_msgs`, `sensor_msgs`, `nav_msgs`, `tf2_msgs`,
`diagnostic_msgs`, `builtin_interfaces`. See
`devduck/tools/_ros_msgs.py` for the full list; unknown types return a
clear error (opaque-byte fallback is on the roadmap).

## Requirements

- Python 3.10+
- `cyclonedds>=0.10` (bundled with DevDuck's default install profile)
- Same LAN / same DDS domain as the ROS2 fleet (override with
  `ROS_DOMAIN_ID` env var)

No ROS2 install needed on the DevDuck host. If the fleet runs
DDS-Security, see Open Questions below.

## Quick start

From a DevDuck session:

```
use_ros(action="list_nodes")
use_ros(action="list_topics")
use_ros(action="echo", topic="/scan", timeout=3)
use_ros(action="pub",  topic="/cmd_vel",
        type="geometry_msgs/msg/Twist",
        msg={"linear": {"x": 0.2}, "angular": {"z": 0.1}})

# Stream a topic into the agent's dynamic context (2 Hz cap):
use_ros(action="tail", topic="/scan", max_hz=2.0)
use_ros(action="list_tails")
use_ros(action="untail", topic="/scan")
```

Low-level access if you need it:

```
dds_peer(action="start")
dds_peer(action="status")
dds_peer(action="list_participants")
dds_peer(action="list_publications")
```

## How it works

```
┌──────────────────────────────────────────────────────────┐
│                 DevDuck Agent                            │
│                                                          │
│   use_ros  ────▶  dds_peer  ────▶  CycloneDDS            │
│     │              │                                     │
│     └─ tails ──▶  event_bus  ──▶  Dynamic context inject │
│                                   (agent sees live data  │
│                                    on every turn)        │
└──────────────────────────────────────────────────────────┘
            ▼                DDS multicast (UDP)
            ═══════════════════════════════════════════════
            ▼
     ROS2 fleet (Humble / Jazzy / Rolling)
       /cmd_vel  /scan  /tf  /joint_states  /camera/image_raw …
```

- `dds_peer` owns a CycloneDDS `DomainParticipant` and runs a
  discovery loop reading `DCPSParticipant`, `DCPSPublication`,
  `DCPSSubscription` built-in topics. Every ROS2 node on the domain
  becomes a live entry in its registry.
- `use_ros` is the opinionated wrapper: it hides DDS typename mangling
  (`geometry_msgs::msg::dds_::Twist_`), maps ROS2 topic names
  (`/cmd_vel` ↔ `rt/cmd_vel`), and serialises messages to / from
  ordinary Python dicts.
- `_ros_msgs` ships hand-written `IdlStruct` stubs for the most common
  types, wire-compatible with what an `rclpy` node would publish.
- `tail` spins up a background subscriber per topic, rate-limits the
  emits, and pushes every sample onto the DevDuck event bus. The
  agent's context builder picks those up and injects recent ROS2
  activity into every turn.

## Verified live on Thor

Thor (NVIDIA AGX, aarch64, Ubuntu 24.04) with `cyclonedds` installed:

- Two-participant domain: DevDuck's participant + an external Twist
  talker. `list_topics` correctly shows
  `/cmd_vel [geometry_msgs::msg::dds_::Twist_] (topic, known)`.
- CDR round-trip confirmed for flat (`Twist`), nested-2-level
  (`TransformStamped`), sequence-of-struct (`TFMessage`), deeply
  nested (`Odometry`, 3 levels), and seq-of-seq (`DiagnosticArray`).
- `tail` against a 2 Hz talker emitted 5 `ros.cmd_vel` events on the
  event bus in 3 s, respecting the 2 Hz cap.
- All `dds_peer` events (`dds.start`, `dds.participant.join`,
  `dds.endpoint.new`) visible on the event bus.

## Roadmap

See `docs/research/ros2-native-integration.md` for the full design
document. Commits 1-7 are shipped; 8-11 (vision pipeline via
`use_aws` / `use_google`, routing via `use_agents`, and the live
fleet integration test README) are next.

## Non-goals

- This is not a full `rclpy` replacement. Parameters, lifecycle, and
  executor behaviours are deliberately out of scope.
- DDS-Security (certs, access control) is not configured — a fleet
  running with security enabled needs credentials injected before
  start. Deferred.
- Point clouds and compressed images work, but tuning for their
  bandwidth is future work.

## Troubleshooting

- `cyclonedds not installed` error from `dds_peer`: `pip install
  cyclonedds` and restart DevDuck. On Thor, the package is already
  available in the default Python; just ensure DevDuck's venv can
  see it, or install system-wide.
- Empty `list_topics` output: check your network interface supports
  multicast and the fleet is on the same `ROS_DOMAIN_ID` (env var).
  DDS discovery needs ~1 s to converge.
- "type X not in bundled registry": pass `type=` explicitly or add
  the IDL stub to `devduck/tools/_ros_msgs.py`.
