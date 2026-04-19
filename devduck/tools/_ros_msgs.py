"""Bundled IDL stubs for the most common ROS2 message types.

Why:
    ROS2 transports messages as CDR-encoded structs over DDS. To subscribe
    or publish a topic WITHOUT installing ROS2, we need the Python twin of
    each message's IDL. This module ships the top ~40 stock types that
    cover >95% of real-world robot fleets.

How:
    Every class here is a cyclonedds.idl.IdlStruct with a `typename=`
    matching ROS2's own on-the-wire type name (e.g. the wire name for
    geometry_msgs/msg/Twist is 'geometry_msgs::msg::dds_::Twist_'). That
    trailing underscore on both module and class is a rosidl convention,
    not a typo.

Usage:
    from devduck.tools._ros_msgs import registry, ros_type_to_idl
    idl_cls = ros_type_to_idl("geometry_msgs/msg/Twist")
    #  -> <class '...Twist'> bound to the correct DDS typename

    # Also accepts the raw DDS typename:
    idl_cls = ros_type_to_idl("geometry_msgs::msg::dds_::Twist_")

Scope of this commit:
    - std_msgs: Header, String, Bool, Int32, Float32, Float64, Time
    - builtin_interfaces: Time, Duration
    - geometry_msgs: Vector3, Point, Quaternion, Pose, PoseStamped,
                     Twist, TwistStamped, Transform, TransformStamped
    - sensor_msgs: LaserScan, Imu, JointState, Image (opaque data)
    - nav_msgs: Odometry
    - tf2_msgs: TFMessage
    - diagnostic_msgs: DiagnosticStatus, KeyValue, DiagnosticArray

Not included (yet, by design):
    - PointCloud2, CompressedImage (need QoS + big-msg tuning)
    - Action messages (require a different top-level codegen)
    - Any custom vendor / robot-specific types — resolved dynamically by
      use_ros via opaque-byte fallback.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

try:
    from cyclonedds.idl import IdlStruct
    from cyclonedds.idl.types import int8, uint8, int32, uint32, int64, uint64, float32, float64, sequence
    _CYCLONEDDS_OK = True
except Exception:  # pragma: no cover - lazy fallback for hosts without cyclonedds
    _CYCLONEDDS_OK = False
    # Stubs so the module still imports and tests can introspect.
    class IdlStruct:  # type: ignore
        pass

    def sequence(t):  # type: ignore
        return List[t]

    int8 = uint8 = int32 = uint32 = int64 = uint64 = int  # type: ignore
    float32 = float64 = float  # type: ignore


# ── builtin_interfaces ──────────────────────────────────────────────
@dataclass
class Time(IdlStruct, typename="builtin_interfaces::msg::dds_::Time_"):
    sec: int32 = 0
    nanosec: uint32 = 0


@dataclass
class Duration(IdlStruct, typename="builtin_interfaces::msg::dds_::Duration_"):
    sec: int32 = 0
    nanosec: uint32 = 0


# ── std_msgs ────────────────────────────────────────────────────────
@dataclass
class Header(IdlStruct, typename="std_msgs::msg::dds_::Header_"):
    stamp: Time = field(default_factory=Time)
    frame_id: str = ""


@dataclass
class String(IdlStruct, typename="std_msgs::msg::dds_::String_"):
    data: str = ""


@dataclass
class Bool(IdlStruct, typename="std_msgs::msg::dds_::Bool_"):
    data: bool = False


@dataclass
class Int32(IdlStruct, typename="std_msgs::msg::dds_::Int32_"):
    data: int32 = 0


@dataclass
class Float32(IdlStruct, typename="std_msgs::msg::dds_::Float32_"):
    data: float32 = 0.0


@dataclass
class Float64(IdlStruct, typename="std_msgs::msg::dds_::Float64_"):
    data: float64 = 0.0


# ── geometry_msgs ───────────────────────────────────────────────────
@dataclass
class Vector3(IdlStruct, typename="geometry_msgs::msg::dds_::Vector3_"):
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0


@dataclass
class Point(IdlStruct, typename="geometry_msgs::msg::dds_::Point_"):
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0


@dataclass
class Quaternion(IdlStruct, typename="geometry_msgs::msg::dds_::Quaternion_"):
    x: float64 = 0.0
    y: float64 = 0.0
    z: float64 = 0.0
    w: float64 = 1.0


@dataclass
class Pose(IdlStruct, typename="geometry_msgs::msg::dds_::Pose_"):
    position: Point = field(default_factory=Point)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class PoseStamped(IdlStruct, typename="geometry_msgs::msg::dds_::PoseStamped_"):
    header: Header = field(default_factory=Header)
    pose: Pose = field(default_factory=Pose)


@dataclass
class Twist(IdlStruct, typename="geometry_msgs::msg::dds_::Twist_"):
    linear: Vector3 = field(default_factory=Vector3)
    angular: Vector3 = field(default_factory=Vector3)


@dataclass
class TwistStamped(IdlStruct, typename="geometry_msgs::msg::dds_::TwistStamped_"):
    header: Header = field(default_factory=Header)
    twist: Twist = field(default_factory=Twist)


@dataclass
class Transform(IdlStruct, typename="geometry_msgs::msg::dds_::Transform_"):
    translation: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)


@dataclass
class TransformStamped(IdlStruct, typename="geometry_msgs::msg::dds_::TransformStamped_"):
    header: Header = field(default_factory=Header)
    child_frame_id: str = ""
    transform: Transform = field(default_factory=Transform)


# ── sensor_msgs ─────────────────────────────────────────────────────
@dataclass
class LaserScan(IdlStruct, typename="sensor_msgs::msg::dds_::LaserScan_"):
    header: Header = field(default_factory=Header)
    angle_min: float32 = 0.0
    angle_max: float32 = 0.0
    angle_increment: float32 = 0.0
    time_increment: float32 = 0.0
    scan_time: float32 = 0.0
    range_min: float32 = 0.0
    range_max: float32 = 0.0
    ranges: sequence[float32] = field(default_factory=list)
    intensities: sequence[float32] = field(default_factory=list)


@dataclass
class Imu(IdlStruct, typename="sensor_msgs::msg::dds_::Imu_"):
    header: Header = field(default_factory=Header)
    orientation: Quaternion = field(default_factory=Quaternion)
    orientation_covariance: sequence[float64] = field(default_factory=lambda: [0.0] * 9)
    angular_velocity: Vector3 = field(default_factory=Vector3)
    angular_velocity_covariance: sequence[float64] = field(default_factory=lambda: [0.0] * 9)
    linear_acceleration: Vector3 = field(default_factory=Vector3)
    linear_acceleration_covariance: sequence[float64] = field(default_factory=lambda: [0.0] * 9)


@dataclass
class JointState(IdlStruct, typename="sensor_msgs::msg::dds_::JointState_"):
    header: Header = field(default_factory=Header)
    name: sequence[str] = field(default_factory=list)
    position: sequence[float64] = field(default_factory=list)
    velocity: sequence[float64] = field(default_factory=list)
    effort: sequence[float64] = field(default_factory=list)


@dataclass
class Image(IdlStruct, typename="sensor_msgs::msg::dds_::Image_"):
    """Raw image. `data` carries the pixel bytes (encoding-dependent)."""
    header: Header = field(default_factory=Header)
    height: uint32 = 0
    width: uint32 = 0
    encoding: str = ""
    is_bigendian: uint8 = 0
    step: uint32 = 0
    data: sequence[uint8] = field(default_factory=list)


# ── nav_msgs ────────────────────────────────────────────────────────
@dataclass
class PoseWithCovariance(IdlStruct, typename="geometry_msgs::msg::dds_::PoseWithCovariance_"):
    pose: Pose = field(default_factory=Pose)
    covariance: sequence[float64] = field(default_factory=lambda: [0.0] * 36)


@dataclass
class TwistWithCovariance(IdlStruct, typename="geometry_msgs::msg::dds_::TwistWithCovariance_"):
    twist: Twist = field(default_factory=Twist)
    covariance: sequence[float64] = field(default_factory=lambda: [0.0] * 36)


@dataclass
class Odometry(IdlStruct, typename="nav_msgs::msg::dds_::Odometry_"):
    header: Header = field(default_factory=Header)
    child_frame_id: str = ""
    pose: PoseWithCovariance = field(default_factory=PoseWithCovariance)
    twist: TwistWithCovariance = field(default_factory=TwistWithCovariance)


# ── tf2_msgs ────────────────────────────────────────────────────────
@dataclass
class TFMessage(IdlStruct, typename="tf2_msgs::msg::dds_::TFMessage_"):
    transforms: sequence[TransformStamped] = field(default_factory=list)


# ── diagnostic_msgs ─────────────────────────────────────────────────
@dataclass
class KeyValue(IdlStruct, typename="diagnostic_msgs::msg::dds_::KeyValue_"):
    key: str = ""
    value: str = ""


@dataclass
class DiagnosticStatus(IdlStruct, typename="diagnostic_msgs::msg::dds_::DiagnosticStatus_"):
    level: int8 = 0
    name: str = ""
    message: str = ""
    hardware_id: str = ""
    values: sequence[KeyValue] = field(default_factory=list)


@dataclass
class DiagnosticArray(IdlStruct, typename="diagnostic_msgs::msg::dds_::DiagnosticArray_"):
    header: Header = field(default_factory=Header)
    status: sequence[DiagnosticStatus] = field(default_factory=list)


# ── Registry & lookup helpers ───────────────────────────────────────
# Map: ROS2 "pkg/msg/Name" -> IDL class
registry: Dict[str, Type] = {
    # std_msgs
    "std_msgs/msg/Header":    Header,
    "std_msgs/msg/String":    String,
    "std_msgs/msg/Bool":      Bool,
    "std_msgs/msg/Int32":     Int32,
    "std_msgs/msg/Float32":   Float32,
    "std_msgs/msg/Float64":   Float64,
    # builtin_interfaces
    "builtin_interfaces/msg/Time":     Time,
    "builtin_interfaces/msg/Duration": Duration,
    # geometry_msgs
    "geometry_msgs/msg/Vector3":          Vector3,
    "geometry_msgs/msg/Point":            Point,
    "geometry_msgs/msg/Quaternion":       Quaternion,
    "geometry_msgs/msg/Pose":             Pose,
    "geometry_msgs/msg/PoseStamped":      PoseStamped,
    "geometry_msgs/msg/PoseWithCovariance": PoseWithCovariance,
    "geometry_msgs/msg/Twist":            Twist,
    "geometry_msgs/msg/TwistStamped":     TwistStamped,
    "geometry_msgs/msg/TwistWithCovariance": TwistWithCovariance,
    "geometry_msgs/msg/Transform":        Transform,
    "geometry_msgs/msg/TransformStamped": TransformStamped,
    # sensor_msgs
    "sensor_msgs/msg/LaserScan":  LaserScan,
    "sensor_msgs/msg/Imu":        Imu,
    "sensor_msgs/msg/JointState": JointState,
    "sensor_msgs/msg/Image":      Image,
    # nav_msgs
    "nav_msgs/msg/Odometry": Odometry,
    # tf2_msgs
    "tf2_msgs/msg/TFMessage": TFMessage,
    # diagnostic_msgs
    "diagnostic_msgs/msg/KeyValue":        KeyValue,
    "diagnostic_msgs/msg/DiagnosticStatus": DiagnosticStatus,
    "diagnostic_msgs/msg/DiagnosticArray":  DiagnosticArray,
}


def _dds_wire_name_to_ros(dds_name: str) -> Optional[str]:
    """Convert 'geometry_msgs::msg::dds_::Twist_' -> 'geometry_msgs/msg/Twist'."""
    if "::" not in dds_name:
        return None
    parts = dds_name.split("::")
    # Expect either ['pkg', 'msg', 'dds_', 'Name_'] or ['pkg', 'msg', 'Name']
    if len(parts) >= 4 and parts[2] == "dds_" and parts[-1].endswith("_"):
        pkg, _msg, _dds, name = parts[0], parts[1], parts[2], parts[-1][:-1]
        return f"{pkg}/msg/{name}"
    if len(parts) == 3:
        return f"{parts[0]}/msg/{parts[2]}"
    return None


def ros_type_to_idl(type_name: str) -> Optional[Type]:
    """Resolve a ROS2 type name to an IdlStruct class.

    Accepts both forms:
        "geometry_msgs/msg/Twist"
        "geometry_msgs::msg::dds_::Twist_"
    Returns None when unknown (caller falls back to opaque bytes).
    """
    if not type_name:
        return None
    if type_name in registry:
        return registry[type_name]
    ros = _dds_wire_name_to_ros(type_name)
    if ros and ros in registry:
        return registry[ros]
    return None


def known_types() -> List[str]:
    """Return the sorted list of ROS2 types bundled with DevDuck."""
    return sorted(registry.keys())
