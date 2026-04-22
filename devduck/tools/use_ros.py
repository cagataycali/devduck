"""ROS2 tool — future-proof rclpy wrapper for DevDuck agents.

Philosophy
----------
One tool, entire ROS2 API. Leverages rclpy + rosidl_runtime_py for
dynamic message resolution — no hardcoded types. Any new msg package
"just works" once installed in the target environment.

Transport
---------
Two execution modes:
  1. native     — rclpy imported in the devduck process (Linux + ROS2)
  2. docker     — rclpy runs inside a container, we `docker exec`
                  (Mac development loop, zero-dep on host)

Mode auto-detected. Override with ROS2_MODE=native|docker env var.

Actions
-------
  list_topics, list_nodes, list_services, list_types
  echo(topic, type?, timeout, count)          — subscribe, return samples
  publish(topic, type, fields, count, rate)   — build msg from dict, pub
  service_call(service, type, fields, timeout)
  info(topic|node|service)                    — introspection
  exec_raw(command)                           — escape hatch for `ros2 *`

Message construction
--------------------
  rosidl_runtime_py.utilities.get_message("pkg/msg/Name") → class
  rosidl_runtime_py.set_message.set_message_fields(msg, dict)

That's the entire "**kwargs getattr" pattern — already solved by
ROS2 itself. We just pipe JSON through it.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import textwrap
from typing import Any, Dict, List, Optional

from strands import tool

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Backend detection
# ─────────────────────────────────────────────────────────────

ROS2_DOCKER_CONTAINER = os.getenv("ROS2_DOCKER_CONTAINER", "ros-dev")
ROS2_DOCKER_SETUP = os.getenv(
    "ROS2_DOCKER_SETUP", "/opt/ros/jazzy/setup.bash"
)
ROS2_MODE_OVERRIDE = os.getenv("ROS2_MODE")  # "native" | "docker"


def _detect_mode() -> str:
    if ROS2_MODE_OVERRIDE:
        return ROS2_MODE_OVERRIDE
    # Native: rclpy importable?
    try:
        import rclpy  # noqa: F401
        return "native"
    except ImportError:
        pass
    # Docker: container exists and is running?
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", ROS2_DOCKER_CONTAINER],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip() == "true":
            return "docker"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "none"


def _err(msg: str) -> Dict[str, Any]:
    return {"status": "error", "content": [{"text": f"🦆 use_ros: {msg}"}]}


def _ok(msg: str) -> Dict[str, Any]:
    return {"status": "success", "content": [{"text": msg}]}


# ─────────────────────────────────────────────────────────────
# Docker backend — exec a python script inside the container
# ─────────────────────────────────────────────────────────────

def _docker_run_python(py_code: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a Python snippet inside the ROS2 container with setup.bash sourced."""
    shell_cmd = f"source {ROS2_DOCKER_SETUP} && python3 -"
    return subprocess.run(
        ["docker", "exec", "-i", ROS2_DOCKER_CONTAINER, "bash", "-c", shell_cmd],
        input=py_code,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _docker_run_cli(args: List[str], timeout: int = 10) -> subprocess.CompletedProcess:
    """Run a `ros2 <args>` CLI command inside the container."""
    shell_cmd = f"source {ROS2_DOCKER_SETUP} && ros2 " + " ".join(shlex.quote(a) for a in args)
    return subprocess.run(
        ["docker", "exec", ROS2_DOCKER_CONTAINER, "bash", "-c", shell_cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ─────────────────────────────────────────────────────────────
# Python snippets executed in the backend (native or docker)
# Each snippet prints a single JSON line to stdout: {"ok":..., "data":...}
# ─────────────────────────────────────────────────────────────

_SNIPPET_HEADER = textwrap.dedent("""
    import json, sys, time
    import rclpy
    from rclpy.node import Node
    from rosidl_runtime_py.utilities import get_message, get_service
    from rosidl_runtime_py.set_message import set_message_fields
    from rosidl_runtime_py.convert import message_to_ordereddict

    def _out(ok, data=None, err=None):
        print(json.dumps({"ok": ok, "data": data, "err": err}))
        sys.stdout.flush()
""")


def _snippet_echo(topic: str, msg_type: str, timeout: float, count: int) -> str:
    return _SNIPPET_HEADER + textwrap.dedent(f"""
        rclpy.init()
        node = Node("devduck_echo")
        MsgCls = get_message({msg_type!r})
        received = []
        def _cb(msg):
            received.append(dict(message_to_ordereddict(msg)))
        sub = node.create_subscription(MsgCls, {topic!r}, _cb, 10)
        deadline = time.time() + {timeout}
        while len(received) < {count} and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()
        rclpy.shutdown()
        _out(True, data={{"samples": received, "count": len(received)}})
    """)


def _snippet_publish(topic: str, msg_type: str, fields: dict, count: int, rate: float) -> str:
    return _SNIPPET_HEADER + textwrap.dedent(f"""
        rclpy.init()
        node = Node("devduck_pub")
        MsgCls = get_message({msg_type!r})
        pub = node.create_publisher(MsgCls, {topic!r}, 10)
        msg = MsgCls()
        set_message_fields(msg, {json.dumps(fields)})
        # let discovery settle
        time.sleep(0.3)
        for _ in range({count}):
            pub.publish(msg)
            time.sleep(1.0 / {rate} if {rate} > 0 else 0)
        node.destroy_node()
        rclpy.shutdown()
        _out(True, data={{"published": {count}, "topic": {topic!r}, "type": {msg_type!r}}})
    """)


def _snippet_service_call(service: str, srv_type: str, fields: dict, timeout: float) -> str:
    return _SNIPPET_HEADER + textwrap.dedent(f"""
        rclpy.init()
        node = Node("devduck_srv")
        SrvCls = get_service({srv_type!r})
        client = node.create_client(SrvCls, {service!r})
        if not client.wait_for_service(timeout_sec={timeout}):
            _out(False, err="service not available within timeout")
            sys.exit(0)
        req = SrvCls.Request()
        set_message_fields(req, {json.dumps(fields)})
        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec={timeout})
        if future.result() is None:
            _out(False, err="service call timed out")
        else:
            _out(True, data=dict(message_to_ordereddict(future.result())))
        node.destroy_node()
        rclpy.shutdown()
    """)


# ─────────────────────────────────────────────────────────────
# Backend dispatch
# ─────────────────────────────────────────────────────────────

def _run_python(py_code: str, timeout: int = 30) -> Dict[str, Any]:
    mode = _detect_mode()
    if mode == "native":
        proc = subprocess.run(
            ["python3", "-c", py_code],
            capture_output=True, text=True, timeout=timeout,
        )
    elif mode == "docker":
        proc = _docker_run_python(py_code, timeout=timeout)
    else:
        return {"ok": False, "err": "no ROS2 backend (install rclpy or run a container named ros-dev)"}

    if proc.returncode != 0:
        return {"ok": False, "err": f"backend error: {proc.stderr.strip() or proc.stdout.strip()}"}

    # Last JSON line from stdout
    last_line = None
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            last_line = line
    if not last_line:
        return {"ok": False, "err": f"no JSON output: {proc.stdout!r}"}
    try:
        return json.loads(last_line)
    except json.JSONDecodeError as e:
        return {"ok": False, "err": f"invalid JSON: {e} in {last_line!r}"}


def _run_cli(args: List[str], timeout: int = 10) -> Dict[str, Any]:
    mode = _detect_mode()
    if mode == "native":
        proc = subprocess.run(
            ["ros2"] + args, capture_output=True, text=True, timeout=timeout,
        )
    elif mode == "docker":
        proc = _docker_run_cli(args, timeout=timeout)
    else:
        return {"ok": False, "err": "no ROS2 backend"}
    if proc.returncode != 0:
        return {"ok": False, "err": proc.stderr.strip() or proc.stdout.strip()}
    return {"ok": True, "data": proc.stdout.strip()}


# ─────────────────────────────────────────────────────────────
# Public tool
# ─────────────────────────────────────────────────────────────

@tool
def use_ros(
    action: str,
    topic: Optional[str] = None,
    service: Optional[str] = None,
    type: Optional[str] = None,
    fields: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
    count: int = 1,
    rate: float = 10.0,
    command: Optional[str] = None,
) -> Dict[str, Any]:
    """🤖 ROS2 tool — publish, subscribe, service call, introspect.

    Future-proof: uses rosidl_runtime_py for dynamic type resolution.
    Any msg type installed in the ROS2 env works — no hardcoding.

    Actions:
        status          — show backend (native/docker) and container health
        list_topics     — list all topics (with types)
        list_nodes      — list all nodes
        list_services   — list all services
        info            — `ros2 topic/node/service info <name>` (auto-detects)
        echo            — subscribe, collect samples, return as JSON
                          args: topic, type?, timeout=5, count=1
        publish         — build msg from fields dict, publish
                          args: topic, type, fields, count=1, rate=10
        service_call    — call a service with fields dict
                          args: service, type, fields, timeout=5
        exec_raw        — escape hatch: run `ros2 <command>` verbatim
                          args: command="topic list"

    Examples:
        use_ros(action="list_topics")
        use_ros(action="echo", topic="/turtle1/pose", timeout=2)
        use_ros(action="publish", topic="/turtle1/cmd_vel",
             type="geometry_msgs/msg/Twist",
             fields={"linear": {"x": 2.0}, "angular": {"z": 1.5}})
        use_ros(action="service_call", service="/spawn",
             type="turtlesim/srv/Spawn",
             fields={"x": 3.0, "y": 3.0, "name": "t2"})
    """
    fields = fields or {}

    # ── status / backend info ────────────────────────────────
    if action == "status":
        mode = _detect_mode()
        text = f"🦆 use_ros backend: {mode}\n"
        if mode == "docker":
            text += f"   container: {ROS2_DOCKER_CONTAINER}\n   setup: {ROS2_DOCKER_SETUP}"
        elif mode == "none":
            text += "   no backend — set ROS2_MODE, install rclpy, or start docker container 'ros-dev'"
        return _ok(text)

    # ── list_* actions (use CLI — simpler than rclpy introspection) ──
    if action == "list_topics":
        res = _run_cli(["topic", "list", "-t"])
        if not res["ok"]:
            return _err(res["err"])
        return _ok(f"🦆 topics:\n{res['data']}")

    if action == "list_nodes":
        res = _run_cli(["node", "list"])
        if not res["ok"]:
            return _err(res["err"])
        return _ok(f"🦆 nodes:\n{res['data']}")

    if action == "list_services":
        res = _run_cli(["service", "list", "-t"])
        if not res["ok"]:
            return _err(res["err"])
        return _ok(f"🦆 services:\n{res['data']}")

    if action == "info":
        # Auto-detect: topic? node? service?
        target = topic or service or command
        if not target:
            return _err("info requires topic/service/command")
        for kind in ("topic", "node", "service"):
            res = _run_cli([kind, "info", target])
            if res["ok"] and res["data"]:
                return _ok(f"🦆 {kind} info {target}:\n{res['data']}")
        return _err(f"no info found for {target}")

    # ── echo / publish / service_call (use rclpy snippets) ───
    if action == "echo":
        if not topic:
            return _err("echo requires topic")
        if not type:
            # auto-resolve from `ros2 topic list -t`
            types_res = _run_cli(["topic", "list", "-t"])
            if types_res["ok"]:
                for line in types_res["data"].splitlines():
                    # format: "/name [type/Name]"
                    if line.startswith(topic + " "):
                        type = line.split("[", 1)[1].rstrip("]").strip()
                        break
            if not type:
                return _err(f"cannot resolve type for {topic}; pass type=...")
        res = _run_python(_snippet_echo(topic, type, timeout, count), timeout=int(timeout) + 5)
        if not res["ok"]:
            return _err(res["err"])
        samples = res["data"]["samples"]
        if not samples:
            return _ok(f"🦆 echo {topic}: no samples in {timeout}s")
        text = f"🦆 echo {topic} ({type}) — {len(samples)} sample(s):\n"
        text += json.dumps(samples, indent=2, default=str)
        return _ok(text)

    if action == "publish":
        if not topic or not type:
            return _err("publish requires topic and type")
        res = _run_python(
            _snippet_publish(topic, type, fields, count, rate),
            timeout=int(count / max(rate, 0.1)) + 10,
        )
        if not res["ok"]:
            return _err(res["err"])
        return _ok(f"🦆 published {count}× to {topic} ({type})")

    if action == "service_call":
        if not service or not type:
            return _err("service_call requires service and type")
        res = _run_python(
            _snippet_service_call(service, type, fields, timeout),
            timeout=int(timeout) + 10,
        )
        if not res["ok"]:
            return _err(res["err"])
        text = f"🦆 service {service} ({type}) response:\n"
        text += json.dumps(res["data"], indent=2, default=str)
        return _ok(text)

    # ── exec_raw — agent escape hatch ────────────────────────
    if action == "exec_raw":
        if not command:
            return _err("exec_raw requires command (e.g. 'topic hz /turtle1/pose')")
        res = _run_cli(shlex.split(command), timeout=int(timeout))
        if not res["ok"]:
            return _err(res["err"])
        return _ok(f"🦆 ros2 {command}:\n{res['data']}")

    return _err(f"unknown action: {action}")
