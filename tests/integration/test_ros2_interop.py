"""End-to-end integration test: DevDuck ↔ real ROS2 Jazzy over CycloneDDS.

Strategy
--------
Docker Desktop on macOS cannot bridge host-level multicast into a
container, so we run *everything* inside a single Ubuntu container that
has both ROS2 Jazzy and DevDuck's DDS/ROS2 tools loaded. A real
`demo_nodes_cpp talker` publishes `/chatter`; DevDuck's `dds_peer` +
`use_ros` connect on the same CycloneDDS domain and must:

  1. Discover the ROS2 node via SPDP
  2. Enumerate its topics (incl. rq/rr service topics)
  3. Echo a sample from /chatter (decode real std_msgs/String bytes)
  4. Stream /chatter into event_bus via `tail`
  5. Publish a new topic and round-trip it
  6. Enumerate services and call `/add_two_ints` on a live demo server

Exit code 0 = all assertions passed. Any failure aborts with detail.

Run manually
------------
    python tests/integration/test_ros2_interop.py

The harness is self-contained: it pulls/launches the ROS2 container,
installs cyclonedds Python inside it, copies DevDuck's tool sources in,
runs the inner test, and cleans up. Takes ~60–120 s on first run
(image pull + apt-get), ~30 s on subsequent runs.

Requirements
------------
    * docker (with Desktop on macOS; root docker on Linux)
    * 2+ GB free for the ros:jazzy image
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TOOLS_DIR = ROOT / "devduck" / "tools"
CONTAINER = "devduck-ros2-interop"
IMAGE = "ros:jazzy"


def _run(cmd: list[str], *, check: bool = True, capture: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a shell command with a sane default timeout."""
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
        timeout=timeout,
    )


def _docker_exec(cmd: str, *, timeout: int = 120) -> str:
    """Exec a bash command inside the test container and return stdout."""
    r = _run(
        ["docker", "exec", CONTAINER, "bash", "-c", cmd],
        timeout=timeout,
    )
    return r.stdout


def ensure_docker() -> None:
    if not shutil.which("docker"):
        print("FAIL: docker binary not found on PATH", file=sys.stderr)
        sys.exit(2)
    try:
        _run(["docker", "info"], timeout=15)
    except Exception as e:
        print(f"FAIL: docker not running: {e}", file=sys.stderr)
        sys.exit(2)


def teardown() -> None:
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER],
        capture_output=True,
        check=False,
        timeout=30,
    )


def launch_container() -> None:
    """Launch the ROS2 Jazzy container with demo nodes installed and the
    talker running as the foreground command."""
    teardown()
    print(f"🐳 launching {CONTAINER} from {IMAGE} (with demo_nodes)…")
    # Pull first so we can time-bound apt-get separately.
    _run(["docker", "pull", IMAGE], timeout=600)

    _run(
        [
            "docker", "run", "-d",
            "--name", CONTAINER,
            "--network", "host",
            "-e", "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
            IMAGE,
            "bash", "-c",
            textwrap.dedent(
                """
                set -e
                apt-get update -qq
                apt-get install -y -qq \
                    ros-jazzy-demo-nodes-cpp \
                    ros-jazzy-example-interfaces \
                    ros-jazzy-rmw-cyclonedds-cpp \
                    cyclonedds-dev \
                    python3-pip \
                    > /dev/null
                # Symlink headers/libs into a layout the cyclonedds wheel builder recognises.
                mkdir -p /opt/cyclone/lib /opt/cyclone/bin /opt/cyclone/include
                arch=$(uname -m)
                lnsrc=/usr/lib/${arch}-linux-gnu
                ln -sf ${lnsrc}/libddsc.so       /opt/cyclone/lib/libddsc.so
                ln -sf ${lnsrc}/libcycloneddsidl.so /opt/cyclone/lib/ 2>/dev/null || true
                ln -sf /usr/include/dds  /opt/cyclone/include/dds
                ln -sf /usr/include/ddsc /opt/cyclone/include/ddsc
                ln -sf /usr/bin/idlc     /opt/cyclone/bin/idlc
                CYCLONEDDS_HOME=/opt/cyclone pip install --break-system-packages --no-build-isolation \
                    scikit-build setuptools wheel 'cyclonedds==0.10.4' > /dev/null 2>&1
                source /opt/ros/jazzy/setup.bash
                # Start the demo services alongside the talker.
                ros2 run demo_nodes_cpp add_two_ints_server &
                exec ros2 run demo_nodes_cpp talker
                """
            ).strip(),
        ],
        timeout=60,
    )

    # Wait for the talker to start publishing.
    print("⏳ waiting for talker to publish…")
    deadline = time.time() + 300
    while time.time() < deadline:
        _r = subprocess.run(
            ["docker", "logs", CONTAINER],
            capture_output=True, text=True, timeout=10,
        )
        logs = _r.stdout + _r.stderr
        if "Publishing: 'Hello World:" in logs:
            print("✅ talker is publishing /chatter")
            # Give the add_two_ints server a couple extra seconds to come up.
            time.sleep(2)
            return
        time.sleep(2)
    print("FAIL: talker never started. Last logs:", file=sys.stderr)
    _fr = subprocess.run(["docker", "logs", CONTAINER], capture_output=True, text=True)
    print((_fr.stdout + _fr.stderr)[-2000:])
    sys.exit(3)


def copy_tools_in() -> None:
    print("📦 copying devduck tool sources into container…")
    _run([
        "docker", "exec", CONTAINER,
        "bash", "-c",
        "mkdir -p /opt/devduck_tools && touch /opt/devduck_tools/__init__.py",
    ], timeout=10)
    for fname in ("_ros_msgs.py", "dds_peer.py", "use_ros.py"):
        src = TOOLS_DIR / fname
        if not src.exists():
            print(f"FAIL: missing {src}", file=sys.stderr)
            sys.exit(4)
        _run([
            "docker", "cp",
            str(src),
            f"{CONTAINER}:/opt/devduck_tools/{fname}",
        ], timeout=15)


# -- The script executed *inside* the container to exercise our tools.
INNER_SCRIPT = r'''
import sys, os, time, types

# Stub strands so @tool returns the function directly.
stub = types.ModuleType("strands"); stub.tool = lambda fn: fn
sys.modules["strands"] = stub

# Stub the event_bus and capture everything the tools emit.
events = []
class FakeBus:
    def emit(self, **kw):
        events.append(kw)
fake = types.ModuleType("devduck.tools.event_bus")
fake.bus = FakeBus()
sys.modules["devduck.tools.event_bus"] = fake

# Simulate the devduck package layout.
sys.path.insert(0, "/opt")
pkg = types.ModuleType("devduck"); sys.modules["devduck"] = pkg
tpkg = types.ModuleType("devduck.tools"); tpkg.__path__ = ["/opt/devduck_tools"]
sys.modules["devduck.tools"] = tpkg

import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(f"devduck.tools.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"devduck.tools.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod

_ros_msgs = _load("_ros_msgs", "/opt/devduck_tools/_ros_msgs.py")
dds_peer  = _load("dds_peer",  "/opt/devduck_tools/dds_peer.py")
use_ros   = _load("use_ros",   "/opt/devduck_tools/use_ros.py")

failures = []
def _assert(cond, msg):
    if cond:
        print(f"PASS: {msg}")
    else:
        failures.append(msg)
        print(f"FAIL: {msg}")

# TEST 1 — start, discover real ROS2 participants.
r = dds_peer.dds_peer(action="start")
_assert(r["status"] == "success", "dds_peer start")

# Give SPDP/SEDP a few seconds to settle.
time.sleep(6)

r = dds_peer.dds_peer(action="list_participants")
text = r["content"][0]["text"]
_assert(
    any(f"participants ({n}" in text for n in range(2, 10)),
    f"discovered ≥2 DDS participants (got: {text.splitlines()[0]!r})",
)

r = dds_peer.dds_peer(action="list_topics")
text = r["content"][0]["text"]
_assert("rt/chatter" in text, "/chatter topic discovered via SEDP")
_assert("std_msgs::msg::dds_::String_" in text, "chatter type is std_msgs/String")

# TEST 2 — use_ros sees /chatter mapped to ROS2-style name.
r = use_ros.use_ros(action="list_topics")
text = r["content"][0]["text"]
_assert("/chatter" in text and "[std_msgs::msg::dds_::String_]" in text,
        "use_ros list_topics maps rt/chatter → /chatter")
_assert("/rosout" in text, "/rosout topic visible")
_assert("/parameter_events" in text, "/parameter_events topic visible")

# TEST 3 — echo a real sample from demo_nodes_cpp talker.
r = use_ros.use_ros(action="echo", topic="/chatter", timeout=5.0)
text = r["content"][0]["text"]
_assert(r["status"] == "success", "echo /chatter returns success")
_assert("Hello World" in text, "echo payload contains talker phrase")

# TEST 4 — tail /chatter for a few seconds.
r = use_ros.use_ros(action="tail", topic="/chatter", max_hz=10.0)
_assert(r["status"] == "success", "tail /chatter starts")
time.sleep(4)
r = use_ros.use_ros(action="list_tails")
_assert("/chatter" in r["content"][0]["text"], "tail appears in list_tails")

r = use_ros.use_ros(action="untail", topic="/chatter")
_assert(r["status"] == "success", "untail /chatter stops cleanly")

ros_events = [e for e in events if e.get("event_type", "").startswith("ros.chatter")]
_assert(len(ros_events) >= 1, f"tail emitted ≥1 ros.chatter event on event_bus (got {len(ros_events)})")

# TEST 5 — roundtrip a publish-then-echo on a new topic.
r = use_ros.use_ros(
    action="pub", topic="/devduck_roundtrip",
    type="std_msgs/msg/String",
    msg={"data": "integration test round trip"},
)
_assert(r["status"] == "success", "pub /devduck_roundtrip succeeds")
time.sleep(1)
r = use_ros.use_ros(
    action="echo", topic="/devduck_roundtrip",
    type="std_msgs/msg/String", timeout=3.0,
)
_assert("integration test round trip" in r["content"][0]["text"],
        "roundtrip payload matches what we published")

# TEST 6 — /rosout Log decoding (new type we just added).
r = use_ros.use_ros(action="echo", topic="/rosout",
                    type="rcl_interfaces/msg/Log", timeout=5.0)
# demo_nodes_cpp logs an INFO on every publish so we should get one fast.
_assert(r["status"] == "success" and "Hello World" in r["content"][0]["text"],
        "echo /rosout decodes rcl_interfaces/msg/Log with real talker log line")

# TEST 7 — /parameter_events type decoding.
# We don't expect a sample in the 2s window (talker rarely changes params),
# but the TYPE must at least resolve without the old "unknown" tag.
idl_cls = _ros_msgs.ros_type_to_idl("rcl_interfaces/msg/ParameterEvent")
_assert(idl_cls is not None, "ParameterEvent idl class resolves (type registry lookup)")

# TEST 8 — list_services + call /add_two_ints on a live demo server.
r = use_ros.use_ros(action="list_services")
text = r["content"][0]["text"]
_assert("/add_two_ints" in text,
        f"list_services finds /add_two_ints (got: {text!r})")

r = use_ros.use_ros(
    action="call",
    service="/add_two_ints",
    srv_type="example_interfaces/srv/AddTwoInts",
    msg={"a": 17, "b": 25},
    timeout=6.0,
)
text = r["content"][0]["text"]
_assert(r["status"] == "success", f"call /add_two_ints returns success (got: {r['status']}, {text!r})")
_assert('"sum": 42' in text, f"call /add_two_ints returns 17+25=42 (got: {text!r})")

# Clean up
dds_peer.dds_peer(action="stop")

print()
print("=" * 60)
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"ALL {sum(1 for _ in events if True) and 'TESTS'} PASSED")
sys.exit(0)
'''


def run_inner_test() -> int:
    print("🧪 running inner test harness inside container…")
    script_path = "/opt/test_inner.py"
    # Write the inner script into the container.
    _run([
        "docker", "exec", CONTAINER,
        "bash", "-c",
        f"cat > {script_path} <<'EOF'\n{INNER_SCRIPT}\nEOF",
    ], timeout=10)
    # Run it. Stream output so the user sees it.
    cp = subprocess.run(
        [
            "docker", "exec", CONTAINER,
            "bash", "-c",
            f"source /opt/ros/jazzy/setup.bash && python3 {script_path}",
        ],
        timeout=180,
    )
    return cp.returncode


def main() -> int:
    ensure_docker()
    try:
        launch_container()
        copy_tools_in()
        rc = run_inner_test()
    finally:
        if os.environ.get("KEEP_CONTAINER") != "1":
            teardown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
