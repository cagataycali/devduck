"""Tests for the CycloneDDS peer tool (``devduck.tools.dds_peer``).

These tests deliberately bypass the full ``devduck`` package import chain
(which pulls in heavy optional deps like ``bs4``) and load the module
directly with ``importlib``. This keeps the test suite self-contained
and runnable from any environment where only ``cyclonedds`` is
installed.
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

cyclonedds = pytest.importorskip("cyclonedds")

MODULE_PATH = Path(__file__).parent.parent / "devduck" / "tools" / "dds_peer.py"


def _load_module():
    """Load dds_peer by file path and register in ``sys.modules``.

    The dataclass-based IDL type defined at module scope needs the module
    to be registered in ``sys.modules`` before ``@dataclass`` executes
    (otherwise ``dataclasses._is_type`` can't find the owning module).
    """
    mod_name = "dds_peer_test_target"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, str(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dds():
    """Yield a freshly-loaded module with guaranteed clean shutdown."""
    module = _load_module()
    yield module
    # Always try to shut down to release the domain participant.
    try:
        module.dds_peer(action="stop")
    except Exception:
        pass


def test_import(dds):
    """The tool module imports and exposes the expected surface."""
    assert hasattr(dds, "dds_peer")
    assert hasattr(dds, "DDS_STATE")
    assert dds.DDS_STATE["running"] is False


def test_start_stop(dds):
    """start followed by stop should leave DDS_STATE clean."""
    r = dds.dds_peer(action="start", domain_id=0)
    assert r["status"] == "success"
    assert dds.DDS_STATE["running"] is True
    assert dds.DDS_STATE["instance_id"].startswith("dds-")

    r = dds.dds_peer(action="status")
    texts = " ".join(c["text"] for c in r["content"])
    assert "running" in texts.lower()
    assert "Domain: 0" in texts

    r = dds.dds_peer(action="stop")
    assert r["status"] == "success"
    assert dds.DDS_STATE["running"] is False


def test_publish_subscribe_roundtrip(dds):
    """A local publish should be readable by a local subscribe."""
    dds.dds_peer(action="start", domain_id=0)

    # Create the reader BEFORE the write so it latches new samples
    # (default QoS is KEEP_LAST depth 1 and the reader doesn't request history).
    topic = "devduck/test_roundtrip"
    dds.dds_peer(action="subscribe", topic=topic, wait_time=0.2)

    dds.dds_peer(action="publish", topic=topic, message="hello-roundtrip")
    time.sleep(0.5)

    r = dds.dds_peer(action="subscribe", topic=topic, wait_time=0.3)
    assert r["status"] == "success"
    texts = " ".join(c["text"] for c in r["content"])
    assert "hello-roundtrip" in texts


def test_discovery_lists_self(dds):
    """The builtin participant reader must discover our own participant."""
    dds.dds_peer(action="start", domain_id=0)
    # Give discovery a moment to settle.
    time.sleep(0.5)
    dds.dds_peer(action="discover")

    r = dds.dds_peer(action="list_participants")
    assert r["status"] == "success"
    text = r["content"][0]["text"]
    # At least one participant (ourselves) should be reported.
    assert "participant" in text.lower()
    assert len(dds.DDS_STATE["discovered_participants"]) >= 1
