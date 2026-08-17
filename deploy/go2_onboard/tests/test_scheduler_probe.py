import ast
import types
from pathlib import Path

from deploy.go2_onboard import scheduler_probe


def test_lowstate_probe_uses_read_only_subscriber_and_never_writes(monkeypatch):
    calls = {"factory": [], "subscriber": [], "publisher": 0, "writes": 0}

    class ReadOnlyChannelModule:
        def __getattr__(self, name):
            if name == "ChannelPublisher":
                calls["publisher"] += 1
                raise AssertionError("lowstate probe must not access ChannelPublisher")
            raise AttributeError(name)

        @staticmethod
        def ChannelFactoryInitialize(domain, net):
            calls["factory"].append((domain, net))

        class ChannelSubscriber:
            def __init__(self, topic, message_type):
                calls["subscriber"].append((topic, message_type))

            def Init(self, callback, queue_len):
                assert queue_len == 10
                callback(object())

            def Close(self):
                pass

    class LowState:
        pass

    def fake_import(name):
        if name == "unitree_sdk2py.core.channel":
            return ReadOnlyChannelModule
        if name == "unitree_sdk2py.idl.unitree_go.msg.dds_":
            return types.SimpleNamespace(LowState_=LowState)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(scheduler_probe.importlib, "import_module", fake_import)
    result = scheduler_probe.run_lowstate_probe(0.01, "enp3s0")

    assert calls["factory"] == [(0, "enp3s0")]
    assert calls["subscriber"] == [("rt/lowstate", LowState)]
    assert calls["publisher"] == 0
    assert calls["writes"] == 0
    assert result["topic"] == "rt/lowstate"
    assert result["queue_len"] == 10
    assert result["LOWCMD_PUBLISHER_CREATED"] == "NO"
    assert result["LOWCMD_WRITE_COUNT"] == 0
    assert result["lowstate"]["rx_count"] == 1


def test_scheduler_probe_source_has_no_write_path():
    source = Path("deploy/go2_onboard/scheduler_probe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "ChannelPublisher" not in source
    assert "LowCmd" not in source
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"Write", "write", "publish"}
        for node in ast.walk(tree)
    )
