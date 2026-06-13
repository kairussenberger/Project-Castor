"""HardwareSink safety wiring, tested without the i2rt SDK: active-side
filtering, the hands-off default, the rest-pose engage gate (fail-closed,
torque released), and shaper init from the MEASURED pose."""
from __future__ import annotations

import numpy as np
import pytest

import bimanual_teleop.arms.yam_driver as yam_driver
from bimanual_teleop.arms.joint_map import JointMap, save_joint_map
from bimanual_teleop.config import load_rig
from bimanual_teleop.hardware import HardwareSink, active_sides


class FakeYamArm:
    """Stands in for the CAN driver: reports a configurable measured pose."""

    instances: list["FakeYamArm"] = []
    measured: dict[str, np.ndarray] = {}

    def __init__(self, channel: str, joint_map=None, *, require_map: bool = True,
                 model_limits=None):
        assert joint_map is not None, "sink must pass the calibrated map"
        assert model_limits is not None, "sink must install the model-limit clamp"
        self.channel = channel
        self.map = joint_map
        self.commands: list[np.ndarray] = []
        self.closed = False
        FakeYamArm.instances.append(self)

    def state(self) -> np.ndarray:
        return FakeYamArm.measured[self.channel].copy()

    def command(self, q) -> None:
        self.commands.append(np.asarray(q, dtype=float).copy())

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def hw_rig(tmp_path, monkeypatch):
    monkeypatch.setattr(yam_driver, "YamArm", FakeYamArm)
    FakeYamArm.instances = []
    rig = load_rig()
    map_file = tmp_path / "hw_joint_map.json"
    rig["hardware"]["joint_map_file"] = str(map_file)
    rig["hardware"]["sides"] = ["right"]
    rig["hardware"]["use_hands"] = False
    neutral = np.asarray(rig["arms"]["right"]["neutral_q"], dtype=float)
    jm = JointMap.anchored_at_rest(np.ones(6), np.linspace(-1, 1, 6), neutral)
    save_joint_map(map_file, "right", jm, channel="can0", q_motor_rest=np.linspace(-1, 1, 6))
    FakeYamArm.measured = {rig["arms"]["right"]["can_channel"]: neutral.copy()}
    return rig


def test_active_sides_validation():
    assert active_sides({"hardware": {"sides": ["right"]}}) == ("right",)
    assert active_sides({"hardware": {}}) == ("left", "right")
    with pytest.raises(ValueError):
        active_sides({"hardware": {"sides": ["up"]}})
    with pytest.raises(ValueError):
        active_sides({"hardware": {"sides": []}})


def test_only_wired_sides_open_and_left_commands_drop(hw_rig):
    sink = HardwareSink(hw_rig)
    assert list(sink.arms) == ["right"] and len(FakeYamArm.instances) == 1
    assert sink.hands == {}
    sink.set_arm("left", np.zeros(6))            # dropped, no crash
    sink.set_hand("right", {"thumb": 0.0})       # hands disabled, no crash
    sink.set_arm("right", np.asarray(hw_rig["arms"]["right"]["neutral_q"], dtype=float))
    assert len(sink.arms["right"].commands) == 1


def test_shaper_glides_from_measured_pose(hw_rig):
    neutral = np.asarray(hw_rig["arms"]["right"]["neutral_q"], dtype=float)
    nudged = neutral + 0.05                       # at rest, within the gate tol
    FakeYamArm.measured[hw_rig["arms"]["right"]["can_channel"]] = nudged
    sink = HardwareSink(hw_rig)
    np.testing.assert_allclose(sink.shapers["right"].q, nudged, atol=1e-9)
    sink.set_arm("right", neutral)                # first shaped command starts AT measured
    np.testing.assert_allclose(sink.arms["right"].commands[0], nudged, atol=1e-6)


def test_rest_pose_gate_blocks_and_releases(hw_rig):
    neutral = np.asarray(hw_rig["arms"]["right"]["neutral_q"], dtype=float)
    lifted = neutral.copy()
    lifted[2] += 0.4                              # arm not at rest
    FakeYamArm.measured[hw_rig["arms"]["right"]["can_channel"]] = lifted
    with pytest.raises(RuntimeError, match="REST-POSE GATE"):
        HardwareSink(hw_rig)
    assert FakeYamArm.instances and all(a.closed for a in FakeYamArm.instances)


def test_missing_calibration_refuses_with_instructions(hw_rig, tmp_path):
    hw_rig["hardware"]["joint_map_file"] = str(tmp_path / "absent.json")
    with pytest.raises(RuntimeError, match="hw_bringup"):
        HardwareSink(hw_rig)
