"""The motor↔model joint map is safety-critical: a wrong sign/offset is a
commanded sweep on real metal. Pin the affine math, the rest-pose anchoring,
the engage gate, and the per-machine persistence."""
from __future__ import annotations

import numpy as np
import pytest

from bimanual_teleop.arms.joint_map import (JointMap, load_joint_map,
                                            map_file_from_rig, rest_pose_gate,
                                            save_joint_map)

NEUTRAL = np.array([3.14, -0.001, 0.305, -0.152, 0.001, 1.571])


def test_round_trip_mixed_signs():
    jm = JointMap(signs=[1, -1, 1, -1, -1, 1], offsets=[0.1, -2.0, 0.4, 0.0, 3.0, -1.0])
    q = np.array([0.3, -1.2, 2.0, 0.7, -0.4, 1.5])
    np.testing.assert_allclose(jm.to_model(jm.to_motor(q)), q, atol=1e-12)
    np.testing.assert_allclose(jm.to_motor(jm.to_model(q)), q, atol=1e-12)


def test_anchoring_makes_rest_map_to_neutral_exactly():
    q_rest_motor = np.array([2.9, 0.4, 3.1, -0.2, 0.05, -1.3])
    for signs in ([1] * 6, [1, -1, -1, 1, -1, 1]):
        jm = JointMap.anchored_at_rest(signs, q_rest_motor, NEUTRAL)
        np.testing.assert_allclose(jm.to_model(q_rest_motor), NEUTRAL, atol=1e-12)
        np.testing.assert_allclose(jm.to_motor(NEUTRAL), q_rest_motor, atol=1e-12)


def test_only_unit_signs_accepted():
    with pytest.raises(ValueError):
        JointMap(signs=[1, 1, 0, 1, 1, 1], offsets=np.zeros(6))
    with pytest.raises(ValueError):
        JointMap(signs=[2, 1, 1, 1, 1, 1], offsets=np.zeros(6))
    with pytest.raises(ValueError):
        JointMap(signs=np.ones(5), offsets=np.zeros(5))


def test_rest_pose_gate_pass_and_fail():
    ok = rest_pose_gate(NEUTRAL + 0.05, NEUTRAL, tol_rad=0.15)
    assert ok.ok and "within" in ok.describe()
    bad = rest_pose_gate(NEUTRAL + np.array([0, 0, 0.3, 0, 0, 0]), NEUTRAL, tol_rad=0.15)
    assert not bad.ok and bad.worst == 2 and "j3" in bad.describe() and "EXCEEDS" in bad.describe()


def test_rest_pose_gate_fails_closed_on_garbage():
    assert not rest_pose_gate([np.nan] * 6, NEUTRAL, 0.15).ok
    assert not rest_pose_gate([0.0] * 5, NEUTRAL, 0.15).ok


def test_wrong_side_on_bus_trips_the_gate_via_j6():
    """left vs right rest differ by π on j6 (palms inward) — the gate must see it."""
    left_neutral = np.array([3.137, -0.004, 0.305, -0.162, -0.003, -1.571])
    gate = rest_pose_gate(left_neutral, NEUTRAL, tol_rad=0.15)
    assert not gate.ok and gate.worst == 5


def test_persistence_merge_and_reload(tmp_path):
    f = tmp_path / "hw_joint_map.json"
    assert load_joint_map(f, "right") is None
    jm_r = JointMap.anchored_at_rest([1, -1, 1, 1, -1, 1], np.linspace(-1, 1, 6), NEUTRAL)
    save_joint_map(f, "right", jm_r, channel="can0", q_motor_rest=np.linspace(-1, 1, 6))
    jm_l = JointMap(np.ones(6), np.zeros(6))
    save_joint_map(f, "left", jm_l, channel="can1", q_motor_rest=np.zeros(6))
    back_r, back_l = load_joint_map(f, "right"), load_joint_map(f, "left")
    np.testing.assert_allclose(back_r.signs, jm_r.signs)
    np.testing.assert_allclose(back_r.offsets, jm_r.offsets, atol=1e-12)
    np.testing.assert_allclose(back_l.offsets, np.zeros(6))


def test_wrap_corrections_normalize_multiturn_readings():
    """p16 encoders legitimately report ±2π·k (metal 2026-06-11: j6 read +311.6°);
    the chain re-anchors offsets so angles land in (−π, π], commands coherent."""
    from bimanual_teleop.arms.yam_chain import wrap_corrections
    pos = np.array([5.4378, -3.0882, 9.7, -7.0, 0.3, np.pi])
    corr = wrap_corrections(pos)
    out = pos - corr
    assert np.all(out > -np.pi - 1e-9) and np.all(out <= np.pi + 1e-9)
    np.testing.assert_allclose(corr / (2 * np.pi), [1, 0, 2, -1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(out[0], 5.4378 - 2 * np.pi, atol=1e-12)
    np.testing.assert_allclose(out[1], -3.0882, atol=1e-12)  # already in range — untouched


def test_map_file_resolution(tmp_path):
    p = map_file_from_rig({"hardware": {"joint_map_file": str(tmp_path / "m.json")}})
    assert p == tmp_path / "m.json"
    rel = map_file_from_rig({"hardware": {"joint_map_file": "config/hw_joint_map.json"}})
    assert rel.is_absolute() and rel.name == "hw_joint_map.json" and rel.parent.name == "config"
