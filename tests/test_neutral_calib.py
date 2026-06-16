"""Operator neutral-pose calibration + pairwise hand-separation guard.

Covers the pure math (fit, separation projection, mapper application), the
capture state machine (gates, hold, cancel), persistence validation, and the
engine integration end-to-end: arms freeze during capture, the fit applies and
persists on completion, and clapping hands can never push the two wrist targets
closer than safety.hand_min_separation.

    uv run pytest tests/test_neutral_calib.py -q
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from bimanual_teleop.config import SIDES, load_rig
from bimanual_teleop.engine import TeleopEngine
from bimanual_teleop.safety.separation import separate_targets
from bimanual_teleop.vr import neutral_calib as nc
from bimanual_teleop.vr.frames import SE3, VRFrame, HandSample, ClutchMapper
from bimanual_teleop.vr.neutral_calib import (
    CalibResult, NeutralPoseCalibration, fit_neutral, load_calibration)


# --------------------------------------------------------------------------- #
# fit math
# --------------------------------------------------------------------------- #
ROBOT_NEUTRAL = {"left": np.array([-0.22, 0.02, 0.46]),
                 "right": np.array([0.22, 0.02, 0.46])}


def test_fit_taller_robot_scales_up_reach():
    """Operator with a shorter reach than the robot's neutral → scale > 1."""
    op = {"left": np.array([-0.18, 0.05, 0.40]), "right": np.array([0.18, 0.05, 0.40])}
    r = fit_neutral(op, ROBOT_NEUTRAL)
    assert r.axis_scale[0] == pytest.approx(0.22 / 0.18, abs=1e-6)      # lateral
    assert r.axis_scale[2] == pytest.approx(0.46 / 0.40, abs=1e-6)      # reach
    assert r.axis_scale[1] == r.axis_scale[2]                           # up shares reach
    # offset: lateral forced to zero; forward fitted exactly by the scale
    assert r.body_offset[0] == 0.0
    assert r.body_offset[2] == pytest.approx(0.0, abs=1e-9)
    # up offset aligns neutrals: rb_u - s*op_u
    assert r.body_offset[1] == pytest.approx(0.02 - (0.46 / 0.40) * 0.05, abs=1e-6)
    # the fitted map sends the operator neutral exactly onto the robot neutral
    for s in SIDES:
        mapped = r.axis_scale * op[s] + r.body_offset
        np.testing.assert_allclose(mapped[1:], ROBOT_NEUTRAL[s][1:], atol=1e-9)


def test_fit_clamps_absurd_scales_and_offsets():
    op = {"left": np.array([-0.04, 0.0, 0.12]), "right": np.array([0.04, 0.0, 0.12])}
    r = fit_neutral(op, ROBOT_NEUTRAL)
    assert np.all(r.axis_scale <= nc.SCALE_MAX + 1e-9)
    assert np.all(r.axis_scale >= nc.SCALE_MIN - 1e-9)
    assert np.all(np.abs(r.body_offset) <= nc.OFFSET_MAX + 1e-9)


def test_fit_asymmetric_operator_keeps_midline():
    """L/R asymmetry in the held pose must average out, never bias one side."""
    op = {"left": np.array([-0.20, 0.04, 0.42]), "right": np.array([0.16, 0.06, 0.38])}
    r = fit_neutral(op, ROBOT_NEUTRAL)
    assert r.body_offset[0] == 0.0


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_save_load_round_trip(tmp_path):
    r = fit_neutral({"left": np.array([-0.18, 0.05, 0.40]),
                     "right": np.array([0.18, 0.05, 0.40])}, ROBOT_NEUTRAL)
    p = tmp_path / "calib.json"
    r.save(p)
    back = load_calibration(p)
    assert back is not None
    np.testing.assert_allclose(back.axis_scale, r.axis_scale, atol=1e-9)
    np.testing.assert_allclose(back.body_offset, r.body_offset, atol=1e-9)


def test_load_rejects_garbage(tmp_path):
    p = tmp_path / "calib.json"
    assert load_calibration(p) is None                       # absent
    p.write_text("not json {")
    assert load_calibration(p) is None                       # corrupt
    p.write_text(json.dumps({"axis_scale": [9.0, 1.0, 1.0], "body_offset": [0, 0, 0]}))
    assert load_calibration(p) is None                       # out-of-range scale
    p.write_text(json.dumps({"axis_scale": [1.0, 1.0, 1.0], "body_offset": [0, 0.9, 0]}))
    assert load_calibration(p) is None                       # out-of-range offset


# --------------------------------------------------------------------------- #
# separation projection
# --------------------------------------------------------------------------- #
def test_separation_noop_when_apart():
    l, r = np.array([0.0, -0.2, 1.0]), np.array([0.0, 0.2, 1.0])
    nl, nr = separate_targets(l, r, 0.12)
    np.testing.assert_allclose(nl, l)
    np.testing.assert_allclose(nr, r)


def test_separation_symmetric_push():
    l, r = np.array([0.0, -0.02, 1.0]), np.array([0.0, 0.02, 1.0])
    nl, nr = separate_targets(l, r, 0.12)
    assert np.linalg.norm(nr - nl) == pytest.approx(0.12, abs=1e-9)
    # symmetric: midpoint preserved, push purely along the connecting line (Y)
    np.testing.assert_allclose((nl + nr) / 2, (l + r) / 2, atol=1e-9)
    assert nl[1] < l[1] and nr[1] > r[1]
    assert nl[0] == l[0] and nl[2] == l[2]


def test_separation_one_sided_push_parks_obstacle():
    l, r = np.array([0.0, -0.02, 1.0]), np.array([0.0, 0.02, 1.0])
    nl, nr = separate_targets(l, r, 0.12, move_left=False)
    np.testing.assert_allclose(nl, l)                        # parked arm untouched
    assert np.linalg.norm(nr - nl) == pytest.approx(0.12, abs=1e-9)


def test_separation_degenerate_uses_lateral_axis():
    p = np.array([0.1, 0.0, 1.0])
    nl, nr = separate_targets(p, p.copy(), 0.12)
    assert np.linalg.norm(nr - nl) == pytest.approx(0.12, abs=1e-9)
    assert nl[1] < nr[1]                                     # left pushed to its own (−Y) side


def test_separation_full_3d_line():
    """Push happens along the actual connecting line, not just Y."""
    l = np.array([0.00, -0.03, 1.00])
    r = np.array([0.04, 0.03, 1.05])
    nl, nr = separate_targets(l, r, 0.20)
    assert np.linalg.norm(nr - nl) == pytest.approx(0.20, abs=1e-9)
    d0 = (r - l) / np.linalg.norm(r - l)
    d1 = (nr - nl) / np.linalg.norm(nr - nl)
    np.testing.assert_allclose(d0, d1, atol=1e-9)


# --------------------------------------------------------------------------- #
# mapper application
# --------------------------------------------------------------------------- #
def test_mapper_applies_axis_scale_and_offset():
    R = np.eye(3)
    m = ClutchMapper(R, position_mode="absolute", chest_base=np.zeros(3),
                     orientation_mode="relative")
    m.set_calibration(np.array([1.5, 1.2, 1.2]), np.array([0.0, -0.03, 0.02]))
    ctrl = SE3.from_translation(np.array([0.1, 0.2, 0.3]))
    expected = np.array([1.5, 1.2, 1.2]) * np.array([0.1, 0.2, 0.3]) + np.array([0.0, -0.03, 0.02])
    np.testing.assert_allclose(m._p_abs(ctrl), expected, atol=1e-12)


def test_mapper_set_calibration_releases_anchor():
    m = ClutchMapper(np.eye(3), position_mode="absolute", chest_base=np.zeros(3),
                     orientation_mode="relative")
    ctrl = SE3.from_translation(np.zeros(3))
    m.engage(ctrl, SE3.from_translation(np.zeros(3)), 0.0)
    assert m.engaged
    m.set_calibration(np.ones(3), np.zeros(3))
    assert not m.engaged                                     # glide path re-engages next tick


# --------------------------------------------------------------------------- #
# capture state machine
# --------------------------------------------------------------------------- #
def _rig():
    return load_rig()


def _drive(npc: NeutralPoseCalibration, w_left, w_right, t0, t1, hz=30.0):
    t = t0
    while t <= t1:
        npc.tick({"left": None if w_left is None else np.asarray(w_left, float),
                  "right": None if w_right is None else np.asarray(w_right, float)}, t)
        if not npc.active:
            return t
        t += 1.0 / hz
    return t


def test_capture_completes_on_still_extended_pose():
    npc = NeutralPoseCalibration(_rig())
    npc.start(0.0)
    t_end = _drive(npc, [-0.18, 0.05, 0.40], [0.18, 0.05, 0.40], 0.0, 10.0)
    assert npc.phase == "done" and npc.result is not None
    assert t_end < 5.0                                       # window + hold, not the timeout
    assert npc.result.axis_scale[2] == pytest.approx(0.46 / 0.40, rel=1e-3)


def test_capture_rejects_arms_down_and_crossed():
    npc = NeutralPoseCalibration(_rig())
    npc.start(0.0)
    _drive(npc, [-0.20, -0.45, 0.05], [0.20, -0.45, 0.05], 0.0, 6.0)   # ragdoll hang
    assert npc.active and npc.phase == "wait"
    _drive(npc, [0.10, 0.0, 0.40], [-0.10, 0.0, 0.40], 6.0, 12.0)      # crossed hands
    assert npc.active and npc.phase == "wait"


def test_capture_waits_for_both_hands():
    npc = NeutralPoseCalibration(_rig())
    npc.start(0.0)
    _drive(npc, [-0.18, 0.05, 0.40], None, 0.0, 6.0)
    assert npc.active and npc.phase == "wait"
    st = npc.status(6.0)
    assert st["left"] and not st["right"]


def test_capture_motion_resets_hold():
    npc = NeutralPoseCalibration(_rig())
    npc.start(0.0)
    # still for 1.5 s (less than HOLD_S) …
    _drive(npc, [-0.18, 0.05, 0.40], [0.18, 0.05, 0.40], 0.0, 1.5)
    assert npc.active
    # … then a SUSTAINED 8 cm shift (a single-sample glitch is tolerated by
    # design — the window std absorbs it; real motion must reset the hold)
    _drive(npc, [-0.18, 0.13, 0.40], [0.18, 0.05, 0.40], 1.5 + 1 / 30, 1.8)
    assert npc._hold_t0 is None                    # mixed window → not still → hold reset
    assert npc.active


def test_capture_timeout_cancels():
    npc = NeutralPoseCalibration(_rig())
    npc.start(0.0)
    npc.tick({"left": None, "right": None}, nc.TIMEOUT_S + 1.0)
    assert not npc.active and npc.phase == "cancelled"
    assert "timed out" in npc.status(nc.TIMEOUT_S + 1.0)["msg"]


# --------------------------------------------------------------------------- #
# engine integration
# --------------------------------------------------------------------------- #
class DummySink:
    def __init__(self):
        self.arm = {}
        self.hand = {}

    def set_arm(self, side, q):
        self.arm[side] = np.asarray(q, dtype=float).copy()

    def set_hand(self, side, joints):
        self.hand[side] = dict(joints)


def _frame_with_wrist_body(w_by_side: dict, t: float) -> VRFrame:
    """Build a raw VRFrame whose body-relative wrist positions equal w_by_side.
    Head at origin/identity: op_axes = [x, y, -z], torso at [0, -0.35, 0]."""
    op_axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    torso_world = np.array([0.0, -0.35, 0.0])
    hands = {}
    for s, w in w_by_side.items():
        if w is None:
            continue
        W = np.eye(4)
        W[:3, 3] = torso_world + op_axes @ np.asarray(w, dtype=float)
        hands[s] = HandSample(tracked=True, wrist=W, landmarks=None, pinch=0.0)
    return VRFrame(stamp=t, head=np.eye(4), hands=hands)


def test_engine_capture_freezes_arms_applies_and_persists(tmp_path):
    rig = load_rig()
    calib_path = tmp_path / "operator_calib.json"
    rig["mapping"]["calib_file"] = str(calib_path)
    rig["mapping"]["swap_sides"] = False   # this test pins per-side capture, not the operator swap
    sink = DummySink()
    eng = TeleopEngine(rig, sink)
    assert eng.calib_summary is None
    q_before = {s: eng.arm[s].ik.q.copy() for s in SIDES}

    eng.request_calibration()
    w = {"left": [-0.18, 0.05, 0.40], "right": [0.18, 0.05, 0.40]}
    t, dt = 0.0, 1.0 / 30.0
    while t < 10.0:
        eng.tick(_frame_with_wrist_body(w, t), {"left": True, "right": True}, t)
        if eng.calib_summary is not None:
            break
        # while capturing, the arms hold their pose exactly
        for s in SIDES:
            np.testing.assert_allclose(sink.arm[s], q_before[s], atol=1e-12)
        t += dt
    assert eng.calib_summary is not None, "capture never completed"
    assert calib_path.exists()
    for s in SIDES:
        assert eng.arm[s].mapper.axis_scale[2] == pytest.approx(0.46 / 0.40, rel=1e-3)
    # banner: done message present, then fades after 2.5 s of normal ticks
    assert eng.calib_status and eng.calib_status["phase"] == "done"
    for _ in range(int(3.0 / dt)):
        t += dt
        eng.tick(_frame_with_wrist_body(w, t), {"left": True, "right": True}, t)
    assert eng.calib_status is None


def test_engine_autoloads_for_live_transport_only(tmp_path):
    calib_path = tmp_path / "operator_calib.json"
    fit_neutral({"left": np.array([-0.18, 0.05, 0.40]),
                 "right": np.array([0.18, 0.05, 0.40])}, ROBOT_NEUTRAL).save(calib_path)
    rig = load_rig()
    rig["mapping"]["calib_file"] = str(calib_path)
    rig["vr"]["transport"] = "fake"
    assert TeleopEngine(rig, DummySink()).calib_summary is None      # gate stays deterministic
    rig["vr"]["transport"] = "orbit"
    eng = TeleopEngine(rig, DummySink())
    assert eng.calib_summary is not None                             # live session restores fit
    eng.request_calibration_clear()
    eng.tick(None, {}, 0.0)
    assert eng.calib_summary is None
    assert not calib_path.exists()
    for s in SIDES:
        np.testing.assert_allclose(eng.arm[s].mapper.axis_scale, np.ones(3))


def test_engine_clap_respects_min_separation():
    rig = load_rig()
    rig["vr"]["transport"] = "fake"
    d_min = float(rig["safety"]["hand_min_separation"])
    sink = DummySink()
    eng = TeleopEngine(rig, sink)
    # hands clapped together right in front of the chest
    w = {"left": [-0.01, -0.10, 0.35], "right": [0.01, -0.10, 0.35]}
    t, dt = 0.0, 1.0 / 60.0
    for _ in range(240):                                     # 4 s — well past the engage glide
        eng.tick(_frame_with_wrist_body(w, t), {"left": True, "right": True}, t)
        t += dt
    from bimanual_teleop.safety.separation import closest_points_segments
    cap_len = float(rig["safety"]["hand_capsule_len"])
    pw, tip = {}, {}
    for s in SIDES:
        ac = eng.arm[s]
        assert ac.cmd_pos is not None
        pw[s] = ac.base_R @ ac.cmd_pos + ac.base_pos
        tip[s] = pw[s] + ac.fingers_dir_world(ac.cmd_R) * cap_len
    # the guard protects the whole HAND CAPSULE (wrist → fingertips)
    cl, cr = closest_points_segments(pw["left"], tip["left"], pw["right"], tip["right"])
    gap = float(np.linalg.norm(cr - cl))
    assert gap >= d_min - 1e-6, f"hand capsules only {gap*100:.1f} cm apart"


def test_engine_clap_one_engaged_vs_parked():
    rig = load_rig()
    d_min = float(rig["safety"]["hand_min_separation"])
    eng = TeleopEngine(rig, DummySink())
    parked = eng.arm["left"].wrist_world()
    # drive the right hand INTO the parked left wrist (its body-coords position)
    from bimanual_teleop.vr.calibrate import W_AXES
    anchor = None
    # recover the anchor the same way ArmController built chest_base
    ac = eng.arm["right"]
    anchor = ac.base_R @ ac.mapper.chest + ac.base_pos
    w_right = (W_AXES.T @ (parked - anchor)).tolist()
    t, dt = 0.0, 1.0 / 60.0
    for _ in range(240):
        eng.tick(_frame_with_wrist_body({"right": w_right}, t), {"right": True}, t)
        t += dt
    from bimanual_teleop.safety.separation import closest_points_segments
    cap_len = float(rig["safety"]["hand_capsule_len"])
    ac = eng.arm["right"]
    pw_r = ac.base_R @ ac.cmd_pos + ac.base_pos
    tip_r = pw_r + ac.fingers_dir_world(ac.cmd_R) * cap_len
    acl = eng.arm["left"]
    tip_l = parked + acl.fingers_dir_world() * cap_len
    cl, cr = closest_points_segments(parked, tip_l, pw_r, tip_r)
    gap = float(np.linalg.norm(cr - cl))
    assert gap >= d_min - 1e-6
