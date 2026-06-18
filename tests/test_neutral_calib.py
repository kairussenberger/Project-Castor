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
    CalibResult, NeutralPoseCalibration, fit_two_pose, load_calibration)


# --------------------------------------------------------------------------- #
# fit math (two-pose: anchor- and head-yaw-proof)
# --------------------------------------------------------------------------- #
ROBOT_NEUTRAL = {"left": np.array([-0.22, 0.02, 0.46]),
                 "right": np.array([0.22, 0.02, 0.46])}
ROBOT_REST = {"left": np.array([-0.221, -0.437, -0.032]),
              "right": np.array([0.222, -0.440, 0.032])}
POSE_A = {"left": np.array([-0.18, 0.10, 0.42]), "right": np.array([0.18, 0.10, 0.42])}
POSE_B = {"left": np.array([-0.16, -0.45, 0.05]), "right": np.array([0.16, -0.45, 0.05])}
POSE_C = {"left": np.array([-0.04, -0.05, 0.30]), "right": np.array([0.04, -0.05, 0.30])}


def _fit(pa=POSE_A, pb=POSE_B, pc=None):
    return fit_two_pose(pa, pb, ROBOT_NEUTRAL, ROBOT_REST, pose_c=pc)


def test_fit_two_pose_scales_from_differences():
    r = _fit()
    assert r is not None
    assert r.axis_scale[0] == pytest.approx(0.44 / 0.36, rel=1e-6)        # spread ratio
    # up: robot (0.02-(-0.4385))=0.4585 vs operator 0.55; fwd: side-avg 0.46 vs 0.37
    assert r.axis_scale[1] == pytest.approx(0.4585 / 0.55, rel=1e-3)
    assert r.axis_scale[2] == pytest.approx(0.46 / 0.37, rel=1e-3)
    assert r.lat_ref == pytest.approx(0.18, abs=1e-9)
    assert r.lat_center == pytest.approx(0.0, abs=1e-9)
    # pose A maps exactly onto the robot neutral (up/fwd via offset)
    for s in SIDES:
        mapped = r.axis_scale * POSE_A[s] + r.body_offset
        np.testing.assert_allclose(mapped[1:], ROBOT_NEUTRAL[s][1:], atol=1e-9)


def test_fit_two_pose_cancels_recenter_anchor():
    """THE regression: a recenter/desk-start shifts every measurement by one
    constant vector (measured 0.5 m). Scales must be identical and the mapped
    neutral must still land on the robot neutral."""
    delta = np.array([0.12, -0.50, 0.21])
    r0 = _fit()
    r = _fit({s: POSE_A[s] + delta for s in SIDES}, {s: POSE_B[s] + delta for s in SIDES})
    assert r is not None
    np.testing.assert_allclose(r.axis_scale, r0.axis_scale, atol=1e-9)
    assert r.lat_center == pytest.approx(delta[0], abs=1e-9)              # midline absorbed
    for s in SIDES:
        m = (POSE_A[s] + delta)
        mapped_up_fwd = r.axis_scale[1:] * m[1:] + r.body_offset[1:]
        np.testing.assert_allclose(mapped_up_fwd, ROBOT_NEUTRAL[s][1:], atol=1e-9)
        lat = r.axis_scale[0] * (m[0] - r.lat_center)                     # mapper lat path
        assert lat == pytest.approx(ROBOT_NEUTRAL[s][0], abs=1e-9)


def test_fit_two_pose_head_yaw_invariant():
    """The operator watches the dashboard — the body frame is yawed vs the
    arms. Forward comes from the A−B delta, so the fit must not change."""
    yaw = np.radians(50.0)
    c, s_ = np.cos(yaw), np.sin(yaw)

    def yawed(w):
        x, u, f = w
        return np.array([c * x + s_ * f, u, -s_ * x + c * f])

    r0 = _fit()
    r = _fit({s: yawed(POSE_A[s]) for s in SIDES}, {s: yawed(POSE_B[s]) for s in SIDES})
    assert r is not None
    np.testing.assert_allclose(r.axis_scale, r0.axis_scale, atol=1e-6)
    assert r.lat_ref == pytest.approx(r0.lat_ref, abs=1e-6)


def test_fit_two_pose_rejects_degenerate():
    assert _fit(POSE_A, POSE_A) is None                       # no A-B delta
    bad_spread = {"left": np.array([-0.05, 0.1, 0.42]), "right": np.array([0.05, 0.1, 0.42])}
    assert _fit(bad_spread, POSE_B) is None                   # hands too close in A


def test_fit_pose_c_anchors_contact_and_midline():
    """Pose C (palms together): the measured clap gap maps to the robot contact
    gap, the midline is measured where the palms meet, and the curve still hits
    the robot spread at the pose-A spread."""
    r = _fit(pc=POSE_C)
    assert r is not None and r.lat_knots is not None
    (xc, yc), (xa, ya) = r.lat_knots
    assert xc == pytest.approx(0.04, abs=1e-9)               # operator clap half-gap
    assert yc == pytest.approx(0.06, abs=1e-9)               # robot contact half-gap
    assert xa == pytest.approx(0.18, abs=1e-9)
    assert ya == pytest.approx(0.22, abs=1e-9)
    assert r.lat_center == pytest.approx(0.0, abs=1e-9)
    assert r.forward_body is not None


def test_fit_pose_c_anchor_shift_still_cancels():
    delta = np.array([0.12, -0.50, 0.21])
    r = _fit({s: POSE_A[s] + delta for s in SIDES}, {s: POSE_B[s] + delta for s in SIDES},
             {s: POSE_C[s] + delta for s in SIDES})
    assert r is not None and r.lat_knots is not None
    assert r.lat_center == pytest.approx(delta[0], abs=1e-9)
    (xc, yc), (xa, ya) = r.lat_knots
    assert xc == pytest.approx(0.04, abs=1e-9) and xa == pytest.approx(0.18, abs=1e-9)


def test_fit_absorbs_meter_scale_stream_anchor_mismatch():
    """THE 2026-06-11 regression, real captured poses: ORBIT's wrist and head
    streams recenter-anchor INDEPENDENTLY — the wrists measured ~1.35 m above
    the torso proxy for the whole session. The old ±0.8 m offset clip silently
    truncated the fitted up-offset (needed −2.19 m) and every runtime target
    landed ~1.4 m above the chest: arms pinned at the workspace ceiling. The
    offset must absorb the anchor EXACTLY — pose A maps onto the robot
    neutral, unclipped."""
    pa = {"left": np.array([-0.189, 1.492, 0.1353]),
          "right": np.array([0.196, 1.5033, 0.1543])}
    pb = {"left": np.array([-0.2541, 1.1824, -0.2473]),
          "right": np.array([0.2611, 1.1917, -0.2302])}
    pc = {"left": np.array([-0.0351, 1.3494, -0.1041]),
          "right": np.array([0.0498, 1.3484, -0.0981])}
    r = fit_two_pose(pa, pb, ROBOT_NEUTRAL, ROBOT_REST, pose_c=pc)
    assert r is not None
    assert r.body_offset[1] < -1.5                       # NOT clipped to −0.8
    A = {s: np.array(r.meta["pose_a"][s]) for s in SIDES}    # fit-frame pose A
    # The MEAN of pose A maps onto the robot neutral exactly; per-side
    # residuals are the operator's own asymmetry (~1 cm here), not anchor.
    mapped = {s: r.axis_scale * A[s] + r.body_offset for s in SIDES}
    mean_mapped = np.mean([mapped[s] for s in SIDES], axis=0)
    mean_neutral = np.mean([ROBOT_NEUTRAL[s] for s in SIDES], axis=0)
    np.testing.assert_allclose(mean_mapped[1:], mean_neutral[1:], atol=1e-3)
    for s in SIDES:
        np.testing.assert_allclose(mapped[s][1:], ROBOT_NEUTRAL[s][1:], atol=0.02)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_save_load_round_trip(tmp_path):
    r = _fit()
    p = tmp_path / "calib.json"
    r.save(p)
    back = load_calibration(p)
    assert back is not None
    np.testing.assert_allclose(back.axis_scale, r.axis_scale, atol=1e-9)
    np.testing.assert_allclose(back.body_offset, r.body_offset, atol=1e-9)
    assert back.lat_ref == pytest.approx(r.lat_ref, abs=1e-9)
    assert back.lat_center == pytest.approx(r.lat_center, abs=1e-9)


def test_load_rejects_garbage(tmp_path):
    p = tmp_path / "calib.json"
    assert load_calibration(p) is None                       # absent
    p.write_text("not json {")
    assert load_calibration(p) is None                       # corrupt
    p.write_text(json.dumps({"axis_scale": [9.0, 1.0, 1.0], "body_offset": [0, 0, 0]}))
    assert load_calibration(p) is None                       # out-of-range scale
    p.write_text(json.dumps({"axis_scale": [1.0, 1.0, 1.0], "body_offset": [0, 10.5, 0]}))
    assert load_calibration(p) is None                       # out-of-range offset
    # a real wrist↔head stream anchor mismatch (measured −2.19 m) must load
    p.write_text(json.dumps({"axis_scale": [1.0, 1.0, 1.0], "body_offset": [0, -2.19, 0]}))
    assert load_calibration(p) is not None


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
# capture state machine (two-pose)
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


def _drive_two_pose(npc, pa=POSE_A, pb=POSE_B, pc=POSE_C, t0=0.0):
    # Capture order: rest → clap → extended-forward LAST.
    t = _drive(npc, pb["left"], pb["right"], t0, t0 + 8.0)
    if npc.phase != "wait_clap":
        return t
    t = _drive(npc, pc["left"], pc["right"], t + 0.1, t + 12.0)
    if npc.phase != "wait_fwd":
        return t
    return _drive(npc, pa["left"], pa["right"], t + 0.1, t + 24.0)


def test_capture_completes_three_poses():
    npc = NeutralPoseCalibration(_rig())
    npc.start(0.0)
    t_end = _drive_two_pose(npc)
    assert npc.phase == "done" and npc.result is not None
    assert t_end < 30.0
    assert npc.result.axis_scale[0] == pytest.approx(0.44 / 0.36, rel=1e-3)
    assert npc.result.lat_knots is not None


def test_capture_with_recentered_anchor_completes():
    """The measured failure: a desk-start shifted everything ~0.5 m and the old
    absolute pose gate refused forever. The pose gates are relative."""
    delta = np.array([0.1, -0.5, 0.2])
    npc = NeutralPoseCalibration(_rig())
    npc.start(0.0)
    _drive(npc, POSE_B["left"] + delta, POSE_B["right"] + delta, 0.0, 8.0)
    assert npc.phase == "wait_clap", "rest pose refused under anchor shift"
    _drive(npc, POSE_C["left"] + delta, POSE_C["right"] + delta, 8.1, 20.0)
    assert npc.phase == "wait_fwd"
    _drive(npc, POSE_A["left"] + delta, POSE_A["right"] + delta, 20.2, 32.0)
    assert npc.phase == "done" and npc.result is not None
    assert npc.result.lat_center == pytest.approx(0.1, abs=1e-3)


def test_capture_pose_fwd_requires_arm_raise():
    """Holding the rest pose again at step 3/3 must not complete — the
    extended pose needs the wrists RAISED ≥ DROP_MIN above the rest pose."""
    npc = NeutralPoseCalibration(_rig())
    npc.start(0.0)
    _drive(npc, POSE_B["left"], POSE_B["right"], 0.0, 8.0)
    assert npc.phase == "wait_clap"
    _drive(npc, POSE_C["left"], POSE_C["right"], 8.1, 20.0)
    assert npc.phase == "wait_fwd"
    _drive(npc, POSE_B["left"], POSE_B["right"], 20.2, 28.0)
    assert npc.active and npc.phase == "wait_fwd"            # still waiting for the raise


def test_capture_rejects_crossed_or_narrow_hands():
    npc = NeutralPoseCalibration(_rig())
    npc.start(0.0)
    _drive(npc, [0.02, 0.1, 0.40], [-0.02, 0.1, 0.40], 0.0, 6.0)   # crossed/narrow
    assert npc.active and npc.phase == "wait_rest"


def test_capture_waits_for_both_hands():
    npc = NeutralPoseCalibration(_rig())
    npc.start(0.0)
    _drive(npc, POSE_B["left"], None, 0.0, 6.0)
    assert npc.active and npc.phase == "wait_rest"
    st = npc.status(6.0)
    assert st["left"] and not st["right"]


def test_capture_motion_resets_hold():
    npc = NeutralPoseCalibration(_rig())
    npc.start(0.0)
    _drive(npc, POSE_B["left"], POSE_B["right"], 0.0, 1.5)
    assert npc.active
    _drive(npc, POSE_B["left"] + [0, 0.08, 0], POSE_B["right"], 1.5 + 1 / 30, 1.8)
    assert npc._hold_t0 is None
    assert npc.active


def test_capture_timeout_cancels():
    npc = NeutralPoseCalibration(_rig())
    npc.start(0.0)
    npc.tick({"left": None, "right": None}, nc.TIMEOUT_S + 1.0)
    assert not npc.active and npc.phase == "cancelled"
    assert "timed out" in npc.status(nc.TIMEOUT_S + 1.0)["msg"]


def test_status_prompts_walk_the_operator():
    npc = NeutralPoseCalibration(_rig())
    npc.start(0.0)
    npc.tick({"left": POSE_B["left"], "right": POSE_B["right"]}, 0.0)
    assert "1/3" in npc.status(0.0)["msg"] or "RELAX" in npc.status(0.0)["msg"]
    t = _drive(npc, POSE_B["left"], POSE_B["right"], 0.0, 8.0)
    assert npc.phase == "wait_clap"
    assert "2/3" in npc.status(t)["msg"]
    t = _drive(npc, POSE_C["left"], POSE_C["right"], t + 0.1, t + 12.0)
    assert npc.phase == "wait_fwd"
    assert "3/3" in npc.status(t)["msg"]


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
    # This fixture TELEPORTS between poses (real operators glide); the anchor
    # guard would rightly read that as an anchor event. Capture mechanics are
    # the subject here — guard/capture interplay is pinned in test_anchor_guard.
    rig["safety"]["anchor_guard"] = {"enabled": False}
    sink = DummySink()
    eng = TeleopEngine(rig, sink)
    assert eng.calib_summary is None
    q_before = {s: eng.arm[s].ik.q.copy() for s in SIDES}

    eng.request_calibration()
    wa = {"left": POSE_A["left"].tolist(), "right": POSE_A["right"].tolist()}
    wb = {"left": POSE_B["left"].tolist(), "right": POSE_B["right"].tolist()}
    wc = {"left": POSE_C["left"].tolist(), "right": POSE_C["right"].tolist()}
    t, dt = 0.0, 1.0 / 30.0
    while t < 30.0:
        ph = eng.neutral.phase
        w = wb if (ph == "wait_rest" or t < 0.1) else (wc if ph == "wait_clap" else wa)
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
        assert eng.arm[s].mapper.axis_scale[0] == pytest.approx(0.44 / 0.36, rel=1e-3)
    # banner: done message present, then fades after 2.5 s of normal ticks
    assert eng.calib_status and eng.calib_status["phase"] == "done"
    for _ in range(int(3.0 / dt)):
        t += dt
        eng.tick(_frame_with_wrist_body(wb, t), {"left": True, "right": True}, t)
    assert eng.calib_status is None


def test_engine_yaw_latch_skips_degenerate_head_samples():
    """A NaN warm-up or looking-straight-down head must NOT latch the session
    yaw frame — a degenerate latch poisons every body-relative sample after it
    (a real replay's first head sample is NaN; a real session's first sample
    can be the operator looking down at the desk). Until a sane head arrives
    the engine fails closed: samples come back untracked."""
    eng = TeleopEngine(load_rig(), DummySink())
    assert not TeleopEngine._head_latchable(np.full((4, 4), np.nan))
    down = np.eye(4)        # camera −Z pointing straight down → no horizontal fwd
    down[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float).T
    assert not TeleopEngine._head_latchable(down)
    hs = HandSample(tracked=True, wrist=np.eye(4), landmarks=None, pinch=0.0)
    out = eng._arm_hand_sample(hs, VRFrame(stamp=0.0, head=down,
                                           hands={"left": hs, "right": hs}))
    assert out is not None and not out.tracked           # fail closed, no latch
    assert eng._yaw_R is None
    out = eng._arm_hand_sample(hs, VRFrame(stamp=0.1, head=np.eye(4),
                                           hands={"left": hs, "right": hs}))
    assert eng._yaw_R is not None and out.tracked        # sane head latches


def test_engine_calibration_required_locks_live_transports(tmp_path):
    """SAFETY: with require_calibration (default), a live transport must NOT
    auto-load an old fit and must NOT follow hands until an in-session
    calibration completes — a fresh ORBIT recenter anchor invalidates any
    previous absolute fit."""
    calib_path = tmp_path / "operator_calib.json"
    _fit().save(calib_path)
    rig = load_rig()
    rig["mapping"]["calib_file"] = str(calib_path)
    rig["vr"]["transport"] = "fake"
    assert TeleopEngine(rig, DummySink()).calib_summary is None      # gate stays deterministic
    rig["vr"]["transport"] = "orbit"
    eng = TeleopEngine(rig, DummySink())
    assert eng.calib_summary is None                                 # no stale auto-load
    assert eng.follow_locked                                         # arms locked
    q0 = {s: eng.arm[s].ik.q.copy() for s in SIDES}
    t = 0.0
    for _ in range(120):                                             # hands wave; arms must hold
        t += 1 / 60
        eng.tick(_frame_with_wrist_body({"left": [-0.2, 0.1 * np.sin(t * 3), 0.4],
                                         "right": [0.2, 0.1 * np.cos(t * 3), 0.4]}, t),
                 {"left": True, "right": True}, t)
    for s in SIDES:
        assert float(np.linalg.norm(eng.arm[s].ik.q - q0[s])) < 1e-9, "locked arm moved"
    # legacy opt-out still auto-loads
    rig2 = load_rig()
    rig2["mapping"]["calib_file"] = str(calib_path)
    rig2["vr"]["transport"] = "orbit"
    rig2["vr"]["require_calibration"] = False
    eng2 = TeleopEngine(rig2, DummySink())
    assert eng2.calib_summary is not None and not eng2.follow_locked
    eng2.request_calibration_clear()
    eng2.tick(None, {}, 0.0)
    assert eng2.calib_summary is None
    assert not calib_path.exists()


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

