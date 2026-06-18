"""Calibration QUALITY readout (vr/neutral_calib._fit_quality + the live
workspace-clamp signal). The 2026-06-11 floating-arms regression shipped a
self-inconsistent fit that was only diagnosed by forensics on a recording —
these tests pin that a fit now grades itself on the spot, the grade survives
persistence, and a wrong mapping shows up live as sustained clamping."""
from __future__ import annotations

import numpy as np

from bimanual_teleop.config import SIDES, load_rig
from bimanual_teleop.arms.arm_control import ArmController
from bimanual_teleop.vr.frames import ClutchMapper, HandSample, lateral_curve
from bimanual_teleop.vr.neutral_calib import (ROBOT_NEUTRAL_DEFAULT, ROBOT_REST_DEFAULT,
                                              fit_two_pose, load_calibration)

# Clean symmetric capture: spread 0.44 (matches the robot ±0.22 → s_lat 1.0),
# 0.50 raise, 0.45 reach, clap gap 0.08.
POSE_A = {"left": np.array([-0.22, 0.05, 0.45]), "right": np.array([0.22, 0.05, 0.45])}
POSE_B = {"left": np.array([-0.20, -0.45, 0.0]), "right": np.array([0.20, -0.45, 0.0])}
POSE_C = {"left": np.array([-0.04, -0.10, 0.30]), "right": np.array([0.04, -0.10, 0.30])}
RBN = {s: np.asarray(ROBOT_NEUTRAL_DEFAULT[s]) for s in SIDES}
RBR = {s: np.asarray(ROBOT_REST_DEFAULT[s]) for s in SIDES}


def _fit(pa=POSE_A, pb=POSE_B, pc=POSE_C):
    return fit_two_pose(pa, pb, RBN, RBR, pose_c=pc)


def test_clean_fit_grades_good_with_small_residuals():
    res = _fit()
    assert res is not None
    q = res.meta["quality"]
    assert q["grade"] == "good"
    assert q["worst_cm"] < 5.0
    assert q["reasons"] == []
    assert not any(q["clipped"])
    # the held neutral lands on the robot neutral through the real map
    for s in SIDES:
        assert q["residual_cm"]["neutral"][s] < 2.0


def test_asymmetric_capture_grades_check_with_named_reason():
    """One arm held 16 cm higher than the other in the 'symmetric' neutral
    pose: the fit anchors the MEAN, so each side carries ~6 cm of residual —
    the operator must SEE that, not discover it mid-teleop."""
    pa = {"left": POSE_A["left"] + np.array([0.0, 0.16, 0.0]), "right": POSE_A["right"]}
    q = _fit(pa=pa).meta["quality"]
    assert q["grade"] == "check"
    assert any("neutral residual" in r for r in q["reasons"])
    assert q["worst_cm"] > 5.0


def test_badly_clipped_scale_grades_bad():
    """A capture whose reach delta is far too small forces the raw forward
    scale way past SCALE_MAX — the clipped value WILL mis-map reach
    proportionally, and the grade must say so."""
    pb = {s: POSE_B[s] + np.array([0.0, 0.0, 0.29]) for s in SIDES}   # d_fwd = 0.16
    res = _fit(pb=pb)
    assert res is not None
    q = res.meta["quality"]
    assert q["grade"] == "bad"
    assert any("reach scale clipped" in r for r in q["reasons"])
    assert q["clipped"][2]


def test_quality_survives_save_load_and_summary(tmp_path):
    res = _fit()
    assert res.summary()["quality"]["grade"] == "good"
    p = tmp_path / "calib.json"
    res.save(p)
    loaded = load_calibration(p)
    assert loaded is not None
    assert loaded.meta["quality"]["grade"] == "good"
    assert loaded.summary()["quality"]["worst_cm"] == res.meta["quality"]["worst_cm"]


def test_lateral_curve_is_the_mappers_curve():
    """The grader must measure through EXACTLY the runtime lateral map."""
    m = ClutchMapper(np.eye(3), position_mode="absolute", chest_base=[0, 0, 0])
    m.set_calibration([1.4, 1.0, 1.0], [0, 0, 0], lat_ref=0.22, lat_center=0.03,
                      lat_knots=[[0.04, 0.06], [0.22, 0.22]])
    for lat in (-0.3, -0.1, -0.02, 0.0, 0.05, 0.18, 0.4):
        assert m._lat_scaled(lat) == lateral_curve(lat - 0.03, 1.4, 0.22,
                                                   [[0.04, 0.06], [0.22, 0.22]])
    m.set_calibration([1.4, 1.0, 1.0], [0, 0, 0], lat_ref=0.22, lat_center=0.0)
    for lat in (-0.3, -0.05, 0.0, 0.1, 0.5):
        assert m._lat_scaled(lat) == lateral_curve(lat, 1.4, 0.22, None)


# --------------------------------------------------------------------------- #
# live workspace-clamp signal
# --------------------------------------------------------------------------- #
def _hs(p) -> HandSample:
    W = np.eye(4)
    W[:3, 3] = np.asarray(p, dtype=float)
    return HandSample(tracked=True, wrist=W)


def test_clamp_dist_flags_out_of_workspace_targets_live():
    """Glide the (body-frame) hand to a pose whose absolute target sits far
    outside the workspace box: clamp_dist must report the excess while pinned,
    and fall back to ~0 when the hand returns / disengages. This is the live
    symptom of the 2026-06-11 broken fit (targets pinned at the box top)."""
    rig = load_rig()
    ac = ArmController(rig, "right")
    t, dt = 0.0, 1 / 60
    for _ in range(30):                                  # engage + settle inside
        ac.plan(_hs([0.25, 0.0, 0.35]), True, t)
        t += dt
    assert ac.clamp_dist < 0.05
    p = np.array([0.25, 0.0, 0.35])
    hi = np.array([0.25, 1.6, 0.35])                     # 1.6 m above the torso proxy
    for k in range(120):                                 # 2 s glide ≈ 0.8 m/s (honest)
        f = (k + 1) / 120
        ac.plan(_hs((1 - f) * p + f * hi), True, t)
        t += dt
    for _ in range(180):                                 # hold 3 s — engage blend decays
        ac.plan(_hs(hi), True, t)
        t += dt
    assert ac.clamp_dist > 0.3, "pinned target must report its excess"
    for k in range(120):                                 # glide back inside
        f = (k + 1) / 120
        ac.plan(_hs((1 - f) * hi + f * p), True, t)
        t += dt
    for _ in range(120):
        ac.plan(_hs(p), True, t)
        t += dt
    assert ac.clamp_dist < 0.05
    ac.plan(None, False, t)                              # disengage clears the signal
    assert ac.clamp_dist == 0.0
