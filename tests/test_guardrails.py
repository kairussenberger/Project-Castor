"""Motion guardrails: target governor (speed cap, teleport rejection, angular
rate cap), the in-loop joint command shaper, and the hand-capsule separation.

Sized from a real session (recordings/live_0611_130501.npz): operator motion
peaks ≈2 m/s, tracking glitches 58–65 m/s, and the left-wrist-roll singularity
drove j4/j6 at 20–25 rad/s before the shaper entered the sim path.

    uv run pytest tests/test_guardrails.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from bimanual_teleop.arms.arm_control import ArmController
from bimanual_teleop.config import SIDES, load_rig
from bimanual_teleop.safety.separation import closest_points_segments, separate_capsules
from bimanual_teleop.vr.frames import HandSample

DT = 1.0 / 120.0


def _hs(p, R=None) -> HandSample:
    W = np.eye(4)
    if R is not None:
        W[:3, :3] = R
    W[:3, 3] = np.asarray(p, dtype=float)
    return HandSample(tracked=True, wrist=W)


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# --------------------------------------------------------------------------- #
# target governor
# --------------------------------------------------------------------------- #
def test_governor_caps_target_speed():
    rig = load_rig()
    ac = ArmController(rig, "right")
    v_max = float(rig["safety"]["target_speed_max"])
    t, p = 0.0, np.array([0.25, 0.0, 0.30])
    prev = None
    for i in range(240):                       # 2 s sweep at ~1.8 m/s requested
        p = p + np.array([0.0, 0.015, 0.0])    # 1.8 m/s in body axes
        plan = ac.plan(_hs(p), True, t)
        assert plan is not None
        if prev is not None:
            step = float(np.linalg.norm(plan["pw"] - prev))
            assert step <= v_max * DT * 1.05 + 1e-9, f"target moved {step/DT:.2f} m/s"
        prev = plan["pw"]
        t += DT

def test_governor_rejects_teleport_and_recovers():
    rig = load_rig()
    ac = ArmController(rig, "right")
    t = 0.0
    for _ in range(60):                        # settle engaged
        ac.plan(_hs([0.25, 0.0, 0.30]), True, t)
        t += DT
    assert ac.mapper.engaged
    # 0.6 m in one tick = 72 m/s — a tracking glitch, not a movement
    plan = ac.plan(_hs([0.25, 0.6, 0.30]), True, t)
    assert plan is None                        # the movement does not happen
    assert not ac.mapper.engaged               # re-anchored → glide on next tick
    t += DT
    plan = ac.plan(_hs([0.25, 0.6, 0.30]), True, t)
    assert plan is not None                    # following resumes (snap-free)


def test_governor_caps_angular_speed():
    rig = load_rig()
    ac = ArmController(rig, "right")
    w_max = float(rig["safety"]["target_ang_speed_max"])
    t = 0.0
    for _ in range(60):
        ac.plan(_hs([0.25, 0.0, 0.30]), True, t)
        t += DT
    prev = None
    for i in range(120):                       # demand a fast 180°/s… × 4 roll
        R = _rotx(12.0 * (i + 1) * DT)         # 12 rad/s requested
        plan = ac.plan(_hs([0.25, 0.0, 0.30], R), True, t)
        R_w = ac.base_R @ plan["R"].as_matrix()
        if prev is not None:
            d = prev.T @ R_w
            ang = float(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1)))
            assert ang <= w_max * DT * 1.05 + 1e-6, f"attitude turned {ang/DT:.1f} rad/s"
        prev = R_w
        t += DT


def test_governor_bounds_target_acceleration():
    """The smoothing contract: the governed target's velocity may never JUMP —
    it ramps at ≤ target_accel_max (S-curve onset/landing), so a fast operator
    sweep leaves as the smooth second-order response, not the old
    rectangle-velocity 'blocky' glide."""
    rig = load_rig()
    ac = ArmController(rig, "right")
    a_max = float(rig["safety"]["target_accel_max"])
    t = 0.0
    for _ in range(240):                       # settle the engage glide fully
        ac.plan(_hs([0.25, 0.0, 0.30]), True, t)
        t += DT
    p, prev_p, prev_v = np.array([0.25, 0.0, 0.30]), None, None
    first_tick_speed = None
    for i in range(360):                       # 1.8 m/s sweep, then a hard stop
        if i < 240:
            p = p + np.array([0.0, 0.015, 0.0])
        plan = ac.plan(_hs(p), True, t)
        if prev_p is not None:
            v = (plan["pw"] - prev_p) / DT
            if first_tick_speed is None:
                first_tick_speed = float(np.linalg.norm(v))
            if prev_v is not None:
                dv = float(np.linalg.norm(v - prev_v)) / DT
                assert dv <= a_max * 1.10 + 1e-6, \
                    f"target accelerated at {dv:.1f} m/s² on tick {i}"
            prev_v = v
        prev_p = plan["pw"]
        t += DT
    # the sweep starts from rest: the very first tick must NOT already move at
    # full speed (that instant jump is exactly the old blockiness)
    assert first_tick_speed <= a_max * DT * 2.0, \
        f"first tick already at {first_tick_speed:.2f} m/s — velocity slammed, not ramped"


def test_governor_bounds_angular_acceleration():
    rig = load_rig()
    ac = ArmController(rig, "right")
    aw_max = float(rig["safety"]["target_ang_accel_max"])
    t = 0.0
    for _ in range(240):
        ac.plan(_hs([0.25, 0.0, 0.30]), True, t)
        t += DT
    prev_R, prev_w = None, None
    for i in range(240):                       # 12 rad/s roll demand, then hold
        R = _rotx(min(12.0 * (i + 1) * DT, 3.0))
        plan = ac.plan(_hs([0.25, 0.0, 0.30], R), True, t)
        R_w = ac.base_R @ plan["R"].as_matrix()
        if prev_R is not None:
            d = prev_R.T @ R_w
            ang = float(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1)))
            w = ang / DT
            if prev_w is not None:
                assert abs(w - prev_w) / DT <= aw_max * 1.15 + 1e-6, \
                    f"attitude rate jumped {abs(w - prev_w)/DT:.0f} rad/s² on tick {i}"
            prev_w = w
        prev_R = R_w
        t += DT


def test_governed_motion_is_frame_rate_independent():
    """The caps are m/s and m/s², NOT metres-per-frame: the same operator
    trajectory sampled at 120 Hz and 40 Hz must govern to the same target —
    the irl frame moves identically whatever rate the loop achieves."""
    rig = load_rig()

    def run(hz: int) -> np.ndarray:
        ac = ArmController(rig, "right")
        n, t, plan = int(3.0 * hz), 0.0, None
        for i in range(n):
            t = i / hz
            # smooth 0.2 m sweep over 1 s, then hold still for 2 s
            s = min(t, 1.0)
            p = np.array([0.25, 0.2 * (3 * s * s - 2 * s ** 3), 0.30])
            plan = ac.plan(_hs(p), True, t)
        return plan["pw"]

    np.testing.assert_allclose(run(120), run(40), atol=5e-3)


# --------------------------------------------------------------------------- #
# in-loop joint shaper
# --------------------------------------------------------------------------- #
def test_shaper_bounds_joint_speed_through_violent_input():
    rig = load_rig()
    ac = ArmController(rig, "right")
    rate = float(rig["safety"]["sim_rate_limit"])
    t, q_prev = 0.0, None
    rng = np.random.default_rng(0)
    p = np.array([0.25, 0.0, 0.30])
    for i in range(480):                       # 4 s of erratic, fast, rolling input
        if i % 40 == 0:
            p = np.array([0.25, 0, 0.30]) + rng.uniform(-0.25, 0.25, 3)
        R = _rotx(rng.uniform(-2.5, 2.5))
        q = ac.update(_hs(p, R), True, t)
        if q_prev is not None:
            dq = np.abs(q - q_prev) / DT
            assert dq.max() <= rate + 1e-6, f"joint moved {dq.max():.1f} rad/s at tick {i}"
        q_prev = q
        t += DT


def test_shaper_keeps_render_state_consistent():
    """The shaped pose is seeded back: FK (render stream) and the returned
    command must be the same robot."""
    rig = load_rig()
    ac = ArmController(rig, "right")
    t = 0.0
    for _ in range(30):
        q = ac.update(_hs([0.3, 0.1, 0.35]), True, t)
        np.testing.assert_allclose(q, ac.ik.q, atol=1e-12)
        t += DT


# --------------------------------------------------------------------------- #
# capsule separation
# --------------------------------------------------------------------------- #
def test_capsules_fingers_pointing_at_each_other():
    """The case the point-pair guard missed: fingertips aimed at the other palm."""
    w_l = np.array([0.0, -0.15, 1.0])
    w_r = np.array([0.0, 0.15, 1.0])
    d_l = np.array([0.0, 1.0, 0.0])            # left fingers point right…
    d_r = np.array([0.0, -1.0, 0.0])           # …right fingers point left
    n_l, n_r = separate_capsules(w_l, w_r, d_l, d_r, 0.19, 0.12)
    cl, cr = closest_points_segments(n_l, n_l + d_l * 0.19, n_r, n_r + d_r * 0.19)
    assert np.linalg.norm(cr - cl) == pytest.approx(0.12, abs=1e-9)


def test_capsules_parallel_palms_and_noop():
    w_l = np.array([0.0, -0.04, 1.0])
    w_r = np.array([0.0, 0.04, 1.0])
    fwd = np.array([-1.0, 0.0, 0.0])
    n_l, n_r = separate_capsules(w_l, w_r, fwd, fwd, 0.19, 0.12)
    assert np.linalg.norm(n_r - n_l) == pytest.approx(0.12, abs=1e-9)
    far_l, far_r = np.array([0.0, -0.4, 1.0]), np.array([0.0, 0.4, 1.0])
    n_l, n_r = separate_capsules(far_l, far_r, fwd, fwd, 0.19, 0.12)
    np.testing.assert_allclose(n_l, far_l)
    np.testing.assert_allclose(n_r, far_r)


def test_closest_points_crossing_segments():
    cl, cr = closest_points_segments([-1, 0, 0], [1, 0, 0], [0, -1, 0.2], [0, 1, 0.2])
    np.testing.assert_allclose(cl, [0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(cr, [0, 0, 0.2], atol=1e-9)


def test_full_clasp_push_is_lateral_not_shear():
    """The 'crossing hands' clap bug (reported live 2026-06-12, measured on
    clap.npz: 48% of pushes were majority fore/aft+vertical SHEAR): real
    clasped hands are near-parallel but staggered, the closest pair sits at
    fingertip-vs-palm, and the gradient axis shears the hands past each other.
    Near-parallel capsules must resolve along the WRIST line instead."""
    w_l = np.array([0.02, -0.05, 1.0])
    w_r = np.array([-0.03, 0.05, 0.97])              # staggered, 10 cm lateral
    d_l = np.array([-0.94, 0.20, -0.28]); d_l = d_l / np.linalg.norm(d_l)
    d_r = np.array([-0.90, -0.25, -0.36]); d_r = d_r / np.linalg.norm(d_r)   # ~26° apart
    n_l, n_r = separate_capsules(w_l, w_r, d_l, d_r, 0.19, 0.12)
    cl, cr = closest_points_segments(n_l, n_l + d_l * 0.19, n_r, n_r + d_r * 0.19)
    assert np.linalg.norm(cr - cl) >= 0.12 - 1e-9
    wrist_line = (w_r - w_l) / np.linalg.norm(w_r - w_l)
    push_r = n_r - w_r
    push_l = n_l - w_l
    # the pair must separate ALONG ITS OWN LINE (apart), not slide past each
    # other on the gradient's fore/aft tilt
    assert push_r @ wrist_line > 0.9 * np.linalg.norm(push_r)
    assert -push_l @ wrist_line > 0.9 * np.linalg.norm(push_l)
    assert n_r[1] - n_l[1] > w_r[1] - w_l[1], "hands must move APART laterally"


def test_perpendicular_poke_keeps_gradient_axis():
    """A fingertip poke INTO the other palm (axes ~orthogonal) genuinely needs
    the closest-point axis — pushing laterally would let the poke advance."""
    w_l = np.array([0.0, -0.30, 1.0])
    d_l = np.array([0.0, 1.0, 0.0])                  # left fingers point right…
    w_r = np.array([-0.02, -0.06, 1.0])
    d_r = np.array([-1.0, 0.0, 0.0])                 # …into the right palm (fwd axis)
    cl0, cr0 = closest_points_segments(w_l, w_l + d_l * 0.19, w_r, w_r + d_r * 0.19)
    grad = (cr0 - cl0) / np.linalg.norm(cr0 - cl0)
    n_l, n_r = separate_capsules(w_l, w_r, d_l, d_r, 0.19, 0.12)
    cl, cr = closest_points_segments(n_l, n_l + d_l * 0.19, n_r, n_r + d_r * 0.19)
    assert np.linalg.norm(cr - cl) >= 0.12 - 1e-9
    push_r = n_r - w_r
    assert push_r @ grad > 0.8 * np.linalg.norm(push_r), "poke must resolve along the gradient"


# --------------------------------------------------------------------------- #
# j6 saturation hysteresis (the left-roll pivot from live_0611_133349)
# --------------------------------------------------------------------------- #
def _roll_beyond_stop(side: str, direction: float):
    """Drive a pure tool-axis roll well past the j6 stop (beyond the ±π wrap of
    the relative twist) and back. Returns the joint trajectory."""
    from bimanual_teleop.arms.ik import ArmIK
    from bimanual_teleop.vr.frames import SE3, SO3
    rig = load_rig()
    ik = ArmIK(rig, side)
    R0 = ik.fk_ee().rotation().as_matrix()
    p0 = ik.fk_wrist().translation()
    a = ik._joint_axis_base(ik.joints[5])
    a = a / np.linalg.norm(a)

    def rot(th):
        c, s = np.cos(th), np.sin(th)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + s * K + (1 - c) * (K @ K)

    qs = []
    thetas = list(np.linspace(0, direction * 6.0, 120)) + \
             list(np.linspace(direction * 6.0, 0, 120))
    for th in thetas:
        ik.solve(SE3.from_rotation_and_translation(SO3.from_matrix(rot(th) @ R0), p0))
        qs.append(ik.q)
    return np.array(qs), np.asarray(rig["arms"][side]["neutral_q"], float)


def test_left_roll_past_wrap_stays_pinned_no_pivot():
    """The measured failure: left j6 has only ~30° of headroom rolling negative;
    past the ±π wrap the old code walked j6 the LONG way across its range
    through the wrist singularity while j4/j5 wandered ~0.5 rad."""
    qs, q_rest = _roll_beyond_stop("left", -1.0)
    settle = qs[20:]                       # after reaching the stop
    assert settle[:, 5].max() <= -1.55, \
        f"left j6 crossed back to {settle[:,5].max():+.2f} — long-way travel"
    for j in (3, 4):
        wander = np.abs(qs[:, j] - q_rest[j]).max()
        assert wander < 0.30, f"left j{j+1} pivoted {wander:.2f} rad during saturated roll"


def test_right_roll_past_wrap_stays_pinned_no_pivot():
    qs, q_rest = _roll_beyond_stop("right", +1.0)
    settle = qs[20:]
    assert settle[:, 5].min() >= +1.55, \
        f"right j6 crossed back to {settle[:,5].min():+.2f} — long-way travel"
    for j in (3, 4):
        wander = np.abs(qs[:, j] - q_rest[j]).max()
        assert wander < 0.30, f"right j{j+1} pivoted {wander:.2f} rad during saturated roll"
