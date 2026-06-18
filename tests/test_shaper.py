"""JointCommandShaper — the no-rash-movements guarantee, unit-tested.

Every property the hardware boundary promises is asserted here: bounded speed
under target jumps, no overshoot, convergence, hardstop clamping, fail-closed on
non-finite input, and stability across loop hiccups.
"""
from __future__ import annotations

import numpy as np

from bimanual_teleop.safety.shaper import JointCommandShaper


def _make(rate=1.0, hz=3.0, q0=None, accel=None):
    lo = -np.ones(6) * 3.0
    hi = np.ones(6) * 3.0
    return JointCommandShaper(q0 if q0 is not None else np.zeros(6),
                              rate_limit=rate, smooth_hz=hz, lo=lo, hi=hi,
                              accel_limit=accel)


def test_target_jump_is_speed_capped_and_converges_without_overshoot():
    sh = _make(rate=1.0)
    tgt = np.array([2.0, -2.0, 0.5, 0.0, 0.0, 0.0])
    t, dt = 0.0, 1 / 120
    prev = sh.shape(tgt, t)
    max_speed = 0.0
    overshoot = 0.0
    for _ in range(int(8.0 / dt)):
        t += dt
        q = sh.shape(tgt, t)
        max_speed = max(max_speed, float(np.max(np.abs(q - prev))) / dt)
        overshoot = max(overshoot, float(np.max(np.sign(tgt) * (q - tgt))))
        prev = q
    assert max_speed <= 1.0 + 1e-6                  # the hard speed guarantee
    assert overshoot < 1e-3                         # critically damped: no overshoot
    assert np.allclose(prev, tgt, atol=1e-3)        # and it does get there


def test_nonfinite_target_holds_last_safe_command():
    sh = _make()
    q1 = sh.shape(np.full(6, 0.5), 0.0)
    for i in range(1, 120):
        q1 = sh.shape(np.full(6, 0.5), i / 120)
    bad = np.full(6, np.nan)
    q2 = sh.shape(bad, 1.1)
    assert np.all(np.isfinite(q2))
    for i in range(2, 60):
        q2 = sh.shape(bad, 1.0 + i / 120)
    assert np.allclose(q2, q1, atol=1e-6)           # parked, not drifting


def test_targets_beyond_hardstops_are_clamped():
    sh = _make(rate=10.0)
    tgt = np.array([99.0, -99.0, 0.0, 0.0, 0.0, 0.0])
    q = sh.shape(tgt, 0.0)
    for i in range(1, 600):
        q = sh.shape(tgt, i / 120)
    assert q[0] <= 3.0 + 1e-9 and q[1] >= -3.0 - 1e-9
    assert np.allclose(q[:2], [3.0, -3.0], atol=1e-3)


def test_loop_hiccup_does_not_violate_rate():
    """A 0.4 s scheduler stall must not produce a 0.4 s × kp jump — sub-stepping
    keeps the worst-case instantaneous speed at the cap."""
    sh = _make(rate=1.0)
    sh.shape(np.zeros(6), 0.0)
    q_before = sh.shape(np.full(6, 2.0), 1 / 120)
    q_after = sh.shape(np.full(6, 2.0), 1 / 120 + 0.4)   # hiccup
    assert float(np.max(np.abs(q_after - q_before))) <= 1.0 * 0.4 + 1e-6


def test_reset_reanchors_at_measured_pose():
    sh = _make()
    sh.shape(np.full(6, 1.0), 0.0)
    sh.reset(np.full(6, -0.5), t=10.0)
    q = sh.shape(np.full(6, -0.5), 10.0 + 1 / 120)
    assert np.allclose(q, -0.5, atol=1e-2)


def test_accel_limit_ramps_velocity_instead_of_slamming():
    """With accel_limit set, a step target produces an S-CURVE: velocity ramps
    to the cap at ≤ accel_limit (never the old instant slam), cruises, and the
    critically-damped landing still does not overshoot."""
    accel = 12.0
    sh = _make(rate=1.0, accel=accel)
    tgt = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    t, dt = 0.0, 1 / 120
    prev_q = sh.shape(tgt, t)
    prev_v = 0.0
    max_dv = 0.0
    overshoot = 0.0
    ramp_ticks = 0
    for i in range(int(8.0 / dt)):
        t += dt
        q = sh.shape(tgt, t)
        v = float((q[0] - prev_q[0]) / dt)
        max_dv = max(max_dv, abs(v - prev_v) / dt)
        overshoot = max(overshoot, float(q[0] - 2.0))
        if v < 0.9 and ramp_ticks == i:        # still ramping up, count ticks
            ramp_ticks += 1
        prev_q, prev_v = q, v
    assert max_dv <= accel * 1.05 + 1e-6, f"accel hit {max_dv:.1f} rad/s²"
    assert ramp_ticks >= 5, "velocity reached the cap near-instantly — still blocky"
    assert overshoot < 5e-3
    assert abs(prev_q[0] - 2.0) < 1e-3


def test_shaped_motion_is_frame_rate_independent():
    """The caps are per SECOND, not per frame: the same step tracked at 120 Hz
    and at 30 Hz must land on the same trajectory (sub-stepping makes the
    integration rate-independent) — what the sim shows is what hardware does."""
    tgt = np.full(6, 1.5)
    qs = {}
    for hz in (120, 30):
        sh = _make(rate=1.0, accel=12.0)
        sh.shape(tgt, 0.0)
        q_mid = None
        for i in range(1, int(4.0 * hz) + 1):
            q = sh.shape(tgt, i / hz)
            if q_mid is None and i / hz >= 0.75:
                q_mid = q
        qs[hz] = (q_mid, q)
    assert np.allclose(qs[120][0], qs[30][0], atol=0.05), "mid-glide diverged across rates"
    assert np.allclose(qs[120][1], qs[30][1], atol=1e-3), "settled pose diverged across rates"


def test_arm_shaper_built_from_rig_clamps_to_yam_hardstops():
    """The hardware-boundary shaper wiring (hardware.arm_shaper) must use the
    REAL YAM joint limits and config caps, without needing the i2rt SDK."""
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.hardware import arm_shaper

    rig = load_rig()
    q0 = np.asarray(rig["arms"]["left"]["neutral_q"], float)
    sh = arm_shaper(rig, q0)
    assert sh.rate <= 3.0
    crazy = q0 + 100.0
    q = sh.shape(crazy, 0.0)
    for i in range(1, 2400):
        q = sh.shape(crazy, i / 120)
    hi = np.asarray(rig["arms"]["joint_limits"]["upper"], float)
    assert np.all(q <= hi + 1e-9)                  # physical hardstops enforced
    assert np.allclose(q, hi, atol=5e-2)           # parked at the stop, not past it
