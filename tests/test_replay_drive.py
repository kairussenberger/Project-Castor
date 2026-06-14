"""ReplayConductor + the externally-governed ReplaySource clock — pure, no hardware.

The conductor paces the replay clock from the live tracking error so an absolute
recorded trajectory never outruns the gravity-loaded arm (the trip that killed
every prior hardware replay), and auto-converges onto the start pose first. The
CRUCIAL test is closed-loop: a "laggy arm" that can only track at a bounded joint
rate stays UNDER the per-joint tracking ceiling for the whole tape WITH the
conductor, and is driven OVER it at fixed rate_scale=1 — proving the governor is
what prevents the trip.

    uv run pytest tests/test_replay_drive.py -q
"""
from __future__ import annotations

import math

import numpy as np

from bimanual_teleop.safety.replay_drive import ReplayConductor
from bimanual_teleop.vr.replay import ReplaySource, SessionRecorder
from bimanual_teleop.vr.ingest import FakeVRSource

MT = np.array([0.6, 0.6, 0.6, 1.4, 3.0, 3.0])   # rig safety.runtime.max_tracking_error
CFG = {"converge_tol": 0.10, "converge_hold_s": 0.4, "converge_timeout_s": 20.0,
       "soft_frac": 0.5, "hard_frac": 0.85}


def _c(**over):
    cfg = {**CFG, **over}
    return ReplayConductor(max_tracking_error=MT, sides=("right",), cfg=cfg)


# ---- govern() shape -------------------------------------------------------- #
def test_govern_is_one_below_soft():
    c = _c()
    assert c.govern(0.0) == 1.0
    assert c.govern(CFG["soft_frac"]) == 1.0
    assert c.govern(CFG["soft_frac"] - 0.01) == 1.0


def test_govern_is_zero_at_and_above_hard():
    c = _c()
    assert c.govern(CFG["hard_frac"]) == 0.0
    assert c.govern(CFG["hard_frac"] + 0.5) == 0.0
    assert c.govern(10.0) == 0.0


def test_govern_ramps_linearly_and_monotone_between():
    c = _c()
    soft, hard = CFG["soft_frac"], CFG["hard_frac"]
    mid = 0.5 * (soft + hard)
    assert abs(c.govern(mid) - 0.5) < 1e-9            # halfway → half rate
    xs = np.linspace(soft, hard, 25)
    ys = [c.govern(x) for x in xs]
    assert all(b <= a + 1e-12 for a, b in zip(ys, ys[1:]))   # monotone non-increasing
    assert all(0.0 <= y <= 1.0 for y in ys)


def test_govern_nan_or_none_is_full_rate():
    c = _c()
    assert c.govern(float("nan")) == 1.0
    assert c.govern(None) == 1.0


# ---- converge → follow ----------------------------------------------------- #
def test_converge_holds_clock_frozen_until_sustained_convergence():
    c = _c()
    c.start(0.0)
    # Gap above tol → frozen, still converging.
    assert c.update(0.5, 0.5 / 0.6, 0.0) == 0.0
    assert c.phase == "converge"
    # Gap drops in but the hold window has not elapsed → still frozen.
    assert c.update(0.05, 0.05 / 0.6, 0.1) == 0.0
    assert c.phase == "converge"
    assert c.update(0.05, 0.05 / 0.6, 0.3) == 0.0
    assert c.phase == "converge"
    # converge_hold_s (0.4) elapsed since first in-tol sample (t=0.1) → follow.
    s = c.update(0.05, 0.05 / 0.6, 0.55)
    assert c.phase == "follow"
    assert s == 1.0                                   # ratio well below soft → full rate
    assert c.converged is True


def test_convergence_hold_restarts_if_gap_reopens():
    c = _c()
    c.start(0.0)
    c.update(0.05, 0.08, 0.0)                          # in tol
    c.update(0.5, 0.8, 0.1)                            # gap re-opens — restart the hold
    assert c.phase == "converge"
    c.update(0.05, 0.08, 0.2)                          # in tol again, t=0.2 is the NEW anchor
    assert c.update(0.05, 0.08, 0.55) == 0.0           # only 0.35s < hold → still converging
    assert c.phase == "converge"
    c.update(0.05, 0.08, 0.65)                         # 0.45s ≥ hold → follow
    assert c.phase == "follow"


def test_converge_timeout_follows_anyway_flagged():
    c = _c(converge_timeout_s=1.0)
    c.start(0.0)
    # Never converges (gap always huge), but after the timeout it follows, flagged.
    for t in np.arange(0.0, 1.5, 0.1):
        c.update(1.0, 2.0, float(t))
    assert c.phase == "follow"
    assert c.timed_out is True
    st = c.status()
    assert st["timed_out"] is True and st["converged"] is True


def test_note_trip_reenters_converge():
    c = _c()
    c.start(0.0)
    for t in [0.0, 0.1, 0.3, 0.55]:                    # converge → follow
        c.update(0.05, 0.08, t)
    assert c.phase == "follow"
    c.note_trip(1.0)                                   # genuine trip caught upstream
    assert c.phase == "converge"
    assert c.update(0.5, 0.8, 1.0) == 0.0              # frozen again, re-converging


def test_status_is_json_native():
    import json
    c = _c()
    c.start(0.0)
    c.update(float("nan"), float("nan"), 0.0)          # no telemetry yet
    st = c.status()
    json.dumps(st)                                     # must not raise (NaN→None)
    assert st["ratio"] is None
    assert set(st) == {"phase", "scale", "ratio", "converged", "timed_out"}
    assert isinstance(st["scale"], float) and isinstance(st["phase"], str)


# ---- ReplaySource external clock ------------------------------------------- #
def _record(n=30, hz=60.0) -> SessionRecorder:
    src = FakeVRSource()
    rec = SessionRecorder()
    for i in range(n):
        t = i / hz
        rec.add(src.frame_at(t), {"left": i >= 5, "right": i >= 12}, t)
    return rec


def test_set_rate_scale_zero_freezes_the_clock():
    import time
    rs = ReplaySource.from_recorder(_record(n=20, hz=10.0))
    rs.start()
    try:
        rs.set_rate_scale(0.0)
        rs.latest()
        t_a = rs._last_replay_t
        time.sleep(0.05)
        rs.latest()
        assert rs._last_replay_t == t_a                # frozen — clock did not advance
        rs.set_rate_scale(1.0)
        time.sleep(0.05)
        rs.latest()
        assert rs._last_replay_t > t_a                 # resumes when ungoverned
    finally:
        rs.stop()


def test_rate_scale_one_reproduces_prior_trajectory():
    """scale 1 (the default) must reproduce the old speed*(now-t0) clock within
    tolerance, so run_teleop/sim replay is unchanged."""
    import time
    rs = ReplaySource.from_recorder(_record(n=40, hz=20.0))
    rs.start()
    try:
        t0 = rs._t0_wall
        for _ in range(8):
            time.sleep(0.02)
            rs.latest()
        now = time.monotonic()
        expected = float(rs.t[0]) + 1.0 * (now - t0)   # old absolute formula, speed=1
        assert abs(rs._last_replay_t - expected) < 0.02
    finally:
        rs.stop()


def test_hold_is_rate_scale_zero():
    import time
    rs = ReplaySource.from_recorder(_record(n=10, hz=10.0))
    rs.start()
    try:
        rs.latest()
        rs.hold()
        a = rs._last_replay_t
        time.sleep(0.05)
        rs.latest()
        assert rs._last_replay_t == a
    finally:
        rs.stop()


def test_set_clock_jumps_the_clock():
    rs = ReplaySource.from_recorder(_record(n=30, hz=10.0))
    rs.start()
    try:
        rs.hold()                                      # don't let wall-advance perturb it
        rs.set_clock(1.5)
        f = rs.latest()
        assert abs(rs._last_replay_t - 1.5) < 0.02
        assert f is not None
    finally:
        rs.stop()


def test_first_engaged_time_correctness():
    rs = ReplaySource.from_recorder(_record(n=30, hz=10.0))   # left@i>=5, right@i>=12
    # left engages first at i=5 → t=0.5; right at i=12 → t=1.2.
    assert abs(rs.first_engaged_time(("left",)) - 0.5) < 1e-9
    assert abs(rs.first_engaged_time(("right",)) - 1.2) < 1e-9
    assert abs(rs.first_engaged_time() - 0.5) < 1e-9          # any side → earliest
    assert abs(rs.first_engaged_time(("left", "right")) - 0.5) < 1e-9


def test_first_engaged_time_none_when_never_engaged():
    src = FakeVRSource()
    rec = SessionRecorder()
    for i in range(10):
        rec.add(src.frame_at(i / 10.0), {"left": False, "right": False}, i / 10.0)
    rs = ReplaySource.from_recorder(rec)
    assert rs.first_engaged_time() is None


# ---- CRUCIAL closed-loop: the governor prevents the trip ------------------- #
def _run_closed_loop(*, governed: bool, dt=0.01, joint=0):
    """Simulate a single joint replaying an absolute trajectory against a 'laggy
    arm'. The TAPE demands the joint sweep at `tape_rate` rad/s of recorded time;
    the arm can only physically move at `arm_rate` rad/s (< tape_rate) — exactly
    the gravity-loaded/derated/shaped real arm. The conductor scales the replay
    CLOCK from the worst tracking ratio; the control run keeps it pinned at 1.

    Returns the peak tracking ratio (gap/max_track) observed over the run.
    """
    tape_rate = 3.0           # rad / recorded-second the tape demands on this joint
    arm_rate = 0.8            # rad/s the laggy arm can actually move (well below)
    duration = 4.0            # recorded seconds of trajectory
    mt = float(MT[joint])

    c = _c() if governed else None
    if c is not None:
        c.start(0.0)
    clock = 0.0               # recorded-time clock (what set_clock/rate_scale drives)
    measured = 0.0            # the laggy arm's actual joint angle
    peak_ratio = 0.0
    t = 0.0
    # Pre-converge phase is trivial here (start already at the tape origin), so the
    # interesting part is the follow governor; give the conductor a few in-tol ticks
    # to leave converge, mirroring the real start-pose glide.
    while clock < duration:
        target = tape_rate * clock                     # absolute commanded joint angle
        gap = abs(target - measured)
        ratio = gap / mt
        if t > 0.6:                                     # ignore the very first ticks
            peak_ratio = max(peak_ratio, ratio)
        if c is not None:
            scale = c.update(gap, ratio, t)
        else:
            scale = 1.0
        clock += tape_rate_scale(scale) * dt           # advance the replay clock
        # laggy arm integrates toward the current target at its bounded rate
        step = math.copysign(min(arm_rate * dt, abs(target - measured)), target - measured)
        measured += step
        t += dt
    return peak_ratio


def tape_rate_scale(scale: float) -> float:
    # The clock advances at `scale` × real time (speed=1). Factored out for clarity.
    return scale


def test_governor_keeps_gap_under_trip_threshold():
    """WITH the conductor the worst tracking ratio stays below 1.0 (no trip) for
    the whole tape, BECAUSE the clock is paced to the metal."""
    peak = _run_closed_loop(governed=True)
    assert peak < 1.0, f"governed replay tripped the guard (peak ratio {peak:.2f} ≥ 1)"
    # And it stays comfortably under — the governor targets the soft/hard band.
    assert peak < 0.95


def test_fixed_rate_one_drives_gap_over_threshold():
    """The control: at fixed rate_scale=1 the same laggy arm is driven PAST the
    ceiling — proving the governor (not the arm/tape) is what averts the trip."""
    peak = _run_closed_loop(governed=False)
    assert peak > 1.0, f"control run did not exceed the ceiling (peak ratio {peak:.2f})"
