"""RuntimeGuard trip matrix + the absolute hard speed ceiling — pure, no hardware."""
import math

import numpy as np
import pytest

from bimanual_teleop.safety.runtime_guard import (
    GuardTrip,
    RuntimeGuard,
    effective_rate_limit,
)

CFG = {
    "enabled": True,
    "max_tracking_error": 0.35,
    "tracking_grace_s": 0.5,
    "tracking_trip_s": 0.3,
    "max_motor_temp_c": 75.0,
    "max_motor_current": 8.0,
    "loop_stall_s": 0.5,
    "warn_fraction": 0.8,
}
REST = np.zeros(6)


def _g():
    return RuntimeGuard(dict(CFG), sides=("right",), n=6)


def test_clean_tracking_never_trips():
    g = _g()
    t = 0.0
    for _ in range(50):
        # command leads measured by a small lag (normal gravity sag), well under limit
        assert g.check("right", REST + 0.05, REST, effort=np.zeros(6), temp=np.full(6, 40.0), t=t) == []
        t += 0.02
    assert g.tripped is None


def test_tracking_trip_after_debounce_and_grace():
    g = _g()
    # during grace: a big gap must NOT trip yet
    g.check("right", np.full(6, 1.0), REST, t=0.0)
    g.check("right", np.full(6, 1.0), REST, t=0.4)
    assert g.tripped is None
    # after grace, the gap must PERSIST past trip_s before it fires
    g.check("right", np.full(6, 1.0), REST, t=0.6)      # gap opens (first time past grace)
    assert g.tripped is None                              # debounce not elapsed
    with pytest.raises(GuardTrip) as e:
        g.check("right", np.full(6, 1.0), REST, t=1.0)   # 0.4s later > trip_s
    assert e.value.kind == "tracking"
    assert g.tripped["kind"] == "tracking"


def test_tracking_gap_that_recovers_does_not_trip():
    g = _g()
    g.check("right", REST, REST, t=0.0)                   # anchor t0=0 (grace from here)
    g.check("right", REST, REST, t=0.3)                   # still in grace, no gap
    g.check("right", np.full(6, 1.0), REST, t=0.6)        # gap opens just past grace
    g.check("right", np.full(6, 1.0), REST, t=0.7)        # still open, < trip_s
    # recovers before trip_s elapses → debounce resets, no trip
    assert g.check("right", REST + 0.01, REST, t=0.8) == []
    assert g.tripped is None
    # warning band (>80% of limit) surfaces without tripping (next tick, no stall)
    warns = g.check("right", REST + 0.30, REST, t=1.0)
    assert any("tracking" in w for w in warns)


def test_thermal_trip():
    g = _g()
    hot = np.array([40, 40, 78, 40, 40, 40], float)
    with pytest.raises(GuardTrip) as e:
        g.check("right", REST, REST, temp=hot, t=0.0)
    assert e.value.kind == "thermal"
    assert "j3" in e.value.detail


def test_thermal_warning_band():
    g = _g()
    warm = np.full(6, 62.0)         # >80% of 75 but under it
    warns = g.check("right", REST, REST, temp=warm, t=0.0)
    assert any("°C" in w for w in warns)
    assert g.tripped is None


def test_overcurrent_trip():
    g = _g()
    eff = np.array([1, 1, 1, 1, 1, 9.5], float)
    with pytest.raises(GuardTrip) as e:
        g.check("right", REST, REST, effort=eff, t=0.0)
    assert e.value.kind == "overcurrent"
    assert "j6" in e.value.detail


def test_loop_stall_trip():
    g = _g()
    g.check("right", REST, REST, t=0.0)
    with pytest.raises(GuardTrip) as e:
        g.check("right", REST, REST, t=1.0)      # 1.0s gap > loop_stall_s 0.5
    assert e.value.kind == "loop-stall"


def test_normal_loop_cadence_no_stall():
    g = _g()
    t = 0.0
    for _ in range(100):
        g.check("right", REST, REST, t=t)
        t += 0.02                                 # 50 Hz, well under stall_s
    assert g.tripped is None


def test_near_pi_wrap_is_not_a_false_gap():
    g = _g()
    # commanded +3.14, measured −3.14: physically ~0 apart, must not read as ~6.28
    g.check("right", np.full(6, math.pi - 0.01), np.full(6, -math.pi + 0.01), t=0.6)
    g.check("right", np.full(6, math.pi - 0.01), np.full(6, -math.pi + 0.01), t=0.9)
    assert g.tripped is None


def test_disabled_guard_is_inert():
    g = RuntimeGuard({"enabled": False}, sides=("right",))
    assert g.check("right", np.full(6, 5.0), REST, temp=np.full(6, 200.0), effort=np.full(6, 99.0), t=0.0) == []
    assert g.tripped is None


def test_reset_clears_latched_trip():
    g = _g()
    with pytest.raises(GuardTrip):
        g.check("right", REST, REST, t=0.0)
        g.check("right", REST, REST, t=2.0)
    assert g.tripped is not None
    g.reset()
    assert g.tripped is None
    assert g.check("right", REST + 0.05, REST, t=0.0) == []


def test_telemetry_recorded_for_dashboard():
    g = _g()
    g.check("right", REST + 0.1, REST, effort=np.arange(6.0), temp=np.full(6, 50.0), t=0.0)
    rec = g.last["right"]
    assert rec["gap_joint"] >= 1 and "gap_max" in rec
    assert rec["temp_max"] == 50.0 and "effort_max" in rec


def test_per_joint_tracking_weak_wrist_vs_strong_shoulder():
    """Per-joint limits: the weak wrist (j6) may lag far without tripping, but a
    strong shoulder joint (j1) over its own tight limit still trips. Pins the metal
    fix where kp=10 j6 lagged 83° during a normal roll yet kp=80 j1 tracks tight."""
    cfg = dict(CFG, max_tracking_error=[0.6, 0.6, 0.6, 1.4, 3.0, 3.0])
    # j6 lags 83° for well past the debounce — must NOT trip
    g = RuntimeGuard(cfg, sides=("right",))
    m = REST.copy(); m[5] = math.radians(83)
    for i in range(80):
        g.check("right", REST, m, t=i * 0.02)
    assert g.tripped is None
    # j1 stuck 40° (> its 34° limit) — MUST trip
    g = RuntimeGuard(cfg, sides=("right",))
    c = REST.copy(); c[0] = math.radians(40)
    with pytest.raises(GuardTrip) as e:
        for i in range(80):
            g.check("right", c, REST, t=i * 0.02)
    assert e.value.kind == "tracking" and "j1" in e.value.detail


def test_hard_speed_ceiling_clamps_rate_limit():
    # config asks for 5.0 rad/s, ceiling is 1.5 → effective is 1.5
    rig = {"hardware": {"rate_limit": 5.0}, "safety": {"runtime": {"hard_max_joint_speed": 1.5}}}
    assert effective_rate_limit(rig) == 1.5
    # a slow request stays slow (ceiling never speeds anything up)
    rig["hardware"]["rate_limit"] = 0.8
    assert effective_rate_limit(rig) == 0.8
    # no ceiling configured → passthrough
    assert effective_rate_limit({"hardware": {"rate_limit": 2.0}}) == 2.0
