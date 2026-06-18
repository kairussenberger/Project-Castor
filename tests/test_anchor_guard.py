"""Mid-session anchor-jump guard (safety/anchor_guard.py + engine wiring).

The threat model is from real forensics (2026-06-11): ORBIT stream anchors move
on recenter / app restart / headset sleep, silently shifting the absolute
mapping under an applied calibration. The guard must trip on those — and must
NOT trip on what real sessions are full of: single-frame tracking glitches of
0.2–1.5 m (measured in most recordings), dropouts with the hands moving while
untracked, and honest fast motion.
"""
from __future__ import annotations

import numpy as np
import pytest

from bimanual_teleop.config import SIDES, load_rig
from bimanual_teleop.engine import TeleopEngine
from bimanual_teleop.safety.anchor_guard import AnchorGuard
from bimanual_teleop.vr.frames import HandSample, VRFrame

HZ = 110.0
DT = 1.0 / HZ


def _rig(transport="orbit", **guard_over):
    rig = {"safety": {"anchor_guard": {"enabled": True, **guard_over}},
           "vr": {"transport": transport}}
    return rig


def _run(g, seq, t0=0.0, dt=DT, fresh=True, armed=True):
    """Feed a list of {side: pos-or-None} dicts; return (last_holds, times)."""
    holds = {}
    t = t0
    for step in seq:
        wb = {s: (None if step.get(s) is None else np.asarray(step[s], float))
              for s in SIDES}
        holds = g.observe(wb, fresh, t, armed=armed)
        t += dt
    return holds, t


def _smooth(n, base=(0.2, 0.0, 0.4), amp=0.15, hz_motion=1.5, mirror=True):
    """n frames of honest two-hand motion (peaks ~1.4 m/s at these defaults)."""
    out = []
    for i in range(n):
        ph = 2 * np.pi * hz_motion * i * DT
        l = np.array([-base[0] - amp * np.sin(ph), base[1] + amp * np.cos(ph), base[2]])
        r = np.array([base[0] + amp * np.sin(ph), base[1] + (amp * np.cos(ph) if mirror else 0.0), base[2]])
        out.append({"left": l, "right": r})
    return out


# --------------------------------------------------------------------------- #
# pure guard
# --------------------------------------------------------------------------- #
def test_smooth_motion_never_trips_or_holds():
    g = AnchorGuard(_rig())
    seq = _smooth(400)
    for i, step in enumerate(seq):
        holds = g.observe({s: step[s] for s in SIDES}, True, i * DT)
        assert not any(holds.values())
    assert not g.tripped


def test_single_frame_glitch_holds_then_resumes_without_trip():
    g = AnchorGuard(_rig())
    seq = _smooth(50)
    glitch = dict(seq[25])
    glitch["right"] = seq[25]["right"] + np.array([0.0, 0.9, 0.0])   # 0.9 m teleport, 1 frame
    seq[25] = glitch
    held_any = False
    for i, step in enumerate(seq):
        holds = g.observe({s: step[s] for s in SIDES}, True, i * DT)
        held_any = held_any or holds["right"]
        assert not holds["left"]
    assert held_any, "the glitch frame must hold the side"
    assert not g.tripped
    # back to normal: no residual holds
    holds = g.observe(seq[-1], True, len(seq) * DT)
    assert not any(holds.values())


def test_persistent_single_jump_with_other_steady_resumes_no_trip():
    """One hand teleports and STAYS (identity glitch / violent event) while the
    other streams steadily: the shared anchor cannot have moved — accept the
    new regime (the controller re-anchors + glides), never trip."""
    g = AnchorGuard(_rig())
    pre = _smooth(30)
    _run(g, pre)
    post = []
    for i in range(20):
        step = {"right": pre[-1]["right"] + np.array([0.0, 0.6, 0.0]) + 0.002 * np.sin(i),
                "left": pre[-1]["left"] + np.array([0.001 * i, 0.0, 0.0])}  # steady-ish
        post.append(step)
    holds, _ = _run(g, post, t0=30 * DT)
    assert not g.tripped
    assert not holds["right"], "side must resume after the confirm window"


def test_coherent_two_hand_jump_trips():
    g = AnchorGuard(_rig())
    pre = _smooth(30)
    _run(g, pre)
    delta = np.array([0.05, -0.85, 0.1])                 # one anchor delta, both hands
    post = [{s: pre[-1][s] + delta for s in SIDES} for _ in range(8)]
    for i, step in enumerate(post):                       # vary slightly so frames are "new"
        for s in SIDES:
            step[s] = step[s] + 0.002 * np.sin(i + (0 if s == "left" else 1))
    _run(g, post, t0=30 * DT)
    assert g.tripped
    assert "anchor" in (g.trip_reason or "")
    assert g.take_trip() and not g.take_trip()            # edge consumed once


def test_jump_with_other_hand_untracked_trips():
    g = AnchorGuard(_rig())
    pre = [{"left": None, "right": np.array([0.2, 0.0, 0.4]) + 0.01 * np.sin(i) * np.ones(3)}
           for i in range(30)]
    _run(g, pre)
    post = [{"left": None, "right": pre[-1]["right"] + np.array([0.0, 0.7, 0.0]) + 0.002 * i * np.ones(3)}
            for i in range(8)]
    _run(g, post, t0=30 * DT)
    assert g.tripped
    assert "untracked" in (g.trip_reason or "")


def test_incoherent_dual_jump_trips_fail_safe():
    g = AnchorGuard(_rig())
    pre = _smooth(30)
    _run(g, pre)
    post = []
    for i in range(8):
        post.append({"left": pre[-1]["left"] + np.array([0.0, 0.8, 0.0]) + 0.002 * i,
                     "right": pre[-1]["right"] + np.array([0.6, -0.4, 0.3]) + 0.002 * i})
    _run(g, post, t0=30 * DT)
    assert g.tripped
    assert "incoherent" in (g.trip_reason or "")


def test_dropout_with_movement_reseeds_quietly():
    """Hands legitimately move while untracked — re-tracking far away is NOT a
    jump (continuity expired)."""
    g = AnchorGuard(_rig())
    _run(g, _smooth(30))
    gap = [{"left": None, "right": None}] * int(0.5 / DT)
    _run(g, gap, t0=30 * DT)
    far = [{"left": np.array([-0.5, 0.4, 0.2]) + 0.002 * i, "right": np.array([0.5, 0.4, 0.2]) + 0.002 * i}
           for i in range(10)]
    holds, _ = _run(g, far, t0=30 * DT + 0.5)
    assert not g.tripped and not any(holds.values())


def test_duplicate_frames_do_not_advance_confirmation():
    """Replay/latest() can deliver one sample across several engine ticks: the
    confirm count and dt must run on CHANGED samples, not ticks."""
    g = AnchorGuard(_rig(confirm_frames=3))
    base = {"left": np.array([-0.2, 0.0, 0.4]), "right": np.array([0.2, 0.0, 0.4])}
    seq = [dict(base) for _ in range(20)]
    for i, step in enumerate(seq):
        for s in SIDES:
            step[s] = step[s] + 0.001 * np.sin(i + (s == "left"))
    _run(g, seq)
    jumped = {s: seq[-1][s] + np.array([0.0, 0.8, 0.0]) for s in SIDES}
    t0 = 20 * DT
    g.observe(jumped, True, t0)                                  # jump sample (n_far=1)
    for k in range(1, 5):
        holds = g.observe(jumped, True, t0 + k * DT)             # exact duplicates
        assert holds == {s: True for s in SIDES}
    assert not g.tripped, "duplicates must not confirm a jump"
    # the glitch snaps back -> resume, still no trip
    g.observe({s: seq[-1][s] for s in SIDES}, True, t0 + 5 * DT)
    assert not g.tripped
    holds = g.observe({s: seq[-1][s] + 0.001 for s in SIDES}, True, t0 + 6 * DT)
    assert not any(holds.values())


def test_blackout_trips_live_transport_only():
    for transport, should_trip in (("orbit", True), ("vuer", True),
                                   ("replay", False), ("fake", False)):
        g = AnchorGuard(_rig(transport=transport))
        _run(g, _smooth(10))
        t = 10 * DT
        for _ in range(int(3.0 / DT)):
            g.observe({s: None for s in SIDES}, False, t)        # stream dead
            t += DT
        assert g.tripped is should_trip, transport
        if should_trip:
            assert "stream dead" in (g.trip_reason or "")


def test_blackout_needs_a_live_stream_first():
    """A dead stream BEFORE anything was ever fresh (app not started yet) must
    not trip."""
    g = AnchorGuard(_rig())
    t = 0.0
    for _ in range(int(5.0 / DT)):
        g.observe({s: None for s in SIDES}, False, t)
        t += DT
    assert not g.tripped


def test_unarmed_observation_never_trips_but_keeps_continuity():
    g = AnchorGuard(_rig())
    pre = _smooth(30)
    _run(g, pre, armed=False)
    delta = np.array([0.0, 0.9, 0.0])
    post = [{s: pre[-1][s] + delta + 0.002 * i for s in SIDES} for i in range(8)]
    _run(g, post, t0=30 * DT, armed=False)
    assert not g.tripped
    # now armed: a SECOND coherent jump from the accepted regime trips
    post2 = [{s: post[-1][s] + np.array([0.0, -0.7, 0.0]) + 0.002 * i for s in SIDES}
             for i in range(8)]
    _run(g, post2, t0=40 * DT, armed=True)
    assert g.tripped


def test_reset_forgives_trip_and_continuity():
    g = AnchorGuard(_rig())
    pre = _smooth(30)
    _run(g, pre)
    post = [{s: pre[-1][s] + np.array([0.0, 0.9, 0.0]) + 0.002 * i for s in SIDES}
            for i in range(8)]
    _run(g, post, t0=30 * DT)
    assert g.tripped
    g.reset()
    assert not g.tripped and g.trip_reason is None
    holds = g.observe({s: pre[0][s] for s in SIDES}, True, 50 * DT)  # far from anything prior
    assert not any(holds.values()) and not g.tripped


def test_disabled_guard_is_inert():
    rig = _rig()
    rig["safety"]["anchor_guard"]["enabled"] = False
    g = AnchorGuard(rig)
    pre = _smooth(10)
    _run(g, pre)
    post = [{s: pre[-1][s] + np.array([0.0, 2.0, 0.0]) + 0.002 * i for s in SIDES}
            for i in range(10)]
    holds, _ = _run(g, post, t0=10 * DT)
    assert not g.tripped and not any(holds.values())


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


OP_AXES = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
TORSO_W = np.array([0.0, -0.35, 0.0])


def _frame(w_by_side: dict, t: float) -> VRFrame:
    hands = {}
    for s, w in w_by_side.items():
        if w is None:
            continue
        W = np.eye(4)
        W[:3, 3] = TORSO_W + OP_AXES @ np.asarray(w, dtype=float)
        hands[s] = HandSample(tracked=True, wrist=W, landmarks=None, pinch=0.0)
    return VRFrame(stamp=t, head=np.eye(4), hands=hands)


def _glide(a, b, n):
    """n frames gliding both sides a→b (real operators do not teleport)."""
    out = []
    for k in range(1, n + 1):
        f = k / n
        out.append({s: (1 - f) * np.asarray(a[s], float) + f * np.asarray(b[s], float)
                    for s in SIDES})
    return out


def test_engine_trip_locks_follow_and_holds_arms():
    rig = load_rig()                                   # transport fake → follow unlocked
    eng = TeleopEngine(rig, DummySink())
    engaged = {s: True for s in SIDES}
    base = {"left": np.array([-0.2, 0.05, 0.42]), "right": np.array([0.2, 0.05, 0.42])}
    t = 0.0
    for i in range(120):                               # arms engage + follow smoothly
        w = {s: base[s] + np.array([0.0, 0.04 * np.sin(3 * t), 0.0]) for s in SIDES}
        eng.tick(_frame(w, t), engaged, t)
        t += DT
    assert not eng.follow_locked and not eng.guard.tripped
    delta = np.array([0.0, 1.0, 0.0])                  # the 2026-06-11 class of shift
    for i in range(10):
        w = {s: base[s] + delta + 0.002 * np.sin(i) for s in SIDES}
        eng.tick(_frame(w, t), engaged, t)
        t += DT
    assert eng.guard.tripped and eng.follow_locked
    assert eng.calib_status and eng.calib_status.get("phase") == "tripped"
    # locked arms hold while the hands keep waving in the shifted frame
    for _ in range(60):                                # let the shaper bleed velocity
        w = {s: base[s] + delta + np.array([0.0, 0.05 * np.sin(5 * t), 0.0]) for s in SIDES}
        eng.tick(_frame(w, t), engaged, t)
        t += DT
    q0 = {s: eng.arm[s].ik.q.copy() for s in SIDES}
    for _ in range(120):
        w = {s: base[s] + delta + np.array([0.0, 0.08 * np.sin(5 * t), 0.0]) for s in SIDES}
        eng.tick(_frame(w, t), engaged, t)
        t += DT
    for s in SIDES:
        assert float(np.linalg.norm(eng.arm[s].ik.q - q0[s])) < 1e-9, "locked arm moved"
    # banner persists (no fade while tripped)
    assert eng.calib_status and eng.calib_status.get("phase") == "tripped"


def test_engine_trip_cancels_inflight_capture():
    rig = load_rig()
    eng = TeleopEngine(rig, DummySink())
    engaged = {s: True for s in SIDES}
    rest = {"left": np.array([-0.3, -0.5, 0.05]), "right": np.array([0.3, -0.5, 0.05])}

    def jitter(w, t):                                  # real streams never repeat bit-exact
        return {s: w[s] + 0.0015 * np.sin(37.0 * t) for s in SIDES}

    t = 0.0
    for _ in range(30):
        eng.tick(_frame(jitter(rest, t), t), engaged, t)
        t += DT
    eng.request_calibration()
    for _ in range(15):                                # capture running, operator at rest
        eng.tick(_frame(jitter(rest, t), t), engaged, t)
        t += DT
    assert eng.neutral.active
    delta = np.array([0.0, -1.2, 0.0])
    for i in range(10):                                # anchor moves mid-capture
        w = {s: rest[s] + delta + 0.002 * np.sin(i) for s in SIDES}
        eng.tick(_frame(w, t), engaged, t)
        t += DT
    assert eng.guard.tripped
    assert not eng.neutral.active, "capture must cancel — poses straddle the anchor change"
    assert eng.follow_locked


def test_engine_recalibration_clears_trip_and_unlocks(tmp_path):
    rig = load_rig()
    rig["mapping"]["calib_file"] = str(tmp_path / "operator_calib.json")
    eng = TeleopEngine(rig, DummySink())
    engaged = {s: True for s in SIDES}
    base = {"left": np.array([-0.2, 0.05, 0.42]), "right": np.array([0.2, 0.05, 0.42])}
    t = 0.0
    for _ in range(60):
        eng.tick(_frame({s: base[s] + 0.0015 * np.sin(37.0 * t) for s in SIDES}, t),
                 engaged, t)
        t += DT
    delta = np.array([0.0, 0.9, 0.0])
    shifted = {s: base[s] + delta for s in SIDES}
    for i in range(10):
        eng.tick(_frame({s: shifted[s] + 0.002 * np.sin(1 + i) for s in SIDES}, t), engaged, t)
        t += DT
    assert eng.guard.tripped and eng.follow_locked

    # operator recalibrates IN THE NEW ANCHOR FRAME (poses shifted by delta too)
    rest = {"left": np.array([-0.3, -0.5, 0.05]) + delta, "right": np.array([0.3, -0.5, 0.05]) + delta}
    clap = {"left": np.array([-0.05, -0.1, 0.3]) + delta, "right": np.array([0.05, -0.1, 0.3]) + delta}
    fwd = {"left": np.array([-0.18, 0.0, 0.5]) + delta, "right": np.array([0.18, 0.0, 0.5]) + delta}
    eng.request_calibration()
    plan = [(rest, 4.0), (clap, 4.0), (fwd, 6.0)]
    cur = shifted
    for target, hold_s in plan:
        for step in _glide(cur, target, 40):           # ~0.36 s glide, honest speeds
            eng.tick(_frame(step, t), engaged, t)
            t += DT
        cur = target
        t_end = t + hold_s
        while t < t_end:
            eng.tick(_frame(cur, t), engaged, t)
            t += DT
        if eng.calib_summary is not None:
            break
    assert eng.calib_summary is not None, "recalibration never completed"
    assert not eng.guard.tripped and not eng.follow_locked
    assert eng.guard.status()["trips"] >= 1            # history kept for forensics


def test_guard_status_published_in_render_state():
    from bimanual_teleop.render_sink import RenderSink
    rig = load_rig()
    rig["vr"]["render_endpoint"] = "tcp://127.0.0.1:18901"     # avoid live-port clashes
    rig["vr"]["unity_json_endpoint"] = None
    eng = TeleopEngine(rig, DummySink())
    sink = RenderSink(rig)
    try:
        eng.tick(None, {s: False for s in SIDES}, 0.0)
        msg = sink.build_state(eng, None, {s: False for s in SIDES}, 60.0, 0.0)
        gd = msg["status"]["guard"]
        assert set(gd) >= {"enabled", "tripped", "reason", "trips", "holds"}
        assert gd["tripped"] is False
    finally:
        sink.close()
