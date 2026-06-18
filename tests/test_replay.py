"""Record -> save -> load -> replay round-trip (spec Section 7, Replay mode).

The launcher-level fake-source record/replay smoke is covered by verify_stack; this
file pins the deterministic replay schema and body-relative fidelity directly.

    uv run pytest tests/test_replay.py -q
"""
from __future__ import annotations

import numpy as np

from bimanual_teleop.config import SIDES
from bimanual_teleop.vr.ingest import FakeVRSource
from bimanual_teleop.vr.replay import ReplaySource, SessionRecorder


def _record(n=30, hz=60.0) -> SessionRecorder:
    src = FakeVRSource()
    rec = SessionRecorder()
    for i in range(n):
        t = i / hz
        eng = {"left": i % 2 == 0, "right": i > 10}
        rec.add(src.frame_at(t), eng, t)
    return rec


def test_replay_roundtrip_matches_recording(tmp_path):
    rec = _record()
    path = tmp_path / "session.npz"
    rec.save(str(path))
    rs = ReplaySource(str(path))
    assert len(rs) == len(rec)
    src = FakeVRSource()
    for i in range(len(rec)):
        t = i / 60.0
        orig = src.frame_at(t)
        got = rs.frame_at(t)
        for s in SIDES:
            assert got.hands[s].tracked == orig.hands[s].tracked
            assert np.allclose(got.hands[s].wrist, orig.hands[s].wrist, atol=1e-6)
            assert np.allclose(got.hands[s].landmarks, orig.hands[s].landmarks, atol=1e-6)
        assert rs.engaged_at(t) == {"left": i % 2 == 0, "right": i > 10}


def test_replay_from_recorder_no_file():
    rs = ReplaySource.from_recorder(_record())
    assert len(rs) == 30
    assert abs(rs.duration - 29 / 60.0) < 1e-9
    f0 = rs.frame_at(-5.0)                     # clamp below start
    assert f0.hands["left"].tracked


def test_recorder_save_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "recordings" / "session.npz"
    saved = _record(n=3).save(path)
    assert saved == str(path)
    assert path.exists()
    assert len(ReplaySource(str(path))) == 3


def test_replay_none_landmarks_survive_roundtrip(tmp_path):
    """A hand with no landmarks (e.g. controller-only) must come back as None, not
    an array of NaNs."""
    from bimanual_teleop.vr.frames import HandSample, VRFrame
    rec = SessionRecorder()
    fr = VRFrame(stamp=0.0, head=np.eye(4),
                 hands={"left": HandSample(tracked=True, wrist=np.eye(4), landmarks=None),
                        "right": HandSample(tracked=False, wrist=np.eye(4), landmarks=None)})
    rec.add(fr, {"left": True, "right": False}, 0.0)
    p = tmp_path / "s.npz"
    rec.save(str(p))
    got = ReplaySource(str(p)).frame_at(0.0)
    assert got.hands["left"].landmarks is None
    assert got.hands["left"].tracked and not got.hands["right"].tracked


def test_replay_none_head_survives_roundtrip(tmp_path):
    """Missing headset pose must be replayed as None, not as an identity matrix.
    Body-relative arm control relies on this to avoid raw room-space fallback."""
    from bimanual_teleop.vr.frames import HandSample, VRFrame

    rec = SessionRecorder()
    wrist = np.eye(4)
    wrist[:3, 3] = [1.0, 1.2, -0.4]
    rec.add(
        VRFrame(
            stamp=0.0,
            head=None,
            hands={"left": HandSample(tracked=True, wrist=wrist, landmarks=None)},
        ),
        {"left": True, "right": False},
        0.0,
    )
    p = tmp_path / "missing-head.npz"
    rec.save(str(p))
    got = ReplaySource(str(p)).frame_at(0.0)
    assert got.head is None
    assert got.hands["left"].tracked is True
    assert np.allclose(got.hands["left"].wrist, wrist, atol=1e-9)


def test_replay_preserves_body_relative_torso_to_wrist_vector(tmp_path):
    """A recorded headset+wrist pair must replay to the same torso-to-wrist vector
    Unity renders and arm control consumes."""
    from bimanual_teleop.render_sink import operator_debug_state
    from bimanual_teleop.vr.calibrate import head_op_axes
    from bimanual_teleop.vr.frames import HandSample, VRFrame, euler_to_R

    def pose(R, p):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    torso_from_head = np.array([0.0, -0.35, 0.0])
    wrist_body = np.array([0.24, 0.31, 0.52])
    head = pose(euler_to_R([0.1, 0.6, -0.2]), [0.3, 1.65, -0.2])
    op_axes = head_op_axes(head)
    wrist = pose(np.eye(3), head[:3, 3] + op_axes @ (torso_from_head + wrist_body))
    frame = VRFrame(
        stamp=0.0,
        head=head,
        hands={s: HandSample(tracked=True, wrist=wrist.copy(), landmarks=None) for s in SIDES},
    )

    rec = SessionRecorder()
    rec.add(frame, {"left": True, "right": True}, 0.0)
    p = tmp_path / "body-relative.npz"
    rec.save(str(p))
    replayed = ReplaySource(str(p)).frame_at(0.0)

    orig_op = operator_debug_state(frame, torso_from_head)
    replay_op = operator_debug_state(replayed, torso_from_head)
    for side in SIDES:
        assert np.allclose(orig_op["hands"][side]["wrist_body"], wrist_body, atol=1e-9)
        assert np.allclose(replay_op["hands"][side]["wrist_body"], wrist_body, atol=1e-9)


def test_replay_preserves_wrist_rotation(tmp_path):
    """A non-identity wrist ORIENTATION must survive the round-trip (not just the
    translation) — orientation is the whole point of the teleop bug."""
    from bimanual_teleop.vr.frames import HandSample, VRFrame, euler_to_R
    R = euler_to_R([0.4, -0.7, 1.2])
    wrist = np.eye(4)
    wrist[:3, :3] = R
    wrist[:3, 3] = [0.1, -0.2, 0.3]
    rec = SessionRecorder()
    rec.add(VRFrame(stamp=0.0, head=np.eye(4),
                    hands={"left": HandSample(tracked=True, wrist=wrist, landmarks=None),
                           "right": HandSample(tracked=True, wrist=np.eye(4), landmarks=None)}),
            {"left": True, "right": True}, 0.0)
    p = tmp_path / "rot.npz"
    rec.save(str(p))
    got = ReplaySource(str(p)).frame_at(0.0).hands["left"].wrist
    assert np.allclose(got, wrist, atol=1e-9)


def test_replay_loop_wraps():
    rs = ReplaySource.from_recorder(_record(n=10, hz=10.0), loop=True)
    assert abs(rs.duration - 0.9) < 1e-9
    t0 = 0.35                                   # mid-bin (avoids float boundary)
    a = rs.frame_at(t0)
    b = rs.frame_at(t0 + rs.duration)           # one full loop later -> same sample
    assert np.allclose(a.hands["left"].wrist, b.hands["left"].wrist, atol=1e-9)
    c = rs.frame_at(0.75)                        # different time -> different wrist
    assert not np.allclose(a.hands["left"].wrist, c.hands["left"].wrist, atol=1e-3)


def test_replay_latest_refreshes_stamp_for_staleness_gate():
    """Replay uses recorded time for sample selection, but latest() must stamp frames
    with the current monotonic clock so Supervisor does not reject old recordings."""
    import time
    rs = ReplaySource.from_recorder(_record(n=10, hz=10.0))
    rs.start()
    try:
        f = rs.latest()
        now = time.monotonic()
        assert f is not None
        assert abs(now - f.stamp) < 0.2
    finally:
        rs.stop()


def test_replay_current_engaged_tracks_latest_sample():
    import time
    rs = ReplaySource.from_recorder(_record(n=20, hz=10.0))
    rs.start()
    try:
        rs._t0_wall = time.monotonic() - 1.2
        f = rs.latest()
        assert f is not None
        assert rs.current_engaged() == {"left": True, "right": True}
        rs._t0_wall = time.monotonic() - 0.35
        rs.latest()
        assert rs.current_engaged() == {"left": False, "right": False}
    finally:
        rs.stop()


def test_replay_exhausted_and_rewind():
    """A non-looping replay reports exhausted once its clock passes the last sample;
    rewind() restarts it at the top (engagement included). Drives run_hw --loop-home."""
    import time
    rs = ReplaySource.from_recorder(_record(n=20, hz=10.0))
    assert rs.exhausted is False                 # not started yet
    rs.start()
    try:
        rs.latest()
        assert rs.exhausted is False             # just started, at t[0]
        rs._t0_wall = time.monotonic() - (rs.duration + 0.5)   # clock past the end
        rs.latest()
        assert rs.exhausted is True              # played past the last sample
        rs.rewind()
        rs.latest()
        assert rs.exhausted is False             # back at the top
        assert rs.current_engaged() == {"left": True, "right": False}   # start-of-tape
    finally:
        rs.stop()


def test_replay_loop_never_exhausts():
    """A looping source plays forever — --loop-home's home transition is gated on
    exhausted, so a loop source must never trip it."""
    import time
    rs = ReplaySource.from_recorder(_record(n=20, hz=10.0), loop=True)
    rs.start()
    try:
        rs._t0_wall = time.monotonic() - (rs.duration * 3 + 0.5)
        rs.latest()
        assert rs.exhausted is False
    finally:
        rs.stop()


def test_recorded_clutch_uses_replay_engagement_and_tracking():
    import time
    from bimanual_teleop.safety.clutch import RecordedClutch

    rs = ReplaySource.from_recorder(_record(n=20, hz=10.0))
    clutch = RecordedClutch(rs)
    rs.start()
    try:
        rs._t0_wall = time.monotonic() - 1.2
        f = rs.latest()
        assert clutch.engaged("left", f) is True
        assert clutch.engaged("right", f) is True
        f.hands["right"].tracked = False
        assert clutch.engaged("right", f) is False
    finally:
        rs.stop()
