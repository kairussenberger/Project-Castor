"""Per-session calibration EMBEDDING in recordings + the trim curation tool.

Why: ORBIT stream anchors move metres between sessions (2026-06-12: four
consecutive sessions fitted offsets ~1 m apart). Raw frames are only
meaningful together with the fit that ran while they were recorded — a tape
without its fit replays/scores through identity with metre-scale phantom
errors. The recorder embeds the applied fit; replay/analyze apply it; old
recordings and synthetic fixtures (no embedded fit) stay identity-exact."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from bimanual_teleop.config import SIDES, load_rig
from bimanual_teleop.engine import TeleopEngine
from bimanual_teleop.vr.frames import HandSample, VRFrame
from bimanual_teleop.vr.neutral_calib import (ROBOT_NEUTRAL_DEFAULT, ROBOT_REST_DEFAULT,
                                              fit_two_pose, parse_calibration)
from bimanual_teleop.vr.replay import ReplaySource, SessionRecorder

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("trim_session", REPO / "scripts" / "trim_session.py")
trim_session = importlib.util.module_from_spec(_spec)
sys.modules["trim_session"] = _spec.loader and trim_session
_spec.loader.exec_module(trim_session)


def _payload():
    pa = {"left": np.array([-0.22, 0.05, 0.45]), "right": np.array([0.22, 0.05, 0.45])}
    pb = {"left": np.array([-0.20, -0.45, 0.0]), "right": np.array([0.20, -0.45, 0.0])}
    rn = {s: np.asarray(ROBOT_NEUTRAL_DEFAULT[s]) for s in SIDES}
    rr = {s: np.asarray(ROBOT_REST_DEFAULT[s]) for s in SIDES}
    return fit_two_pose(pa, pb, rn, rr).payload()


def _record(n=30, dt=1 / 30):
    rec = SessionRecorder()
    for i in range(n):
        hands = {s: HandSample(tracked=True, wrist=np.eye(4)) for s in SIDES}
        rec.add(VRFrame(stamp=i * dt, head=np.eye(4), hands=hands),
                {s: True for s in SIDES}, i * dt)
    return rec


def test_recorder_embeds_and_replay_exposes(tmp_path):
    payload = _payload()
    p = tmp_path / "s.npz"
    _record().save(p, calib=payload)
    src = ReplaySource(str(p))
    assert src.calib == payload
    assert parse_calibration(src.calib) is not None


def test_old_recordings_have_no_calib(tmp_path):
    p = tmp_path / "s.npz"
    _record().save(p)                                  # no calib kwarg → old format
    assert ReplaySource(str(p)).calib is None


def test_engine_applies_embedded_calibration():
    payload = _payload()
    rig = load_rig()
    rig["vr"]["_embedded_calib"] = payload

    class Sink:
        def set_arm(self, *a):
            pass

        def set_hand(self, *a):
            pass
    eng = TeleopEngine(rig, Sink())
    assert eng.calib_summary is not None
    np.testing.assert_allclose(eng.arm["left"].mapper.axis_scale,
                               np.asarray(payload["axis_scale"]), atol=1e-12)
    np.testing.assert_allclose(eng.arm["left"].mapper.body_offset,
                               np.asarray(payload["body_offset"]), atol=1e-12)

    rig2 = load_rig()
    rig2["vr"]["_embedded_calib"] = {"axis_scale": [99, 1, 1], "body_offset": [0, 0, 0]}
    eng2 = TeleopEngine(rig2, Sink())                  # fails the load screen → identity
    assert eng2.calib_summary is None
    np.testing.assert_allclose(eng2.arm["left"].mapper.axis_scale, np.ones(3))


def test_live_engine_records_its_applied_fit(tmp_path):
    """engine.calib_result.payload() round-trips the applied fit exactly —
    what run_teleop hands the recorder at save time."""
    rig = load_rig()
    rig["vr"]["_embedded_calib"] = _payload()

    class Sink:
        def set_arm(self, *a):
            pass

        def set_hand(self, *a):
            pass
    eng = TeleopEngine(rig, Sink())
    assert eng.calib_result is not None
    p = tmp_path / "s.npz"
    _record().save(p, calib=eng.calib_result.payload())
    assert ReplaySource(str(p)).calib["axis_scale"] == _payload()["axis_scale"]


# --------------------------------------------------------------------------- #
# trim tool
# --------------------------------------------------------------------------- #
def test_trim_arrays_slices_consistently(tmp_path):
    p = tmp_path / "s.npz"
    _record(n=300).save(p, calib=_payload())           # 10 s at 30 Hz
    data = dict(np.load(p, allow_pickle=False))
    out = trim_session.trim_arrays(data, head_s=2.0, tail_s=3.0)
    t = out["t"]
    assert t[0] >= data["t"][0] + 2.0 - 1e-9
    assert t[-1] <= data["t"][-1] - 3.0 + 1e-9
    n = len(t)
    for k in ("head", "engaged", "left_wrist", "right_tracked", "left_landmarks"):
        assert out[k].shape[0] == n, k
    assert out["calib_json"] == data["calib_json"]     # 0-d passthrough


def test_trim_empty_window_raises(tmp_path):
    p = tmp_path / "s.npz"
    _record(n=30).save(p)                              # 1 s
    data = dict(np.load(p, allow_pickle=False))
    with pytest.raises(ValueError):
        trim_session.trim_arrays(data, head_s=2.0, tail_s=2.0)


def test_trim_cli_in_place_keeps_raw_and_embeds(tmp_path, monkeypatch, capsys):
    p = tmp_path / "tape.npz"
    _record(n=300).save(p)
    calib = tmp_path / "fit.json"
    import json
    calib.write_text(json.dumps(_payload()))
    monkeypatch.setattr(sys, "argv",
                        ["trim_session", str(p), "--tail", "3", "--calib", str(calib)])
    assert trim_session.main() == 0
    raw = tmp_path / "tape.raw.npz"
    assert raw.exists()
    assert len(np.load(raw)["t"]) == 300               # original intact
    src = ReplaySource(str(p))
    assert src.calib == _payload()                     # trimmed + embedded
    assert src.duration <= 7.0 + 1e-6
    # re-running never clobbers the original
    monkeypatch.setattr(sys, "argv", ["trim_session", str(p), "--tail", "1"])
    assert trim_session.main() == 0
    assert len(np.load(raw)["t"]) == 300
