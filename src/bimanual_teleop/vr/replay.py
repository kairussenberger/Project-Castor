"""Record + deterministic replay of a teleop session (spec Section 7, "Replay mode").

Record a Quest session to disk, then replay it through the FULL pipeline so the loop
can be debugged and bisected without wearing the headset. `ReplaySource` is a drop-in
`VRSource` (start/stop/latest/frame_at), so the engine + supervisor see exactly what
they would live — replay a recording, change a gain, replay again, compare.

`run_teleop --record` and `run_hw --record` wire this recorder into the live
launchers, and `scripts/verify_stack.py` covers a fake-source record/replay launch
smoke. Capturing a real Quest/operator session is still external hardware
validation.

On-disk format (.npz):
    t[N], head[N,4,4]  (NaN where the frame had no headset pose);
    per side:  {side}_wrist[N,4,4], {side}_tracked[N] bool, {side}_pinch[N],
               {side}_landmarks[N,25,3]  (NaN where the hand had no landmarks);
    engaged[N,2] bool in SIDES order;
    calib_json[()] str (optional) — the neutral-pose calibration that was
    APPLIED while recording (CalibResult.payload()). ORBIT stream anchors
    differ per session by metres (measured 2026-06-12: consecutive sessions
    fitted offsets ~1 m apart), so raw frames are only meaningful together
    with their session's fit: replay/analyze apply it automatically.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from ..config import SIDES
from .frames import HandSample, VRFrame

_N_LM = 25   # WebXR joints per hand


class SessionRecorder:
    """Accumulate (VRFrame, engaged, t) tuples and dump them to a .npz."""

    def __init__(self):
        self._t: list[float] = []
        self._frames: list[VRFrame] = []
        self._engaged: list[dict[str, bool]] = []

    def __len__(self) -> int:
        return len(self._t)

    def add(self, frame: VRFrame, engaged: dict[str, bool], t: float) -> None:
        self._t.append(float(t))
        self._frames.append(frame)
        self._engaged.append({s: bool(engaged.get(s, False)) for s in SIDES})

    def save(self, path, calib: dict | None = None) -> str:
        n = len(self._t)
        cols: dict[str, np.ndarray] = {
            "t": np.asarray(self._t, float),
            "head": np.stack([
                np.asarray(f.head, float) if f.head is not None else np.full((4, 4), np.nan)
                for f in self._frames
            ]) if n
            else np.empty((0, 4, 4)),
            "engaged": np.array([[e[s] for s in SIDES] for e in self._engaged], bool) if n
            else np.empty((0, len(SIDES)), bool),
        }
        for s in SIDES:
            wrist, tracked, pinch, lms = [], [], [], []
            for f in self._frames:
                h = f.hands.get(s)
                if h is None:
                    wrist.append(np.eye(4)); tracked.append(False); pinch.append(0.0)
                    lms.append(np.full((_N_LM, 3), np.nan))
                else:
                    wrist.append(np.asarray(h.wrist, float))
                    tracked.append(bool(h.tracked))
                    pinch.append(float(h.pinch))
                    lms.append(np.asarray(h.landmarks, float) if h.landmarks is not None
                               else np.full((_N_LM, 3), np.nan))
            cols[f"{s}_wrist"] = np.stack(wrist) if n else np.empty((0, 4, 4))
            cols[f"{s}_tracked"] = np.asarray(tracked, bool)
            cols[f"{s}_pinch"] = np.asarray(pinch, float)
            cols[f"{s}_landmarks"] = np.stack(lms) if n else np.empty((0, _N_LM, 3))
        if calib is not None:
            cols["calib_json"] = np.array(json.dumps(calib))   # 0-d unicode, pickle-free
        if isinstance(path, (str, bytes, os.PathLike)):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            path = str(p)
        np.savez_compressed(path, **cols)
        return path


class ReplaySource:
    """A VRSource that replays a recording. Matches FakeVRSource's interface
    (start/stop/latest/frame_at) so it drops straight into make_source / the engine."""

    def __init__(self, path: str | None = None, *, data: dict | None = None, loop: bool = False,
                 speed: float = 1.0):
        self.loop = bool(loop)
        self.speed = max(1e-3, float(speed))   # 0.2 = play 5x slower than recorded
        # External clock governor (hardware replay): the replay clock advances by
        # speed * _rate_scale * dt_wall each latest(). _rate_scale defaults to 1.0
        # so run_teleop/sim replay and the verify_stack record/replay smoke are
        # byte-for-byte unchanged; the ReplayConductor drives it to 0..1 on the
        # hardware path so the tape never outruns the metal (see safety/replay_drive).
        self._rate_scale = 1.0
        d = data if data is not None else dict(np.load(path, allow_pickle=False))
        self.t = np.asarray(d["t"], float)
        self.head = np.asarray(d["head"], float)
        self.engaged_arr = np.asarray(d["engaged"], bool)
        # The calibration APPLIED while this session was recorded (None for
        # older recordings and synthetic fixtures → identity, as before).
        self.calib: dict | None = None
        if "calib_json" in d:
            try:
                self.calib = json.loads(str(np.asarray(d["calib_json"]).item()))
            except (json.JSONDecodeError, ValueError):
                self.calib = None
        self._side = {s: {"wrist": np.asarray(d[f"{s}_wrist"], float),
                          "tracked": np.asarray(d[f"{s}_tracked"], bool),
                          "pinch": np.asarray(d[f"{s}_pinch"], float),
                          "landmarks": np.asarray(d[f"{s}_landmarks"], float)} for s in SIDES}
        self._t0_wall: float | None = None
        self._prev_now: float | None = None       # wall clock of the previous latest()
        self._seen_t0_wall: float | None = None    # _t0_wall observed last latest() (re-anchor on change)
        # The replay CLOCK (recorded-time seconds). Advanced incrementally so an
        # external governor can scale or freeze it; clamped into the recording span
        # (or wrapped when looping) by _index/frame_at.
        self._clock = float(self.t[0]) if len(self.t) else 0.0
        self._last_replay_t = self._clock

    @classmethod
    def from_recorder(cls, rec: SessionRecorder, **kw) -> "ReplaySource":
        """Build a ReplaySource directly from a recorder, round-tripping through the
        on-disk .npz schema (so replay is bit-identical to a saved+loaded session)."""
        import io
        buf = io.BytesIO()
        rec.save(buf)                  # np.savez_compressed accepts a file-like
        buf.seek(0)
        return cls(data=dict(np.load(buf, allow_pickle=False)), **kw)

    # --- introspection ------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    def _index(self, t: float) -> int:
        """Index of the recorded sample at or just before time t (clamped)."""
        if len(self.t) == 0:
            raise IndexError("empty recording")
        if self.loop and self.duration > 0:
            t = self.t[0] + (t - self.t[0]) % self.duration
        i = int(np.searchsorted(self.t, t, side="right") - 1)
        return max(0, min(i, len(self.t) - 1))

    # --- VRSource API -------------------------------------------------------- #
    def frame_at(self, t: float) -> VRFrame:
        i = self._index(t)
        head = self.head[i]
        hands = {}
        for s in SIDES:
            d = self._side[s]
            lm = d["landmarks"][i]
            hands[s] = HandSample(tracked=bool(d["tracked"][i]), wrist=d["wrist"][i].copy(),
                                  landmarks=(None if np.isnan(lm).all() else lm.copy()),
                                  pinch=float(d["pinch"][i]))
        return VRFrame(stamp=float(self.t[i]),
                       head=(None if np.isnan(head).all() else head.copy()),
                       hands=hands)

    def engaged_at(self, t: float) -> dict[str, bool]:
        row = self.engaged_arr[self._index(t)]
        return {s: bool(row[k]) for k, s in enumerate(SIDES)}

    def current_engaged(self) -> dict[str, bool]:
        return self.engaged_at(self._last_replay_t)

    # --- external clock governor (hardware replay) --------------------------- #
    def set_rate_scale(self, s: float) -> None:
        """Scale how fast the replay clock advances each latest(), clamped to
        [0, 1]. 1.0 is the default (unchanged behaviour); 0.0 freezes the clock.
        The ReplayConductor sets this from the live tracking error so the tape
        never outruns the gravity-loaded arm."""
        self._rate_scale = float(np.clip(float(s), 0.0, 1.0))

    def hold(self) -> None:
        """Freeze the replay clock (rate scale 0) — convenience for the conductor's
        converge phase."""
        self._rate_scale = 0.0

    def set_clock(self, t: float) -> None:
        """Jump the replay clock to recorded time t (clamped into the span by the
        sampler). Used to position the tape at the first engaged frame before the
        hardware loop starts following."""
        self._clock = float(t)
        self._last_replay_t = self._clock

    def first_engaged_time(self, sides=None) -> float | None:
        """Recorded time of the first frame whose engaged flag is True for ANY of
        `sides` (default: all SIDES), reading self.engaged_arr in SIDES order.
        None if no frame ever engages — the caller then starts at the recording
        start. This lets the loop skip a long idle pre-roll straight to motion."""
        if len(self.t) == 0 or self.engaged_arr.size == 0:
            return None
        sides = SIDES if sides is None else tuple(sides)
        cols = [k for k, s in enumerate(SIDES) if s in sides]
        if not cols:
            return None
        any_eng = self.engaged_arr[:, cols].any(axis=1)
        idx = np.flatnonzero(any_eng)
        return float(self.t[int(idx[0])]) if idx.size else None

    def latest(self) -> VRFrame | None:
        if len(self.t) == 0:
            return None
        if self._t0_wall is None:                 # synchronous: first recorded frame
            self._last_replay_t = float(self.t[0])
            return self.frame_at(self._last_replay_t)
        now = time.monotonic()
        # A caller that re-anchors _t0_wall between ticks (fast-forward / rewind by
        # hand) re-bases the clock to that origin: clock = t[0] + speed*(now - t0).
        # This keeps the old absolute semantics for that idiom while the normal
        # path runs incrementally.
        if self._t0_wall != self._seen_t0_wall:
            self._seen_t0_wall = self._t0_wall
            self._prev_now = None
            self._clock = float(self.t[0])
        # INCREMENTAL, externally-scaled clock: advance by speed*_rate_scale*dt_wall
        # since the previous latest(), so a governor can freeze/slow it tick-by-tick.
        # _rate_scale defaults to 1.0 → equivalent to the old speed*(now-t0) advance
        # (verified by the round-trip + staleness tests). The FIRST latest() after a
        # (re-)anchor references _t0_wall (not the previous tick), so a caller that
        # winds _t0_wall back in time still sees the full elapsed offset.
        prev = self._prev_now if self._prev_now is not None else self._t0_wall
        dt_wall = max(0.0, now - prev)
        self._prev_now = now
        self._clock += self.speed * self._rate_scale * dt_wall
        # Clamp into the recording span (looping is handled in _index/frame_at; for
        # a non-looping tape the clock saturates at the last sample like before).
        if not self.loop and self.duration > 0:
            self._clock = min(self._clock, float(self.t[-1]))
        self._last_replay_t = float(self._clock)
        f = self.frame_at(self._last_replay_t)
        # The recorded timestamp drives deterministic sample selection, but live
        # supervisors compare frame.stamp to the current monotonic clock for
        # staleness. Refresh the delivery stamp so `run_teleop --vr replay` does
        # not immediately classify a valid recording as stale.
        f.stamp = now
        return f

    def start(self) -> None:
        self._t0_wall = time.monotonic()
        self._seen_t0_wall = self._t0_wall
        self._prev_now = None                     # first latest() references _t0_wall
        self._clock = float(self.t[0]) if len(self.t) else 0.0

    def stop(self) -> None:
        self._t0_wall = None
        self._seen_t0_wall = None
        self._prev_now = None
