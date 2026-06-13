"""VR pose sources. A source produces the latest VRFrame on demand (latest-value
semantics). Two implementations:

- FakeVRSource: synthetic motion (wrists trace arcs, fingers open/close). Lets the
  whole pipeline run and be verified with no headset.
- VuerVRSource: real Quest 3 hand-tracking over WebXR (see vr/vuer_source.py).

The control loop only depends on `latest() -> VRFrame | None`.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from ..config import SIDES
from .frames import HandSample, VRFrame
from ..hands.quest_retarget import synthetic_webxr_hand


class VRSource:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest(self) -> VRFrame | None: ...


def _wrist_mat(pos, R=None) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = pos
    if R is not None:
        T[:3, :3] = R
    return T


class FakeVRSource(VRSource):
    """Synthetic operator: each wrist traces a small circle in front of the body
    and the fingers cyclically open/close. Useful to verify arms-follow + grasp
    end-to-end without a Quest."""

    def __init__(self, rate_hz: float = 72.0, t0: float = 0.0):
        self.rate = rate_hz
        self._t0 = t0
        self._lock = threading.Lock()
        self._frame: VRFrame | None = None
        self._run = False
        self._thread: threading.Thread | None = None

    def frame_at(self, t: float, stamp: float | None = None) -> VRFrame:
        """`t` drives the synthetic motion; `stamp` is the freshness clock the
        supervisor reads (defaults to `t` for deterministic headless runs)."""
        hands = {}
        for side in SIDES:
            s = 1.0 if side == "left" else -1.0
            # wrist circles in the headset frame (x right, y up, z toward operator)
            cx = 0.10 * math.sin(t * 0.7) + 0.15 * s
            cy = 0.08 * math.cos(t * 0.9)
            cz = 0.06 * math.sin(t * 1.1)
            curl = 0.5 - 0.5 * math.cos(t * 1.5)
            hands[side] = HandSample(
                tracked=True,
                wrist=_wrist_mat([cx, cy, cz]),
                landmarks=synthetic_webxr_hand(curl),
                pinch=curl,
            )
        return VRFrame(stamp=(t if stamp is None else stamp), head=np.eye(4), hands=hands)

    def latest(self) -> VRFrame | None:
        if self._thread is None:                 # synchronous mode: compute on demand
            now = time.monotonic()
            return self.frame_at(now - self._t0 if self._t0 else 0.0, stamp=now)
        with self._lock:
            return self._frame

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._run:
            now = time.monotonic()
            f = self.frame_at(now - self._t0, stamp=now)   # stamp on the supervisor's clock
            with self._lock:
                self._frame = f
            time.sleep(1.0 / self.rate)

    def stop(self) -> None:
        self._run = False


def make_source(rig: dict) -> VRSource:
    transport = rig.get("vr", {}).get("transport", "fake")
    if transport == "fake":
        return FakeVRSource(rate_hz=rig["control"]["vr_hz"])
    if transport == "vuer":
        from .vuer_source import VuerVRSource
        return VuerVRSource(rig)
    if transport == "orbit":
        from .orbit_source import OrbitVRSource
        return OrbitVRSource(rig)
    if transport == "replay":
        from .replay import ReplaySource
        path = rig.get("vr", {}).get("replay_path")
        if not path:
            raise ValueError("vr.transport='replay' requires vr.replay_path in the rig config")
        return ReplaySource(path, loop=rig["vr"].get("replay_loop", False),
                            speed=float(rig["vr"].get("replay_speed", 1.0)))
    raise ValueError(f"unknown vr.transport: {transport!r}")
