"""Real Quest 3 / 3S hand-tracking via the **ORBIT** native app (com.ORBIT.Teleoperation),
as an alternative to the Vuer/WebXR browser source.

ORBIT is an installed APK that PUSHes operator tracking over ZeroMQ/NetMQ to
127.0.0.1; with `adb reverse tcp:<p> tcp:<p>` (USB) those land on this machine,
where we BIND a PULL socket per channel. Verified wire format (GestureDetector.cs):

    8087 / 8088   right / left hand   "relative:x,y,z|...|x,y,z:"  = 26 joint POSITIONS
    8122 / 8123   right / left wrist   "relative,x,y,z,qx,qy,qz,qw" (pose, XYZW quat)
    8200          head pose            "relative,x,y,z,qx,qy,qz,qw"
    8095 / 8100   resolution / pause   strings (drained so ORBIT's PUSH sends don't stall)

Frames (see project memory): ORBIT is **Unity** — x-right, y-up, **+z FORWARD,
LEFT-handed**. Our engine wants the **WebXR** frame — x-right, y-up, **-z forward,
RIGHT-handed** (see frames.WEBXR_TO_WORLD / FakeVRSource "z toward operator").
The map between them is a single Z-flip, which converts handedness AND the forward
axis together. We apply it as a congruence (rotation + translation together), the
same trick the reference bridge (Orca-Yam-teleop/orca-teleop/vr_zmq.py) uses:

    M_webxr = S4 @ M_unity @ S4,   S4 = diag(1, 1, -1, 1)

`orbit_flip` (rig vr.orbit_flip, default "z") is an empirical knob: if motion is
mirrored on hardware, change which axis S negates.

FRAME ORIGINS (measured on a real session, 2026-06-11 — recordings/live_0611_124653):
ORBIT streams the HEAD pose in a floor-origin world frame (standing head y ≈ 1.4)
but the WRIST poses in an EYE-HEIGHT anchored frame (the Unity XR recenter origin:
resting hands read y ≈ −0.3, NOT ≈ +1.0). The old note here claimed the offset
cancels because "the clutch mapper is RELATIVE" — that stopped being true when the
mapping went ABSOLUTE: subtracting the head-derived torso from a wrist in a
different origin put a phantom ~−1 m on the body-relative UP axis (operator
calibration then measured arms-forward as hip-height). `latest()` therefore
re-anchors each wrist TRANSLATION at the live head position (rotation untouched —
wrist attitudes were already world-axes and the verified absolute-orientation
mapping worked). Exact when the operator's posture matches the recenter pose;
a deep crouch/lean couples in by the head's displacement since recenter (the
hand keypoints are wrist-relative downstream, so they need no fix). Set
`vr.orbit_wrist_anchor: world` to restore raw passthrough for a future ORBIT
build that streams both poses in one frame.

ORBIT sends **26** joints (it adds a `Palm` at index 1 that WebXR lacks); dropping
index 1 yields exactly the **25** WebXR joints in W3C order, so landmark indices
(thumb-tip 4, index-tip 9, middle-prox 11) line up with the rest of the pipeline.
"""
from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
import threading
import time

import numpy as np

from ..config import SIDES
from .frames import HandSample, VRFrame, quat_to_R
from .ingest import VRSource

# Channel -> port. All OUTBOUND ports must be bound or ORBIT's blocking PUSH sends stall.
HAND_PORTS = {"right": 8087, "left": 8088}
WRIST_PORTS = {"right": 8122, "left": 8123}
HEAD_PORT = 8200
DRAIN_PORTS = (8095, 8100)
# In-headset screen view (scripts/headset_view.py binds its own PUB here) —
# tunnel-only: the link watchdog keeps its reverse alive, but it is NOT bound here.
VIDEO_PANEL_PORT = 10505
_ALL_PORTS = (*HAND_PORTS.values(), *WRIST_PORTS.values(), HEAD_PORT, *DRAIN_PORTS)
_TUNNEL_PORTS = (*_ALL_PORTS, VIDEO_PANEL_PORT)


def _S4(flip: str = "z") -> np.ndarray:
    """4x4 handedness-flip (congruence) for flip in {x,y,z,none}."""
    s = np.ones(3)
    if flip in ("x", "y", "z"):
        s["xyz".index(flip)] = -1.0
    M = np.eye(4)
    M[:3, :3] = np.diag(s)
    return M


def _parse_hand(raw: str) -> np.ndarray | None:
    """'relative:x,y,z|...|x,y,z:' -> (26,3) array, or None."""
    body = raw.split(":", 1)[1] if ":" in raw else raw
    pts = [t for t in body.strip(":").split("|") if t.strip()]
    if not pts:
        return None
    try:
        arr = np.array([[float(v) for v in t.split(",")] for t in pts], dtype=float)
    except ValueError:
        return None
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _parse_pose(raw: str) -> tuple[np.ndarray, np.ndarray] | None:
    """'relative,x,y,z,qx,qy,qz,qw' -> (pos[3], quat_xyzw[4]) or None."""
    t = raw.split(",")
    if len(t) != 8:
        return None
    try:
        pos = np.array([float(t[1]), float(t[2]), float(t[3])])
        q = np.array([float(t[4]), float(t[5]), float(t[6]), float(t[7])])
    except ValueError:
        return None
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(q)):
        return None
    n = np.linalg.norm(q)
    return (pos, q / n) if n > 1e-9 else None


def _pinch_from_landmarks(lm: np.ndarray | None) -> float:
    """Thumb-tip<->index-tip distance normalized by hand size -> 1 pinched, 0 open
    (matches vuer_source._pinch_from_landmarks; same 25-joint WebXR indices)."""
    if lm is None or len(lm) < 25:
        return 0.0
    d = np.linalg.norm(lm[4] - lm[9])              # thumb tip (4) <-> index tip (9)
    scale = np.linalg.norm(lm[11] - lm[0]) + 1e-6  # wrist (0) -> middle proximal (11)
    return float(np.clip((0.6 - d / scale) / (0.6 - 0.2), 0.0, 1.0))


class OrbitVRSource(VRSource):
    """ORBIT Quest app -> VRFrame. One PULL bind per channel (latest-wins), Unity->WebXR
    congruence, Palm dropped. Engine-side calibration + clutch do the rest."""

    def __init__(self, rig: dict, debug: bool = False):
        v = rig.get("vr", {})
        self.debug = bool(v.get("debug", debug))
        self.timeout = float(v.get("orbit_timeout", 0.3))   # s before a hand reads "not tracked"
        self.head_timeout = float(v.get("orbit_head_timeout", max(1.0, self.timeout)))
        self.auto_reverse = bool(v.get("orbit_adb_reverse", True))
        self.S4 = _S4(str(v.get("orbit_flip", "z")))
        # Wrist translation reconstruction (see module docstring):
        #  'keypoint' (default): head_pos + hand keypoint[0] — the keypoints are
        #     streamed HEAD-ANCHORED (world axes), so this needs NO assumption
        #     about the wrist stream's recenter anchor. Measured failure of
        #     'head': starting/recentering ORBIT with the headset on the desk
        #     moved the eye-anchor ~0.5 m down and every wrist read 0.5 m high
        #     (calibration refused; mapping shifted).
        #  'head': wrist-stream translation + live head (exact only while the
        #     live head matches the recenter pose). 'world': raw passthrough.
        self.wrist_anchor = str(v.get("orbit_wrist_anchor", "keypoint"))
        self.viz_port = int(v.get("orbit_viz_port", 8099))
        self.viz_enabled = bool(v.get("orbit_viz", True))
        self.viz_url = None
        self._viz_srv = None
        self.overlay = {}          # optional debug/calibration overlays rendered by orbit_viz
        self._lock = threading.Lock()
        self._wrist: dict[str, np.ndarray | None] = {s: None for s in SIDES}
        self._lm: dict[str, np.ndarray | None] = {s: None for s in SIDES}
        self._wrist_last: dict[str, float] = {s: 0.0 for s in SIDES}
        self._lm_last: dict[str, float] = {s: 0.0 for s in SIDES}
        self._head = np.eye(4)
        self._head_last = 0.0
        self.counts = {"hand": 0, "wrist": 0, "head": 0}
        self._ctx = None
        self._stop = False
        self._thread: threading.Thread | None = None

    # ---- Unity -> WebXR conversions ---------------------------------------- #
    def _conv_pose(self, pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        M = np.eye(4)
        qx, qy, qz, qw = quat_xyzw
        M[:3, :3] = quat_to_R((qw, qx, qy, qz))   # quat_to_R wants (w,x,y,z)
        M[:3, 3] = pos
        return self.S4 @ M @ self.S4

    def _conv_points(self, pts: np.ndarray) -> np.ndarray:
        # Nx3: same diagonal flip as the pose congruence (points transform under S)
        return pts * np.diag(self.S4[:3, :3])

    # ---- VRSource API ------------------------------------------------------ #
    def latest(self) -> VRFrame | None:
        now = time.monotonic()
        with self._lock:
            head = self._head.copy() if self._head_last > 0 and (now - self._head_last) < self.head_timeout else None
            hands = {}
            for s in SIDES:
                w = self._wrist[s]
                wrist_fresh = w is not None and (now - self._wrist_last[s]) < self.timeout
                lm = self._lm[s] if (now - self._lm_last[s]) < self.timeout else None
                if wrist_fresh:
                    if head is not None and self.wrist_anchor != "world":
                        # Rebuild the TRANSLATION in one consistent origin (the
                        # body-relative subtraction downstream then cancels the
                        # head exactly). Rotation is already world-axes — the
                        # wrist stream's attitude is kept untouched.
                        w = w.copy()
                        if self.wrist_anchor == "keypoint" and lm is not None:
                            w[:3, 3] = head[:3, 3] + lm[0]   # anchor-independent
                        else:
                            w[:3, 3] += head[:3, 3]          # 'head' / no landmarks
                    hands[s] = HandSample(tracked=True, wrist=w, landmarks=lm,
                                          pinch=_pinch_from_landmarks(lm))
                else:
                    hands[s] = HandSample(tracked=False)
            return VRFrame(stamp=now, head=head, hands=hands)

    def start(self) -> None:
        if self.auto_reverse:
            self._adb_reverse()
        self._banner()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if self.auto_reverse:
            threading.Thread(target=self._adb_watchdog, daemon=True).start()
        if self.viz_enabled and self._viz_srv is None:
            try:
                from .orbit_viz import start_viz
                self._viz_srv, self.viz_url = start_viz(self.viz_snapshot, self.viz_port)
                print(f"[orbit] hand viz: {self.viz_url}", flush=True)
            except OSError as e:
                print(f"[orbit] viz disabled (port {self.viz_port} busy: {e})", flush=True)

    @staticmethod
    def _mat_to_pos_quat(M):
        """4x4 -> (pos[3], quat_xyzw[4]) for the viz."""
        m = M[:3, :3]
        t = float(np.trace(m))
        if t > 0:
            s = np.sqrt(t + 1.0) * 2; w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
        else:
            i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
            if i == 0:
                s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
                w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
            elif i == 1:
                s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
                w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
            else:
                s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
                w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
        return M[:3, 3].round(4).tolist(), [float(round(x, 4)), float(round(y, 4)),
                                            float(round(z, 4)), float(round(w, 4))]

    def viz_snapshot(self) -> dict:
        """Live state as JSON-able dict for the browser viz (WebXR frame)."""
        now = time.monotonic()
        with self._lock:
            def hand(side):
                lm = self._lm[side]
                if lm is None:
                    return None
                d = {"pts": lm.round(4).tolist(), "age": round(now - self._lm_last[side], 3)}
                if self._wrist[side] is not None:
                    pos, quat = self._mat_to_pos_quat(self._wrist[side])
                    d["wrist"] = {"pos": pos, "quat": quat, "age": round(now - self._wrist_last[side], 3)}
                return d
            head = None
            if self._head_last > 0:
                pos, quat = self._mat_to_pos_quat(self._head)
                head = {"pos": pos, "quat": quat, "age": round(now - self._head_last, 3)}
            return {"hands": {"right": hand("right"), "left": hand("left")}, "head": head,
                    "overlay": dict(self.overlay)}

    def stop(self) -> None:
        self._stop = True

    # ---- internals --------------------------------------------------------- #
    def _adb_reverse(self) -> None:
        if not shutil.which("adb"):
            print("[orbit] adb not found — set up `adb reverse` for "
                  f"{', '.join(map(str, _TUNNEL_PORTS))} yourself.", flush=True)
            return
        ok = 0
        for p in _TUNNEL_PORTS:
            try:
                r = subprocess.run(["adb", "reverse", f"tcp:{p}", f"tcp:{p}"],
                                   capture_output=True, timeout=5)
                ok += (r.returncode == 0)
            except (subprocess.SubprocessError, OSError):
                pass
        print(f"[orbit] adb reverse set on {ok}/{len(_TUNNEL_PORTS)} ports.", flush=True)

    @staticmethod
    def _adb_device_state() -> str:
        """'device' | 'unauthorized' | 'absent' | 'no-adb'."""
        if not shutil.which("adb"):
            return "no-adb"
        try:
            r = subprocess.run(["adb", "get-state"], capture_output=True, timeout=3, text=True)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
            return "unauthorized" if "unauthorized" in (r.stderr or "") else "absent"
        except (subprocess.SubprocessError, OSError):
            return "absent"

    def _adb_watchdog(self) -> None:
        """Self-heal the Quest link: `adb reverse` forwards silently die when the
        headset disconnects (cable, sleep, adb server restart) and nothing brings
        them back — the stream then stays dead until a full engine restart. Poll
        the device every few seconds and re-assert any missing forwards."""
        state = "device"                       # start() just ran _adb_reverse()
        while not self._stop:
            for _ in range(12):                # 3 s, responsive to stop()
                if self._stop:
                    return
                time.sleep(0.25)
            new = self._adb_device_state()
            if new == "no-adb":
                return
            if new == "device":
                try:
                    r = subprocess.run(["adb", "reverse", "--list"],
                                       capture_output=True, timeout=3, text=True)
                    have = {int(a) for a, b in re.findall(r"tcp:(\d+)\s+tcp:(\d+)", r.stdout)
                            if a == b}
                except (subprocess.SubprocessError, OSError, ValueError):
                    have = set()
                missing = [p for p in _TUNNEL_PORTS if p not in have]
                if missing:
                    print(f"[orbit] quest link restored, reverses missing {missing} "
                          "— re-asserting", flush=True)
                    self._adb_reverse()
            elif new != state:
                hint = " — put the headset ON and tap 'Allow USB debugging'" \
                    if new == "unauthorized" else " — check the USB cable"
                print(f"[orbit] quest adb state: {new}{hint}", flush=True)
            state = new

    def _banner(self) -> None:
        print("\n" + "=" * 64
              + "\n  ORBIT source: launch com.ORBIT.Teleoperation on the Quest, then\n"
              "  WEAR it and SET BOTH CONTROLLERS DOWN (hand tracking only streams\n"
              "  with controllers asleep). Hold both hands out front for calibration.\n"
              + "=" * 64 + "\n", flush=True)

    def _run(self) -> None:
        import zmq
        self._ctx = zmq.Context.instance()
        poller = zmq.Poller()
        socks = {}
        for s, p in HAND_PORTS.items():
            socks[self._bind(p)] = ("hand", s)
        for s, p in WRIST_PORTS.items():
            socks[self._bind(p)] = ("wrist", s)
        socks[self._bind(HEAD_PORT)] = ("head", None)
        for p in DRAIN_PORTS:
            socks[self._bind(p)] = ("drain", p)
        for sock in socks:
            poller.register(sock, zmq.POLLIN)
        if self.debug:
            print(f"[orbit] bound PULL on {sorted(_ALL_PORTS)}", flush=True)

        last_dbg = 0.0
        while not self._stop:
            events = dict(poller.poll(timeout=200))
            now = time.monotonic()
            for sock, (kind, key) in socks.items():
                if sock not in events:
                    continue
                msg = self._drain(sock)               # newest message only
                if msg is None:
                    continue
                self._ingest(kind, key, msg, now)
            if self.debug and now - last_dbg > 2.0:
                last_dbg = now
                print(f"[orbit] hands={self.counts['hand']} wrists={self.counts['wrist']} "
                      f"heads={self.counts['head']}", flush=True)
        for sock in socks:
            sock.close(0)

    def _bind(self, port: int):
        import zmq
        sock = self._ctx.socket(zmq.PULL)
        sock.bind(f"tcp://127.0.0.1:{port}")
        return sock

    @staticmethod
    def _drain(sock):
        import zmq
        msg = None
        while True:
            try:
                msg = sock.recv_string(zmq.NOBLOCK)
            except zmq.Again:
                return msg

    def _ingest(self, kind: str, key, msg: str, now: float) -> None:
        if kind == "hand":
            pts = _parse_hand(msg)
            if pts is None or len(pts) < 26:
                return
            lm = self._conv_points(np.delete(pts, 1, axis=0))   # drop Palm[1] -> 25 WebXR joints
            with self._lock:
                self._lm[key] = lm
                self._lm_last[key] = now
            self.counts["hand"] += 1
        elif kind == "wrist":
            parsed = _parse_pose(msg)
            if parsed is None:
                return
            M = self._conv_pose(*parsed)
            with self._lock:
                self._wrist[key] = M
                self._wrist_last[key] = now
            self.counts["wrist"] += 1
        elif kind == "head":
            parsed = _parse_pose(msg)
            if parsed is not None:
                with self._lock:
                    self._head = self._conv_pose(*parsed)
                    self._head_last = now
                self.counts["head"] += 1
        # drain: discard


# --------------------------------------------------------------------------- #
# Headless self-test: transport round-trip + Unity->WebXR conversion + palm drop
# --------------------------------------------------------------------------- #
def selftest() -> int:
    import zmq

    ok = True
    rig = {"vr": {"orbit_flip": "z", "orbit_adb_reverse": False, "orbit_timeout": 2.0}}
    src = OrbitVRSource(rig)
    # rebind to test ports to avoid clashing with a live run
    global HAND_PORTS, WRIST_PORTS, HEAD_PORT, DRAIN_PORTS
    HAND_PORTS = {"right": 18087, "left": 18088}
    WRIST_PORTS = {"right": 18122, "left": 18123}
    HEAD_PORT = 18200
    DRAIN_PORTS = (18095, 18100)
    src.start()
    time.sleep(0.4)

    ctx = zmq.Context.instance()
    pushers = {p: ctx.socket(zmq.PUSH) for p in (18087, 18122)}
    for p, s in pushers.items():
        s.connect(f"tcp://127.0.0.1:{p}")
    time.sleep(0.2)

    # 26 keypoints: encode a recognizable index ramp on z; Palm (index 1) is a sentinel
    pts = np.zeros((26, 3))
    pts[:, 2] = np.arange(26) * 0.01           # z = +0.00,+0.01,... (Unity forward)
    pts[1] = [9.0, 9.0, 9.0]                    # Palm sentinel — must be dropped
    hand_msg = "relative:" + "|".join(f"{x},{y},{z}" for x, y, z in pts) + ":"
    pushers[18087].send_string(hand_msg)
    # wrist pose: Unity (0.1,0.2,0.3), identity rot -> WebXR z flips -> (0.1,0.2,-0.3)
    pushers[18122].send_string("relative,0.1,0.2,0.3,0,0,0,1")
    time.sleep(0.3)

    f = src.latest()
    hr = f.hands["right"]
    t_tracked = hr.tracked and hr.landmarks is not None
    t_palm = hr.landmarks.shape == (25, 3) and not np.allclose(hr.landmarks[1], [9, 9, -9])
    # WebXR z = -Unity z: joint k (k>=2 in orig) lands at index k-1 with z=-(k*0.01)
    t_zflip = np.isclose(hr.landmarks[1, 2], -0.02) and np.isclose(hr.landmarks[24, 2], -0.25)
    t_wrist = np.allclose(hr.wrist[:3, 3], [0.1, 0.2, -0.3], atol=1e-6)
    print(f"  tracked + landmarks present:           {'ok' if t_tracked else 'BAD'}")
    print(f"  Palm[1] dropped -> (25,3):             {'ok' if t_palm else 'BAD'}")
    print(f"  Unity->WebXR z-flip on landmarks:      {'ok' if t_zflip else 'BAD'}")
    print(f"  wrist (0.1,0.2,0.3)->(0.1,0.2,-0.3):   {'ok' if t_wrist else 'BAD'}")
    ok &= bool(t_tracked and t_palm and t_zflip and t_wrist)
    src.stop()
    print("ORBIT-SOURCE SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
