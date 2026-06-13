#!/usr/bin/env python
"""Bimanual GIF-style episode: both hands raise, reach, sweep, roll wrists,
bob, curl fingers, return. Right arm drives the metal; left exists for the
render only (hardware sink drops it). Proper rotations throughout."""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.config import load_rig, SIDES
from bimanual_teleop.arms.ik import ArmIK
from bimanual_teleop.vr.frames import quat_to_R
from bimanual_teleop.vr.ingest import _wrist_mat, synthetic_webxr_hand, HandSample, VRFrame
from bimanual_teleop.vr.replay import SessionRecorder

rig = load_rig()


def rest_wb(side: str) -> np.ndarray:
    ik = ArmIK(rig, side)
    base_R = quat_to_R(rig["arms"][side]["base_quat"])
    base_p = np.asarray(rig["arms"][side]["base_pos"], float)
    w = base_R @ ik.fk_wrist().translation() + base_p
    drop = float(rig["mapping"].get("body_anchor_drop", 0.15))
    anchor = 0.5 * (np.asarray(rig["arms"]["left"]["base_pos"], float)
                    + np.asarray(rig["arms"]["right"]["base_pos"], float)) - np.array([0.0, 0.0, drop])
    d = (w - anchor) / float(rig["mapping"].get("pos_scale", 1.0))
    return np.array([d[1], d[2], -d[0]])        # body axes [right, up, fwd]


WB0 = {s: rest_wb(s) for s in SIDES}
tfh = np.asarray(rig["vr"].get("torso_from_head", [0.0, -0.35, 0.0]), float)
TORSO = np.array([tfh[0], tfh[1], -tfh[2]])


def webxr_pos(wb):
    return TORSO + np.array([wb[0], wb[1], -wb[2]])


def rot(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


# Keyframes for the RIGHT hand: (t, offset [right, up, fwd]); left mirrors right axis.
KEY = [
    (0.0,  [0.00, 0.00, 0.00]),
    (2.0,  [0.00, 0.00, 0.00]),
    (8.0,  [0.02, 0.30, 0.12]),     # raise to lower-chest
    (13.0, [0.05, 0.32, 0.25]),     # reach forward
    (18.0, [0.14, 0.30, 0.22]),     # sweep out
    (24.0, [-0.04, 0.30, 0.22]),    # sweep in past center
    (28.0, [0.03, 0.32, 0.22]),     # recenter   (wrist ROLL plays 18-30s)
    (33.0, [0.03, 0.24, 0.20]),     # bob down   (wrist PITCH plays 30-38s)
    (38.0, [0.03, 0.36, 0.20]),     # bob up
    (44.0, [0.00, 0.00, 0.00]),     # return to hang
    (46.0, [0.00, 0.00, 0.00]),
]


def offset_at(t):
    for (t0, a), (t1, b) in zip(KEY, KEY[1:]):
        if t0 <= t <= t1:
            u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            s = 0.5 - 0.5 * np.cos(np.pi * u)
            return (1 - s) * np.asarray(a) + s * np.asarray(b)
    return np.asarray(KEY[-1][1], float)


def ramp(t, t0, t1):
    """0..1 cosine window inside [t0, t1], 0 outside."""
    if t <= t0 or t >= t1:
        return 0.0
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * (t - t0) / (t1 - t0))


HZ = 72.0
T = KEY[-1][0]
rec = SessionRecorder()
for i in range(int(T * HZ)):
    t = i / HZ
    off = offset_at(t)
    roll = np.radians(35.0) * np.sin(2.0 * np.pi * (t - 18.0) / 6.0) * ramp(t, 18.0, 30.0)
    pitch = np.radians(20.0) * np.sin(2.0 * np.pi * (t - 30.0) / 4.0) * ramp(t, 30.0, 38.0)
    curl = 0.15 + 0.35 * (0.5 - 0.5 * np.cos(2.0 * np.pi * t / 8.0)) * (1.0 if 4.0 < t < 42.0 else 0.0)
    hands = {}
    for side in SIDES:
        m = 1.0 if side == "right" else -1.0
        wb = WB0[side] + off * np.array([m, 1.0, 1.0])
        # wrist attitude: roll about the forward axis (webxr -z), pitch about right (+x)
        R = rot([0, 0, -1], m * roll) @ rot([1, 0, 0], pitch)
        hands[side] = HandSample(tracked=True, wrist=_wrist_mat(webxr_pos(wb), R),
                                 landmarks=synthetic_webxr_hand(curl), pinch=0.0)
    rec.add(VRFrame(stamp=t, head=np.eye(4), hands=hands),
            {"left": t > 1.0, "right": t > 1.0}, t)

out = rec.save("recordings/bimanual_demo.npz")
print(f"saved {out}: {int(T * HZ)} frames, {T:.0f}s, wrist roll ±35° @18-30s, pitch ±20° @30-38s, finger curls")
