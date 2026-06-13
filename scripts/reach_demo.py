#!/usr/bin/env python
"""Synthesize a substantial-but-bounded replay episode: raise to chest height,
reach forward, lateral sweep, vertical bob, return to rest. Right hand only,
constant attitude, smooth cosine blends between keyframes. Contract-clean by
construction (same frame builders as the fake source, start at the robot's
rest-wrist correspondence)."""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.config import load_rig
from bimanual_teleop.arms.ik import ArmIK
from bimanual_teleop.vr.frames import quat_to_R
from bimanual_teleop.vr.ingest import _wrist_mat, synthetic_webxr_hand, HandSample, VRFrame
from bimanual_teleop.vr.replay import SessionRecorder

rig = load_rig()
ik = ArmIK(rig, "right")
base_R = quat_to_R(rig["arms"]["right"]["base_quat"])
base_p = np.asarray(rig["arms"]["right"]["base_pos"], float)
w_rest = base_R @ ik.fk_wrist().translation() + base_p
drop = float(rig["mapping"].get("body_anchor_drop", 0.15))
anchor = 0.5 * (np.asarray(rig["arms"]["left"]["base_pos"], float)
                + np.asarray(rig["arms"]["right"]["base_pos"], float)) - np.array([0.0, 0.0, drop])
d = (w_rest - anchor) / float(rig["mapping"].get("pos_scale", 1.0))
wb_rest = np.array([d[1], d[2], -d[0]])            # body axes [right, up, fwd]

tfh = np.asarray(rig["vr"].get("torso_from_head", [0.0, -0.35, 0.0]), float)
torso = np.array([tfh[0], tfh[1], -tfh[2]])


def webxr(wb):
    return torso + np.array([wb[0], wb[1], -wb[2]])


# Keyframes: (t_seconds, offset from rest in body axes [right, up, fwd], metres)
KEY = [
    (0.0,  [0.00, 0.00, 0.00]),     # hang
    (2.0,  [0.00, 0.00, 0.00]),     # hold
    (9.0,  [0.02, 0.30, 0.10]),     # raise to lower-chest height, slight forward
    (15.0, [0.05, 0.32, 0.25]),     # reach forward
    (20.0, [0.15, 0.30, 0.22]),     # sweep right
    (26.0, [-0.05, 0.30, 0.22]),    # sweep left through center
    (30.0, [0.02, 0.32, 0.22]),     # back to center
    (34.0, [0.02, 0.24, 0.20]),     # bob down
    (38.0, [0.02, 0.36, 0.20]),     # bob up
    (44.0, [0.00, 0.00, 0.00]),     # return to hang
    (46.0, [0.00, 0.00, 0.00]),     # hold
]


def offset_at(t: float) -> np.ndarray:
    for (t0, a), (t1, b) in zip(KEY, KEY[1:]):
        if t0 <= t <= t1:
            u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            s = 0.5 - 0.5 * np.cos(np.pi * u)          # cosine ease in/out
            return (1 - s) * np.asarray(a) + s * np.asarray(b)
    return np.asarray(KEY[-1][1], float)


HZ = 72.0
T = KEY[-1][0]
rec = SessionRecorder()
n = int(T * HZ)
peak_v = 0.0
prev = None
for i in range(n):
    t = i / HZ
    wb = wb_rest + offset_at(t)
    if prev is not None:
        peak_v = max(peak_v, float(np.linalg.norm(wb - prev)) * HZ)
    prev = wb
    hands = {
        "right": HandSample(tracked=True, wrist=_wrist_mat(webxr(wb)),
                            landmarks=synthetic_webxr_hand(0.15), pinch=0.0),
        "left": HandSample(tracked=False, wrist=np.eye(4),
                           landmarks=synthetic_webxr_hand(0.15), pinch=0.0),
    }
    rec.add(VRFrame(stamp=t, head=np.eye(4), hands=hands), {"left": False, "right": t > 1.0}, t)

out = rec.save("recordings/reach_demo.npz")
print(f"saved {out}: {n} frames, {T:.0f}s, peak hand speed {peak_v * 100:.1f} cm/s "
      f"(x0.2 replay -> {peak_v * 20:.1f} cm/s on metal)")
