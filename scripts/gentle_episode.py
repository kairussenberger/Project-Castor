#!/usr/bin/env python
"""Generate a GENTLE synthetic replay episode for the first metal replay.

Start pose maps exactly onto the robot's rest wrist (minimal engage-glide),
3 s hold, one slow 5 cm circle in the right/up plane over 12 s, 3 s hold.
Constant identity wrist attitude -> near-zero orientation travel. Right hand
only; left is untracked (and gated off at the hardware sink anyway).
"""
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
wb_rest = np.array([d[1], d[2], -d[0]])        # world -> body axes: right=+Y, up=+Z, fwd=-X

tfh = np.asarray(rig["vr"].get("torso_from_head", [0.0, -0.35, 0.0]), float)
torso = np.array([tfh[0], tfh[1], -tfh[2]])    # [right,up,fwd] -> webxr xyz (head = identity)


def webxr(wb):
    return torso + np.array([wb[0], wb[1], -wb[2]])


HZ, HOLD, CIRC, RAD = 72.0, 3.0, 12.0, 0.05
T = HOLD + CIRC + 3.0
rec = SessionRecorder()
n = int(T * HZ)
for i in range(n):
    t = i / HZ
    wb = wb_rest.copy()
    if HOLD <= t < HOLD + CIRC:
        ph = 2.0 * np.pi * (t - HOLD) / CIRC
        wb = wb_rest + np.array([RAD * np.sin(ph),                  # lateral +-5 cm
                                 0.5 * RAD * (1.0 - np.cos(ph)),    # vertical 0..5 cm..0
                                 0.0])
    hands = {
        "right": HandSample(tracked=True, wrist=_wrist_mat(webxr(wb)),
                            landmarks=synthetic_webxr_hand(0.15), pinch=0.0),
        "left": HandSample(tracked=False, wrist=np.eye(4),
                           landmarks=synthetic_webxr_hand(0.15), pinch=0.0),
    }
    frame = VRFrame(stamp=t, head=np.eye(4), hands=hands)
    rec.add(frame, {"left": False, "right": t > 1.0}, t)

out = rec.save("recordings/gentle_right.npz")
print(f"saved {out}: {n} frames, {T:.0f}s")
print(f"rest wrist (world) {np.round(w_rest, 3).tolist()}  ->  wrist_body {np.round(wb_rest, 3).tolist()}")
