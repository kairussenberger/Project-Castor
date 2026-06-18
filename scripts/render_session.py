#!/usr/bin/env python
"""Render a recorded teleop session as a watchable side-by-side movie/sheet:
LEFT = the operator's tracked hands (25 Quest joints each + wrist frames, placed
torso-relative), RIGHT = the robot with REAL YAM mesh geometry (from the MJCF
source assets) following them through the actual TeleopEngine. No headset, no
Unity, no Rerun — just files you can open.

    uv run --with matplotlib python scripts/render_session.py recordings/session.npz
    uv run --with matplotlib python scripts/render_session.py s.npz --gif out/s.gif --fps 13
    uv run --with matplotlib python scripts/render_session.py s.npz --sheet out/s.png
    uv run --with matplotlib python scripts/render_session.py s.npz --static out/frame.png --at 8.3

matplotlib is intentionally NOT a project dependency — run via `uv run --with
matplotlib`. Robot hands are drawn as EE triads (the ORCA model dir ships no
meshes); finger tracking is visible on the operator side.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pinocchio as pin                                              # noqa: E402

from bimanual_teleop.config import SIDES, load_rig                   # noqa: E402
from bimanual_teleop.engine import TeleopEngine                      # noqa: E402
from bimanual_teleop.vr.calibrate import body_relative_hand_sample, head_op_axes  # noqa: E402
from bimanual_teleop.vr.frames import quat_to_R, rotvec              # noqa: E402
from bimanual_teleop.vr.replay import ReplaySource                   # noqa: E402

from bimanual_teleop.viz.hand_geom import hand_basis_for_side, orca_hand_tris_ee  # noqa: E402
from bimanual_teleop.viz.yam_meshes import (  # noqa: E402
    geom_transforms, load_arm_meshes, load_orca_hand, load_stand_meshes,
    orca_description_available, orca_q_from_degrees, world_tris)

# WebXR 25-joint finger chains (W3C order), wrist = 0
FINGER_CHAINS = [([0, 1, 2, 3, 4], "#d4699e"), ([0, 5, 6, 7, 8, 9], "#4a90d9"),
                 ([0, 10, 11, 12, 13, 14], "#2aa198"), ([0, 15, 16, 17, 18, 19], "#caa520"),
                 ([0, 20, 21, 22, 23, 24], "#a66cc9")]
TRIAD_RGB = ("#e03030", "#1faa1f", "#3060ff")
ARM_RGB = {"left": (0.62, 0.74, 0.93), "right": (0.92, 0.56, 0.35)}


class NullSink:
    def __init__(self):
        self.hands = {}

    def set_arm(self, side, q):
        pass

    def set_hand(self, side, joints):
        self.hands[side] = dict(joints)


def shaded_colors(tris: np.ndarray, base_rgb, light=(0.4, 0.3, 0.85)) -> np.ndarray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    light = np.asarray(light) / np.linalg.norm(light)
    lam = np.clip(np.abs(n @ light), 0.0, 1.0) * 0.65 + 0.35     # double-sided, ambient floor
    cols = np.clip(lam[:, None] * np.asarray(base_rgb)[None, :], 0, 1)
    return np.concatenate([cols, np.ones((len(cols), 1))], axis=1)


# --------------------------------------------------------------------------- #
# Capture: replay the session through the real engine, recording everything the
# panels need per frame.
# --------------------------------------------------------------------------- #
def capture(path: str, rig: dict):
    src = ReplaySource(path)
    if src.calib:
        # Replay through the calibration that ran during the session — raw
        # ORBIT frames are only meaningful with their fit (see REPLAY_LIBRARY).
        rig["vr"]["_embedded_calib"] = src.calib
        print("[render] applying the session calibration embedded in the recording")
    sink = NullSink()
    engine = TeleopEngine(rig, sink)
    torso = np.asarray(rig["vr"]["torso_from_head"], float)
    blend = float(rig.get("mapping", {}).get("engage_blend_s", 1.0))
    anchors = {s: {"R": None, "ee": None, "prev": None, "t_eng": None, "settled": False}
               for s in SIDES}
    frames = []
    for i in range(len(src.t)):
        t_i = float(src.t[i])
        fr = src.frame_at(t_i)
        engine.tick(fr, src.engaged_at(t_i), t_i)
        rec = {"t": t_i - float(src.t[0]), "q": {}, "hand": {}, "ang": {}, "hand_q": {}, "ee_T": {}}
        for s in SIDES:
            arm = engine.arm[s]
            rec["q"][s] = arm.ik.q
            rec["hand_q"][s] = dict(sink.hands.get(s, {}))
            ee = arm.ik.fk_ee()
            T = np.eye(4)
            T[:3, :3] = arm.base_R @ ee.rotation().as_matrix()
            T[:3, 3] = arm.base_R @ ee.translation() + arm.base_pos
            rec["ee_T"][s] = T
            hs = fr.hands.get(s)
            a = anchors[s]
            m = arm.mapper
            if m.anchor_ctrl is not None and m.anchor_ctrl is not a["prev"] and hs is not None:
                a["prev"] = m.anchor_ctrl
                a["R"] = np.asarray(hs.wrist, float)[:3, :3].copy()
                a["ee"] = m.anchor_ee.rotation().as_matrix()
                a["t_eng"], a["settled"] = t_i, False
            elif (not a["settled"] and a["t_eng"] is not None
                  and t_i - a["t_eng"] > 2.0 * blend
                  and hs is not None and hs.tracked and arm.cmd_R is not None):
                # Re-zero the readout reference POST-GLIDE: a replay engages
                # from the rest pose, and the engage glide would otherwise bake
                # ~120° of one-time attitude convergence into "rotation since
                # engage" while the hand readout starts at 0.
                a["R"] = np.asarray(hs.wrist, float)[:3, :3].copy()
                a["ee"] = arm.cmd_R.copy()
                a["settled"] = True
            hand_ang = cmd_ang = 0.0
            if a["R"] is not None and arm.cmd_R is not None and hs is not None and hs.tracked:
                hand_ang = np.degrees(np.linalg.norm(rotvec(np.asarray(hs.wrist)[:3, :3] @ a["R"].T)))
                cmd_ang = np.degrees(np.linalg.norm(rotvec(arm.cmd_R @ a["ee"].T)))
            rec["ang"][s] = (hand_ang, cmd_ang)
            body = body_relative_hand_sample(hs, fr.head, torso)
            if body is not None and body.tracked and hs.landmarks is not None and fr.head is not None:
                op = head_op_axes(fr.head)
                rec["hand"][s] = {"lm_body": (hs.landmarks - hs.landmarks[0]) @ op,   # rows·op == opᵀ·v
                                  "wrist_body": body.wrist[:3, 3].copy(),
                                  "R_body": body.wrist[:3, :3].copy()}
        frames.append(rec)
    return frames


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def make_renderer(rig: dict, frames: list, fig, debug_links: bool = False):
    import matplotlib.pyplot as plt  # noqa: F401  (backend chosen by caller)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    arms = {s: load_arm_meshes(s) for s in SIDES}
    stand = load_stand_meshes(rig["stand"]["pos"][2], 600)
    from bimanual_teleop.arms.ik import ArmIK
    from bimanual_teleop.vr.frames import quat_to_R as _q2R
    hand_basis = {}
    orca = {}
    for s in SIDES:
        if orca_description_available():
            orca[s] = load_orca_hand(s, 400)        # the REAL ORCA hand model
        else:
            ikb = ArmIK(rig, s)
            hand_basis[s] = hand_basis_for_side(ikb, _q2R(rig["arms"][s]["base_quat"]), s)
    base_T = {}
    for s in SIDES:
        T = np.eye(4)
        T[:3, :3] = quat_to_R(rig["arms"][s]["base_quat"])
        T[:3, 3] = rig["arms"][s]["base_pos"]
        base_T[s] = T
    axh = fig.add_subplot(1, 2, 1, projection="3d")
    axr = fig.add_subplot(1, 2, 2, projection="3d")

    # Auto-centre the hands panel on THIS session's data: ORBIT anchors put the
    # raw torso-relative joints metres from the origin on some sessions (the
    # embedded calibration absorbs that for the ROBOT side; the legacy fixed
    # box just rendered an empty panel). Same framing, recentred.
    _ws = [f["hand"][s]["wrist_body"] for f in frames for s in SIDES if f["hand"].get(s)]
    _hc = np.median(np.stack(_ws), axis=0) if _ws else np.zeros(3)
    h_off = _hc - np.array([0.0, 0.1, 0.325])      # legacy box centre, body [right, up, fwd]

    def draw(idx: int):
        f = frames[idx]
        axh.cla(), axr.cla()
        # ---- operator: both tracked hands, torso-relative [right, up, forward] --
        for s in SIDES:
            h = f["hand"].get(s)
            if h is None:
                continue
            origin = h["wrist_body"]
            for chain, c in FINGER_CHAINS:
                seg = h["lm_body"][chain] + origin
                axh.plot(seg[:, 0], seg[:, 2], seg[:, 1], "-o", c=c, lw=2.0, ms=2.6)
            for k, c in enumerate(TRIAD_RGB):
                v = origin + 0.05 * h["R_body"][:, k]
                axh.plot([origin[0], v[0]], [origin[2], v[2]], [origin[1], v[1]],
                         c=c, lw=2.8, solid_capstyle="round")
        if (abs(h_off[0]) <= 0.45 and -0.65 <= h_off[2] <= 0.0
                and -0.55 <= h_off[1] <= 0.35):                            # torso proxy, if in view
            axh.scatter([0], [0], [0], c="#caa520", s=40, marker="s")
        axh.set_xlim(-0.45 + h_off[0], 0.45 + h_off[0])
        axh.set_ylim(0.0 + h_off[2], 0.65 + h_off[2])
        axh.set_zlim(-0.35 + h_off[1], 0.55 + h_off[1])
        axh.set_box_aspect((1, 0.72, 1)), axh.view_init(elev=16, azim=-125)
        axh.set_xticks([]), axh.set_yticks([]), axh.set_zticks([])
        la, ra = f["ang"]["left"][0], f["ang"]["right"][0]
        axh.set_title(f"YOUR HANDS — Quest joints, torso-relative\n"
                      f"rotation since engage   L {la:3.0f}°   R {ra:3.0f}°", fontsize=11)
        # ---- robot: real YAM meshes via engine joint state ----------------------
        for tw in stand:
            axr.add_collection3d(Poly3DCollection(
                tw, facecolors=shaded_colors(tw, (0.40, 0.44, 0.50)), edgecolors="none"))
        for s in SIDES:
            model, data, items = arms[s]
            for tw in world_tris(model, data, items, f["q"][s], base_T[s]):
                axr.add_collection3d(Poly3DCollection(
                    tw, facecolors=shaded_colors(tw, ARM_RGB[s]), edgecolors="none"))
            if f.get("hand_q", {}).get(s) is not None and f.get("ee_T", {}).get(s) is not None:
                Th = f["ee_T"][s]
                if s in orca:
                    hmodel, hdata, hitems = orca[s]
                    qh = orca_q_from_degrees(hmodel, f["hand_q"][s], s)
                    for it, T in zip(hitems, geom_transforms(hmodel, hdata, hitems, qh, Th), strict=True):
                        tw = it["tris"] @ T[:3, :3].T + T[:3, 3]
                        axr.add_collection3d(Poly3DCollection(
                            tw, facecolors=shaded_colors(tw, it["rgb"]), edgecolors="none"))
                else:
                    ht = orca_hand_tris_ee(f["hand_q"][s], hand_basis[s], mirror=(s == "left"))
                    ht = ht @ Th[:3, :3].T + Th[:3, 3]
                    axr.add_collection3d(Poly3DCollection(
                        ht, facecolors=shaded_colors(ht, (0.82, 0.77, 0.70)), edgecolors="none"))
            T_ee = base_T[s] @ data.oMi[model.njoints - 1].homogeneous
            for k, c in enumerate(TRIAD_RGB):
                v = T_ee[:3, 3] + 0.11 * T_ee[:3, k]
                axr.plot([T_ee[0, 3], v[0]], [T_ee[1, 3], v[1]], [T_ee[2, 3], v[2]],
                         c=c, lw=2.8, solid_capstyle="round")
            if debug_links:
                jp = np.stack([(base_T[s] @ data.oMi[j].homogeneous)[:3, 3]
                               for j in range(1, model.njoints)])
                axr.plot(jp[:, 0], jp[:, 1], jp[:, 2], "o-", c="k", ms=3, lw=1.2, alpha=0.8)
        axr.set_xlim(-0.62, 0.34), axr.set_ylim(-0.52, 0.44), axr.set_zlim(0.0, 1.45)
        axr.set_box_aspect((1, 1, 1)), axr.view_init(elev=14, azim=-38)
        axr.set_xticks([]), axr.set_yticks([]), axr.set_zticks([])
        lc, rc = f["ang"]["left"][1], f["ang"]["right"][1]
        axr.set_title(f"ROBOT — both YAM arms (real mesh geometry)\n"
                      f"EE rotation since engage   L {lc:3.0f}°   R {rc:3.0f}°", fontsize=11)
        fig.suptitle(f"t = {f['t']:4.1f}s", fontsize=13)

    return draw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="session .npz (run_teleop --record / check_roll --save)")
    ap.add_argument("--gif", metavar="OUT", default=None, help="write an animated GIF")
    ap.add_argument("--sheet", metavar="OUT", default=None, help="write a 5-moment contact sheet PNG")
    ap.add_argument("--static", metavar="OUT", default=None, help="write a single frame PNG")
    ap.add_argument("--at", type=float, default=None, help="timestamp (s) for --static")
    ap.add_argument("--fps", type=float, default=13.0, help="GIF playback fps")
    ap.add_argument("--step", type=int, default=8, help="render every Nth recorded frame for the GIF")
    ap.add_argument("--dpi", type=float, default=88.0)
    ap.add_argument("--debug-links", action="store_true",
                    help="overlay the joint-origin polyline on the meshes (alignment check)")
    args = ap.parse_args()
    if not (args.gif or args.sheet or args.static):
        args.gif = "out/session_render.gif"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    rig = load_rig()
    frames = capture(args.path, rig)
    print(f"[render] captured {len(frames)} engine frames from {args.path}")

    if args.gif:
        fig = plt.figure(figsize=(12.5, 6.2))
        draw = make_renderer(rig, frames, fig, debug_links=args.debug_links)
        idxs = list(range(0, len(frames), max(1, args.step)))
        anim = FuncAnimation(fig, lambda k: draw(idxs[k]), frames=len(idxs))
        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        anim.save(args.gif, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
        plt.close(fig)
        print(f"[render] gif: {args.gif} ({len(idxs)} frames @ {args.fps:.0f}fps)")

    if args.sheet:
        ts = np.array([f["t"] for f in frames])
        picks = np.linspace(ts[0] + 0.05 * ts[-1], ts[-1] * 0.95, 5)
        figs = []
        import matplotlib.image as mpimg
        for j, st in enumerate(picks):
            fig = plt.figure(figsize=(12.5, 6.2))
            draw = make_renderer(rig, frames, fig, debug_links=args.debug_links)
            draw(int(np.argmin(np.abs(ts - st))))
            tmp = f"/tmp/render_sheet_{j}.png"
            fig.savefig(tmp, dpi=args.dpi)
            plt.close(fig)
            figs.append(mpimg.imread(tmp))
        h = max(im.shape[0] for im in figs)
        strip = np.vstack([np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0)), constant_values=1)
                           for im in figs])
        Path(args.sheet).parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(args.sheet, strip)
        print(f"[render] sheet: {args.sheet}")

    if args.static:
        ts = np.array([f["t"] for f in frames])
        idx = int(np.argmin(np.abs(ts - (args.at if args.at is not None else ts[-1] / 2))))
        fig = plt.figure(figsize=(12.5, 6.2))
        draw = make_renderer(rig, frames, fig, debug_links=args.debug_links)
        draw(idx)
        Path(args.static).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.static, dpi=max(args.dpi, 110))
        plt.close(fig)
        print(f"[render] static frame @t={ts[idx]:.1f}s: {args.static}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
