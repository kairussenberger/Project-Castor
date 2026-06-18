#!/usr/bin/env python
"""Offline session analyzer — replay a recorded Quest session (.npz) through the
REAL TeleopEngine and measure, frame by frame, whether the commanded robot motion
matches the operator's hand motion. No headset, no Unity, no robot.

The teleop contract (CLAUDE.md): rotate/translate your hand relative to your BODY
axes (right/up/forward) and the EE must rotate/translate about the corresponding
robot WORLD axes (+Y/+Z/−X) by the same amount. This script quantifies both:

  - ORIENTATION: angle between the world-frame rotation axis your hand actually
    moved about and the world-frame rotation axis the engine commanded the EE
    about, plus the angle-magnitude ratio (1.0 = same amount of rotation).
  - TRANSLATION: direction cosine + magnitude ratio between the body-frame hand
    displacement (mapped to world) and the commanded EE displacement.
  - IK TRACKING: orientation/position gap between the commanded target and the
    pose the IK actually achieved (isolates mapping bugs from solver bugs).

    uv run python scripts/analyze_session.py recordings/roll_right.npz
    uv run python scripts/analyze_session.py session.npz --side left --min-angle 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.config import SIDES, load_rig                  # noqa: E402
from bimanual_teleop.engine import TeleopEngine                     # noqa: E402
from bimanual_teleop.vr.calibrate import W_AXES, head_op_axes       # noqa: E402
from bimanual_teleop.vr.frames import (                             # noqa: E402
    SE3, quat_from_axis_angle, quat_to_R, rotvec, swing_twist_angle)
from bimanual_teleop.vr.replay import ReplaySource                  # noqa: E402


def _rot(axis, ang):
    return quat_to_R(quat_from_axis_angle(axis, ang))


class NullSink:
    def set_arm(self, side, q):
        pass

    def set_hand(self, side, joints):
        pass


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="session .npz from run_teleop --record / check_roll --save")
    ap.add_argument("--side", choices=["left", "right"], default="right")
    ap.add_argument("--min-angle", type=float, default=15.0,
                    help="only score frames where the hand has rotated at least this many deg from anchor")
    ap.add_argument("--calib-seconds", type=float, default=None,
                    help="override vr.calib_seconds (default: rig value, i.e. what a live session does)")
    ap.add_argument("--no-calib", action="store_true",
                    help="score through IDENTITY even when the recording embeds its session fit")
    args = ap.parse_args()

    rig = load_rig()
    if args.calib_seconds is not None:
        rig["vr"]["calib_seconds"] = max(0.0, float(args.calib_seconds))

    src = ReplaySource(args.path)
    if src.calib and not args.no_calib:
        # Score through the calibration that actually RAN during the session:
        # raw ORBIT frames are only meaningful with their fit (stream anchors
        # move metres between sessions — identity scoring of a calibrated
        # session reads as a metre-scale 'mapping error' that never happened).
        rig["vr"]["_embedded_calib"] = src.calib
        print(f"applying the session calibration embedded in the recording "
              f"(stamp {(src.calib.get('meta') or {}).get('stamp', '?')}; --no-calib for identity)")
    t = src.t
    side = args.side
    base_R = quat_to_R(rig["arms"][side]["base_quat"])              # arm base → world
    print(f"session: {args.path}")
    print(f"  frames={len(t)}  duration={src.duration:.1f}s  side={side}")
    head_ok = ~np.isnan(src.head).any(axis=(1, 2))
    lm_ok = ~np.isnan(src._side[side]["landmarks"]).all(axis=(1, 2))
    print(f"  head present {head_ok.mean()*100:.0f}%  | {side} tracked "
          f"{src._side[side]['tracked'].mean()*100:.0f}%  landmarks {lm_ok.mean()*100:.0f}%  "
          f"engaged {src.engaged_arr[:, SIDES.index(side)].mean()*100:.0f}%")

    engine = TeleopEngine(rig, NullSink())
    arm = engine.arm[side]
    twist_mode = str(rig.get("mapping", {}).get("twist_mode", "intrinsic"))
    ori_mode = str(rig.get("mapping", {}).get("orientation_mode", "absolute"))
    from bimanual_teleop.config import side_axis
    hand_axis = np.asarray(side_axis(rig.get("mapping", {}), "hand_twist_axis", side,
                                     [0.0, 0.456, 0.890]), float)
    hand_axis = hand_axis / np.linalg.norm(hand_axis)

    # Anchor snapshot, re-captured whenever the mapper (re-)engages.
    anchor = None          # dict(raw_R, op_axes, ee_R, ee_p, q)
    prev_anchor_obj = None
    rows = []              # per-frame mapping scores

    for i in range(len(t)):
        frame = src.frame_at(t[i])
        engaged = src.engaged_at(t[i])
        engine.tick(frame, engaged, float(t[i]))

        m = arm.mapper
        hs = frame.hands.get(side)
        if frame.head is None or hs is None or not hs.tracked:
            continue
        # Score in the SAME body frame the engine maps in: the LOCKED session
        # yaw (vr.body_yaw: locked, the default). Using the live head here
        # mis-scored every session where the operator looked around — the
        # prediction rotated with the head while the command (correctly)
        # did not (measured: a fingers session with the head wandering the
        # full circle scored 71° of phantom orientation error).
        if getattr(engine, "_yaw_R", None) is not None:
            T = np.eye(4)
            T[:3, :3] = engine._yaw_R
            op_axes_now = head_op_axes(T)
        else:
            op_axes_now = head_op_axes(frame.head)
        if m.anchor_ctrl is not None and m.anchor_ctrl is not prev_anchor_obj:
            prev_anchor_obj = m.anchor_ctrl
            anchor = {
                "raw_R": np.asarray(hs.wrist, float)[:3, :3].copy(),
                "raw_p": np.asarray(hs.wrist, float)[:3, 3].copy(),
                "op_axes": op_axes_now,
                "ee_R": m.anchor_ee.rotation().as_matrix(),
                "ee_p": m.anchor_ee.translation(),
                "t": float(t[i]),
                "q": arm.ik.q,
            }
        if anchor is None or not m.engaged or arm.cmd_R is None:
            continue

        # --- what the OPERATOR did, in robot-world terms ---------------------- #
        W = np.asarray(hs.wrist, float)
        D_xr = W[:3, :3] @ anchor["raw_R"].T                # hand rotation since anchor (WebXR world)
        rv = rotvec(D_xr)
        hand_ang = float(np.degrees(np.linalg.norm(rv)))
        Q = W_AXES @ anchor["op_axes"].T                    # WebXR world → robot world (proper)
        want_axis_w = Q @ (rv / (np.linalg.norm(rv) + 1e-12))

        # --- what the ENGINE commanded ---------------------------------------- #
        D_ee = arm.cmd_R @ anchor["ee_R"].T                 # commanded EE rotation since anchor (base)
        rv_ee = rotvec(D_ee)
        cmd_ang = float(np.degrees(np.linalg.norm(rv_ee)))
        cmd_axis_w = base_R @ (rv_ee / (np.linalg.norm(rv_ee) + 1e-12))

        # --- intrinsic-twist contract, verified end-to-end --------------------- #
        # Independent reconstruction from RAW data + rig constants: decompose the
        # physical hand rotation about the forearm axis, map the swing through the
        # body↔world axes, apply the twist about the EE's own tool axis, and
        # compare against what the engine actually commanded.
        h_now = W[:3, :3] @ hand_axis                       # forearm axis in the room
        phi_h = swing_twist_angle(D_xr, h_now)
        D_sw = D_xr @ _rot(h_now, -phi_h)
        Q_b = base_R.T @ Q                                  # room → arm base (proper)
        S_b = Q_b @ D_sw @ Q_b.T
        A_R = anchor["ee_R"]
        D_pred = S_b @ (A_R @ _rot(arm.ik.ee_tool_axis_local, phi_h) @ A_R.T)
        contract_err = float(np.degrees(np.linalg.norm(rotvec(D_pred.T @ D_ee))))
        ee_ax_b = A_R @ arm.ik.ee_tool_axis_local
        phi_e = swing_twist_angle(D_ee, ee_ax_b)
        swing_h = float(np.degrees(np.linalg.norm(rotvec(D_sw))))
        abs_ori_err = float("nan")
        if arm.mapper.C is not None:
            pred_abs = base_R.T @ Q @ W[:3, :3] @ arm.mapper.C      # skeleton attitude ∘ convention
            abs_ori_err = float(np.degrees(np.linalg.norm(rotvec(pred_abs.T @ arm.cmd_R))))

        # --- translation ------------------------------------------------------- #
        # Body-frame torso→wrist vector, recomputed the same way the ENGINE's
        # _arm_hand_sample does: LOCKED yaw axes, live head position; compared
        # post-hoc via windowed deltas so the scoring works for both 'absolute'
        # (with its engage glide) and 'relative' position modes.
        op_axes = op_axes_now
        torso_w = frame.head[:3, 3] + op_axes @ np.asarray(rig["vr"]["torso_from_head"], float)
        ctrl_p = op_axes.T @ (W[:3, 3] - torso_w)
        cmd_w = base_R @ arm.cmd_pos + np.asarray(rig["arms"][side]["base_pos"], float)

        # --- IK tracking (achieved vs commanded) ------------------------------- #
        ach_R = arm.ik.fk_ee().rotation().as_matrix()
        ik_ori_err = float(np.degrees(np.linalg.norm(rotvec(ach_R.T @ arm.cmd_R))))

        rows.append({
            "t": float(t[i] - t[0]),
            "t_engage": anchor["t"] - float(t[0]),
            "hand_ang": hand_ang,
            "cmd_ang": cmd_ang,
            "axis_err": _angle_between(want_axis_w, cmd_axis_w),
            "want_axis": want_axis_w,
            "cmd_axis": cmd_axis_w,
            "twist_h": float(np.degrees(phi_h)),
            "twist_e": float(np.degrees(phi_e)),
            "swing_h": swing_h,
            "contract_err": contract_err,
            "abs_ori_err": abs_ori_err,
            "ctrl_p": ctrl_p,
            "cmd_w": cmd_w,
            "ik_ori_err": ik_ori_err,
            "q": arm.ik.q,
        })

    if not rows:
        print("\nno engaged+tracked frames with a command — nothing to score "
              "(did calibration ever finish? try --calib-seconds 0)")
        return 1

    sc = [r for r in rows if r["hand_ang"] >= args.min_angle]
    print(f"\nscored {len(sc)}/{len(rows)} engaged frames with hand rotation ≥ {args.min_angle:.0f}°")
    if ori_mode == "absolute":
        print("\n---- ORIENTATION mapping (orientation_mode=absolute) ----")
        blend = float(rig.get("mapping", {}).get("engage_blend_s", 1.0))
        settled = [r for r in rows if r["t"] - r["t_engage"] > 2.0 * blend and np.isfinite(r["abs_ori_err"])]
        if settled:
            ae = np.array([r["abs_ori_err"] for r in settled])
            print(f"  |skeleton attitude ∘ convention − commanded|: median {np.median(ae):5.2f}°  "
                  f"p90 {np.percentile(ae, 90):5.2f}°   (post-glide; the overlay-overlap guarantee)")
            ok_ori = np.median(ae) < 5.0
        else:
            ok_ori = True
            print("  (no settled frames; orientation unscored)")
        print(f"  VERDICT: {'OK — robot hand wears your hand attitude' if ok_ori else 'BROKEN — absolute orientation contract violated'}")
    elif twist_mode == "intrinsic":
        print("\n---- ORIENTATION mapping (twist_mode=intrinsic) ----")
        ce = np.array([r["contract_err"] for r in sc]) if sc else np.array([])
        ok_ori = True
        if len(ce):
            print(f"  contract error |predicted − commanded|: median {np.median(ce):5.1f}°   "
                  f"p90 {np.percentile(ce, 90):5.1f}°")
            print(f"  (prediction reconstructed from raw data: hand twist about your forearm axis → EE")
            print(f"   roll about its own tool axis; residual hand swing → world-frame EE swing)")
            tw = [r for r in sc if abs(r['twist_h']) >= args.min_angle and r['swing_h'] < 10.0]
            if tw:
                ratio = np.array([r["twist_e"] / r["twist_h"] for r in tw])
                print(f"  twist-dominant frames: EE/hand twist ratio median {np.median(ratio):+5.2f} ({len(tw)} frames)")
            ok_ori = np.median(ce) < 5.0
        else:
            print("  (no frames exceeded --min-angle; orientation unscored)")
        print(f"  VERDICT: {'OK — commanded orientation matches the intrinsic-twist contract' if ok_ori else 'BROKEN — engine output deviates from the intrinsic-twist contract'}")
    elif sc:
        ax = np.array([r["axis_err"] for r in sc if np.isfinite(r["axis_err"])])
        ratio = np.array([r["cmd_ang"] / max(r["hand_ang"], 1e-9) for r in sc])
        print("\n---- ORIENTATION mapping (twist_mode=world) ----")
        print(f"  world-axis error:    median {np.median(ax):6.1f}°   p90 {np.percentile(ax, 90):6.1f}°")
        print(f"  angle ratio cmd/hand: median {np.median(ratio):5.2f}    (1.00 = same magnitude)")
        mid = sc[len(sc) // 2]
        print(f"  example @t={mid['t']:.1f}s: hand {mid['hand_ang']:.0f}° about world "
              f"[{mid['want_axis'][0]:+.2f} {mid['want_axis'][1]:+.2f} {mid['want_axis'][2]:+.2f}]  "
              f"→ cmd {mid['cmd_ang']:.0f}° about [{mid['cmd_axis'][0]:+.2f} {mid['cmd_axis'][1]:+.2f} {mid['cmd_axis'][2]:+.2f}]")
        ok_ori = np.median(ax) < 15.0 and 0.8 < np.median(ratio) < 1.25
        print(f"  VERDICT: {'OK — EE rotates about the same world axis as your hand' if ok_ori else 'BROKEN — commanded rotation axis does not match the hand'}")
    else:
        ok_ori = True
        print("  (no frames exceeded --min-angle; orientation unscored)")

    mode = str(rig.get("mapping", {}).get("position_mode", "absolute"))
    scale = float(rig.get("mapping", {}).get("pos_scale", 1.0))
    blend = float(rig.get("mapping", {}).get("engage_blend_s", 1.0))
    print(f"\n---- TRANSLATION mapping (position_mode={mode}) ----")
    win = max(1, int(0.3 * len(rows) / max(rows[-1]["t"] - rows[0]["t"], 1e-6)))   # ~0.3 s
    pairs = []
    for i in range(len(rows) - win):
        a, b = rows[i], rows[i + win]
        d_want = scale * (W_AXES @ (b["ctrl_p"] - a["ctrl_p"]))
        if np.linalg.norm(d_want) < 0.04:
            continue
        d_cmd = b["cmd_w"] - a["cmd_w"]
        pairs.append((_angle_between(d_want, d_cmd),
                      float(np.linalg.norm(d_cmd) / (np.linalg.norm(d_want) + 1e-9))))
    if pairs:
        de = np.array([p[0] for p in pairs if np.isfinite(p[0])])
        dr = np.array([p[1] for p in pairs])
        print(f"  windowed (0.3s) displacement direction error: median {np.median(de):6.1f}°   p90 {np.percentile(de, 90):6.1f}°")
        print(f"  windowed magnitude ratio:                     median {np.median(dr):5.2f}   (clamps/filters can shrink it)")
        ok_pos = np.median(de) < 20.0
    else:
        ok_pos = True
        print("  hand never displaced >4cm within a window; direction unscored")
    if mode == "absolute":
        settled = [r for r in rows if r["t"] - r["t_engage"] > 2.0 * blend]
        if settled:
            # The reference goes through the mapper's OWN body-axes map
            # (_p_abs: lateral curve + per-axis scale/offset + chest anchor),
            # so a calibrated session is scored against the calibrated
            # correspondence. With identity calibration this is byte-identical
            # to the old `chest + scale·(W_AXES @ torso→wrist)` formula.
            base_pos = np.asarray(rig["arms"][side]["base_pos"], float)
            m = arm.mapper
            err = np.array([np.linalg.norm(
                r["cmd_w"] - (base_R @ m._p_abs(SE3.from_translation(r["ctrl_p"])) + base_pos))
                for r in settled])
            print(f"  absolute correspondence |cmd − map(torso→wrist)|: median {np.median(err)*100:5.1f} cm  "
                  f"p90 {np.percentile(err, 90)*100:5.1f} cm   (post-glide; workspace clamps add to this)")
            ok_pos = ok_pos and np.median(err) < 0.10
    print(f"  VERDICT: {'OK' if ok_pos else 'BROKEN'}")

    ik = np.array([r["ik_ori_err"] for r in rows])
    print("\n---- IK tracking (achieved vs commanded orientation) ----")
    print(f"  median {np.median(ik):5.1f}°   p90 {np.percentile(ik, 90):5.1f}°"
          "   (large = solver/limits, NOT the mapping)")

    q = np.stack([r["q"] for r in rows])
    rng = np.degrees(q.max(axis=0) - q.min(axis=0))
    print("\n---- joint travel while engaged (deg) ----")
    print("  " + "  ".join(f"j{i+1}:{rng[i]:6.1f}" for i in range(6)))

    print(f"\nOVERALL: {'PASS' if (ok_ori and ok_pos) else 'FAIL — mapping does not honor the body↔world contract'}")
    return 0 if (ok_ori and ok_pos) else 1


if __name__ == "__main__":
    raise SystemExit(main())
