#!/usr/bin/env python
"""Validate the default rig contract for body-relative Unity teleop.

This catches config drift that would undo the main rework even if lower-level
tests still pass with hand-built rigs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bimanual_teleop.arms import yam_pin
from bimanual_teleop.config import SIDES, load_rig


REPO = Path(__file__).resolve().parents[1]

FROZEN_NEUTRAL = {
    "left": np.array([3.137, -0.004, 0.305, -0.162, -0.003, -1.571]),
    "right": np.array([3.14, -0.001, 0.305, -0.152, 0.001, 1.571]),
}

FROZEN_BASE_POS = {
    "left": np.array([-0.0248, -0.1700, 1.1908]),
    "right": np.array([0.0101, 0.0801, 1.1875]),
}

FROZEN_BASE_QUAT = {
    "left": np.array([0.49791, 0.50194, -0.50011, -0.50004]),
    "right": np.array([0.49989, 0.49988, 0.50055, 0.49968]),
}

EXPECTED_LIMITS_LO = np.array([row[4] for row in yam_pin._CHAIN])
EXPECTED_LIMITS_HI = np.array([row[5] for row in yam_pin._CHAIN])

REMOVED_RUNTIME_FILES = [
    "src/bimanual_teleop/sim/model.py",
    "src/bimanual_teleop/sim/sim_world.py",
    "src/bimanual_teleop/launch/run_sim.py",
]


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def check_vector(name: str, value, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    require(arr.shape == (n,), f"{name} must be a length-{n} numeric vector")
    require(np.all(np.isfinite(arr)), f"{name} must contain finite numbers")
    return arr


def main() -> int:
    rig = load_rig()
    vr = rig.get("vr", {})
    require(vr.get("body_relative") is True, "vr.body_relative must default to true")
    torso = check_vector("vr.torso_from_head", vr.get("torso_from_head"), 3)
    require(-0.8 < torso[1] < -0.05, "vr.torso_from_head must place torso below headset in body-up axis")
    require(str(vr.get("render_endpoint", "")).startswith("tcp://"), "vr.render_endpoint must be a tcp:// endpoint")
    require(str(vr.get("unity_json_endpoint", "")).startswith("tcp://"), "vr.unity_json_endpoint must be a tcp:// endpoint")

    mapping = rig.get("mapping", {})
    pos_scale = float(mapping.get("pos_scale", float("nan")))
    require(np.isfinite(pos_scale) and pos_scale > 0.0, "mapping.pos_scale must be finite and positive")
    require(mapping.get("position_mode") == "absolute",
            "mapping.position_mode must default to 'absolute' — hands in front of the operator "
            "put the robot's hands in front of the robot ('relative' is per-run diagnostics only)")
    blend = float(mapping.get("engage_blend_s", float("nan")))
    require(np.isfinite(blend) and blend >= 0.0, "mapping.engage_blend_s must be finite and >= 0")
    require(mapping.get("twist_mode") == "intrinsic",
            "mapping.twist_mode must default to 'intrinsic' — a wrist turn is a pure j6 roll, "
            "never a j4/j5 swing through the wrist singularity ('world' is diagnostics only)")
    require(mapping.get("orientation_mode") == "absolute",
            "mapping.orientation_mode must default to 'absolute' — the robot hand wears the "
            "operator's hand attitude (overlay skeleton and ORCA hand coincide); 'relative' is "
            "a per-run diagnostic choice")
    # Hand-local axes are PER-SIDE (the two hands are anatomical mirrors; shared
    # constants were measured 21-45 deg off and leaked rolls into swing).
    for key in ("hand_twist_axis", "hand_finger_axis", "hand_palm_axis"):
        v = mapping.get(key)
        require(isinstance(v, dict) and set(v) >= {"left", "right"},
                f"mapping.{key} must be per-side ({{left: …, right: …}})")
        for side in SIDES:
            a = check_vector(f"mapping.{key}.{side}", v[side], 3)
            require(abs(np.linalg.norm(a) - 1.0) < 0.05,
                    f"mapping.{key}.{side} must be (near) unit length")
    for side in SIDES:
        hfa = np.asarray(mapping["hand_finger_axis"][side], float)
        hpa = np.asarray(mapping["hand_palm_axis"][side], float)
        require(abs(float(hfa @ hpa)) < 0.2,
                f"hand finger/palm axes must be roughly orthogonal ({side})")
        # mirror-pair sanity: lateral (x) components flip sign between sides
    require(np.sign(mapping["hand_palm_axis"]["left"][0]) != np.sign(mapping["hand_palm_axis"]["right"][0]),
            "hand_palm_axis left/right must be a mirror pair (x flips sign)")
    anchor = mapping.get("body_anchor_world")
    if anchor is not None:
        check_vector("mapping.body_anchor_world", anchor, 3)
    for stale in ("abs_orientation", "ori_tweak_euler"):
        require(stale not in mapping,
                f"mapping.{stale} is a removed knob — orientation now uses the calibration-free "
                "world-frame relative mapping (ClutchMapper.target); delete the stale key")
    require(float(rig.get("vr", {}).get("calib_seconds", 0.0)) == 0.0,
            "vr.calib_seconds must default to 0 — the arm mapping is calibration-free; "
            "pass --calib-seconds for legacy stance diagnostics instead of baking it in")

    trims = mapping.get("r_base_from_vr_euler", {})
    for side in SIDES:
        trim = check_vector(f"mapping.r_base_from_vr_euler.{side}", trims.get(side), 3)
        require(np.linalg.norm(trim) < 1e-12, f"legacy mapper trim for {side} must stay zero in body-relative mode")

        arm = rig.get("arms", {}).get(side, {})
        base_pos = check_vector(f"arms.{side}.base_pos", arm.get("base_pos"), 3)
        require(np.allclose(base_pos, FROZEN_BASE_POS[side], atol=1e-9),
                f"arms.{side}.base_pos changed from the measured elongated-stand placement")
        quat = check_vector(f"arms.{side}.base_quat", arm.get("base_quat"), 4)
        require(abs(np.linalg.norm(quat) - 1.0) < 2e-3, f"arms.{side}.base_quat must be normalized")
        require(np.allclose(quat, FROZEN_BASE_QUAT[side], atol=1e-9),
                f"arms.{side}.base_quat changed from the measured elongated-stand placement")
        hand_pos = check_vector(f"arms.{side}.hand_pos", arm.get("hand_pos"), 3)
        hand_euler = check_vector(f"arms.{side}.hand_euler", arm.get("hand_euler"), 3)
        _, ee_pos, ee_euler = yam_pin._EE_SITE[side]
        require(np.allclose(hand_pos, np.asarray(ee_pos, dtype=float), atol=1e-9),
                f"arms.{side}.hand_pos must match the MJCF-derived ORCA flange site")
        require(np.allclose(hand_euler, np.asarray(ee_euler, dtype=float), atol=1e-9),
                f"arms.{side}.hand_euler must match the MJCF-derived ORCA flange site")
        q = check_vector(f"arms.{side}.neutral_q", arm.get("neutral_q"), 6)
        require(np.allclose(q, FROZEN_NEUTRAL[side], atol=1e-9),
                f"arms.{side}.neutral_q changed from docs/RESTING_POSE.md")

    limits = rig.get("arms", {}).get("joint_limits", {})
    lo = check_vector("arms.joint_limits.lower", limits.get("lower"), 6)
    hi = check_vector("arms.joint_limits.upper", limits.get("upper"), 6)
    require(np.all(lo < hi), "joint lower limits must be below upper limits")
    require(np.allclose(lo, EXPECTED_LIMITS_LO, atol=1e-9),
            "arms.joint_limits.lower must match the MJCF-derived YAM joint limits")
    require(np.allclose(hi, EXPECTED_LIMITS_HI, atol=1e-9),
            "arms.joint_limits.upper must match the MJCF-derived YAM joint limits")

    hw = rig.get("hardware", {})
    scale = float(hw.get("max_vel_scale", float("nan")))
    require(np.isfinite(scale) and 0.0 < scale <= 1.0,
            "hardware.max_vel_scale must be in (0, 1] — real motors run derated, never faster than sim")
    rate = float(hw.get("rate_limit", float("nan")))
    require(np.isfinite(rate) and 0.0 < rate <= 3.0,
            "hardware.rate_limit must be a sane per-joint speed cap (0, 3] rad/s")
    smooth = float(hw.get("smooth_hz", float("nan")))
    require(np.isfinite(smooth) and 0.0 < smooth <= 12.0,
            "hardware.smooth_hz must be finite in (0, 12] (command-shaper bandwidth)")
    sides = hw.get("sides", list(SIDES))
    require(isinstance(sides, list) and len(sides) > 0 and all(s in SIDES for s in sides),
            "hardware.sides must be a non-empty subset of left/right — the arms actually wired")
    require(isinstance(hw.get("use_hands"), bool),
            "hardware.use_hands must be an explicit bool (RealHand MOVES the hand at connect)")
    # engage_pose_tol is scalar OR per-joint (the rig went per-joint so the limp
    # wrist's free drift doesn't reject every start): the shoulder/elbow (j1–j4)
    # must stay TIGHT (≤ 0.35 rad) for the rest-pose gate to catch a wrong-arm /
    # not-at-rest start, while the wrist (j5/j6) is intentionally loose but must
    # stay < π so a ±π wrong-arm flip still trips.
    tol = np.atleast_1d(np.asarray(hw.get("engage_pose_tol", float("nan")), dtype=float))
    require(bool(np.all(np.isfinite(tol))) and bool(np.all(tol > 0.0)),
            "hardware.engage_pose_tol entries must be finite and > 0 rad")
    shoulder = tol[:4] if tol.size > 1 else tol
    require(bool(np.all(shoulder <= 0.35)),
            "hardware.engage_pose_tol shoulder/elbow (j1–j4) must be ≤ 0.35 rad — the rest-pose gate must stay meaningful")
    require(bool(np.all(tol <= np.pi)),
            "hardware.engage_pose_tol must stay < π so a ±π wrong-arm flip still trips the rest-pose gate")
    require(str(hw.get("joint_map_file", "")).endswith(".json"),
            "hardware.joint_map_file must point at the per-machine motor↔model calibration json")

    workspace = rig.get("safety", {}).get("workspace", {})
    wmin = check_vector("safety.workspace.min", workspace.get("min"), 3)
    wmax = check_vector("safety.workspace.max", workspace.get("max"), 3)
    require(np.all(wmin < wmax), "workspace min must be below max")
    require(wmin[1] <= -0.6, "workspace y-min must contain the frozen arms-down home wrist")

    for rel in REMOVED_RUNTIME_FILES:
        require(not (REPO / rel).exists(), f"removed MuJoCo runtime file came back: {rel}")

    print("rig contract preserves body-relative Unity runtime defaults")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
