#!/usr/bin/env python
"""Hardware-free probe for torso-relative arm control.

This is the acceptance check for the original teleop bug: arm control must consume
the vector from the operator torso proxy to the wrist, not a blind room-space hand
direction. Translating/yawing the headset while that torso-to-wrist vector is fixed
must not move the arm target. Lifting the wrist relative to the torso must move the
achieved wrist upward in robot world.
"""
from __future__ import annotations

import numpy as np

from bimanual_teleop.config import SIDES, load_rig
from bimanual_teleop.engine import TeleopEngine
from bimanual_teleop.hands.quest_retarget import synthetic_webxr_hand
from bimanual_teleop.render_sink import operator_debug_state
from bimanual_teleop.vr.calibrate import head_op_axes
from bimanual_teleop.vr.frames import HandSample, VRFrame, euler_to_R


class ProbeSink:
    def __init__(self) -> None:
        self.arm: dict[str, np.ndarray] = {}
        self.hand: dict[str, dict[str, float]] = {}

    def set_arm(self, side: str, q) -> None:
        self.arm[side] = np.asarray(q, dtype=float)

    def set_hand(self, side: str, joints: dict) -> None:
        self.hand[side] = dict(joints)


def pose(R, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(p, dtype=float)
    return T


def make_frame(head: np.ndarray, torso_from_head: np.ndarray,
               torso_to_wrist: dict[str, np.ndarray]) -> VRFrame:
    op = head_op_axes(head)
    hands = {}
    for side in SIDES:
        # RIGID body motion: the hand carries the head's rotation (a real hand
        # attached to a turning body does), with a PROPER rotation matrix.
        wrist = pose(head[:3, :3], head[:3, 3] + op @ (torso_from_head + torso_to_wrist[side]))
        hands[side] = HandSample(
            tracked=True,
            wrist=wrist,
            landmarks=synthetic_webxr_hand(0.0),
        )
    return VRFrame(stamp=0.0, head=head, hands=hands)


def wrist_world(engine: TeleopEngine, side: str) -> np.ndarray:
    arm = engine.arm[side]
    return arm.base_R @ arm.ik.fk_wrist().translation() + arm.base_pos


def assert_close(name: str, got, expected, atol: float) -> None:
    if not np.allclose(got, expected, atol=atol):
        raise AssertionError(f"{name}: got {np.asarray(got)}, expected {np.asarray(expected)}")


def main() -> int:
    rig = load_rig()
    rig["vr"]["calib_seconds"] = 0
    rig["vr"]["body_relative"] = True
    rig["vr"]["torso_from_head"] = [0.0, -0.35, 0.0]
    rig.setdefault("mapping", {})["swap_sides"] = False   # this probe tests same-side mapping mechanics
    torso = np.asarray(rig["vr"]["torso_from_head"], dtype=float)

    # Torso-height, own-side, forward — inside the absolute-mapping workspace the
    # YAM can actually reach (it cannot follow far above its base plates).
    base_vec = {
        "left": np.array([-0.22, 0.0, 0.42]),
        "right": np.array([0.22, 0.0, 0.42]),
    }
    lifted_vec = {side: vec + np.array([0.0, 0.16, 0.0]) for side, vec in base_vec.items()}
    head0 = pose(np.eye(3), [0.0, 1.6, 0.0])
    head_moved = pose(euler_to_R([0.0, 0.6, 0.0]), [0.35, 1.72, -0.25])

    # First prove the debug/render payload exposes the same body vector independent
    # of room-space headset translation and yaw.
    frame0 = make_frame(head0, torso, base_vec)
    frame_moved = make_frame(head_moved, torso, base_vec)
    debug0 = operator_debug_state(frame0, torso)
    debug_moved = operator_debug_state(frame_moved, torso)
    for side in SIDES:
        assert_close(f"{side} wrist_body initial", debug0["hands"][side]["wrist_body"], base_vec[side], 1e-9)
        assert_close(f"{side} wrist_body moved-head", debug_moved["hands"][side]["wrist_body"], base_vec[side], 1e-9)

    sink = ProbeSink()
    engine = TeleopEngine(rig, sink)
    engaged = {side: True for side in SIDES}

    # Settle the absolute-mode engage glide + IK onto the static target first.
    for i in range(480):
        engine.tick(frame0, engaged, i / 120.0)
    p0 = {side: wrist_world(engine, side).copy() for side in SIDES}

    # Move/yaw the headset but keep the torso-to-wrist vector unchanged. The arm
    # should stay anchored because body-relative mapping cancels whole-body motion.
    for i in range(480, 520):
        engine.tick(frame_moved, engaged, i / 120.0)
    p_same = {side: wrist_world(engine, side).copy() for side in SIDES}
    for side in SIDES:
        drift = float(np.linalg.norm(p_same[side] - p0[side]))
        if drift > 1e-4:
            raise AssertionError(f"{side} drifted under pure head/body motion: {drift:.6f} m")

    # Lift both wrists relative to the torso. In robot world, this must increase Z.
    frame_lifted = make_frame(head_moved, torso, lifted_vec)
    for i in range(520, 760):
        engine.tick(frame_lifted, engaged, i / 120.0)
    p_lift = {side: wrist_world(engine, side).copy() for side in SIDES}
    for side in SIDES:
        dz = float(p_lift[side][2] - p_same[side][2])
        if dz < 0.015:
            raise AssertionError(f"{side} wrist did not move upward enough after lift: dz={dz:.6f} m")
        print(f"{side}: head-motion drift={np.linalg.norm(p_same[side] - p0[side]):.6f} m, "
              f"lift dz={dz:.3f} m, wrist_body={lifted_vec[side].round(3).tolist()}")

    print("body-relative teleop probe passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
