#!/usr/bin/env python
"""Verify body-relative motion reaches the Unity render payload.

This complements `check_body_relative.py`: instead of looking only at the IK state,
it inspects the actual `render.state` payload that Unity consumes. A fixed
torso-to-wrist vector must keep `arms.*.cmd_pos` stable under headset translation
and yaw; lifting the wrist in the body frame must lift the Unity command target.
"""
from __future__ import annotations

import uuid

import numpy as np

from bimanual_teleop.config import SIDES, load_rig
from bimanual_teleop.engine import TeleopEngine
from bimanual_teleop.hands.quest_retarget import synthetic_webxr_hand
from bimanual_teleop.render_sink import RenderSink
from bimanual_teleop.vr.calibrate import head_op_axes
from bimanual_teleop.vr.frames import HandSample, VRFrame, euler_to_R


def pose(R, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(p, dtype=float)
    return T


def make_frame(head: np.ndarray, torso_from_head: np.ndarray,
               torso_to_wrist: dict[str, np.ndarray], stamp: float,
               ref_head: np.ndarray | None = None) -> VRFrame:
    # Models the real ORBIT stream (see check_body_relative.py): wrist values
    # ride the head POSITION but not its ROTATION; offsets defined in REF_HEAD's
    # frame so head motion leaves the streamed hand values physically unchanged.
    ref = head if ref_head is None else ref_head
    op = head_op_axes(ref)
    hands = {}
    for side in SIDES:
        wrist = pose(ref[:3, :3], head[:3, 3] + op @ (torso_from_head + torso_to_wrist[side]))
        hands[side] = HandSample(
            tracked=True,
            wrist=wrist,
            landmarks=synthetic_webxr_hand(0.15),
            pinch=0.15,
        )
    return VRFrame(stamp=stamp, head=head, hands=hands)


def cmd_pos(state: dict, side: str) -> np.ndarray:
    value = state["arms"][side].get("cmd_pos")
    if value is None:
        raise AssertionError(f"arms.{side}.cmd_pos is null")
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise AssertionError(f"arms.{side}.cmd_pos must be a finite vec3, got {value!r}")
    return arr


def build(engine: TeleopEngine, sink: RenderSink, frame: VRFrame, engaged: dict[str, bool], t: float) -> dict:
    engine.tick(frame, engaged, t)
    return sink.build_state(engine, frame, engaged, hz=120.0, t=t)


def main() -> int:
    rig = load_rig()
    rig["vr"]["calib_seconds"] = 0
    rig["vr"]["body_relative"] = True
    rig["vr"]["torso_from_head"] = [0.0, -0.35, 0.0]
    rig["vr"]["unity_json_endpoint"] = None
    rig["vr"]["render_endpoint"] = f"inproc://body-relative-render-{uuid.uuid4()}"
    torso = np.asarray(rig["vr"]["torso_from_head"], dtype=float)

    # Torso-height, own-side, forward — inside the absolute-mapping workspace.
    base_vec = {
        "left": np.array([-0.22, 0.0, 0.42]),
        "right": np.array([0.22, 0.0, 0.42]),
    }
    lifted_vec = {side: vec + np.array([0.0, 0.16, 0.0]) for side, vec in base_vec.items()}
    head0 = pose(np.eye(3), [0.0, 1.6, 0.0])
    head_moved = pose(euler_to_R([0.0, 0.6, 0.0]), [0.35, 1.72, -0.25])
    engaged = {side: True for side in SIDES}

    sink = RenderSink(rig)
    try:
        engine = TeleopEngine(rig, sink)

        # Settle the absolute-mode engage glide onto the static target first.
        state0 = None
        for i in range(480):
            t = i / 120.0
            state0 = build(engine, sink, make_frame(head0, torso, base_vec, t), engaged, t)
        cmd0 = {side: cmd_pos(state0, side) for side in SIDES}

        state_same = state0
        for i in range(480, 530):
            t = i / 120.0
            state_same = build(engine, sink, make_frame(head_moved, torso, base_vec, t, ref_head=head0), engaged, t)
        cmd_same = {side: cmd_pos(state_same, side) for side in SIDES}

        state_lift = state_same
        for i in range(530, 720):
            t = i / 120.0
            state_lift = build(engine, sink, make_frame(head_moved, torso, lifted_vec, t, ref_head=head0), engaged, t)
        cmd_lift = {side: cmd_pos(state_lift, side) for side in SIDES}

        for side in SIDES:
            op_same = np.asarray(state_same["op"]["hands"][side]["wrist_body"], dtype=float)
            op_lift = np.asarray(state_lift["op"]["hands"][side]["wrist_body"], dtype=float)
            if not np.allclose(op_same, base_vec[side], atol=1e-9):
                raise AssertionError(f"{side} Unity wrist_body changed under head motion: {op_same}")
            if not np.allclose(op_lift, lifted_vec[side], atol=1e-9):
                raise AssertionError(f"{side} Unity wrist_body did not expose lifted vector: {op_lift}")

            drift = float(np.linalg.norm(cmd_same[side] - cmd0[side]))
            dz = float(cmd_lift[side][2] - cmd_same[side][2])
            if drift > 1e-4:
                raise AssertionError(f"{side} Unity cmd_pos drifted under pure head/body motion: {drift:.6f} m")
            if dz < 0.05:
                raise AssertionError(f"{side} Unity cmd_pos did not lift enough: dz={dz:.6f} m")
            print(f"{side}: Unity cmd_pos head-drift={drift:.6f} m, lift dz={dz:.3f} m, "
                  f"wrist_body={op_lift.round(3).tolist()}")

        print("body-relative Unity render payload probe passed")
        return 0
    finally:
        sink.close()


if __name__ == "__main__":
    raise SystemExit(main())
