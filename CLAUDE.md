# Teleop Runtime Notes

System overview, failsafe inventory, tooling index, and the sim→real checklist
live in `docs/ARCHITECTURE.md` — keep that page current when changing any of
them. The hardware boundary (`HardwareSink`) must always command through
`safety/shaper.py` (limit-clamp + speed cap + PD smoothing); never bypass it.

THE MOTOR BOUNDARY IS A DIFFERENT JOINT CONVENTION: this rig's motor zeros do
NOT match this repo's model NOR i2rt's own convention (measured on metal: the
official hang reads j2≈−177°, j6≈+312° in motor space). Every real
command/state crosses the per-side affine map in `arms/joint_map.py`, MEASURED
by `scripts/hw_bringup.py` and anchored at the official rest pose; `YamArm`
refuses model-space commands without it, and `HardwareSink` refuses to start
unless the arm MEASURES at rest through it (`hardware.engage_pose_tol`). The
runtime drives the CAN chain DIRECTLY via `arms/yam_chain.py` — never through
i2rt's `MotorChainRobot` (its qpos checks reject this rig's zeros at boot, and
its gravity compensation assumes a TABLE mount while these arms hang SIDEWAYS;
no model-based feedforward is ever sent). `hardware.sides` /
`hardware.use_hands` must reflect what is physically wired.

This repo no longer uses MuJoCo as the runtime simulator. The current target is a
headless Python teleop process that:

- ingests Quest/ORBIT, Vuer/WebXR, replay, or synthetic poses;
- converts raw headset/hand poses into body-relative torso-to-wrist vectors;
- solves each YAM arm with a standalone Pinocchio/pink IK model;
- retargets ORCA hand landmarks to hardware joint degrees;
- publishes `render.state` for Unity over ZMQ/msgpack and plain TCP JSON;
- can swap the render sink for the real hardware sink on the Linux robot host.

The remaining MJCF files under `src/bimanual_teleop/sim/models/yam_real/` are
source geometry for the programmatic Pinocchio model in
`src/bimanual_teleop/arms/yam_pin.py`. They are not loaded by the runtime.

## Coordinate Contract

Robot world is right-handed:

- `+Z` is up.
- `+Y` is operator/robot right.
- `-X` is forward.

Quest/ORBIT Unity poses are converted before they reach the teleop engine. Arm
control then uses body-relative wrist samples:

```text
raw head + raw wrist -> operator body axes -> torso proxy -> wrist_body
```

`wrist_body` is `[right, up, forward]` from the torso proxy to the wrist. Whole
head/body translation should cancel. A hand lift should appear as a positive
`wrist_body[1]` delta and drive the corresponding arm target upward.

The Unity render stream exposes the same vector at
`render.state.op.hands.*.wrist_body` so the operator overlay and the robot arm
state can be compared directly.

## Motion Mapping Contract

POSITION is ABSOLUTE and body-anchored (`mapping.position_mode: absolute`): the
operator's torso→wrist vector maps 1:1 (×`pos_scale`) onto the robot's
chest→wrist vector, where the chest anchor defaults to the midpoint of the arm
bases dropped by `body_anchor_drop`. Hands held in front of the operator put the
robot's wrists in front of the robot — verified at 0.1 cm median correspondence
on a real session. On (re)engage the EE GLIDES onto correspondence over
`mapping.engage_blend_s` (continuous at the engage instant; displacements map 1:1
from the first tick). The `ik.soft_margin` values are sized so this front-of-body
workspace is reachable — re-tightening them below the measured excursions in
config/rig.yaml's comment will pin the arm short of its targets again.

EE ORIENTATION is ABSOLUTE and calibration-free
(`mapping.orientation_mode: absolute`): the robot hand WEARS the operator's hand
attitude — mapped through the body↔world axes and a DERIVED hand↔EE convention
(EE side from the frozen rest contract via `viz.hand_geom.hand_basis_for_side`;
hand side from `mapping.hand_finger_axis`/`hand_palm_axis`, measured from a real
session) — gliding over `engage_blend_s` on (re)engage. Consequences: the
dashboard-overlay hand skeleton and the rendered ORCA hand coincide by
construction (verified 0.00° post-glide on a real recording), and commanded
wrist attitudes are reachable ones (IK orientation tracking 68°→0.1° median on
the same session). A wrist turn is inherently a pure j6 roll because the whole
attitude follows.

`orientation_mode: relative` is a per-run diagnostic; in that mode
`mapping.twist_mode: intrinsic` decomposes deltas about the operator's forearm
axis so the twist still lands on j6. `vr.calib_seconds` defaults to 0; the
legacy stance hold only steers arms when `vr.body_relative` is false.

FRAME HYGIENE (hard-won): real devices only ever produce PROPER wrist rotations;
`head_op_axes`/`W_AXES` are left-handed bases whose reflections must cancel in
pairs. Synthetic fixtures must NEVER use `op_axes` as a wrist rotation — an
improper raw wrist makes the absolute attitude a reflection and the QP explodes
(the mapper now fails closed on det<0, and
`test_absolute_orientation_fails_closed_on_improper_ctrl_rotation` pins it).
Rigid body-motion fixtures must carry the head rotation on the wrist.

Do not reintroduce a stance-calibrated hand-local↔EE-local correspondence: an
imperfect calibration pose scrambles every commanded rotation axis (measured 145°
median axis error on a real Quest session; `tests/test_frames.py::
test_clutch_orientation_body_relative_real_rig_axes` pins the contract). Note that
`head_op_axes`/`W_AXES` are left-handed `[right, up, forward]` bases (det −1); the
ClutchMapper conjugation cancels the two reflections, so keep orientation math
going through BOTH or neither.

To grade a recorded session against this contract (axis error, angle ratio,
translation direction, IK tracking gap — no headset needed):

```sh
uv run python scripts/analyze_session.py recordings/session.npz
```

To watch it in 3D without Unity (`uv sync --extra telemetry` once):

```sh
uv run python -m bimanual_teleop.launch.run_teleop --vr replay session.npz --viz
```

## IK Contract

Arm IK lives in `src/bimanual_teleop/arms/ik.py` and uses Pinocchio/pink:

1. Solve wrist position with j1-j3 while j4-j6 are velocity-limited near zero.
2. Assign the TWIST (roll about the current j6/tool axis) of the orientation
   error DIRECTLY to j6 via the analytic swing–twist decomposition —
   rate-limited and clamped, so roll beyond j6's range saturates gracefully.
3. Solve the residual SWING with j4-j5 only (j6 frozen), with the unrealizable
   twist remainder removed from the target so the swing joints never contort
   through the wrist singularity to fake a roll.
4. Enforce soft limits around the configured rest pose, including the elbow floor.

Pure wrist roll lands on j6 BY CONSTRUCTION
(`tests/test_ik.py::test_roll_beyond_j6_range_saturates_without_contortion` pins
the saturation behavior). Mind the physical j6 range: ±120° motor with the frozen
rest at ∓90° leaves asymmetric roll headroom per side.

Do not reintroduce a hand-rolled Jacobian solver for arm control. If IK behavior
regresses, use `scripts/run_synthetic.py` first; it isolates line/circle/roll/pitch
and yaw targets without any headset or Unity dependency.

## Replay Contract

Use `run_teleop --record session.npz` to capture the exact head/wrist/finger stream
and engagement decisions for later debugging. `run_teleop --vr replay session.npz`
must drive the same `TeleopEngine`, `RenderSink`, and Unity JSON stream. Replay
sample selection uses recorded time, but `ReplaySource.latest()` refreshes
`VRFrame.stamp` to the current monotonic clock so the live staleness gate does not
drop valid recordings. `run_teleop --vr replay` uses the recorded engagement
decisions by default; overriding `--clutch` is for deliberate policy experiments.

To repeat a recording on the REAL arms, `run_hw --vr replay --loop-home` (dashboard:
RUN ON ROBOT + the `loop+home` toggle) plays the tape, then — when `ReplaySource.exhausted`
trips — glides the arms back to rest through `glide_arms_home` and `src.rewind()`s for the
next take. The home transition is a CONTROLLED, ENERGIZED glide: it commands the rest pose
through the existing `HardwareSink` shaper (never bypasses `safety/shaper.py`, never drops
the arm limp) and then re-syncs the engine IK and BOTH shapers to rest so the re-armed take
does not yank the arm back to the replay-end pose. No re-anchor happens between cycles (the
joint map is untouched); drift correction stays the manual RETURN HOME / RE-ANCHOR path.
`--home-dwell-s` / `hardware.replay_loop_dwell_s` set the hold at home; `--cycles N` caps
the run. The loop never re-runs the rest-pose engage gate mid-run because the chain stays
open for the whole session.

## Unity Contract

`src/bimanual_teleop/render_sink.py` publishes robot state. Unity consumes the TCP
JSON stream by default because it needs only `TcpClient` and `JsonUtility`.

Arm geometry is authoritative in Python. The render payload includes
`arms.*.link_pos`, a flattened base, j1..j6, EE polyline from the live Pinocchio
state. Unity should convert and draw those points; it should not duplicate FK
constants.

See `docs/UNITY_BRIDGE.md` and `unity/TeleopRenderer/README.md`.

## Bring-Up Gates

Use this hardware-free acceptance gate after runtime changes:

```sh
uv run python scripts/verify_stack.py
```

It runs:

- `uv run pytest -q`
- the body-relative teleop probe (`scripts/check_body_relative.py`)
- the YAM source-geometry provenance check (`scripts/check_yam_geometry.py`)
- synthetic YAM trajectories
- the static Unity render contract and render-state fixture freshness check
- launch CLI parsing for `run_teleop` and `run_hw`
- a headless `run_teleop --vr fake` smoke
- a record/replay launch smoke
- a Unity JSON monitor smoke

On a machine with Unity installed, also run:

```sh
uv run python scripts/run_unity_validation.py --require
uv run python scripts/verify_stack.py --unity-editor
```

Unity Editor compilation and Quest rendering are external to this machine and must
be validated on a machine with Unity installed.
