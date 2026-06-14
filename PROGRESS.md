# Progress

The repository has been reworked away from the old local MuJoCo simulator toward a
headless body-relative teleop runtime with a Unity render stream.

## 2026-06-14 (Quest live teleop on BOTH arms; calibration reordered to home-last; dashboard cleanup)

Live Quest→metal teleop is wired end-to-end on this host: both arms calibrated
and homing, ORBIT hand-tracking reaching the engine over adb, and a single
**TELEOP LIVE** button driving both arms continuously.

**Host / Quest bring-up**
- `adb` installed WITHOUT sudo (Google platform-tools in `~/platform-tools`,
  symlinked into `~/.local/bin`). Quest authorized via a udev rule
  (`/etc/udev/rules.d/51-oculus-adb.rules`, `MODE=0666` for idVendor `2833`).
  The ORBIT app (`com.ORBIT.Teleoperation`) launches over adb; the engine's
  automatic `adb reverse` lands ORBIT's NetMQ PUSH on the host.
- NOTE: the Quest must be in **hand-tracking** mode (controllers down/off) or it
  streams nothing — controller-constellation tracking suppresses hand tracking.

**Calibration** (`vr/neutral_calib.py`) — fixes the "stuck measuring" stall + ends at home
- capture order is now **extended-forward → clap → REST (arms-down) LAST** (was
  rest→clap→forward). Arms-down is the robot's HOME and the easiest pose to hold
  still, so the capture closes reliably and the operator finishes at home.
- `HOLD_S` 2.5→1.5 s and `STILL_TOL` 0.03→0.06 m so the hold actually closes.
- `tests/test_neutral_calib.py` updated to the new order/contract.

**Hardware engage / homing**
- `hardware.engage_pose_tol` loosened j1–j4 `0.15→0.30` rad (still ≪ wrong-arm
  180° / ±2π wrap, so those are still caught) so a slightly-drifted hang engages.
- **RETURN HOME** (`launch/return_home.py` + dashboard button) drives BOTH wired
  arms to home (rate-limited) and RE-ANCHORS the rest offset at the settled hang
  (the automatic `--step rest`), so the engage gate then passes. Guard loosened
  to 0.50 rad so normal drift (~13° j4 seen on the right arm) is re-anchored, not
  rejected. Replaced the old RE-ANCHOR REST button.
- `run_hw --clutch always` added (`AlwaysOn`) so hardware teleop follows
  continuously.

**Dashboard** (`scripts/dashboard.py`)
- **TELEOP LIVE** = `run_hw --vr orbit --clutch always --sides right,left` —
  drives BOTH arms, follows continuously (gesture/deadman clutch removed per
  operator request). Replaced the right-arm-only HW REPLAY / HW TELEOP buttons.
- kill-strays sped up: `_busy_ports` uses `SO_REUSEADDR` so a TIME_WAIT socket is
  no longer counted busy (engine restart ~8 s → ~0.7 s).
- fixed the flickering `#hint`: a stale engine-log trip was overwriting it every
  ~1 s while `hint(d)` rewrote it every frame.

**Calibration data / CAN**
- `config/hw_joint_map.json` carries the good left+right map (signs
  `[1,1,-1,-1,1,1]`). LEFT = `can0` (bytewerk candleLight, serial `002E…`),
  RIGHT = `can1` (canable.io, serial `0044…`). `canN` is USB-enumeration order
  and can flip on replug.

**Open / next**
- CAN-channel udev locking (`config/99-yam-can.rules`,
  `scripts/setup_can_persistence.sh`) exists locally but is NOT yet applied or
  committed — do this before relying on two-arm teleop across reboots/replugs.
- one anchor-guard test + a cosmetic calib-start log line (`engine.py:321`) still
  reference the old pose order — harmless, cleanup pending.
- both-arm live teleop is first-contact: keep speed modest, hand on the e-stop.

## 2026-06-12 (metal day complete: jog + real-speed replay on the right arm; dashboard replay studio)

The right arm passed the full sim→real checklist on can0. Bring-up ran clean:
`--step scan` found exactly motors 1..6 (no stock gripper on the bus), `--step
rest` measured a still hang (drift 0.02°), `--step signs` measured the
motor↔model direction signs **[+1,+1,−1,−1,+1,+1]** (j3/j4 inverted vs the
model) and wrote config/hw_joint_map.json (offsets anchored at the official
rest, gate 0.0°/joint), `--step verify` + `--step watchdog` passed, the keyboard
jog moved every joint and EE-nudged with the metal tracking the dashboard, and a
synthesized 46 s bimanual episode (recordings/bimanual_demo.npz — raise / reach /
sweep + ±35° wrist rolls + finger curls; analyzer PASS, 0.1 cm correspondence)
replayed on the metal at REAL speed. Operator verdict: "mapping is good for the
right arm."

Tooling hardened during the session (each was a real fault that made a working
system look dead):

- **rest-pose gate** now compares SHORTEST-ARC: j2 rests 1.8° from ±180°, so the
  wrap-normalizer's branch choice flips between sessions; the raw difference
  false-alarmed −352.7°.
- **jog** (`scripts/jog_arms.py`): re-asserts the target every 60 Hz tick (the
  hardware shaper is a tracker — one call per keypress moved one rate-limited
  step and parked, silently diverging metal from page); ARROW keys jog/select
  (raw `os.read` — buffered stdin made ↑ quit the session); motion keys gated to
  the metal's tracking speed (autorepeat ran the commanded target away to +320°);
  a live `meas jN / gap` encoder readout; a render-mirror tee so the dashboard
  shadows the jog 1:1; `--side` for single-arm bring-up.
- **replay speed**: `ReplaySource.speed` time-stretches a recording;
  `run_hw` / `run_teleop --speed 0.2` plays 5× slower; `run_hw --rate-limit` and a
  dashboard mirror tee added. `probe_nudge.py` = one-command instrumented nudge
  (telemetry proof of motion); `test_pattern.py` = no-robot dashboard wave.
- **dashboard replay studio** (`scripts/dashboard.py`): recordings browser with
  per-file metadata (duration / frames / engaged %) and an `analyze` button
  (offline contract grader), a speed slider (drives PREVIEW and the composed
  metal command), a copy-paste `run_hw` command for the right arm, and a red
  **STOP ALL** that SIGINTs every teleop / jog / bring-up / replay process on the
  host (clean torque release). The page still launches PREVIEW only — metal stays
  a terminal command with a hand on the e-stop. New `/recinfo` + `/analyze`
  read-only endpoints.

Reminders for the other side: only the RIGHT arm is wired (can0;
`hardware.sides: [right]`, `use_hands: false`); the joint map is per-machine
(config/hw_joint_map.json, gitignored — re-measure with `hw_bringup` on a
different rig); after any torque release the limp wrist sags, so re-run `--step
rest` (re-anchors offsets) at the start of a session, and expect a few degrees of
gravity sag when the arm is held far from the hang (PD without feedforward).
Browse the dashboard at http://&lt;host&gt;:8180 (binds 0.0.0.0; no ssh tunnel).

## 2026-06-11 (sim→real boundary: measured joint map, rest-pose gate, guided bring-up)

First metal session prep (right arm only, single gs_usb adapter on can0). Reading
the i2rt SDK (v1.1.2, editable at ~/i2rt) surfaced three pre-metal hazards:

1. **i2rt's energize defaults are wrong for this rig.** Constructing
   `get_yam_robot()` starts a 250 Hz stream immediately, in zero-gravity mode
   with gravity-comp torques from a MuJoCo model that assumes a TABLE-MOUNTED
   YAM — our arms hang SIDEWAYS, so those torques push the wrong way at process
   start. `YamArm` now constructs with `zero_gravity_mode=False` +
   `gravity_comp_factor=zeros(6)`: energize = MIT PD holding the MEASURED pose,
   zero motion, gravity as a bounded PD disturbance (kp 80/80/80/40/10/10).
2. **The firmware joint convention is NOT the repo convention.** i2rt yam.xml:
   j1∈[−2.62, 3.05], j2∈[0, 3.65], j3∈[0, 3.67] vs rig.yaml j1∈[0, 2π] (hang at
   j1≈π!), j2∈[−2.88, 0.44], j3∈[−1.40, 2.44] — different zeros, unknown signs,
   and i2rt ±2π-wraps any boot reading past π. Raw model-space commands could
   sweep the arm. New `arms/joint_map.py`: per-side affine map (signs measured
   on metal, offsets anchored at the official rest pose), persisted per-machine
   in `config/hw_joint_map.json` (gitignored). `YamArm` refuses to command
   without it.
3. **Partial rig + wrong-bus hazards.** rig.yaml had left→can0 but the physical
   right arm is on can0 (only adapter present; can1 doesn't exist). Channels
   swapped; `hardware.sides: [right]` + `use_hands: false` (RealHand MOVES the
   hand at connect) gate what HardwareSink may open; and a REST-POSE ENGAGE GATE
   (±`engage_pose_tol`/joint through the map) refuses sessions where the arm
   isn't measurably at rest — which also catches a wrong arm on the bus (left vs
   right j6 differ by π), ±2π boot wraps, and zero drift, before any tracking.

New guided bring-up: `scripts/hw_bringup.py` (links → scan → rest → signs →
verify → watchdog), chain-level (no i2rt model assumptions), every motion ±2-3°
around rest at ≤10°/s with explicit confirmations; `signs` mirrors each nudge on
the dashboard so the operator just answers same/opposite; `watchdog` validates
the motor-side deadman by stopping the stream mid-hold. `jog_arms --sink hw` now
defaults to a 10°/s cap, starts on a wired side, prints MEASURED pose on `m`;
`jog_right.py` is a deprecation shim (its old form pushed unmapped model angles
with i2rt default energize). `run_hw` gained `--sides/--no-hands`. Failsafe rows
12a–12d + a rewritten hardware-day checklist in docs/ARCHITECTURE.md; rig
contract now pins the new hardware keys. Tests: `test_joint_map.py` (8) +
`test_hardware_gate.py` (5); verify_stack passes (212 tests + probes/smokes).

Open items for the metal day: does the hanging pose pass i2rt's boot qpos check
(rest step predicts it); is the motor CAN watchdog configured (watchdog step
verifies; `set_timeout.py` fixes); measure the actual signs.

**Metal addendum (same day):** first contact answered the boot-check question —
NO. A default-config i2rt run on can0 (LINEAR_4310 path, not our stack) reached
`MotorChainRobot`'s qpos check and was rejected: at the hang the motors read
j2≈−3.088 rad (−177°) and j6≈+5.438 rad (+312°) — the zeros follow neither
i2rt's convention nor ours, and a >π reading survived i2rt's single-shot boot
wrap fix. (That run also got far enough to prove the whole chain answers —
including, apparently, a motor at ID 7: scan will confirm whether the stock
gripper is still on the bus.) Consequence: the runtime CANNOT go through
MotorChainRobot at all. `YamArm` now drives the chain directly via the new
`arms/yam_chain.py` (promoted from hw_bringup's ChainSession): enable limp →
multi-turn wrap normalization from a FRESH post-thread read (`wrap_corrections`,
unit-pinned — handles the observed +312°) → MIT PD hold at measured, stock yam
gains, no feedforward, motor-space clamp = the mapped model limits (i2rt xml
limits are meaningless under these zeros). i2rt's boot verdict table in
`--step rest` is now informational (it only governs i2rt's own tools).
verify_stack green (213 tests).

## 2026-06-10 (absolute orientation — the overlay-overlap fix)

Operator, looking at the dashboard overlay: the tracked hand skeleton and the
ORCA hand "are not overlapping" (≈90° off on some axis). Measured: under
RELATIVE orientation the skeleton↔robot-hand attitude offset was 127–181° and
DRIFTING — relative mapping can never make them coincide.

Change (`mapping.orientation_mode: absolute`, default): the robot hand wears the
operator's hand attitude — R_t = R·ctrl_R·C (proper: the two reflections cancel)
with the hand↔EE convention C DERIVED, not calibrated: EE side from the frozen
rest contract (hand_basis_for_side), hand side from measured hand-local
finger/palm axes (`mapping.hand_finger_axis/hand_palm_axis`, from the recording's
palm-on-desk phase). Engage-latched offset decays with the shared glide.

Results on the real recording: skeleton↔commanded attitude 0.00° post-glide
(2.6° via the analyzer's stricter reconstruction), and IK orientation tracking
improved 68°→0.1° median — absolute attitudes are reachable ones, so the wrist
now works mid-range instead of pinned at ±89°.

Bug found on the way (the probes "exploded"): several synthetic fixtures built
RAW wrist rotations from `op_axes` — a REFLECTION (det −1) that real devices
never produce. Relative deltas cancelled it; absolute attitude inherited it and
handed the QP a reflection target → instant joint-limit wedge. Fixed all
fixtures (proper rotations; rigid body-motion carries the head rotation), added
a det<0 fail-closed guard in the mapper + regression test, and recorded the
frame-hygiene rule in CLAUDE.md.

165 tests + verify_stack pass; overlay coincidence screenshot-verified.

## 2026-06-10 (real ORCA hands)

Operator pushback ("don't you have a model of the orca hands? that is not the
model") was justified: the official description lives in the SIBLING repo
`~/Developer/orcahand_description` (v2 MJCF body fragments + per-side STL sets —
the same split-file pattern as the YAM sources, i.e. from the same original
scene). The earlier "no meshes exist" conclusion came from searching only inside
`orca_core` — lesson recorded: search the whole Developer tree before concluding
an asset is missing.

- `viz/yam_meshes.load_orca_hand`: synthesizes a loadable MJCF per side (assets
  + body fragment, materials stripped, paths absolutized), 30 visual geoms, 17
  joints whose names are exactly `<side>_<orca_to_sim_short(name)>` — this
  repo's joint_map was built for these models. `orca_q_from_degrees` maps the
  streamed 17 hardware angles (degrees) onto the model (radians, limit-clamped).
- Dense organic shells (8k–100k tris each) defeat largest-face decimation, so
  `simplify()` does quadric decimation via the optional `fast-simplification`
  dep (telemetry extra), disk-cached per mesh; crude fallback when absent.
- Dashboard ships hand geometry once via /meshes (with per-geom skin/structure
  colors) and per-state finger FK transforms (`hand_T`); render_session draws
  the same models per frame. The parametric hand (`viz/hand_geom`) remains only
  as a fallback when the description repo is absent.

163 tests + verify_stack pass; dashboard screenshot-verified with the real
hands articulating on the looping recording.

## 2026-06-10 (intrinsic twist + full-robot render)

Operator feedback: position perfect, but "the wrist still gets into singularities
when I'm turning it — should really just be turning J6"; also asked to render the
entire robot.

Root cause of the remaining singularity feel: orientation mapped fully
EXTRINSICALLY (world axes). A wrist turn about the operator's forearm axis only
lands on j6 when the EE tool axis happens to be parallel to that world axis;
otherwise it decomposes at the robot into twist + SWING, and the swing drags
j4/j5 through the wrist singularity.

Change (`mapping.twist_mode: intrinsic`, default): the hand rotation since engage
is decomposed about the operator's forearm axis (`mapping.hand_twist_axis` =
[0, 0.456, 0.890] in the ORBIT hand frame, measured from the real roll session at
77% single-axis energy). The twist is applied about the EE's OWN tool/j6 axis
(the body-relative reflection compensated via det(R), pinned by an
aligned-equivalence test where intrinsic must equal world mapping exactly); the
residual swing maps through world axes as before. ArmIK exposes
`ee_tool_axis_local` (constant). The analyzer now verifies the full contract by
independently reconstructing the predicted EE rotation from raw data: 1.9° median
/ 3.5° p90 on the real recording.

Honest physics surfaced by the recording: with always-on engagement from a
desk-rest anchor, raising the hand without re-orienting it keeps commanding the
rest attitude (tool axis down) at chest height — near the wrist's reach/limit
envelope, so achieved-vs-commanded orientation strains (IK tracking median ~68°
on this worst-case session). Live gesture-clutch re-engages re-anchor the
attitude and clear the offset; if live feel demands more, the designed next step
is absolute orientation with a fixed hand↔EE convention (no stance calibration
needed). Documented in CLAUDE.md.

Full-robot render: the AgileX stand frame STLs self-assemble (one assembly frame,
mm) — `viz/yam_meshes.load_stand_meshes` places them at `stand.pos`; drawn in the
dashboard (static /meshes entry) and the render_session GIFs. ORCA hands ship no
meshes anywhere, so robot hands remain EE triads.

## 2026-06-10 (hardening pass) — Hardware Safety, Dashboard, Keyboard Jog, Architecture

Operator sign-off on the mapping ("working now, perfect") triggered the
production-readiness pass:

- `safety/shaper.py` — JointCommandShaper, the last line of defense at the
  hardware boundary: every CAN command limit-clamped to the PHYSICAL hardstops,
  per-joint speed-capped (`hardware.rate_limit`, default 1.2 rad/s), and smoothed
  by a critically-damped second-order tracker (`hardware.smooth_hz`, the
  command-side "PD" feeding the YAM's motor-side MIT PD). Initializes from the
  arm's MEASURED pose (no startup snap; supersedes the never-implemented
  `safety.ramp_s`). Fail-closed on non-finite targets; sub-stepped so loop
  hiccups cannot violate the rate cap. Six unit tests pin all guarantees.
- `HardwareSink` now commands ONLY through the shaper; `run_hw` derates
  `ik.max_vel` by `hardware.max_vel_scale` (default 0.35). Rig contract enforces
  sane hardware-section values.
- `scripts/dashboard.py` — stdlib-only browser dashboard consuming the existing
  Unity TCP JSON stream: connection/age/Hz, per-side TRACKED/ENGAGED chips,
  calibration banner, drag-to-rotate 3D view (both arms, cmd-vs-achieved EE,
  operator torso→wrist vectors), per-joint angle tables with limit-margin
  highlighting, cmd−ee errors. Verified end-to-end against a live fake session
  (connected, 105 Hz, real q + wrist vectors over HTTP).
- `scripts/jog_arms.py` — keyboard jog through the REAL ArmIK and the same sinks
  (render for sim preview, `--sink hw` for the Linux host through the shaper):
  per-joint stepping with soft-limit clamps, world-frame EE nudges via the
  two-stage solve, home, live dashboard publishing. JogSession unit-tested.
- `docs/ARCHITECTURE.md` — the one-page system map: data flow Quest→…→CAN, the
  mapping contracts, a 13-row failsafe inventory (each layer + its test), the
  tooling index, and the ordered sim→real hardware-day checklist for the Linux
  host. README/CLAUDE.md point to it.
- Fixed in passing: rig.yaml restructure had orphaned `safety.workspace` under
  the new `hardware:` section (caught by the live smoke; full suite would have
  caught it too — lesson: run the full suite after config surgery).

158 tests + full verify_stack pass. Hardware-day items that can only be verified
on the Linux host with motors: CAN latency under load, YamArm motor-count
padding, ORCA serial throughput, watchdog/e-stop behavior on metal — listed in
ARCHITECTURE.md.

## 2026-06-10 (later still) — Swing–Twist Wrist: Roll Goes Straight To j6

Operator feedback: wrist orientation causes "singularities"; suspected j6 missing
from the model. j6 IS modeled (the synthetic/IK roll tests prove a pure tool-axis
roll lands on j6), but two real failure modes existed: j6 roll headroom from the
frozen ∓90° rest is asymmetric (±120° motor ⇒ ~+97°/−30° left, mirrored right),
and once j6 saturated — or the roll axis wasn't exactly the tool axis — the
orientation QP smeared the remaining roll onto j4/j5, folding the wrist through
its singularity.

Change (`arms/ik.py`): the orientation stage is now swing–twist. The twist
component of the orientation error about the CURRENT j6/tool axis is computed
analytically and assigned DIRECTLY to j6 (rate-limited, limit-clamped); the
remaining swing is solved by the QP on j4-j5 only, with the unrealizable twist
remainder REMOVED from its target so the swing joints never contort to fake a
roll. Position/swing remain real QP diff-IK.

New tests: roll beyond j6 range ⇒ j6 pins at its limit, j4/j5 move <3°, arm
still, residual error ≈ exactly the dropped twist; combined twist+swing converges
<2° with j6 carrying the twist. 149 tests + full verify_stack pass.

Honest physics that remains: genuinely large tool-axis RE-AIMS (the recorded
diagonal-axis roll demands them) are real j4/j5 swings bounded by j5's hard ±90°,
and roll headroom depends on roll direction per side because the rest pose is
frozen at ∓90°. The analyzer's IK-tracking metric reports exactly these
saturations (median ~11° on the high-roll recording) without contortion.

## 2026-06-10 (later) — Absolute Body-Anchored Position Mapping

Operator feedback on the rendered replay: "this is all happening in front of me,
yet the robot arms stay to the sides and down." Correct observation — position was
clutch-RELATIVE, so the robot only mirrored displacements from its hanging rest.

Change: `mapping.position_mode: absolute` (new default). The torso→wrist vector
maps 1:1 (×`pos_scale`) onto the robot's chest→wrist vector; the chest anchor
defaults to the arm-base midpoint dropped by `mapping.body_anchor_drop` (0.15 m —
the plates are the shoulder line; the workspace, like a human's, is below it). On
(re)engage the EE glides onto correspondence over `mapping.engage_blend_s` (no
snap; displacement deltas map 1:1 from the first tick). Orientation stays
relative-latched (the proven world-frame mapping). Legacy `body_relative: false`
forces relative position mode.

Enablers found while validating:
- `ArmController` used to build its mapper with the legacy raw-WebXR basis and
  relied on the engine to patch in the body-relative basis afterwards; it now
  selects the right basis at construction (absolute mode made this load-bearing).
- The conservative `ik.soft_margin` values (tuned around the hanging rest) made
  the front-of-chest workspace unreachable — j1/j3/j5 pinned at their margins
  0.4 m short of target. Margins re-sized from measured IK excursions over the
  absolute workspace (max |q−home| ≈ [2.4, 0.65, 1.4, 0.5, 1.55, 0.5]) + headroom.
- The anti-cross guard correctly pins a hand that crosses the body midline.

Verification: 147 tests + full `verify_stack` pass (probes updated to settle the
engage glide and to use reachable torso-height poses). `analyze_session` is now
position-mode aware; on the real recording: windowed translation direction error
1.9° median, magnitude ratio 1.03, **absolute correspondence |cmd − (chest +
torso→wrist)| = 0.1 cm median (0.7 cm p90)**, orientation unchanged at 1.4°. The
`render_session.py` side-by-side now shows both arms raised in front, matching
the operator's hands. UNVERIFIED: live feel; reach-ceiling clamping when hands go
far above the operator's chest is geometry-honest but worth feeling out.

## 2026-06-10 — Scrambled-Wrist Root Cause Found And Fixed (On Real Quest Data)

Symptom (reported from live use): hand tracking is good and holding the hands
forward roughly works, but any wrist rotation makes the robot move about seemingly
random axes.

Diagnosis — `scripts/analyze_session.py` (new) replays a recorded session through
the REAL `TeleopEngine` and grades the commanded motion against the operator's
motion. On `recordings/roll_right.npz` (real Quest capture of a right-wrist roll):

- translation mapping: 0.7° median direction error → fine (matches "hands forward
  works");
- orientation mapping: **145.5° median world-axis error**, angle magnitude
  preserved → every rotation came out about the wrong axis;
- IK tracking: 0.0° — the solver faithfully executed the wrong command (the
  earlier synthetic-roll finding "IK is sound" confirmed end-to-end);
- the 5 s stance calibration in that session graded itself SHAKY (33–37 mm) and
  the stance was not arms-at-sides — and the engine used the result anyway.

Root cause: EE orientation went through a hand-local→EE-local correspondence `P`
built from the startup arms-at-sides hold. The correspondence is only right if the
operator exactly mirrors the robot's rest stance during calibration; any deviation
re-labels which hand axis is which and scrambles all commanded rotation axes.

Fix (`ClutchMapper.target`): orientation now uses the SAME change of basis as
translation — the wrist rotation since clutch-engage as a left/world-frame delta,
conjugated into the arm base frame by the one constant `R`, applied about the EE
anchor. Calibration-free; roll→roll/pitch→pitch/yaw→yaw from any starting pose;
body-turn invariant. The left-handed `[right,up,forward]` bases (`head_op_axes`,
`W_AXES`) each carry a reflection and the conjugation cancels them exactly —
`tests/test_frames.py::test_clutch_orientation_body_relative_real_rig_axes` pins
this on the real per-side `base_quat`s. Same real recording after the fix: **1.4°
median axis error (p90 4.6°), angle ratio 0.99**.

Consequences folded in:

- `vr.calib_seconds` defaults to **0** — no startup ritual; arms follow once
  tracked + engaged. The stillness hold remains opt-in for legacy
  `body_relative: false` diagnostics, and the rig contract rejects a nonzero
  default and the removed knobs (`mapping.abs_orientation`,
  `mapping.ori_tweak_euler`).
- Dead machinery removed: `ArmController.set_ori_calib`, `set_ref_frame`,
  `ArmIK.ee_semantic_frame_local`, mapper `P`/`set_P`/`abs_orientation`/
  `freeze_ori`/`_R_off`.
- `run_teleop --viz` (and `--viz-save out.rrd`): local Rerun 3D viewer — both arm
  link chains, commanded vs achieved EE triads, the operator torso→wrist vector +
  wrist triad mapped through the SAME body→world axes arm control uses, error and
  clutch/pinch plots. Works with fake/synthetic/replay/orbit sources; `--vr
  replay` scrubs the whole session on the Rerun timeline. No Unity, no headset.
  Requires `uv sync --extra telemetry` (optional dependency, runtime stays lean).
- README gained "Debugging Without A Headset" + rewrote the mapping section;
  CLAUDE.md gained the Orientation Mapping Contract.

Verification: `uv run pytest -q` → 144 passed (new: world-frame orientation
contract, real-rig reflection cancellation, body-turn orientation invariance,
engage continuity, Rerun viz smoke). `uv run python scripts/verify_stack.py` →
all gates pass; headless loop ~105 Hz on this Mac (no MuJoCo anywhere in the
runtime). UNVERIFIED until the next headset session: the subjective feel of the
fixed mapping live (the recording-based scorer says the axes are now right).

## Current State

- `TeleopEngine` uses `vr.body_relative` and `vr.torso_from_head` so arm motion is
  driven by torso-to-wrist vectors instead of raw room-space hand positions.
- Calibration and calibration stillness use the same body-relative wrist frame when
  head samples are available.
- Arm IK runs on standalone Pinocchio/pink YAM models, with two-stage wrist-position
  and hand-orientation solves.
- The render path is `RenderSink`, not `SimWorld`.
- Unity receives `render.state` over newline-delimited TCP JSON at
  `127.0.0.1:8102` by default, plus ZMQ/msgpack at `tcp://127.0.0.1:8101` for
  Python tools.
- The Unity scaffold under `unity/TeleopRenderer` draws primitive YAM arms from
  Python-published `arms.*.link_pos` and draws operator torso-to-wrist vectors from
  `op.hands.*.wrist_body`; `scripts/check_unity_contract.py` statically checks
  the Unity DTOs, endpoint defaults, coordinate conversions, and scene bootstrap.
- `run_teleop --record session.npz` captures head/wrist/finger frames plus engage
  state; `run_teleop --vr replay session.npz` replays them through the same engine
  and render stream for deterministic debugging, using recorded engagement by
  default.
- The old `run_sim.py`, MuJoCo viewer, mapping studio, and overlay tools have been
  removed from the runtime surface.

## Verification

Latest local hardware-free gate:

```sh
uv run python scripts/verify_stack.py
```

Result:

- pytest: 141 passed
- rig contract: pass; default config keeps body-relative mode, Unity render
  endpoints, zero legacy mapper trim, measured elongated-stand base poses/quats,
  MJCF-derived ORCA flange transforms and YAM joint limits, frozen rest pose, and
  removed MuJoCo runtime entrypoints; regression tests also reject disabled
  body-relative mode, non-finite torso offsets, non-positive mapping scale, and
  disabled absolute wrist orientation
- no MuJoCo runtime: pass; runtime Python imports, project dependencies, and
  locked packages do not include `mujoco`, `mink`, or `dm_control`
- body-relative teleop probe: pass on both arms; headset translation/yaw drift is
  effectively zero and wrist lift increases robot-world Z
- body-relative Unity render payload probe: pass on both arms; the Unity-facing
  `render.state` payload keeps `arms.*.cmd_pos` stable under headset translation
  and yaw when `op.hands.*.wrist_body` is unchanged, then lifts the commanded
  target by 16 cm when the torso-relative wrist vector is lifted by 16 cm
- body-relative arm gating: pass; tracked hand samples without a head pose are not
  allowed to fall back to raw XR-world wrist coordinates for arm control; ORBIT and
  Vuer startup frames now expose `head=None` rather than an identity placeholder
  until a real headset pose arrives; malformed/non-finite Vuer and ORBIT
  messages fail closed instead of becoming identity hand/head poses; custom/replay
  body-relative samples with non-finite headset or wrist matrices fail closed at the
  arm-control boundary, and calibration ignores non-finite pose samples instead of
  anchoring to them; ORBIT wrist pose freshness is tracked separately from finger
  landmarks, so fresh landmark packets cannot revive a stale wrist pose for arm
  control
- body-relative render gating: pass; Unity `status.tracked` and operator overlay
  wrist vectors are false/null when a hand is tracked but the headset pose or
  finite wrist pose needed to form torso-to-wrist motion is missing; non-finite
  custom/replay headset or wrist matrices fail closed in the arm command path and
  operator overlay without leaking `NaN` or `Infinity` into strict Unity JSON;
  malformed/non-finite `vr.torso_from_head` values also fail closed for arm-control
  samples and fall back to a finite default in the Unity overlay
- render publisher headset-gating contract: pass; even with legacy
  `vr.body_relative=false`, the publisher does not fabricate Unity
  `op.hands.*.wrist_body` without a headset pose, while preserving the separate
  status tracking semantics
- render monitor body-state contract: pass; strict monitor mode requires
  `status.tracked.*` to match `op.hands.*.tracked` and requires `wrist_body` to be
  null when a side is gated/untracked; it also validates `op.torso_from_head` and
  the nullable `op.head_pos`/`op.torso_pos` pair before Unity consumes the stream,
  rejects stale `render.state` schema versions, rejects non-finite arm, hand,
  operator, and status numeric payloads, and prints achieved-vs-commanded EE error
  (`cmd_err`) for terminal diagnosis without Unity Editor
- render schema nullability: pass; when no frame/head pose exists, operator debug
  state keeps both hand entries present with `tracked=false`, `wrist_body=null`,
  `head_pos=null`, and `torso_pos=null`
- YAM geometry provenance: pass; runtime Pinocchio joints/sites match the source
  MJCF body trees
- synthetic IK trajectories: pass for line, circle, roll, pitch, yaw on both arms
- Unity render contract: pass, including Unity C# DTO/static checks and
  `render_state_sample.json` freshness against Python's `RenderSink.build_state()`;
  Unity's `ExpectedSchemaVersion` is parsed and checked against Python's
  `topics.SCHEMA_VERSION`, and Unity TCP client host/port defaults are checked
  against `config/rig.yaml`'s `vr.unity_json_endpoint`; Unity Editor DTO
  validation now checks both left and right fixed-shape arm, hand, operator vector,
  and commanded-target payloads; the render schema includes `arms.*.cmd_pos`, the
  post-filter/post-clamp commanded EE target in robot world coordinates, so Unity
  can draw the command separately from the achieved EE pose and connect them with
  an achieved-to-command error line; the scene bootstrap contract and Editor
  validation require
  headset-oriented runtime settings for vSync, 72 FPS target frame rate, sleep
  prevention, and a dependency-free status HUD for stream, schema/stale/error,
  loop-rate, engagement/tracking, calibration, operator head/wrist state, and
  numeric left/right achieved-to-command EE error
- Unity renderer fail-closed behavior: pass; malformed fixed-shape arm and hand
  payloads on both left and right sides, null commanded targets, and
  schema-version mismatches hide the affected command marker or primitives instead
  of drawing stale/default geometry; the individual YAM arm, ORCA hand, and
  operator-vector renderers also reject non-finite numeric payloads on direct
  `Apply()` calls; the Unity TCP client rejects malformed
  top-level arm, hand, operator vector, `torso_from_head`, status, and calibration
  shapes plus non-finite numeric values before accepting a render state as current, hides
  renderers if no valid render state arrives before the configured stale-state
  timeout, rejects version-correct but incomplete top-level render states, and
  clears the HUD's latest-state backing data and shows the corresponding HUD status
  on stale, invalid, schema-mismatched, or malformed-JSON payloads
- Unity operator overlay fail-closed behavior: pass; nullable, untracked, or
  malformed fixed-shape `wrist_body` payloads on both left and right sides hide the
  corresponding torso-to-wrist marker and line without hiding the unaffected side
- Unity renderer initialization: pass; the YAM arm, ORCA hand, and operator-vector
  renderers initialize idempotently before `Apply()`, and the operator overlay keeps
  line renderers parented under the overlay object so Editor validation, scene
  cleanup, and Play mode use the same object hierarchy
- Unity material resilience: pass; primitive renderers and the bootstrap floor use
  shared material creation with built-in/URP/unlit shader fallbacks instead of
  duplicating `Shader.Find("Standard")` in each renderer; Unity Editor validation
  also checks that the material factory returns a non-null shader-backed material
  and preserves the requested color
- Unity scene/build utility: pass; Editor validation now ensures a saved
  `Assets/Scenes/TeleopRenderer.unity` scene and registers it in Unity build
  settings for desktop/Quest builds; bootstrap validation also proves existing
  cameras, lights, and floor objects are reused instead of duplicated when the
  renderer is embedded in an existing scene, and the runtime bootstrap still
  ensures those support objects when a saved/manual `TeleopRenderClient` already
  exists
- Unity project contract: pass; static checks require the scaffold to stay
  dependency-free, require the batch validator to target `unity/TeleopRenderer`,
  require Unity-generated project folders to stay ignored, and require every
  committed Unity asset/folder under `Assets/` to have a stable sidecar `.meta`
  file with a unique 32-character hex GUID and the expected importer type
- Unity validation runner contract: pass; optional/missing Unity behavior and the
  required editor-success-marker check are unit-tested without launching Unity; the
  runner also enforces a bounded Unity batchmode timeout and reports the log tail on
  timeout
- verify-stack Unity gate contract: pass; `--unity-editor` is unit-tested to call
  `scripts/run_unity_validation.py --require`, while the default hardware-free
  gate skips Unity Editor validation unless explicitly requested
- launch/diagnostic CLI help (`run_teleop`, `run_hw`, `check_quest`, `check_roll`): pass
- headless teleop smoke: pass
- record/replay launch smoke: pass
- replay body-relative fidelity: pass; saved sessions preserve missing headset poses
  as `head=None` on replay, and recorded headset+wrist pairs replay to the same
  `op.hands.*.wrist_body` torso-to-wrist vectors used by Unity and arm control
- Quest ingest diagnostic: pass; `scripts/check_quest.py` now prints the same
  body-frame torso-to-wrist vector (`body=[right up forward]`) when a head pose is
  present, and prints `body=NO_HEAD` instead of falling back to raw room-space wrist
  coordinates when the headset pose is missing
- Quest roll diagnostic: pass; `scripts/check_roll.py` analyzes wrist roll in the
  operator body frame and refuses to accept tracked hand samples without a valid
  headset pose
- Unity TCP JSON monitor smoke: pass; runs fake teleop with `--calib-seconds 0`
  and requires an observed active-command frame with both arms, both fixed-shape
  hand payloads, finite commanded EE targets, status flags, operator pose fields,
  and torso-relative wrist vectors; the TCP bridge serializes with strict JSON and
  drops non-finite frames instead of sending `NaN`/`Infinity` tokens to Unity; the
  monitor, Unity fixture generator, and static contract reject non-finite sample
  values too
- Unity Editor batch validation hook: present (`scripts/run_unity_validation.py
  --require`), but not run on this machine because Unity Editor is not installed

Unity Editor, `dotnet`, and `mcs` are not installed on this machine, so Unity C#
compilation and an in-editor/Quest visual test remain unverified here. On a Unity
machine, run `uv run python scripts/verify_stack.py --unity-editor`.

## Useful Commands

```sh
uv run python -m bimanual_teleop.launch.run_teleop --vr fake
uv run python scripts/render_monitor.py --seconds 5
uv run python scripts/render_monitor.py --transport json --require-hand-render --require-bimanual-state --require-command-target --require-frame --seconds 5
uv run python scripts/check_rig_contract.py
uv run python scripts/check_no_mujoco_runtime.py
uv run python scripts/check_body_relative.py
uv run python scripts/check_yam_geometry.py
uv run python scripts/run_synthetic.py
uv run pytest -q
uv run python scripts/check_unity_contract.py
uv run python scripts/update_unity_fixture.py --check
uv run python scripts/run_unity_validation.py --require
```

For ORBIT on Quest:

```sh
uv run python -m bimanual_teleop.launch.run_teleop --vr orbit --clutch gesture
```

`run_teleop --vr orbit` attempts `adb reverse` for render ports when `adb` is
available.

## Remaining External Validation

- Open `unity/TeleopRenderer` in Unity.
- Run `uv run python scripts/run_unity_validation.py --require` on a machine with
  Unity Editor installed.
- Press Play while `run_teleop --vr fake` is running.
- Confirm both arms draw from `link_pos` and the torso-to-wrist overlay moves as
  expected.
- Build/run on Quest or the intended Unity target.
- On hardware, validate CAN watchdogs, e-stop, and conservative velocity/limit
  settings before using gesture clutch with real YAM arms.
