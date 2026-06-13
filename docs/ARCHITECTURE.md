# Architecture — How Every Piece Fits

One page to understand what you are running and what you are deploying.

## Data Flow

```
Quest 3 headset                     macOS / Linux teleop host                          outputs
───────────────                     ─────────────────────────────────────────────     ─────────────
ORBIT app (Unity)  ──ZMQ/adb──►  vr/orbit_source.py    converts Unity→WebXR frames
  hands 8087/8088                   │  (alternatives: vuer WebXR browser, fake
  wrists 8122/8123                  │   synthetic, replay of a recorded .npz)
  head  8200                        ▼
                                 VRFrame { head pose, per-hand wrist pose,
                                           25 landmarks, pinch }
                                    │
                    safety/supervisor.py + clutch ──► engaged? (staleness, hold,
                                    │                  e-stop, gesture/always)
                                    ▼
                                 engine.py (TeleopEngine)
                                    │ body_relative_hand_sample: torso→wrist in
                                    │ BODY axes [right, up, forward] (head pose +
                                    │ vr.torso_from_head; cancels walking/turning)
                                    ▼
                                 ClutchMapper (vr/frames.py)
                                    │ POSITION: absolute — chest + torso→wrist,
                                    │   glide-in on engage (engage_blend_s)
                                    │ ORIENTATION: relative world-frame — hand
                                    │   rotation since engage about body axes →
                                    │   EE rotation about matching world axes
                                    ▼
                                 arms/ik.py (Pinocchio/pink, per arm)
                                    │ 1 position (j1-j3) → 2 TWIST analytic on j6
                                    │ → 3 SWING QP (j4-j5); soft+hard limits,
                                    │ elbow floor; workspace box + anti-cross in
                                    │ arm_control.py
                                    ▼ joint targets q (6 per arm) + 17 hand dofs
                     ┌──────────────┴───────────────┐
                     ▼                              ▼
              render_sink.py                 hardware.py (Linux)
              ZMQ + TCP JSON render.state    JointCommandShaper per arm:
                     │                       limit-clamp + 1.2 rad/s cap +
       ┌─────────┬───┴────┬──────────┐       critically-damped PD smoothing,
       ▼         ▼        ▼          ▼       init from MEASURED pose
   Unity     dashboard  Rerun    analyzers          │
   headset   (browser)  (--viz)  render_session     ▼
   renderer                      analyze_session  YAM CAN (i2rt, MIT-mode motor
                                                  PD, 400 ms motor watchdog)
                                                  + ORCA hands (serial)
```

The hand path is parallel: raw landmarks → `hands/quest_retarget` → 17 ORCA joint
degrees (never goes through arm IK).

## The Mapping Contracts (CLAUDE.md is normative)

- **Position — absolute, body-anchored.** Robot chest = your torso. No
  calibration; per-user variation is absorbed by `vr.torso_from_head` (sternum
  offset below the headset — the only knob that is *about the operator*, default
  fits most adults) and `mapping.pos_scale`.
- **Orientation — relative, world-frame.** Rotation since clutch-engage about
  your body axes maps to the same rotation about robot world axes. No stance
  calibration; `vr.calib_seconds` defaults to 0 (the old 5 s hold is legacy).
- Proven on a real recorded session via `scripts/analyze_session.py`:
  orientation axis error 1.4° median, absolute position correspondence 0.1 cm
  median.

## Failsafe Inventory (every layer, where it lives, how it's tested)

| # | Failsafe | Layer | Behavior | Test |
|---|----------|-------|----------|------|
| 1 | Staleness gate | `safety/supervisor.py` | VR sample older than `safety.staleness_s` ⇒ not engaged | `test_supervisor_estop_and_staleness` |
| 2 | Dropout HOLD | supervisor | brief tracking loss ⇒ hold last pose ≤ `hold_s`, then idle (never chases a frozen target) | same |
| 3 | Deadman release | supervisor + clutch | releasing the clutch on a LIVE feed disengages immediately | `test_clutch_release_disengages_immediately` |
| 4 | Latched e-stop | supervisor | zeros engagement until deliberate `reset()`; run_hw releases torque on exit | same |
| 5 | Fail-closed parsing | sources + calibrate | malformed/non-finite poses, missing head ⇒ hand reads UNTRACKED, never identity/raw fallback | pipeline gating tests |
| 6 | Engage glide | `ClutchMapper` | target equals current EE at engage; absolute correspondence reached over `engage_blend_s` | `test_absolute_position_glides_to_chest_correspondence` |
| 7 | Workspace box | `arm_control.py` | EE targets clamped to `safety.workspace` (base frame) | motion tests |
| 8 | Anti-cross guard | `arm_control.py` | each hand pinned to its own side of world-Y ⇒ arms can never collide at the midline | `test_calibration_aligns_forward_and_no_cross` |
| 8b | Hand min-separation | `safety/separation.py` via `engine.py` | the two wrist targets (or a parked arm's actual wrist) kept ≥ `safety.hand_min_separation` apart in 3D — clapped hands meet at contact distance, never interpenetrate | `tests/test_neutral_calib.py` separation + clap tests |
| 9 | Soft joint limits + elbow floor | `arms/ik.py` | home ± measured-workspace margins; j3 floor prevents hyperextension | `test_limit_margins_and_within_limits` |
| 10 | j6 roll saturation | `arms/ik.py` swing–twist | roll beyond the ±120° motor range pins j6 gracefully; NEVER smeared onto j4/j5 | `test_roll_beyond_j6_range_saturates_without_contortion` |
| 11 | IK velocity budget | `arms/ik.py` | per-joint `ik.max_vel`, derated ×`hardware.max_vel_scale` on run_hw | run_hw derate print |
| 12 | **Command shaper** | `safety/shaper.py` @ `HardwareSink` | every CAN command limit-clamped to the PHYSICAL hardstops, speed-capped (`hardware.rate_limit`), critically-damped PD smoothing, init from measured pose, NaN ⇒ hold | `tests/test_shaper.py` (6 tests) |
| 12a | **Motor↔model joint map** | `arms/joint_map.py` @ `YamArm` | the i2rt firmware and this repo use DIFFERENT joint zeros/signs (i2rt yam.xml j1∈[−2.6,3.05] vs rig j1∈[0,2π]); every command/state crosses a per-side affine map MEASURED on metal (`scripts/hw_bringup.py`); an unmapped YamArm refuses model-space commands | `tests/test_joint_map.py` |
| 12b | **Rest-pose engage gate** | `hardware.py` | a session only starts commanding an arm that MEASURES at the official rest pose (±`hardware.engage_pose_tol`/joint through the map) — catches wrong-arm-on-bus (j6 differs by π), ±2π boot wraps, zero drift, not-at-rest starts; fail = abort with per-joint deltas, torque released | `tests/test_hardware_gate.py` |
| 12c | **Energize policy** | `arms/yam_chain.py` @ `YamArm` | the runtime drives the CAN chain DIRECTLY — i2rt's MotorChainRobot is not used (its boot/runtime qpos checks assume i2rt zero conventions this rig's motors don't follow — confirmed on metal: hang reads j2≈−177°, j6≈+312°; its gravity comp assumes a TABLE mount). Energize = enable limp → wrap-normalize offsets from a fresh read → MIT PD holds the MEASURED pose; zero motion, no feedforward, motor-space clamp = mapped model limits; close() releases torque then switches motors OFF | `test_wrap_corrections_normalize_multiturn_readings` + bring-up steps |
| 12d | Partial-rig guard | `hardware.py` | only arms in `hardware.sides` are opened/commanded (engine output for others is dropped); `hardware.use_hands: false` skips ORCA entirely (RealHand MOVES the hand at connect) | `tests/test_hardware_gate.py` |
| 13 | Motor-side backstops | YAM firmware | MIT-mode PD + motor CAN watchdog (commands stop ⇒ motors stop) — VERIFY per rig: `scripts/hw_bringup.py --step watchdog`; configure via i2rt `motor_config_tool/set_timeout.py` | bring-up watchdog step |

Order matters: 1–5 decide *whether* to follow, 6–11 decide *where* to go, 12–13
bound *how fast anything can physically move* no matter what upstream does.

## Tooling Index

| Need | Command |
|------|---------|
| Full hardware-free acceptance gate | `uv run python scripts/verify_stack.py` |
| Live dashboard (status, 3D, joint angles) | `uv run python scripts/dashboard.py` → http://127.0.0.1:8180 |
| Guided first-contact bring-up (scan/rest/signs/verify/watchdog) | `uv run python scripts/hw_bringup.py [--step STEP]` |
| Keyboard jog (sim→real verification) | `uv run python scripts/jog_arms.py [--sink hw] [--side right]` |
| Instrumented single nudge (telemetry proof of motion) | `uv run python scripts/probe_nudge.py` |
| No-robot dashboard test pattern | `uv run python scripts/test_pattern.py` |
| Replay a recording at reduced speed | `run_hw --vr replay s.npz --speed 0.2 --rate-limit 0.5` |
| Dashboard replay studio (browser: recordings, analyze, speed, STOP ALL) | `uv run python scripts/dashboard.py --host 0.0.0.0` → http://&lt;host&gt;:8180 |
| Record a headset session | `run_teleop --vr orbit --record recordings/s.npz` |
| Score a recording vs the contracts | `uv run python scripts/analyze_session.py recordings/s.npz` |
| Watchable movie (hands vs robot meshes) | `uv run --with matplotlib python scripts/render_session.py recordings/s.npz --gif out/s.gif` |
| Rerun 3D viewer (live or replay) | `run_teleop --vr replay s.npz --viz` |
| Quest ingest diagnostics | `scripts/check_quest.py`, `scripts/check_roll.py` |
| Synthetic IK isolation | `scripts/run_synthetic.py` |

## Sim→Real Checklist (the Linux hardware day)

The Mac never talks to motors; the Linux host runs the SAME engine with the
HardwareSink. The arm starts AND ends every session hanging at the official rest
pose (docs/RESTING_POSE.md) — that pose anchors the motor↔model calibration and
the engage gate. In order:

1. **Host prep**: Ubuntu, `sudo ip link set can0 up type can bitrate 1000000`
   (and can1 when the second arm arrives), `uv pip install -e i2rt`. Set
   `hardware.sides` / `hardware.use_hands` in rig.yaml to what is PHYSICALLY
   wired (hands stay false until mounted — RealHand moves them at connect).
2. **Gate**: `uv run python scripts/verify_stack.py` on the Linux host (everything
   that passes on the Mac must pass there).
3. **Guided bring-up (first metal contact, per arm)**: arm hanging at rest,
   e-stop in hand, dashboard up — `uv run python scripts/hw_bringup.py`.
   Steps: `links` (bus up) → `scan` (exactly motors 1..6 answer) → `rest`
   (capture the hanging pose; i2rt boot-check verdict) → `signs` (±2°
   motor-space nudges vs dashboard preview ⇒ measured per-joint signs)
   → `verify` (±3° model-space wiggles THROUGH the saved map: dashboard and
   metal must match on every joint) → `watchdog` (stop the stream mid-hold;
   motors must go limp on their own). Writes `config/hw_joint_map.json`.
   Re-run `--step rest` whenever the rest-pose gate complains (±2π boot wraps).
4. **Keyboard jog — no headset**: `scripts/jog_arms.py --sink hw` (10°/s cap by
   default; `m` prints the MEASURED pose). Single joint ±3°, every joint, then
   EE nudges. Confirm direction matches the dashboard, speed feels like the cap,
   hardstops respected. Hand on the e-stop.
5. **Replay on hardware**: `run_hw --vr replay recordings/roll_right.npz`
   — a session you have already watched in sim, now on metal. No surprises
   allowed: same motion, slower (derated).
6. **Live teleop, gesture clutch**: `run_hw --vr orbit --clutch gesture`.
   Engage one hand at a time (lift the right hand first, confirm, then the
   left). Verify dropout HOLD by covering a hand; verify deadman by releasing
   the gesture; verify Ctrl+C releases torque and the arm settles to the hang.
7. Only then consider raising `hardware.max_vel_scale` / `rate_limit`
   incrementally.

Known unknowns to verify on metal (cannot be tested here): CAN bus latency under
both arms + hands, ORCA serial throughput, thermal behavior, and whether the
motor CAN watchdog is actually configured on THIS rig's motors (`hw_bringup
--step watchdog`). Confirmed on metal 2026-06-11: the motor zeros do NOT follow
i2rt's convention (hang reads j2≈−177°, j6≈+312°) — expected and fine for the
runtime (chain-level, measured map), but i2rt's OWN tools/examples will reject
this rig's poses at boot; don't "fix" that by re-zeroing motors without
re-running the full bring-up.

## What Is Deliberately NOT Here

- No MuJoCo at runtime (`scripts/check_no_mujoco_runtime.py` enforces it).
- No startup calibration ritual; no per-user stance fitting (see contracts).
- No hand-rolled Jacobian solvers — position/swing are pink QP diff-IK; the only
  analytic step is the exact 1-DoF j6 twist assignment.
