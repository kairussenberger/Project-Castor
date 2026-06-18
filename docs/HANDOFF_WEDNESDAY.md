# Handoff — Dashboard bring-up tooling (Wed session, 2026-06-16 → 06-18)

All work from this session is committed on branch **`dashboard-bringup-tooling`**
and pushed to `origin`. It is **NOT on `main`** — see "Git state" below before doing
anything with git. `uv run pytest -q` = **214 passed** on the branch.

## Continuation 2026-06-18 (loop-on-robot + Quest app launcher)

Added after the above, same branch (`uv run pytest -q` now **219 passed**):

- **Loop a replay on the real arms** — `run_hw --vr replay --loop-home` (dashboard:
  RUN ON ROBOT + the new `loop+home` toggle). Plays the tape, then glides the arms back
  to rest and replays — the next take WAITS for the home transition. The glide is
  CONTROLLED + ENERGIZED (commands rest through the existing `HardwareSink` shaper, never
  drops limp, never bypasses `safety/shaper.py`), then re-syncs the engine IK + both
  shapers to rest so the re-armed take doesn't yank the arm back to the replay-end pose.
  No re-anchor between cycles. New: `ReplaySource.exhausted`/`rewind()`
  (`vr/replay.py`), `glide_arms_home()` + `--loop-home/--home-dwell-s/--cycles`
  (`launch/run_hw.py`), `hardware.replay_loop_dwell_s` (`rig.yaml`). Tests:
  `tests/test_replay.py` (exhausted/rewind), `tests/test_replay_loop.py` (glide,
  hardware-free).
- **📱 LAUNCH QUEST APP** dashboard button (top toolbar) — runs
  `adb shell monkey -p com.ORBIT.Teleoperation -c android.intent.category.LAUNCHER 1`
  to start the ORBIT app on the connected Quest; reports the adb outcome on the status
  line. **Restart the dashboard process to pick up new buttons** (the HTML is built in
  `scripts/dashboard.py`; a browser refresh alone won't do it).

**On-metal verify still needed** for `--loop-home`: confirm the home glide is smooth
across several cycles and watch `out/hw_telemetry.json` motor temps (over-temp on j1 is
the known risk from below).

## TL;DR of what shipped

The goal was making the real-hardware bring-up doable entirely from the dashboard,
plus fixing a chain of real bugs found along the way.

1. **Motor wrap-frame fix** (`arms/yam_chain.py`) — encoder reads now fold to the
   rest-anchor frame (`wrap_ref`) instead of `(−π, π]`. The rig's j2 rests at ≈±177°,
   right on the fold boundary, so reads were flipping a full turn between samples →
   flaky engage gate and a **350° HOME sweep** on the right arm. Threaded through
   `YamArm` ← `HardwareSink` (`wrap_ref = jm.to_motor(neutral)`) and `return_home`.
   Pinned by `tests/test_joint_map.py`.

2. **Live motor telemetry** — `YamArm.telemetry()` / `HardwareSink.telemetry()` expose
   per-joint **temp / effort / measured pose**. `run_hw` (and the jog server) write
   `out/hw_telemetry.json` ~5 Hz; the dashboard reads it for:
   - per-joint **motor temp °C** with a **"⚠ MOTOR HOT"** banner (thresholds
     `TEMP_WARN=58`, `TEMP_HOT=72` in `dashboard.py` — **guessed; tune to the real
     DM4340 trip**),
   - **"motor measured ° (vs cmd)"** per joint — a joint moving the wrong way shows a
     big red delta (the live orientation check).

3. **LIVE LOG panel** on the dashboard — streams `out/engine.log` with color, plus
   **copy failures** (grabs the last crash/traceback) and **copy all** buttons.

4. **Fluidity** — `hardware.smooth_hz` 3.0 → **4.0**; new `hardware.replay_rate_limit`
   (**1.0** rad/s) so RUN ON ROBOT isn't clipped at the old hardcoded 0.5.

5. **By-hand calibration on the dashboard** (the main ask) — a CALIBRATE row:
   - **↑ CAN UP** — `ip link set canN up` for both buses (needs passwordless sudo for
     `ip`, else it reports the error).
   - **⌂ RE-ANCHOR REST** — `return_home --anchor-only`: captures the current limp hang
     as rest, **no motion**. Recovery after a channel swap.
   - **▶ JOG hw** — starts `jog_arms.py --sink hw --server` (holds arms, UDP 8202).
   - **per-joint −10° / +10°** — jog one joint on the real motor to **see direction**.
   - **per-joint ± (sign flip)** — flips that joint's motor direction and re-anchors.
   - **swap L/R arms** — swaps `arms.*.can_channel` in `rig.yaml` (comments preserved).

6. **Mapping knobs** (`mapping.*`, engine-applied, all default off now):
   - `swap_sides` — right arm does the left hand's motion (and vice versa).
   - `mirror_forward` / `mirror_lateral` — single-axis reflections (proper rotation
     for orientation, no QP blowup). Launch flags `--swap-sides`, `--mirror-fb`,
     `--mirror-lr` on `run_hw` and `run_teleop`; dashboard toggles on the replay row.

## Hardware state the operator set by hand (in `rig.yaml`, currently)

- **Channels swapped: `left.can_channel: can1`, `right.can_channel: can0`** (operator's
  edit — the physical arms were reversed vs the old config).
- **`mapping.swap_sides: false`** — L/R is now fixed at the channel level, so the
  software swap is OFF (having both on double-swapped and made `roll_right` look wrong).
- Joint map (`config/hw_joint_map.json`, **gitignored**) was re-anchored and had signs
  flipped by hand via the dashboard. Operator reported **orientation is now correct**.
- `config/operator_calib.json` (gitignored) has `axis_scale ≈ [1.1, 1.66, 1.66]` — it
  **amplifies** a small recorded hand motion into a ~1 m EE swing; that's why the left
  arm chased an unreachable target in `roll_right`. Re-record without it for clean tapes.

## Things verified vs still open

Verified (sim / tests / metal): wrap fix (right j2 +350° → −9.5°, stable); swap moves
the reach left↔right through the real launcher; `roll_right` restored to original with
`swap_sides:false` (left arm tall reach); body-relative contract passes; jog server
applies UDP nudges; 214 tests pass.

**Open / next:**
- **Reconcile git** (see below) — the branch must be merged into `origin/main`.
- **Motor over-temp is a real risk**: j1 (DM4340) tripped during repeated `roll_right`
  runs — sideways mount + **no gravity-comp feedforward** → high holding current. When
  `run_hw` crashed on the fault it **left the arm energized** (the rig's 400 ms CAN
  watchdog is still unverified). Two fixes worth doing: make `run_hw` force torque-release
  on a hard fault, and actually verify/enable the motor watchdog. JOG mode also holds the
  arms energized — STOP between joints, watch the temp readout.
- **Tune temp thresholds** to the real DM4340 trip once observed.
- **On-metal verify** still needed: jog each joint's direction, then a live teleop run.

## Git state ⚠️ (read before pushing)

- Work is on branch **`dashboard-bringup-tooling`** (commit `510ff5c`), pushed to
  `origin`. PR link:
  `https://github.com/kairussenberger/Project-Castor/pull/new/dashboard-bringup-tooling`
- **`origin/main` has DIVERGED**: split at `f9da703`; origin/main is **19 commits ahead**
  on a different line (S-curve smoothing, curated replay library, preflight doctor, and a
  commit that already fixed the same dashboard `\n`-in-JS bug), local is **7 ahead**.
- Do **not** force-push to `main` — it would erase those 19 commits.
- To integrate: open the PR, or rebase `dashboard-bringup-tooling` onto `origin/main`
  and resolve conflicts (heavy overlap in `scripts/dashboard.py` — both lines rewrote it).

## Files touched

`config/rig.yaml`, `scripts/{dashboard,jog_arms,check_body_relative,update_unity_fixture}.py`,
`src/bimanual_teleop/arms/{yam_chain,yam_driver}.py`, `src/bimanual_teleop/{engine,hardware}.py`,
`src/bimanual_teleop/vr/calibrate.py`, `src/bimanual_teleop/launch/{run_hw,run_teleop,return_home}.py`
(return_home is new), `tests/{test_joint_map,test_hardware_gate,test_neutral_calib}.py`.
