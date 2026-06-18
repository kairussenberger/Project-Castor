# Session change notes

Running log of changes made by Claude during working sessions, newest first.
Each entry: what changed, why, and exactly which files were touched.

---

## 2026-06-12 (afternoon) — replay library shipped, clasp crossing fixed, fingers forensics

Live session day (first with the morning's guard/grading/doctor). 262 tests.

**Replay library** (`replay_library/`, committed, 34 MB; docs/REPLAY_LIBRARY.md
pins scores): six probe tapes recorded via dashboard sessions, trimmed
(`scripts/trim_session.py`), each EMBEDDING its session's calibration
(`calib_json` in the npz; recorder/replay/engine/analyze all wired) — anchors
moved ~1 m BETWEEN consecutive sessions, so a tape without its fit is
meaningless (reach_box scored 77 cm phantom error through identity, 1.0 cm
through its own fit). run_teleop now embeds automatically at save.

**Clasp crossing FIXED** (user report, live): clap.npz dissection showed 48%
of separation pushes were majority fore/aft SHEAR (closest-point axis at
fingertip-vs-palm for near-parallel staggered capsules) — the hands slid past
each other. separate_capsules blends the push axis to the WRIST LINE as axes
approach (anti-)parallel (λ=|cos|), magnitude scaled by the gradient
component; perpendicular pokes keep the gradient. Tape re-dissect: 0%
majority-shear, 0 direction flips.

**Fingers/“right wrist moves too much” forensics**: split into (a) an ANALYZER
artifact — it scored through the LIVE head frame while the engine maps through
the LOCKED yaw; fixed → pick_place/reach_box/clap became clean PASSes
(1.3°/1.4°/0.7° median — real manipulation tracking was always fine); and
(b) a REAL tracker limitation — sustained finger articulation makes the
whole-hand hypothesis wander (slow 120–160° attitude swings + ~0.5 m drift,
coherent across landmarks AND wrist stream → uncatchable by cross-checks;
engine faithfully danced: j5/j6 450–1000° cumulative on a parked-arms tape).
Remedy = operational: dashboard START LIVE gained a clutch selector
(always | gesture); finger-only work runs on the gesture deadman.

**Anchor guard validated live, organically**: the abandoned 11:11 session =
the user RECENTERED mid-calibration → guard tripped ("both wrists jumped
together (1.46 m)"), capture cancelled, arms locked — exactly as designed.
Offline replay of that tape reproduces the same trip deterministically. The
user read it as "reanchoring the hands did not work": recenter is NOT a
re-zero mechanism — the 3-pose recalibration is (the banner says so). Note
the CALIBRATE-button toggle (press-again = CANCEL) likely ate the two
follow-up capture attempts that session. Still to do live: a trip while
DRIVING (the 11:11 trip fired while locked-but-capturing), plus the
occlusion-no-trip and sloppy-calibration probes, engage_cycles tape.

Also: dashboard page-bricking JS SyntaxError fixed mid-morning (a Python-
interpreted \n inside the new CAL-chip tooltip killed the whole inline script
— every button dead while the server was healthy; node --check now pins the
page in CI), and a headset_view ops note: the startup reaper kills the
PREVIOUS instance's ffmpeg — one instance at a time, last one wins.

---

## 2026-06-12 — anchor-jump guard, fit grading, preflight doctor

Detail-ironing day (no headset). Three pieces, each closing a seam the
2026-06-11 sessions exposed. 252 tests green (was 209), `verify_stack` passes,
Unity fixture regenerated twice (status.guard, arms.clamp_dist — both
additive). NOT yet validated live — first live session should deliberately
recenter mid-drive and watch the guard trip + recalibrate.

**1. Mid-session anchor-jump guard (`safety/anchor_guard.py`, failsafe 5c).**
*Why:* the 7th-pass `follow_locked` only guards session START; a recenter,
ORBIT app restart, or headset sleep mid-session moves the stream anchors and
the applied fit goes silently wrong — and the existing teleport rejection then
re-anchors and GLIDES the arms onto the shifted mapping (wrong, smoothly).
*What:* the guard watches the BODY-RELATIVE wrist signal (downstream of
whatever `vr.orbit_wrist_anchor` reconstructs — anchors that cancel can never
false-trip). Channel A: one-sample discontinuity beyond
`jump_m + speed_allow·dt` → side SUSPECT (arm holds, mapper fed nothing);
snap-back within `return_tol` = glitch → resume; persistence
`confirm_frames` → verdict via the OTHER hand (shared stream anchor): coherent
both-hand jump or other-hand-untracked → TRIP; other hand steady → accept
(same policy as teleport rejection). Channel B (live transports only): whole
stream dead > `blackout_s` → TRIP (app restart/sleep reset anchors — and
previously resumed silently on the stale fit!). TRIP = `follow_locked`,
in-flight capture CANCELLED (poses straddling an anchor change must not fit),
persistent dashboard/in-headset banner; only a completed calibration (which
absorbs the new anchors) unlocks + resets. Re-trips re-fire the response (a
jump during the post-trip recapture must cancel THAT too). Thresholds sized
from 19 real recordings: normal per-frame deltas p99 < 3 cm @ ~9 ms; *one-frame
0.2–1.5 m glitch teleports appear in MOST sessions* (why single-hand jumps must
never trip alone); benign stream hiccups ≤ 0.9 s, real blackouts 6 s / 31 s
(why `blackout_s: 2.0` separates cleanly). Duplicate frames (latest() repeats)
are skipped — confirm counts run on real samples, not engine ticks.
Config `safety.anchor_guard.*` in rig.yaml. Engine: samples hoisted in tick(),
`_guard_tick` before capture/plan paths, guard armed when following OR
capturing, reset inside `_apply_calibration`. `status.guard` published;
dashboard shows the red trip banner + `⚠ TRACKING TRIP` chip.

**2. Calibration fit grading + live WS-clamp indicator.**
*Why:* the 9th-pass broken fit was only diagnosable by evening forensics on a
recording; a bad fit must announce itself at completion, and a stale one must
show up while driving.
*What:* `fit_two_pose` now grades itself through EXACTLY the runtime map —
`lateral_curve` extracted from `ClutchMapper._lat_scaled` (one implementation,
shared): per-side neutral residual (3D) and rest residual (up/forward only —
the lateral map is anchored at clap width + extended spread, rest width is
outside its claim), plus raw-scale clip detection. `quality` =
grade good/check/bad + named reasons + residual table, persisted in the calib
JSON meta, in `summary()` → dashboard CAL chip (colour + `±worst cm`, reasons
in tooltip) and the completion banner. Validated on the REAL 2026-06-11
capture: refit reproduces the famous −2.191 m offset, capture grades GOOD at
2.3 cm worst (the capture WAS good — the clip was the bug); the old clipped
parameters would have screamed BAD at 139 cm in the banner. Live signal:
`ArmController.plan` publishes `clamp_dist` (how far the raw mapped target sat
OUTSIDE the workspace box — a broken mapping pins targets at a box face, it
never just "feels far"); dashboard shows a per-arm row + a sustained-EMA red
`WS CLAMP — mapping off? recalibrate` header chip.

**3. Preflight doctor (`scripts/doctor.py`).**
*Why:* 06-11 lost ~30 min to a 4-day-old `orbit_to_unity.py` squatting all
seven ORBIT ports, and twice to wedged ffmpeg screen-captures (zero frames, no
error).
*What:* report → `--fix` → `--json`; exit 1 on failures. Checks: stray
processes (old bridge = always fail; ffmpeg avfoundation with no live
headset_view; engines > 12 h = orphans), foreign listeners on the teleop ports
(reported, NEVER auto-killed — `--fix` only signals OUR OWN tooling patterns,
INT before KILL so engines save recordings), adb state + missing reverse
tunnels (fixable; matters for an already-running engine), persisted-calib
load-screen + grade + age, the PROVEN iCloud venv trap (Mobile Documents path
or UF_HIDDEN .pth → CPython ≥3.12 skips them), recordings disk usage.
README gains step-0 preflight + operator notes for the guard/grades.

**Files.** NEW: `src/bimanual_teleop/safety/anchor_guard.py`,
`scripts/doctor.py`, `tests/test_anchor_guard.py` (17),
`tests/test_calib_quality.py` (6), `tests/test_doctor.py` (20).
TOUCHED: `engine.py` (guard wiring, `_neutral_tick(wb,…)` signature),
`vr/frames.py` (`lateral_curve` extraction), `vr/neutral_calib.py`
(`_fit_quality`, summary, banner msg), `arms/arm_control.py` (`clamp_dist`),
`render_sink.py` (`status.guard`, `arms.*.clamp_dist`), `scripts/dashboard.py`
(trip banner/chip, CAL grade chip, WS CLAMP chip + card row),
`config/rig.yaml` (`safety.anchor_guard`), `.gitignore`
(`operator_calib.json.*` — the .bak forensics copy escaped the pattern),
`tests/test_neutral_calib.py` (guard disabled in the teleporting-fixture
capture test), README, ARCHITECTURE (failsafe 5c + tooling rows), Unity
fixture.

**Behavioral notes.**
- Replaying a recording that CONTAINS an anchor jump now locks follow at the
  jump (correct mirror of live). For deliberate glitch studies:
  `safety.anchor_guard.enabled: false`.
- `--vr fake` never trips (synthetic motion is smooth, fake/replay exempt from
  the blackout channel).
- Parked for later (from the priorities discussion): finger retargeting
  quality grading (#3), curated hardware-day replay library (#5), latency
  instrumentation.

---

## 2026-06-11 (9th pass) — headset view WORKS + the floating-arms calibration bug

**Why (headset view).** The 8th-pass stream decoded but was unusable in the
headset. Three stacked causes: (1) `StereoSbsRenderTexture.shader` ALWAYS
splits the texture side-by-side — left half→left eye, right half→right eye —
so a MONO frame put a different half of the screen in each eye (binocular
rivalry); (2) the reprojection config type must be exactly
`orbit_stereo_reprojection_config` with PER-EYE dims (ours was silently
rejected as `unexpected_type`); (3) the avfoundation screen device needs an
explicit `-pixel_format nv12`, and WEDGED ffmpeg captures from earlier runs
starve new ones (zero frames, no error — two 4-hour-old orphans were doing
exactly that).

**What (headset view, `scripts/headset_view.py` + `vr/orbit_source.py`).**
The captured screen is letterboxed to one eye then DUPLICATED into both SBS
halves (header 2880×810, per-eye 1440×810, zero disparity → flat 2D panel);
config type/dims fixed; `fps=` filter clamps CFR-sync duplicate frames
(~270/s → 30); stale pipelines reaped at startup + an 8 s no-frames watchdog;
`VIDEO_PANEL_PORT 10505` added to the quest-link watchdog's tunnel list
(tunnel-only, never PULL-bound). **CONFIRMED working in-headset.**

**Why (calibration).** After the 7th-pass clap calibration, the arms hovered
at the workspace ceiling. Forensics on the recorded session
(`live_0611_204016.npz` + the persisted fit's meta): ORBIT's wrist and head
streams recenter-anchor INDEPENDENTLY — the wrists measured ~1.35 m above
the torso proxy ALL session. The fit's offset is the designed absorber, but
`fit_two_pose` clipped it at ±0.8 m (needed −2.19) → +1.4 m baked into every
up-target. Earlier sessions only worked because the mismatch stayed < 0.8 m.

**What (calibration, `vr/neutral_calib.py` + `engine.py`).** The fitted
offset is NEVER clipped (finite-only; `OFFSET_MAX` 0.8→10.0, demoted to a
load-time corruption screen). Capture order is now REST → CLAP →
EXTENDED-FORWARD LAST (pose A maps onto the robot neutral by construction,
so the arms engage gliding onto a correspondence the operator already
holds). Engine yaw latch hardened: NaN warm-up / looking-straight-down head
samples never latch the session yaw frame (`TeleopEngine._head_latchable`,
fail-closed until a sane head arrives — a replay's first head sample IS
NaN). Validated against the broken recording end-to-end: the held
outstretched pose lands within 1.5 cm of the robot neutral through the live
mapper (the clipped offset commanded +1.39 m above it). README steps 4–5
updated; regression tests pin the real-session fit, the load bounds, and the
latch guard. **CONFIRMED working live.**

---

## 2026-06-11 (8th pass) — see the dashboard INSIDE the headset (solo operation)

**Why.** The ORBIT app blocks sight through the headset; working alone meant
operating blind. The app has a built-in in-headset video panel that SUBs on
tcp://127.0.0.1:10505 (meant for a robot camera) — it was sitting blank.

**What.** NEW `scripts/headset_view.py`: captures the Mac screen with ffmpeg
(hardware HEVC via VideoToolbox, all-intra gop=1), packetizes it in ORBIT's
wire format (reverse-engineered from CameraOneStreamer.cs: ZMQ PUB, 'ORBT' +
version + flags + dims + ptsUs + payloadLen + Annex-B access unit) and
publishes on 10505 (sets up its own adb reverse). A JSON reprojection-config
packet (pinhole intrinsics, FLAG_CONFIG) is sent every 2 s — gotcha found the
hard way: FLAG_CONFIG packets bypass the decoder entirely, so video frames
must never carry it. Verified end-to-end via the app's own diagnostics:
render_state=LIVE_HEVC, packets parsed/enqueued 1:1, renderer live.

**Usage.** `uv run python scripts/headset_view.py` (first run: grant the
terminal Screen Recording permission in System Settings → Privacy & Security).
`--list-screens`, `--fps`, `--width/--height`, `--bitrate` to tune. Quest's
built-in double-tap-for-passthrough (Settings → Physical Space) is the
zero-code complement for glancing at the real keyboard.

---

## 2026-06-11 (7th pass) — safety: yaw lock + calibration-required, clap as pose 3/3

**User requirements.** (1) Head motion (looking left/right, removing the
headset) must NEVER produce arm motion — safety-critical. (2) No joint
commands until calibration completes — force a calibration every session.
(3) Claps still landed displaced; add the clap to the calibration.

**1. Body-frame YAW LOCK (`vr.body_yaw: locked`, default).** Head POSITION
already cancels in the body-relative subtraction (the ORBIT reconstruction
rides the live head), but head ROTATION entered through the head-derived
body axes: turning the head rotated every wrist target around the chest
anchor — both a safety hazard and (since the operator watches the dashboard
mid-clap) a likely contributor to the displaced claps. The yaw frame is now
LATCHED: from the first head sample at startup, then re-latched to the
operator's ARM-DEFINED forward when a calibration completes (stored as
`forward_body` in the fit). Looking around or pulling the headset off is
now ZERO input by construction; removal also drops hand tracking →
supervisor HOLD → idle (existing layers). Trade-off (documented): turning
your whole BODY now moves the frame with you only after recalibrating —
correct for a stationary operator station, and ORBIT data cannot
distinguish head from body turns anyway. The dashboard's operator panels
use the same locked frame (`operator_debug_state(head_R_override=…)`), so
display == control. `check_body_relative*.py` probes and the pipeline test
now model the REAL stream semantics (wrist values ride head position, not
rotation) and pin the zero-input contract.

**2. Calibration REQUIRED per session (`vr.require_calibration: true`).**
Live transports (orbit/vuer) start with `engine.follow_locked`: arms hold
rest and ignore all hand input until the in-session capture completes; the
stale-file auto-load is skipped (a fresh ORBIT recenter anchor invalidates
any previous absolute fit — measured 0.5 m shifts). The dashboard shows a
persistent gold banner: "ARMS LOCKED — press ⊕ CALIBRATE…". Clearing the
calibration re-locks. fake/replay are exempt (deterministic gate/analysis).
run_hw inherits the same engine behavior → hardware also cannot move
pre-calibration.

**3. Clap = calibration pose 3/3.** The capture is now extend-forward →
arms-at-sides → PALMS PRESSED TOGETHER. Pose C anchors the LATERAL map at
contact width via piecewise-linear knots [[x_clap → robot contact half-gap
(`mapping.robot_clap_gap`/2)], [x_spread → robot half-spread]]: the
operator's measured clap maps to the robot's hands touching BY CONSTRUCTION
(no scale guessing), and the midline (`lat_center`) is measured where the
palms actually meet — the best possible midline estimate. Gates stay
relative/anchor-proof (pose C: hands close + raised back up from pose B).
Calibration file v4 (`lat_knots`, `forward_body`); older files still load
(legacy quadratic ramp path kept in the mapper).

**Files.** `vr/neutral_calib.py` (pose C, knots, forward_body, prompts
1/3–3/3), `vr/frames.py` (piecewise lateral curve, lat_knots), `engine.py`
(yaw latch + re-latch, follow_locked, knots wiring), `render_sink.py`
(`status.follow_locked`, locked-frame operator display), `scripts/dashboard.py`
(ARMS LOCKED banner), `config/rig.yaml` (`vr.body_yaw`,
`vr.require_calibration`, `mapping.robot_clap_gap`),
`scripts/check_body_relative.py` + `_render.py` (new stream-semantics
fixtures + zero-input contract), tests updated/extended.
Gate: **207 tests + probes green.**

---

## 2026-06-11 (6th pass) — TWO-POSE calibration (recenter-anchor-proof)

**Why.** "The calibrate function did not work": two live sessions
(live_0611_152433/152622) started the capture but never completed it.
Forensics: (1) hand tracking was only ~43% that session; (2) EVERY hand
position read ~0.5 m too high and ~0.2 m short — the ORBIT app's positions
live in a RECENTER-ANCHORED frame, and starting the app/recentering with the
headset on the desk moved that anchor; the head-anchored reconstruction from
the 2nd pass assumed anchor ≈ live head and broke. There is NO in-data
absolute reference (the hand keypoints share the same anchor — verified:
keypoint[0] ≡ wrist-stream translation). A one-pose calibration fundamentally
cannot separate an unknown anchor shift from the operator's proportions, and
its absolute pose gate (forward ≥ 0.25 m) trusted the broken frame and
refused forever. A secondary trap found on the way: the operator watches the
dashboard during calibration, so the HEAD (and the body frame's yaw) points
at the monitor — an absolute "forward" gate also fails for that reason.

**What.** `vr/neutral_calib.py` rewritten as a TWO-POSE capture:
  - pose A: both arms extended straight forward, hold 2.5 s;
  - pose B: arms relaxed at the sides, hold 2.5 s (banner walks through 1/2,
    2/2 with progress).
  All fitted quantities are now anchor- and head-yaw-proof:
  - operator FORWARD = horizontal direction of the A−B wrist-midpoint delta;
  - lateral scale from the pose-A wrist SPREAD; up/forward scales from the
    per-axis A−B deltas vs the robot's matching references
    (`mapping.robot_neutral_wrist` ↔ A, NEW `mapping.robot_rest_wrist` — the
    robot's actual rest pose — ↔ B): anchors cancel in every difference;
  - the offset (pose A) absorbs the anchor; the operator's measured midline
    (`lat_center`, new mapper field) maps to the robot's midline — the
    non-linear lateral ramp now centers there;
  - capture gates are RELATIVE only (stillness, lateral spread 0.2–0.8 m,
    pose-B drop ≥ 0.22 m below pose A) — nothing trusts the broken frame.
  Calibration file v3 (`lat_center`); v1/v2 files still load.
  Also: `vr.orbit_wrist_anchor: keypoint` (head + keypoint[0]; numerically
  equal to 'head' for current ORBIT builds since keypoint[0] ≡ wrist stream,
  but conceptually direct), OFFSET_MAX 0.4 → 0.8 (must cover anatomy + anchor).

**Validation.** Replaying the exact session where calibration refused: pose A
accepted at 7.1 s, completed at 32.7 s (the operator's later arms-rest served
as pose B), sane scales [1.41, 1.48, 1.84]. Unit suite includes THE
regression: a constant 0.5 m anchor shift on all inputs must produce
identical scales and map pose A exactly onto the robot neutral.
Gate: **205 tests + probes green.**

**Operator notes.** Recentering the headset mid-session moves the anchor
again — if the mapping suddenly feels shifted, just recalibrate (8 s).
The stale single-pose calibration file was deleted; recalibrate on the next
session. Hand tracking was 43% in the failed session — keep hands in the
Quest cameras' view; the TRACKED chips show per-hand status live.

---

## 2026-06-11 (5th pass) — per-side hand axes, off-center claps, pair-order anti-cross

**From recordings/live_0611_143958.npz** (clap landed with the hands ~26 cm
apart and diagonally displaced; left wrist roll still sweeping).

**1. Left-roll sweep — root cause finally found: the shared hand-axis
constants were 21–45° wrong.** Measured per-side from 14.7k tracked frames
across three sessions (4–5° scatter): the two hands are clean anatomical
MIRRORS (x flips sign), and the old shared `hand_finger_axis` /
`hand_palm_axis` (measured once, palm on a desk) sat 21° / 45° off. The
hand↔EE convention C built from them mis-mapped every attitude — rolls leaked
into swing. All three hand-local axes are now PER-SIDE dicts in rig.yaml with
the measured values (`config.side_axis()` helper reads either form).
REPLAY RESULT: left q4 range [−1.56,+0.18] → [−0.63,+0.71], zero ticks at
the soft floor (was pinned for stretches), now symmetric with the right arm
([−0.95,+0.55]) — "copy exactly what the right arm does" achieved.
`check_rig_contract` now requires the per-side form + mirror-pair sanity.

**2. Clap displacement — three stacked causes, measured:**
   - The user clapped LEFT OF CENTER (right hand 2 cm left of the body
     midline). The old anti-cross guard was per-side HALF-SPACES about the
     midline: the robot's right hand was pinned at +5 cm and could never
     follow — 24 of the 26 cm gap. REPLACED with a PAIR-ORDER constraint in
     the engine's separation step: the right wrist stays ≥ 2·cross_gap to +Y
     OF THE LEFT WRIST — the pair may sit anywhere laterally (off-center
     claps are legal); only their order and minimum lateral gap are enforced.
   - The calibration's lateral scale (×1.6) amplified absolute lateral
     positions — clapped hands (±4–9 cm) were stretched apart. The lateral
     scale is now NON-LINEAR: ≈1:1 near the midline, quadratic ramp to the
     full calibrated scale at the operator's neutral width (`lat_ref`, stored
     in the calibration file v2; v1 files derive it from their meta — no
     recalibration needed).
   - The capsule guard provides the final contact floor.
   REPLAY RESULT: closest commanded wrist gap 26.6 → **16.3 cm**, almost
   purely lateral (vertical offset 2.0 cm — the diagonal displacement from
   the screenshot is gone).

**3. Cleanups.** Shaper seed clipped to the soft limits (pink warned about
1e-6 overspill); body-motion-invariance test fixture had BOTH hands at the
same pose (degenerate — the guards pushed coincident targets around), now
mirrored; the legacy no-cross test asserts the new pair semantics.

**Files.** `config/rig.yaml` (per-side hand axes — measured values),
`src/bimanual_teleop/config.py` (`side_axis`), `arms/arm_control.py`
(per-side axes, half-space clamp removed, seed clipping),
`engine.py` (pair-order anti-cross in `_separate_hands`), `vr/frames.py`
(non-linear `_lat_scaled`, `lat_ref`), `vr/neutral_calib.py` (lat_ref in
fit/persist/load-with-derivation), `scripts/analyze_session.py` +
`scripts/check_rig_contract.py` (per-side schema), tests updated.
Gate: **199 tests + probes green.**

---

## 2026-06-11 (4th pass) — mirrored dashboard fix, left-roll taming, attitude smoothing

**From recordings/live_0611_133349.npz** (user: right-arm roll fixed, left
wrist still pivots sideways at the end; "left hand moves the right arm on the
model"; smoothing looked good).

**1. "Arms the wrong way around" = the dashboard 3D projection was MIRRORED.**
Engine-side mapping verified correct on the recording (left-hand lateral ↔
left-arm EE world-Y corr +0.80; cross-corr negative). The page's `camOf` built
the camera basis as `right = up × fwd` — LEFT-handed — so every panel rendered
horizontally mirrored: anatomically the robot's left arm drew where its right
should be (the L/R letters follow the mirror, hence "labeled correctly" yet
crossed). Fixed to `right = fwd × up`, `up = right × fwd`; the view buttons'
values were visually swapped by the same bug → swapped back, and the DEFAULT
camera is now a true BEHIND-the-robot view (over-the-shoulder embodiment: your
left hand drives the screen-left arm). `scripts/dashboard.py` only.

**2. Left-wrist roll pivot — three compounding causes found by replay:**
   - j6 headroom is asymmetric by the frozen rest contract (rest ∓90°, motor
     ±120°): the user's roll direction gives the left arm 30° where the right
     gets 210° — the left pins constantly, the right almost never (why "right
     is fixed, left isn't").
   - Past the stop, the wrapped twist error (±π) can FLIP SIGN and walk j6 the
     long way through the wrist singularity. Two-layer fix: continuity unwrap
     in `ik._apply_twist` while pinned (`ik.reset_twist()` on engage), plus a
     stateless demand-level roll clamp in the controller
     (`_saturate_roll`: roll measured from REST about the EE tool axis,
     clipped to the j6 window + 0.35 rad slack — small overspills keep the
     wears-the-attitude contract, the wrap zone is unreachable).
     `tests/test_guardrails.py` pins both sides rolled 6 rad past the stop:
     pinned, no long-way travel, j4/j5 < 0.3 rad wander.
   - Raw Quest wrist quats carry ~200°/s of HIGH-FREQUENCY jitter (worse on
     the rolled/occluded hand; position had One-Euro, orientation had NO
     filter). Added an attitude slerp low-pass to the governor
     (`safety.target_ori_smooth_s` = 0.12 s), merged with the angular cap.
   - Also re-measured `mapping.hand_twist_axis` from 9.5k fast-rotation
     samples across both sessions: [0.031, 0.311, 0.950] (old value ~10° off).
   Replay scoreboard (left arm, the roll exercise): j5 wander while pinned
   0.90 → 0.46 rad, q4-at-soft-floor ticks 98 → 59, no more multi-second
   pinned thrash stretches. HONEST RESIDUAL: j4 still sweeps up to ~1.2 rad
   during extreme left rolls — the demanded attitude genuinely contains that
   swing (hand-axis convention error + degraded tracking of the rolled hand);
   needs a live check and possibly per-side hand-axis measurement next.

**Files.** `scripts/dashboard.py` (camera basis, default view, button swap),
`arms/ik.py` (pinned twist unwrap + reset_twist), `arms/arm_control.py`
(_saturate_roll, attitude low-pass, rest references), `config/rig.yaml`
(hand_twist_axis re-measured, safety.target_ori_smooth_s),
`tests/test_guardrails.py` (+2 roll-saturation contract tests). Gate:
**199 tests + probes green.**

---

## 2026-06-11 (3rd pass) — singularity taming, motion guardrails, capsule clap fix

**Why.** Session recordings/live_0611_130501.npz (first with working
calibration): (1) left-wrist roll drove the IK through wrist singularities —
measured j4 20.5 / j6 25.1 rad/s with full-range j6 flips as j5 crossed zero
(j6 headroom is asymmetric: rest −90°, range ±120°); (2) operator asked for
deploy-grade guardrails: "a lot of movement in a short timeframe → simply
don't do that movement", smooth/slow everything; (3) hands STILL clapped
through each other — the point-pair guard held the wrist points 17 cm apart
while the fingertips reached 0.4 cm from the other palm.

**What.** Three layers, all sized from the recording (operator real-motion
p99 ≈ 2 m/s; tracking glitches 58–65 m/s):

1. **Target governor** (`arm_control.plan`): teleport rejection on the RAW
   body-frame wrist signal (> `safety.target_jump_speed` 3 m/s ⇒ the motion
   does not happen — mapper re-anchors, arm glides from its current pose);
   world-frame caps on commanded target speed (`target_speed_max` 0.8 m/s)
   and attitude rate (`target_ang_speed_max` 2.5 rad/s, clamped axis-angle
   slerp). Jump test deliberately reads the operator signal, NOT the mapped
   target — the engage glide legitimately demands ~1.5 m/s of target motion
   and must never read as a jump (first implementation got this wrong and
   livelocked under sustained fast motion).
2. **In-loop joint shaper** (`arm_control.commit`): the hardware-boundary
   `JointCommandShaper` (limit clamp + hard rate cap + critically-damped
   tracker) now runs in the sim/render path too — `safety.sim_rate_limit`
   1.8 rad/s, `sim_smooth_hz` 4.0. The shaped pose is seeded back into the
   IK (`ik.seed`) so FK/render/next-solve see ONE consistent robot. Replay
   verification: every joint-velocity peak in the session collapsed from
   17–25 rad/s to exactly 1.8 rad/s; singularity flips leave as bounded
   glides. Set sim_rate_limit = hardware.rate_limit (1.2) for exact
   hardware-day parity. Idle arms bleed velocity to a stop instead of
   freezing. NOTE: analyze_session IK-tracking-gap scores now include shaper
   lag during fast motion — that is the honest new behavior.
3. **Capsule separation** (`safety/separation.py`, `engine._separate_hands`):
   each hand is a capsule wrist → wrist + `safety.hand_capsule_len` (0.19 m)
   along the EE fingers axis; minimum segment-segment distance ≥
   `hand_min_separation` (0.12), iterated 4× (a single distance push
   under-resolves INTERSECTING anti-parallel capsules). Replaces the
   palm-point pass (`safety.palm_center_offset` removed). Additionally each
   arm's COMMAND must clear the other arm's ACHIEVED capsule — the shaper
   lag let achieved poses pass closer than their separated commands
   (measured 0.9 cm achieved vs 12 cm commanded); commanding only into
   unoccupied space fixed it. Replay: minimum achieved capsule gap across
   the whole session (clap included) 0.9 cm → 5.3 cm. Raise
   `hand_min_separation` to ~0.15 for more visual clearance.

**Files.** `arms/arm_control.py` (governor, shaper, fingers_dir refactor,
`commit(plan, t)`), `engine.py` (capsule + achieved-pose separation),
`safety/separation.py` (`closest_points_segments`, `separate_capsules`),
`config/rig.yaml` (guardrail knobs), `tests/test_guardrails.py` (NEW: 8
tests), `tests/test_neutral_calib.py` (capsule clap asserts),
`tests/test_pipeline.py` (legacy step-input test gets an explicit
`target_jump_speed` override — it teleports the wrist by design), Unity
fixture regenerated. Gate: **197 tests + probes green.**

---

## 2026-06-11 (later) — ORBIT frame-origin fix + palm-center separation

**Why.** First real calibrated session (recordings/live_0611_124653.npz)
failed two ways: (1) after calibration the arms sat at HIP height while the
operator held them extended forward; (2) clapping still drove the hands
through each other.

**Root cause 1 — mixed frame origins in the ORBIT stream.** Forensics on the
recording: the HEAD pose streams floor-anchored (head y ≈ 1.11–1.56 m) while
the WRIST poses stream EYE-anchored (resting hands y ≈ −0.3, range −0.74..
+0.57 — nowhere near floor-origin heights). The old orbit_source docstring
even said origin offsets "cancel because the clutch mapper is RELATIVE" —
stale since the mapping went ABSOLUTE. The body-relative subtraction
(wrist − head-derived torso) therefore carried a phantom ≈ −1.3 m on the UP
axis: the saved calibration meta recorded the operator's neutral at
up = −0.96 m (anatomically impossible), the fit clamped its offset at +0.4,
and the arms landed at hip height. Verified fix candidate against the
recording: re-anchoring wrist TRANSLATIONS at the live head position makes
every still window anatomically correct (hands-on-lap −0.26 up, raised
forward +0.10..0.15 up), and corr(Δhead_y, Δwrist_raw_y) ≈ +0.2 (not the
strong negative a camera-anchored stream would show) pins the wrist frame as
a FIXED eye-height recenter anchor. Rotation untouched (wrist attitudes were
already world-axes — which is why absolute ORIENTATION worked all along).
NOT a missing dependency; the friend's bridge is delta-based so origin
offsets cancel for him — our absolute mapping exposes them.

- `vr/orbit_source.py` — `latest()` re-anchors each fresh wrist translation
  at the live head position (knob `vr.orbit_wrist_anchor: head|world`,
  default head); module docstring rewritten with the measured evidence.
  Caveat documented: exact at normal posture; a deep crouch/lean couples in
  by the head's displacement since recenter.
- `config/rig.yaml` — `vr.orbit_wrist_anchor: head`.
- `tests/test_orbit_anchor.py` — NEW: recombination math, world passthrough,
  fail-closed without a head pose.
- Deleted `config/operator_calib.json` (fitted on corrupted frames — must
  not auto-load). **Operator must RE-CALIBRATE on the next live session.**
- NOTE: recordings made BEFORE this fix store the mismatched frames (the
  recorder captures post-source VRFrames), so absolute-position metrics on
  them are not meaningful; orientation metrics remain valid.

**Root cause 2 — separation guarded the wrong point.** The guard kept the
WRIST targets 12 cm apart, but the ORCA hand volume sits ~9 cm beyond the
wrist along the fingers axis — palms-facing claps collide at the palm
centers, which could still coincide. The guard now runs TWO passes: palm
centers first (projected via the commanded EE orientation; a parked arm uses
its live FK), then wrists, each pass shifting the engaged sides' wrist
targets along the violating line.

- `arms/arm_control.py` — `palm_off_ee` from the rest-contract hand basis ×
  `safety.palm_center_offset` (new knob, 0.09 m); `plan()` returns
  `palm_dir`; `palm_dir_world()` for parked arms.
- `engine.py` — `_separate_hands()` two-pass palm+wrist projection.
- `config/rig.yaml` — `safety.palm_center_offset: 0.09`.
- `tests/test_neutral_calib.py` — clap test now asserts palm-center gap too.

Gate after both fixes: **189 tests + all probes green.**

---

## 2026-06-11 — Operator neutral-pose calibration + hand minimum-separation guard

**Why.** Two operator-reported issues from the first live Quest sessions:
1. The 1:1 absolute position mapping doesn't fit the operator — the YAM's
   reach/mounting proportions differ from a human arm, so the mapping felt
   "off" with no way to adapt it.
2. Clapping the hands together drove the robot hands *through* each other in
   sim — the only guard was the world-Y anti-cross clamp (`vr.cross_gap`,
   ±5 cm), which neither models hand volume nor acts in 3D.

**What.**
- **Neutral-pose calibration** (position-only; orientation mapping remains
  calibration-free BY CONTRACT — see CLAUDE.md, the 145°-axis-error lesson):
  a runtime, operator-triggered capture. The operator extends both arms
  straight forward at shoulder height and holds ~2.5 s of stillness; from the
  per-side mean torso→wrist vector we fit, in body axes:
  - lateral scale `s_lat` = robot neutral lateral / operator neutral lateral,
  - reach scale `s_fwd` = robot neutral forward / operator neutral forward
    (shared by the vertical axis — both are arm-length bound),
  - an up/forward offset aligning operator-neutral with robot-neutral
    (lateral offset forced to 0 so the midline stays the midline).
  Applied live inside `ClutchMapper._p_abs` (scale/offset on the body-axes
  wrist vector, before the body→base rotation); mapper anchors release so the
  arms GLIDE onto the new correspondence (same snap-free path as re-engage).
  Robot neutral reference per side comes from `mapping.robot_neutral_wrist`
  in `config/rig.yaml` — probed with the real IK on 2026-06-11:
  comfortable max forward ≈ 0.52 m at lateral ±0.22 → neutral set to
  `[∓0.22, +0.02, +0.46]` (≈90% of comfortable max).
  Persisted to `config/operator_calib.json` (gitignored, like
  `viz_calib.json`); auto-loaded on engine start for LIVE transports
  (orbit/vuer) only — fake/replay stay identity so `verify_stack` and
  `analyze_session` remain deterministic.
- **Dashboard CALIBRATE button + guided prompts.** New engine control channel
  (stdlib TCP line-JSON on `vr.control_port` = 8201, localhost) lets the
  dashboard trigger `calibrate` / `calibrate_cancel` / `calibrate_clear` on
  the RUNNING engine. The page shows a prominent calibration banner driven by
  `status.calib` from the render stream (prompt text, per-side tracked ticks,
  hold-still progress), and a persistent `CAL ✓` chip from the new
  `status.calib_applied` field. During capture the arms FREEZE at their
  current pose (fingers keep tracking), then glide on completion.
- **Hand minimum-separation guard.** New pairwise 3D guard
  (`safety/separation.py` + `safety.hand_min_separation` in rig.yaml,
  default 0.12 m): every tick, the two wrist TARGETS (or the parked arm's
  actual wrist for a disengaged side) are kept ≥ d_min apart by symmetric
  projection along their connecting line — clap and the robot hands meet and
  STOP at contact distance instead of interpenetrating. Runs after the
  per-side workspace/anti-cross clamps (order is safe: the push moves each
  hand deeper into its own half-space, never across the midline). Engaged
  sides get pushed; parked sides are treated as obstacles.
- `ArmController` split into `plan()` (mapping + clamps → world target) and
  `commit()` (IK solve) so the engine can coordinate the pair between the
  two; `update()` remains as plan+commit so every existing call-site and test
  is untouched.

**Files.**
- `src/bimanual_teleop/vr/neutral_calib.py` — NEW: state machine
  (wait→hold→done/cancelled, stillness + extended-pose gates, 120 s timeout),
  scale/offset math, JSON persistence with range validation.
- `src/bimanual_teleop/safety/separation.py` — NEW: pure
  `separate_targets()` (symmetric/one-sided push, degenerate fallback axis).
- `src/bimanual_teleop/control_server.py` — NEW: stdlib TCP line-JSON
  command server bound to 127.0.0.1, one command per connection.
- `src/bimanual_teleop/vr/frames.py` — ClutchMapper: `axis_scale` /
  `body_offset` fields, `set_calibration()` (releases anchors → glide),
  `_p_abs` applies them in body axes.
- `src/bimanual_teleop/arms/arm_control.py` — plan/commit split + `wrist_world()`.
- `src/bimanual_teleop/engine.py` — neutral-calibration tick path (arms hold,
  fingers track, status published), thread-safe request/cancel/clear flags,
  calib file load/apply/save, pairwise separation in the normal tick,
  `calib_summary` for the render stream.
- `src/bimanual_teleop/render_sink.py` — `status.calib_applied` (additive
  schema field; Unity's JsonUtility ignores it).
- `src/bimanual_teleop/launch/run_teleop.py` — starts/stops the ControlServer.
- `scripts/dashboard.py` — CALIBRATE/CANCEL button, `clear cal` button,
  calibration banner with progress bar, `CAL ✓` chip, control-port proxy
  actions, ENGINE_PORTS += 8201.
- `config/rig.yaml` — `mapping.robot_neutral_wrist`, `mapping.calib_file`,
  `vr.control_port`, `safety.hand_min_separation`.
- `.gitignore` — `config/operator_calib.json`.
- `tests/test_neutral_calib.py` — NEW: mapper math, separation geometry,
  state-machine lifecycle, persistence round-trip, engine integration
  (freeze during capture, apply on done, clap-separation end-to-end).

**Behavioral notes.**
- Replaying a session recorded WITH calibration applies identity unless the
  rig sets `vr.use_calib: true` — raw VR frames are recorded, so replays
  re-map through whatever calibration is active at replay time.
- The separation guard also binds in `--vr fake` (the synthetic hands pass
  within 10 cm laterally); targets get pushed apart — expected, not a bug.

---

## 2026-06-11 — Branch switch + environment repair (no code changes)

- Re-pointed `~/Developer/orca_core` symlink →
  `~/Developer/Bimanual_Humanoid/orca_core` after the user moved repos off
  the iCloud Desktop (editable install was failing with "Distribution not
  found"). The `auto/teleop-foundation` branch later made orca-core optional.
- Checked out `auto/teleop-foundation` (replacing local `main` per user
  request); the two uncommitted local changes (`link-mode=copy` pyproject
  workaround, FakeVRSource natural-pose fix) are parked in
  `git stash list` → "pre-branch-switch: link-mode=copy + FakeVRSource
  natural pose fix".
- Killed a 4-day-old orphaned `orbit_to_unity.py --debug` (old
  `orca-teleop-unity` project) that was squatting on all seven ORBIT ports
  and blocking the dashboard's engine spawns.

---

## 2026-06-12 (late afternoon) — S-curve motion smoothing: accel-bounded governor + shaper

User report: motion reads "blocky" — targets glide at constant max speed with
instant starts/stops (rectangle velocity profile). Goal: the classic smooth
second-order step response, accepting a little lag/slower following. Also:
verify the speed limiting is per-SECOND (frame-rate independent), not
per-frame, so sim behavior translates to the real frame.

**Root cause.** Two hard clamps with infinite acceleration at the corners:
`_govern`'s `speed·dt` position clamp (orientation had a first-order blend —
smooth decay but instant velocity ONSET), and the joint shaper's stiff
`kp = ω²` slamming velocity to `rate_limit` in ~6 ms before cruising.

**Change.**
- `safety/shaper.py` — optional `accel_limit` (rad/s²): PD accel is clipped
  before integrating, so velocity RAMPS to the cap and back (S-curve onset,
  critically-damped no-overshoot landing). No-overshoot bound documented:
  accel ≥ 0.37·rate·ω; rig defaults carry ≥1.4× margin. Class default None
  (legacy); both call sites pass config values.
- `arms/arm_control.py` `_govern` — rewritten as a critically-damped
  second-order TRACKER for position and attitude (SO(3), world-frame rotvec
  error, angular-velocity state) under norm-clipped velocity AND acceleration
  caps, sub-stepped with real dt (stable across hiccups, identical at any
  loop rate). Teleport rejection unchanged. Legacy `target_ori_smooth_s` τ
  still honored (ω = 2/τ) when `target_ori_smooth_hz` is absent.
- `config/rig.yaml` — new: `target_accel_max: 5.0` m/s²,
  `target_smooth_hz: 2.0`, `target_ang_accel_max: 25.0` rad/s²,
  `sim_accel_limit: 25.0`, `hardware.accel_limit: 12.0`.
- Tests: +5 (shaper accel ramp + frame-rate independence; governor linear +
  angular accel bounds; end-to-end governed-motion rate independence).
  267 pass; `verify_stack.py` green.

**Measured cost/benefit (library tapes, all 6 PASS).** Correspondence during
fast translation: reach_box 1.2→7.2 cm, pick_place →7.0 cm median (the
expected tracker lag — user explicitly accepted); slow/parked tapes stay at
0.2–3.0 cm. Orientation overlay 2.8°→2.65° and IK tracking 4.0°→2.0°
medians IMPROVED on reach_box (smooth demands are easier to solve).

**Follow-ups.** Re-pin the REPLAY_LIBRARY.md score table + re-render the
movies (the table's provenance note anticipated this rework; the render
agent was mid-flight on the OLD dynamics for 5/6 GIFs — fingers.npz picked
up the new config). Feel-test on the headset: if following feels laggy,
raise `target_smooth_hz` (2.0 → 2.5–3.0) and keep
`target_accel_max ≥ 0.37·speed·ω` for the no-overshoot margin.
