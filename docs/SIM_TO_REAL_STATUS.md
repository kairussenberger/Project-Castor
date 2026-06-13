# Sim-to-Real: Status & Roadmap

> First metal day is done — the **right arm is mapped, gated, jogged, and replays
> on hardware at real speed**. This document is the working map for taking it the
> rest of the way to operational bimanual teleop on the Ubuntu control host.
> Written 2026-06-12 (continue from here on the Ubuntu machine).

Host: `ethrc-System-Product-Name` (10.5.0.114), Ubuntu 24.04, repo at
`~/Bimanual-Teleop`. Right arm on **can0** (single gs_usb adapter). Dashboard at
**http://&lt;host&gt;:8180** (binds 0.0.0.0 — no ssh tunnel).

---

## 1. What works now (done this session)

Right-arm bring-up on can0 ran the full guided sequence
(`scripts/hw_bringup.py`), in order:

| Step | Result |
|------|--------|
| `--step links` | can0 UP @ 1 Mbit (gs_usb) |
| `--step scan` | motors **1..6 online, no id-7 gripper** — 6-joint chain confirmed |
| `--step rest` | still hang (drift 0.02°); j2 boot wrap (−181°→+178°) auto-normalized |
| `--step signs` | direction signs **[+1, +1, −1, −1, +1, +1]** measured; map written |
| `--step verify` | model-space ±3° per joint through the map — metal matched the dashboard |
| `--step watchdog` | motors went limp on stream stop (deadman validated) |
| keyboard jog | every joint + EE nudges, metal tracked the dashboard 1:1 |
| replay on metal | `bimanual_demo.npz` at **real speed**, analyzer PASS, 0.1 cm correspondence |

Operator verdict: **"mapping is good for the right arm."**

Key infrastructure landed (5 local commits — see §3):
- **`arms/joint_map.py`** — per-side affine motor↔model map (signs measured, offsets
  anchored at the official rest); gitignored per-machine file `config/hw_joint_map.json`.
- **`arms/yam_chain.py`** — drives the DM CAN chain directly (i2rt `MotorChainRobot`
  is bypassed: it rejects this rig's motor zeros at boot, and its gravity comp assumes
  a table mount while these arms hang sideways). Limp enable, multi-turn wrap normalize,
  MIT PD, no feedforward, motor-space clamp.
- **Rest-pose engage gate** — `HardwareSink` refuses to start unless the wired arm
  measures at the official rest through its map (shortest-arc comparison).
- **Replay speed** — `run_hw / run_teleop --speed 0.2` time-stretches a recording;
  `--rate-limit` overrides the shaper cap.
- **Dashboard replay studio** — recordings browser (duration/frames/engaged%),
  `analyze` button (offline contract grader), speed slider, copy-paste `run_hw`
  command, red **STOP ALL** (SIGINTs every teleop/jog/bring-up/replay process).

---

## 2. The per-joint direction (CW/CCW) matching test — KEEP USING THIS

This is the core sim-to-real orientation/direction check and the thing to re-run
on any rig, any re-cabling, any map change. It exists because the motor joint
convention is NOT the model convention — a joint can rotate the *opposite* way
to what the model commands, and that must be caught before any tracking torque.

Two stages, both at ≤10°/s, hand on the e-stop, dashboard visible:

- **`signs` (±2°, the measurement):** each joint is nudged **+2° then −2°** in
  *motor* space while the dashboard previews the *assumed* model-space direction.
  The operator answers **same / opposite** per joint. This writes the sign vector
  into `config/hw_joint_map.json`.
- **`verify` (±3°, the proof):** each joint is wiggled **+3° (CW) then −3° (CCW)**
  in *model* space *through the saved map*. The physical joint and the dashboard
  must now rotate the **same direction** on every joint — that is the orientation
  match. Any disagreement = a wrong sign answer; re-run `signs` for that joint.

**Dashboard preview is exaggerated ×12 on purpose** (±2–3° true-scale is ~2 px,
invisible). The prompt says so. Jog and live teleop render 1:1 — only the
calibration steps exaggerate.

**Pass looks like:** verify shows the same CW/CCW on metal and screen for all 6
joints; the rest-gate reports ~0°/joint; jog gap settles to ~0 after each move.
**Investigate if:** a joint moves opposite the screen (sign error), or the
commanded-vs-measured gap stays large while idle at the hang (not gravity sag).

---

## 3. Repo state & how to ship it (push is currently blocked)

**5 commits are made locally on the host but NOT pushed:**

```
aea751e docs: metal day complete (right arm) — PROGRESS + ARCHITECTURE
4844ca9 dashboard: replay studio — recordings, speed slider, analyze, STOP ALL
409bdf9 replay: time-stretch speed + run_hw dashboard mirror + synthesized episodes
3a00a06 bring-up tooling + jog overhaul: guided first-contact + tracker-correct jog
0d40333 hardware boundary: measured joint map + direct CAN chain + rest-pose gate
```

`verify_stack` passes on the host (all tests + probes + smokes). The tree is clean.

**Why push failed:** the GitHub account on this host (`alerest285`) is **not a
collaborator** on `kairussenberger/Bimanual-Teleop` (403 denied), and the pasted
token was a **fine-grained PAT without write** (can't push even to own repos /
can't fork). To deliver, pick one:

1. **Get collaborator access** on `kairussenberger/Bimanual-Teleop`, then a
   **classic token with `repo` scope**: `echo TOKEN | gh auth login --with-token`,
   then `git push origin main` (fast-forwards Kai's main — these 5 commits are
   exactly his `f9da703` + 5).
2. **Push to your own repo + Kai pulls:** classic `repo` token, then
   `git push https://github.com/alerest285/<repo>.git main:metal-day-rightarm`,
   and Kai runs `git fetch <that> metal-day-rightarm && git merge --ff-only …`.
3. **Working locally only:** the commits live in the host's git — keep working on
   this machine; push whenever access is sorted.

---

## 4. Missing — MONITORING (live, during hardware runs)

The dashboard currently mirrors *commanded* state. For real hardware operation it
must also show what the metal is actually doing.

- [ ] **Commanded-vs-measured tracking error on the dashboard.** `HardwareSink`
      should read encoders each tick and publish measured `q` alongside commanded
      `q` in `render.state`; the dashboard draws per-joint gap (the jog already
      computes this in text — promote it into `render_sink` + the arm cards).
- [ ] **Motor telemetry.** `yam_chain.read()` already returns effort and
      `temp_mos` per joint — surface temperature + current on the dashboard, with
      a warning threshold (thermal was an explicit known-unknown).
- [ ] **Bus / loop health.** CAN error counters, dropped-frame count, actual
      control-loop Hz on hardware (not just render Hz), staleness of the last
      motor read.
- [ ] **Torque / e-stop state indicator.** Is torque enabled? Is the arm in the
      rest-gate, engaged, or e-stopped? A clear banner.
- [ ] **End-to-end latency monitor.** VR sample → command → measured-move latency,
      to catch CAN saturation under both arms + hands.
- [ ] **Alerting.** Visual + log alert when gap exceeds a limit, temp high, or
      loop rate drops.

## 5. Missing — TESTING (sim-to-real validation)

- [ ] **Left arm bring-up.** Everything in §1/§2 for `left` on can1 once its
      adapter is wired: scan, signs, verify, watchdog, jog. Set `hardware.sides:
      [right, left]`. The map file already supports per-side entries.
- [ ] **ORCA hands on metal.** `hardware.use_hands` is `false` (RealHand MOVES the
      hand to neutral at connect). Bring up each hand's serial bus, per-joint range
      test, finger-retarget sanity on metal, then enable in `run_hw`.
- [ ] **Repeatability test.** Command a pose, return to rest, measure return error
      over N cycles per joint — a number that should stay small.
- [ ] **Workspace + anti-cross on metal.** Verify the clamps and the world-Y
      anti-cross guard physically (currently only proven in sim/analyzer).
- [ ] **Automated replay suite.** A battery of recorded episodes run through the
      analyzer + on-metal tracking-error capture, producing a pass/fail report
      (extend `analyze_session.py` to fold in measured-vs-commanded from a hw run).
- [ ] **Calibration history.** Log each `signs`/`verify` result + measured rest
      with a timestamp so drift between sessions is visible.
- [ ] **Jitter/throughput under full load.** Both arms + both hands streaming —
      confirm the single-process loop holds rate (it likely won't; see §6).

## 6. Missing — DEPLOYMENT (bring-up → operational)

- [ ] **Per-arm CAN loop process.** The current `HardwareSink`/`run_hw` is the
      synchronous single-process *bring-up* form (its docstring says so). Production
      wants a dedicated ~250 Hz CAN loop per arm in its own process (SCHED_FIFO),
      decoupled from vision/IK via latest-value buffers. This is the big one for
      smooth, safe high-rate control.
- [ ] **Live Quest teleop on the host.** Metal currently does *replay* only. Live
      needs `adb` installed (NOT present on the host), the ORBIT app on the Quest,
      and the `--vr orbit` path (`vr/orbit_source.py`) exercised on metal. This is
      the next headline milestone after monitoring lands.
- [ ] **Startup/shutdown automation.** A launcher (or systemd units) that brings up
      can0/can1 at the right bitrate, runs the rest-gate check, then starts the
      dashboard + engine; clean torque-release on shutdown.
- [ ] **Rate ramp procedure.** Raising `hardware.max_vel_scale` / `rate_limit`
      from the conservative bring-up values is documented but manual — script a
      stepped, logged ramp with an abort.
- [ ] **Physical e-stop integration.** A monitored hardware e-stop in the loop, and
      a software deadman on the dashboard (STOP ALL is the start, but it is not an
      e-stop substitute).
- [ ] **Recording retention / naming** as sessions accumulate.

## 7. Immediate next actions (suggested order)

1. **Monitoring first** — publish measured `q` + temp from `HardwareSink` into
   `render.state` and draw the gap/temp on the dashboard. You want eyes on the
   metal before driving it more.
2. **Left arm bring-up** (when wired) — repeat §2 on can1.
3. **ORCA hands** — bring up + enable `use_hands`.
4. **Per-arm CAN loop process** — the rate/safety upgrade for real operation.
5. **Live Quest on metal** — install adb, run `--vr orbit`.

## Operational reminders

- After any torque release the limp wrist sags a few degrees — re-run
  `--step rest` (re-anchors offsets) at the start of each session.
- Away from the hang, expect a few degrees of gravity sag in the gap (PD without
  feedforward) — that is normal, not a fault.
- The joint map is **per-machine** (`config/hw_joint_map.json`, gitignored) —
  re-measure with `hw_bringup` on any different rig.
- Dashboard tool: `scripts/dashboard.py --host 0.0.0.0`; preview-replay launches
  from the page (render only); metal stays a terminal `run_hw` command with a hand
  on the e-stop. **STOP ALL** kills every teleop/jog/bring-up/replay process.
- Loose end: `render_session.py --gif` errors on `bimanual_demo.npz` (the demo GIF
  render) — cosmetic, unrelated to the runtime.
