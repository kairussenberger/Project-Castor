# Mac Development Plan — working from home (week of 2026-06-22)

You'll develop on a **MacBook Air M3** for ~a week, get the whole thing working
**in simulation** (including calibrating with the Quest and the Quest app itself),
then bring it to the **lab Linux host** to test on the **real robot**.

This is the workflow the project was built for: the control stack runs natively on
Apple Silicon (~105 Hz headless), the **Mac never talks to motors**, and the Linux
host runs the *same* engine with the hardware sink. See `docs/ARCHITECTURE.md`.

---

## The machine split

| Machine | Role | Runs |
|---|---|---|
| **MacBook Air M3** (home) | Development + **simulation** | Python control stack (sim), dashboard, Quest ingest, Unity (build the Quest app) |
| **Lab Linux host** (later) | **Real robot** | Same control stack with the hardware sink (CAN motors), the Quest for running, the robot |

The Mac produces (a) a working, calibrated sim and (b) a built Quest APK. The lab
box is only for the real-motor test at the end.

## Repos & branches (both already cloned)

| Repo | What | Branch to use |
|---|---|---|
| **Project-Castor** (was `Bimanual-Teleop`) | control runtime, sim, dashboard | **`hans-dev1`** |
| **castor-quest-app** | Quest VR Unity app (ORBIT) | **`feature/quest-calibration`** |

> The control repo was renamed `Bimanual-Teleop → Project-Castor` (old URL
> redirects). Canonical: `git@github.com:kairussenberger/Project-Castor.git`.

---

## The week's goal (all achievable from home, no robot)

1. **Sim stack runs on the Mac** — `verify_stack` green, `run_teleop --vr fake`
   drives the robot model in the dashboard.
2. **Calibrate in simulation with the Quest** — put the headset on, run the
   3-pose calibration, drive the sim robot 1:1.
3. **Quest app works seamlessly** — build/iterate the app: passthrough first,
   then the in-headset robot view, then in-headset calibration (prompt + progress
   bar). Plan: `castor-quest-app/docs/castor-quest-changes.md`.

Then, at the lab: same calibration + app, now on the **real arms**.

---

## Part A — Control / sim stack on the Mac

### Dependencies — install these first

**System tools (Homebrew):**
```sh
brew install uv ffmpeg android-platform-tools
```
- **`uv`** — the **only** package manager this repo uses (**not conda, not pip
  directly**). It creates the `.venv`, installs Python **3.12** itself if missing,
  and resolves everything from the committed **`uv.lock`** (reproducible,
  cross-platform — same versions on Mac and the lab Linux box).
- **`ffmpeg`** — only for `scripts/headset_view.py` (streams the dashboard into
  the Quest's video panel; on macOS it uses VideoToolbox natively). Recommended.
- **`android-platform-tools`** — `adb`, for the Quest (pose ingest + installing
  the APK).

**Python environment (one command installs all Python deps):**
```sh
git clone git@github.com:kairussenberger/Project-Castor.git
cd Project-Castor && git checkout hans-dev1
uv sync --extra telemetry          # core runtime + Rerun 3D viewer
```
`uv sync` installs everything from `pyproject.toml` / `uv.lock`:
- **core:** `numpy`, `pyzmq`, `msgpack`, `loop-rate-limiters`, `pyyaml`,
  **`pin`** (Pinocchio) + **`pin-pink`** (the IK solver), **`daqp`** (QP solver),
  `typing-extensions`
- **`--extra telemetry`:** `rerun-sdk`, `fast-simplification` (the 3D viewer +
  mesh decimation — recommended)
- **`--extra vr`** *(optional)*: `vuer` — only if you use the browser/WebXR ingest
  instead of the native ORBIT app (you won't, for normal use)
- **dev group:** `pytest` (pulled in automatically by `uv run`)

**⚠️ Pinocchio / conda — you do NOT need conda.** Pinocchio is the historic
"conda-only" library, but here it ships as the **pip wheels `pin` / `pin-pink`**
(cmeel), which have native **Apple-Silicon** wheels — `uv sync` just works on the
M3, no conda. *Only* if those wheels ever fail would you fall back to
`conda install -c conda-forge pinocchio` — and then keep the **entire** env in
conda; never mix conda + uv. Default: **stick with uv.**

**Not installed on the Mac:** the hardware drivers `i2rt` (YAM arms) and
`orca_core` (hands) — Linux-only, only on the lab host. Sim needs neither.

**Optional GIF renders:** `uv run --with matplotlib python scripts/render_session.py ...`

### Verify it
```sh
uv run python scripts/verify_stack.py   # must pass: tests + probes + smokes
```

### See it move (no headset)
```sh
# terminal 1 — synthetic operator
uv run python -m bimanual_teleop.launch.run_teleop --vr fake
# terminal 2 — dashboard
uv run python scripts/dashboard.py        # http://127.0.0.1:8180
```

### Calibrate in simulation with the Quest (the key milestone)
1. Install the ORBIT app on the Quest (your build, see Part B), connect USB-C,
   `adb devices` lists it.
2. ```sh
   uv run python -m bimanual_teleop.launch.run_teleop --vr orbit --clutch gesture
   uv run python scripts/dashboard.py
   ```
3. Headset on, controllers **down**. Press **⊕ CALIBRATE** on the dashboard, do
   the 3 poses (sides → palms together → reach forward). Pinch to engage and
   drive the **sim** robot. **No motors move** — this is `run_teleop` (render
   sink). Grade the result: `uv run python scripts/analyze_session.py <rec>.npz`.

> The Mac is the native platform for the dashboard-into-headset stream too:
> `scripts/headset_view.py` uses macOS VideoToolbox directly (the Linux x11grab +
> Main-profile port is for the lab box; on the Mac it just works).

---

## Part B — Quest app in Unity on the Mac

### Dependencies — install these first
```sh
git clone git@github.com:kairussenberger/castor-quest-app.git
cd castor-quest-app && git checkout feature/quest-calibration
```
- **Unity Hub** — download from unity.com/download.
- **Unity account** (free) — sign in once in Hub to get the **Personal** license
  (easy on the Mac GUI; none of the headless activation pain).
- **Unity Editor `6000.3.9f1`** — the **exact** version (the project pins it; Hub
  offers it when you open the project). Native Apple Silicon.
- **Android Build Support** module — and tick its **sub-modules**:
  - **Android SDK & NDK Tools**
  - **OpenJDK**
- **`adb`** — bundled with the Android SDK Unity installs, or reuse
  `android-platform-tools` from Part A.
- Unity **packages auto-resolve** from `Packages/manifest.json` on first open
  (OpenXR 1.16.1, XR Hands, XR Interaction Toolkit, XR Management, vendored
  NetMQ) — nothing to install by hand.
- For **passthrough** you'll add **one** package in the Editor's Package Manager:
  **`com.unity.xr.meta-openxr`** (it pulls in **AR Foundation**) — steps in
  `docs/castor-quest-changes.md`.

Then: Hub → **Add** → open `vr_quest_code/VR_Unity_Project`. (Skip the Linux fix
script — that's only for building on Linux.)

### Build → install → test loop
1. File → Build Settings → **Android** → **Build** (produces `.apk`), or **Build
   And Run** with the Quest plugged into the Mac.
2. Install: `adb install -r app.apk` then
   `adb shell monkey -p com.ORBIT.Teleoperation -c android.intent.category.LAUNCHER 1`.
3. Set Player Settings → Package Name = **`com.ORBIT.Teleoperation`** so the build
   replaces the current app (and the control repo's launcher keeps working).

### Implementation order (do passthrough first)
Follow `castor-quest-app/docs/castor-quest-changes.md`:
1. **Passthrough** — add `com.unity.xr.meta-openxr` (Package Manager), enable the
   OpenXR **Meta Quest → Passthrough** feature, wire AR Session + AR Camera
   Background, camera clear alpha = 0. Manifest feature already declared.
2. **Robot view in-headset** — feed the app's video panel (`:10505`) a robot
   render from the control host (reuse `headset_view.py`'s path).
3. **In-headset calibration** — world-space canvas with prompt + progress bar,
   driven by the control host's `calib_status` (via the existing GraphStream).

---

## Part C — Bring it to the lab (real robot)

On the lab Linux host (`Project-Castor`, `hans-dev1`):
1. `git pull` your week's work; `adb install -r` the tested APK to the Quest here.
2. CAN up (`can0`/`can1` @ 1 Mbit), `verify_stack.py` on the host.
3. Follow the **sim→real checklist** in `docs/ARCHITECTURE.md`: `jog_arms --sink hw`
   first (e-stop in hand) → replay a known tape → live `run_hw --vr orbit`.
   Nothing new should happen — same motion you saw in sim, slower (derated).

---

## Git workflow (Mac ↔ lab)
- Push from the Mac, pull on the lab box. You're **owner** (kairussenberger) of
  both repos → full push from any machine signed in as you.
- Set up git auth on the Mac once: add `~/.ssh/id_ed25519.pub` to your GitHub SSH
  keys, or `gh auth login`.
- Keep app work on `feature/quest-calibration`, control work on `hans-dev1`; open
  PRs into `main` when a piece is solid.

## Gotchas
- **Unity version must be exactly `6000.3.9f1`** (the project pins it). Accept
  Hub's offer to install it when opening the project.
- **Quest hand tracking needs the controllers asleep** — set them down.
- **Live sessions require a fresh calibration each run** (ORBIT re-centers its
  origin on restart). Expected, not a bug.
- The **real arms only move via `run_hw`** on the lab box — `run_teleop` (Mac) is
  always sim, regardless of what's plugged in.
- Record every Quest session (`--record recordings/<name>.npz`) so debugging is
  headset-free afterward.
