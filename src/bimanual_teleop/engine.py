"""The teleop engine: per-side arm + hand controllers.

The arm ORIENTATION mapping is CALIBRATION-FREE in the default body-relative
mode: positions are torso→wrist vectors in body axes, and orientation is the
wrist attitude mapped to the corresponding robot-world axes (see ClutchMapper).
`vr.calib_seconds` therefore defaults to 0 and the arms follow as soon as a hand
is tracked + engaged.

POSITION can additionally be fitted to the operator at runtime via the
NEUTRAL-POSE calibration (vr/neutral_calib.py): triggered from the dashboard
(control_server), the operator extends both arms forward and holds still; the
fitted body-axes scale/offset is applied to both ClutchMappers (arms glide onto
the new correspondence) and persisted. While the capture runs, the ARMS FREEZE
at their current pose and the fingers keep tracking. Orientation is never
calibrated — that contract stands (see CLAUDE.md).

Setting `vr.calib_seconds > 0` enables the optional legacy startup STILLNESS hold
(operator stands arms at sides for N seconds). It measures the operator's body
frame from the head pose, which only steers arm motion when `vr.body_relative`
is explicitly disabled for diagnostics; otherwise it is a stillness/quality gate.

`calib_status` is published every tick for the dashboard banner and the
in-headset visual (vr/vuer_source): the operator can SEE the prompt/progress.

Every tick also runs the pairwise HAND SEPARATION guard (safety/separation.py):
the two wrist targets — or a parked arm's actual wrist — are kept at least
`safety.hand_min_separation` apart, so clashing the operator's hands brings the
robot hands to contact distance, never through each other.

Sink = anything with set_arm(side, q) / set_hand(side, joints_deg): Unity render
now, the hardware drivers later.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .arms.arm_control import ArmController
from .config import REPO_ROOT, SIDES
from .hands.hand_control import HandController
from .safety.separation import separate_capsules
from .vr.calibrate import Calibrator, R_base_from_body, body_relative_hand_sample
from .vr.frames import HandSample, VRFrame
from .vr.neutral_calib import NeutralPoseCalibration, load_calibration


class TeleopEngine:
    def __init__(self, rig: dict, sink):
        self.rig = rig
        self.sink = sink
        self.arm = {s: ArmController(rig, s) for s in SIDES}
        self.hand = {s: HandController(rig, s) for s in SIDES}
        self.calibrator = Calibrator(rig)
        self.calib_s = float(rig.get("vr", {}).get("calib_seconds", 0.0))
        self.calibrated = self.calib_s <= 0
        self.body_relative = bool(rig.get("vr", {}).get("body_relative", True))
        self.torso_from_head = rig.get("vr", {}).get("torso_from_head", [0.0, -0.35, 0.0])
        # Mirror the motion when the operator observes the robot from a mirrored
        # viewpoint: forward = front↔back, lateral = left↔right (independent). Default off.
        _m = rig.get("mapping", {})
        self.mirror_forward = bool(_m.get("mirror_forward", False))
        self.mirror_lateral = bool(_m.get("mirror_lateral", False))
        # Swap which ARM follows which hand: right arm does the LEFT hand's motion and
        # vice versa. Applied once here so EVERY path (live/replay/robot) is consistent.
        self.swap_sides = bool(_m.get("swap_sides", False))
        self._calib_t0 = None
        self._prompted = False
        self._done_t = None
        # Published each tick for the in-headset visual. None once the post-calib
        # banner has faded. See vr/vuer_source.set_calib.
        self.calib_status: dict | None = None
        if not self.calibrated:
            self.calib_status = {"active": True, "phase": "wait", "progress": 0.0,
                                 "remaining": self.calib_s, "left": False, "right": False,
                                 "msg": "DROP YOUR ARMS TO YOUR SIDES"}
        elif self.body_relative:
            self._set_body_relative_mapping()
        # --- operator neutral-pose calibration (position-only, runtime) ----- #
        self.neutral = NeutralPoseCalibration(rig)
        self.calib_summary: dict | None = None     # applied scale/offset, for the dashboard chip
        self._calib_file = self._resolve_calib_path(rig)
        self._req_calib = False                    # set from the control-server thread,
        self._req_calib_cancel = False             # consumed by tick() on the control loop
        self._req_calib_clear = False
        # Auto-load a persisted fit for LIVE transports only: fake/replay stay
        # identity so the acceptance gate and the session scorer are deterministic
        # (override with vr.use_calib: true for calibrated replays).
        transport = rig.get("vr", {}).get("transport", "fake")
        if self._calib_file and (transport in ("orbit", "vuer") or rig.get("vr", {}).get("use_calib")):
            res = load_calibration(self._calib_file)
            if res is not None:
                self._apply_calibration(res, announce=f"loaded {self._calib_file}")
        # Pairwise hand guard: capsule length (wrist → fingertips) + min distance
        # between the two capsules (0 disables).
        self.hand_min_sep = float(rig.get("safety", {}).get("hand_min_separation", 0.12))
        self.hand_capsule_len = float(rig.get("safety", {}).get("hand_capsule_len", 0.19))
        self.cross_gap = float(rig.get("vr", {}).get("cross_gap", 0.05))

    def tick(self, frame: VRFrame | None, engaged: dict[str, bool], t: float) -> None:
        if self.swap_sides and frame is not None and "left" in frame.hands and "right" in frame.hands:
            frame.hands["left"], frame.hands["right"] = frame.hands["right"], frame.hands["left"]
            engaged = {"left": engaged.get("right", False), "right": engaged.get("left", False)}
        self._drain_calib_requests(t)
        if not self.calibrated:
            self._calibration_tick(frame, t)
            return
        if self.neutral.active:
            self._neutral_tick(frame, t)
            return
        # Keep the "CALIBRATED ✓" banner up briefly, then clear it.
        if self.calib_status is not None:
            if self._done_t is not None and (t - self._done_t) > 2.5:
                self.calib_status = None
        plans = {}
        for s in SIDES:
            hs = frame.hands.get(s) if frame else None
            plans[s] = self.arm[s].plan(self._arm_hand_sample(hs, frame), engaged.get(s, False), t)
        self._separate_hands(plans)
        for s in SIDES:
            self.sink.set_arm(s, self.arm[s].commit(plans[s], t))
            self.sink.set_hand(s, self.hand[s].update(frame.hands.get(s) if frame else None, t))

    # ---- pairwise hand separation ----------------------------------------- #
    def _separate_hands(self, plans: dict[str, dict | None]) -> None:
        """Keep the two hand CAPSULES (wrist → wrist + capsule_len·fingers_dir,
        the full ORCA volume from wrist to fingertips) ≥ hand_min_separation
        apart, in 3D. Point-pair guards proved insufficient on a real clap:
        with the wrist points 17 cm apart the fingertips still reached 0.4 cm
        from the other palm. The push shifts the engaged sides' wrist TARGETS
        along the line between the capsules' closest points; a disengaged side
        is a fixed obstacle (its live pose enters the math but never moves)."""
        if self.hand_min_sep <= 0.0:
            return
        mv = {s: plans[s] is not None for s in SIDES}
        if not (mv["left"] or mv["right"]):
            return
        cw = {s: (plans[s]["pw"] if mv[s] else self.arm[s].wrist_world()) for s in SIDES}
        cd = {s: (plans[s]["fingers_dir"] if mv[s] else self.arm[s].fingers_dir_world())
              for s in SIDES}
        # The robot tracks its commands with lag (shaper), so command-vs-command
        # clearance alone lets the lagging ACHIEVED poses pass closer mid-flight
        # (measured 0.9 cm achieved with 12 cm commanded on a real clap). Each
        # command must therefore also clear the other arm's ACHIEVED capsule —
        # never command into space the other hand still occupies.
        aw = {s: self.arm[s].wrist_world() for s in SIDES}
        ad = {s: self.arm[s].fingers_dir_world() for s in SIDES}
        L, d = self.hand_capsule_len, self.hand_min_sep
        for _ in range(2):                       # interleaved constraints → settle
            cw["left"], cw["right"] = separate_capsules(
                cw["left"], cw["right"], cd["left"], cd["right"], L, d,
                move_left=mv["left"], move_right=mv["right"])
            if mv["left"]:
                cw["left"], _ = separate_capsules(cw["left"], aw["right"], cd["left"],
                                                  ad["right"], L, d, move_right=False)
            if mv["right"]:
                _, cw["right"] = separate_capsules(aw["left"], cw["right"], ad["left"],
                                                   cd["right"], L, d, move_left=False)
        # Anti-cross as a PAIR-ORDER constraint: the right wrist stays at least
        # 2·cross_gap to the +Y side OF THE LEFT WRIST. Unlike the old per-side
        # midline half-spaces (which pinned a hand at ±gap and tore off-center
        # claps 24 cm apart — measured), the pair may sit anywhere laterally;
        # only their ORDER and minimum lateral separation are enforced.
        need = 2.0 * self.cross_gap - (cw["right"][1] - cw["left"][1])
        if need > 0.0:
            if mv["left"] and mv["right"]:
                cw["left"][1] -= need / 2.0
                cw["right"][1] += need / 2.0
            elif mv["left"]:
                cw["left"][1] -= need
            elif mv["right"]:
                cw["right"][1] += need
        for s in SIDES:
            if mv[s]:
                plans[s]["pw"] = cw[s]

    # ---- neutral-pose calibration plumbing --------------------------------- #
    @staticmethod
    def _resolve_calib_path(rig: dict) -> Path | None:
        raw = rig.get("mapping", {}).get("calib_file", "config/operator_calib.json")
        if not raw:
            return None
        p = Path(raw)
        return p if p.is_absolute() else REPO_ROOT / p

    def _apply_calibration(self, res, announce: str) -> None:
        for s in SIDES:
            self.arm[s].mapper.set_calibration(res.axis_scale, res.body_offset,
                                               getattr(res, "lat_ref", 0.0))
        self.calib_summary = res.summary()
        print(f"[calib] {announce}: axis_scale={np.round(res.axis_scale, 3).tolist()} "
              f"body_offset={np.round(res.body_offset, 3).tolist()}", flush=True)

    # Request methods are called from the control-server thread: they only set
    # flags (atomic under the GIL); all real work happens in tick().
    def request_calibration(self) -> None:
        self._req_calib = True

    def request_calibration_cancel(self) -> None:
        self._req_calib_cancel = True

    def request_calibration_clear(self) -> None:
        self._req_calib_clear = True

    def _drain_calib_requests(self, t: float) -> None:
        if self._req_calib_clear:
            self._req_calib_clear = False
            if self.neutral.active:
                self.neutral.cancel("calibration cleared")
            for s in SIDES:
                self.arm[s].mapper.set_calibration(np.ones(3), np.zeros(3), 0.0)
            self.calib_summary = None
            if self._calib_file is not None:
                try:
                    self._calib_file.unlink(missing_ok=True)
                except OSError:
                    pass
            self.calib_status = {"active": False, "kind": "neutral", "phase": "cancelled",
                                 "progress": 0.0, "remaining": 0.0, "left": False,
                                 "right": False, "msg": "calibration cleared — back to 1:1"}
            self._done_t = t
        if self._req_calib_cancel:
            self._req_calib_cancel = False
            if self.neutral.active:
                self.neutral.cancel()
                self.calib_status = self.neutral.status(t)
                self._done_t = t
        if self._req_calib:
            self._req_calib = False
            if not self.neutral.active:
                self.neutral.start(t)
                print("[calib] neutral-pose capture started — extend both arms forward "
                      "and hold still", flush=True)

    def _neutral_tick(self, frame: VRFrame | None, t: float) -> None:
        """One capture tick: arms FREEZE at their current pose, fingers keep
        tracking, the state machine eats body-relative wrist samples."""
        samples: dict[str, np.ndarray | None] = {}
        for s in SIDES:
            hs_raw = frame.hands.get(s) if frame else None
            hs = self._arm_hand_sample(hs_raw, frame)
            w = None
            if hs is not None and hs.tracked:
                W = np.asarray(hs.wrist, dtype=float)
                if W.shape == (4, 4) and np.all(np.isfinite(W[:3, 3])):
                    w = W[:3, 3]
            samples[s] = w
            self.sink.set_arm(s, self.arm[s].ik.q)                  # hold current pose
            self.sink.set_hand(s, self.hand[s].update(hs_raw, t))   # fingers can track meanwhile
        self.neutral.tick(samples, t)
        self.calib_status = self.neutral.status(t)
        if self.neutral.phase == "done" and self.neutral.result is not None:
            res = self.neutral.result
            self.neutral.result = None                              # consume once
            self._apply_calibration(res, announce="neutral-pose fit")
            if self._calib_file is not None:
                try:
                    res.save(self._calib_file)
                    print(f"[calib] saved → {self._calib_file}", flush=True)
                except OSError as e:
                    print(f"[calib] save failed: {e}", flush=True)
            self._done_t = t                                        # banner fades in 2.5 s
        elif not self.neutral.active:                               # cancelled / timed out
            self._done_t = t

    def _set_body_relative_mapping(self) -> None:
        for s in SIDES:
            self.arm[s].mapper.set_R(R_base_from_body(self.rig["arms"][s]["base_quat"]))

    def _arm_hand_sample(self, hs: HandSample | None, frame: VRFrame | None) -> HandSample | None:
        if not self.body_relative:
            return hs
        return body_relative_hand_sample(hs, frame.head if frame else None, self.torso_from_head,
                                         mirror_forward=self.mirror_forward,
                                         mirror_lateral=self.mirror_lateral)

    def _calibration_tick(self, frame: VRFrame | None, t: float) -> None:
        """Collect resting-stance samples; hold arms at the rest pose; fingers track."""
        head = frame.head if frame else None
        seen = {s: False for s in SIDES}
        for s in SIDES:
            hs = frame.hands.get(s) if frame else None
            if hs and hs.tracked and hs.landmarks is not None:
                self.calibrator.add(s, hs.landmarks, hs.wrist, head)
                seen[s] = True
            self.sink.set_arm(s, self.arm[s].ik.q0)            # hold rest pose
            self.sink.set_hand(s, self.hand[s].update(hs, t))  # fingers can track meanwhile
        any_tracked = seen["left"] or seen["right"]

        if any_tracked and self._calib_t0 is None:
            self._calib_t0 = t
            if not self._prompted:
                print("[calib] ARMS DOWN at your sides, relaxed, palms rolled INWARD — "
                      f"matching the robot's rest. Hold still {self.calib_s:.0f}s...", flush=True)
                self._prompted = True

        # Publish status for the in-headset visual.
        if self._calib_t0 is None:
            self.calib_status = {"active": True, "phase": "wait", "progress": 0.0,
                                 "remaining": self.calib_s, "left": seen["left"],
                                 "right": seen["right"], "msg": "DROP YOUR ARMS TO YOUR SIDES"}
        else:
            elapsed = t - self._calib_t0
            self.calib_status = {"active": True, "phase": "hold",
                                 "progress": max(0.0, min(1.0, elapsed / self.calib_s)),
                                 "remaining": max(0.0, self.calib_s - elapsed),
                                 "left": seen["left"], "right": seen["right"],
                                 "msg": "HOLD STILL — ARMS AT YOUR SIDES"}

        if self._calib_t0 is not None and (t - self._calib_t0) >= self.calib_s:
            allok = True
            for s in SIDES:
                r = self.calibrator.result(s)
                if r is None:
                    print(f"[calib] {s}: NOT tracked — using default frame.", flush=True)
                    allok = False
                    continue
                self.arm[s].mapper.set_R(R_base_from_body(self.rig["arms"][s]["base_quat"])
                                         if self.body_relative else r["R"])
                allok = allok and r["ok"]
                tag = "OK" if r["ok"] else "SHAKY (hold stiller & recalibrate)"
                print(f"[calib] {s}: {tag} | stillness={r['std']*1000:.0f}mm "
                      f"| forward≈{r['forward'].round(2)} up≈{r['up'].round(2)}", flush=True)
            print("[calib] done — arms now follow your hands. (Restart to recalibrate.)", flush=True)
            self._done_t = t
            self.calib_status = {"active": True, "phase": "done", "progress": 1.0,
                                 "remaining": 0.0, "left": seen["left"], "right": seen["right"],
                                 "msg": "CALIBRATED" if allok else "CALIBRATED (shaky — recheck)"}
            self.calibrated = True
