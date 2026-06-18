"""Per-arm controller: turns a tracked wrist pose into YAM joint targets.

Transport-agnostic — the sim loop and the (future) ZMQ arm process both just call
`update(hand_sample, engaged, t)`. Wraps ArmIK + ClutchMapper + One-Euro target
smoothing + a workspace bounding box. Holds the last pose when not engaged.
"""
from __future__ import annotations

import numpy as np

from ..config import side_axis
from ..filters import OneEuroFilter
from ..safety.shaper import JointCommandShaper
from ..vr.calibrate import R_base_from_body
from ..vr.frames import (SE3, SO3, ClutchMapper, HandSample, mat_to_se3,
                         quat_from_axis_angle, quat_to_R, r_base_from_vr, rotvec,
                         swing_twist_angle)
from .ik import ArmIK


class ArmController:
    def __init__(self, rig: dict, side: str):
        self.rig = rig
        self.side = side
        self.ik = ArmIK(rig, side)
        m = rig["mapping"]
        body_relative = bool(rig.get("vr", {}).get("body_relative", True))
        # The mapper R must match the frame the ctrl samples actually live in:
        # body axes (default) or legacy raw WebXR room coords. Built here, not
        # patched in later by the engine, because the absolute position mapping
        # depends on it from the very first engage.
        if body_relative:
            R = R_base_from_body(rig["arms"][side]["base_quat"])
        else:
            R = r_base_from_vr(rig["arms"][side]["base_quat"], m["r_base_from_vr_euler"][side])
        self.base_R = quat_to_R(rig["arms"][side]["base_quat"])   # base → world
        self.base_pos = np.asarray(rig["arms"][side]["base_pos"], dtype=float)
        # Absolute position mapping: operator torso→wrist lands 1:1 on robot
        # chest→wrist. The chest anchor defaults to the midpoint of the two arm
        # bases; absolute only makes sense for body-relative ctrl samples, so the
        # legacy raw-room mode falls back to clutch deltas.
        mode = str(m.get("position_mode", "absolute"))
        if not body_relative:
            mode = "relative"
        anchor_w = m.get("body_anchor_world")
        if anchor_w is None:
            # The base plates are the SHOULDER line; the operator's torso proxy is
            # sternum-ish, below their shoulders — and the YAM workspace, like a
            # human arm's, lives below the shoulder mounts. Drop the chest anchor
            # accordingly so torso-height hands land in reachable space.
            drop = float(m.get("body_anchor_drop", 0.15))
            anchor_w = 0.5 * (np.asarray(rig["arms"]["left"]["base_pos"], dtype=float)
                              + np.asarray(rig["arms"]["right"]["base_pos"], dtype=float)) \
                - np.array([0.0, 0.0, drop])
        chest_base = self.base_R.T @ (np.asarray(anchor_w, dtype=float) - self.base_pos)
        # Intrinsic wrist twist: pronation about YOUR forearm axis becomes a pure
        # j6 roll about the EE's own tool axis (never a j4/j5 swing through the
        # wrist singularity). Forced to 'world' in legacy raw-room mode.
        twist_mode = str(m.get("twist_mode", "intrinsic"))
        ori_mode = str(m.get("orientation_mode", "absolute"))
        if not body_relative:
            twist_mode = "world"
            ori_mode = "relative"
        # EE-local hand basis from the frozen rest contract: [lat|palm-back|fingers]
        # columns. Used for the hand↔EE orientation convention AND as the axis of
        # the hand CAPSULE the pairwise separation guard protects (the ORCA hand
        # volume runs from the wrist toward the fingertips along this axis).
        from ..viz.hand_geom import hand_basis_for_side
        B = hand_basis_for_side(self.ik, self.base_R, side)            # EE-local
        self.fingers_ee = B[:, 2] / (np.linalg.norm(B[:, 2]) + 1e-12)
        C = None
        if ori_mode == "absolute":
            # Hand↔EE convention, derived (no calibration): the EE basis comes from
            # the frozen rest contract (fingers along approach, palm inward); the
            # hand basis from measured hand-local finger/palm-back axes.
            f = np.asarray(side_axis(m, "hand_finger_axis", side,
                                     [0.345, -0.363, -0.866]), dtype=float)
            f = f / np.linalg.norm(f)
            p = np.asarray(side_axis(m, "hand_palm_axis", side,
                                     [-0.496, 0.713, -0.496]), dtype=float)
            p = p - (p @ f) * f
            p = p / np.linalg.norm(p)
            H = np.column_stack([np.cross(p, f), p, f])                # hand-local [lat|palm-back|fingers]
            C = H @ B.T                                                # EE-local → hand-local
        self.mapper = ClutchMapper(R, pos_scale=m["pos_scale"], position_mode=mode,
                                   chest_base=chest_base if mode == "absolute" else None,
                                   engage_blend_s=float(m.get("engage_blend_s", 1.0)),
                                   twist_mode=twist_mode,
                                   hand_twist_axis=side_axis(m, "hand_twist_axis", side,
                                                             [0.0, 0.456, 0.890])
                                   if twist_mode == "intrinsic" else None,
                                   ee_tool_axis=self.ik.ee_tool_axis_local
                                   if twist_mode == "intrinsic" else None,
                                   orientation_mode=ori_mode, hand_ee_convention=C)
        # Anti-cross is now a PAIR constraint enforced by the engine alongside the
        # capsule separation (right hand stays ≥ 2·cross_gap right OF THE LEFT
        # HAND — not of the body midline, so off-center claps work anywhere).
        ws = rig["safety"]["workspace"]
        self.ws_min = np.asarray(ws["min"], dtype=float)
        self.ws_max = np.asarray(ws["max"], dtype=float)
        self.iters = int(rig["ik"].get("iters", 1))
        # lighter smoothing than the fingers -> less arm lag (snappier baseline,
        # beta cuts lag on fast motion). Tune if jittery.
        self._filt_params = dict(mincutoff=4.0, beta=1.0)
        self.pos_filt = OneEuroFilter(**self._filt_params)
        self._was_engaged = False
        self.cmd_R = None   # last commanded EE orientation (base frame) — for the on-screen viz
        self.cmd_pos = None  # last commanded EE position (base frame) after workspace/cross clamps
        # Live calibration-health signal: how far the raw mapped target sat
        # OUTSIDE the workspace box this tick (0 inside). Sustained large values
        # mean the mapping is off — a stale/broken calibration pins targets at
        # the box face (the 2026-06-11 failure mode), it never just "feels far".
        self.clamp_dist = 0.0
        # --- motion guardrails (safety section) ------------------------------ #
        # Target governor: a critically-damped second-order TRACKER of the
        # commanded target (position AND attitude), with world-frame velocity
        # and ACCELERATION caps, plus teleport rejection. The old hard
        # speed·dt clamp produced rectangle-velocity glides — instant jump to
        # max speed, instant stop: "blocky". Bounding acceleration too makes
        # every demand an S-curve (the classic smooth step response): velocity
        # ramps in, cruises at the cap, and the critically-damped PD lands it
        # without overshoot. All caps are per SECOND (integrated with real dt
        # in stable sub-steps), so the governed motion is identical at any
        # frame rate — what the sim shows is what the hardware does.
        # Real operator motion peaks ≈2 m/s; tracking glitches measured
        # 58–65 m/s — anything implying more than target_jump_speed is NOT a
        # movement: the mapper re-anchors and the arm GLIDES instead (the
        # violent motion simply does not happen).
        s = rig.get("safety", {})
        self.speed_max = float(s.get("target_speed_max", 0.8))        # m/s
        self.accel_max = float(s.get("target_accel_max", 5.0))        # m/s²
        self.jump_speed = float(s.get("target_jump_speed", 3.0))      # m/s → reject
        self.ang_speed_max = float(s.get("target_ang_speed_max", 2.5))  # rad/s
        self.ang_accel_max = float(s.get("target_ang_accel_max", 25.0))  # rad/s²
        # Tracker bandwidths (critically damped: kp = ω², kd = 2ω). Orientation
        # keeps backward compat with the legacy first-order τ when no Hz key is
        # set: ω = 2/τ roughly matches the old settle time.
        hz_p = float(s.get("target_smooth_hz", 2.0))
        tau = float(s.get("target_ori_smooth_s", 0.12))
        hz_o = float(s.get("target_ori_smooth_hz",
                           1.0 / (np.pi * tau) if tau > 0 else 2.5))
        self._wp = 2.0 * np.pi * hz_p
        self._wo = 2.0 * np.pi * hz_o
        # Forward-Euler stability needs dt < 2/ω; integrate well under it.
        self._gov_max_dt = min(0.05, 0.5 / max(self._wp, self._wo))
        self._gov: dict | None = None       # tracker state {p, v, R, w, t}
        self._prev_wrist: tuple | None = None   # previous raw wrist sample (p, t) for the jump test
        # Rest references for the stateless roll saturation (_saturate_roll):
        # the ik is freshly constructed AT the rest pose right above. The roll is
        # measured/removed about the OPERATOR'S FOREARM axis mapped into the EE
        # frame (C.T @ hand_twist_axis) — NOT the j6 tool axis: the two differ by
        # ~10°, and removing 150° of unrealized roll about the wrong axis injects
        # sin(10°)·150° ≈ 0.45 rad of artificial swing into the fed demand (the
        # measured j4 pivot). About the true roll axis, what remains after
        # removal is exactly the operator's REAL swing.
        self._R_rest_w = self.base_R @ self.ik.fk_ee().rotation().as_matrix()
        self._q6_rest = float(self.ik.q[5])
        a_rm = np.asarray(self.ik.ee_tool_axis_local, dtype=float)
        self._roll_rm_ee = a_rm / (np.linalg.norm(a_rm) + 1e-12)
        self._a_roll_w = self._R_rest_w @ self._roll_rm_ee     # same axis, world, at rest
        # Output shaper: the same limit-clamp + rate-cap + critically-damped
        # tracker the hardware boundary uses, now ALSO in the sim/render path —
        # the published joint command can never move faster than sim_rate_limit,
        # whatever the solver does (singularity flips become bounded glides, and
        # sim motion matches what the real robot will be allowed to do).
        self.shaper = JointCommandShaper(
            self.ik.q,
            rate_limit=float(s.get("sim_rate_limit", 1.8)),
            smooth_hz=float(s.get("sim_smooth_hz", 4.0)),
            accel_limit=float(s.get("sim_accel_limit", 25.0)),
            lo=self.ik.hard_lo, hi=self.ik.hard_hi)

    def wrist_world(self) -> np.ndarray:
        """Current wrist-site position in WORLD (the point the position mapping
        drives) — what the pair separation guard uses for a parked arm."""
        return self.base_R @ self.ik.fk_wrist().translation() + self.base_pos

    def fingers_dir_world(self, R_base: np.ndarray | None = None) -> np.ndarray:
        """WORLD-frame UNIT vector from the wrist toward the fingertips, given
        the EE orientation `R_base` (base frame; None = current FK). The
        separation guard protects the capsule wrist → wrist + len·this."""
        if R_base is None:
            R_base = self.ik.fk_ee().rotation().as_matrix()
        return self.base_R @ (np.asarray(R_base, dtype=float) @ self.fingers_ee)

    def _saturate_roll(self, R_w: np.ndarray) -> np.ndarray:
        """Clamp the commanded attitude's ROLL to what j6 can actually reach,
        BEFORE the IK sees it — measured STATELESSLY against the REST attitude.

        Why: the wrapped twist of the attitude error vs the CURRENT pose flips
        sign at ±π. The side with only ~30° of roll headroom (j6 rest ∓90°,
        soft range asymmetric) pins early; as the operator keeps rolling, the
        error grows past π and the solver suddenly believes the short way is
        the OTHER direction — j6 walks its whole range through the wrist
        singularity while j4/j5 wander (the measured left-arm pivot; the right
        arm has 210° of headroom in the same physical direction and never
        reaches the wrap). Measured FROM REST instead, the roll demand equals
        the operator's physical roll from the calibration neutral — a human
        wrist cannot roll ±180° from neutral, so the wrapped angle is always
        unambiguous, with no integrator to drift and no state to reset. The
        unreachable remainder is rotated out on the TWIST side of the
        decomposition (EE-local axis, right-multiplied), leaving the swing
        component untouched; both arms then degrade identically at their
        stops: pinned, clean, no long-way travel."""
        phi = swing_twist_angle(R_w @ self._R_rest_w.T, self._a_roll_w)
        dem = self._q6_rest + phi                      # j6 needed for this roll
        # Slack: small overspill beyond the soft window stays untouched — the
        # IK's own twist clamp handles it gracefully (j6 pins, remainder
        # excluded from the swing) and the wears-the-attitude contract holds
        # for it. Anything beyond clips, for two measured reasons: (1) the
        # twist error vs a pinned j6 must never approach ±π (the wrap flips
        # its sign and walks j6 the LONG way through the wrist singularity);
        # (2) the hand↔EE convention axis is ~10° off the operator's true
        # forearm axis, so every radian of unrealized roll left in the demand
        # leaks ~sin(10°) of GENUINE swing into j4/j5 — 150° of unrealized
        # roll measured ~0.5 rad of pivot-to-the-side on the 30°-headroom arm,
        # feeding back to the joint stops. 0.35 rad of remainder caps the leak
        # at a harmless ~0.06 rad.
        slack = 0.35
        lo = self.ik.soft_lo[5] - slack
        hi = self.ik.soft_hi[5] + slack
        remainder = dem - float(np.clip(dem, lo, hi))
        if abs(remainder) < 1e-9:
            return R_w
        # R_w·Rot(a_ee, −rem) with the constant EE-local roll axis ≡
        # R_err·Rot(a_roll_w, −rem)·R_rest — removal on the twist side.
        return R_w @ quat_to_R(quat_from_axis_angle(self._roll_rm_ee, -remainder))

    def _govern(self, pw: np.ndarray, R_t: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Track the would-be target (world frame) with a critically-damped
        second-order tracker under velocity AND acceleration caps — position
        and attitude alike. A step demand leaves as a smooth S-curve: velocity
        ramps in at ≤ accel_max, cruises at ≤ speed_max, and the PD lands it
        without overshoot. High-frequency tracking jitter (the raw wrist quat
        carries ~200°/s of it) dies in the tracker's −40 dB/decade roll-off.
        The caller's plan() returns None separately when the motion is a
        TELEPORT (implied raw-wrist speed > jump_speed): the mapper re-anchors
        and the movement simply does not happen."""
        if self._gov is None:
            self._gov = {"p": pw.copy(), "v": np.zeros(3),
                         "R": R_t.copy(), "w": np.zeros(3), "t": float(t)}
            return pw, R_t
        g = self._gov
        remaining = max(0.0, min(float(t) - g["t"], 0.5))   # cap runaway gaps
        g["t"] = float(t)
        while remaining > 1e-9:
            dt = min(remaining, self._gov_max_dt)
            remaining -= dt
            # position: PD accel toward the demand, norm-clipped (direction-
            # preserving) to accel_max, velocity to speed_max
            a = self._wp * self._wp * (pw - g["p"]) - 2.0 * self._wp * g["v"]
            n = float(np.linalg.norm(a))
            if n > self.accel_max:
                a *= self.accel_max / n
            v = g["v"] + a * dt
            n = float(np.linalg.norm(v))
            if n > self.speed_max:
                v *= self.speed_max / n
            g["v"] = v
            g["p"] = g["p"] + v * dt
            # attitude: the same tracker on SO(3) — angular velocity state w
            # (world axes), PD accel on the world-frame rotation-vector error
            e = g["R"] @ rotvec(g["R"].T @ R_t)
            aw = self._wo * self._wo * e - 2.0 * self._wo * g["w"]
            n = float(np.linalg.norm(aw))
            if n > self.ang_accel_max:
                aw *= self.ang_accel_max / n
            w = g["w"] + aw * dt
            n = float(np.linalg.norm(w))
            if n > self.ang_speed_max:
                w *= self.ang_speed_max / n
            g["w"] = w
            ang = float(np.linalg.norm(w)) * dt
            if ang > 1e-12:
                g["R"] = quat_to_R(quat_from_axis_angle(w / np.linalg.norm(w), ang)) @ g["R"]
        # many tiny rotation products accumulate float drift — re-orthonormalize
        u, _, vt = np.linalg.svd(g["R"])
        g["R"] = u @ vt
        return g["p"].copy(), g["R"].copy()

    def plan(self, hand: HandSample | None, engaged: bool, t: float) -> dict | None:
        """Mapping half of a tick: wrist pose → clamped WORLD target. Returns
        None when idle (not engaged/tracked). The engine may adjust the
        returned `pw` (pairwise hand separation) before commit() solves it —
        that is the whole reason plan and solve are separate steps."""
        active = bool(engaged and hand is not None and hand.tracked)
        # (Re)anchor on the clutch rising edge OR whenever the mapper has dropped its
        # anchor while still active — set_R()/set_calibration() call release() (e.g.
        # after a retune or a neutral-pose calibration), and without re-engaging here
        # the next target() would assert on a null anchor. engage() makes the target
        # equal the current EE pose, so re-anchoring is continuous (only the
        # displacement-since-clutch resets to zero, which is the desired behaviour
        # after a retune).
        if active and (not self._was_engaged or not self.mapper.engaged):
            # anchor POSITION to the wrist site, ORIENTATION to the hand
            anchor = SE3.from_rotation_and_translation(
                self.ik.fk_ee().rotation(), self.ik.fk_wrist().translation())
            self.mapper.engage(mat_to_se3(hand.wrist), anchor, t)
            self.pos_filt = OneEuroFilter(**self._filt_params)   # reset smoothing on engage
            self.ik.reset_twist()    # glide makes the demand continuous from here
        if not active and self._was_engaged:                 # release
            self.mapper.release()
            self._gov = None
            self._prev_wrist = None
        self._was_engaged = active
        if not active:
            self.clamp_dist = 0.0
            return None
        # Teleport rejection on the OPERATOR signal (body-frame wrist sample,
        # real metres): tracking glitches measured 58–65 m/s vs ≤2 m/s for real
        # motion. Testing the raw wrist — not the mapped target — keeps the
        # engage GLIDE (which legitimately demands ~1.5 m/s of target motion)
        # from ever reading as a jump. On rejection the movement simply does
        # not happen: re-anchor and glide from the current pose instead.
        w_now = np.asarray(hand.wrist, dtype=float)[:3, 3]
        if self._prev_wrist is not None:
            pdt = max(float(t) - self._prev_wrist[1], 1e-6)
            if float(np.linalg.norm(w_now - self._prev_wrist[0])) / pdt > self.jump_speed:
                self.mapper.release()
                self._gov = None
                self._prev_wrist = (w_now.copy(), float(t))
                return None
        self._prev_wrist = (w_now.copy(), float(t))
        target = self.mapper.target(mat_to_se3(hand.wrist), t)
        p = np.clip(target.translation(), self.ws_min, self.ws_max)  # workspace box
        self.clamp_dist = float(np.linalg.norm(target.translation() - p))
        sm = self.pos_filt({"x": p[0], "y": p[1], "z": p[2]}, t)     # One-Euro
        pb = np.array([sm["x"], sm["y"], sm["z"]])                   # target in base frame
        pw = self.base_R @ pb + self.base_pos                        # → world
        R_w = self.base_R @ target.rotation().as_matrix()            # attitude, world frame
        governed = self._govern(pw, R_w, t)
        if governed is None:                                         # teleport rejected
            return None
        pw, R_w = governed
        R_w = self._saturate_roll(R_w)         # j6-reachable roll only — see method
        R_b = self.base_R.T @ R_w                                    # back to base frame
        return {"pw": pw, "R": SO3.from_matrix(R_b),
                "fingers_dir": self.fingers_dir_world(R_b)}

    def commit(self, plan: dict | None, t: float) -> np.ndarray:
        """IK + shaper half of a tick: solve toward the (possibly pair-adjusted)
        world target from plan(), then pass the solution through the joint-space
        shaper — the published command is limit-clamped, rate-capped
        (`safety.sim_rate_limit`) and critically-damped no matter what the
        solver did (a singularity flip leaves as a bounded glide). The shaped
        pose is seeded back into the IK so FK, the render stream, and the next
        solve all see ONE consistent robot. Idle (None) bleeds velocity to a
        smooth stop instead of freezing mid-motion."""
        if plan is None:
            self.cmd_pos = None
            self.cmd_R = None
            q = self.shaper.shape(self.ik.q, t)
            self.ik.seed(np.clip(q, self.ik.soft_lo, self.ik.soft_hi))
            return q
        pb = self.base_R.T @ (plan["pw"] - self.base_pos)            # world → base
        target = SE3.from_rotation_and_translation(plan["R"], pb)
        self.cmd_pos = pb.copy()
        self.cmd_R = target.rotation().as_matrix()   # commanded orientation (base frame), for viz
        q_raw = self.ik.solve(target, iters=self.iters)   # two-stage: position(j1-3) then orientation(j4-6)
        q = self.shaper.shape(q_raw, t)
        # seed clipped to the SOFT limits: the shaper clamps to the hardstops,
        # so integration eps can sit 1e-6 outside soft and spam pink warnings
        self.ik.seed(np.clip(q, self.ik.soft_lo, self.ik.soft_hi))
        return q

    def update(self, hand: HandSample | None, engaged: bool, t: float) -> np.ndarray:
        """plan + commit in one call (single-arm paths and tests; the engine
        calls the halves itself so it can run the pair separation between)."""
        return self.commit(self.plan(hand, engaged, t), t)
