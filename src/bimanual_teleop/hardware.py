"""HardwareSink: the real-robot backend behind the same set_arm/set_hand interface
as RenderSink, so TeleopEngine drives visualization or hardware unchanged.

Every arm command passes through a per-side JointCommandShaper before touching
CAN: clamped to the physical joint limits, per-joint speed-capped
(`hardware.rate_limit`), acceleration-capped (`hardware.accel_limit` — velocity
ramps instead of slamming, so the physical frame is never jerked), and smoothed
by a critically-damped tracker (`hardware.smooth_hz`) feeding the YAM's
motor-side MIT PD. All caps are per second of wall-clock time, so the motion is
the same whatever rate the loop achieves. The shaper initializes from the arm's
MEASURED pose, so the first command glides from wherever the robot actually is
— no startup snap.

PARTIAL RIGS: `hardware.sides` lists the arms that are physically wired —
only those are opened/commanded; the engine's commands for other sides are
dropped here. `hardware.use_hands: false` skips the ORCA hands entirely
(RealHand MOVES the hand to neutral at connect — never construct it for a
hand that isn't mounted).

REST-POSE GATE: each active arm must measure AT the official resting pose
(docs/RESTING_POSE.md, ±hardware.engage_pose_tol per joint, via the calibrated
motor↔model JointMap) before this sink accepts any command for it. This single
check catches the wrong arm on a bus (left/right differ by π on j6), a ±2π
boot wrap, motor-zero drift, and an arm that simply isn't at rest. On failure
construction aborts with the per-joint deltas; the arm is left PD-holding its
measured pose (and goes limp via the watchdog when the process exits).

NOTE: this is the synchronous, single-process bring-up sink. For production the
arms want a dedicated ~250 Hz CAN loop per side (separate process / SCHED_FIFO),
decoupled from vision/IK via latest-value buffers — see README "Hardware day" and
the recon architecture. This class is the correct *logic*; wrap each arm in its
own process when you need the rate.
"""
from __future__ import annotations

import time

import numpy as np

from .arms.joint_map import load_joint_map, map_file_from_rig, rest_pose_gate
from .config import SIDES
from .logging_utils import get_logger
from .safety.shaper import JointCommandShaper

log = get_logger("hardware")


def arm_shaper(rig: dict, q0) -> JointCommandShaper:
    """The hardware-boundary shaper for one YAM arm, from rig config. Factored out
    so the safety wiring is unit-testable without the i2rt SDK."""
    hw = rig.get("hardware", {})
    limits = rig["arms"]["joint_limits"]
    return JointCommandShaper(
        q0,
        rate_limit=float(hw.get("rate_limit", 1.2)),
        smooth_hz=float(hw.get("smooth_hz", 3.0)),
        accel_limit=float(hw.get("accel_limit", 12.0)),
        lo=limits["lower"],
        hi=limits["upper"],
    )


def active_sides(rig: dict) -> tuple[str, ...]:
    """The arms that are physically wired (rig hardware.sides; default: all)."""
    sides = tuple(rig.get("hardware", {}).get("sides", SIDES) or ())
    bad = [s for s in sides if s not in SIDES]
    if bad or not sides:
        raise ValueError(f"hardware.sides must be a non-empty subset of {SIDES}, got {sides}")
    return sides


class HardwareSink:
    def __init__(self, rig: dict):
        from .arms.yam_driver import YamArm
        hw = rig.get("hardware", {})
        self.sides = active_sides(rig)
        self.use_hands = bool(hw.get("use_hands", True))
        tol = float(hw.get("engage_pose_tol", 0.15))
        map_file = map_file_from_rig(rig)

        skipped = [s for s in SIDES if s not in self.sides]
        if skipped:
            log.warning("arms NOT wired this session (hardware.sides): %s — their commands are dropped", skipped)
        if not self.use_hands:
            log.warning("ORCA hands disabled (hardware.use_hands: false)")

        self.arms: dict[str, YamArm] = {}
        self.shapers: dict[str, JointCommandShaper] = {}
        try:
            for s in self.sides:
                jm = load_joint_map(map_file, s)
                if jm is None:
                    raise RuntimeError(
                        f"{s} arm: no motor↔model calibration in {map_file}. "
                        "Run the guided bring-up first: uv run python scripts/hw_bringup.py"
                    )
                neutral = np.asarray(rig["arms"][s]["neutral_q"], dtype=float)
                arm = YamArm(rig["arms"][s]["can_channel"], jm,
                             model_limits=rig["arms"]["joint_limits"],
                             wrap_ref=jm.to_motor(neutral))   # fold reads to the rest frame
                self.arms[s] = arm                     # tracked even if the gate fails → close() releases it
                measured = arm.state()                 # model space — energized = PD-holding this pose
                gate = rest_pose_gate(measured, rig["arms"][s]["neutral_q"], tol)
                log.info("%s arm on %s: %s", s, rig["arms"][s]["can_channel"], gate.describe())
                if not gate.ok:
                    raise RuntimeError(
                        f"{s} arm REST-POSE GATE FAILED on {rig['arms'][s]['can_channel']}: {gate.describe()}.\n"
                        "The arm must hang at the official resting pose before a session "
                        "(docs/RESTING_POSE.md). If it IS at rest, the motor mapping is stale "
                        "(±2π wrap / zero drift / wrong arm on this bus) — re-run: "
                        "uv run python scripts/hw_bringup.py --step rest"
                    )
                self.shapers[s] = arm_shaper(rig, measured)   # glide from the MEASURED pose
        except Exception:
            self.close()
            raise

        self.hands = {}
        if self.use_hands:
            from .hands.real_driver import RealHand
            try:
                for s in self.sides:
                    self.hands[s] = RealHand(model_name=rig["hands"][s]["model_name"])
            except Exception:
                self.close()
                raise

    def set_arm(self, side: str, q: np.ndarray) -> None:
        if side not in self.arms:
            return
        self.arms[side].command(self.shapers[side].shape(q, time.monotonic()))

    def set_hand(self, side: str, joints_deg: dict) -> None:
        if side not in self.hands:
            return
        self.hands[side].set_joint_positions(joints_deg)

    def telemetry(self) -> dict:
        """Per-side motor health (temp/effort/measured) for the live dashboard."""
        out = {}
        for s, arm in self.arms.items():
            try:
                out[s] = arm.telemetry()
            except Exception:
                pass
        return out

    def close(self) -> None:
        for h in getattr(self, "hands", {}).values():
            try:
                h.release()
            except Exception:
                pass
        for a in getattr(self, "arms", {}).values():
            try:
                a.close()
            except Exception:
                pass
