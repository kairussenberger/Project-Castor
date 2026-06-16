"""Return the wired YAM arms to the official HOME / rest pose, then re-anchor the
rest calibration there — the automatic equivalent of `hw_bringup --step rest`.

For each wired arm (rig hardware.sides, or --side / --sides):
  1. open the CAN chain (enable = limp, zero command),
  2. GENTLY glide from the MEASURED pose to model HOME (neutral_q) through the
     saved joint map, rate-limited (<= --rate deg/s),
  3. release torque and let the arm settle to its gravity hang,
  4. if the settle is STILL and lands within REANCHOR_TOL of home (i.e. the arm
     actually reached a clean rest), RE-ANCHOR the map offsets at this hang and
     save it — this defines the current limp pose as neutral_q, so the next
     run_hw engage-gate (hardware.engage_pose_tol) passes. Otherwise WARN and
     leave the map UNTOUCHED (a tangled / propped / obstructed arm must never be
     baked into the calibration),
  5. switch the motors off (limp) — the arm hangs at the home it was driven to.

SAFETY: motion is rate-limited and starts from the measured pose; a side without
a joint map is SKIPPED (it cannot be commanded safely). Hand on the e-stop. SIGINT
(dashboard STOP / Ctrl+C) releases torque via each chain's teardown.

    uv run python -m bimanual_teleop.launch.return_home              # all wired arms
    uv run python -m bimanual_teleop.launch.return_home --side left
    uv run python -m bimanual_teleop.launch.return_home --no-reanchor  # move home only
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ..config import SIDES, load_rig
from ..arms.joint_map import (JointMap, load_joint_map, map_file_from_rig,
                              rest_pose_gate, save_joint_map)
from ..logging_utils import get_logger

log = get_logger("return_home")

REANCHOR_TOL = 0.20   # rad (~11.5°): max settle-vs-home error we still trust as "at rest"
SETTLE_S = 1.5        # stillness window after releasing torque


def _deg(v) -> str:
    return "[" + " ".join(f"{np.degrees(x):+6.1f}" for x in np.asarray(v).reshape(-1)) + "]°"


def home_one(side: str, rig: dict, map_file, rate_deg_s: float, reanchor: bool,
             anchor_only: bool = False) -> bool:
    neutral = np.asarray(rig["arms"][side]["neutral_q"], dtype=float)
    channel = rig["arms"][side]["can_channel"]
    engage_tol = float(rig.get("hardware", {}).get("engage_pose_tol", 0.15))

    jm = load_joint_map(map_file, side)
    if jm is None:
        log.warning("%s arm: no joint map (%s) — SKIPPED (cannot command without calibration). "
                    "Run scripts/hw_bringup.py --side %s", side, map_file, side)
        return True   # not a failure; just nothing to do for this side

    home_motor = jm.to_motor(neutral)
    from ..arms.yam_chain import YamChain        # hardware-only import, guarded here

    # Anchor-only: NO motion. Capture the current limp hang and define it as the rest
    # pose (keeping the measured signs), so the engage gate + HOME target are correct
    # again after a channel swap / drift. The arm must already be hanging at rest.
    if anchor_only:
        chain = YamChain(channel, rate_deg_s=rate_deg_s)
        try:
            still, q_rest, vmax, ptp = chain.settle_check(SETTLE_S)
            if not still:
                log.warning("%s arm: NOT re-anchoring (moving, v=%.3f) — steady it and retry", side, vmax)
                return False
            jm2 = JointMap.anchored_at_rest(jm.signs, q_rest, neutral)
            save_joint_map(map_file, side, jm2, channel=channel, q_motor_rest=q_rest)
            log.info("%s arm on %s: ✓ rest RE-ANCHORED in place (no move) at %s → engage gate ±%.1f° will pass",
                     side, channel, _deg(jm2.to_model(q_rest)), np.degrees(engage_tol))
            return True
        finally:
            chain.off()

    # Fold reads to the HOME frame so a joint parked on the ±π boundary (j2 ≈ ±177°)
    # never glides the long way round — the move is always the short way to rest.
    chain = YamChain(channel, rate_deg_s=rate_deg_s, wrap_ref=home_motor)
    try:
        q_now = chain.read_pos()[:6]
        log.info("%s arm on %s: at %s → gliding HOME %s at <=%.0f°/s",
                 side, channel, _deg(jm.to_model(q_now)), _deg(neutral), rate_deg_s)
        chain.waypoints(q_now, [home_motor], dwell_s=1.2)
        chain.hold(home_motor)
        time.sleep(0.8)

        # Release and let the arm settle to its gravity hang, then re-anchor there.
        chain.idle()
        time.sleep(0.6)
        still, q_rest, vmax, ptp = chain.settle_check(SETTLE_S)
        gate = rest_pose_gate(jm.to_model(q_rest), neutral, REANCHOR_TOL)
        log.info("%s arm: settled %s still=%s (v=%.3f drift=%.2f°)  %s",
                 side, _deg(jm.to_model(q_rest)), still, vmax, np.degrees(ptp), gate.describe())

        if not reanchor:
            log.info("%s arm: --no-reanchor → calibration left as-is", side)
        elif still and gate.ok:
            jm2 = JointMap.anchored_at_rest(jm.signs, q_rest, neutral)
            save_joint_map(map_file, side, jm2, channel=channel, q_motor_rest=q_rest)
            # By construction the current limp pose now reads as neutral_q → the
            # run_hw engage gate (±engage_pose_tol) will pass.
            log.info("%s arm: ✓ re-anchored rest at this hang → %s (engage gate ±%.1f° will pass)",
                     side, map_file, np.degrees(engage_tol))
        else:
            why = "not still" if not still else f"off home by >{np.degrees(REANCHOR_TOL):.0f}°"
            log.warning("%s arm: NOT re-anchoring (%s) — map UNCHANGED. The arm did not reach a "
                        "clean rest (tangled/propped/obstructed?). Clear it and retry.", side, why)
            return False
        return True
    finally:
        chain.off()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=list(SIDES), default=None,
                    help="home only this side (default: all rig hardware.sides)")
    ap.add_argument("--sides", default=None, metavar="right[,left]",
                    help="override rig hardware.sides")
    ap.add_argument("--rate", type=float, default=10.0, metavar="DEG_S",
                    help="per-joint glide speed cap (default 10°/s, gentle)")
    ap.add_argument("--no-reanchor", action="store_true",
                    help="move to home only; do NOT re-anchor the rest calibration")
    ap.add_argument("--anchor-only", action="store_true",
                    help="do NOT move; capture the current limp hang as the rest pose "
                         "(fixes the engage gate + HOME target after a channel swap)")
    args = ap.parse_args()

    rig = load_rig()
    map_file = map_file_from_rig(rig)
    if args.side:
        sides = [args.side]
    elif args.sides:
        sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    else:
        sides = list(rig.get("hardware", {}).get("sides", SIDES))
    log.info("return-home: sides=%s rate=%.0f°/s reanchor=%s anchor_only=%s map=%s",
             sides, args.rate, not args.no_reanchor, args.anchor_only, map_file)

    ok = True
    try:
        for side in sides:
            ok = home_one(side, rig, map_file, args.rate, not args.no_reanchor,
                          anchor_only=args.anchor_only) and ok
    except KeyboardInterrupt:
        log.info("interrupted — chains released torque on teardown")
        return 130
    if ok:
        log.info("✓ return-home complete for %s", sides)
        return 0
    log.warning("return-home finished with WARNINGS — see above (some arms not re-anchored)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
