#!/usr/bin/env python
"""Guided first-contact bring-up for one real YAM arm — SAFETY FIRST, tiny moves only.

Why this exists: the i2rt motor firmware and this repo's model use DIFFERENT joint
conventions (zeros/signs — compare i2rt yam.xml ranges with rig.yaml), and the
arms are mounted SIDEWAYS so i2rt's table-mount gravity compensation must never
run. Before run_hw / jog may energize, the motor↔model map has to be MEASURED.
This tool does that with the arm hanging at the official resting pose
(docs/RESTING_POSE.md), moving single joints ±2-3° at ≤10°/s, hand on the e-stop.

    uv run python scripts/hw_bringup.py                  # full guided sequence
    uv run python scripts/hw_bringup.py --step scan      # which motors answer on the bus?
    uv run python scripts/hw_bringup.py --step rest      # capture/re-anchor the resting pose
    uv run python scripts/hw_bringup.py --step signs     # measure per-joint direction signs
    uv run python scripts/hw_bringup.py --step verify    # model-space wiggle through the saved map
    uv run python scripts/hw_bringup.py --step watchdog  # do motors stop when commands stop?

Steps in order (each asks before energizing anything):
  links    SocketCAN interface present + UP (needs sudo bring-up otherwise).
  scan     ping motor IDs 1..7: who is on this bus? (expects exactly 1..6; a 7th
           means the stock gripper is still attached). Motors are switched off after.
  rest     read the hanging pose, check it is STILL (the arm must hang limp at the
           official rest), show the i2rt boot-check verdict, save the snapshot.
           Re-run this any session where the rest-pose gate complains (±2π wraps).
  signs    for each joint: ±2° motor-space nudge while the DASHBOARD previews the
           assumed model-space motion — you answer same/opposite. Writes the map.
  verify   model-space ±3° wiggle per joint THROUGH the saved map — dashboard and
           metal must now agree everywhere; returns to rest.
  watchdog with the arm at rest, stop the command stream mid-hold: the motors must
           go limp on their own (the crash backstop). Validates the deadman chain.

Run the dashboard in a second terminal for the preview steps:
    uv run python scripts/dashboard.py        →  http://127.0.0.1:8180
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.config import SIDES, load_rig                              # noqa: E402
from bimanual_teleop.arms.ik import ArmIK                                        # noqa: E402
from bimanual_teleop.arms.joint_map import (JointMap, load_joint_map,            # noqa: E402
                                            map_file_from_rig, rest_pose_gate,
                                            save_joint_map)
from bimanual_teleop.vr.frames import quat_to_R                                  # noqa: E402

MAX_DEG_S = 10.0                       # hard speed cap for every bring-up motion
SIGN_AMP = np.radians(2.0)             # signs step amplitude
PREVIEW_GAIN = 12.0                    # dashboard DIRECTION preview: ~24 deg on screen per 2 deg of metal
VERIFY_AMP = np.radians(3.0)           # verify step amplitude


# --------------------------------------------------------------------------- #
# UX helpers
# --------------------------------------------------------------------------- #

def banner(txt: str) -> None:
    print("\n" + "=" * 78 + f"\n{txt}\n" + "=" * 78)


def confirm(txt: str) -> bool:
    """ENTER continues, anything starting with n/q aborts the step."""
    a = input(f"{txt}  [ENTER=yes / n=abort] ").strip().lower()
    return not a.startswith(("n", "q"))


def deg(v) -> str:
    return "[" + " ".join(f"{np.degrees(x):+7.1f}" for x in np.asarray(v).reshape(-1)) + "]°"


# --------------------------------------------------------------------------- #
# Chain-level session — shared with the runtime driver (arms/yam_chain.py):
# raw i2rt CAN chain, OUR shaping, no model-based torques, no MotorChainRobot.
# Convention-agnostic on purpose — it works before any mapping exists.
# --------------------------------------------------------------------------- #

def open_chain(channel: str):
    from bimanual_teleop.arms.yam_chain import YamChain
    return YamChain(channel, rate_deg_s=MAX_DEG_S)


# --------------------------------------------------------------------------- #
# Dashboard preview (same render stream the rest of the repo uses)
# --------------------------------------------------------------------------- #

class _ArmShim:
    def __init__(self, rig, side, ik):
        self.ik = ik
        self.base_R = quat_to_R(rig["arms"][side]["base_quat"])
        self.base_pos = np.asarray(rig["arms"][side]["base_pos"], dtype=float)
        self.cmd_pos = None
        self.cmd_R = None


class Preview:
    """Pushes model-space joint previews to the dashboard/Unity render stream."""

    def __init__(self, rig: dict, side: str):
        from bimanual_teleop.render_sink import RenderSink
        self.rig, self.side = rig, side
        self.ik = {s: ArmIK(rig, s) for s in SIDES}
        self.arm = {s: _ArmShim(rig, s, self.ik[s]) for s in SIDES}
        self.calib_status = None
        self.sink = RenderSink(rig)
        self.t0 = time.monotonic()
        for s in SIDES:
            self.sink.set_arm(s, self.ik[s].q)
        self._last = 0.0

    def show(self, q_model: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last < 1 / 30:
            return
        self._last = now
        self.n_pub = getattr(self, "n_pub", 0) + 1
        self.ik[self.side].seed(np.clip(q_model, self.ik[self.side].hard_lo,
                                        self.ik[self.side].hard_hi))
        self.sink.set_arm(self.side, self.ik[self.side].q)
        self.sink.publish(self, None, {s: False for s in SIDES}, 30.0, now - self.t0)

    def close(self) -> None:
        if hasattr(self.sink, "close"):
            self.sink.close()


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #

def step_links(channel: str) -> bool:
    banner(f"STEP links — SocketCAN interface {channel}")
    base = Path("/sys/class/net") / channel
    if not base.exists():
        others = sorted(p.name for p in Path("/sys/class/net").iterdir() if p.name.startswith("can"))
        print(f"✗ {channel} does not exist. CAN interfaces present: {others or 'none'}")
        print("  Bring it up (sudo, on this host):")
        print("    sudo modprobe can can_raw gs_usb")
        print(f"    sudo ip link set {channel} up type can bitrate 1000000")
        return False
    up = bool(int((base / "flags").read_text().strip(), 16) & 0x1)
    if not up:
        print(f"✗ {channel} exists but is DOWN. Run:")
        print(f"    sudo ip link set {channel} up type can bitrate 1000000")
        return False
    print(f"✓ {channel} is UP @ 1 Mbit (gs_usb)")
    return True


def step_scan(channel: str, rig: dict, side: str) -> bool:
    banner(f"STEP scan — who answers on {channel}?")
    print("Each motor ID 1..7 gets an enable ping and an immediate off. Zero command\n"
          "= zero torque; the arm stays limp. Expect IDs 1..6 (7 = stock gripper —\n"
          "should be REMOVED on this rig).")
    if not confirm("Arm hanging free, e-stop in reach?"):
        return False
    from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface, MotorType
    iface = DMSingleMotorCanInterface(channel=channel, bustype="socketcan")
    online = []
    try:
        for mid in range(1, 8):
            try:
                info = iface.motor_on(mid, MotorType.DM4310)
                pos = getattr(info, "position", getattr(info, "pos", float("nan")))
                print(f"  motor {mid}: ONLINE  pos={np.degrees(float(pos)):+8.2f}°")
                online.append(mid)
                iface.motor_off(mid)
            except Exception:
                print(f"  motor {mid}: no answer")
    finally:
        try:
            iface.close()
        except Exception:
            pass
    expected = rig["arms"][side]["can_channel"]
    print(f"\nonline motors on {channel}: {online}")
    if online == list(range(1, 7)):
        print(f"✓ exactly the 6 YAM joint motors — this bus carries an arm")
    elif 7 in online:
        print("⚠ motor 7 answered: the stock gripper is still on the chain — remove it or "
              "expect i2rt enumeration to differ from this repo's 6-joint assumption")
    elif not online:
        print("✗ nothing answered — wrong channel, arm unpowered, or CAN wiring/termination issue")
        return False
    else:
        print(f"⚠ unexpected ID set {online} — investigate before going further")
    if channel != expected:
        print(f"⚠ rig.yaml maps the {side} arm to {expected}, but you scanned {channel}.\n"
              f"  Fix rig.yaml arms.{side}.can_channel before run_hw.")
    return bool(online)


def _boot_verdict_table(q_rest: np.ndarray) -> None:
    """How far are this rig's motor zeros from i2rt's convention? INFORMATIONAL:
    the runtime drives the chain directly (arms/yam_chain.py) and does NOT go
    through i2rt's MotorChainRobot, so its boot qpos check no longer applies —
    but i2rt's OWN tools (examples, motor_chain_robot __main__) still reject
    poses outside these rows, which is expected on this rig."""
    from i2rt.robot_models import ARM_YAM_XML_PATH
    from i2rt.robots.get_robot import _load_joint_limits_from_xml
    lim = _load_joint_limits_from_xml(ARM_YAM_XML_PATH)[:6]
    print("\nmotor zeros vs i2rt's own convention (informational — the runtime does not\n"
          "use i2rt's MotorChainRobot; only i2rt's own tools enforce these rows):")
    for i in range(6):
        lo, hi = lim[i, 0] - 0.25, lim[i, 1] + 0.25
        ok = lo <= q_rest[i] <= hi
        print(f"  j{i + 1}: {np.degrees(q_rest[i]):+8.2f}° in [{np.degrees(lo):+8.2f}, "
              f"{np.degrees(hi):+8.2f}]°  {'✓' if ok else '✗ outside i2rt convention'}")


def step_rest(channel: str, rig: dict, side: str, map_file: Path) -> np.ndarray | None:
    banner(f"STEP rest — capture the {side} arm's resting pose on {channel}")
    print("The arm must hang LIMP at the official rest (docs/RESTING_POSE.md):\n"
          "straight down at the side, slight elbow bend, palm facing the center shaft.\n"
          "Motors will be enabled with ZERO torque + light damping — no motion.")
    if not confirm("Arm is at rest and nobody is touching it?"):
        return None
    sess = open_chain(channel)
    try:
        still, q_rest, vmax, ptp = sess.settle_check()
        print(f"\nmeasured motor-space pose {deg(q_rest)}")
        print(f"stillness: max|vel|={vmax:.3f} rad/s, drift={np.degrees(ptp):.2f}° "
              f"over 1.5 s → {'✓ still' if still else '✗ MOVING'}")
        if not still:
            print("✗ the arm is moving — steady it, then re-run --step rest")
            return None
        _boot_verdict_table(q_rest)
        neutral = np.asarray(rig["arms"][side]["neutral_q"], dtype=float)
        print(f"\nmodel rest (rig neutral_q)  {deg(neutral)}")
        existing = load_joint_map(map_file, side)
        if existing is not None:
            gate = rest_pose_gate(existing.to_model(q_rest), neutral,
                                  float(rig["hardware"].get("engage_pose_tol", 0.15)))
            print(f"through the SAVED map: {gate.describe()}")
            if gate.ok:
                print("✓ saved calibration still matches — nothing to do")
                return q_rest
            print("⚠ saved map disagrees (±2π wrap or drift). Re-anchoring offsets at this rest.")
            jm = JointMap.anchored_at_rest(existing.signs, q_rest, neutral)
            save_joint_map(map_file, side, jm, channel=channel, q_motor_rest=q_rest)
            print(f"✓ offsets re-anchored → {map_file}")
        else:
            print("\nNo joint map yet for this side — continue with --step signs.\n"
                  "(rest snapshot will be taken there; nothing saved now)")
        return q_rest
    finally:
        sess.off()


def _coverage_table(jm: JointMap, rig: dict, side: str = "right") -> None:
    """The model joint limits mapped into MOTOR space — these become the runtime
    chain clamp (YamArm.set_bounds). The i2rt xml limits are NOT compared: under
    this rig's motor zeros they describe nothing physical; rig.yaml's limits
    (from the URDF + the measured rest anchor) are the authoritative hardstops."""
    lo = np.asarray(rig["arms"]["joint_limits"]["lower"], dtype=float)
    hi = np.asarray(rig["arms"]["joint_limits"]["upper"], dtype=float)
    mapped = np.sort(np.stack([jm.to_motor(lo), jm.to_motor(hi)]), axis=0)
    rest = jm.to_motor(np.asarray(rig["arms"][side]["neutral_q"], dtype=float))
    print("\nmodel joint limits in MOTOR space (= the runtime chain clamp):")
    for i in range(6):
        inside = mapped[0, i] - 1e-9 <= rest[i] <= mapped[1, i] + 1e-9
        print(f"  j{i + 1}: [{np.degrees(mapped[0, i]):+8.1f}, {np.degrees(mapped[1, i]):+8.1f}]°  "
              f"rest at {np.degrees(rest[i]):+8.1f}°  {'✓' if inside else '✗ rest outside its own limits?!'}")


def step_signs(channel: str, rig: dict, side: str, map_file: Path) -> bool:
    banner(f"STEP signs — measure per-joint direction signs ({side} arm, {channel})")
    print("For each joint: the arm nudges ±2° in MOTOR space at ≤10°/s while the\n"
          "DASHBOARD previews the assumed MODEL-space motion. You answer whether the\n"
          "real joint moved the same way. Keep the dashboard visible:\n"
          "    uv run python scripts/dashboard.py   →  http://127.0.0.1:8180")
    if not confirm("Arm at rest, dashboard open, e-stop in reach?"):
        return False
    neutral = np.asarray(rig["arms"][side]["neutral_q"], dtype=float)
    sess = open_chain(channel)
    preview = Preview(rig, side)
    try:
        still, q_rest, _, _ = sess.settle_check()
        if not still:
            print("✗ arm not still — abort")
            return False
        print(f"rest (motor) {deg(q_rest)}\nEngaging gentle PD hold at rest…")
        sess.hold(q_rest)
        time.sleep(1.0)
        signs = np.ones(6)
        for j in range(6):
            while True:
                s = signs[j]
                print(f"\n— joint {j + 1}/6: nudging motor j{j + 1} by +2° then −2° "
                      f"(dashboard preview EXAGGERATED x{PREVIEW_GAIN:.0f}, sign {s:+.0f}) —")
                if not confirm("ready?"):
                    return False

                def tick(q_cmd, j=j, s=s):
                    qm = neutral.copy()
                    # Direction preview, not magnitude: 2 deg true-scale is ~2px on the page.
                    qm[j] = neutral[j] + s * PREVIEW_GAIN * (q_cmd[j] - q_rest[j])
                    preview.show(qm)

                tgt_a, tgt_b = q_rest.copy(), q_rest.copy()
                tgt_a[j] += SIGN_AMP
                tgt_b[j] -= SIGN_AMP
                print(f"  nudging (~6 s, +/-2 deg at motor j{j + 1}, dashboard swings "
                      f"x{PREVIEW_GAIN:.0f}) ...", end="", flush=True)
                sess.waypoints(q_rest, [tgt_a, tgt_b, q_rest.copy()], on_tick=tick)
                moved = np.degrees(float(np.abs(sess.read_pos()[j] - q_rest[j])))
                print(f" done (back to within {moved:.2f} deg of rest, "
                      f"{getattr(preview, 'n_pub', 0)} preview frames sent)")
                a = input("real joint vs dashboard: [s]ame / [o]pposite / [r]epeat / [a]bort: ").strip().lower()
                if a.startswith("s"):
                    break
                if a.startswith("o"):
                    signs[j] = -signs[j]
                    print(f"  sign j{j + 1} flipped to {signs[j]:+.0f}; repeating to confirm…")
                elif a.startswith("a"):
                    return False
        jm = JointMap.anchored_at_rest(signs, q_rest, neutral)
        save_joint_map(map_file, side, jm, channel=channel, q_motor_rest=q_rest)
        print(f"\n✓ joint map saved → {map_file}")
        print(f"  signs   {signs.astype(int).tolist()}")
        print(f"  offsets {deg(jm.offsets)}")
        _coverage_table(jm, rig, side)
        print("\nNEXT: uv run python scripts/hw_bringup.py --step verify")
        return True
    finally:
        preview.close()
        sess.off()


def step_verify(channel: str, rig: dict, side: str, map_file: Path) -> bool:
    banner(f"STEP verify — model-space wiggle THROUGH the saved map ({side}, {channel})")
    jm = load_joint_map(map_file, side)
    if jm is None:
        print(f"✗ no calibration for {side} in {map_file} — run --step signs first")
        return False
    neutral = np.asarray(rig["arms"][side]["neutral_q"], dtype=float)
    tol = float(rig["hardware"].get("engage_pose_tol", 0.15))
    if not confirm("Arm at rest, dashboard open, e-stop in reach?"):
        return False
    sess = open_chain(channel)
    preview = Preview(rig, side)
    try:
        still, q_rest, _, _ = sess.settle_check()
        gate = rest_pose_gate(jm.to_model(q_rest), neutral, tol)
        print(f"rest-pose gate: {gate.describe()}")
        if not (still and gate.ok):
            print("✗ gate failed — re-run --step rest (re-anchors offsets), then verify again")
            return False
        sess.hold(q_rest)
        time.sleep(1.0)
        worst = 0.0
        for j in range(6):
            print(f"\n— joint {j + 1}/6: model-space ±3° — page swings x{PREVIEW_GAIN:.0f} for visibility; "
                  f"judge DIRECTION agreement, the metal itself moves only ±3° —")
            if not confirm("ready?"):
                return False

            def tick(q_cmd):
                qm = neutral + PREVIEW_GAIN * (jm.to_model(q_cmd) - neutral)
                preview.show(qm)

            qa, qb = neutral.copy(), neutral.copy()
            qa[j] += VERIFY_AMP
            qb[j] -= VERIFY_AMP
            sess.waypoints(q_rest, [jm.to_motor(qa), jm.to_motor(qb), q_rest.copy()], on_tick=tick)
            track = float(np.max(np.abs(sess.read_pos() - q_rest)))
            worst = max(worst, track)
            a = input("matched the dashboard? [y]/n: ").strip().lower()
            if a.startswith("n"):
                print(f"✗ j{j + 1} mismatch — re-run --step signs (that joint's sign is wrong)")
                return False
        print(f"\n✓ all 6 joints verified through the map (settle residual ≤ {np.degrees(worst):.2f}°)")
        print("NEXT (in order, e-stop in hand):")
        print("  uv run python scripts/hw_bringup.py --step watchdog")
        print("  uv run python scripts/jog_arms.py --sink hw        # free keyboard jog")
        print("  uv run python -m bimanual_teleop.launch.run_hw --vr replay recordings/<known_good>.npz")
        print("  uv run python -m bimanual_teleop.launch.run_hw --vr orbit --clutch gesture")
        return True
    finally:
        preview.close()
        sess.off()


def step_watchdog(channel: str) -> bool:
    banner(f"STEP watchdog — do the motors stop when the host stops? ({channel})")
    print("The arm will PD-hold its rest pose, then the command stream STOPS, exactly\n"
          "like a host crash. Configured DM watchdogs release torque on their own —\n"
          "the arm should go LIMP (it is already hanging, so 'limp' = a tiny sag).")
    if not confirm("Arm at rest, clear of people?"):
        return False
    sess = open_chain(channel)
    try:
        sess.hold()
        time.sleep(1.5)
        print("holding… now STOPPING the command stream.")
        sess.stop_stream_for_watchdog_test()
        time.sleep(2.0)
        a = input("Push gently on the forearm: is the arm LIMP now? [y]/n: ").strip().lower()
        if a.startswith("n"):
            print("✗ motors still stiff ⇒ NO motor-side watchdog. The host process is then a\n"
                  "  single point of failure — configure the timeout before any teleop:\n"
                  "    cd ~/i2rt/i2rt/motor_config_tool && uv run python set_timeout.py \\\n"
                  f"        --channel {channel} --timeout    # writes+persists a CAN timeout\n"
                  "  then re-run this step.")
            return False
        print("✓ watchdog releases torque on stream loss — crash backstop verified")
        return True
    finally:
        sess.off()


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=list(SIDES), default="right")
    ap.add_argument("--channel", default=None, help="override rig arms.<side>.can_channel")
    ap.add_argument("--step", choices=["all", "links", "scan", "rest", "signs", "verify", "watchdog"],
                    default="all")
    args = ap.parse_args()

    rig = load_rig()
    channel = args.channel or rig["arms"][args.side]["can_channel"]
    map_file = map_file_from_rig(rig)
    print(f"side={args.side}  channel={channel}  map={map_file}")
    if args.side not in tuple(rig.get("hardware", {}).get("sides", SIDES)):
        print(f"⚠ note: {args.side} is not in rig hardware.sides — run_hw will ignore it until added")

    steps = [args.step] if args.step != "all" else ["links", "scan", "rest", "signs", "verify", "watchdog"]
    for s in steps:
        ok = {
            "links": lambda: step_links(channel),
            "scan": lambda: step_scan(channel, rig, args.side),
            "rest": lambda: step_rest(channel, rig, args.side, map_file) is not None,
            "signs": lambda: step_signs(channel, rig, args.side, map_file),
            "verify": lambda: step_verify(channel, rig, args.side, map_file),
            "watchdog": lambda: step_watchdog(channel),
        }[s]()
        if not ok:
            print(f"\nstopped at step '{s}' — fix and re-run: "
                  f"uv run python scripts/hw_bringup.py --step {s}")
            return 1
    print("\n✓ bring-up complete for this arm.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\naborted — motors were switched off by the step teardown; "
              "if the arm is stiff, power-cycle the rig")
        raise SystemExit(130)
