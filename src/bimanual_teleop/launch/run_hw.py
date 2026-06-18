"""Run the teleop pipeline against REAL hardware (Linux control host).

This is the hardware bring-up entrypoint. It reuses the exact TeleopEngine +
controllers + supervisor as the render path — only the *sink* changes (HardwareSink →
YAM CAN + ORCA serial). Single-process bring-up form; for production split each
arm into its own ~250 Hz CAN process (see README "Hardware day").

Prereqs (Ubuntu): SocketCAN up (`sudo ip link set can0 up type can bitrate 1000000`,
same for can1), i2rt SDK installed, ORCA hands tensioned + calibrated.

    python -m bimanual_teleop.launch.run_hw --vr orbit
    python -m bimanual_teleop.launch.run_hw --vr orbit --record recordings/hw_session.npz
    python -m bimanual_teleop.launch.run_hw --vr replay recordings/hw_session.npz --clutch recorded

SAFETY: starts in IDLE (not following). Engage via the configured clutch. Ctrl+C
or e-stop releases torque on all devices.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
TELEMETRY_PATH = REPO_ROOT / "out" / "hw_telemetry.json"

from ..config import load_rig
from ..engine import TeleopEngine
from ..safety.clutch import AlwaysOn, GestureClutch, RecordedClutch
from ..safety.supervisor import Supervisor
from ..vr.ingest import make_source
from ..vr.replay import SessionRecorder


def glide_arms_home(engine, sink, sides, period, *, rate_cap, dwell_s,
                    on_tick=None, clock=time.monotonic, sleep=time.sleep, margin_s=0.6):
    """Glide the wired arms from their current commanded pose back to rest (neutral_q),
    ENERGIZED, through the existing hardware shaper (sink.set_arm → shapers[s].shape, so
    the limit-clamp + rate-cap boundary is never bypassed); hold for dwell_s; then
    re-sync the engine IK and BOTH shapers to rest. The arm never goes limp — torque is
    held throughout (STOP/SIGINT still releases it via run_hw's finally).

    The re-sync is essential: without resetting engine.arm[s].ik AND the arm-control
    shaper, the first re-armed (disengaged) tick would shape from the stale replay-end
    pose and yank the already-home arm back up.

    on_tick(t) runs once per control tick for render/telemetry side effects.
    clock/sleep are injectable so the glide is testable without real time or hardware.
    """
    sides = [s for s in sides if s in engine.arm]
    if not sides:
        return
    goals = {s: np.asarray(engine.arm[s].ik.q0, dtype=float) for s in sides}
    # Self-size the glide window from the worst-case joint excursion at the rate cap, so
    # the critically-damped shaper has time to converge before we re-arm the replay.
    span = max((float(np.max(np.abs(np.asarray(engine.arm[s].ik.q) - goals[s])))
                for s in sides), default=0.0)
    glide_s = span / max(float(rate_cap), 1e-3) + float(margin_s)
    t_end = clock() + glide_s + max(0.0, float(dwell_s))
    while clock() < t_end:
        t = clock()
        for s in sides:
            sink.set_arm(s, goals[s])
        if on_tick is not None:
            on_tick(t)
        dt = period - (clock() - t)
        if dt > 0:
            sleep(dt)
    # Re-sync to a coherent rest state (engine IK + arm-control shaper + hardware shaper,
    # all at rest with zero velocity) so the next take starts clean.
    t = clock()
    shapers = getattr(getattr(sink, "hw", sink), "shapers", {})
    for s in sides:
        engine.arm[s].ik.reset()
        engine.arm[s].shaper.reset(goals[s], t)
        shp = shapers.get(s)
        if shp is not None:
            shp.reset(goals[s], t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vr", choices=["vuer", "orbit", "fake", "replay"], default="orbit")
    ap.add_argument("replay_path", nargs="?", help="session .npz when --vr replay")
    ap.add_argument("--clutch", choices=["gesture", "recorded", "always"], default="gesture",
                    help="hardware engage policy. gesture: pinch-to-engage deadman. "
                         "always: follow continuously once calibrated (NO deadman — e-stop in hand). "
                         "recorded: replay engage decisions.")
    ap.add_argument("--record", metavar="PATH", default=None,
                    help="write VR frames + engage state to a replayable .npz session")
    ap.add_argument("--hz", type=float, default=None, help="override control rate")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="replay time-stretch: 0.2 plays a recording 5x slower (replay only)")
    ap.add_argument("--rate-limit", type=float, default=None, metavar="RAD_S",
                    help="override hardware.rate_limit at the CAN shaper for this run")
    ap.add_argument("--sides", default=None, metavar="right[,left]",
                    help="override rig hardware.sides — arms physically wired this session")
    ap.add_argument("--no-hands", action="store_true",
                    help="override rig hardware.use_hands=false (ORCA hands not mounted)")
    ap.add_argument("--swap-sides", action="store_true",
                    help="drive each recorded hand with the OPPOSITE arm (L/R mirror): "
                         "the left hand commands the right arm and vice versa. Calibration "
                         "and the rest-pose gate are unchanged.")
    ap.add_argument("--mirror-fb", action="store_true",
                    help="mirror the motion front↔back only (keeps left/right): mapping.mirror_forward")
    ap.add_argument("--mirror-lr", action="store_true",
                    help="mirror the motion left↔right only (keeps front/back): mapping.mirror_lateral")
    ap.add_argument("--loop-home", action="store_true",
                    help="replay only: after each pass, glide the arms back to rest "
                         "(energized, through the shaper) then replay again — the next "
                         "take waits for the home transition. Runs until STOP or --cycles.")
    ap.add_argument("--home-dwell-s", type=float, default=None, metavar="S",
                    help="loop-home: seconds to hold at rest between takes "
                         "(default hardware.replay_loop_dwell_s)")
    ap.add_argument("--cycles", type=int, default=0, metavar="N",
                    help="loop-home: stop after N takes (default 0 = until STOP)")
    args = ap.parse_args()

    if sys.platform == "darwin":
        print("WARNING: real YAM control needs Linux/SocketCAN; macOS can't run the CAN loop.")

    rig = load_rig()
    hw = rig.setdefault("hardware", {})
    if args.sides:
        hw["sides"] = [s.strip() for s in args.sides.split(",") if s.strip()]
    if args.no_hands:
        hw["use_hands"] = False
    if args.swap_sides:
        rig.setdefault("mapping", {})["swap_sides"] = True
    if args.mirror_fb or args.mirror_lr:
        m = rig.setdefault("mapping", {})
        m["mirror_forward"] = bool(args.mirror_fb)
        m["mirror_lateral"] = bool(args.mirror_lr)
        print(f"[hw] mirror: front↔back={args.mirror_fb}  left↔right={args.mirror_lr}")
    print(f"[hw] arms: {hw.get('sides', ['left', 'right'])}  hands: {hw.get('use_hands', True)}  "
          f"(rig hardware.sides / --sides; rest-pose gate ±"
          f"{float(hw.get('engage_pose_tol', 0.15)):.2f} rad)")
    rig["vr"]["transport"] = args.vr
    if args.vr == "replay":
        if not args.replay_path:
            ap.error("--vr replay needs a session file: run_hw --vr replay session.npz")
        rig["vr"]["replay_path"] = args.replay_path
        rig["vr"]["replay_speed"] = float(args.speed)
        if args.speed != 1.0:
            print(f"[hw] replay time-stretch x{1.0 / args.speed:.1f} slower (speed {args.speed})")
    if args.loop_home and args.vr != "replay":
        ap.error("--loop-home only applies to --vr replay")
    home_dwell_s = (args.home_dwell_s if args.home_dwell_s is not None
                    else float(rig.get("hardware", {}).get("replay_loop_dwell_s", 1.0)))
    if args.loop_home:
        print(f"[hw] loop-home ON: replay → glide home (dwell {home_dwell_s:.1f}s) → replay"
              + (f", stopping after {args.cycles} take(s)" if args.cycles else " (until STOP)"))
    hz = args.hz or rig["control"]["arm_hz"]
    # Hardware speed derating: scale the IK joint-velocity budget down for real
    # motors (the sink's JointCommandShaper independently caps speed again).
    scale = float(rig.get("hardware", {}).get("max_vel_scale", 0.35))
    rig["ik"]["max_vel"] = float(rig["ik"]["max_vel"]) * scale
    print(f"[hw] ik.max_vel derated x{scale:.2f} -> {rig['ik']['max_vel']:.1f} rad/s; "
          f"shaper rate_limit {rig.get('hardware', {}).get('rate_limit', 1.2)} rad/s")

    if args.rate_limit:
        rig["hardware"]["rate_limit"] = float(args.rate_limit)
        print(f"[hw] shaper rate_limit overridden -> {args.rate_limit:.2f} rad/s")
    src = make_source(rig)
    if args.clutch == "recorded":
        clutch = RecordedClutch(src)
    elif args.clutch == "always":
        clutch = AlwaysOn()
        print("[hw] clutch=always — arms FOLLOW CONTINUOUSLY once calibrated "
              "(no pinch deadman); keep the e-stop in hand.")
    else:
        clutch = GestureClutch()

    from ..hardware import HardwareSink
    sink = HardwareSink(rig)
    render = None
    try:
        from ..render_sink import RenderSink

        class _Tee:
            def __init__(self, hw, rd):
                self.hw, self.rd = hw, rd
                self.arms = hw.arms

            def set_arm(self, side, q):
                self.hw.set_arm(side, q)
                try:
                    self.rd.set_arm(side, q)
                except Exception:
                    pass

            def set_hand(self, side, j):
                self.hw.set_hand(side, j)
                try:
                    self.rd.set_hand(side, j)
                except Exception:
                    pass

            def close(self):
                self.hw.close()
                try:
                    self.rd.close()
                except Exception:
                    pass

        render = RenderSink(rig)
        sink = _Tee(sink, render)
        print("[hw] dashboard mirror on (render.state)")
    except Exception as e:
        print(f"[hw] dashboard mirror disabled ({e}) — hardware loop unaffected")
    engine = TeleopEngine(rig, sink)
    supervisor = Supervisor(rig, clutch)
    # Engine control channel (dashboard CALIBRATE button → port 8201). REQUIRED for
    # live transports: --vr orbit/vuer start with follow LOCKED until an IN-SESSION
    # calibration completes (vr.require_calibration — the same gate run_teleop has).
    # Without this the dashboard CALIBRATE button has nothing to talk to and the arms
    # never unlock ("prompted to calibrate but the button does nothing"). Best-effort:
    # a busy port must never block the hardware loop.
    ctl = None
    if args.vr in ("orbit", "vuer"):
        ctl_port = int(rig.get("vr", {}).get("control_port", 8201))
        try:
            from ..control_server import ControlServer
            ctl = ControlServer(engine, ctl_port)
            print(f"[hw] engine control channel on {ctl.endpoint} — dashboard CALIBRATE works")
        except OSError as e:
            print(f"[hw] engine control channel disabled (port {ctl_port}): {e} — "
                  "CALIBRATE button will not reach this process")
    src.start()
    recorder = SessionRecorder() if args.record else None
    push_calib = hasattr(src, "set_calib")   # in-headset calibration countdown (Vuer)
    hw_sink = getattr(sink, "hw", sink)       # the HardwareSink under the render tee
    TELEMETRY_PATH.parent.mkdir(exist_ok=True)
    last_telem = 0.0
    if rig.get("mapping", {}).get("swap_sides"):
        print("[hw] swap_sides ON: right arm does the LEFT hand's motion (and vice versa)")

    period = 1.0 / hz
    sides = list(getattr(hw_sink, "sides", rig.get("hardware", {}).get("sides", [])))

    def emit(frame, engaged, t):
        """Publish render.state and (rate-limited ~5 Hz) motor telemetry for one tick —
        shared by the teleop loop and the between-takes home glide."""
        nonlocal last_telem
        if render is not None:
            try:
                render.publish(engine, frame, engaged, 1.0 / period, t)
            except Exception:
                pass
        if hasattr(hw_sink, "telemetry") and t - last_telem > 0.2:
            last_telem = t
            try:
                TELEMETRY_PATH.write_text(json.dumps({
                    "wall": time.time(),
                    "engaged": {s: bool(engaged.get(s, False)) for s in engaged},
                    "arms": hw_sink.telemetry(),
                }))
            except Exception:
                pass

    cycle = 0
    try:
        while True:
            # --loop-home: a non-looping replay that has played through once glides the
            # arms back to rest (energized, through the shaper), dwells, then replays —
            # the next take WAITS for the home transition. STOP releases torque mid-glide.
            if args.loop_home and getattr(src, "exhausted", False):
                cycle += 1
                print(f"[hw] take {cycle} done — gliding arms home, then replaying", flush=True)
                glide_arms_home(
                    engine, sink, sides, period,
                    rate_cap=float(rig.get("hardware", {}).get("rate_limit", 1.2)),
                    dwell_s=home_dwell_s,
                    on_tick=lambda tt: emit(None, {s: False for s in sides}, tt),
                )
                if args.cycles and cycle >= args.cycles:
                    print(f"[hw] completed {cycle} take(s) — stopping", flush=True)
                    break
                src.rewind()
                continue

            t = time.monotonic()   # shared clock with source stamps + supervisor staleness
            frame = src.latest()
            engaged = supervisor.update(frame, t)   # swap_sides applied inside engine.tick
            if recorder is not None and frame is not None:
                recorder.add(frame, engaged, t)
            engine.tick(frame, engaged, t)
            emit(frame, engaged, t)
            if push_calib:
                src.set_calib(engine.calib_status)
            dt = period - (time.monotonic() - t)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\nstopping — releasing torque")
    finally:
        if ctl is not None:
            ctl.close()
        supervisor.estop()
        src.stop()
        sink.close()
        if recorder is not None:
            recorder.save(args.record)
            print(f"recorded {len(recorder)} frames -> {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
