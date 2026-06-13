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
import sys
import time

from ..config import load_rig
from ..engine import TeleopEngine
from ..safety.clutch import GestureClutch, RecordedClutch
from ..safety.supervisor import Supervisor
from ..vr.ingest import make_source
from ..vr.replay import SessionRecorder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vr", choices=["vuer", "orbit", "fake", "replay"], default="orbit")
    ap.add_argument("replay_path", nargs="?", help="session .npz when --vr replay")
    ap.add_argument("--clutch", choices=["gesture", "recorded"], default="gesture",
                    help="hardware engage policy (default: gesture; recorded is for replay sessions)")
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
    args = ap.parse_args()

    if sys.platform == "darwin":
        print("WARNING: real YAM control needs Linux/SocketCAN; macOS can't run the CAN loop.")

    rig = load_rig()
    hw = rig.setdefault("hardware", {})
    if args.sides:
        hw["sides"] = [s.strip() for s in args.sides.split(",") if s.strip()]
    if args.no_hands:
        hw["use_hands"] = False
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
    clutch = RecordedClutch(src) if args.clutch == "recorded" else GestureClutch()

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
    src.start()
    recorder = SessionRecorder() if args.record else None
    push_calib = hasattr(src, "set_calib")   # in-headset calibration countdown (Vuer)

    period = 1.0 / hz
    try:
        while True:
            t = time.monotonic()   # shared clock with source stamps + supervisor staleness
            frame = src.latest()
            engaged = supervisor.update(frame, t)
            if recorder is not None and frame is not None:
                recorder.add(frame, engaged, t)
            engine.tick(frame, engaged, t)
            if render is not None:
                try:
                    render.publish(engine, frame, engaged, 1.0 / period, t)
                except Exception:
                    pass
            if push_calib:
                src.set_calib(engine.calib_status)
            dt = period - (time.monotonic() - t)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\nstopping — releasing torque")
    finally:
        supervisor.estop()
        src.stop()
        sink.close()
        if recorder is not None:
            recorder.save(args.record)
            print(f"recorded {len(recorder)} frames -> {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
