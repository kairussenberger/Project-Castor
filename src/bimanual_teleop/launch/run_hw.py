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

import numpy as np

from ..config import load_rig
from ..engine import TeleopEngine
from ..hardware import active_sides
from ..safety.clutch import GestureClutch, RecordedClutch
from ..safety.replay_drive import ReplayConductor
from ..safety.runtime_guard import GuardTrip, effective_rate_limit
from ..safety.supervisor import Supervisor
from ..vr.ingest import make_source
from ..vr.replay import SessionRecorder

# Bounded replay-trip recovery: a genuine guard trip is RECOVERABLE (reseed the
# shapers from the measured pose + re-converge), but more than _RECOVER_MAX trips
# inside _RECOVER_WINDOW_S is a real fault (obstacle / stale map), not transient
# lag — latch a fatal stop instead of pulsing torque on/off forever.
_RECOVER_MAX = 3
_RECOVER_WINDOW_S = 10.0


def _worst_track(tele: dict | None, sides, max_track) -> tuple[float, float]:
    """Reduce a HardwareSink.telemetry() snapshot to (worst_gap_rad, worst_ratio)
    across the wired `sides`, where ratio = gap[j]/max_tracking_error[j] (the exact
    quantity the RuntimeGuard trips on). NaN/NaN when no per-arm gap is available
    yet (first ticks, telemetry read skipped) — the conductor reads that as "no
    reason to stall". `max_track` is the per-joint tracking ceiling (scalar or list)."""
    mt = np.atleast_1d(np.asarray(max_track, dtype=float))
    worst_gap = float("nan")
    worst_ratio = float("nan")
    arms = (tele or {}).get("arms", {}) or {}
    for s in sides:
        gap = (arms.get(s) or {}).get("gap")
        if gap is None:
            continue
        g = np.asarray(gap, dtype=float).reshape(-1)
        n = min(len(g), len(mt) if mt.size > 1 else len(g))
        ceil = mt if mt.size > 1 else np.full(len(g), float(mt.reshape(-1)[0]))
        ratio = g[:n] / ceil[:n]
        gmax = float(np.max(g[:n])) if n else float("nan")
        rmax = float(np.max(ratio)) if n else float("nan")
        if np.isnan(worst_gap) or gmax > worst_gap:
            worst_gap = gmax
        if np.isnan(worst_ratio) or rmax > worst_ratio:
            worst_ratio = rmax
    return worst_gap, worst_ratio


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
    _ept = hw.get('engage_pose_tol', 0.15)
    _ept_str = (f"±{float(_ept):.2f} rad" if np.ndim(_ept) == 0
                else "±[" + ",".join(f"{float(x):.2f}" for x in _ept) + "] rad/joint")
    print(f"[hw] arms: {hw.get('sides', ['left', 'right'])}  hands: {hw.get('use_hands', True)}  "
          f"(rig hardware.sides / --sides; rest-pose gate {_ept_str})")
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
    eff_rate = effective_rate_limit(rig)
    ceiling = rig.get("safety", {}).get("runtime", {}).get("hard_max_joint_speed")
    if ceiling is not None and eff_rate < float(rig["hardware"].get("rate_limit", 1.2)) - 1e-9:
        print(f"[hw] HARD SPEED CEILING {ceiling} rad/s -> shaper clamped to {eff_rate:.2f} rad/s")
    print(f"[hw] runtime guard: {'ON' if rig.get('safety', {}).get('runtime', {}).get('enabled', True) else 'OFF'} "
          f"(tracking/thermal/overcurrent/loop-stall — trip releases torque)")
    src = make_source(rig)
    # A recording that embeds its session's calibration MUST replay THROUGH it —
    # raw ORBIT frames are only meaningful with their fit (anchors move metres
    # between sessions). run_teleop does this (run_teleop.py:95-96); the hardware
    # path skipped it, so replays ran through IDENTITY calib, targets landed metres
    # off, and the tracking guard tripped on the first tick. Inject it BEFORE the
    # engine is built (TeleopEngine reads vr._embedded_calib in __init__).
    if args.vr == "replay" and getattr(src, "calib", None):
        rig["vr"]["_embedded_calib"] = src.calib
        stamp = (src.calib.get("meta") or {}).get("stamp", "no stamp")
        print(f"[hw] replay: applying the calibration embedded in the recording ({stamp})")
    elif args.vr == "replay":
        print("[hw] replay: recording has NO embedded calibration — replaying through "
              "IDENTITY (old tape / synthetic fixture); targets may be off")
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

            def telemetry(self):
                return self.hw.telemetry() if hasattr(self.hw, "telemetry") else None

            def release_all_torque(self):
                try:
                    self.hw.release_all_torque()
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

    # Localhost command channel (dashboard CALIBRATE / reanchor button → running
    # engine), mirroring run_teleop.py:125-128. Best-effort like the render mirror:
    # a busy port or missing dep must NEVER block the hardware loop.
    ctl = None
    try:
        from ..control_server import ControlServer
        ctl = ControlServer(engine, int(rig.get("vr", {}).get("control_port", 8201)))
        print(f"[hw] engine control channel on {ctl.endpoint} (dashboard CALIBRATE/reanchor)")
    except Exception as e:
        print(f"[hw] engine control channel disabled ({e}) — hardware loop unaffected")

    # HARDWARE REPLAY governor: pace the replay clock from the live tracking error
    # so the absolute IK targets never outrun the gravity-loaded arm (the trip that
    # killed every prior replay), and auto-converge onto the trajectory start first.
    runtime_cfg = rig.get("safety", {}).get("runtime", {})
    sides = active_sides(rig)
    max_track = runtime_cfg.get("max_tracking_error", 0.35)
    conductor = None
    if args.vr == "replay":
        conductor = ReplayConductor(max_tracking_error=max_track, sides=sides,
                                    cfg=rig.get("replay", {}))

    src.start()
    # Position the tape at the first ENGAGED frame (skip a long idle pre-roll) and
    # freeze the conductor on the start pose so the arm glides onto it before
    # following. first_engaged_time/set_clock exist only on ReplaySource.
    if conductor is not None and hasattr(src, "set_clock"):
        t0 = time.monotonic()
        start_t = src.first_engaged_time(sides)
        if start_t is None:
            start_t = float(src.t[0]) if len(getattr(src, "t", [])) else 0.0
        src.set_clock(start_t)
        src.hold()                                # clock frozen until converged
        conductor.start(t0)
        print(f"[hw] replay conductor: converge→follow, error-governed clock; "
              f"start at recorded t={start_t:.2f}s")

    recorder = SessionRecorder() if args.record else None
    push_calib = hasattr(src, "set_calib")   # in-headset calibration countdown (Vuer)

    period = 1.0 / hz
    # The HardwareSink lives one layer down through the render mirror Tee — reach it
    # for recover() (re-seed shapers + re-arm guard after a genuine trip).
    hw_sink = getattr(sink, "hw", sink)
    trip_times: list[float] = []     # genuine-trip timestamps, for bounded recovery
    try:
        while True:
            t = time.monotonic()   # shared clock with source stamps + supervisor staleness
            frame = src.latest()
            engaged = supervisor.update(frame, t)
            if recorder is not None and frame is not None:
                recorder.add(frame, engaged, t)
            # On REPLAY a GuardTrip is RECOVERABLE: re-seed the shapers from the
            # measured pose, freeze the conductor so playback re-converges, and keep
            # running. The guard stays fully ACTIVE — we recover from a genuine trip,
            # never disable it. LIVE keeps the original fatal behaviour (re-raises).
            try:
                engine.tick(frame, engaged, t)
            except GuardTrip as trip:
                if conductor is None:
                    raise
                trip_times.append(t)
                trip_times[:] = [tt for tt in trip_times if t - tt <= _RECOVER_WINDOW_S]
                if len(trip_times) > _RECOVER_MAX:
                    print(f"[hw] replay: {len(trip_times)} guard trips in "
                          f"{_RECOVER_WINDOW_S:.0f}s — a real fault, not lag. Latching a "
                          f"stop (torque already released).", flush=True)
                    raise trip               # → outer fatal handler + estop
                print(f"[hw] replay: GUARD TRIP caught ({trip}) — recovering "
                      f"({len(trip_times)}/{_RECOVER_MAX} in {_RECOVER_WINDOW_S:.0f}s): "
                      f"reseed shapers from measured pose, re-converge", flush=True)
                try:
                    hw_sink.recover()
                except Exception as e:
                    print(f"[hw] recover() failed ({e}) — aborting", flush=True)
                    raise trip
                conductor.note_trip(t)
                if hasattr(src, "hold"):
                    src.hold()
            # Error-governed clock: measure how far the command leads the metal and
            # scale the replay rate so the gap stays well under the trip ceiling.
            try:
                tele = sink.telemetry() if hasattr(sink, "telemetry") else None
            except Exception:
                tele = None        # telemetry is diagnostic — never let it kill the loop
            if conductor is not None:
                worst_gap, worst_ratio = _worst_track(tele, sides, max_track)
                scale = conductor.update(worst_gap, worst_ratio, t)
                if hasattr(src, "set_rate_scale"):
                    src.set_rate_scale(scale)
            if render is not None:
                try:
                    hw = dict(tele) if isinstance(tele, dict) else (tele or {})
                    if conductor is not None and isinstance(hw, dict):
                        hw = {**hw, "replay": conductor.status()}   # → status.hw.replay
                    render.publish(engine, frame, engaged, 1.0 / period, t, hw=hw or None)
                except Exception:
                    pass
            if push_calib:
                src.set_calib(engine.calib_status)
            dt = period - (time.monotonic() - t)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\nstopping — releasing torque")
    except GuardTrip as trip:
        print(f"\n*** RUNTIME SAFETY TRIP — torque released on all arms, run aborted ***\n"
              f"    {trip}\n"
              f"    Inspect the rig, then re-run. Re-check the rest pose if needed: "
              f"hw_bringup --step rest")
    finally:
        supervisor.estop()
        src.stop()
        if ctl is not None:
            ctl.close()
        sink.close()
        if recorder is not None:
            recorder.save(args.record)
            print(f"recorded {len(recorder)} frames -> {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
