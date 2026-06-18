"""Run the full teleop pipeline HEADLESS, publishing robot state to Unity renderers.
No MuJoCo, no local viewer, plain `python`.

    uv run python -m bimanual_teleop.launch.run_teleop --vr fake             # synthetic operator
    uv run python -m bimanual_teleop.launch.run_teleop --vr orbit            # real Quest (ORBIT/NetMQ over adb)
    uv run python -m bimanual_teleop.launch.run_teleop --vr orbit --record session.npz
    uv run python -m bimanual_teleop.launch.run_teleop --vr replay FILE.npz  # replay a recorded session

Unity can consume either `render.state` on the ZMQ bus or newline-delimited JSON on
`vr.unity_json_endpoint`. The ORBIT Quest app PUSHes hand poses back on the ORBIT
ports (vr/orbit_source). See docs/UNITY_BRIDGE.md. Same TeleopEngine + controllers
as the hardware path (launch/run_hw.py); only the sink differs.
"""
from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import time

from loop_rate_limiters import RateLimiter

from ..bus import topics
from ..config import load_rig
from ..engine import TeleopEngine
from ..logging_utils import RateMeter, get_logger
from ..render_sink import RenderSink
from ..safety.clutch import AlwaysOn, GestureClutch, RecordedClutch
from ..safety.supervisor import Supervisor
from ..vr.ingest import make_source
from ..vr.replay import SessionRecorder

log = get_logger("run_teleop")


def _adb_reverse_render(ports: list[int]) -> None:
    """Let the Quest reach the PC's render publishers over USB (Quest localhost:port →
    PC localhost:port), the same trick the ORBIT ingest uses for the pose ports."""
    if not shutil.which("adb"):
        log.warning("adb not found — set up `adb reverse tcp:<port> tcp:<port>` for render ports %s yourself", ports)
        return
    ok = 0
    for port in ports:
        try:
            r = subprocess.run(["adb", "reverse", f"tcp:{port}", f"tcp:{port}"], capture_output=True, timeout=5)
            ok += (r.returncode == 0)
        except (subprocess.SubprocessError, OSError):
            pass
    log.info("adb reverse set for %d/%d render ports: %s", ok, len(ports), ports)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vr", choices=["fake", "vuer", "orbit", "replay"], default="fake",
                    help="pose transport (default: fake synthetic operator)")
    ap.add_argument("replay_path", nargs="?", help="session .npz when --vr replay")
    ap.add_argument("--clutch", choices=["always", "gesture", "recorded"], default=None,
                    help="engage policy: always-on, pinch-gesture, or recorded replay decisions")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = run until Ctrl+C)")
    ap.add_argument("--calib-seconds", type=float, default=None,
                    help="override vr.calib_seconds for deterministic smoke/replay runs")
    ap.add_argument("--loop", action="store_true",
                    help="with --vr replay: loop the recording forever (demos/dashboard)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="replay time-stretch: 0.2 plays a recording 5x slower (replay only)")
    ap.add_argument("--record", metavar="PATH", default=None,
                    help="write VR frames + engage state to a replayable .npz session")
    ap.add_argument("--viz", action="store_true",
                    help="open the local Rerun 3D viewer (no Unity/headset needed): robot arms, "
                         "commanded vs achieved EE, operator wrist overlay, clutch/error plots")
    ap.add_argument("--viz-save", metavar="PATH", default=None,
                    help="log the same Rerun scene to a .rrd file instead of opening the viewer "
                         "(inspect later with `rerun PATH`)")
    ap.add_argument("--swap-sides", action="store_true",
                    help="drive each recorded hand with the OPPOSITE arm (right arm does the "
                         "left hand's motion and vice versa)")
    ap.add_argument("--mirror-fb", action="store_true",
                    help="mirror motion front↔back only (mapping.mirror_forward)")
    ap.add_argument("--mirror-lr", action="store_true",
                    help="mirror motion left↔right only (mapping.mirror_lateral)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    rig = load_rig()
    rig.setdefault("vr", {})["transport"] = args.vr
    if args.swap_sides:
        rig.setdefault("mapping", {})["swap_sides"] = True
    if args.mirror_fb or args.mirror_lr:
        m = rig.setdefault("mapping", {})
        m["mirror_forward"] = bool(args.mirror_fb)
        m["mirror_lateral"] = bool(args.mirror_lr)
    if args.calib_seconds is not None:
        rig["vr"]["calib_seconds"] = max(0.0, float(args.calib_seconds))
    if args.debug:
        rig["vr"]["debug"] = True
    if args.vr == "replay":
        if not args.replay_path:
            ap.error("--vr replay needs a session file: run_teleop --vr replay session.npz")
        rig["vr"]["replay_path"] = args.replay_path
        rig["vr"]["replay_speed"] = float(args.speed)
        rig["vr"]["replay_loop"] = bool(args.loop)

    src = make_source(rig)
    # A recording that embeds its session's calibration replays THROUGH it —
    # raw ORBIT frames are only meaningful with their fit (anchors move metres
    # between sessions). Old recordings/synthetic fixtures have none → identity.
    if args.vr == "replay" and getattr(src, "calib", None):
        rig["vr"]["_embedded_calib"] = src.calib
        log.info("replay: applying the calibration embedded in the recording (%s)",
                 (src.calib.get("meta") or {}).get("stamp", "no stamp"))
    clutch_name = args.clutch or ("recorded" if args.vr == "replay" else "always")
    if clutch_name == "gesture":
        clutch = GestureClutch()
    elif clutch_name == "recorded":
        clutch = RecordedClutch(src)
    else:
        clutch = AlwaysOn()
    sink = RenderSink(rig)
    engine = TeleopEngine(rig, sink)
    supervisor = Supervisor(rig, clutch)
    viz = None
    if args.viz or args.viz_save:
        from ..viz.live_viz import TeleopViz
        viz = TeleopViz(rig, spawn=bool(args.viz), save_path=args.viz_save)

    if args.vr == "orbit":
        render_ports = []
        if sink.zmq_enabled:
            render_ports.append(int(sink.endpoint.rsplit(":", 1)[1]))
        if sink.json_enabled and sink.json_endpoint:
            render_ports.append(int(sink.json_endpoint.rsplit(":", 1)[1]))
        if render_ports:
            _adb_reverse_render(render_ports)
    # Localhost command channel (dashboard CALIBRATE button → running engine).
    # Best-effort: a busy port must never block teleop itself.
    ctl = None
    ctl_port = int(rig.get("vr", {}).get("control_port", 8201))
    try:
        from ..control_server import ControlServer
        ctl = ControlServer(engine, ctl_port)
    except OSError as e:
        log.warning("engine control channel disabled (port %d): %s", ctl_port, e)

    src.start()
    log.info("teleop running | transport=%s | clutch=%s | render → %s | unity-json → %s | control → %s | Ctrl+C to stop",
             args.vr,
             clutch_name,
             sink.endpoint if sink.zmq_enabled else "disabled",
             sink.json_endpoint if sink.json_enabled else "disabled",
             ctl.endpoint if ctl is not None else "disabled")

    # Engines are routinely children of background/non-interactive shells, which
    # inherit SIGINT=SIG_IGN — Python then never installs KeyboardInterrupt, so a
    # "graceful" stop (dashboard STOP, pkill -INT) silently no-ops until someone
    # escalates to SIGKILL and the recording is lost. Install our own handlers
    # unconditionally; SIGTERM gets the same graceful save path.
    def _request_stop(signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    rate = RateMeter()
    recorder = SessionRecorder() if args.record else None
    limiter = RateLimiter(frequency=float(rig["control"]["arm_hz"]), warn=False)
    t0 = time.monotonic()
    _t_prev = None
    try:
        while True:
            t = time.monotonic()
            if _t_prev is not None:
                rate.update(t - _t_prev)
            _t_prev = t
            frame = src.latest()
            engaged = supervisor.update(frame, t)   # swap_sides applied inside engine.tick
            if recorder is not None and frame is not None:
                recorder.add(frame, engaged, t)
            engine.tick(frame, engaged, t)
            sink.publish(engine, frame, engaged, rate.hz, t)
            if viz is not None:
                viz.tick(frame=frame, engine=engine, engaged=engaged, hz=rate.hz, t=t - t0)
            if args.duration and (t - t0) >= args.duration:
                break
            limiter.sleep()
    except KeyboardInterrupt:
        print()
    finally:
        # FIRST thing in teardown: a second INT/TERM (uv forwarding, impatient
        # killer) must not abort mid-teardown and lose the recording save.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        src.stop()
        if ctl is not None:
            ctl.close()
        sink.close()
        if recorder is not None:
            calib = engine.calib_result.payload() if engine.calib_result is not None else None
            recorder.save(args.record, calib=calib)
            log.info("recorded %d frames → %s%s", len(recorder), args.record,
                     " (session calibration embedded)" if calib else "")
        log.info("stopped (ran %.1fs, ~%.0f Hz)", time.monotonic() - t0, rate.hz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
