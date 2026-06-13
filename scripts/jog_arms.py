#!/usr/bin/env python
"""Keyboard jog — drive the YAM arms MANUALLY, no headset, for sim→real checks.

Runs the same ArmIK + sinks as teleop, so a keypress in sim and a keypress on the
Linux host produce the same joint commands (hardware additionally passes through
the JointCommandShaper). Watch it live on the dashboard / Unity / --viz while
jogging.

    uv run python scripts/jog_arms.py                    # render sink (sim, default)
    uv run python scripts/jog_arms.py --sink hw          # REAL arms (Linux host)

Keys:
    a / d      jog selected joint - / +        1..6   select joint
    [ / ]      halve / double step             TAB    switch side (left/right)
    w / s      EE forward / back               z / c  EE left / right
    r / f      EE up / down                    h      go home (rest pose)
    m          print MEASURED pose (hw)        p      print state
    SPACE / x  PANIC: torque off NOW (limp)    q, ESC quit
    (arrows still work too: UP/DOWN jog joint, LEFT/RIGHT select)
"""
from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.config import SIDES, load_rig                  # noqa: E402
from bimanual_teleop.arms.ik import ArmIK                           # noqa: E402
from bimanual_teleop.safety.runtime_guard import GuardTrip          # noqa: E402
from bimanual_teleop.vr.frames import SE3, quat_to_R                # noqa: E402


class _ArmShim:
    """Just enough controller surface for RenderSink.build_state()."""

    def __init__(self, rig: dict, side: str, ik: ArmIK):
        self.ik = ik
        self.base_R = quat_to_R(rig["arms"][side]["base_quat"])
        self.base_pos = np.asarray(rig["arms"][side]["base_pos"], dtype=float)
        self.cmd_pos = None
        self.cmd_R = None


class _EngineShim:
    def __init__(self, rig: dict, iks: dict):
        self.arm = {s: _ArmShim(rig, s, iks[s]) for s in SIDES}
        self.calib_status = None


class JogSession:
    """Manual joint/EE jogging through the real IK — testable without a TTY."""

    def __init__(self, rig: dict, sink):
        self.rig = rig
        self.sink = sink
        self.ik = {s: ArmIK(rig, s) for s in SIDES}
        self.engine = _EngineShim(rig, self.ik)
        self.side = "right"
        self.joint = 5                      # 0-based; j6 selected by default
        self.joint_step = np.radians(3.0)
        self.ee_step = 0.015                # m per nudge
        for s in SIDES:
            self.sink.set_arm(s, self.ik[s].q)

    # ---- actions ----------------------------------------------------------- #
    def step_joint(self, direction: int) -> np.ndarray:
        ik = self.ik[self.side]
        q = ik.q
        j = self.joint
        q[j] = float(np.clip(q[j] + direction * self.joint_step,
                             ik.soft_lo[j], ik.soft_hi[j]))
        ik.seed(q)
        self._push(self.side)
        return q

    def nudge_ee(self, d_world) -> np.ndarray:
        """Move the wrist target by a world-frame delta through the two-stage IK
        (exactly the solve path teleop uses)."""
        ik = self.ik[self.side]
        shim = self.engine.arm[self.side]
        d_base = shim.base_R.T @ np.asarray(d_world, dtype=float)
        target_p = ik.fk_wrist().translation() + d_base
        target = SE3.from_rotation_and_translation(ik.fk_ee().rotation(), target_p)
        for _ in range(6):
            ik.solve(target)
        shim.cmd_pos = target_p.copy()
        shim.cmd_R = ik.fk_ee().rotation().as_matrix()
        self._push(self.side)
        return ik.q

    def home(self) -> None:
        ik = self.ik[self.side]
        ik.reset()
        self.engine.arm[self.side].cmd_pos = None
        self.engine.arm[self.side].cmd_R = None
        self._push(self.side)

    def _push(self, side: str) -> None:
        self.sink.set_arm(side, self.ik[side].q)

    def publish(self, hz: float, t: float) -> None:
        if hasattr(self.sink, "publish"):
            self.sink.publish(self.engine, None, {s: False for s in SIDES}, hz, t)

    def live_line(self, meas: float | None = None, gap: float | None = None) -> str:
        """Compact ONE-line status for the live in-place display. Kept well under
        80 cols so it never wraps (wrapping is what defeats the \\r overwrite and
        floods the screen). Shows the SELECTED joint — the thing a/d moves."""
        j = self.joint
        qj = np.degrees(self.ik[self.side].q[j])
        s = (f"[{self.side[0].upper()}] j{j+1}  cmd {qj:+7.1f}°"
             f"  step {np.degrees(self.joint_step):.1f}°")
        if meas is not None:
            arrow = "  <- MOVING" if (gap is not None and gap > 0.5) else ""
            s += f"  |  meas {meas:+7.1f}°  gap {gap:4.1f}°{arrow}"
        return s

    def status_line(self) -> str:
        """Full 6-joint pose (the `p` key, and the test). Multi-line-safe: only
        printed on its own fresh line, never as the in-place ticker."""
        q = np.degrees(self.ik[self.side].q)
        qs = " ".join(f"j{i+1}{'*' if i == self.joint else ''}={q[i]:+6.1f}" for i in range(6))
        return (f"[{self.side.upper()}] step={np.degrees(self.joint_step):.1f}°/"
                f"{self.ee_step*100:.1f}cm  {qs}")


class _Tee:
    """Hardware first; the render copy + publish are best-effort cosmetics so the
    dashboard mirrors the jog 1:1 (losing it never blocks the metal)."""

    def __init__(self, hw, render):
        self.hw = hw
        self.render = render
        self.arms = hw.arms                  # wired-side detection keeps working

    def set_arm(self, side, q):
        self.hw.set_arm(side, q)
        try:
            self.render.set_arm(side, q)
        except Exception:
            pass

    def set_hand(self, side, joints_deg):
        self.hw.set_hand(side, joints_deg)
        try:
            self.render.set_hand(side, joints_deg)
        except Exception:
            pass

    def publish(self, *a, **k):
        try:
            k.setdefault("hw", self.telemetry())
            self.render.publish(*a, **k)
        except Exception:
            pass

    def release_all_torque(self):
        try:
            self.hw.release_all_torque()
        except Exception:
            pass

    def telemetry(self):
        return self.hw.telemetry() if hasattr(self.hw, "telemetry") else None

    def close(self):
        self.hw.close()
        try:
            self.render.close()
        except Exception:
            pass


def _make_sink(kind: str, rig: dict):
    if kind == "hw":
        from bimanual_teleop.hardware import HardwareSink
        hw = HardwareSink(rig)
        try:
            from bimanual_teleop.render_sink import RenderSink
            return _Tee(hw, RenderSink(rig))
        except Exception as e:
            print(f"[jog] dashboard mirror disabled ({e}) — hardware jog unaffected")
            return hw
    from bimanual_teleop.render_sink import RenderSink
    return RenderSink(rig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sink", choices=["render", "hw"], default="render",
                    help="render = sim/Unity/dashboard preview; hw = REAL arms (Linux host)")
    ap.add_argument("--max-deg-s", type=float, default=10.0,
                    help="hw only: per-joint speed cap for this jog session "
                         "(default 10°/s — far below hardware.rate_limit; 0 = rig value)")
    args = ap.parse_args()

    rig = load_rig()
    if args.sink == "hw" and args.max_deg_s > 0:
        rig["hardware"]["rate_limit"] = float(np.radians(args.max_deg_s))
        print(f"[jog] hw speed cap {args.max_deg_s:.0f}°/s "
              f"({rig['hardware']['rate_limit']:.3f} rad/s) at the CAN shaper")
    sink = _make_sink(args.sink, rig)
    jog = JogSession(rig, sink)
    if args.sink == "hw" and jog.side not in getattr(sink, "arms", {}):
        wired = list(getattr(sink, "arms", {}))
        if wired:
            jog.side = wired[0]
            print(f"[jog] starting on the wired side: {jog.side}")
    print(__doc__.split("Keys:")[1])
    print(f"sink={args.sink}  |  watch on the dashboard at http://<host>:8180")
    print("Press 1-6 to pick a joint, then tap a / d to move it (start with 1 — the "
          "shoulder is easiest to see).")

    def draw(meas: float | None = None, gap: float | None = None) -> None:
        # ONE in-place line, truncated below the terminal width so it never wraps
        # (a wrapped line is what made \r flood the screen). \033[K clears the rest.
        sys.stdout.write("\r\033[K" + jog.live_line(meas, gap)[:79])
        sys.stdout.flush()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    draw()
    t0 = time.monotonic()
    try:
        last_meas = 0.0
        last_key = {"j": 0.0, "e": 0.0}

        def may_move(kind: str) -> bool:
            # Held keys autorepeat ~30/s but the metal tracks at ~10 deg/s (hw):
            # admit motion keys only as fast as the hardware can actually follow,
            # so the commanded target can NEVER run away from the measured pose.
            if not getattr(sink, "arms", None):
                return True                       # render-only: no physical cap
            need = (jog.joint_step / np.radians(10.0) if kind == "j"
                    else jog.ee_step / 0.05)
            nowk = time.monotonic()
            if nowk - last_key[kind] < need:
                return False
            last_key[kind] = nowk
            return True

        while True:
            t = time.monotonic() - t0
            now = time.monotonic()
            # Re-assert the current target EVERY tick: the hardware shaper is a
            # tracker and must be called continuously to glide the metal all the
            # way to the target — a single keypress-time call moves it one
            # rate-limited step and then parks, silently diverging from the page.
            for s in getattr(sink, "arms", {}):
                sink.set_arm(s, jog.ik[s].q)
            jog.publish(60.0, t)
            if getattr(sink, "arms", None) and now - last_meas > 0.25 and jog.side in sink.arms:
                meas = np.degrees(sink.arms[jog.side].state())
                cmd = np.degrees(jog.ik[jog.side].q)
                j = jog.joint
                gapj = float(abs(((meas[j] - cmd[j]) + 180.0) % 360.0 - 180.0))
                draw(meas[j], gapj)
                last_meas = now
            if not select.select([sys.stdin], [], [], 1 / 60)[0]:
                continue
            ch = os.read(fd, 1).decode(errors="ignore")
            if ch == "\x1b":
                # Arrow keys arrive as ESC [ A/B/C/D; a BARE esc (nothing pending) quits.
                # Raw os.read keeps the fd and select() consistent (buffered
                # sys.stdin.read would swallow the [A and make arrows look like ESC).
                seq = ""
                while len(seq) < 2 and select.select([fd], [], [], 0.05)[0]:
                    seq += os.read(fd, 1).decode(errors="ignore")
                if seq == "[A":
                    if not may_move("j"):
                        continue
                    jog.step_joint(+1)
                elif seq == "[B":
                    if not may_move("j"):
                        continue
                    jog.step_joint(-1)
                elif seq == "[C":
                    jog.joint = (jog.joint + 1) % 6
                elif seq == "[D":
                    jog.joint = (jog.joint - 1) % 6
                elif not seq:
                    break
                else:
                    continue
            elif ch == "q":
                break
            elif ch == "\t":
                jog.side = "left" if jog.side == "right" else "right"
            elif ch in "123456":
                jog.joint = int(ch) - 1
            elif ch == "=":
                if not may_move("j"):
                    continue
                jog.step_joint(+1)
            elif ch == "-":
                if not may_move("j"):
                    continue
                jog.step_joint(-1)
            elif ch == "[":
                jog.joint_step = max(np.radians(0.5), jog.joint_step / 2)
                jog.ee_step = max(0.002, jog.ee_step / 2)
            elif ch == "]":
                jog.joint_step = min(np.radians(12.0), jog.joint_step * 2)
                jog.ee_step = min(0.06, jog.ee_step * 2)
            elif ch == "d":
                if not may_move("j"):
                    continue
                jog.step_joint(+1)
            elif ch == "a":
                if not may_move("j"):
                    continue
                jog.step_joint(-1)
            elif ch in "wsrfzc":
                if not may_move("e"):
                    continue
                d = {"w": [-jog.ee_step, 0, 0],         # forward = world −X
                     "s": [+jog.ee_step, 0, 0],
                     "z": [0, -jog.ee_step, 0],         # EE left = world −Y
                     "c": [0, +jog.ee_step, 0],         # EE right = world +Y
                     "r": [0, 0, +jog.ee_step],
                     "f": [0, 0, -jog.ee_step]}[ch]
                jog.nudge_ee(d)
            elif ch in (" ", "x"):
                rel = getattr(sink, "release_all_torque", None)
                if rel is None:
                    rel = getattr(getattr(sink, "hw", None), "release_all_torque", None)
                if rel:
                    rel()
                sys.stdout.write("\r\033[K*** PANIC: torque released — arm is LIMP (hanging). quitting ***\n")
                break
            elif ch == "h":
                jog.home()
            elif ch == "m":
                arms = getattr(sink, "arms", {})
                if jog.side in arms:
                    meas = np.degrees(arms[jog.side].state())
                    sys.stdout.write("\r\033[K[" + jog.side + "] MEASURED  " +
                          " ".join(f"j{i+1}={meas[i]:+6.1f}" for i in range(6)) + "\n")
            elif ch == "p":
                sys.stdout.write("\r\033[K" + jog.status_line() + "\n")
            else:
                continue
            draw()
    except KeyboardInterrupt:
        pass
    except GuardTrip as trip:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print(f"\n*** RUNTIME SAFETY TRIP — torque released, jog aborted ***\n    {trip}")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()
        if hasattr(sink, "close"):
            sink.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
