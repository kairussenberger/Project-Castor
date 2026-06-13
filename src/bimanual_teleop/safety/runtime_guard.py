"""RuntimeGuard — continuous per-tick safety monitor for HARDWARE sessions.

The startup rest-pose gate (hardware.py) proves the arm is sane BEFORE the first
command. This guard proves it stays sane DURING the session, every tick, and trips
(raising GuardTrip — the caller MUST release torque) the instant something is
wrong. It complements, never replaces, the motor-side CAN watchdog and the
command shaper. Four independent checks, each from `safety.runtime` config:

  1. TRACKING — per joint, |commanded − measured| (shortest arc). A joint that
     stops following its (already shaped, already slow) command — snagged, stalled,
     collided, faulted — opens a gap. Past `max_tracking_error` for longer than
     `tracking_trip_s` (after an initial `tracking_grace_s` so the engage glide and
     normal gravity-sag lag don't trip) ⇒ TRIP.
  2. THERMAL — any motor's MOSFET temperature over `max_motor_temp_c` ⇒ TRIP
     (thermal behaviour was an explicit hardware known-unknown).
  3. OVERCURRENT — any motor's |effort| over `max_motor_current` ⇒ TRIP (a joint
     pushing hard into an obstacle shows here before the gap opens).
  4. LOOP-STALL — wall-clock since the previous check exceeded `loop_stall_s`: the
     control loop hung (GIL stall, blocked CAN, a wedged upstream). The motor
     watchdog will already be releasing torque; this makes it explicit and logged.

Pure logic (no hardware imports) so the whole trip matrix is unit-tested without a
robot. HardwareSink feeds it measured state + motor telemetry each tick.
"""
from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi


class GuardTrip(RuntimeError):
    """A runtime safety guard tripped. Whoever catches this MUST release torque.

    `kind` is one of: tracking | thermal | overcurrent | loop-stall.
    """

    def __init__(self, kind: str, detail: str):
        self.kind = kind
        self.detail = detail
        super().__init__(f"SAFETY TRIP [{kind}] — {detail}")


def _shortest_arc(delta: np.ndarray) -> np.ndarray:
    """Wrap angle differences into (−π, π] so a joint near ±π doesn't read as a
    huge gap against its own wrapped command."""
    return (np.asarray(delta, dtype=float) + math.pi) % TWO_PI - math.pi


class RuntimeGuard:
    def __init__(self, cfg: dict | None, *, sides=("right",), n: int = 6):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.n = int(n)
        # PER-JOINT tracking limit (rad). A scalar broadcasts to all joints; a list
        # gives each joint its own ceiling — essential because the wrist motors are
        # WEAK (j5/j6 kp=10) and lag 80°+ behind a fast roll while the shoulder
        # (kp=80) tracks to a few degrees. One scalar that clears the wrist would be
        # blind on the shoulder, and one that guards the shoulder kills every roll.
        mt = np.asarray(cfg.get("max_tracking_error", 0.35), dtype=float)
        self.max_track = (np.full(self.n, float(mt)) if mt.ndim == 0
                          else np.asarray(mt, dtype=float).reshape(-1)[: self.n])
        self.grace_s = float(cfg.get("tracking_grace_s", 0.6))          # s after start
        self.trip_s = float(cfg.get("tracking_trip_s", 0.3))            # s debounce
        self.max_temp = float(cfg.get("max_motor_temp_c", 75.0))        # °C
        self.max_curr = float(cfg.get("max_motor_current", 8.0))        # |effort|
        self.stall_s = float(cfg.get("loop_stall_s", 0.5))             # s
        self.warn_frac = float(cfg.get("warn_fraction", 0.8))
        self.sides = tuple(sides)
        self._t0: float | None = None
        self._last_t: float | None = None
        self._track_since: dict[str, float | None] = {s: None for s in self.sides}
        self.tripped: dict | None = None        # set once a trip fires
        self.last: dict[str, dict] = {}          # latest per-side telemetry (for the dashboard)

    def reset(self) -> None:
        """Clear a latched trip and the debounce state — e.g. after the operator
        acknowledges and re-engages."""
        self._track_since = {s: None for s in self.sides}
        self._t0 = None
        self._last_t = None
        self.tripped = None

    def check(self, side: str, commanded, measured, *, effort=None, temp=None, t: float) -> list[str]:
        """Run all checks for one arm at time `t` (monotonic seconds). Returns a
        list of human-readable WARNINGS (approaching a limit); raises GuardTrip the
        moment any limit is exceeded. No-op when disabled."""
        if not self.enabled:
            return []
        warns: list[str] = []
        if self._t0 is None:
            self._t0 = t
        # 4. LOOP-STALL — a long gap since the previous check means the loop hung.
        if self._last_t is not None and (t - self._last_t) > self.stall_s:
            self._trip("loop-stall",
                       f"control loop stalled {t - self._last_t:.2f}s (> {self.stall_s:.2f}s) — "
                       "commands stopped flowing")
        self._last_t = t

        commanded = np.asarray(commanded, dtype=float).reshape(-1)[: self.n]
        measured = np.asarray(measured, dtype=float).reshape(-1)[: self.n]
        gap = np.abs(_shortest_arc(commanded - measured))
        # Each joint judged against ITS OWN limit: the joint nearest (or past) its
        # ceiling is the one that matters, not whichever has the biggest raw gap.
        ratio = gap / self.max_track
        worst = int(np.argmax(ratio))
        gmax = float(gap[worst])
        rec: dict = {"gap": [float(x) for x in gap], "gap_max": float(np.max(gap)),
                     "gap_joint": int(np.argmax(gap)) + 1}

        # 1. TRACKING — per-joint, debounced, after the initial glide grace window.
        if (t - self._t0) > self.grace_s:
            if ratio[worst] > 1.0:
                if self._track_since[side] is None:
                    self._track_since[side] = t
                elif (t - self._track_since[side]) > self.trip_s:
                    self._trip("tracking",
                               f"{side} j{worst + 1} not following: gap "
                               f"{math.degrees(gmax):.0f}° > {math.degrees(float(self.max_track[worst])):.0f}° "
                               f"held > {self.trip_s:.2f}s (snag / stall / collision?)")
            else:
                self._track_since[side] = None
                if ratio[worst] > self.warn_frac:
                    warns.append(f"{side} j{worst + 1} tracking {math.degrees(gmax):.0f}°")

        # 2. THERMAL
        if temp is not None:
            temp = np.asarray(temp, dtype=float).reshape(-1)
            rec["temp"] = [float(x) for x in temp]
            tj = int(np.argmax(temp))
            tmax = float(temp[tj])
            rec["temp_max"] = tmax
            if tmax > self.max_temp:
                self._trip("thermal", f"{side} j{tj + 1} motor {tmax:.0f}°C > {self.max_temp:.0f}°C")
            elif tmax > self.warn_frac * self.max_temp:
                warns.append(f"{side} j{tj + 1} {tmax:.0f}°C")

        # 3. OVERCURRENT (effort proxy)
        if effort is not None:
            effort = np.asarray(effort, dtype=float).reshape(-1)
            rec["effort"] = [float(x) for x in effort]
            cj = int(np.argmax(np.abs(effort)))
            cmax = float(abs(effort[cj]))
            rec["effort_max"] = cmax
            if cmax > self.max_curr:
                self._trip("overcurrent",
                           f"{side} j{cj + 1} effort {cmax:.1f} > {self.max_curr:.1f} "
                           "(pushing hard into something?)")
            elif cmax > self.warn_frac * self.max_curr:
                warns.append(f"{side} j{cj + 1} effort {cmax:.1f}")

        self.last[side] = rec
        return warns

    def _trip(self, kind: str, detail: str):
        self.tripped = {"kind": kind, "detail": detail}
        raise GuardTrip(kind, detail)


def effective_rate_limit(rig: dict) -> float:
    """The per-joint speed cap actually used by the hardware shaper: the configured
    `hardware.rate_limit` clamped DOWN to the absolute ceiling
    `safety.runtime.hard_max_joint_speed`. No config, CLI flag, or dashboard control
    can make the arm move faster than the ceiling — it is the floor of all speed
    knobs and exists so 'never faster than X' holds no matter what asks otherwise."""
    hw = rig.get("hardware", {})
    rate = float(hw.get("rate_limit", 1.2))
    ceiling = rig.get("safety", {}).get("runtime", {}).get("hard_max_joint_speed", None)
    if ceiling is not None:
        ceiling = float(ceiling)
        if math.isfinite(ceiling) and ceiling > 0:
            return min(rate, ceiling)
    return rate
