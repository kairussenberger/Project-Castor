"""ReplayConductor — error-governed playback clock for HARDWARE replay.

Why this exists (2026-06-14 forensics): replaying a recorded trajectory on the
real YAM always aborted. A recording is a stream of ABSOLUTE IK targets; played
at the tape's own rate they outrun the gravity-loaded, speed-derated, shaped arm
(hardware.rate_limit 1.2 rad/s, ik.max_vel derated ×0.35). The command then sits
metres ahead of the metal, the RuntimeGuard TRACKING gap (rig safety.runtime
max_tracking_error) opens past its ceiling, the guard trips, torque releases, and
the run loop EXITS — unrecoverable. The cure is the same discipline jog_arms.py's
may_move() already uses for held keys: never let the commanded target run away
from the MEASURED pose. Here the dial is the replay CLOCK, not a keyboard.

This object is PURE LOGIC (no hardware imports, fully unit-testable). It does two
things, both driven by the live tracking error reported back from the sink:

  1. CONVERGE-BEFORE-FOLLOW. A tape rarely starts at the arm's rest pose, so the
     very first frame is itself a teleport. On start() the clock FREEZES at the
     trajectory start (rate scale 0.0) and the arm is allowed to GLIDE onto that
     first pose under its own shaper. Only once the worst-joint gap has stayed
     within `converge_tol` (rad) for `converge_hold_s` does playback begin. If it
     never converges within `converge_timeout_s` we follow anyway (flagged, never
     aborted) — a stuck convergence must not strand the operator.

  2. ERROR-GOVERNED PLAYBACK. While following, the clock RATE SCALE is set from
     the worst per-joint tracking RATIO (gap[j] / max_tracking_error[j], the exact
     quantity the guard trips on). At/below `soft_frac` the tape plays at full
     rate (scale 1); from soft_frac to `hard_frac` the scale ramps linearly to 0;
     above hard_frac the clock is frozen. The metal thus pulls the clock along at
     exactly the rate it can physically track, and the ratio is held well under
     1.0 — the trip can never be reached by lag alone. No telemetry yet (NaN/None
     ratio) means "no reason to stall" → scale 1.

State machine: "converge" → "follow", with note_trip() (a GENUINE guard trip
caught + recovered upstream) re-entering "converge" so playback re-glides onto the
recovered measured pose. status() is JSON-native for the dashboard (status.hw.replay).
"""
from __future__ import annotations

import math

import numpy as np


# Tunables live in the rig 'replay:' block; these defaults match config/rig.yaml
# so the conductor is usable with cfg=None (tests, ad-hoc).
_DEFAULTS = {
    "converge_tol": 0.10,       # rad — worst-joint gap that counts as converged
    "converge_hold_s": 0.4,     # s — gap must stay converged this long before following
    "converge_timeout_s": 20.0,  # s — give up waiting and follow anyway (flagged)
    "soft_frac": 0.5,           # ratio at/below which the tape plays full rate
    "hard_frac": 0.85,          # ratio at/above which the clock is frozen
}


class ReplayConductor:
    """Paces a replay clock from the live tracking error. See module docstring.

    Constructor:
      max_tracking_error: per-joint array (rig safety.runtime.max_tracking_error) —
          the SAME ceiling the RuntimeGuard trips on; the governor keeps the ratio
          below it.
      sides: the arms wired this session (informational; the caller reduces its
          telemetry to a single worst gap/ratio across these before calling update).
      cfg: the rig 'replay' block (or None → _DEFAULTS).

    The convergence test is judged on the actual worst-joint GAP in rad
    (`<= converge_tol`), NOT on the ratio — a wrist joint with a 3.0 rad ceiling
    would otherwise read "converged" with a half-radian gap. So update() takes BOTH
    the worst gap (rad) and the worst ratio; govern() (playback) uses the ratio.
    """

    def __init__(self, *, max_tracking_error, sides=("right",), cfg: dict | None = None):
        self.max_track = np.asarray(max_tracking_error, dtype=float).reshape(-1)
        self.sides = tuple(sides)
        c = {**_DEFAULTS, **(cfg or {})}
        self.converge_tol = float(c["converge_tol"])
        self.converge_hold_s = float(c["converge_hold_s"])
        self.converge_timeout_s = float(c["converge_timeout_s"])
        self.soft_frac = float(c["soft_frac"])
        self.hard_frac = float(c["hard_frac"])
        if not (self.hard_frac > self.soft_frac):
            raise ValueError("replay.hard_frac must be > replay.soft_frac")
        # State machine.
        self.phase = "converge"
        self._t_enter: float | None = None       # when this phase was entered
        self._t_converged: float | None = None    # when the gap first fell within tol
        self.timed_out = False                     # converge_timeout_s elapsed at least once
        self._last_scale = 0.0                     # last rate scale returned (for status)
        self._last_ratio = float("nan")            # last worst ratio seen (for status)

    # ---- lifecycle --------------------------------------------------------- #
    def start(self, t: float) -> None:
        """Enter 'converge' at time t — clock frozen at the trajectory start."""
        self.phase = "converge"
        self._t_enter = float(t)
        self._t_converged = None
        self._last_scale = 0.0
        self.timed_out = False        # reflects THIS converge attempt, not run history

    def note_trip(self, t: float) -> None:
        """A genuine GuardTrip was caught + recovered upstream (sink.recover()
        re-seeded the shapers from the measured pose). Re-enter 'converge' so the
        clock freezes and playback re-glides onto the recovered pose from rest —
        never resume mid-flight straight back into the gap that tripped."""
        self.start(t)

    # ---- playback governor ------------------------------------------------- #
    def govern(self, ratio) -> float:
        """Map the worst tracking RATIO to a clock rate scale in [0, 1]:
        1 at/below soft_frac, linear down to 0 at hard_frac, 0 above. A NaN/None
        ratio (no telemetry yet) returns 1.0 — absence of a reason to stall must
        not stall the clock (the converge phase, not this, owns the initial freeze)."""
        if ratio is None:
            return 1.0
        r = float(ratio)
        if not math.isfinite(r):
            return 1.0
        if r <= self.soft_frac:
            return 1.0
        if r >= self.hard_frac:
            return 0.0
        return float((self.hard_frac - r) / (self.hard_frac - self.soft_frac))

    # ---- per-tick ---------------------------------------------------------- #
    def update(self, worst_gap: float, worst_ratio, t: float) -> float:
        """Advance the state machine one tick and return the replay-clock RATE
        SCALE to apply now.

          worst_gap   — worst-joint |commanded − measured| in RAD across the wired
                        sides (judges convergence; clean of the per-joint ceilings).
          worst_ratio — worst-joint gap/max_tracking_error across the wired sides
                        (governs playback; the quantity the guard trips on). NaN/None
                        when there is no telemetry yet.
          t           — monotonic seconds (the loop clock).

        CONVERGE: scale 0.0 (clock frozen at the start pose) until the gap has
        stayed ≤ converge_tol for converge_hold_s → switch to FOLLOW. If
        converge_timeout_s elapses first, switch to FOLLOW anyway with
        `timed_out` set (do NOT abort).
        FOLLOW: scale = govern(worst_ratio).
        """
        self._last_ratio = (float("nan") if worst_ratio is None
                            or not math.isfinite(float(worst_ratio)) else float(worst_ratio))
        if self._t_enter is None:
            self._t_enter = float(t)
        if self.phase == "converge":
            g = float(worst_gap) if (worst_gap is not None
                                     and math.isfinite(float(worst_gap))) else float("inf")
            if g <= self.converge_tol:
                if self._t_converged is None:
                    self._t_converged = float(t)
                elif (t - self._t_converged) >= self.converge_hold_s:
                    self.phase = "follow"
                    self._t_enter = float(t)
            else:
                self._t_converged = None              # gap re-opened — restart the hold
            # Timeout is a fallback, NOT an abort: follow from wherever we are.
            if self.phase == "converge" and (t - self._t_enter) >= self.converge_timeout_s:
                self.phase = "follow"
                self.timed_out = True
                self._t_enter = float(t)
            self._last_scale = (self.govern(worst_ratio) if self.phase == "follow" else 0.0)
            return self._last_scale
        # FOLLOW
        self._last_scale = self.govern(worst_ratio)
        return self._last_scale

    # ---- introspection ----------------------------------------------------- #
    @property
    def converged(self) -> bool:
        """True once playback has begun (left the converge phase)."""
        return self.phase != "converge"

    def status(self) -> dict:
        """JSON-native snapshot for the dashboard (status.hw.replay). ratio is
        emitted as None when not-a-number so json.dumps stays valid (NaN is not
        legal JSON)."""
        r = self._last_ratio
        return {
            "phase": str(self.phase),
            "scale": round(float(self._last_scale), 4),
            "ratio": (None if not math.isfinite(r) else round(float(r), 4)),
            "converged": bool(self.converged),
            "timed_out": bool(self.timed_out),
        }
