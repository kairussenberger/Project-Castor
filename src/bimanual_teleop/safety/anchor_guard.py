"""Mid-session ANCHOR-JUMP guard: detect a tracking-anchor change while the
operator is driving, and stop following before the stale calibration steers the
arms somewhere wrong.

Why this exists (2026-06-11 forensics): ORBIT's pose streams live in
recenter-anchored frames whose origins are NOT trustworthy — the wrist and head
streams were measured anchored 1.35 m apart, and a recenter/desk-start moves an
anchor by an arbitrary vector. The neutral-pose calibration absorbs whatever the
anchors did AT FIT TIME; if an anchor moves AFTER the fit (recenter gesture,
headset sleep/wake, ORBIT app restart, Guardian re-detection), every absolute
target silently shifts by the anchor delta. The per-side teleport rejection in
ArmController then makes it WORSE in slow motion: it re-anchors and GLIDES the
arm onto the shifted mapping at the governed speed — wrong, smoothly.

The guard watches the BODY-RELATIVE wrist vector (`wrist_body`) — the exact
signal that steers the arms, downstream of whatever `vr.orbit_wrist_anchor`
mode reconstructs — so every anchor that can move the mapping is covered, and
anchors that cancel (the live head position cancels in the body-relative
subtraction) can never false-trip. Two channels:

  (A) JUMP while tracked: a one-sample discontinuity beyond what human motion
      explains (measured on 19 real sessions: normal per-frame deltas p99
      < 3 cm; single-frame tracking glitches of 0.2–1.5 m are COMMON) marks the
      side SUSPECT and its arm HOLDS (the engine feeds the mapper nothing). If
      the position snaps back within `return_tol`, it was a glitch — resume.
      If it persists `confirm_frames` real samples, the verdict comes from the
      OTHER hand, because both hands share the wrist stream's anchor:
        - other hand jumped COHERENTLY (same delta) → anchor event → TRIP;
        - other hand untracked → cannot disambiguate → TRIP (fail safe);
        - other hand tracked and steady → NOT an anchor — accept the move and
          resume (the side re-engages and glides, same policy as the existing
          teleport rejection).
  (B) BLACKOUT (live transports only): the whole stream dead for longer than
      `blackout_s` means the ORBIT app restarted or the headset slept — both
      reset anchors (real sessions show 6 s and 31 s mid-session blackouts;
      benign hiccups measured ≤ 0.9 s). Trip immediately; the supervisor
      already disengaged, the banner explains why the arms will not come back
      on their own.

A TRIP is latched until `reset()` — the engine locks `follow_locked`, cancels
any in-flight neutral capture (poses straddling an anchor change cannot be
fitted), and the only way back is a fresh calibration, which re-fits the new
anchors and resets this guard.

Deliberately NOT detected: a coherent shift across a both-hands tracking gap
(hands legitimately move during dropouts; channel B catches the realistic
causes), and pure anchor ROTATION about the wrist point itself (measure-zero —
real recenters translate). Clock-injected and pure; the engine owns the policy.
"""
from __future__ import annotations

import numpy as np

from ..config import SIDES

_OTHER = {"left": "right", "right": "left"}


class _SideTrack:
    """Continuity state for one side's wrist_body samples."""

    __slots__ = ("pos", "t", "ref", "ref_t", "last_p", "last_t", "n_far", "confirmed")

    def __init__(self):
        self.pos: np.ndarray | None = None    # last ACCEPTED sample
        self.t = 0.0
        self.ref: np.ndarray | None = None    # pre-jump position while suspect
        self.ref_t = 0.0
        self.last_p: np.ndarray | None = None  # most recent RAW sample (dedupe + resume point)
        self.last_t = 0.0
        self.n_far = 0
        self.confirmed = False

    @property
    def suspect(self) -> bool:
        return self.ref is not None

    @property
    def jump(self) -> np.ndarray | None:
        if self.ref is None or self.last_p is None:
            return None
        return self.last_p - self.ref

    def seed(self, p: np.ndarray, t: float) -> None:
        self.pos = np.asarray(p, dtype=float).copy()
        self.t = float(t)
        self.last_p = self.pos.copy()
        self.last_t = float(t)
        self.clear_suspect()

    def clear_suspect(self) -> None:
        self.ref = None
        self.n_far = 0
        self.confirmed = False

    def accept_new_regime(self) -> None:
        """Resolve a suspect window by accepting the post-jump position."""
        if self.last_p is not None:
            self.seed(self.last_p, self.last_t)

    def drop(self) -> None:
        self.pos = None
        self.last_p = None
        self.clear_suspect()


class AnchorGuard:
    """See module docstring. `observe()` per tick; `tripped`/`trip_reason` are
    latched until `reset()` (a completed calibration)."""

    def __init__(self, rig: dict):
        g = (rig.get("safety", {}) or {}).get("anchor_guard", {}) or {}
        self.enabled = bool(g.get("enabled", True))
        # One-sample discontinuity beyond jump_m + speed_allow*dt = suspect.
        # Margins vs reality: normal motion p99 < 3 cm/frame at ~9 ms cadence;
        # an anchor shift that matters is decimetres. speed_allow covers honest
        # fast motion scaling with the sample gap.
        self.jump_m = float(g.get("jump_m", 0.25))
        self.speed_allow = float(g.get("speed_allow", 4.0))
        self.confirm_frames = max(1, int(g.get("confirm_frames", 3)))
        self.return_tol = float(g.get("return_tol", 0.10))
        # Two jump vectors are "the same anchor delta" when they differ by less
        # than max(coherence_tol, 25% of the larger jump).
        self.coherence_tol = float(g.get("coherence_tol", 0.15))
        self.max_gap_s = float(g.get("max_sample_gap_s", 0.3))
        self.blackout_s = float(g.get("blackout_s", 2.0))
        transport = str(rig.get("vr", {}).get("transport", "fake"))
        # Blackout semantics only exist for a live headset link; replay/fake
        # streams pause for benign reasons (loop restart, debugger).
        self.blackout_armed = transport in ("orbit", "vuer")
        self._track = {s: _SideTrack() for s in SIDES}
        self._last_fresh_t: float | None = None
        self._ever_fresh = False
        self.tripped = False
        self.trip_reason: str | None = None
        self.trip_count = 0
        self._trip_is_new = False
        self.holds = {s: False for s in SIDES}

    # ---- lifecycle --------------------------------------------------------- #
    def reset(self) -> None:
        """Forgive everything — called when a fresh calibration is applied (the
        new fit absorbs whatever the anchors are NOW) and when the engine's
        body axes change under the signal (yaw re-latch)."""
        for tr in self._track.values():
            tr.drop()
        self.tripped = False
        self.trip_reason = None
        self._trip_is_new = False
        self.holds = {s: False for s in SIDES}
        self._last_fresh_t = None

    def take_trip(self) -> bool:
        """True exactly once per trip (the engine consumes the edge)."""
        if self._trip_is_new:
            self._trip_is_new = False
            return True
        return False

    def status(self) -> dict:
        return {"enabled": self.enabled, "tripped": self.tripped,
                "reason": self.trip_reason, "trips": self.trip_count,
                "holds": dict(self.holds)}

    # ---- per-tick ---------------------------------------------------------- #
    def observe(self, wrist_body: dict[str, np.ndarray | None], fresh: bool,
                t: float, armed: bool = True) -> dict[str, bool]:
        """Feed one tick of body-relative wrist positions (None = untracked /
        unusable) + stream freshness. Returns per-side HOLD flags (suspect
        window in flight — feed the mapper nothing for that side this tick).
        `armed=False` (follow already locked) keeps continuity tracking but
        never trips. Duplicate samples (replay/latest() holding one frame
        across engine ticks) are skipped so dt and the confirm count run on
        real sample-to-sample time."""
        if not self.enabled:
            return {s: False for s in SIDES}
        if fresh:
            self._last_fresh_t = t
            self._ever_fresh = True
        elif (self.blackout_armed and armed and not self.tripped
              and self._ever_fresh and self._last_fresh_t is not None
              and (t - self._last_fresh_t) > self.blackout_s):
            self._trip(f"stream dead {t - self._last_fresh_t:.1f}s "
                       "(app restart / headset sleep resets anchors)")
        for s in SIDES:
            self._observe_side(s, wrist_body.get(s), t)
        self._judge(armed)
        self.holds = {s: self._track[s].suspect for s in SIDES}
        return dict(self.holds)

    def _observe_side(self, side: str, p, t: float) -> None:
        tr = self._track[side]
        if p is None:
            tr.drop()                                  # dropout forgives everything
            return
        p = np.asarray(p, dtype=float).reshape(3)
        if not np.all(np.isfinite(p)):
            tr.drop()
            return
        if tr.pos is None:
            tr.seed(p, t)
            return
        if tr.last_p is not None and np.array_equal(p, tr.last_p):
            return                                     # duplicate frame — no new information
        dt = t - tr.last_t
        tr.last_p = p.copy()
        tr.last_t = float(t)
        if dt <= 0.0:
            return
        if not tr.suspect:
            if dt > self.max_gap_s:
                tr.seed(p, t)                          # continuity expired — reseed
            elif float(np.linalg.norm(p - tr.pos)) > self.jump_m + self.speed_allow * dt:
                tr.ref = tr.pos.copy()                 # pre-jump reference
                tr.ref_t = tr.t
                tr.n_far = 1
            else:
                tr.pos = p.copy()
                tr.t = t
            return
        # suspect: did it come back, or does the new regime persist?
        if float(np.linalg.norm(p - tr.ref)) <= self.return_tol + self.speed_allow * (t - tr.ref_t):
            tr.seed(p, t)                              # glitch bounced back — resume
            return
        tr.n_far += 1
        if tr.n_far >= self.confirm_frames:
            tr.confirmed = True

    def _judge(self, armed: bool) -> None:
        """Resolve confirmed suspects against the other hand (shared anchor)."""
        for s in SIDES:
            tr = self._track[s]
            if not tr.confirmed:
                continue
            other = self._track[_OTHER[s]]
            jump = tr.jump
            if jump is None:                           # defensive: cannot happen while confirmed
                tr.clear_suspect()
                continue
            if other.pos is None and not other.suspect:
                self._resolve_trip(armed, tr,
                                   f"{s} wrist jumped {float(np.linalg.norm(jump)):.2f} m "
                                   "with the other hand untracked")
            elif other.confirmed:
                a, b = jump, other.jump
                gap = float(np.linalg.norm(a - b))
                big = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
                if gap <= max(self.coherence_tol, 0.25 * big):
                    reason = (f"both wrists jumped together ({big:.2f} m) — "
                              "tracking anchor moved")
                else:
                    # two simultaneous unrelated teleports: not explainable as
                    # motion OR a shared anchor — fail safe.
                    reason = (f"both wrists jumped {float(np.linalg.norm(a)):.2f} / "
                              f"{float(np.linalg.norm(b)):.2f} m incoherently")
                self._resolve_trip(armed, tr, reason)
                self._resolve_trip(armed, other, None)   # release the partner's hold too
            elif other.suspect:
                continue                               # let the other side resolve first
            else:
                # Other hand tracked and steady through the window → the shared
                # anchor did not move; this was a real (if violent) hand event.
                # Accept the new position and resume — same policy as the
                # ArmController teleport rejection (re-anchor + glide).
                tr.accept_new_regime()

    def _resolve_trip(self, armed: bool, tr: _SideTrack, reason: str | None) -> None:
        """Resolve a suspect window either way (holds must release: the operator
        needs working continuity tracking even while tripped) and latch the trip
        when armed. reason=None resolves without attempting a second latch."""
        tr.accept_new_regime()
        if armed and reason is not None:
            self._trip(reason)

    def _trip(self, reason: str) -> None:
        # Re-trips are NOT collapsed: each armed anchor event must re-fire the
        # engine response (a jump during the post-trip RE-calibration has to
        # cancel that capture too). The blackout channel self-gates on
        # `not self.tripped`, so a dead stream cannot spam this.
        self.tripped = True
        self.trip_reason = reason
        self.trip_count += 1
        self._trip_is_new = True
