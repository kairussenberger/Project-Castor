"""Operator-triggered TWO-POSE calibration (position-only, runtime).

The absolute mapping is 1:1 by default: your torso→wrist vector in metres becomes
the robot's chest→wrist vector. Two things break a naive fit of that map:

  1. Operators are not robot-sized (the YAM's reach/mounting proportions differ).
  2. ORBIT's hand positions live in a RECENTER-ANCHORED frame: starting the app
     or recentering with the headset somewhere arbitrary (a desk) shifts EVERY
     hand position by an unknown constant 3-vector (measured: 0.5 m down after a
     desk start — a one-pose fit then reads "arms at hip height" and either
     refuses or fits garbage). There is no in-data absolute reference: the hand
     keypoints share the same anchor as the wrist stream.

The guided THREE-POSE capture solves both at once (captured rest → clap →
extended-forward; the fit itself is order-free, A/B/C below name the ROLES):

    pose 1 (B) — relax both arms down at your sides, hold ~2.5 s
    pose 2 (C) — press your palms together in front of your chest, hold ~2.5 s
             (anchors the LATERAL map at contact width: YOUR clap maps to the
             ROBOT's hands touching, by construction, and your true midline
             is measured where your palms meet)
    pose 3 (A) — extend both arms straight forward at shoulder height, hold ~2.5 s
             (LAST on purpose: pose A maps onto the robot's neutral by
             construction, so the fit completes while you are ALREADY HOLDING
             that correspondence — the arms engage and glide to the natural
             forward neutral, instead of engaging mid-clap at the chest)

Everything the fit needs comes out anchor-proof and head-yaw-proof:
  - the operator's FORWARD direction = the horizontal direction of the A−B
    wrist-midpoint delta (raising your arms from your sides to extended-forward
    IS forward — wherever your head points, e.g. at the dashboard);
  - LATERAL scale from the pose-A wrist SPREAD (anchor cancels in the spread);
  - FORWARD and UP scales from the per-axis A−B DELTAS against the robot's
    matching references (robot_neutral_wrist ↔ A, robot_rest_wrist — its actual
    rest pose — ↔ B): the anchor cancels in every difference;
  - the OFFSET (computed last, from pose A) absorbs whatever the anchor did;
    the operator's measured midline (`lat_center`) maps to the robot's midline.

ORIENTATION IS NEVER TOUCHED. The absolute attitude mapping stays
calibration-free by contract (see CLAUDE.md / tests/test_frames.py).

The result is applied inside ClutchMapper._p_abs and persisted as JSON
(per-machine, gitignored). NOTE: recentering the headset mid-session moves the
anchor again — if the mapping suddenly feels shifted, recalibrate (8 seconds)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import SIDES
from .frames import lateral_curve

# Capture gates.
HOLD_S = 2.5          # continuous still time to accept each pose
WINDOW_S = 0.6        # rolling stillness window
STILL_TOL = 0.030     # m — std-norm of wrist_body over the window
SPREAD_MIN = 0.20     # m — pose-A lateral wrist spread (right − left), anchor-proof
SPREAD_MAX = 0.80
DROP_MIN = 0.22       # m — pose-A wrists must sit at least this far ABOVE pose B
DELTA_MIN = 0.15      # m — horizontal A−B midpoint delta needed to define forward
CLAP_SPREAD_MAX = 0.18  # m — pose-C wrists must be close together (palms pressed)
RAISE_MIN = 0.15      # m — pose-C wrists must come UP from pose B
TIMEOUT_S = 180.0     # all poses
SCALE_MIN, SCALE_MAX = 0.6, 2.0
# LOAD-time corruption screen ONLY. The offset must absorb anatomy AND the
# wrist↔head stream ANCHOR MISMATCH, which is unbounded in practice: ORBIT's
# wrist and head streams recenter-anchor INDEPENDENTLY (measured 1.35 m apart
# vertically on a real 2026-06-11 session, larger in offline reconstructions).
# NEVER clip or bound the FITTED offset — a clipped offset is self-inconsistent
# and parks the arms at the workspace ceiling (the 2026-06-11 regression: the
# old ±0.8 m clip turned a correct −2.19 m up-offset into −0.8 and every
# runtime target landed ~1.4 m above the chest).
OFFSET_MAX = 10.0     # m

# Robot-side references (body coords [right, up, forward] relative to the chest
# anchor) used when the rig does not provide them. Probed with the real IK.
ROBOT_NEUTRAL_DEFAULT = {"left": (-0.22, 0.02, 0.46), "right": (0.22, 0.02, 0.46)}
ROBOT_REST_DEFAULT = {"left": (-0.221, -0.437, -0.032), "right": (0.222, -0.440, 0.032)}


@dataclass
class CalibResult:
    """A fitted position calibration, in OPERATOR body axes:
    out_lat = s_lat-ramp(lat − lat_center); out_up/fwd = S·in + offset.
    `lat_ref` = half the measured pose-A spread (the non-linear lateral ramp
    reaches full scale there); `lat_center` = the operator's measured midline
    (absorbs the anchor's lateral shift — maps to the robot's midline)."""
    axis_scale: np.ndarray                  # (3,) [right, up, forward]
    body_offset: np.ndarray                 # (3,) metres ([0] unused — see lat_center)
    lat_ref: float = 0.0
    lat_center: float = 0.0
    # Piecewise-linear lateral curve knots [[x_clap, y_contact], [x_spread,
    # y_robot_half_spread]] (|lat−center| → robot |lat|) from pose C; None →
    # the legacy quadratic ramp. forward_body = the operator's arm-defined
    # forward (2D, measured body frame) — the engine latches its yaw lock to it.
    lat_knots: list | None = None
    forward_body: list | None = None
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"axis_scale": [round(float(v), 3) for v in self.axis_scale],
                "body_offset": [round(float(v), 3) for v in self.body_offset],
                "lat_ref": round(float(self.lat_ref), 3),
                "lat_center": round(float(self.lat_center), 3),
                "lat_knots": ([[round(float(v), 3) for v in k] for k in self.lat_knots]
                              if self.lat_knots else None),
                "quality": self.meta.get("quality"),
                "stamp": self.meta.get("stamp")}

    def payload(self) -> dict:
        """The canonical persisted form — written to disk by save() and EMBEDDED
        in session recordings (vr/replay.py), so one parser handles both."""
        return {"version": 4,
                "axis_scale": [float(v) for v in self.axis_scale],
                "body_offset": [float(v) for v in self.body_offset],
                "lat_ref": float(self.lat_ref),
                "lat_center": float(self.lat_center),
                "lat_knots": ([[float(v) for v in k] for k in self.lat_knots]
                              if self.lat_knots else None),
                "forward_body": ([float(v) for v in self.forward_body]
                                 if self.forward_body else None),
                "meta": self.meta}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.payload(), indent=2) + "\n")


def load_calibration(path: str | Path) -> CalibResult | None:
    """Load + validate a persisted calibration; None when absent or implausible
    (a corrupt/out-of-range file must never steer the arms)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return parse_calibration(json.loads(p.read_text()))
    except (json.JSONDecodeError, OSError):
        return None


def parse_calibration(d: dict) -> CalibResult | None:
    """Validate a calibration payload (persisted file OR embedded in a
    recording); None when implausible — corrupt data must never steer arms."""
    try:
        scale = np.asarray(d["axis_scale"], dtype=float).reshape(3)
        off = np.asarray(d["body_offset"], dtype=float).reshape(3)
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.all(np.isfinite(scale)) and np.all(np.isfinite(off))):
        return None
    if np.any(scale < SCALE_MIN - 1e-9) or np.any(scale > SCALE_MAX + 1e-9):
        return None
    if np.any(np.abs(off) > OFFSET_MAX + 1e-9):
        return None
    lat_ref = float(d.get("lat_ref", 0.0))
    lat_center = float(d.get("lat_center", 0.0))
    meta = d.get("meta", {})
    if lat_ref <= 0.0 and isinstance(meta.get("op_neutral"), dict):
        try:   # version-1/2 files: derive the ramp reference from the stored neutral
            lat_ref = float(np.mean([abs(meta["op_neutral"][s][0]) for s in SIDES]))
        except (KeyError, TypeError, IndexError):
            lat_ref = 0.0
    if not (0.0 <= lat_ref <= 0.6) or not np.isfinite(lat_center) or abs(lat_center) > 0.6:
        lat_ref, lat_center = max(0.0, min(lat_ref, 0.6)) if np.isfinite(lat_ref) else 0.0, 0.0
    knots = d.get("lat_knots")
    if knots is not None:
        try:
            knots = [[float(a), float(b)] for a, b in knots]
            ok = (len(knots) == 2 and 0.0 < knots[0][0] < knots[1][0] <= 0.6
                  and 0.0 < knots[0][1] < knots[1][1] <= 0.6)
            knots = knots if ok else None
        except (TypeError, ValueError):
            knots = None
    fb = d.get("forward_body")
    return CalibResult(axis_scale=scale, body_offset=off, lat_ref=lat_ref,
                       lat_center=lat_center, lat_knots=knots,
                       forward_body=list(fb) if fb else None, meta=meta)


def _reyaw_frame(mid_a: np.ndarray, mid_b: np.ndarray):
    """Forward/right horizontal unit vectors from the A−B midpoint delta
    (raising the arms from the sides to extended-forward IS forward). Returns
    (f2, r2) 2-vectors over (lat, fwd) components, or None if the delta is too
    small to define a direction."""
    d = np.asarray(mid_a, float) - np.asarray(mid_b, float)
    h = np.array([d[0], d[2]])
    n = float(np.linalg.norm(h))
    if n < DELTA_MIN:
        return None
    f2 = h / n
    r2 = np.array([f2[1], -f2[0]])
    return f2, r2


def _reyaw(v: np.ndarray, f2: np.ndarray, r2: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    xz = np.array([v[0], v[2]])
    return np.array([float(xz @ r2), v[1], float(xz @ f2)])


def _fit_quality(A, B, rbN, rbR, scale, raw_scale, off, lat_center, lat_ref,
                 lat_knots, ps) -> dict:
    """Grade a fit ON THE SPOT, through exactly the map the runtime will apply
    (shared `lateral_curve`). The 2026-06-11 floating-arms regression was only
    discoverable by post-mortem forensics on a recording — these numbers make a
    broken fit visible in the completion banner instead.

    - NEUTRAL residual (3D, per side): the held extended pose vs the robot
      neutral. The fit anchors the MEAN here by construction, so what remains
      is left/right asymmetry of the capture (and the lateral knot anchoring).
    - REST residual (up/forward only, per side): the rest pose vs the robot
      rest through the fitted scales. LATERAL is excluded by design — the
      lateral map is anchored at clap width and extended spread, and an
      operator's at-rest hand width is genuinely outside its claim.
    - CLIPPED scales: a raw scale outside [SCALE_MIN, SCALE_MAX] means the
      capture geometry disagreed with the robot references beyond what the
      model may express — the clipped value WILL mis-map proportionally."""
    def mapped(v):
        return ps * np.array([
            lateral_curve(float(v[0]) - lat_center, float(scale[0]), lat_ref, lat_knots),
            float(scale[1]) * float(v[1]) + float(off[1]),
            float(scale[2]) * float(v[2]) + float(off[2])])

    res = {"neutral": {}, "rest": {}}
    worst = 0.0
    for s in SIDES:
        rn = float(np.linalg.norm(mapped(A[s]) - rbN[s]))
        rr = float(np.linalg.norm((mapped(B[s]) - rbR[s])[1:]))
        res["neutral"][s] = round(rn * 100, 1)                       # cm
        res["rest"][s] = round(rr * 100, 1)
        worst = max(worst, rn, rr)
    reasons = [f"{s} {name} residual {cm:.0f} cm"
               for name, side_res in res.items()
               for s, cm in side_res.items() if cm > 5.0]
    clipped, badly_clipped = [], False
    for i, axis in enumerate(("lat", "up", "reach")):
        r = float(raw_scale[i])
        excess = max(SCALE_MIN / r if r > 0 else np.inf, r / SCALE_MAX)
        clipped.append(bool(excess > 1.0 + 1e-9))
        if clipped[-1]:
            reasons.append(f"{axis} scale clipped (raw {r:.2f})")
            badly_clipped = badly_clipped or excess > 1.15
    grade = ("bad" if worst > 0.10 or badly_clipped
             else ("check" if reasons else "good"))
    return {"grade": grade, "worst_cm": round(worst * 100, 1), "reasons": reasons,
            "residual_cm": res, "scale_raw": [round(float(r), 3) for r in raw_scale],
            "clipped": clipped}


def fit_two_pose(pose_a: dict[str, np.ndarray], pose_b: dict[str, np.ndarray],
                 robot_neutral: dict[str, np.ndarray], robot_rest: dict[str, np.ndarray],
                 pos_scale: float = 1.0, pose_c: dict[str, np.ndarray] | None = None,
                 robot_clap_gap: float = 0.12) -> CalibResult | None:
    """Fit scale/offset from the held poses. Every scale comes from an A−B
    DIFFERENCE or a spread, so the ORBIT recenter anchor cancels exactly; the
    offset (from pose A) absorbs it. With pose C (palms together) the lateral
    map becomes a piecewise-linear curve anchored at CONTACT width — the
    operator's clap maps to the robot's hands touching by construction — and
    the midline is measured where the palms actually meet. Pure math."""
    mid_a = 0.5 * (np.asarray(pose_a["left"], float) + np.asarray(pose_a["right"], float))
    mid_b = 0.5 * (np.asarray(pose_b["left"], float) + np.asarray(pose_b["right"], float))
    fr = _reyaw_frame(mid_a, mid_b)
    if fr is None:
        return None
    f2, r2 = fr
    A = {s: _reyaw(pose_a[s], f2, r2) for s in SIDES}
    B = {s: _reyaw(pose_b[s], f2, r2) for s in SIDES}
    rbN = {s: np.asarray(robot_neutral[s], dtype=float).reshape(3) for s in SIDES}
    rbR = {s: np.asarray(robot_rest[s], dtype=float).reshape(3) for s in SIDES}

    spread_a = A["right"][0] - A["left"][0]
    if not (SPREAD_MIN <= spread_a <= SPREAD_MAX):
        return None
    s_lat = (rbN["right"][0] - rbN["left"][0]) / spread_a
    d_up = float(np.mean([A[s][1] - B[s][1] for s in SIDES]))
    d_fwd = float(np.mean([A[s][2] - B[s][2] for s in SIDES]))
    if d_up < DROP_MIN or d_fwd < DELTA_MIN / 2:
        return None
    s_up = float(np.mean([rbN[s][1] - rbR[s][1] for s in SIDES])) / d_up
    s_fwd = float(np.mean([rbN[s][2] - rbR[s][2] for s in SIDES])) / d_fwd
    scale_raw = np.array([s_lat, s_up, s_fwd])
    scale = np.clip(scale_raw, SCALE_MIN, SCALE_MAX)

    lat_center = 0.5 * (A["right"][0] + A["left"][0])    # operator midline (incl. anchor)
    lat_knots = None
    C = None
    if pose_c is not None:
        C = {s: _reyaw(pose_c[s], f2, r2) for s in SIDES}
        clap_gap = C["right"][0] - C["left"][0]
        if -0.05 <= clap_gap <= CLAP_SPREAD_MAX:
            lat_center = 0.5 * (C["right"][0] + C["left"][0])   # palms meet AT the midline
            x_c = max(abs(clap_gap) / 2.0, 0.01)
            y_c = max(robot_clap_gap, 0.02) / 2.0
            x_a = spread_a / 2.0
            y_a = 0.5 * (rbN["right"][0] - rbN["left"][0])
            if x_c < x_a - 0.02 and y_c < y_a:
                lat_knots = [[x_c, y_c], [x_a, y_a]]
    ps = max(pos_scale, 1e-6)
    off = np.mean([rbN[s] / ps - scale * A[s] for s in SIDES], axis=0)
    # The offset IS the anchor absorber — it must map pose A onto the robot
    # neutral EXACTLY, whatever the stream anchors did (see OFFSET_MAX note).
    # Finite is the only requirement; the capture gates already vet the poses.
    if not np.all(np.isfinite(off)):
        return None
    off[0] = 0.0                                          # lateral handled by lat_center
    quality = _fit_quality(A, B, rbN, rbR, scale, scale_raw, off, lat_center,
                           spread_a / 2.0, lat_knots, ps)
    meta = {"stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "quality": quality,
            "pose_a": {s: [round(float(v), 4) for v in A[s]] for s in SIDES},
            "pose_b": {s: [round(float(v), 4) for v in B[s]] for s in SIDES},
            "pose_c": ({s: [round(float(v), 4) for v in C[s]] for s in SIDES}
                       if C is not None else None),
            "op_neutral": {s: [round(float(v), 4) for v in A[s]] for s in SIDES},
            "robot_neutral": {s: rbN[s].round(4).tolist() for s in SIDES},
            "robot_rest": {s: rbR[s].round(4).tolist() for s in SIDES}}
    return CalibResult(axis_scale=scale, body_offset=off, lat_ref=spread_a / 2.0,
                       lat_center=lat_center, lat_knots=lat_knots,
                       forward_body=[float(f2[0]), float(f2[1])], meta=meta)


class NeutralPoseCalibration:
    """The guided three-pose capture: rest (arms down at the sides) → clap
    (palms together) → extended-forward LAST, each held still HOLD_S. Ending
    on the extended pose means the fit completes while the operator already
    holds the robot's neutral correspondence. Drives `status` dicts the
    dashboard renders as the prompt. Clock-injected and deterministic; the
    engine feeds it body-relative wrist samples each tick."""

    def __init__(self, rig: dict):
        m = rig.get("mapping", {})
        rn = m.get("robot_neutral_wrist") or {}
        rr = m.get("robot_rest_wrist") or {}
        self.robot_neutral = {
            s: np.asarray(rn.get(s, ROBOT_NEUTRAL_DEFAULT[s]), dtype=float).reshape(3)
            for s in SIDES}
        self.robot_rest = {
            s: np.asarray(rr.get(s, ROBOT_REST_DEFAULT[s]), dtype=float).reshape(3)
            for s in SIDES}
        self.pos_scale = float(m.get("pos_scale", 1.0))
        self.robot_clap_gap = float(m.get("robot_clap_gap", 0.12))
        self.active = False
        self.phase = "idle"      # idle | wait_rest | wait_clap | wait_fwd | done | cancelled
        self.result: CalibResult | None = None
        self._t0 = 0.0
        self._hold_t0: float | None = None
        self._buf: dict[str, list[tuple[float, np.ndarray]]] = {s: [] for s in SIDES}
        self._pose_b: dict[str, np.ndarray] | None = None
        self._pose_c: dict[str, np.ndarray] | None = None
        self._msg = ""
        self._seen = {s: False for s in SIDES}

    # ---- lifecycle --------------------------------------------------------- #
    def start(self, t: float) -> None:
        self.active = True
        self.phase = "wait_rest"
        self.result = None
        self._t0 = t
        self._hold_t0 = None
        self._buf = {s: [] for s in SIDES}
        self._pose_b = None
        self._pose_c = None
        self._msg = ""

    def cancel(self, msg: str = "calibration cancelled") -> None:
        self.active = False
        self.phase = "cancelled"
        self._msg = msg

    # ---- per-tick ---------------------------------------------------------- #
    def tick(self, wrist_body: dict[str, np.ndarray | None], t: float) -> None:
        if not self.active:
            return
        if (t - self._t0) > TIMEOUT_S:
            self.cancel("calibration timed out — press CALIBRATE to retry")
            return
        for s in SIDES:
            w = wrist_body.get(s)
            self._seen[s] = w is not None
            if w is not None:
                buf = self._buf[s]
                buf.append((t, np.asarray(w, dtype=float).reshape(3)))
                while buf and (t - buf[0][0]) > max(WINDOW_S, HOLD_S):
                    buf.pop(0)
        ready = self._pose_ready(t)
        if ready:
            if self._hold_t0 is None:
                self._hold_t0 = t
            elif (t - self._hold_t0) >= HOLD_S:
                self._advance(t)
        else:
            self._hold_t0 = None

    def _window(self, side: str, t: float, span: float) -> np.ndarray | None:
        pts = [w for (ts, w) in self._buf[side] if (t - ts) <= span]
        return np.stack(pts) if len(pts) >= 4 else None

    def _still_means(self, t: float) -> dict[str, np.ndarray] | None:
        """Per-side window means, or None unless BOTH hands are tracked + still."""
        means = {}
        for s in SIDES:
            win = self._window(s, t, WINDOW_S)
            if win is None or float(np.linalg.norm(win.std(axis=0))) > STILL_TOL:
                return None
            means[s] = win.mean(axis=0)
        return means

    def _pose_ready(self, t: float) -> bool:
        """Anchor-proof gates (nothing trusts an absolute frame): pose 1 (rest)
        needs a sane lateral SPREAD; pose 2 (palms together) needs the wrists
        CLOSE and raised up from the rest pose; pose 3 (extended forward)
        needs a sane spread AND the wrists RAISED ≥ DROP_MIN above rest."""
        means = self._still_means(t)
        if means is None:
            return False
        spread = abs(means["right"][0] - means["left"][0])
        if self.phase == "wait_clap":
            raised = float(np.mean([means[s][1] - self._pose_b[s][1] for s in SIDES]))
            return spread <= CLAP_SPREAD_MAX and raised >= RAISE_MIN
        if not (SPREAD_MIN <= spread <= SPREAD_MAX):
            return False
        if self.phase == "wait_rest":
            return True
        raised = float(np.mean([means[s][1] - self._pose_b[s][1] for s in SIDES]))
        return raised >= DROP_MIN

    def _advance(self, t: float) -> None:
        means = {}
        for s in SIDES:
            win = self._window(s, t, HOLD_S)
            if win is None:
                self._hold_t0 = None
                return
            means[s] = win.mean(axis=0)
        if self.phase == "wait_rest":
            self._pose_b = means
            self.phase = "wait_clap"
            self._hold_t0 = None
            self._buf = {s: [] for s in SIDES}       # fresh windows for the next pose
            return
        if self.phase == "wait_clap":
            self._pose_c = means
            self.phase = "wait_fwd"
            self._hold_t0 = None
            self._buf = {s: [] for s in SIDES}
            return
        res = fit_two_pose(means, self._pose_b, self.robot_neutral, self.robot_rest,
                           self.pos_scale, pose_c=self._pose_c,
                           robot_clap_gap=self.robot_clap_gap)
        if res is None:                               # degenerate capture — keep waiting
            self._hold_t0 = None
            return
        self.result = res
        self.active = False
        self.phase = "done"
        sc = res.axis_scale
        q = (res.meta or {}).get("quality") or {}
        grade = q.get("grade", "?")
        mark = {"good": "✓", "check": "⚠", "bad": "✗"}.get(grade, "✓")
        fit = f" · fit {grade.upper()} (worst {q.get('worst_cm', '?')} cm)"
        if grade != "good" and q.get("reasons"):
            fit += f" — {q['reasons'][0]}"
        self._msg = (f"CALIBRATED {mark} scale lat {sc[0]:.2f} / up {sc[1]:.2f} / reach {sc[2]:.2f}, "
                     f"midline {res.lat_center:+.2f} m"
                     + ("" if res.lat_knots else " (no clap anchor)") + fit)

    # ---- display ----------------------------------------------------------- #
    def status(self, t: float) -> dict:
        if self.active and self._hold_t0 is not None:
            elapsed = t - self._hold_t0
            step = {"wait_rest": "1/3", "wait_clap": "2/3"}.get(self.phase, "3/3")
            return {"active": True, "kind": "neutral", "phase": "hold",
                    "progress": min(1.0, elapsed / HOLD_S),
                    "remaining": max(0.0, HOLD_S - elapsed),
                    "left": self._seen["left"], "right": self._seen["right"],
                    "msg": f"HOLD STILL ({step}) — measuring… {max(0.0, HOLD_S - elapsed):.1f}s"}
        if self.active:
            if not all(self._seen.values()):
                msg = "CALIBRATION: wear the headset, controllers down — both hands in view"
            elif self.phase == "wait_clap":
                msg = "CALIBRATION 2/3: PRESS YOUR PALMS TOGETHER in front of your chest — and hold"
            elif self.phase == "wait_fwd":
                msg = "CALIBRATION 3/3: EXTEND BOTH ARMS straight forward at shoulder height — and hold"
            else:
                msg = "CALIBRATION 1/3: RELAX BOTH ARMS DOWN at your sides — and hold"
            return {"active": True, "kind": "neutral", "phase": self.phase, "progress": 0.0,
                    "remaining": HOLD_S, "left": self._seen["left"],
                    "right": self._seen["right"], "msg": msg}
        return {"active": False, "kind": "neutral", "phase": self.phase,
                "progress": 1.0 if self.phase == "done" else 0.0, "remaining": 0.0,
                "left": self._seen["left"], "right": self._seen["right"], "msg": self._msg}
