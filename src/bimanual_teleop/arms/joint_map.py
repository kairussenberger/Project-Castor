"""Motor↔model joint-space mapping for the real YAM arms — measured, never assumed.

The runtime model (rig.yaml / yam_pin) and the i2rt motor firmware use DIFFERENT
joint conventions: the model's j1 lives in [0, 2π] with the ragdoll hang at ≈π,
while the i2rt YAM XML declares j1 ∈ [-2.62, 3.05], j2 ∈ [0, 3.65], j3 ∈ [0, 3.67]
— different zeros, possibly different signs, per joint. Commanding model-space
angles straight onto the bus would sweep the arm toward a wrong pose at whatever
speed the shaper allows. This module is the explicit, per-side affine bridge:

    q_motor = signs * q_model + offsets          (signs ∈ {−1, +1}, per joint)

Signs are geometric constants measured ONCE per side with scripts/hw_bringup.py
(±2° motor-space nudges, operator confirms direction against the dashboard).
Offsets are anchored at the OFFICIAL REST POSE (docs/RESTING_POSE.md): with the
arm hanging limp, offsets = q_motor_measured − signs · neutral_q. Both persist in
a per-machine JSON (rig: hardware.joint_map_file, gitignored) because they encode
this rig's motor zero calibration, which is not portable.

The rest pose doubles as the runtime gate: a session may only start commanding an
arm whose measured pose maps back to neutral_q within hardware.engage_pose_tol.
That one check catches a wrong arm on the bus (the ±90° palms-inward j6 differs
left↔right by π), a ±2π boot-wrap on a motor, zero-calibration drift, and an arm
that simply isn't at rest — all BEFORE any torque tracks a model command.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

N_JOINTS = 6


def _vec(x, name: str) -> np.ndarray:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.shape != (N_JOINTS,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be {N_JOINTS} finite numbers, got {x!r}")
    return v


@dataclass(frozen=True)
class JointMap:
    """Affine per-joint bridge between model space and motor space."""

    signs: np.ndarray    # (6,) of ±1
    offsets: np.ndarray  # (6,) rad: q_motor = signs*q_model + offsets

    def __post_init__(self):
        object.__setattr__(self, "signs", _vec(self.signs, "signs"))
        object.__setattr__(self, "offsets", _vec(self.offsets, "offsets"))
        if not np.all(np.isin(self.signs, (-1.0, 1.0))):
            raise ValueError(f"signs must be ±1 per joint, got {self.signs}")

    def to_motor(self, q_model) -> np.ndarray:
        return self.signs * _vec(q_model, "q_model") + self.offsets

    def to_model(self, q_motor) -> np.ndarray:
        return (_vec(q_motor, "q_motor") - self.offsets) * self.signs  # signs⁻¹ == signs

    @staticmethod
    def anchored_at_rest(signs, q_motor_rest, neutral_q) -> "JointMap":
        """Build the map from measured motor angles AT the official rest pose."""
        signs = _vec(signs, "signs")
        return JointMap(signs, _vec(q_motor_rest, "q_motor_rest") - signs * _vec(neutral_q, "neutral_q"))


@dataclass
class PoseGate:
    """Verdict of the rest-pose engage gate (kept whole for error messages)."""

    ok: bool
    deltas: np.ndarray              # measured_model − neutral_q, rad
    tol: float
    worst: int = field(init=False)  # joint index of the largest violation

    def __post_init__(self):
        self.worst = int(np.argmax(np.abs(self.deltas)))

    def describe(self) -> str:
        rows = ", ".join(
            f"j{i + 1}={np.degrees(d):+.1f}°{'  ←' if i == self.worst and not self.ok else ''}"
            for i, d in enumerate(self.deltas)
        )
        verdict = "within" if self.ok else "EXCEEDS"
        return f"measured−rest [{rows}] {verdict} ±{np.degrees(self.tol):.1f}°"


def rest_pose_gate(measured_model, neutral_q, tol_rad: float) -> PoseGate:
    """May this arm start tracking model commands? Only if it is measurably AT
    the official rest pose. Fail-closed on any non-finite input."""
    try:
        raw = _vec(measured_model, "measured") - _vec(neutral_q, "neutral_q")
    except ValueError:
        return PoseGate(False, np.full(N_JOINTS, np.inf), float(tol_rad))
    # Shortest-arc comparison: the multi-turn wrap normalizer can land on either
    # 2pi branch when a joint rests near +-180 deg (j2 does), and that branch
    # choice is not a pose error. A true full-turn offset cannot survive the
    # fresh-read normalization at chain open, so modulo-2pi distance is exact.
    deltas = (raw + np.pi) % (2.0 * np.pi) - np.pi
    return PoseGate(bool(np.all(np.abs(deltas) <= float(tol_rad))), deltas, float(tol_rad))


# --------------------------------------------------------------------------- #
# Per-machine persistence (config/hw_joint_map.json — see rig hardware.joint_map_file)
# --------------------------------------------------------------------------- #

def load_joint_map(path: str | Path, side: str) -> JointMap | None:
    """Load one side's calibrated map; None when the side was never calibrated."""
    p = Path(path)
    if not p.exists():
        return None
    entry = json.loads(p.read_text()).get(side)
    if entry is None:
        return None
    return JointMap(entry["signs"], entry["offsets"])


def save_joint_map(path: str | Path, side: str, jm: JointMap, *,
                   channel: str, q_motor_rest) -> None:
    """Merge-write one side's calibration (other sides preserved)."""
    p = Path(path)
    doc = json.loads(p.read_text()) if p.exists() else {}
    doc[side] = {
        "signs": jm.signs.tolist(),
        "offsets": jm.offsets.tolist(),
        "q_motor_rest": _vec(q_motor_rest, "q_motor_rest").tolist(),
        "channel": channel,
        "calibrated_unix": time.time(),
        "calibrated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2) + "\n")


def map_file_from_rig(rig: dict) -> Path:
    """Resolve hardware.joint_map_file relative to the repo root."""
    rel = str(rig.get("hardware", {}).get("joint_map_file", "config/hw_joint_map.json"))
    p = Path(rel)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[3] / rel
    return p
