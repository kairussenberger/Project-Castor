"""Operator body-frame helpers + the OPTIONAL legacy startup stillness hold.

The pieces the runtime actually depends on are stateless:
  - `head_op_axes`: operator body axes (right/up/forward) from the head pose;
  - `body_relative_hand_sample`: wrist pose re-expressed relative to the torso
    proxy in those body axes (what arm control consumes);
  - `R_base_from_body` / `W_AXES`: the fixed body→robot-world axis map shared by
    translation AND rotation in ClutchMapper.

There is NO stance/orientation calibration anymore: the old `vr.calib_seconds`
hold built a hand-local↔EE-local correspondence (P) from an assumed arms-at-sides
pose, and any deviation from that stance scrambled every commanded rotation axis
(measured ≈145° median axis error on a real Quest session). The `Calibrator` class
remains for the optional stillness/quality gate and for the legacy
non-body-relative mapping R used only in diagnostics (`vr.body_relative: false`).
"""
from __future__ import annotations

import numpy as np

from .frames import HandSample, quat_to_R

# WebXR 25-joint indices (W3C order)
W_WRIST = 0
W_INDEX_PROX, W_INDEX_TIP = 6, 9
W_MID_TIP = 14
W_RING_TIP = 19
W_PINKY_PROX = 21

# Desired robot WORLD axes for the reference stance: operator right → +Y,
# operator up → +Z, operator forward → −X (the robot faces −X).
W_AXES = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])

# WebXR gravity-up (the headset floor frame is +Y up).
_UP = np.array([0.0, 1.0, 0.0])
_DEFAULT_TORSO_FROM_HEAD = np.array([0.0, -0.35, 0.0])


def _finite_array(value, shape: tuple[int, ...]) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=float).reshape(shape)
    except (TypeError, ValueError):
        return None
    return arr if np.all(np.isfinite(arr)) else None


def _safe_untracked_copy(hand: HandSample) -> HandSample:
    wrist = _finite_array(hand.wrist, (4, 4))
    return HandSample(tracked=False, wrist=wrist.copy() if wrist is not None else np.eye(4),
                      landmarks=hand.landmarks, pinch=hand.pinch)


def head_op_axes(head_mat: np.ndarray) -> np.ndarray:
    """(right, up, forward) operator BODY axes (columns) in the WebXR frame, from
    the head pose. Forward = the way you're facing, flattened to horizontal; up =
    gravity. Robust regardless of where the hands are (works with arms at sides).

    The WebXR camera looks down its local −Z, so view-forward = −(head R)·ẑ."""
    H = np.asarray(head_mat, dtype=float).reshape(4, 4)[:3, :3]
    fwd = -H[:, 2]                                   # camera −Z = view forward
    fwd = fwd - (fwd @ _UP) * _UP                    # flatten to horizontal
    n = np.linalg.norm(fwd)
    if n < 1e-6:                                     # looking straight up/down → fall back to raw
        fwd = -H[:, 2]
        n = np.linalg.norm(fwd) + 1e-9
    f = fwd / n
    r = np.cross(f, _UP); r /= (np.linalg.norm(r) + 1e-9)   # operator right
    u = np.cross(r, f); u /= (np.linalg.norm(u) + 1e-9)
    return np.column_stack([r, u, f])


def operator_axes(lm: np.ndarray) -> np.ndarray:
    """(right, up, forward) operator axes (columns) in the WebXR frame from a
    palms-down-FORWARD hand (legacy fallback only, when no head pose is available).
    Fingertips give forward, the back-of-hand normal gives up."""
    lm = np.asarray(lm, dtype=float).reshape(25, 3)
    wrist = lm[W_WRIST]
    fwd = lm[[W_INDEX_TIP, W_MID_TIP, W_RING_TIP]].mean(0) - wrist     # fingertips → forward
    palm_lat = lm[W_INDEX_PROX] - lm[W_PINKY_PROX]                     # across the palm
    n = np.cross(fwd, palm_lat)
    if n[1] < 0:                                                       # up points away from gravity
        n = -n
    f = fwd / (np.linalg.norm(fwd) + 1e-9)
    r = np.cross(f, n); r /= (np.linalg.norm(r) + 1e-9)               # right = forward × up
    u = np.cross(r, f); u /= (np.linalg.norm(u) + 1e-9)
    return np.column_stack([r, u, f])


def _avg_rotation(mats: list) -> np.ndarray | None:
    """Average a list of rotation matrices, re-projected onto SO(3) via SVD."""
    if not mats:
        return None
    M = np.mean(np.stack(mats), axis=0)
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:                 # guard against a reflection
        U[:, -1] *= -1
        Rm = U @ Vt
    return Rm


def R_base_from_op(op_axes: np.ndarray, base_quat) -> np.ndarray:
    """R_base_from_vr for one arm: maps a WebXR wrist displacement into the arm's
    IK base frame so the measured operator axes align with the robot world axes."""
    R_world_from_vr = W_AXES @ np.asarray(op_axes, float).T   # WebXR → world
    return quat_to_R(base_quat).T @ R_world_from_vr           # world → base


def R_base_from_body(base_quat) -> np.ndarray:
    """Map an operator BODY-frame wrist displacement into one arm's IK base frame.

    Body-frame coordinates are `[right, up, forward]`. The robot world convention is
    right→+Y, up→+Z, forward→−X, which is exactly `W_AXES`.
    """
    return quat_to_R(base_quat).T @ W_AXES


def _mirror_body(mirror_forward: bool, mirror_lateral: bool) -> np.ndarray | None:
    """Reflection M in body [right, up, forward] coords. mirror_forward negates the
    FORWARD axis (front↔back); mirror_lateral negates the RIGHT axis (left↔right).
    Position mirrors as M·p; orientation mirrors as M·R·M, which is the mirror-image
    rotation and stays PROPER (det +1) even for a single-axis reflection — so the
    absolute-orientation QP never sees an improper rotation. None = no mirror."""
    if not (mirror_forward or mirror_lateral):
        return None
    return np.diag([-1.0 if mirror_lateral else 1.0, 1.0, -1.0 if mirror_forward else 1.0])


def body_relative_hand_sample(hand: HandSample | None, head_mat: np.ndarray | None,
                              torso_from_head=(0.0, -0.35, 0.0),
                              mirror_forward: bool = False,
                              mirror_lateral: bool = False) -> HandSample | None:
    """Return a copy of `hand` whose wrist pose is expressed relative to the current
    operator torso/body frame.

    `torso_from_head` is in body coordinates `[right, up, forward]`; the default is
    roughly upper-torso/shoulder height below the headset. Translation becomes the
    vector from this torso proxy to the wrist in body axes. Rotation becomes wrist
    orientation in the same body axes.
    `mirror_forward` / `mirror_lateral` reflect the motion about the frontal / sagittal
    plane when the operator is observing the robot from a mirrored viewpoint.
    Landmarks are intentionally left unchanged because finger retargeting consumes
    their raw local hand geometry, not the arm-control wrist origin.
    """
    if hand is None or not hand.tracked:
        return hand
    if head_mat is None:
        return _safe_untracked_copy(hand)
    H = _finite_array(head_mat, (4, 4))
    W = _finite_array(hand.wrist, (4, 4))
    torso_body = _finite_array(torso_from_head, (3,))
    if H is None or W is None or torso_body is None:
        return _safe_untracked_copy(hand)
    op_axes = head_op_axes(H)
    torso_world = H[:3, 3] + op_axes @ torso_body
    wrist = W.copy()
    wrist[:3, 3] = op_axes.T @ (W[:3, 3] - torso_world)
    wrist[:3, :3] = op_axes.T @ W[:3, :3]
    M = _mirror_body(mirror_forward, mirror_lateral)
    if M is not None:
        wrist[:3, 3] = M @ wrist[:3, 3]
        wrist[:3, :3] = M @ wrist[:3, :3] @ M
    return HandSample(tracked=hand.tracked, wrist=wrist, landmarks=hand.landmarks, pinch=hand.pinch)


def calibrate_R(lm_avg: np.ndarray, base_quat) -> np.ndarray:
    """Legacy hand-stance R (kept for back-compat / fallback)."""
    return R_base_from_op(operator_axes(lm_avg), base_quat)


class Calibrator:
    def __init__(self, rig: dict):
        self.rig = rig
        self._samples: dict[str, list] = {"left": [], "right": []}
        self._wrists: dict[str, list] = {"left": [], "right": []}   # wrist rotation 3x3
        self._wrist_pos: dict[str, list] = {"left": [], "right": []}  # wrist translation
        self._wrist_heads: dict[str, list] = {"left": [], "right": []}  # head 4x4 paired with wrist samples
        self._heads: list = []                                       # head 3x3, shared
        torso = _finite_array(rig.get("vr", {}).get("torso_from_head", _DEFAULT_TORSO_FROM_HEAD), (3,))
        self._torso_from_head = torso if torso is not None else _DEFAULT_TORSO_FROM_HEAD.copy()

    def add(self, side: str, landmarks, wrist=None, head=None) -> None:
        lm = _finite_array(landmarks, (25, 3)) if landmarks is not None else None
        if lm is not None:
            self._samples[side].append(lm)
        H = _finite_array(head, (4, 4)) if head is not None else None
        if wrist is not None:
            w = _finite_array(wrist, (4, 4))
            if w is None:
                if H is not None:
                    self._heads.append(H[:3, :3])
                return
            self._wrists[side].append(w[:3, :3])
            self._wrist_pos[side].append(w[:3, 3])
            if H is not None:
                self._wrist_heads[side].append(H)
        if H is not None:
            self._heads.append(H[:3, :3])

    def count(self, side: str) -> int:
        return len(self._wrists[side]) or len(self._samples[side])

    def tracked(self, side: str) -> bool:
        return self.count(side) > 0

    def _op_axes(self, side: str):
        """Operator body axes — head-derived (robust at any hand pose). Falls back to
        the legacy hand-stance measurement only if no head pose was seen."""
        if self._heads:
            H = _avg_rotation(self._heads[-30:])
            T = np.eye(4); T[:3, :3] = H
            return head_op_axes(T)
        if self._samples[side]:
            print(f"[WARN] calibrate: no head pose for {side}; falling back to hand-stance "
                  "axes (expects palms-down-forward, NOT arms-at-sides).", flush=True)
            return operator_axes(np.stack(self._samples[side][-30:]).mean(0))
        return None

    def result(self, side: str) -> dict | None:
        """Calibration for one side from the at-rest window. Returns
        {R, ref, op_axes, wrist_ref, ok, std, forward, up} or None if too few samples.
        `ok` is False when the hand wasn't held still enough → re-calibrate."""
        w = self._wrists[side]
        if len(w) < 8:
            return None
        op = self._op_axes(side)
        if op is None:
            return None
        base_quat = self.rig["arms"][side]["base_quat"]
        R = R_base_from_op(op, base_quat)
        wrist_ref = _avg_rotation(w[-30:])
        # Stillness from wrist TRANSLATION jitter over the window. Prefer torso/body-
        # relative positions when head samples are paired with wrists, so normal
        # headset/body sway does not falsely mark a stable torso-to-hand pose as shaky.
        pos = self._body_relative_wrist_positions(side)
        if pos is None:
            pos = np.stack(self._wrist_pos[side][-30:]) if self._wrist_pos[side] else None
        std = float(np.linalg.norm(pos.std(axis=0))) if pos is not None and len(pos) > 1 else 0.0
        ok = std < 0.02 and bool(np.isfinite(R).all()) and wrist_ref is not None
        return {"R": R, "ref": None, "op_axes": op, "wrist_ref": wrist_ref,
                "ok": ok, "std": std, "forward": op[:, 2], "up": op[:, 1]}

    def _body_relative_wrist_positions(self, side: str) -> np.ndarray | None:
        pos = self._wrist_pos[side][-30:]
        heads = self._wrist_heads[side][-30:]
        if not pos or len(pos) != len(heads):
            return None
        out = []
        for p, H in zip(pos, heads, strict=True):
            op_axes = head_op_axes(H)
            torso_world = H[:3, 3] + op_axes @ self._torso_from_head
            out.append(op_axes.T @ (np.asarray(p, dtype=float) - torso_world))
        return np.stack(out)

    # Back-compat thin wrappers
    def compute(self, side: str) -> np.ndarray | None:
        r = self.result(side)
        return r["R"] if r else None

    def ref_frame(self, side: str) -> np.ndarray | None:
        r = self.result(side)
        return r["ref"] if r else None
