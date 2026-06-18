"""VR pose types + the relative/clutch SE(3) mapping from a tracked wrist to an
arm end-effector target.

Mapping (per OpenTeleVision / Quest2ROS best practice): on clutch *engage* we
latch the current wrist pose and the current EE pose as anchors. While engaged,
the EE target is the anchored EE pose composed with the operator's wrist motion
*relative* to its anchor. Translation AND rotation use the same change of basis:
the wrist delta, measured in the ctrl frame, is re-expressed into the arm base
frame by the one constant `R_base_from_vr` and applied about the anchor. Because
it's relative, absolute origin offsets cancel — there is NO stance calibration:
rotate your hand about a body axis and the EE rotates about the corresponding
robot-world axis, from any starting pose. See ClutchMapper.target().
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HandSample:
    tracked: bool = False
    wrist: np.ndarray = field(default_factory=lambda: np.eye(4))  # 4x4 in headset/world frame
    landmarks: np.ndarray | None = None                          # (25,3) WebXR joints
    pinch: float = 0.0                                           # 0..1 pinch strength


@dataclass
class VRFrame:
    stamp: float = 0.0
    head: np.ndarray | None = field(default_factory=lambda: np.eye(4))
    hands: dict[str, HandSample] = field(default_factory=dict)


def quat_to_R(q) -> np.ndarray:
    """Quaternion (w, x, y, z) → 3x3 rotation matrix (MuJoCo quat convention)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


# --- quaternion algebra (w, x, y, z convention, matching quat_to_R) --------- #
def quat_mul(a, b) -> np.ndarray:
    """Hamilton product a ⊗ b of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_inv(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return quat_conj(q) / float(q @ q)


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    h = 0.5 * float(angle)
    return np.array([np.cos(h), *(np.sin(h) * (a / n))])


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → unit (w, x, y, z) quaternion (Shepperd's method, numerically
    stable). Inverse of quat_to_R up to global sign (q and −q are the same rotation)."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def swing_twist_angle(R_err: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle of the TWIST component of R_err about unit `axis`, from the
    swing–twist decomposition R_err = R_swing · R_twist(axis, angle). Wrapped to
    [-π, π]; 0 when R_err has no component about the axis."""
    q = R_to_quat(R_err)
    if q[0] < 0.0:
        q = -q                                  # shortest-arc representation
    proj = float(q[1:] @ axis)
    n = float(np.hypot(q[0], proj))
    if n < 1e-12:
        return 0.0
    ang = 2.0 * np.arctan2(proj / n, q[0] / n)
    return float((ang + np.pi) % (2.0 * np.pi) - np.pi)


def conjugate_rotation(R_basis: np.ndarray, R_delta: np.ndarray) -> np.ndarray:
    """Change-of-basis for a ROTATION (spec Section 3): re-express R_delta — a
    rotation given in frame A — in the frame that R_basis (A→B) maps into:

        R_delta_in_B = R_basis · R_delta · R_basisᵀ

    The angle is preserved; the axis transforms by R_basis. This is the matrix form
    of the quaternion conjugation q_align · dq · q_align⁻¹ — the step people forget,
    which is exactly what scrambles a pure wrist roll into pitch+yaw if skipped."""
    R_basis = np.asarray(R_basis, dtype=float).reshape(3, 3)
    R_delta = np.asarray(R_delta, dtype=float).reshape(3, 3)
    return R_basis @ R_delta @ R_basis.T


def change_basis_quat(q_basis, dq) -> np.ndarray:
    """Quaternion form of conjugate_rotation: q_basis ⊗ dq ⊗ q_basis⁻¹."""
    return quat_mul(quat_mul(q_basis, dq), quat_inv(q_basis))


# WebXR reference frame (x=right, y=up, -z=forward) → robot WORLD frame
# (x, y, z; the robot faces world -X). Columns = where webxr +x,+y,+z land in world:
# webxr +z (back) → world +X  (so forward -z → -X = robot forward),
# webxr +y (up)   → world +Z,  webxr +x (right) → world +Y.
WEBXR_TO_WORLD = np.array([[0.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])

# The spec (Section 2/3) names the static OpenXR→robot basis change `R_align`.
# In this repo that is exactly WEBXR_TO_WORLD (headset world → robot world); the
# per-arm IK-base rotation adds the arm's base_quat on top (see r_base_from_vr).
# Exposed under the spec's name so call-sites and tests can speak the same language.
R_ALIGN = WEBXR_TO_WORLD


def r_base_from_vr(base_quat, tweak_euler=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Rotation mapping a wrist displacement in the WebXR frame to a displacement
    in this arm's IK base frame, so 'hand forward' → 'robot reaches forward'.

    The arm's standalone IK base frame is rotated into the world by `base_quat`,
    so: R = (world←base)ᵀ · (webxr→world) · tweak  =  base_quatᵀ · WEBXR_TO_WORLD · tweak.
    `tweak_euler` is an optional small correction (radians) applied in the WebXR
    frame if an axis still reads inverted."""
    R_bw = quat_to_R(base_quat)                 # base → world
    return R_bw.T @ WEBXR_TO_WORLD @ euler_to_R(tweak_euler)


def euler_to_R(euler_xyz) -> np.ndarray:
    """Intrinsic XYZ euler (rad) → 3x3 rotation (matches MuJoCo eulerseq XYZ)."""
    cx, cy, cz = np.cos(euler_xyz)
    sx, sy, sz = np.sin(euler_xyz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → rotation vector (axis * angle)."""
    R = np.asarray(R, dtype=float)
    ang = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if ang < 1e-7:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return ang / (2.0 * np.sin(ang)) * v


def lateral_curve(x: float, s: float, lat_ref: float, lat_knots=None) -> float:
    """The calibrated LATERAL map: operator lateral offset from their measured
    midline `x` → robot lateral (robot-midline-centred). ONE implementation
    shared by the live mapper (ClutchMapper._lat_scaled) and the calibration
    quality grader (vr/neutral_calib) so residuals are measured through
    exactly the map the runtime will apply.

    With knots: piecewise-linear |x|→|out| through (0,0), (x_clap → contact
    half-gap) and (x_spread → robot half-spread), extended with the last
    slope — the operator's clap maps to the robot's hands touching, their
    full spread to the robot's full spread, by construction. Without knots:
    QUADRATIC ramp from 1:1 at the midline to full scale `s` at |x| ≥ lat_ref
    for expanding scales (a linear ramp still amplified a real clap ×1.26 —
    measured); linear ramp for shrinking scales (the quadratic form would
    fold the map below s ≈ 0.67)."""
    if lat_knots is not None:
        (xc, yc), (xa, ya) = lat_knots
        ax = abs(x)
        if ax <= xc:
            out = ax * (yc / xc)
        elif ax <= xa:
            out = yc + (ax - xc) * (ya - yc) / (xa - xc)
        else:
            out = ya + (ax - xa) * (ya - yc) / (xa - xc)
        return float(np.sign(x) * out)
    if lat_ref <= 0.0:
        return s * x
    a = min(abs(x) / lat_ref, 1.0)
    ramp = a * a if s >= 1.0 else a
    return (1.0 + (s - 1.0) * ramp) * x


# --------------------------------------------------------------------------- #
# SE3 / SO3 — minimal pose types with the mink API surface this repo uses,
# decoupled from any IK backend. The Pinocchio/pink solver is touched ONLY at the
# task boundary in arms/ik.py via SE3.to_pin()/from_pin(); everything else (clutch
# mapper, controllers, tests, synthetic harness) speaks these classes. Replaces the
# former dependency on mink.SE3 / mink.SO3.
# --------------------------------------------------------------------------- #
class SO3:
    __slots__ = ("_R",)

    def __init__(self, R: np.ndarray):
        self._R = np.asarray(R, dtype=float).reshape(3, 3)

    @classmethod
    def from_matrix(cls, R: np.ndarray) -> "SO3":
        return cls(R)

    def as_matrix(self) -> np.ndarray:
        return self._R.copy()

    def inverse(self) -> "SO3":
        return SO3(self._R.T)


class SE3:
    __slots__ = ("_R", "_t")

    def __init__(self, R: np.ndarray, t: np.ndarray):
        self._R = np.asarray(R, dtype=float).reshape(3, 3)
        self._t = np.asarray(t, dtype=float).reshape(3)

    @classmethod
    def from_rotation_and_translation(cls, rotation, translation) -> "SE3":
        R = rotation.as_matrix() if isinstance(rotation, SO3) else np.asarray(rotation, float).reshape(3, 3)
        return cls(R, translation)

    @classmethod
    def from_translation(cls, translation) -> "SE3":
        return cls(np.eye(3), translation)

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> "SE3":
        T = np.asarray(T, dtype=float).reshape(4, 4)
        return cls(T[:3, :3], T[:3, 3])

    @classmethod
    def from_pin(cls, M) -> "SE3":
        """Build from a pinocchio.SE3 (rotation/translation are attributes there)."""
        return cls(np.asarray(M.rotation), np.asarray(M.translation))

    def to_pin(self):
        """Convert to pinocchio.SE3 for a pink FrameTask target (the only IK-backend touchpoint)."""
        import pinocchio as pin
        return pin.SE3(self._R.copy(), self._t.copy())

    def rotation(self) -> SO3:
        return SO3(self._R)

    def translation(self) -> np.ndarray:
        return self._t.copy()

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self._R
        T[:3, 3] = self._t
        return T


def mat_to_se3(T: np.ndarray) -> SE3:
    return SE3.from_matrix(T)


class ClutchMapper:
    """Clutch wrist→EE mapping for one arm.

    ROTATION is always relative: the wrist rotation since engage, measured in the
    ctrl frame, is re-expressed into the arm base frame by the single constant `R`
    and applied about the anchor. No stance calibration.

    POSITION has two modes:
      - 'absolute' (default rig config): the operator's torso→wrist vector maps
        1:1 (×scale) onto the robot's chest→wrist vector — hands held in front of
        the operator put the robot's hands in front of the robot. `chest_base` is
        the robot's chest/torso anchor in this arm's base frame. To stay snap-free,
        the offset between the current EE and the absolute target is latched at
        (re)engage and decays over `engage_blend_s` seconds, so the arm GLIDES
        onto correspondence instead of jumping.
      - 'relative': classic clutch deltas about the engage anchor (legacy, and the
        only valid choice when ctrl is a raw room-frame pose).
    """

    def __init__(self, R_base_from_vr: np.ndarray, pos_scale: float = 1.0, *,
                 position_mode: str = "relative", chest_base=None,
                 engage_blend_s: float = 1.0, twist_mode: str = "world",
                 hand_twist_axis=None, ee_tool_axis=None,
                 orientation_mode: str = "relative", hand_ee_convention=None):
        if position_mode not in ("relative", "absolute"):
            raise ValueError(f"position_mode must be 'relative' or 'absolute', got {position_mode!r}")
        if position_mode == "absolute" and chest_base is None:
            raise ValueError("absolute position_mode needs chest_base (robot chest in arm base frame)")
        if twist_mode not in ("world", "intrinsic"):
            raise ValueError(f"twist_mode must be 'world' or 'intrinsic', got {twist_mode!r}")
        if twist_mode == "intrinsic" and (hand_twist_axis is None or ee_tool_axis is None):
            raise ValueError("intrinsic twist_mode needs hand_twist_axis (hand-local) and "
                             "ee_tool_axis (EE-local j6 axis)")
        if orientation_mode not in ("relative", "absolute"):
            raise ValueError(f"orientation_mode must be 'relative' or 'absolute', got {orientation_mode!r}")
        if orientation_mode == "absolute" and hand_ee_convention is None:
            raise ValueError("absolute orientation_mode needs hand_ee_convention (EE-local→hand-local)")
        self.R = np.asarray(R_base_from_vr, dtype=float).reshape(3, 3)
        self.scale = float(pos_scale)
        self.mode = position_mode
        self.chest = None if chest_base is None else np.asarray(chest_base, dtype=float).reshape(3)
        self.blend_s = float(engage_blend_s)
        self.twist_mode = twist_mode
        self.hand_axis = None
        self.ee_axis = None
        if twist_mode == "intrinsic":
            h = np.asarray(hand_twist_axis, dtype=float).reshape(3)
            e = np.asarray(ee_tool_axis, dtype=float).reshape(3)
            self.hand_axis = h / (np.linalg.norm(h) + 1e-12)
            self.ee_axis = e / (np.linalg.norm(e) + 1e-12)
        self.orientation_mode = orientation_mode
        self.C = None
        if orientation_mode == "absolute":
            self.C = np.asarray(hand_ee_convention, dtype=float).reshape(3, 3)
        # Operator neutral-pose calibration (POSITION ONLY, body axes — see
        # vr/neutral_calib.py): per-axis scale + offset applied to the ctrl
        # translation before the body→base rotation. Identity by default; only
        # meaningful for body-relative ctrl samples in absolute position mode.
        # The LATERAL scale is non-linear when lat_ref > 0: ~1:1 near the body
        # midline (clapped operator hands stay clapped-width on the robot,
        # instead of being amplified apart), ramping linearly to the full
        # calibrated scale at the operator's neutral lateral |y| = lat_ref.
        self.axis_scale = np.ones(3)
        self.body_offset = np.zeros(3)
        self.lat_ref = 0.0
        self.lat_center = 0.0
        self.lat_knots = None     # [[x_clap, y_contact], [x_spread, y_half_spread]]
        self.anchor_ctrl: SE3 | None = None
        self.anchor_ee: SE3 | None = None
        self._blend_t0: float | None = None
        self._blend_off = np.zeros(3)
        self._ori_off_axis = np.zeros(3)
        self._ori_off_ang = 0.0

    def set_R(self, R: np.ndarray) -> None:
        """Replace the ctrl→base rotation (legacy stance calibration only)."""
        self.R = np.asarray(R, dtype=float).reshape(3, 3)
        self.release()   # force a fresh anchor on next engage

    def set_calibration(self, axis_scale, body_offset, lat_ref: float = 0.0,
                        lat_center: float = 0.0, lat_knots=None) -> None:
        """Install an operator POSITION calibration (body-axes per-axis scale +
        offset; lat_ref enables the non-linear lateral ramp; lat_center = the
        operator's measured midline, mapped onto the robot's midline — absorbs
        the ORBIT recenter-anchor's lateral shift). Releases the anchors so the
        next engage latches a fresh offset and the arm GLIDES onto the new
        correspondence (the same snap-free path as any re-engage)."""
        self.axis_scale = np.asarray(axis_scale, dtype=float).reshape(3)
        self.body_offset = np.asarray(body_offset, dtype=float).reshape(3)
        self.lat_ref = max(0.0, float(lat_ref))
        self.lat_center = float(lat_center)
        self.lat_knots = ([[float(a), float(b)] for a, b in lat_knots]
                          if lat_knots else None)
        self.release()

    @property
    def engaged(self) -> bool:
        return self.anchor_ctrl is not None

    def _lat_scaled(self, lat: float) -> float:
        """Lateral component about the OPERATOR'S measured midline (lat_center —
        the ORBIT anchor shifts it), centred on the ROBOT midline. The curve
        itself lives in `lateral_curve` (shared with the calibration grader)."""
        return lateral_curve(lat - self.lat_center, float(self.axis_scale[0]),
                             self.lat_ref, self.lat_knots)

    def _p_abs(self, ctrl: SE3) -> np.ndarray:
        w = ctrl.translation()
        w = np.array([self._lat_scaled(w[0]),
                      self.axis_scale[1] * w[1],
                      self.axis_scale[2] * w[2]]) + self.body_offset
        return self.chest + self.scale * (self.R @ w)

    def _R_abs(self, ctrl: SE3) -> np.ndarray:
        """Absolute EE attitude (base frame): the operator's hand attitude mapped
        through the body↔world axes (self.R @ ctrl.R is PROPER — the two
        reflections cancel) composed with the fixed hand↔EE convention."""
        return self.R @ ctrl.rotation().as_matrix() @ self.C

    def engage(self, ctrl: SE3, ee: SE3, t: float | None = None) -> None:
        """Latch position AND orientation anchors on the clutch rising edge, so the
        target equals the current EE pose at the engage instant (no jump). In
        absolute modes also latch the EE−absolute offsets; they decay from `t` over
        `engage_blend_s` (t=None holds the offsets until a timed target() call)."""
        self.anchor_ctrl = ctrl
        self.anchor_ee = ee
        if self.mode == "absolute":
            self._blend_off = ee.translation() - self._p_abs(ctrl)
            self._blend_t0 = t
        if self.orientation_mode == "absolute":
            rv = rotvec(self._R_abs(ctrl).T @ ee.rotation().as_matrix())
            self._ori_off_ang = float(np.linalg.norm(rv))
            self._ori_off_axis = rv / self._ori_off_ang if self._ori_off_ang > 1e-9 else np.zeros(3)
            self._blend_t0 = t if t is not None else self._blend_t0

    def release(self) -> None:
        self.anchor_ctrl = None
        self.anchor_ee = None
        self._blend_t0 = None
        self._blend_off = np.zeros(3)

    def target(self, ctrl: SE3, t: float | None = None) -> SE3:
        """EE target (arm base frame) for the current wrist pose while engaged.

        Position: absolute mode targets chest + scale·R·(torso→wrist), plus the
        engage-latched offset decayed over `engage_blend_s` (exact at the engage
        instant, ~5%% left after blend_s, → pure absolute). Hand displacements map
        1:1 in BOTH modes — the blend offset is constant per engage, so deltas are
        identical to relative mode. With t=None the offset is held un-decayed (the
        runtime always passes t; un-timed calls stay snap-free).

        Orientation: the wrist rotation SINCE engage as a left/world-frame delta in
        the ctrl frame (D = R_now·R_anchorᵀ), conjugated into the arm base frame by
        the SAME `R` that maps translations, then applied to the EE anchor in the
        base frame:

            R_target = (R · D · Rᵀ) · R_ee_anchor

        So a hand rotation of θ about a ctrl-frame axis `a` becomes an EE rotation
        of θ about the base-frame axis `R·a` — roll→roll, pitch→pitch, yaw→yaw,
        from ANY starting pose, with no stance/orientation calibration. Continuous
        at engage (D=I ⇒ target=anchor). In body-relative mode both the ctrl basis
        (head_op_axes) and `R` (R_base_from_body) carry one reflection (det −1);
        the conjugation cancels them, so the mapped rotation is proper and lands on
        the physically corresponding world axis (tests/test_frames.py asserts this
        on the real per-side base_quat). The earlier hand-local correspondence
        (anchorᵀ·now conjugated by a calibrated P) is gone: it depended on a 5 s
        arms-at-sides hold that, done imperfectly, scrambled every rotation axis —
        measured at ~145° median axis error on a real Quest session.
        """
        assert self.anchor_ctrl is not None and self.anchor_ee is not None
        if t is None or self._blend_t0 is None:
            k = 1.0
        else:
            k = float(np.exp(-3.0 * max(0.0, t - self._blend_t0) / max(self.blend_s, 1e-6)))
        if self.mode == "absolute":
            p = self._p_abs(ctrl) + k * self._blend_off
        else:
            dp_vr = ctrl.translation() - self.anchor_ctrl.translation()
            p = self.anchor_ee.translation() + self.scale * (self.R @ dp_vr)
        if self.orientation_mode == "absolute":
            # The robot hand WEARS your hand's attitude (mapped through the body↔
            # world axes + the fixed hand↔EE convention). The engage-latched offset
            # decays with the same blend as position, so re-engages glide instead
            # of snapping; after the glide, the overlay hand skeleton and the
            # rendered ORCA hand coincide by construction.
            R_abs = self._R_abs(ctrl)
            if np.linalg.det(R_abs) < 0.0:
                # A REFLECTION here means the ctrl sample's rotation was improper
                # (real devices never produce one; synthetic fixtures can). Never
                # hand the IK a reflection — fail closed on the anchor attitude.
                return SE3.from_rotation_and_translation(self.anchor_ee.rotation(), p)
            Rt = R_abs @ quat_to_R(
                quat_from_axis_angle(self._ori_off_axis, k * self._ori_off_ang))
            return SE3.from_rotation_and_translation(SO3.from_matrix(Rt), p)
        dR = ctrl.rotation().as_matrix() @ self.anchor_ctrl.rotation().as_matrix().T
        A = self.anchor_ee.rotation().as_matrix()
        if self.twist_mode == "intrinsic":
            # Decompose the hand delta into TWIST about the hand's own forearm axis
            # and the residual SWING. The twist drives the EE about ITS OWN tool/j6
            # axis (intrinsic — a wrist turn is always a pure j6 roll, never a j4/j5
            # contortion through the wrist singularity); only real pitch/yaw of the
            # hand swings the EE, mapped through the world frame like translation.
            h = ctrl.rotation().as_matrix() @ self.hand_axis      # hand axis NOW (ctrl frame)
            phi = swing_twist_angle(dR, h)
            dR_swing = dR @ quat_to_R(quat_from_axis_angle(h, -phi))
            Rt = (self.R @ dR_swing @ self.R.T) @ A @ quat_to_R(
                quat_from_axis_angle(self.ee_axis, self._twist_sign() * phi))
        else:
            Rt = (self.R @ dR @ self.R.T) @ A
        return SE3.from_rotation_and_translation(SO3.from_matrix(Rt), p)

    def _twist_sign(self) -> float:
        """Sense pairing between the hand-axis twist angle and the EE tool-axis
        rotation. In body-relative mode the ctrl basis carries a reflection
        (det −1), which mirrors the measured twist angle relative to the physical
        pronation; the world-frame swing path cancels it via conjugation, but the
        intrinsic path must compensate explicitly so that, when the hand axis and
        the EE tool axis happen to align in space, intrinsic and world mapping
        agree (pinned by tests/test_frames.py)."""
        return -1.0 if np.linalg.det(self.R) < 0 else 1.0
