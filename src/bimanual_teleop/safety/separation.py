"""Pairwise hand minimum-separation guard.

The per-side anti-cross clamp (world-Y half-spaces) keeps each arm on its own
side, but it neither models hand VOLUME nor acts along the actual line between
the hands — clapping the operator's hands drove the robot hands through each
other. This guard keeps the two wrist target points at least `d_min` apart in
full 3D: when they get closer, both are pushed apart symmetrically along their
connecting line, so the hands meet and STOP at contact distance, and slide along
the contact plane instead of interpenetrating.

A disengaged/parked arm is an obstacle, not a participant: its actual wrist
position enters the math but only engaged sides get moved (move_left/move_right).

Run AFTER the workspace and anti-cross clamps. The order is safe by
construction: the anti-cross clamp guarantees p_right.y ≥ +gap > −gap ≥
p_left.y, so the connecting line always has a rightward Y component — the push
moves each hand DEEPER into its own half-space, never across the midline.
"""
from __future__ import annotations

import numpy as np

# Degenerate fallback (targets coincide): push along world Y, the left/right axis.
_FALLBACK_AXIS = np.array([0.0, 1.0, 0.0])


def separate_targets(p_left: np.ndarray, p_right: np.ndarray, d_min: float, *,
                     move_left: bool = True, move_right: bool = True
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return (p_left', p_right') at least `d_min` apart (world frame, metres).

    Already-apart pairs come back unchanged. A movable pair splits the push
    half/half; with one side frozen the movable side takes the full push; with
    both frozen the input is returned as-is (nothing to actuate)."""
    p_l = np.asarray(p_left, dtype=float).reshape(3).copy()
    p_r = np.asarray(p_right, dtype=float).reshape(3).copy()
    if d_min <= 0.0 or not (move_left or move_right):
        return p_l, p_r
    d = p_r - p_l
    dist = float(np.linalg.norm(d))
    if dist >= d_min:
        return p_l, p_r
    axis = d / dist if dist > 1e-9 else _FALLBACK_AXIS
    need = d_min - dist
    if move_left and move_right:
        p_l -= axis * (need / 2.0)
        p_r += axis * (need / 2.0)
    elif move_left:
        p_l -= axis * need
    else:
        p_r += axis * need
    return p_l, p_r


def closest_points_segments(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Closest points between segments [a0,a1] and [b0,b1] (world frame).
    Standard clamped parametric solution; robust to degenerate (point) segments."""
    a0 = np.asarray(a0, dtype=float); a1 = np.asarray(a1, dtype=float)
    b0 = np.asarray(b0, dtype=float); b1 = np.asarray(b1, dtype=float)
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    A = float(u @ u); B = float(u @ v); C = float(v @ v)
    D = float(u @ w); E = float(v @ w)
    den = A * C - B * B
    s = 0.0 if den < 1e-12 else np.clip((B * E - C * D) / den, 0.0, 1.0)
    t = (B * s + E) / C if C > 1e-12 else 0.0
    t = float(np.clip(t, 0.0, 1.0))
    # re-clamp s against the clamped t (one Gauss-Seidel pass closes the corner cases)
    s = float(np.clip((B * t - D) / A, 0.0, 1.0)) if A > 1e-12 else 0.0
    return a0 + s * u, b0 + t * v


def separate_capsules(w_l: np.ndarray, w_r: np.ndarray, dir_l: np.ndarray, dir_r: np.ndarray,
                      length: float, d_min: float, *,
                      move_left: bool = True, move_right: bool = True
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Keep two HAND CAPSULES ≥ d_min apart; returns adjusted WRIST anchors.

    Each hand is modeled as the segment wrist → wrist + length·fingers_dir (the
    ORCA volume from wrist to fingertips). The minimum distance between the two
    segments is what claps actually violate — palms-facing hands collide at the
    palm centers, fingers-pointing hands at the tips, and a point-pair guard at
    any single offset misses one or the other (measured: fingertips reached
    0.4 cm from the other palm with the wrist points dutifully 17 cm apart).
    The push shifts whole capsules (their wrist anchors) along the line between
    the closest points; a frozen side is an obstacle."""
    w_l = np.asarray(w_l, dtype=float).reshape(3).copy()
    w_r = np.asarray(w_r, dtype=float).reshape(3).copy()
    if d_min <= 0.0 or not (move_left or move_right):
        return w_l, w_r
    u_l = np.asarray(dir_l, dtype=float).reshape(3) * length
    u_r = np.asarray(dir_r, dtype=float).reshape(3) * length
    n_l = u_l / (np.linalg.norm(u_l) + 1e-12)
    n_r = u_r / (np.linalg.norm(u_r) + 1e-12)
    # PUSH DIRECTION: the closest-point axis is the true distance gradient, but
    # for NEAR-PARALLEL overlapping capsules (a full clasp — real hands meet
    # within ~±30° of parallel) the closest pair sits at staggered points
    # (fingertip vs palm middle) and that axis tilts into fore/aft+vertical:
    # the guard then SHEARS the hands past each other instead of stopping them
    # face to face (measured on a real clap tape: 48% of pushes were majority
    # shear, p90 ≈ pure shear — the operator sees the hands "cross"). Blend the
    # axis toward the WRIST-TO-WRIST line as the capsule axes approach
    # (anti-)parallel: a clasp resolves laterally apart, while a perpendicular
    # fingertip poke (λ≈0) keeps the gradient axis it genuinely needs.
    lam = abs(float(n_l @ n_r))
    # Iterate: one distance-based push under-resolves INTERSECTING capsules
    # (anti-parallel overlap — distance 0 says nothing about depth) and a
    # blended axis is not the exact gradient; each pass recomputes the closest
    # points after the previous shift and the pair converges in 2-3 passes.
    for _ in range(4):
        c_l, c_r = closest_points_segments(w_l, w_l + u_l, w_r, w_r + u_r)
        d = c_r - c_l
        dist = float(np.linalg.norm(d))
        if dist >= d_min:
            break
        wd = w_r - w_l
        wn = float(np.linalg.norm(wd))
        need = d_min - dist
        if dist > 1e-9:
            grad = d / dist
            axis = grad
            if lam > 0.0 and wn > 1e-9 and float(grad @ (wd / wn)) >= 0.0:
                axis = (1.0 - lam) * grad + lam * (wd / wn)
                axis /= (np.linalg.norm(axis) + 1e-12)
                # the blended axis is not the exact gradient: scale the push so
                # its gradient COMPONENT closes the full gap (clamped — the
                # iteration finishes pathological geometries)
                need /= max(float(axis @ grad), 0.3)
        else:
            # Deep intersection: no gradient. The wrist line (anti-cross keeps
            # it laterally ordered) beats a fixed world axis — e.g. a vertical
            # clap resolves vertically; world-Y only when wrists coincide too.
            axis = wd / wn if wn > 1e-9 else _FALLBACK_AXIS
        if move_left and move_right:
            w_l -= axis * (need / 2.0)
            w_r += axis * (need / 2.0)
        elif move_left:
            w_l -= axis * need
        else:
            w_r += axis * need
    return w_l, w_r
