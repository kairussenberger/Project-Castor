"""ORBIT frame-origin reconciliation (vr/orbit_source.py).

Measured on a real session (2026-06-11): ORBIT streams the HEAD pose
floor-anchored (standing head y ≈ 1.4) but the WRIST poses eye-anchored
(resting hands y ≈ −0.3). The absolute body-relative mapping subtracts a
head-derived torso from the wrist, so the two streams MUST share an origin —
`latest()` re-anchors wrist translations at the live head position (rotation
untouched). These tests pin that reconstruction and the passthrough mode.

    uv run pytest tests/test_orbit_anchor.py -q
"""
from __future__ import annotations

import time

import numpy as np

from bimanual_teleop.vr.calibrate import body_relative_hand_sample
from bimanual_teleop.vr.orbit_source import OrbitVRSource


def _src(anchor: str) -> OrbitVRSource:
    rig = {"vr": {"orbit_adb_reverse": False, "orbit_viz": False,
                  "orbit_timeout": 5.0, "orbit_head_timeout": 5.0,
                  "orbit_wrist_anchor": anchor}}
    return OrbitVRSource(rig)


def _inject(src: OrbitVRSource, head_pos, wrist_pos, side="right", lm0=None) -> None:
    now = time.monotonic()
    H = np.eye(4)
    H[:3, 3] = head_pos
    W = np.eye(4)
    W[:3, 3] = wrist_pos
    src._head = H
    src._head_last = now
    src._wrist[side] = W
    src._wrist_last[side] = now
    if lm0 is not None:
        lm = np.zeros((25, 3))
        lm[0] = lm0
        src._lm[side] = lm
        src._lm_last[side] = now


def test_head_anchor_recombines_origins():
    src = _src("head")
    _inject(src, head_pos=[0.1, 1.40, 0.05], wrist_pos=[0.2, -0.30, -0.35])
    f = src.latest()
    # wrist translation re-anchored at the head; rotation untouched
    np.testing.assert_allclose(f.hands["right"].wrist[:3, 3], [0.3, 1.10, -0.30], atol=1e-12)
    np.testing.assert_allclose(f.hands["right"].wrist[:3, :3], np.eye(3), atol=1e-12)
    # end-to-end: the body-relative UP now reads anatomically (≈ +0.05 for a
    # wrist 30 cm below the eyes with the torso proxy 35 cm below them) instead
    # of the phantom ≈ −1.3 m the origin mismatch produced.
    hs = body_relative_hand_sample(f.hands["right"], f.head, (0.0, -0.35, 0.0))
    assert abs(hs.wrist[1, 3] - 0.05) < 1e-9


def test_world_mode_is_raw_passthrough():
    src = _src("world")
    _inject(src, head_pos=[0.1, 1.40, 0.05], wrist_pos=[0.2, -0.30, -0.35])
    f = src.latest()
    np.testing.assert_allclose(f.hands["right"].wrist[:3, 3], [0.2, -0.30, -0.35], atol=1e-12)


def test_head_anchor_without_head_leaves_wrist_raw():
    """No fresh head ⇒ no re-anchor (and the body-relative gate downstream
    already fails closed without a head pose — the arm never sees it)."""
    src = _src("head")
    _inject(src, head_pos=[0.0, 1.40, 0.0], wrist_pos=[0.2, -0.30, -0.35])
    src._head_last = 0.0                      # head never seen
    f = src.latest()
    assert f.head is None
    np.testing.assert_allclose(f.hands["right"].wrist[:3, 3], [0.2, -0.30, -0.35], atol=1e-12)
    assert body_relative_hand_sample(f.hands["right"], f.head, (0, -0.35, 0)).tracked is False


def test_keypoint_anchor_is_recenter_proof():
    """'keypoint' rebuilds the wrist as head + keypoint[0]: whatever arbitrary
    anchor the wrist stream carries (a recenter moved it 0.5 m once), the
    reconstruction is unaffected. Rotation still comes from the wrist stream."""
    src = _src("keypoint")
    # wrist stream anchored somewhere crazy; keypoint says 30 cm below, 35 fwd
    _inject(src, head_pos=[0.0, 1.40, 0.0], wrist_pos=[9.9, -9.9, 9.9],
            lm0=[0.05, -0.30, -0.35])
    f = src.latest()
    np.testing.assert_allclose(f.hands["right"].wrist[:3, 3], [0.05, 1.10, -0.35], atol=1e-12)
    hs = body_relative_hand_sample(f.hands["right"], f.head, (0.0, -0.35, 0.0))
    assert abs(hs.wrist[1, 3] - 0.05) < 1e-9          # up reads anatomically


def test_keypoint_anchor_falls_back_without_landmarks():
    src = _src("keypoint")
    _inject(src, head_pos=[0.1, 1.40, 0.05], wrist_pos=[0.2, -0.30, -0.35])  # no lm
    f = src.latest()
    np.testing.assert_allclose(f.hands["right"].wrist[:3, 3], [0.3, 1.10, -0.30], atol=1e-12)
