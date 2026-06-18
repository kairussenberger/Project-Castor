"""Fast, hardware-free tests for the teleop pipeline's pure logic + sim wiring.

    uv run python -m pytest tests/ -q      (or just: uv run python tests/test_pipeline.py)
"""
from __future__ import annotations

import time

import numpy as np


def _free_tcp_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_joint_name_map_roundtrip():
    from bimanual_teleop.hands.joint_map import ORCA_JOINT_ORDER, orca_to_sim_short, sim_short_to_orca
    for j in ["wrist", "thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip",
              "index_abd", "index_mcp", "index_pip", "pinky_pip", "middle_mcp"]:
        assert sim_short_to_orca(orca_to_sim_short(j)) == j
    assert orca_to_sim_short("thumb_dip") == "t-pip"      # the tricky one
    assert orca_to_sim_short("index_mcp") == "i-mcp"
    assert len(ORCA_JOINT_ORDER) == 17
    assert ORCA_JOINT_ORDER[0] == "wrist"


def test_quest_retarget_open_to_fist():
    from bimanual_teleop.hands.quest_retarget import quest_to_orca, synthetic_webxr_hand
    from bimanual_teleop.hands.joint_map import ORCA_JOINT_ORDER, load_hand_config
    from bimanual_teleop.hands import retarget_core as rc
    from bimanual_teleop.render_sink import ordered_hand_state
    neutral, roms = load_hand_config("orcahand_right")
    op = rc.clamp_to_rom(quest_to_orca(synthetic_webxr_hand(0.0), neutral, mirror=True), roms)
    fist = rc.clamp_to_rom(quest_to_orca(synthetic_webxr_hand(1.0), neutral, mirror=True), roms)
    # fingers curl more in a fist; all outputs stay within ROM
    for f in ("index", "middle", "ring", "pinky"):
        assert fist[f"{f}_mcp"] > op[f"{f}_mcp"] + 20
        assert fist[f"{f}_pip"] >= op[f"{f}_pip"]
    for j, v in fist.items():
        lo, hi = roms[j]
        assert lo - 1e-6 <= v <= hi + 1e-6
    op_render = ordered_hand_state(op)
    fist_render = ordered_hand_state(fist)
    assert op_render["names"] == ORCA_JOINT_ORDER
    assert len(fist_render["q"]) == len(ORCA_JOINT_ORDER)
    curl_idx = [ORCA_JOINT_ORDER.index(k) for k in ("index_mcp", "index_pip", "middle_mcp", "middle_pip")]
    assert np.mean([fist_render["q"][i] for i in curl_idx]) > np.mean([op_render["q"][i] for i in curl_idx]) + 20


def test_clutch_mapper_relative_zero_motion_on_engage():
    from bimanual_teleop.vr.frames import SE3, SO3
    from bimanual_teleop.vr.frames import ClutchMapper
    m = ClutchMapper(np.eye(3), pos_scale=1.0)
    ee = SE3.from_translation(np.array([0.3, 0.1, 0.5]))
    ctrl = SE3.from_translation(np.array([1.0, 2.0, 3.0]))
    m.engage(ctrl, ee)
    # no controller motion -> target == anchored EE
    tgt = m.target(ctrl)
    assert np.allclose(tgt.translation(), ee.translation(), atol=1e-9)
    # +5cm controller x -> +5cm EE x (scale 1, identity R)
    moved = SE3.from_translation(np.array([1.05, 2.0, 3.0]))
    assert np.allclose(m.target(moved).translation(), ee.translation() + [0.05, 0, 0], atol=1e-9)


def test_arm_ik_converges():
    from bimanual_teleop.vr.frames import SE3, SO3
    from bimanual_teleop.arms.ik import ArmIK
    from bimanual_teleop.config import load_rig
    ik = ArmIK(load_rig(), "left")
    T0 = ik.fk_wrist()                 # position IK targets the WRIST site
    # Move UP+forward into the workspace — the direction teleop actually drives from
    # the arms-down home. (A target further DOWN sits near the hanging arm's reach
    # boundary, where any IK is stiff; that's not what this convergence test checks.)
    tgt = T0.translation() + np.array([0.07, 0.0, 0.05])
    target = SE3.from_rotation_and_translation(ik.fk_ee().rotation(), tgt)
    for _ in range(300):
        ik.solve(target)
    assert np.linalg.norm(ik.fk_wrist().translation() - tgt) < 5e-3


def test_supervisor_estop_and_staleness():
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.safety.supervisor import Supervisor
    from bimanual_teleop.safety.clutch import AlwaysOn
    from bimanual_teleop.vr.frames import VRFrame, HandSample
    rig = load_rig()
    sup = Supervisor(rig, AlwaysOn())
    fresh = VRFrame(stamp=10.0, hands={s: HandSample(tracked=True) for s in ("left", "right")})
    eng = sup.update(fresh, t=10.0)
    assert eng["left"] and eng["right"]
    # stale frame well past hold window -> not engaged
    eng = sup.update(fresh, t=10.0 + rig["safety"]["hold_s"] + 1.0)
    assert not eng["left"] and not eng["right"]
    sup.estop()
    assert not any(sup.update(fresh, t=12.0).values())


def test_zmq_loopback_latest_value():
    from bimanual_teleop.bus.zmq_io import Publisher, LatestSub
    from bimanual_teleop.bus import topics
    ep = f"tcp://127.0.0.1:{_free_tcp_port()}"
    pub = Publisher(ep)
    sub = LatestSub(ep, [topics.ARM_CMD])
    time.sleep(0.2)  # PUB/SUB slow-joiner
    for q in range(5):
        pub.send(topics.ARM_CMD, topics.msg(stamp=float(q), side="left", q=np.full(6, q)))
    time.sleep(0.05)
    sub.poll()
    got = sub.get(topics.ARM_CMD)
    assert got is not None and got["stamp"] == 4.0  # only the latest survives
    pub.close(); sub.close()


def test_clutch_release_disengages_immediately():
    """Releasing the deadman on a LIVE feed must stop following at once (the HOLD
    window is only for tracking dropouts) — the deadman bug the review caught."""
    from bimanual_teleop.config import load_rig, SIDES
    from bimanual_teleop.safety.supervisor import Supervisor
    from bimanual_teleop.safety.clutch import KeyboardClutch
    from bimanual_teleop.vr.frames import VRFrame, HandSample
    rig = load_rig()
    kc = KeyboardClutch()
    sup = Supervisor(rig, kc)
    fr = VRFrame(stamp=100.0, hands={s: HandSample(tracked=True) for s in SIDES})
    kc.held = True
    assert sup.update(fr, 100.0)["left"]
    kc.held = False
    fr2 = VRFrame(stamp=100.05, hands={s: HandSample(tracked=True) for s in SIDES})
    assert not sup.update(fr2, 100.05)["left"]   # immediate, NOT after hold_s


def test_absolute_position_glides_to_chest_correspondence():
    """Absolute mode: the target is chest + R·(torso→wrist) — continuous at the
    engage instant (offset latched), converged onto correspondence after the blend,
    and displacement deltas map 1:1 from the first instant in both modes."""
    from bimanual_teleop.vr.frames import SE3, ClutchMapper
    chest = np.array([0.1, -0.2, 0.4])
    m = ClutchMapper(np.eye(3), pos_scale=1.0, position_mode="absolute",
                     chest_base=chest, engage_blend_s=0.5)
    ee = SE3.from_translation(np.array([0.3, 0.1, 0.5]))
    ctrl0 = SE3.from_translation(np.array([0.05, 0.0, 0.45]))     # torso→wrist (body axes)
    m.engage(ctrl0, ee, t=10.0)
    # engage instant: exactly the current EE (no snap)
    assert np.allclose(m.target(ctrl0, 10.0).translation(), ee.translation(), atol=1e-9)
    # displacement maps 1:1 immediately (blend offset is constant per engage)
    ctrl1 = SE3.from_translation(ctrl0.translation() + [0.07, 0.0, 0.0])
    d = m.target(ctrl1, 10.0).translation() - m.target(ctrl0, 10.0).translation()
    assert np.allclose(d, [0.07, 0, 0], atol=1e-9)
    # long after the blend: pure absolute correspondence
    assert np.allclose(m.target(ctrl0, 20.0).translation(), chest + ctrl0.translation(), atol=1e-6)


def test_absolute_hands_in_front_put_robot_hands_in_front():
    """The complaint this mode fixes: operator holds BOTH hands out in front of
    their torso, so the robot's wrists must end up IN FRONT of the robot (world −X
    of the chest anchor) at comparable height — not hanging at its sides."""
    from bimanual_teleop.arms.arm_control import ArmController
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.vr.calibrate import body_relative_hand_sample, head_op_axes
    from bimanual_teleop.vr.frames import HandSample, quat_to_R

    rig = load_rig()
    assert rig["mapping"]["position_mode"] == "absolute"
    head = np.eye(4)
    head[:3, 3] = [0.0, 1.6, 0.0]
    anchor_w = 0.5 * (np.asarray(rig["arms"]["left"]["base_pos"])
                      + np.asarray(rig["arms"]["right"]["base_pos"])) \
        - np.array([0.0, 0.0, float(rig["mapping"].get("body_anchor_drop", 0.15))])
    for side, lateral in (("left", -0.22), ("right", 0.22)):
        ac = ArmController(rig, side)
        wrist_body = np.array([lateral, 0.05, 0.42])      # in front, near chest height
        W = np.eye(4)
        op = head_op_axes(head)
        torso_w = head[:3, 3] + op @ np.asarray(rig["vr"]["torso_from_head"])
        W[:3, 3] = torso_w + op @ wrist_body
        hs = body_relative_hand_sample(HandSample(tracked=True, wrist=W), head,
                                       rig["vr"]["torso_from_head"])
        t = 0.0
        for _ in range(int(4.0 * 120)):                    # well past the engage blend
            t += 1 / 120
            ac.update(hs, True, t)
        cmd_w = quat_to_R(rig["arms"][side]["base_quat"]) @ ac.cmd_pos \
            + np.asarray(rig["arms"][side]["base_pos"])
        assert cmd_w[0] < anchor_w[0] - 0.25, (side, cmd_w)        # in FRONT (−X)
        assert abs(cmd_w[2] - (anchor_w[2] + 0.05)) < 0.12, (side, cmd_w)  # chest height
        assert (cmd_w[1] < 0) == (side == "left"), (side, cmd_w)   # own side


def test_orientation_continuous_at_engage():
    """The target must equal the anchored EE pose at the engage instant
    (the ~157° wrist snap the review caught)."""
    from bimanual_teleop.vr.frames import SE3, SO3
    from bimanual_teleop.vr.frames import ClutchMapper, euler_to_R
    R_ee = euler_to_R([0.5, -0.3, 0.8])
    R_ctrl = euler_to_R([1.1, 0.2, -0.4])
    ee = SE3.from_rotation_and_translation(SO3.from_matrix(R_ee), np.array([0.3, 0.1, 0.5]))
    ctrl = SE3.from_rotation_and_translation(SO3.from_matrix(R_ctrl), np.array([1.0, 2.0, 3.0]))
    m = ClutchMapper(euler_to_R([0.2, 0.0, 1.5]), pos_scale=1.0)
    m.engage(ctrl, ee)
    tgt = m.target(ctrl)
    assert np.allclose(tgt.rotation().as_matrix(), R_ee, atol=1e-9)
    assert np.allclose(tgt.translation(), ee.translation(), atol=1e-9)


def test_body_relative_wrist_cancels_head_translation_and_yaw():
    """The arm-control wrist pose is the operator torso -> wrist vector, not
    raw XR-world wrist position. Moving/yawing the head with the hand fixed relative
    to the body must produce the same controller pose."""
    from bimanual_teleop.vr.calibrate import body_relative_hand_sample, head_op_axes
    from bimanual_teleop.vr.frames import HandSample, euler_to_R

    def pose(R, p):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    torso_from_head = np.array([0.0, -0.35, 0.0])
    torso_to_wrist = np.array([0.25, 0.27, 0.55])  # right, up, forward from torso
    head_to_wrist = torso_from_head + torso_to_wrist

    head0 = pose(np.eye(3), [0.0, 1.6, 0.0])
    op0 = head_op_axes(head0)
    hand0 = HandSample(tracked=True, wrist=pose(np.eye(3), head0[:3, 3] + op0 @ head_to_wrist))

    head1 = pose(euler_to_R([0.0, 0.7, 0.0]), [0.4, 1.7, -0.2])
    op1 = head_op_axes(head1)
    hand1 = HandSample(tracked=True, wrist=pose(head1[:3, :3], head1[:3, 3] + op1 @ head_to_wrist))  # rigid: hand carries the body turn

    rel0 = body_relative_hand_sample(hand0, head0, torso_from_head)
    rel1 = body_relative_hand_sample(hand1, head1, torso_from_head)
    assert np.allclose(rel0.wrist[:3, 3], torso_to_wrist, atol=1e-9)
    assert np.allclose(rel1.wrist[:3, 3], torso_to_wrist, atol=1e-9)
    assert np.allclose(rel0.wrist[:3, :3], rel1.wrist[:3, :3], atol=1e-9)


def test_calibration_stillness_uses_body_relative_wrist_positions():
    """Calibration quality should not fail just because the headset/torso shifts
    while the wrist is stable relative to the torso."""
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.hands.quest_retarget import synthetic_webxr_hand
    from bimanual_teleop.vr.calibrate import Calibrator, head_op_axes

    rig = load_rig()
    rig["vr"]["torso_from_head"] = [0.0, -0.35, 0.0]
    cal = Calibrator(rig)
    torso = np.asarray(rig["vr"]["torso_from_head"], dtype=float)
    torso_to_wrist = np.array([0.22, 0.30, 0.45])
    lm = synthetic_webxr_hand(0.0)
    for i in range(12):
        head = np.eye(4)
        head[:3, 3] = [0.003 * i, 1.6 + 0.002 * i, -0.004 * i]
        op = head_op_axes(head)
        wrist = np.eye(4)
        wrist[:3, :3] = op
        wrist[:3, 3] = head[:3, 3] + op @ (torso + torso_to_wrist)
        cal.add("left", lm, wrist, head)
    res = cal.result("left")
    assert res is not None
    assert res["ok"] is True
    assert res["std"] < 1e-9


def test_calibration_ignores_non_finite_pose_samples():
    """Bad custom/replay matrices must not poison calibration averages with NaNs."""
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.hands.quest_retarget import synthetic_webxr_hand
    from bimanual_teleop.vr.calibrate import Calibrator

    cal = Calibrator(load_rig())
    lm = synthetic_webxr_hand(0.0)
    head = np.eye(4)
    wrist = np.eye(4)

    bad_lm = lm.copy()
    bad_lm[0, 0] = float("nan")
    cal.add("left", bad_lm, wrist, head)
    assert cal.count("left") == 1  # finite wrist still counts; bad landmarks ignored
    assert len(cal._samples["left"]) == 0

    bad_wrist = wrist.copy()
    bad_wrist[0, 3] = float("inf")
    for _ in range(10):
        cal.add("right", lm, bad_wrist, head)
    assert len(cal._wrists["right"]) == 0
    assert cal.result("right") is None


def test_body_relative_wrist_orientation_delta_is_head_invariant():
    """Wrist orientation is also expressed in body coordinates: the same hand-local
    twist should produce the same controller delta even if the head pose changes."""
    from bimanual_teleop.vr.calibrate import body_relative_hand_sample, head_op_axes
    from bimanual_teleop.vr.frames import HandSample, euler_to_R

    def pose(R, p):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def ctrl_delta(head):
        op = head_op_axes(head)
        hand_rel = euler_to_R([0.2, -0.4, 0.7])
        twist = euler_to_R([0.0, 0.0, 0.6])  # hand-local twist
        ref = HandSample(tracked=True, wrist=pose(op @ hand_rel, head[:3, 3] + op @ [0.2, 0.0, 0.5]))
        now = HandSample(tracked=True, wrist=pose(op @ hand_rel @ twist, head[:3, 3] + op @ [0.2, 0.0, 0.5]))
        cref = body_relative_hand_sample(ref, head).wrist[:3, :3]
        cnow = body_relative_hand_sample(now, head).wrist[:3, :3]
        return cref, cref.T @ cnow, twist

    cref0, d0, twist = ctrl_delta(pose(np.eye(3), [0.0, 1.6, 0.0]))
    cref1, d1, _ = ctrl_delta(pose(euler_to_R([0.0, 0.8, 0.0]), [0.4, 1.7, -0.2]))
    assert np.allclose(cref0, cref1, atol=1e-8)
    assert np.allclose(d0, twist, atol=1e-8)
    assert np.allclose(d1, twist, atol=1e-8)


def test_body_relative_arm_sample_requires_head_pose():
    """A tracked hand without a head pose must not fall back to raw XR-world arm
    motion in body-relative mode; that would reintroduce blind direction-vector
    control. Finger landmarks remain available to the hand retargeter."""
    from bimanual_teleop.engine import TeleopEngine
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.hands.quest_retarget import synthetic_webxr_hand
    from bimanual_teleop.vr.frames import HandSample, VRFrame

    class Sink:
        def set_arm(self, side, q):
            pass

        def set_hand(self, side, joints):
            pass

    rig = load_rig()
    rig["vr"]["calib_seconds"] = 0
    rig["vr"]["body_relative"] = True
    engine = TeleopEngine(rig, Sink())

    wrist = np.eye(4)
    wrist[:3, 3] = [2.0, 2.0, -1.0]  # deliberately far raw XR-world pose
    landmarks = synthetic_webxr_hand(0.4)
    raw = HandSample(tracked=True, wrist=wrist, landmarks=landmarks, pinch=0.7)
    frame = VRFrame(stamp=0.0, head=None, hands={"left": raw})

    arm_sample = engine._arm_hand_sample(raw, frame)
    assert arm_sample is not None
    assert arm_sample.tracked is False
    assert np.allclose(arm_sample.wrist, raw.wrist)
    assert arm_sample.landmarks is landmarks
    assert raw.tracked is True


def test_body_relative_arm_sample_rejects_non_finite_pose_matrices():
    """A malformed custom/replay source must fail closed at the body-relative
    boundary even if live transports normally filter bad matrices upstream."""
    from bimanual_teleop.hands.quest_retarget import synthetic_webxr_hand
    from bimanual_teleop.vr.calibrate import body_relative_hand_sample
    from bimanual_teleop.vr.frames import HandSample

    lm = synthetic_webxr_hand(0.2)
    wrist = np.eye(4)
    head = np.eye(4)

    bad_head = head.copy()
    bad_head[0, 3] = float("nan")
    out = body_relative_hand_sample(HandSample(tracked=True, wrist=wrist, landmarks=lm), bad_head)
    assert out.tracked is False
    assert out.landmarks is lm
    assert np.all(np.isfinite(out.wrist))

    bad_wrist = wrist.copy()
    bad_wrist[1, 3] = float("inf")
    out = body_relative_hand_sample(HandSample(tracked=True, wrist=bad_wrist, landmarks=lm), head)
    assert out.tracked is False
    assert out.landmarks is lm
    assert np.all(np.isfinite(out.wrist))

    out = body_relative_hand_sample(
        HandSample(tracked=True, wrist=wrist, landmarks=lm),
        head,
        torso_from_head=[0.0, float("nan"), 0.0],
    )
    assert out.tracked is False
    assert out.landmarks is lm
    assert np.all(np.isfinite(out.wrist))


def test_engine_body_motion_does_not_drive_arm_but_hand_lift_does():
    """End-to-end through TeleopEngine: translating/yawing the headset while the
    torso-to-wrist vector is fixed should hold the arm target; lifting the wrist
    relative to the torso should move the arm."""
    from bimanual_teleop.config import load_rig, SIDES
    from bimanual_teleop.engine import TeleopEngine
    from bimanual_teleop.hands.quest_retarget import synthetic_webxr_hand
    from bimanual_teleop.vr.calibrate import head_op_axes
    from bimanual_teleop.vr.frames import HandSample, VRFrame, euler_to_R

    class Sink:
        def __init__(self):
            self.arm = {}
            self.hand = {}

        def set_arm(self, side, q):
            self.arm[side] = np.asarray(q)

        def set_hand(self, side, joints):
            self.hand[side] = dict(joints)

    def pose(R, p):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def frame(head, torso_to_wrist, ref_head=None):
        # Models the REAL ORBIT stream + reconstruction: wrist translation rides
        # the head POSITION (head + keypoint), but NOT the head ROTATION. The
        # offsets are built in REF_HEAD's frame so only the hand pose itself
        # defines them — a later head turn/translation changes nothing physical.
        ref = head if ref_head is None else ref_head
        op = head_op_axes(ref)
        torso = np.array([0.0, -0.35, 0.0])
        hands = {}
        for s in SIDES:
            ttw = np.asarray(torso_to_wrist, dtype=float).copy()
            if s == "right":
                ttw[0] = -ttw[0]   # mirrored: hands APART so no pair/capsule guard binds
            wrist = pose(ref[:3, :3], head[:3, 3] + op @ (torso + ttw))
            hands[s] = HandSample(tracked=True, wrist=wrist.copy(), landmarks=synthetic_webxr_hand(0.0))
        return VRFrame(stamp=0.0, head=head, hands=hands)

    rig = load_rig()
    rig["vr"]["calib_seconds"] = 0
    rig["vr"]["body_relative"] = True
    rig["vr"]["torso_from_head"] = [0.0, -0.35, 0.0]
    engine = TeleopEngine(rig, Sink())
    engaged = {s: True for s in SIDES}

    # A reachable working pose for the LEFT arm under absolute mapping: on the
    # left side of the torso (the anti-cross guard correctly pins a left hand
    # that crosses center), at torso height (the YAM cannot reach far above its
    # base plates), forward where a +16 cm lift stays followable.
    torso_to_wrist = np.array([-0.22, 0.0, 0.40])
    head0 = pose(np.eye(3), [0.0, 1.6, 0.0])
    # settle the absolute-mode engage glide + IK onto the static target first
    for i in range(480):
        engine.tick(frame(head0, torso_to_wrist), engaged, i / 120.0)
    p0 = engine.arm["left"].ik.fk_wrist().translation()

    # SAFETY CONTRACT: head motion (rotation AND translation — looking around,
    # pulling the headset off) must produce ZERO arm input. The hands stay
    # physically still (their head-anchored stream values are unchanged).
    head_moved = pose(euler_to_R([0.0, 0.6, 0.0]), [0.35, 1.72, -0.25])
    for i in range(480, 520):
        engine.tick(frame(head_moved, torso_to_wrist, ref_head=head0), engaged, i / 120.0)
    p_same = engine.arm["left"].ik.fk_wrist().translation()
    assert np.linalg.norm(p_same - p0) < 1e-4

    lifted = torso_to_wrist + np.array([0.0, 0.16, 0.0])
    for i in range(520, 760):
        engine.tick(frame(head_moved, lifted, ref_head=head0), engaged, i / 120.0)
    p_lift = engine.arm["left"].ik.fk_wrist().translation()
    assert np.linalg.norm(p_lift - p_same) > 0.02


def test_calibration_completion_installs_body_relative_mapping():
    """After the rest-pose calibration window, body-relative mode should use the
    body-frame base mapping that runtime wrist poses are expressed in."""
    from bimanual_teleop.config import load_rig, SIDES
    from bimanual_teleop.engine import TeleopEngine
    from bimanual_teleop.hands.quest_retarget import synthetic_webxr_hand
    from bimanual_teleop.vr.calibrate import R_base_from_body, head_op_axes
    from bimanual_teleop.vr.frames import HandSample, VRFrame

    class Sink:
        def set_arm(self, side, q):
            pass

        def set_hand(self, side, joints):
            pass

    rig = load_rig()
    rig["vr"]["calib_seconds"] = 0.08
    rig["vr"]["body_relative"] = True
    engine = TeleopEngine(rig, Sink())
    head = np.eye(4)
    head[:3, 3] = [0.0, 1.6, 0.0]
    op = head_op_axes(head)
    torso = np.asarray(rig["vr"]["torso_from_head"], dtype=float)
    torso_to_wrist = {"left": np.array([-0.25, 0.25, 0.45]),
                      "right": np.array([0.25, 0.25, 0.45])}
    lm = synthetic_webxr_hand(0.0)
    for i in range(10):
        hands = {}
        for s in SIDES:
            wrist = np.eye(4)
            wrist[:3, :3] = op
            wrist[:3, 3] = head[:3, 3] + op @ (torso + torso_to_wrist[s])
            hands[s] = HandSample(tracked=True, wrist=wrist, landmarks=lm)
        engine.tick(VRFrame(stamp=i / 100.0, head=head, hands=hands),
                    {s: False for s in SIDES}, i / 100.0)
    assert engine.calibrated is True
    for s in SIDES:
        assert np.allclose(engine.arm[s].mapper.R, R_base_from_body(rig["arms"][s]["base_quat"]))


def test_operator_debug_state_exposes_invariant_torso_vectors():
    """The Unity-visible op.hands.*.wrist_body payload must expose the same
    body-relative vector as arm control: invariant to head motion, changing on lift."""
    from bimanual_teleop.config import SIDES
    from bimanual_teleop.hands.quest_retarget import synthetic_webxr_hand
    from bimanual_teleop.render_sink import operator_debug_state
    from bimanual_teleop.vr.calibrate import head_op_axes
    from bimanual_teleop.vr.frames import HandSample, VRFrame, euler_to_R

    def pose(R, p):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def frame(head, torso_from_head, torso_to_wrist):
        op = head_op_axes(head)
        wrist = pose(head[:3, :3], head[:3, 3] + op @ (torso_from_head + torso_to_wrist))  # rigid + proper
        hands = {s: HandSample(tracked=True, wrist=wrist.copy(), landmarks=synthetic_webxr_hand(0.0)) for s in SIDES}
        return VRFrame(stamp=0.0, head=head, hands=hands)

    torso = np.array([0.0, -0.35, 0.0])
    torso_to_wrist = np.array([0.2, 0.3, 0.5])
    head0 = pose(np.eye(3), [0.0, 1.6, 0.0])
    head1 = pose(euler_to_R([0.0, 0.5, 0.0]), [0.3, 1.7, -0.2])
    op0 = operator_debug_state(frame(head0, torso, torso_to_wrist), torso)
    op1 = operator_debug_state(frame(head1, torso, torso_to_wrist), torso)
    assert np.allclose(op0["hands"]["left"]["wrist_body"], torso_to_wrist, atol=1e-9)
    assert np.allclose(op1["hands"]["left"]["wrist_body"], torso_to_wrist, atol=1e-9)
    lifted = operator_debug_state(frame(head1, torso, torso_to_wrist + [0.0, 0.1, 0.0]), torso)
    assert lifted["hands"]["left"]["wrist_body"][1] > op1["hands"]["left"]["wrist_body"][1] + 0.09


def test_operator_debug_state_no_frame_is_fixed_shape_untracked():
    from bimanual_teleop.config import SIDES
    from bimanual_teleop.render_sink import operator_debug_state

    op = operator_debug_state(None, [0.0, -0.35, 0.0])
    assert op["torso_from_head"] == [0.0, -0.35, 0.0]
    assert op["head_pos"] is None
    assert op["torso_pos"] is None
    assert set(op["hands"]) == set(SIDES)
    for side in SIDES:
        assert op["hands"][side] == {"tracked": False, "wrist_body": None, "raw_wrist": None, "lm_body": None}


def test_operator_debug_state_rejects_non_finite_pose_matrices():
    """Unity overlay state should fail closed per hand/head without leaking NaN
    into the strict JSON render stream."""
    import json
    from bimanual_teleop.config import SIDES
    from bimanual_teleop.hands.quest_retarget import synthetic_webxr_hand
    from bimanual_teleop.render_sink import operator_debug_state
    from bimanual_teleop.vr.frames import HandSample, VRFrame

    head = np.eye(4)
    good_wrist = np.eye(4)
    good_wrist[:3, 3] = [0.2, 1.3, -0.4]
    bad_wrist = good_wrist.copy()
    bad_wrist[0, 3] = float("nan")
    frame = VRFrame(
        stamp=0.0,
        head=head,
        hands={
            "left": HandSample(tracked=True, wrist=bad_wrist, landmarks=synthetic_webxr_hand(0.0)),
            "right": HandSample(tracked=True, wrist=good_wrist, landmarks=synthetic_webxr_hand(0.0)),
        },
    )

    op = operator_debug_state(frame, [0.0, -0.35, 0.0])
    assert op["hands"]["left"] == {"tracked": False, "wrist_body": None, "raw_wrist": None, "lm_body": None}
    assert op["hands"]["right"]["tracked"] is True
    assert len(op["hands"]["right"]["wrist_body"]) == 3
    json.dumps(op, allow_nan=False)

    bad_head = head.copy()
    bad_head[1, 3] = float("inf")
    op = operator_debug_state(VRFrame(stamp=0.0, head=bad_head, hands=frame.hands), [0.0, -0.35, 0.0])
    assert op["head_pos"] is None
    assert op["torso_pos"] is None
    assert set(op["hands"]) == set(SIDES)
    assert op["hands"]["left"] == {"tracked": False, "wrist_body": None, "raw_wrist": None, "lm_body": None}
    assert op["hands"]["right"]["tracked"] is False
    assert op["hands"]["right"]["wrist_body"] is None
    assert op["hands"]["right"]["raw_wrist"] == [0.2, 1.3, -0.4]
    json.dumps(op, allow_nan=False)


def test_operator_debug_state_uses_finite_default_for_bad_torso_config():
    """A bad rig torso offset should not leak non-finite values into Unity JSON."""
    import json
    from bimanual_teleop.hands.quest_retarget import synthetic_webxr_hand
    from bimanual_teleop.render_sink import operator_debug_state
    from bimanual_teleop.vr.frames import HandSample, VRFrame

    head = np.eye(4)
    head[:3, 3] = [0.0, 1.6, 0.0]
    wrist = np.eye(4)
    wrist[:3, 3] = [0.2, 1.2, -0.4]
    frame = VRFrame(
        stamp=0.0,
        head=head,
        hands={"right": HandSample(tracked=True, wrist=wrist, landmarks=synthetic_webxr_hand(0.0))},
    )

    op = operator_debug_state(frame, [float("nan"), -0.35, 0.0])
    assert op["torso_from_head"] == [0.0, -0.35, 0.0]
    assert op["hands"]["right"]["tracked"] is True
    assert len(op["hands"]["right"]["wrist_body"]) == 3
    json.dumps(op, allow_nan=False)


def test_render_state_marks_body_relative_hand_untracked_without_head_pose():
    """Unity status/overlay must not report valid torso-to-wrist motion when the
    headset pose is missing, even if the hand sensor still reports tracked fingers."""
    import uuid
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.engine import TeleopEngine
    from bimanual_teleop.hands.joint_map import ORCA_JOINT_ORDER
    from bimanual_teleop.hands.quest_retarget import synthetic_webxr_hand
    from bimanual_teleop.render_sink import RenderSink, operator_debug_state
    from bimanual_teleop.vr.frames import HandSample, VRFrame

    rig = load_rig()
    rig["vr"]["calib_seconds"] = 0
    rig["vr"]["body_relative"] = True
    rig["vr"]["unity_json_endpoint"] = None
    endpoint = f"inproc://missing-head-{uuid.uuid4()}"
    sink = RenderSink(rig, endpoint=endpoint)
    engine = TeleopEngine(rig, sink)

    wrist = np.eye(4)
    wrist[:3, 3] = [2.0, 1.8, -1.0]
    frame = VRFrame(
        stamp=0.0,
        head=None,
        hands={"left": HandSample(tracked=True, wrist=wrist, landmarks=synthetic_webxr_hand(0.5))},
    )

    op = operator_debug_state(frame, rig["vr"]["torso_from_head"])
    assert op["head_pos"] is None
    assert op["torso_pos"] is None
    assert op["hands"]["left"]["tracked"] is False
    assert op["hands"]["left"]["wrist_body"] is None
    assert op["hands"]["left"]["raw_wrist"] == [2.0, 1.8, -1.0]

    engine.tick(frame, {"left": True, "right": True}, 0.0)
    state = sink.build_state(engine, frame, {"left": True, "right": True}, 60.0, 0.0)
    assert state["status"]["tracked"]["left"] is False
    assert state["arms"]["left"]["cmd_pos"] is None
    assert state["arms"]["left"]["cmd_quat"] is None
    assert state["op"]["hands"]["left"]["wrist_body"] is None
    assert state["hand_render"]["left"]["names"] == ORCA_JOINT_ORDER
    assert len(state["hand_render"]["left"]["q"]) == len(ORCA_JOINT_ORDER)
    sink.close()


def test_render_state_marks_body_relative_hand_untracked_with_non_finite_wrist():
    """Unity status must match the body-relative vector gate, not just the raw
    hand-tracking flag. A tracked hand with a bad wrist pose cannot drive arm
    motion or an operator wrist overlay."""
    import uuid
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.engine import TeleopEngine
    from bimanual_teleop.hands.quest_retarget import synthetic_webxr_hand
    from bimanual_teleop.render_sink import RenderSink
    from bimanual_teleop.vr.frames import HandSample, VRFrame

    rig = load_rig()
    rig["vr"]["calib_seconds"] = 0
    rig["vr"]["body_relative"] = True
    rig["vr"]["unity_json_endpoint"] = None
    endpoint = f"inproc://bad-wrist-{uuid.uuid4()}"
    sink = RenderSink(rig, endpoint=endpoint)
    engine = TeleopEngine(rig, sink)

    head = np.eye(4)
    head[:3, 3] = [0.0, 1.6, 0.0]
    wrist = np.eye(4)
    wrist[:3, 3] = [0.2, 1.2, -0.4]
    wrist[0, 3] = float("nan")
    frame = VRFrame(
        stamp=0.0,
        head=head,
        hands={"left": HandSample(tracked=True, wrist=wrist, landmarks=synthetic_webxr_hand(0.5))},
    )

    engine.tick(frame, {"left": True, "right": True}, 0.0)
    state = sink.build_state(engine, frame, {"left": True, "right": True}, 60.0, 0.0)
    assert state["status"]["tracked"]["left"] is False
    assert state["arms"]["left"]["cmd_pos"] is None
    assert state["arms"]["left"]["cmd_quat"] is None
    assert state["op"]["hands"]["left"] == {"tracked": False, "wrist_body": None, "raw_wrist": None, "lm_body": None}
    sink.close()


def test_render_state_non_body_relative_status_does_not_fabricate_operator_vector():
    """If body-relative mode is disabled, a hand can still be reported as tracked
    for legacy arm-control semantics, but Unity's torso-vector overlay must not
    invent wrist_body without a headset pose."""
    import uuid
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.engine import TeleopEngine
    from bimanual_teleop.hands.quest_retarget import synthetic_webxr_hand
    from bimanual_teleop.render_sink import RenderSink
    from bimanual_teleop.vr.frames import HandSample, VRFrame

    rig = load_rig()
    rig["vr"]["calib_seconds"] = 0
    rig["vr"]["body_relative"] = False
    rig["vr"]["unity_json_endpoint"] = None
    endpoint = f"inproc://non-body-relative-missing-head-{uuid.uuid4()}"
    sink = RenderSink(rig, endpoint=endpoint)
    engine = TeleopEngine(rig, sink)

    wrist = np.eye(4)
    wrist[:3, 3] = [1.2, 1.1, -0.4]
    frame = VRFrame(
        stamp=0.0,
        head=None,
        hands={"left": HandSample(tracked=True, wrist=wrist, landmarks=synthetic_webxr_hand(0.2))},
    )

    engine.tick(frame, {"left": True, "right": True}, 0.0)
    state = sink.build_state(engine, frame, {"left": True, "right": True}, 60.0, 0.0)
    assert state["status"]["tracked"]["left"] is True
    assert state["op"]["head_pos"] is None
    assert state["op"]["torso_pos"] is None
    assert state["op"]["hands"]["left"]["tracked"] is False
    assert state["op"]["hands"]["left"]["wrist_body"] is None
    assert state["op"]["hands"]["left"]["raw_wrist"] == [1.2, 1.1, -0.4]
    sink.close()


def test_calibration_aligns_forward_and_no_cross():
    """Reference-stance calibration → 'hand forward' maps to robot forward (−X),
    and the anti-cross clamp keeps each hand on its own side of center."""
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.arms.arm_control import ArmController
    from bimanual_teleop.vr.calibrate import calibrate_R
    from bimanual_teleop.vr.frames import HandSample, quat_to_R
    lm = np.zeros((25, 3))                              # fingers forward (−z), palm down
    lm[6] = [0.03, 0, -0.03]; lm[21] = [-0.03, 0, -0.03]
    lm[9] = [0.03, 0, -0.15]; lm[14] = [0, 0, -0.16]; lm[19] = [-0.01, 0, -0.15]
    wm = lambda p: np.block([[np.eye(3), np.array(p).reshape(3, 1)], [0, 0, 0, 1]])
    rig = load_rig()
    rig["vr"]["body_relative"] = False      # this is the LEGACY raw-room mapping under test
    # This contract test drives STEP inputs (0.25-0.5 m wrist jumps per tick) on
    # purpose; disable the teleport guardrail so the alignment/no-cross contract
    # stays the thing under test (the guardrail has its own tests).
    rig["safety"]["target_jump_speed"] = 1e9
    for side, sign in (("left", -1), ("right", +1)):
        ac = ArmController(rig, side)
        ac.mapper.set_R(calibrate_R(lm, rig["arms"][side]["base_quat"]))
        bR = quat_to_R(rig["arms"][side]["base_quat"]); bp = np.array(rig["arms"][side]["base_pos"])
        wp = lambda: bR @ ac.ik.fk_wrist().translation() + bp     # wrist position in world (the IK target)
        ac.ik.reset(); t = 0.0
        ac.update(HandSample(tracked=True, wrist=wm([0, 0, 0]), landmarks=lm), True, t)
        x0 = wp()[0]
        for _ in range(120):            # from the arms-down home the arm must swing
            t += 1 / 120                # toward forward first, so it needs more ticks
            ac.update(HandSample(tracked=True, wrist=wm([0, 0, -0.25]), landmarks=lm), True, t)
        assert wp()[0] < x0 - 0.02      # forward = −X
    # shove the hands ACROSS each other; the engine's pair-order guard must keep
    # the right wrist ≥ 2*cross_gap to +Y of the left (the pair may sit anywhere
    # laterally — off-center claps are legal — but the hands can never cross)
    from bimanual_teleop.engine import TeleopEngine
    from bimanual_teleop.vr.frames import VRFrame

    class _Sink:
        def set_arm(self, *a):
            pass

        def set_hand(self, *a):
            pass

    eng = TeleopEngine(rig, _Sink())
    for s, sign in (("left", -1), ("right", +1)):
        eng.arm[s].mapper.set_R(calibrate_R(lm, rig["arms"][s]["base_quat"]))
    t = 0.0
    for _ in range(200):
        t += 1 / 120
        hands = {s: HandSample(tracked=True, wrist=wm([-sg * 0.5, 0, 0]), landmarks=lm)
                 for s, sg in (("left", -1), ("right", +1))}   # crossed demands
        eng.tick(VRFrame(stamp=t, head=None, hands=hands), {"left": True, "right": True}, t)
    gap_y = eng.arm["right"].wrist_world()[1] - eng.arm["left"].wrist_world()[1]
    min_gap = 2 * float(rig["vr"]["cross_gap"])
    assert gap_y >= min_gap - 0.02, f"hands crossed: pair Y gap {gap_y:.3f} < {min_gap}"


def test_end_to_end_render_tick():
    """Fake VR → engine → RenderSink: the engine moves the arms (EE FK changes) and
    publishes a well-formed render.state that ZMQ and Unity TCP subscribers receive."""
    import json
    import socket
    from bimanual_teleop.config import load_rig, SIDES
    from bimanual_teleop.engine import TeleopEngine
    from bimanual_teleop.render_sink import RenderSink
    from bimanual_teleop.bus.zmq_io import LatestSub
    from bimanual_teleop.bus import topics
    from bimanual_teleop.vr.ingest import FakeVRSource
    rig = load_rig()
    rig["vr"]["calib_seconds"] = 0          # skip the calibration phase for this motion test
    render_port = _free_tcp_port()
    json_port = _free_tcp_port()
    rig["vr"]["render_endpoint"] = f"tcp://127.0.0.1:{render_port}"
    rig["vr"]["unity_json_endpoint"] = f"tcp://127.0.0.1:{json_port}"
    sink = RenderSink(rig)
    engine = TeleopEngine(rig, sink)
    sub = LatestSub(f"tcp://127.0.0.1:{render_port}", topics.RENDER_STATE)
    tcp = socket.create_connection(("127.0.0.1", json_port), timeout=1.0)
    tcp.settimeout(1.0)
    src = FakeVRSource()
    time.sleep(0.2)  # PUB/SUB slow-joiner
    ee0 = engine.arm["left"].ik.fk_ee().translation().copy()
    for i in range(240):                     # past the absolute-mode engage glide
        t = i / 60.0
        frame = src.frame_at(t)
        eng = {s: True for s in SIDES}
        engine.tick(frame, eng, t)
        sink.publish(engine, frame, eng, 60.0, t)
    sub.poll()
    st = sub.get(topics.RENDER_STATE)
    assert st is not None and len(st["arms"]["left"]["q"]) == 6
    assert len(st["arms"]["left"]["link_pos"]) == 24
    assert len(st["arms"]["left"]["cmd_pos"]) == 3
    assert np.all(np.isfinite(st["arms"]["left"]["cmd_pos"]))
    # absolute mode parks cmd at the operator's (clamped) target even when the
    # fake source holds its hands beyond the YAM's reach ceiling, so the
    # cmd-vs-achieved gap is bounded by workspace geometry, not ~0.
    assert np.linalg.norm(np.asarray(st["arms"]["left"]["cmd_pos"]) - np.asarray(st["arms"]["left"]["ee_pos"])) < 0.7
    assert len(st["hand_render"]["left"]["names"]) == 17
    assert len(st["hand_render"]["left"]["q"]) == 17
    assert st["status"]["engaged"]["left"] is True
    raw = tcp.makefile("r").readline()
    js = json.loads(raw)
    assert js["v"] == topics.SCHEMA_VERSION
    assert len(js["arms"]["right"]["q"]) == 6
    assert len(js["arms"]["right"]["link_pos"]) == 24
    assert len(js["arms"]["right"]["cmd_pos"]) == 3
    assert np.all(np.isfinite(js["arms"]["right"]["cmd_pos"]))
    assert len(js["hand_render"]["right"]["names"]) == 17
    assert len(js["hand_render"]["right"]["q"]) == 17
    assert js["status"]["tracked"]["right"] is True
    assert js["op"]["torso_from_head"] == rig["vr"]["torso_from_head"]
    assert js["op"]["hands"]["left"]["tracked"] is True
    assert len(js["op"]["hands"]["left"]["wrist_body"]) == 3
    assert np.linalg.norm(engine.arm["left"].ik.fk_ee().translation() - ee0) > 0.02
    tcp.close(); sink.close(); sub.close()


def test_unity_render_fixture_matches_python_publisher():
    """The Unity Editor validation fixture must be generated from RenderSink's
    current render-state builder, not a hand-maintained JSON sample."""
    import importlib.util
    import json
    from pathlib import Path
    from bimanual_teleop.bus import topics
    from bimanual_teleop.config import SIDES
    from bimanual_teleop.hands.joint_map import ORCA_JOINT_ORDER

    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "update_unity_fixture.py"
    spec = importlib.util.spec_from_file_location("update_unity_fixture", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fixture_path = repo / "unity" / "TeleopRenderer" / "Assets" / "Editor" / "render_state_sample.json"
    actual = fixture_path.read_text(encoding="utf-8")
    generated = mod.normalized_text(mod.make_fixture())
    assert actual == generated

    sample = json.loads(actual)
    assert sample["v"] == topics.SCHEMA_VERSION
    assert set(sample["arms"]) == set(SIDES)
    assert set(sample["hand_render"]) == set(SIDES)
    for side in SIDES:
        assert len(sample["arms"][side]["link_pos"]) == 24
        assert len(sample["arms"][side]["q"]) == 6
        assert sample["hand_render"][side]["names"] == ORCA_JOINT_ORDER
        assert len(sample["op"]["hands"][side]["wrist_body"]) == 3


def test_render_sink_survives_unity_json_port_busy():
    """Unity JSON is useful but non-critical: if that port is already held by an
    editor/debug process, teleop must keep running and still publish ZMQ state."""
    import socket
    import time
    from bimanual_teleop.config import load_rig, SIDES
    from bimanual_teleop.engine import TeleopEngine
    from bimanual_teleop.render_sink import RenderSink
    from bimanual_teleop.bus.zmq_io import LatestSub
    from bimanual_teleop.bus import topics
    from bimanual_teleop.vr.ingest import FakeVRSource

    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    busy.bind(("127.0.0.1", 0))
    busy_port = int(busy.getsockname()[1])
    busy.listen()

    rig = load_rig()
    rig["vr"]["calib_seconds"] = 0
    render_port = _free_tcp_port()
    rig["vr"]["render_endpoint"] = f"tcp://127.0.0.1:{render_port}"
    rig["vr"]["unity_json_endpoint"] = f"tcp://127.0.0.1:{busy_port}"
    sink = RenderSink(rig)
    assert sink.json_enabled is False
    engine = TeleopEngine(rig, sink)
    sub = LatestSub(f"tcp://127.0.0.1:{render_port}", topics.RENDER_STATE)
    src = FakeVRSource()
    time.sleep(0.2)
    eng = {s: True for s in SIDES}
    for i in range(10):
        t = i / 60.0
        frame = src.frame_at(t)
        engine.tick(frame, eng, t)
        sink.publish(engine, frame, eng, 60.0, t)
        time.sleep(0.005)
    sub.poll()
    assert sub.get(topics.RENDER_STATE) is not None
    sink.close(); sub.close(); busy.close()


def test_render_sink_survives_zmq_port_busy_with_unity_json_fallback():
    """If the ZMQ render port is held by a stale process, Unity JSON should still
    start and receive render states."""
    import json
    import socket
    import time
    from bimanual_teleop.config import load_rig, SIDES
    from bimanual_teleop.engine import TeleopEngine
    from bimanual_teleop.render_sink import RenderSink
    from bimanual_teleop.vr.ingest import FakeVRSource

    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    busy.bind(("127.0.0.1", 0))
    busy_port = int(busy.getsockname()[1])
    busy.listen()

    rig = load_rig()
    rig["vr"]["calib_seconds"] = 0
    json_port = _free_tcp_port()
    rig["vr"]["render_endpoint"] = f"tcp://127.0.0.1:{busy_port}"
    rig["vr"]["unity_json_endpoint"] = f"tcp://127.0.0.1:{json_port}"
    sink = RenderSink(rig)
    assert sink.zmq_enabled is False
    assert sink.json_enabled is True
    engine = TeleopEngine(rig, sink)
    tcp = socket.create_connection(("127.0.0.1", json_port), timeout=1.0)
    tcp.settimeout(1.0)
    src = FakeVRSource()
    time.sleep(0.05)
    frame = src.frame_at(0.0)
    eng = {s: True for s in SIDES}
    engine.tick(frame, eng, 0.0)
    sink.publish(engine, frame, eng, 60.0, 0.0)
    js = json.loads(tcp.makefile("r").readline())
    assert js["status"]["tracked"]["left"] is True
    assert len(js["arms"]["left"]["q"]) == 6
    tcp.close(); sink.close(); busy.close()


def test_unity_json_broadcaster_drops_non_finite_frames():
    """Unity receives strict JSON only; NaN/Infinity render frames are dropped
    instead of being serialized as non-standard JavaScript tokens."""
    import json
    import socket
    import time
    from bimanual_teleop.render_sink import TcpJsonBroadcaster

    port = _free_tcp_port()
    broadcaster = TcpJsonBroadcaster(f"tcp://127.0.0.1:{port}")
    tcp = socket.create_connection(("127.0.0.1", port), timeout=1.0)
    tcp.settimeout(0.2)
    reader = tcp.makefile("r")
    time.sleep(0.05)

    try:
        broadcaster.send({"v": 1, "value": 1.0})
        assert json.loads(reader.readline()) == {"v": 1, "value": 1.0}

        broadcaster.send({"v": 1, "value": float("nan")})
        try:
            raw = reader.readline()
        except socket.timeout:
            raw = ""
        assert raw == ""
    finally:
        tcp.close()
        broadcaster.close()


def test_orbit_source_unity_to_webxr():
    """ORBIT (Unity, +z fwd, LH) -> engine (WebXR, -z fwd, RH) is the single Z-flip
    congruence: Palm[1] dropped (26->25 W3C joints), wrist pose converted, staleness honored."""
    import time
    from bimanual_teleop.vr.orbit_source import OrbitVRSource
    src = OrbitVRSource({"vr": {"orbit_flip": "z", "orbit_adb_reverse": False, "orbit_timeout": 5.0}})
    pts = np.zeros((26, 3)); pts[:, 2] = np.arange(26) * 0.01; pts[1] = [9.0, 9.0, 9.0]  # Palm sentinel
    hand_msg = "relative:" + "|".join(f"{x},{y},{z}" for x, y, z in pts) + ":"
    now = time.monotonic()
    src._ingest("hand", "right", hand_msg, now)
    src._ingest("wrist", "right", "relative,0.1,0.2,0.3,0,0,0,1", now)
    f = src.latest()
    assert f.head is None                                             # no placeholder identity head
    hr = f.hands["right"]
    assert hr.tracked and hr.landmarks.shape == (25, 3)               # tracked, palm dropped
    assert not np.allclose(hr.landmarks[1], [9.0, 9.0, -9.0])         # the Palm sentinel is gone
    assert np.isclose(hr.landmarks[1, 2], -0.02)                     # orig joint 2, z negated
    assert np.allclose(hr.wrist[:3, 3], [0.1, 0.2, -0.3], atol=1e-6)  # wrist z flipped
    src._wrist_last["right"] = now - 10.0                            # staleness -> not tracked
    assert not src.latest().hands["right"].tracked


def test_orbit_source_tracks_wrist_and_landmark_freshness_separately():
    """Fresh hand-landmark packets must not keep stale wrist poses tracked, and
    stale landmarks must not invalidate a fresh wrist pose used for arm control."""
    import time
    from bimanual_teleop.vr.orbit_source import OrbitVRSource

    src = OrbitVRSource({"vr": {"orbit_flip": "z", "orbit_adb_reverse": False, "orbit_timeout": 5.0}})
    pts = np.zeros((26, 3))
    pts[:, 0] = np.arange(26) * 0.01
    hand_msg = "relative:" + "|".join(f"{x},{y},{z}" for x, y, z in pts) + ":"
    now = time.monotonic()
    src._ingest("hand", "right", hand_msg, now)
    src._ingest("wrist", "right", "relative,0.1,0.2,0.3,0,0,0,1", now)

    src._wrist_last["right"] = now - 10.0
    src._lm_last["right"] = now
    assert src.latest().hands["right"].tracked is False

    src._ingest("wrist", "right", "relative,0.1,0.2,0.3,0,0,0,1", now)
    src._lm_last["right"] = now - 10.0
    hand = src.latest().hands["right"]
    assert hand.tracked is True
    assert hand.landmarks is None
    assert hand.pinch == 0.0


def test_orbit_source_rejects_malformed_or_non_finite_messages():
    """ORBIT parser failures must leave hands/head untracked rather than
    accepting malformed points or non-finite pose values."""
    import time
    from bimanual_teleop.vr.orbit_source import OrbitVRSource, _parse_hand, _parse_pose

    assert _parse_hand("relative:1,2|3,4:") is None
    assert _parse_hand("relative:1,2,nan|3,4,5:") is None
    assert _parse_pose("relative,0,0,0,0,0,0,0") is None
    assert _parse_pose("relative,0,0,nan,0,0,0,1") is None

    src = OrbitVRSource({"vr": {"orbit_flip": "z", "orbit_adb_reverse": False, "orbit_timeout": 5.0}})
    now = time.monotonic()
    src._ingest("hand", "right", "relative:1,2,nan|3,4,5:", now)
    src._ingest("wrist", "right", "relative,0.1,0.2,nan,0,0,0,1", now)
    f = src.latest()
    assert f.head is None
    assert f.hands["right"].tracked is False


def test_vuer_source_starts_without_placeholder_head_pose():
    """Before WebXR CAMERA_MOVE arrives, Vuer must expose head=None instead of an
    identity matrix, so body-relative arm control cannot fabricate torso vectors."""
    from bimanual_teleop.config import SIDES
    from bimanual_teleop.vr.vuer_source import VuerVRSource

    src = VuerVRSource({"vr": {}})
    f = src.latest()
    assert f is not None
    assert f.head is None
    assert set(f.hands) == set(SIDES)


def test_vuer_source_rejects_malformed_pose_matrices():
    """Malformed Vuer hand/camera matrices must fail closed instead of becoming
    identity poses that body-relative control would treat as real tracking."""
    from bimanual_teleop.vr.vuer_source import VuerVRSource, _mat4

    assert _mat4([1.0, 2.0]) is None
    assert _mat4([float("nan")] * 16) is None

    src = VuerVRSource({"vr": {}})
    bad_hand = [0.0] * (25 * 16)
    bad_hand[0] = float("nan")
    src._update_hand("left", bad_hand, {})
    f = src.latest()
    assert f.head is None
    assert f.hands["left"].tracked is False

    good_hand = np.zeros((25, 16), dtype=float)
    for i in range(25):
        M = np.eye(4)
        M[:3, 3] = [0.01 * i, 0.02 * i, -0.03 * i]
        good_hand[i] = M.reshape(16, order="F")
    src._update_hand("left", good_hand.reshape(-1).tolist(), {"pinchValue": 0.4})
    f = src.latest()
    assert f.hands["left"].tracked is True
    assert np.allclose(f.hands["left"].wrist, np.eye(4))
    assert f.hands["left"].pinch == 0.4


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fail = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:
            fail += 1
            print("FAIL", fn.__name__, "->", type(e).__name__, e)
    sys.exit(1 if fail else 0)


def test_absolute_orientation_wears_hand_attitude_with_glide():
    """orientation_mode=absolute: continuous at engage, and after the glide the
    commanded EE attitude equals the operator's hand attitude mapped through the
    body↔world axes and the derived hand↔EE convention — the overlay-overlap
    guarantee — for both sides."""
    from bimanual_teleop.arms.arm_control import ArmController
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.vr.calibrate import W_AXES, body_relative_hand_sample, head_op_axes
    from bimanual_teleop.vr.frames import HandSample, euler_to_R, quat_to_R, rotvec

    rig = load_rig()
    assert rig["mapping"]["orientation_mode"] == "absolute"
    head = np.eye(4)
    head[:3, 3] = [0.0, 1.6, 0.0]
    for side, lat in (("left", -0.22), ("right", 0.22)):
        ac = ArmController(rig, side)
        W = np.eye(4)
        op = head_op_axes(head)
        W[:3, :3] = euler_to_R([0.4, -0.2, 0.3])
        torso_w = head[:3, 3] + op @ np.asarray(rig["vr"]["torso_from_head"])
        W[:3, 3] = torso_w + op @ np.array([lat, 0.0, 0.42])
        hs = body_relative_hand_sample(HandSample(tracked=True, wrist=W), head,
                                       rig["vr"]["torso_from_head"])
        # engage instant: command equals the current EE pose (no snap)
        ac.update(hs, True, 0.0)
        ee0 = ac.ik.fk_ee().rotation().as_matrix()
        assert np.degrees(np.linalg.norm(rotvec(ac.cmd_R.T @ ee0))) < 5.0
        # after the glide: command equals skeleton-attitude ∘ convention exactly
        t = 0.0
        for _ in range(int(4.0 * 120)):
            t += 1 / 120
            ac.update(hs, True, t)
        M_w = W_AXES @ op.T @ W[:3, :3]
        pred = quat_to_R(rig["arms"][side]["base_quat"]).T @ M_w @ ac.mapper.C
        err = np.degrees(np.linalg.norm(rotvec(pred.T @ ac.cmd_R)))
        assert err < 0.5, (side, err)


def test_absolute_orientation_fails_closed_on_improper_ctrl_rotation():
    """A reflection (det −1) in the ctrl sample must never reach the IK as a
    target attitude: the mapper holds the anchor attitude instead."""
    from bimanual_teleop.vr.frames import SE3, SO3, ClutchMapper, euler_to_R

    C = euler_to_R([0.2, -0.3, 0.5])
    m = ClutchMapper(np.diag([1.0, 1.0, -1.0]) @ euler_to_R([0.1, 0.2, 0.3]),  # improper R (body map)
                     orientation_mode="absolute", hand_ee_convention=C)
    A = euler_to_R([0.4, 0.1, -0.2])
    ee = SE3.from_rotation_and_translation(SO3.from_matrix(A), np.array([0.3, 0.1, 0.5]))
    proper = SE3.from_rotation_and_translation(SO3.from_matrix(euler_to_R([0.5, 0.0, 0.2])),
                                               np.array([0.1, 0.2, 0.3]))
    m.engage(proper, ee, t=0.0)
    # PROPER ctrl with improper R*ctrl*C → improper target → must hold anchor
    out = m.target(proper, 10.0).rotation().as_matrix()
    assert np.allclose(out, A, atol=1e-12)
    assert np.linalg.det(out) > 0
