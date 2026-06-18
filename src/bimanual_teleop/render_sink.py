"""RenderSink — the teleop engine's sink for the Unity renderer.

Replaces the MuJoCo SimWorld: instead of stepping a local physics/viewer, it
publishes one latest-wins `render.state` message per tick over the ZMQ bus and,
optionally, the same payload as newline-delimited JSON over plain TCP. The TCP JSON
path is intentionally dependency-light for Unity: a C# `TcpClient` can connect,
read lines, and draw the robot + HUD with no NetMQ/msgpack packages.

It satisfies the same sink interface the engine drives (`set_arm`, `set_hand`), so
the engine and controllers are untouched. Forward kinematics for the achieved EE
frame comes from the per-arm pink/Pinocchio IK that already ran this tick (read off
engine.arm[side].ik) — no MuJoCo, no duplicate model.

The HardwareSink (real YAM CAN + ORCA serial) is the OTHER sink; both share this
interface. RenderSink can run ALONGSIDE the hardware sink (tee) so the operator sees
in VR exactly what the robot is doing.
"""
from __future__ import annotations

import json
import socket
import threading

import numpy as np

from .bus import topics
from .bus.zmq_io import Publisher
from .config import SIDES
from .hands.joint_map import ORCA_JOINT_ORDER
from .logging_utils import get_logger
from .vr.calibrate import body_relative_hand_sample, head_op_axes
from .vr.frames import R_to_quat

log = get_logger("render")
_DEFAULT_TORSO_FROM_HEAD = np.array([0.0, -0.35, 0.0])


def _finite_mat4(value) -> np.ndarray | None:
    try:
        mat = np.asarray(value, dtype=float).reshape(4, 4)
    except (TypeError, ValueError):
        return None
    return mat if np.all(np.isfinite(mat)) else None


def _finite_vec3(value, default: np.ndarray | None = None) -> np.ndarray | None:
    try:
        vec = np.asarray(value, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return default.copy() if default is not None else None
    if not np.all(np.isfinite(vec)):
        return default.copy() if default is not None else None
    return vec


def ordered_hand_state(joints: dict[str, float]) -> dict:
    """Fixed-shape ORCA hand state for Unity's field-based JsonUtility parser."""
    return {
        "names": list(ORCA_JOINT_ORDER),
        "q": [float(joints.get(name, 0.0)) for name in ORCA_JOINT_ORDER],
    }


def _landmarks_body(hs, op_axes) -> list | None:
    """25 hand landmarks relative to the wrist joint, rotated into operator BODY
    axes; flattened 75 floats for the dashboard's hand-skeleton overlay (additive
    schema field — Unity's JsonUtility ignores it)."""
    lm = getattr(hs, "landmarks", None)
    if lm is None:
        return None
    try:
        arr = np.asarray(lm, dtype=float).reshape(25, 3)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return ((arr - arr[0]) @ op_axes).reshape(-1).tolist()


def operator_debug_state(frame, torso_from_head, head_R_override=None) -> dict:
    """Unity/debug overlay state: the body-frame torso→wrist vectors that drive
    IK. `head_R_override` = the engine's locked yaw frame, so the display uses
    exactly the frame arm control uses (head turns don't rotate the overlay)."""
    torso = _finite_vec3(torso_from_head, _DEFAULT_TORSO_FROM_HEAD)
    if frame is None:
        return {
            "torso_from_head": torso.tolist(),
            "head_pos": None,
            "torso_pos": None,
            "hands": {
                s: {"tracked": False, "wrist_body": None, "raw_wrist": None, "lm_body": None}
                for s in SIDES
            },
        }
    head = _finite_mat4(getattr(frame, "head", None))
    if head is not None and head_R_override is not None:
        head = head.copy()
        head[:3, :3] = np.asarray(head_R_override, dtype=float)
    if head is None:
        hands = {}
        for s in SIDES:
            hs = frame.hands.get(s) if frame and frame.hands else None
            wrist = _finite_mat4(hs.wrist) if hs is not None else None
            raw = wrist[:3, 3].tolist() if wrist is not None else None
            hands[s] = {"tracked": False, "wrist_body": None, "raw_wrist": raw, "lm_body": None}
        return {"torso_from_head": torso.tolist(), "head_pos": None, "torso_pos": None, "hands": hands}
    op_axes = head_op_axes(head)
    torso_world = head[:3, 3] + op_axes @ torso
    hands = {}
    for s in SIDES:
        hs = frame.hands.get(s) if frame and frame.hands else None
        if hs is None or not hs.tracked:
            hands[s] = {"tracked": False, "wrist_body": None, "raw_wrist": None, "lm_body": None}
            continue
        wrist = _finite_mat4(hs.wrist)
        raw = wrist[:3, 3].tolist() if wrist is not None else None
        rel = body_relative_hand_sample(hs, head, torso)
        if rel is None or not rel.tracked or _finite_mat4(rel.wrist) is None:
            hands[s] = {"tracked": False, "wrist_body": None, "raw_wrist": raw, "lm_body": None}
            continue
        hands[s] = {
            "tracked": True,
            "wrist_body": rel.wrist[:3, 3].tolist(),
            "raw_wrist": raw,
            "lm_body": _landmarks_body(hs, op_axes),
        }
    return {
        "torso_from_head": torso.tolist(),
        "head_pos": head[:3, 3].tolist(),
        "torso_pos": torso_world.tolist(),
        "hands": hands,
    }


class TcpJsonBroadcaster:
    """Small non-blocking TCP fanout for Unity clients.

    It is deliberately best-effort: render frames are latest-value state, so a slow
    Unity client is dropped instead of stalling the control loop.
    """

    def __init__(self, endpoint: str):
        host, port_s = endpoint.removeprefix("tcp://").rsplit(":", 1)
        self.endpoint = endpoint
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, int(port_s)))
        self._sock.listen()
        self._sock.settimeout(0.2)
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        while not self._closed:
            try:
                client, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client.settimeout(0.001)
            with self._lock:
                self._clients.append(client)

    def send(self, obj: dict) -> None:
        try:
            line = (json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        except ValueError as e:
            log.warning("dropping non-finite Unity JSON render frame: %s", e)
            return
        dead: list[socket.socket] = []
        with self._lock:
            for client in self._clients:
                try:
                    client.sendall(line)
                except (BlockingIOError, TimeoutError, OSError):
                    dead.append(client)
            for client in dead:
                self._clients.remove(client)
                try:
                    client.close()
                except OSError:
                    pass

    def close(self) -> None:
        self._closed = True
        try:
            self._sock.close()
        except OSError:
            pass
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            try:
                client.close()
            except OSError:
                pass


class RenderSink:
    def __init__(self, rig: dict, endpoint: str | None = None):
        self.rig = rig
        ep = endpoint or rig.get("vr", {}).get("render_endpoint", topics.RENDER_ENDPOINT)
        self.endpoint = ep
        self.pub: Publisher | None = None
        try:
            self.pub = Publisher(ep)
        except Exception as e:
            # Render output is latest-state diagnostics. If an old process is still
            # holding the ZMQ port, keep teleop alive and rely on the JSON bridge if
            # available.
            log.warning("ZMQ render publisher disabled (%s): %s", ep, e)
        json_ep = rig.get("vr", {}).get("unity_json_endpoint")
        self.json_endpoint = json_ep
        self.json: TcpJsonBroadcaster | None = None
        if json_ep:
            try:
                self.json = TcpJsonBroadcaster(json_ep)
            except (OSError, ValueError) as e:
                # Rendering is diagnostic/latest-state only; losing the Unity TCP
                # helper must not prevent teleop/IK/hardware from starting.
                log.warning("Unity JSON render bridge disabled (%s): %s", json_ep, e)
        self._arm: dict[str, np.ndarray] = {s: np.asarray(rig["arms"][s]["neutral_q"], float) for s in SIDES}
        self._hand: dict[str, dict[str, float]] = {s: {} for s in SIDES}
        self._torso_from_head = _finite_vec3(
            rig.get("vr", {}).get("torso_from_head", _DEFAULT_TORSO_FROM_HEAD),
            _DEFAULT_TORSO_FROM_HEAD,
        )
        self._body_relative = bool(rig.get("vr", {}).get("body_relative", True))

    @property
    def json_enabled(self) -> bool:
        return self.json is not None

    @property
    def zmq_enabled(self) -> bool:
        return self.pub is not None

    # ---- sink interface (driven by TeleopEngine) -------------------------- #
    def set_arm(self, side: str, q) -> None:
        self._arm[side] = np.asarray(q, dtype=float)

    def set_hand(self, side: str, joints_deg: dict) -> None:
        self._hand[side] = dict(joints_deg)

    # ---- build / publish one render frame --------------------------------- #
    def build_state(self, engine, frame, engaged: dict, hz: float, t: float) -> dict:
        """Build + send the render.state message. Reads achieved/commanded EE frames
        off the engine's per-arm controllers (FK already computed this tick)."""
        arms = {}
        for s in SIDES:
            ac = engine.arm[s]
            ee = ac.ik.fk_ee()
            ee_p = ac.base_R @ ee.translation() + ac.base_pos          # → world
            ee_R = ac.base_R @ ee.rotation().as_matrix()
            link_pos = np.array([ac.base_R @ p + ac.base_pos for p in ac.ik.link_points()])
            cmd_quat = None
            cmd_pos = None
            if getattr(ac, "cmd_pos", None) is not None:
                cmd_pos = (ac.base_R @ ac.cmd_pos + ac.base_pos).tolist()
            if getattr(ac, "cmd_R", None) is not None:
                cmd_quat = R_to_quat(ac.base_R @ ac.cmd_R).tolist()
            arms[s] = {
                "q": self._arm[s].tolist(),
                "link_pos": link_pos.reshape(-1).tolist(),
                "ee_pos": ee_p.tolist(),
                "ee_quat": R_to_quat(ee_R).tolist(),       # (w,x,y,z)
                "cmd_pos": cmd_pos,
                "cmd_quat": cmd_quat,
                "margins": ac.ik.limit_margins(self._arm[s]).tolist(),
                # Pre-clamp target excess outside the workspace box (m) —
                # sustained values = the mapping/calibration is off (additive
                # field; the dashboard turns it into the WS CLAMP indicator).
                "clamp_dist": round(float(getattr(ac, "clamp_dist", 0.0)), 4),
            }
        op_state = operator_debug_state(frame, self._torso_from_head,
                                        getattr(engine, "_yaw_R", None))
        if self._body_relative:
            tracked = {s: bool(op_state["hands"][s]["tracked"]) for s in SIDES}
        else:
            tracked = {
                s: bool(frame and s in frame.hands and frame.hands[s].tracked)
                for s in SIDES
            } if frame else {s: False for s in SIDES}
        status = {
            "engaged": {s: bool(engaged.get(s, False)) for s in SIDES},
            "tracked": tracked,
            "calib": engine.calib_status,
            # Applied neutral-pose fit (additive schema field — Unity's
            # JsonUtility ignores unknown fields; the dashboard shows a chip).
            "calib_applied": getattr(engine, "calib_summary", None),
            # SAFETY state: live transports hold the arms until an in-session
            # calibration completes (additive field; Unity ignores it).
            "follow_locked": bool(getattr(engine, "follow_locked", False)),
            # Anchor-jump guard state (additive field): trips/holds/reason for
            # the dashboard banner and post-session forensics.
            "guard": (engine.guard.status()
                      if getattr(engine, "guard", None) is not None else None),
            "hz": float(hz),
        }
        hand_render = {s: ordered_hand_state(self._hand[s]) for s in SIDES}
        return topics.msg(stamp=float(t), arms=arms, hands=dict(self._hand),
                          hand_render=hand_render, op=op_state, status=status)

    def publish(self, engine, frame, engaged: dict, hz: float, t: float) -> None:
        """Build and send the latest Unity/Python render state."""
        msg = self.build_state(engine, frame, engaged, hz, t)
        if self.pub is not None:
            self.pub.send(topics.RENDER_STATE, msg)
        if self.json is not None:
            self.json.send(msg)

    def close(self) -> None:
        if self.pub is not None:
            self.pub.close()
        if self.json is not None:
            self.json.close()
