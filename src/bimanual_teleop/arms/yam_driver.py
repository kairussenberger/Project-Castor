"""Real i2rt YAM arm driver (Linux + SocketCAN only).

Import-guarded: importing this module does NOT require the i2rt SDK; it's only
needed when you actually construct a YamArm (i.e. on the Linux control host with
the arms wired up).

ENERGIZE POLICY (safety-critical): YamArm drives the CAN chain DIRECTLY through
arms/yam_chain.py — i2rt's MotorChainRobot wrapper is deliberately not used:

  - its boot/runtime qpos checks assume i2rt's joint zero conventions, which this
    rig's motors do NOT follow (confirmed on metal 2026-06-11: the official hang
    reads j2≈−177°, j6≈+312° in motor space → MotorChainRobot raises at boot);
  - its gravity compensation assumes a TABLE-MOUNTED YAM — wrong directions for
    this rig's SIDEWAYS arm mounts.

Energizing produces NO motion: motors enable limp (zero command), wrap offsets
normalize from a fresh read, then the MIT PD holds the MEASURED pose. Gravity is
a bounded PD disturbance (stock gains kp 80/80/80/40/10/10), deterministic.

JOINT CONVENTION (safety-critical): every command/state crosses the per-side
JointMap measured by scripts/hw_bringup.py (signs from ±2° nudges, offsets
anchored at the official rest pose). A YamArm without a map refuses model-space
commands instead of guessing; the calibrated image of the model joint limits is
also installed as a motor-space clamp at the chain.

Bring-up (Ubuntu):
    sudo ip link set can0 up type can bitrate 1000000
    git clone https://github.com/i2rt-robotics/i2rt && uv pip install -e i2rt
    uv run python scripts/hw_bringup.py          # guided: scan→rest→signs→verify
"""
from __future__ import annotations

import numpy as np

from ..logging_utils import get_logger
from .joint_map import JointMap

log = get_logger("yam")


class YamArm:
    """6-DoF YAM over CAN. Exposes the same state()/command() seam as the sim arm,
    in MODEL space (rig.yaml convention) — the JointMap crosses the boundary."""

    def __init__(self, channel: str, joint_map: JointMap | None = None, *,
                 require_map: bool = True, model_limits=None):
        try:
            from .yam_chain import YamChain
            self.chain = YamChain(channel)
        except ImportError as e:  # pragma: no cover - hardware only
            raise RuntimeError(
                "i2rt SDK not installed. Real YAM control is Linux/SocketCAN only — "
                "install on the control host: `uv pip install -e i2rt` (see module docstring)."
            ) from e
        if joint_map is None and require_map:
            self.chain.off()
            raise RuntimeError(
                f"YamArm({channel}): no motor↔model joint map. The motor firmware uses a "
                "different joint convention than the runtime model; commanding without the "
                "measured map can sweep the arm. Run: uv run python scripts/hw_bringup.py"
            )
        self.channel = channel
        self.map = joint_map
        if self.map is not None and model_limits is not None:
            lo = self.map.to_motor(np.asarray(model_limits["lower"], dtype=float))
            hi = self.map.to_motor(np.asarray(model_limits["upper"], dtype=float))
            self.chain.set_bounds(lo, hi)          # motor-space clamp = mapped model limits
        # Energize = PD hold at the measured pose: zero motion at startup.
        self.chain.hold()

    # ---- state ------------------------------------------------------------- #
    def state_motor(self) -> np.ndarray:
        """Current joint positions (6,) in MOTOR space, radians."""
        return self.chain.read_pos()[:6]

    def state(self) -> np.ndarray:
        """Current joint positions (6,) in MODEL space, radians."""
        if self.map is None:
            raise RuntimeError(f"YamArm({self.channel}): state() needs the joint map (hw_bringup)")
        return self.map.to_model(self.state_motor())

    # ---- commands ---------------------------------------------------------- #
    def command(self, q) -> None:
        """Command MODEL-space joint targets (radians); MIT-mode motor PD tracks
        them at 250 Hz. Callers MUST shape commands first (safety/shaper.py) —
        the chain clamps into the calibrated motor range but does not rate-limit."""
        if self.map is None:
            raise RuntimeError(f"YamArm({self.channel}): command() needs the joint map (hw_bringup)")
        self.chain.command_pd(self.map.to_motor(np.asarray(q, dtype=float)[:6]))

    def command_motor(self, q_motor) -> None:
        """MOTOR-space command (bring-up tooling; runtime goes through command())."""
        self.chain.command_pd(q_motor)

    # ---- torque modes ------------------------------------------------------ #
    def hold(self) -> None:
        """PD-hold the current measured pose."""
        self.chain.hold()

    def release_torque(self) -> None:
        """Zero gains/torques — the arm goes limp (it hangs; at rest this is the
        official resting pose). hold() re-engages."""
        self.chain.idle()

    def close(self) -> None:  # pragma: no cover - hardware only
        """Release torque, stop the stream, switch the motors OFF, close the bus."""
        self.chain.off()
