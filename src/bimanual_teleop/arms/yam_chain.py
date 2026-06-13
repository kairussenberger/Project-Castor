"""Direct CAN-chain control for one YAM arm — no i2rt MotorChainRobot.

Why not i2rt's robot wrapper: it boot-checks the measured pose against ITS xml
joint conventions and raises when they disagree — and this rig's motor zeros do
NOT follow i2rt's convention (confirmed on metal 2026-06-11: at the official
hang the motors read j2≈−177°, j6≈+312°, which MotorChainRobot rejects). It also
defaults to table-mount gravity-compensation torques that are wrong for these
SIDEWAYS-mounted arms. This class drives i2rt's DMChainCanInterface directly:

  - enable = torque-capable but ZERO command (limp); the chain thread streams at
    250 Hz from the first frame (we pre-store a zero-torque+damping command so
    i2rt's enable-torque echo never goes out),
  - multi-turn wrap normalization from a FRESH post-thread read (a p16 encoder
    can legitimately report ±2π·k; reported state and commanded positions share
    the same offset transform, so re-anchoring keeps them coherent),
  - MIT PD position mode with the stock yam.yml gains (kp 80/80/80/40/10/10) and
    NO feedforward — gravity is a bounded PD disturbance, deterministic,
  - optional MOTOR-SPACE command bounds (the calibrated image of the model
    limits — i2rt's xml limits are meaningless under this rig's zeros),
  - fail-closed: on unrecoverable motor errors the chain thread stops streaming
    → the motor-side CAN watchdog releases torque (verify per rig with
    scripts/hw_bringup.py --step watchdog).

Used by YamArm (runtime, through the measured JointMap) and scripts/hw_bringup.py
(calibration, motor space). Hardware-only construction; import is guarded by the
caller (yam_driver / hw_bringup).
"""
from __future__ import annotations

import time

import numpy as np

from ..logging_utils import get_logger
from ..safety.shaper import JointCommandShaper

log = get_logger("yam_chain")

STILL_VEL = 0.05    # rad/s: "the arm is not moving"
STILL_PTP = 0.02    # rad: positions stable across the window
TWO_PI = 2.0 * np.pi


def wrap_corrections(positions: np.ndarray) -> np.ndarray:
    """Per-joint k·2π offset corrections that bring reported angles into (−π, π].
    Pure so it is unit-testable; handles multi-turn readings (e.g. +5.44 → −0.85
    via one +2π offset, +9.0 → +2.72 via one, +9.7 → −2.86 via two)."""
    pos = np.asarray(positions, dtype=float)
    return np.round(pos / TWO_PI) * TWO_PI


class YamChain:
    """One YAM arm over SocketCAN, MOTOR space, our shaping — see module docstring."""

    def __init__(self, channel: str, *, rate_deg_s: float = 10.0):
        from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorCmd, ReceiveMode
        from i2rt.robots.utils import ArmType, _load_arm_config
        cfg = _load_arm_config(ArmType.YAM)
        self._MotorCmd = MotorCmd
        self.kp = np.asarray(cfg.kp, dtype=float)
        self.kd = np.asarray(cfg.kd, dtype=float)
        self.idle_kd = np.asarray(cfg.grav_comp_kd, dtype=float)
        self.n = len(cfg.motor_list)
        self.channel = channel
        self.rate = float(np.radians(rate_deg_s))
        self.bounds: np.ndarray | None = None       # (n,2) motor-space command clamp
        self._clip_warned = np.zeros(self.n, dtype=bool)
        log.info("%s: opening %d motors %s (enable = torque-capable, ZERO command — limp)",
                 channel, self.n, [m[0] for m in cfg.motor_list])
        self.chain = DMChainCanInterface(
            [list(m) for m in cfg.motor_list],
            [0.0] * self.n,
            list(cfg.directions),
            channel,
            motor_chain_name=f"yam_chain_{channel}",
            receive_mode=ReceiveMode.p16,
            start_thread=False,
            use_buffered_reader=False,
        )
        # Zero-torque + light damping from the very first streamed frame (replaces
        # i2rt's default of echoing the enable-time torque reading).
        self._store(self._idle_cmds())
        self.chain.start_thread()
        self.streaming = True
        self._normalize_wraps()

    # -- command plumbing ----------------------------------------------------- #
    def _idle_cmds(self):
        return [self._MotorCmd(torque=0.0, pos=0.0, vel=0.0, kp=0.0, kd=float(self.idle_kd[i]))
                for i in range(self.n)]

    def _store(self, cmds) -> None:
        with self.chain.command_lock:
            self.chain.commands = cmds

    def _normalize_wraps(self) -> None:
        """Re-anchor offsets so reported angles land in (−π, π] — from a FRESH
        read after the stream is up (the enable-frame echo is not trusted)."""
        time.sleep(0.05)
        pos = self.read_pos()
        corr = wrap_corrections(pos)
        if np.any(corr != 0.0):
            self.chain.motor_offset = self.chain.motor_offset + corr
            log.info("%s: wrap normalization %s turns → %s", self.channel,
                     (corr / TWO_PI).astype(int).tolist(),
                     np.round(np.degrees(self.read_pos()), 1).tolist())

    def set_bounds(self, lo, hi) -> None:
        """Motor-space command clamp (the calibrated image of the model limits)."""
        lo = np.asarray(lo, dtype=float).reshape(-1)[: self.n]
        hi = np.asarray(hi, dtype=float).reshape(-1)[: self.n]
        self.bounds = np.stack([np.minimum(lo, hi), np.maximum(lo, hi)], axis=1)

    def _clamp(self, q: np.ndarray) -> np.ndarray:
        if self.bounds is None:
            return q
        clipped = np.clip(q, self.bounds[:, 0], self.bounds[:, 1])
        off = np.abs(clipped - q) > 1e-3
        for j in np.flatnonzero(off & ~self._clip_warned):
            log.warning("%s j%d: command outside calibrated motor range [%.3f, %.3f] — "
                        "clipped (further clips silent)", self.channel, j + 1,
                        self.bounds[j, 0], self.bounds[j, 1])
            self._clip_warned[j] = True
        return clipped

    # -- modes ----------------------------------------------------------------- #
    def idle(self) -> None:
        """Zero torque + light damping — limp, safe to touch/move by hand."""
        self._store(self._idle_cmds())

    def command_pd(self, q_motor) -> np.ndarray:
        """MIT PD toward q_motor (MOTOR space) with the stock gains. No feedforward.
        Returns the (possibly clamped) command actually stored."""
        q = self._clamp(np.asarray(q_motor, dtype=float).reshape(-1)[: self.n])
        self._store([self._MotorCmd(torque=0.0, pos=float(q[i]), vel=0.0,
                                    kp=float(self.kp[i]), kd=float(self.kd[i]))
                     for i in range(self.n)])
        return q

    def hold(self, q=None) -> np.ndarray:
        """PD hold at q (default: the current measured pose)."""
        return self.command_pd(self.read_pos() if q is None else q)

    # -- state ------------------------------------------------------------------ #
    def read(self):
        infos = self.chain.read_states()
        pos = np.array([m.pos for m in infos])
        vel = np.array([m.vel for m in infos])
        eff = np.array([m.eff for m in infos])
        temp = np.array([m.temp_mos for m in infos])
        return pos, vel, eff, temp

    def read_pos(self) -> np.ndarray:
        return self.read()[0]

    def settle_check(self, seconds: float = 1.5):
        """(still, pos_mean, vmax, ptp): the arm must be hanging STILL."""
        t0 = time.monotonic()
        ps, vmax = [], 0.0
        while time.monotonic() - t0 < seconds:
            p, v, _, _ = self.read()
            ps.append(p)
            vmax = max(vmax, float(np.max(np.abs(v))))
            time.sleep(0.05)
        ps = np.array(ps)
        ptp = float(np.max(ps.max(0) - ps.min(0)))
        return (vmax < STILL_VEL and ptp < STILL_PTP), ps.mean(0), vmax, ptp

    # -- shaped multi-waypoint move (bring-up) ----------------------------------- #
    def waypoints(self, q_from: np.ndarray, targets: list[np.ndarray],
                  on_tick=None, dwell_s: float = 1.5) -> None:
        """Glide through targets with PD hold, ≤rate per joint, ~100 Hz.
        on_tick(q_cmd) lets the caller mirror the exact command elsewhere."""
        lo = np.minimum(q_from, np.min(targets, axis=0)) - 0.05
        hi = np.maximum(q_from, np.max(targets, axis=0)) + 0.05
        shaper = JointCommandShaper(q_from, rate_limit=self.rate, smooth_hz=2.0, lo=lo, hi=hi)
        for tgt in targets:
            t_end = time.monotonic() + dwell_s + float(np.max(np.abs(tgt - q_from))) / self.rate
            while time.monotonic() < t_end:
                q_cmd = shaper.shape(tgt, time.monotonic())
                self.command_pd(q_cmd)
                if on_tick is not None:
                    on_tick(q_cmd)
                time.sleep(0.01)
            q_from = tgt

    # -- teardown ----------------------------------------------------------------- #
    def stop_stream_for_watchdog_test(self) -> None:
        """Simulate a process crash: stop sending frames while motors hold."""
        self.chain.running = False
        self.streaming = False
        time.sleep(0.3)

    def off(self) -> None:
        """Release torque, stop the stream, switch the motors OFF, close the bus."""
        try:
            if self.streaming:
                self.idle()
                time.sleep(0.1)
                self.chain.running = False
                self.streaming = False
                time.sleep(0.3)
            for mid, _ in self.chain.motor_list:
                try:
                    self.chain.motor_interface.motor_off(mid)
                except Exception:
                    pass
            self.chain.motor_interface.close()
            log.info("%s: motors off, bus closed", self.channel)
        except Exception as e:
            log.warning("%s: teardown issue (%s) — if the arm is stiff, power-cycle the rig",
                        self.channel, e)
