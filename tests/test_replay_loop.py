"""run_hw --loop-home: the between-takes home glide (hardware-free).

glide_arms_home() is the only new control-path code in the loop; this pins its contract
with a fake sink + stub engine (no Pinocchio, no CAN, no real time):
  - every tick commands the REST pose through the sink (the shaper does the rate-limited
    interpolation — it is never bypassed),
  - render/telemetry side effects fire each tick via on_tick,
  - on exit, the engine IK and BOTH shapers are re-synced to rest so the next take starts
    clean instead of yanking the already-home arm back to the replay-end pose.

    uv run pytest tests/test_replay_loop.py -q
"""
from __future__ import annotations

import numpy as np

from bimanual_teleop.launch.run_hw import glide_arms_home


class _FakeClock:
    """Monotonic clock that advances a fixed step every call (lets the timed glide loop
    terminate deterministically without real sleeps)."""

    def __init__(self, step=0.05):
        self.t = 0.0
        self.step = step

    def __call__(self):
        v = self.t
        self.t += self.step
        return v


class _FakeIK:
    def __init__(self, q0, q):
        self.q0 = np.asarray(q0, float)
        self._q = np.asarray(q, float)
        self.reset_called = 0

    @property
    def q(self):
        return self._q.copy()

    def reset(self):
        self.reset_called += 1
        self._q = self.q0.copy()


class _FakeShaper:
    def __init__(self):
        self.reset_args = []

    def reset(self, q0, t):
        self.reset_args.append((np.asarray(q0, float).copy(), t))


class _FakeArm:
    def __init__(self, q0, q):
        self.ik = _FakeIK(q0, q)
        self.shaper = _FakeShaper()


class _FakeEngine:
    def __init__(self, arms):
        self.arm = arms


class _FakeSink:
    def __init__(self, sides):
        self.shapers = {s: _FakeShaper() for s in sides}
        self.commands = {s: [] for s in sides}

    def set_arm(self, side, q):
        self.commands[side].append(np.asarray(q, float).copy())


def test_glide_arms_home_commands_rest_and_resyncs():
    sides = ["right", "left"]
    q0 = {"right": np.zeros(6), "left": np.zeros(6)}
    q_end = {"right": np.full(6, 0.5), "left": np.full(6, -0.4)}   # lifted at replay end
    arms = {s: _FakeArm(q0[s], q_end[s]) for s in sides}
    engine = _FakeEngine(arms)
    sink = _FakeSink(sides)
    ticks = []

    glide_arms_home(engine, sink, sides, period=0.01,
                    rate_cap=1.0, dwell_s=0.2,
                    on_tick=lambda t: ticks.append(t),
                    clock=_FakeClock(0.05), sleep=lambda _dt: None, margin_s=0.1)

    assert ticks, "on_tick never fired during the glide"
    for s in sides:
        assert sink.commands[s], f"{s}: no commands issued"
        # Every commanded target is the REST pose — the shaper (not this helper) does the
        # rate-limited interpolation, so the helper just holds the goal.
        assert all(np.allclose(c, q0[s]) for c in sink.commands[s])
        # Re-sync: engine IK reset once, both shapers re-seeded at rest.
        assert arms[s].ik.reset_called == 1
        assert arms[s].shaper.reset_args, f"{s}: arm-control shaper not reset"
        assert sink.shapers[s].reset_args, f"{s}: hardware shaper not reset"
        assert np.allclose(arms[s].shaper.reset_args[-1][0], q0[s])
        assert np.allclose(sink.shapers[s].reset_args[-1][0], q0[s])


def test_glide_arms_home_skips_unwired_sides():
    """Only sides present in engine.arm are touched; an unknown side is ignored (no
    KeyError) so passing the full hardware.sides list is always safe."""
    arms = {"right": _FakeArm(np.zeros(6), np.full(6, 0.3))}
    engine = _FakeEngine(arms)
    sink = _FakeSink(["right"])

    glide_arms_home(engine, sink, ["right", "left"], period=0.01,
                    rate_cap=1.0, dwell_s=0.0,
                    clock=_FakeClock(0.1), sleep=lambda _dt: None, margin_s=0.05)

    assert sink.commands["right"]
    assert arms["right"].ik.reset_called == 1


def test_glide_arms_home_no_sides_is_noop():
    engine = _FakeEngine({})
    sink = _FakeSink([])
    # Must not raise (and the clock is never consulted past the guard).
    glide_arms_home(engine, sink, [], period=0.01, rate_cap=1.0, dwell_s=0.0,
                    clock=_FakeClock(0.1), sleep=lambda _dt: None)
