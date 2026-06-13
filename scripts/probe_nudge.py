#!/usr/bin/env python
"""ONE instrumented nudge on motor j1: telemetry proof of motion + preview frames.

Opens the chain, settle-checks, PD-holds, sweeps j1 +2/-2/rest at <=10 deg/s while
sampling MEASURED positions at 20 Hz, previews on the dashboard (x12), then releases.
Prints the achieved deflection so nobody has to trust their eyes.
"""
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.config import load_rig  # noqa: E402

spec = importlib.util.spec_from_file_location("hw_bringup", REPO_ROOT / "scripts" / "hw_bringup.py")
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)

rig = load_rig()
side = "right"
channel = rig["arms"][side]["can_channel"]
amp = np.radians(2.0)

sess = hb.open_chain(channel)
preview = None
try:
    try:
        preview = hb.Preview(rig, side)
        print("preview: publishing on render ports OK")
    except Exception as e:
        print(f"preview: DISABLED ({e}) - nudge continues without dashboard")
    still, q_rest, vmax, ptp = sess.settle_check()
    print(f"rest pose (motor): {np.round(np.degrees(q_rest), 1).tolist()}")
    print(f"still: {still} (vmax={vmax:.3f} rad/s, drift={np.degrees(ptp):.2f} deg)")
    if not still:
        print("ABORT: arm not still")
        sys.exit(1)
    sess.hold(q_rest)
    time.sleep(0.8)
    neutral = np.asarray(rig["arms"][side]["neutral_q"], float)
    trace = []
    last = [0.0]

    def tick(q_cmd):
        now = time.monotonic()
        if preview is not None:
            qm = neutral.copy()
            qm[0] = neutral[0] + 12.0 * (q_cmd[0] - q_rest[0])
            preview.show(qm)
        if now - last[0] > 0.05:
            last[0] = now
            trace.append((now, float(q_cmd[0]), float(sess.read_pos()[0])))

    a, b = q_rest.copy(), q_rest.copy()
    a[0] += amp
    b[0] -= amp
    print("NUDGING j1 now: +2 deg, -2 deg, back to rest (~6 s) - WATCH THE ARM")
    t0 = time.monotonic()
    sess.waypoints(q_rest, [a, b, q_rest.copy()], on_tick=tick)
    dt = time.monotonic() - t0
    tr = np.array(trace)
    cmd_span = np.degrees(tr[:, 1].max() - tr[:, 1].min())
    meas_span = np.degrees(tr[:, 2].max() - tr[:, 2].min())
    gap = np.degrees(np.abs(tr[:, 1] - tr[:, 2]).max())
    print(f"done in {dt:.1f} s, {len(trace)} samples")
    print(f"COMMANDED j1 span: {cmd_span:.2f} deg")
    print(f"MEASURED  j1 span: {meas_span:.2f} deg   <-- the metal physically moved this much")
    print(f"worst cmd-vs-measured gap: {gap:.2f} deg")
    print(f"preview frames published: {getattr(preview, 'n_pub', 'n/a')}")
finally:
    if preview is not None:
        preview.close()
    sess.off()
    print("torque released, bus closed")
