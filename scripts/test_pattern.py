#!/usr/bin/env python
"""Dashboard test pattern: publish a slow j1 wave on the render stream for 120 s.
NO robot, NO CAN — page verification only."""
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
preview = hb.Preview(rig, "right")
neutral = np.asarray(rig["arms"]["right"]["neutral_q"], float)
print("test pattern running: right j1 waving +/-23 deg for 120 s", flush=True)
t0 = time.monotonic()
try:
    while time.monotonic() - t0 < 120:
        qm = neutral.copy()
        qm[0] = neutral[0] + 0.4 * np.sin(0.8 * (time.monotonic() - t0))
        preview.show(qm)
        time.sleep(1 / 30)
finally:
    preview.close()
    print(f"test pattern done ({getattr(preview, 'n_pub', 0)} frames)", flush=True)
