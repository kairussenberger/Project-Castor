#!/usr/bin/env python
"""DEPRECATED — superseded by jog_arms --sink hw.

The original jog_right.py predated the measured motor↔model joint map and the
rest-pose gate: it pushed model-space angles straight onto the bus and let the
i2rt driver start in zero-gravity mode with TABLE-MOUNT gravity compensation —
wrong for this rig's sideways arm mounts. The safe path (joint map + gate +
no-grav-comp energize + 10°/s cap) lives in the shared jog:

    uv run python scripts/jog_arms.py --sink hw

This shim forwards there with the same defaults the old script promised.
First-time bring-up (mapping not measured yet)?  uv run python scripts/hw_bringup.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    print(__doc__)
    target = Path(__file__).resolve().with_name("jog_arms.py")
    os.execv(sys.executable, [sys.executable, str(target), "--sink", "hw", *sys.argv[1:]])
