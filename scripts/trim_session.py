#!/usr/bin/env python
"""Trim a recorded session and/or embed its calibration — the replay-library
curation tool.

    uv run python scripts/trim_session.py recordings/tape.npz --tail 4
    uv run python scripts/trim_session.py recordings/tape.npz --head 10 --tail 4 \\
        --calib recordings/tape.calib.json -o recordings/tape.npz

Trimming drops the walk-to-the-laptop seconds (every dashboard-stopped tape
ends with ~4 s of the operator reaching for STOP). --calib injects a
calibration payload (the persisted operator_calib.json format) into the npz
`calib_json` key for recordings made before the recorder embedded fits
automatically. In-place writes keep the original as `<name>.raw.npz` (first
time only — re-running never clobbers the true original)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def trim_arrays(data: dict, head_s: float = 0.0, tail_s: float = 0.0) -> dict:
    """Slice every per-frame array to t ∈ [t0+head_s, t_end−tail_s]. Arrays
    whose first dimension is not the frame count (e.g. calib_json) pass
    through untouched. Raises when the window would be empty."""
    t = np.asarray(data["t"], float)
    n = len(t)
    lo, hi = t[0] + float(head_s), t[-1] - float(tail_s)
    keep = (t >= lo) & (t <= hi)
    if not keep.any():
        raise ValueError(f"trim window empty: head={head_s}s tail={tail_s}s "
                         f"of a {t[-1] - t[0]:.1f}s recording")
    out = {}
    for k, v in data.items():
        arr = np.asarray(v)
        out[k] = arr[keep] if arr.ndim >= 1 and arr.shape[0] == n else arr
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="session .npz")
    ap.add_argument("--head", type=float, default=0.0, help="seconds to drop from the start")
    ap.add_argument("--tail", type=float, default=0.0, help="seconds to drop from the end")
    ap.add_argument("--calib", default=None,
                    help="calibration JSON (operator_calib.json format) to embed")
    ap.add_argument("-o", "--out", default=None, help="output path (default: in place)")
    args = ap.parse_args()

    src = Path(args.path)
    data = dict(np.load(src, allow_pickle=False))
    n0, dur0 = len(data["t"]), float(data["t"][-1] - data["t"][0])
    data = trim_arrays(data, args.head, args.tail)
    if args.calib:
        payload = json.loads(Path(args.calib).read_text())
        data["calib_json"] = np.array(json.dumps(payload))
    out = Path(args.out) if args.out else src
    if out.resolve() == src.resolve():
        raw = src.with_suffix(".raw.npz")
        if not raw.exists():                      # never clobber the true original
            src.rename(raw)
    np.savez_compressed(out, **data)
    t = data["t"]
    print(f"{src.name}: {n0} frames / {dur0:.1f}s → {len(t)} frames / "
          f"{float(t[-1] - t[0]):.1f}s"
          + (f", calib embedded ({args.calib})" if args.calib else "")
          + f" → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
