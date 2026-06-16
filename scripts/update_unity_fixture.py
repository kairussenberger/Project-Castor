#!/usr/bin/env python
"""Generate or check Unity's representative render-state fixture.

The fixture used by the Unity Editor validation should come from the actual Python
render-state builder, not from a separate hand-written JSON blob. `--check` fails
if the checked-in fixture drifts from the current publisher schema.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

from bimanual_teleop.config import SIDES, load_rig
from bimanual_teleop.engine import TeleopEngine
from bimanual_teleop.render_sink import RenderSink
from bimanual_teleop.vr.ingest import FakeVRSource


REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "unity" / "TeleopRenderer" / "Assets" / "Editor" / "render_state_sample.json"


def make_fixture() -> dict:
    rig = load_rig()
    rig["vr"]["calib_seconds"] = 0
    rig["vr"]["body_relative"] = True
    rig.setdefault("mapping", {})["swap_sides"] = False   # canonical fixture = same-side mapping
    rig["vr"]["render_endpoint"] = "inproc://unity-fixture"
    rig["vr"]["unity_json_endpoint"] = None

    sink = RenderSink(rig)
    try:
        engine = TeleopEngine(rig, sink)
        src = FakeVRSource()
        t = 0.5
        frame = src.frame_at(t)
        engaged = {side: True for side in SIDES}
        engine.tick(frame, engaged, t)
        return sink.build_state(engine, frame, engaged, hz=100.0, t=t)
    finally:
        sink.close()


def normalized_text(obj: dict) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="fail if the fixture is stale")
    group.add_argument("--write", action="store_true", help="rewrite the fixture")
    args = ap.parse_args()

    expected = normalized_text(make_fixture())
    if args.write:
        FIXTURE.write_text(expected, encoding="utf-8")
        print(f"wrote {FIXTURE.relative_to(REPO)}")
        return 0

    try:
        actual = FIXTURE.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"missing fixture: {FIXTURE.relative_to(REPO)}", file=sys.stderr)
        return 1
    if actual != expected:
        diff = difflib.unified_diff(
            actual.splitlines(),
            expected.splitlines(),
            fromfile=str(FIXTURE.relative_to(REPO)),
            tofile="generated",
            lineterm="",
        )
        print("Unity render-state fixture is stale; run:", file=sys.stderr)
        print("  uv run python scripts/update_unity_fixture.py --write", file=sys.stderr)
        print("\n".join(diff), file=sys.stderr)
        return 1
    print("Unity render-state fixture matches Python publisher")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
