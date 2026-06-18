"""The dashboard page's inline <script> must PARSE. PAGE is a plain Python
triple-quoted string, so a stray Python escape (e.g. '\\n' written as a real
newline) lands INSIDE a JS string literal and the browser throws SyntaxError —
which kills the WHOLE script: no handlers bind and every button silently dies
while the HTTP server stays healthy (field incident, 2026-06-12: one '\\n' in
a tooltip bricked the page; the engine could not be started from the UI)."""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _page() -> str:
    spec = importlib.util.spec_from_file_location(
        "dashboard_for_js_check", REPO / "scripts" / "dashboard.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dashboard_for_js_check"] = mod
    spec.loader.exec_module(mod)
    return mod.PAGE


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_dashboard_inline_script_parses(tmp_path):
    m = re.search(r"<script>(.*)</script>", _page(), re.S)
    assert m, "PAGE lost its <script> block?"
    js = tmp_path / "page.js"
    js.write_text(m.group(1), encoding="utf-8")
    r = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, f"dashboard page JS does not parse:\n{r.stderr}"
