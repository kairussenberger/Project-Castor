"""Preflight doctor (scripts/doctor.py): classification and parsing logic.
Process/port/adb access is injected or monkeypatched — these tests never touch
the real machine state (and --fix kill paths are never invoked)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("doctor", REPO / "scripts" / "doctor.py")
doctor = importlib.util.module_from_spec(_spec)
sys.modules["doctor"] = doctor          # @dataclass resolves annotations via sys.modules
_spec.loader.exec_module(doctor)

from bimanual_teleop.config import load_rig  # noqa: E402


# --------------------------------------------------------------------------- #
# parsers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("etime,expect", [
    ("05", 5.0),
    ("11:05", 11 * 60 + 5.0),
    ("02:11:05", 2 * 3600 + 11 * 60 + 5.0),
    ("04-02:11:05", 4 * 86400 + 2 * 3600 + 11 * 60 + 5.0),
])
def test_parse_etime_forms(etime, expect):
    assert doctor.parse_etime(etime) == expect


def test_list_listeners_parses_lsof_field_output(monkeypatch):
    out = "\n".join([
        "p4242", "corbit_to_unity", "n127.0.0.1:8122", "n127.0.0.1:8123",
        "p99", "cpython3.12", "n*:8180",
    ])
    monkeypatch.setattr(doctor, "_run", lambda cmd, timeout=10.0: (0, out))
    ls = doctor.list_listeners()
    assert ls[8122] == {"pid": 4242, "cmd": "orbit_to_unity"}
    assert ls[8123]["pid"] == 4242
    assert ls[8180]["cmd"] == "python3.12"


# --------------------------------------------------------------------------- #
# process classification (procs injected — no system access)
# --------------------------------------------------------------------------- #
def _p(kind, pid=100, age=60.0, cmd="x"):
    return {"pid": pid, "age_s": age, "cmd": cmd, "kind": kind}


def test_old_bridge_is_always_a_failure_with_fix():
    f = doctor.check_processes([_p("old-bridge", pid=4242, age=4 * 86400)])
    assert f[0].status == doctor.FAIL and f[0].fix is not None
    assert "orbit_to_unity" in f[0].detail and "4.0d" in f[0].detail


def test_wedged_ffmpeg_without_headset_view_fails():
    f = doctor.check_processes([_p("screen-ffmpeg", pid=7), _p("screen-ffmpeg", pid=8)])
    assert f[0].status == doctor.FAIL and f[0].data["pids"] == [7, 8]


def test_ffmpeg_with_live_headset_view_is_fine():
    f = doctor.check_processes([_p("screen-ffmpeg"), _p("headset-view")])
    assert len(f) == 1 and f[0].status == doctor.OK
    assert "headset_view up" in f[0].detail


def test_ancient_engine_is_an_orphan_but_young_engine_is_fine():
    f = doctor.check_processes([_p("engine", age=13 * 3600)])
    assert f[0].status == doctor.FAIL and "orphan" in f[0].name
    f = doctor.check_processes([_p("engine", age=120), _p("dashboard")])
    assert len(f) == 1 and f[0].status == doctor.OK


# --------------------------------------------------------------------------- #
# ports
# --------------------------------------------------------------------------- #
def test_foreign_listener_on_teleop_port_fails_without_autofix():
    rig = load_rig()
    f = doctor.check_ports(rig, {8122: {"pid": 31337, "cmd": "node"}})
    assert len(f) == 1 and f[0].status == doctor.FAIL
    assert f[0].fix is None                       # never auto-kill foreign processes
    assert "ORBIT ingest" in f[0].detail and "node" in f[0].detail


def test_our_python_listeners_are_expected():
    rig = load_rig()
    f = doctor.check_ports(rig, {8122: {"pid": 9, "cmd": "python3.12"},
                                 8180: {"pid": 10, "cmd": "Python"},
                                 10505: {"pid": 11, "cmd": "ffmpeg"}})
    assert len(f) == 1 and f[0].status == doctor.OK


# --------------------------------------------------------------------------- #
# adb
# --------------------------------------------------------------------------- #
def test_adb_unauthorized_warns_with_headset_hint(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _: "/usr/bin/adb")
    monkeypatch.setattr(doctor, "_run",
                        lambda cmd, timeout=10.0: (1, "error: device unauthorized"))
    f = doctor.check_adb(load_rig())
    assert f[0].status == doctor.WARN and "Allow USB debugging" in f[0].detail


def test_adb_missing_tunnels_warn_and_are_fixable(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _: "/usr/bin/adb")

    def fake_run(cmd, timeout=10.0):
        if cmd[:2] == ["adb", "get-state"]:
            return 0, "device\n"
        if cmd[:3] == ["adb", "reverse", "--list"]:
            return 0, "host tcp:8122 tcp:8122\nhost tcp:8123 tcp:8123\n"
        return 0, ""
    monkeypatch.setattr(doctor, "_run", fake_run)
    f = doctor.check_adb(load_rig())
    assert f[0].status == doctor.WARN and f[0].fix is not None
    assert 8200 in f[0].data["missing"] and 10505 in f[0].data["missing"]
    assert 8122 not in f[0].data["missing"]


def test_adb_all_tunnels_ok(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _: "/usr/bin/adb")
    tunnels = "\n".join(f"host tcp:{p} tcp:{p}"
                        for p in doctor.orbit_source._TUNNEL_PORTS)

    def fake_run(cmd, timeout=10.0):
        return (0, "device\n") if cmd[:2] == ["adb", "get-state"] else (0, tunnels)
    monkeypatch.setattr(doctor, "_run", fake_run)
    assert doctor.check_adb(load_rig())[0].status == doctor.OK


# --------------------------------------------------------------------------- #
# calibration file
# --------------------------------------------------------------------------- #
def _good_fit():
    from bimanual_teleop.vr.neutral_calib import (ROBOT_NEUTRAL_DEFAULT,
                                                  ROBOT_REST_DEFAULT, fit_two_pose)
    pa = {"left": np.array([-0.22, 0.05, 0.45]), "right": np.array([0.22, 0.05, 0.45])}
    pb = {"left": np.array([-0.20, -0.45, 0.0]), "right": np.array([0.20, -0.45, 0.0])}
    rn = {s: np.asarray(ROBOT_NEUTRAL_DEFAULT[s]) for s in ("left", "right")}
    rr = {s: np.asarray(ROBOT_REST_DEFAULT[s]) for s in ("left", "right")}
    return fit_two_pose(pa, pb, rn, rr)


def _rig_with(tmp_path, name="calib.json"):
    rig = load_rig()
    rig["mapping"]["calib_file"] = str(tmp_path / name)
    return rig, tmp_path / name


def test_calibration_missing_is_normal(tmp_path):
    rig, _ = _rig_with(tmp_path)
    f = doctor.check_calibration(rig)
    assert f[0].status == doctor.OK and "in-session" in f[0].detail


def test_calibration_good_fresh_then_stale(tmp_path):
    import time as _time
    rig, path = _rig_with(tmp_path)
    res = _good_fit()
    res.save(path)
    now = _time.mktime(_time.strptime(res.meta["stamp"], "%Y-%m-%d %H:%M:%S"))
    f = doctor.check_calibration(rig, now=now + 60)
    assert f[0].status == doctor.OK and "GOOD" in f[0].detail
    f = doctor.check_calibration(rig, now=now + 13 * 3600)
    assert f[0].status == doctor.WARN and "recalibrate" in f[0].detail


def test_calibration_corrupt_fails(tmp_path):
    rig, path = _rig_with(tmp_path)
    path.write_text('{"version": 4, "axis_scale": [99, 1, 1], "body_offset": [0, 0, 0]}')
    f = doctor.check_calibration(rig)
    assert f[0].status == doctor.FAIL


# --------------------------------------------------------------------------- #
# venv / iCloud trap
# --------------------------------------------------------------------------- #
def _mk_venv(root: Path) -> Path:
    sp = root / ".venv" / "lib" / "python3.12" / "site-packages"
    sp.mkdir(parents=True)
    (sp / "_proj.pth").write_text("/x\n")
    return root / ".venv"


def test_venv_outside_icloud_ok(tmp_path):
    f = doctor.check_venv(_mk_venv(tmp_path))
    assert f[0].status == doctor.OK


def test_venv_inside_icloud_path_fails(tmp_path):
    root = tmp_path / "Library" / "Mobile Documents" / "proj"
    root.mkdir(parents=True)
    f = doctor.check_venv(_mk_venv(root))
    assert f[0].status == doctor.FAIL and "iCloud" in f[0].detail


@pytest.mark.skipif(sys.platform != "darwin", reason="chflags is macOS-only")
def test_venv_hidden_pth_fails(tmp_path):
    venv = _mk_venv(tmp_path)
    pth = next(venv.glob("lib/python*/site-packages/*.pth"))
    subprocess.run(["chflags", "hidden", str(pth)], check=True)
    f = doctor.check_venv(venv)
    assert f[0].status == doctor.FAIL and "UF_HIDDEN" in f[0].detail
