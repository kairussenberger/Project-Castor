#!/usr/bin/env python
"""Teleop preflight DOCTOR: find (and with --fix, repair) the environmental
wedges that have actually eaten session time, before they eat more of it.

Every failure mode here is from the field, not imagination:
- a 4-day-old `orbit_to_unity.py` (the retired Unity-bridge project) squatting
  all seven ORBIT ports and silently blocking every dashboard engine spawn
  (2026-06-11 — cost the first 20 minutes of the session);
- wedged `ffmpeg` avfoundation screen-captures from dead headset_view runs
  starving new captures: zero frames, no error (same day, twice);
- `adb reverse` tunnels silently dying on cable/sleep events (the engine's own
  watchdog re-asserts them while it runs — but only while it runs);
- a stale/corrupt persisted calibration steering a session (now graded at fit
  time — the doctor surfaces the grade + age before you put the headset on);
- the iCloud venv trap: a .venv inside an iCloud-synced folder gets UF_HIDDEN
  set on its .pth files and CPython ≥3.12 silently skips them →
  ModuleNotFoundError for every editable install (cost most of a day once).

    uv run python scripts/doctor.py            # report, exit 1 if anything FAILS
    uv run python scripts/doctor.py --fix      # also repair what is safely ours
    uv run python scripts/doctor.py --json     # machine-readable findings

--fix only ever kills processes whose command line matches THIS project's own
tooling (engines, the old bridge, headset-view ffmpeg). Foreign processes on
our ports are reported with the kill command left to you."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import stat as stat_mod
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.config import load_rig                      # noqa: E402
from bimanual_teleop.vr import orbit_source                      # noqa: E402
from bimanual_teleop.vr.neutral_calib import load_calibration    # noqa: E402

OK, WARN, FAIL = "ok", "warn", "fail"
_MARK = {OK: "✓", WARN: "⚠", FAIL: "✗"}

# Our own process signatures — the ONLY things --fix may kill. The engine
# pattern matches `python -m bimanual_teleop.launch.run_teleop` (dashboard's
# anchor trick: '[-]m' never matches an editor holding the file open).
OURS = {
    "engine": r"[-]m bimanual_teleop\.launch\.run_(teleop|hw)",
    "old-bridge": r"orbit_to_unity\.py",
    "headset-view": r"headset_view\.py",
    "screen-ffmpeg": r"ffmpeg\b.*avfoundation",
    "dashboard": r"dashboard\.py",
}
ENGINE_MAX_AGE_S = 12 * 3600        # an engine older than this is an orphan
CALIB_STALE_S = 12 * 3600           # a fit older than this deserves a fresh one


@dataclass
class Finding:
    name: str
    status: str
    detail: str
    fix_desc: str = ""
    fix: object = None              # zero-arg callable, set only when --fix can act
    data: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# plumbing (monkeypatch points for tests)
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (subprocess.SubprocessError, OSError) as e:
        return -1, str(e)


def _kill(pids: list[int], grace_s: float = 10.0) -> str:
    """INT first (engines save their recording on INT), escalate to KILL."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.time() + grace_s
    left = set(pids)
    while left and time.time() < deadline:
        time.sleep(0.3)
        left = {p for p in left if _alive(p)}
    for pid in left:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return f"killed {len(pids)} process(es)" + (f" ({len(left)} needed SIGKILL)" if left else "")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def parse_etime(etime: str) -> float:
    """ps etime ('[[dd-]hh:]mm:ss') → seconds."""
    etime = etime.strip()
    days = 0
    if "-" in etime:
        d, etime = etime.split("-", 1)
        days = int(d)
    parts = [int(p) for p in etime.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return float(((days * 24 + h) * 60 + m) * 60 + s)


def _fmt_age(seconds: float) -> str:
    if seconds >= 86400:
        return f"{seconds / 86400:.1f}d"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{int(seconds // 60)}m"
    return f"{int(seconds)}s"


def list_processes() -> list[dict]:
    """All processes matching one of OUR patterns: {pid, age_s, cmd, kind}."""
    rc, out = _run(["ps", "-axo", "pid=,etime=,command="])
    if rc != 0:
        return []
    found = []
    me = os.getpid()
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+(\S+)\s+(.*)", line)
        if not m:
            continue
        pid, etime, cmd = int(m.group(1)), m.group(2), m.group(3)
        if pid == me or "doctor.py" in cmd:
            continue
        for kind, pat in OURS.items():
            if re.search(pat, cmd):
                try:
                    age = parse_etime(etime)
                except ValueError:
                    age = 0.0
                found.append({"pid": pid, "age_s": age, "cmd": cmd, "kind": kind})
                break
    return found


def list_listeners() -> dict[int, dict]:
    """TCP listeners by port: {port: {pid, cmd}} (one lsof call)."""
    rc, out = _run(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-Fpcn"])
    if rc != 0 and not out.strip():
        return {}
    listeners: dict[int, dict] = {}
    pid, cmd = None, ""
    for line in out.splitlines():
        if not line:
            continue
        tag, val = line[0], line[1:]
        if tag == "p":
            pid, cmd = int(val), ""
        elif tag == "c":
            cmd = val
        elif tag == "n":
            m = re.search(r":(\d+)$", val)
            if m and pid is not None:
                listeners.setdefault(int(m.group(1)), {"pid": pid, "cmd": cmd})
    return listeners


def expected_ports(rig: dict) -> dict[int, str]:
    """Port → role, from the same sources the runtime uses."""
    ports: dict[int, str] = {}
    for p in orbit_source._ALL_PORTS:
        ports[int(p)] = "ORBIT ingest (engine binds)"
    ports[int(orbit_source.VIDEO_PANEL_PORT)] = "in-headset video panel (headset_view binds)"
    v = rig.get("vr", {})
    for key, role in (("render_endpoint", "render ZMQ (engine binds)"),
                      ("unity_json_endpoint", "render TCP JSON (engine binds)")):
        ep = v.get(key)
        if ep:
            try:
                ports[int(str(ep).rsplit(":", 1)[1])] = role
            except (ValueError, IndexError):
                pass
    ports[int(v.get("control_port", 8201))] = "engine control (engine binds)"
    ports[int(v.get("orbit_viz_port", 8099))] = "orbit hand viz (engine binds)"
    ports[8180] = "dashboard HTTP"
    return ports


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_processes(procs: list[dict] | None = None) -> list[Finding]:
    procs = list_processes() if procs is None else procs
    out: list[Finding] = []
    bridge = [p for p in procs if p["kind"] == "old-bridge"]
    if bridge:
        pids = [p["pid"] for p in bridge]
        worst = max(p["age_s"] for p in bridge)
        out.append(Finding(
            "old orbit bridge", FAIL,
            f"orbit_to_unity.py running (pid {pids}, up {_fmt_age(worst)}) — the retired "
            "Unity-bridge project; it squats the ORBIT ports and blocks engine spawns",
            fix_desc="kill it", fix=lambda p=pids: _kill(p, grace_s=2.0),
            data={"pids": pids}))
    hv_alive = any(p["kind"] == "headset-view" for p in procs)
    ff = [p for p in procs if p["kind"] == "screen-ffmpeg"]
    if ff and not hv_alive:
        pids = [p["pid"] for p in ff]
        out.append(Finding(
            "screen capture", FAIL,
            f"{len(ff)} ffmpeg avfoundation capture(s) with NO headset_view running "
            f"(pid {pids}) — wedged captures starve new ones: zero frames, no error",
            fix_desc="kill them", fix=lambda p=pids: _kill(p, grace_s=2.0),
            data={"pids": pids}))
    engines = [p for p in procs if p["kind"] == "engine"]
    old = [p for p in engines if p["age_s"] > ENGINE_MAX_AGE_S]
    if old:
        pids = [p["pid"] for p in old]
        worst = max(p["age_s"] for p in old)
        out.append(Finding(
            "orphan engine", FAIL,
            f"engine up {_fmt_age(worst)} (pid {pids}) — almost certainly orphaned; "
            "it holds every engine port",
            fix_desc="kill it (INT first — it saves its recording)",
            fix=lambda p=pids: _kill(p), data={"pids": pids}))
    live = [p for p in engines if p["age_s"] <= ENGINE_MAX_AGE_S]
    bits = []
    if live:
        bits.append(f"engine up {_fmt_age(max(p['age_s'] for p in live))}")
    if any(p["kind"] == "dashboard" for p in procs):
        bits.append("dashboard up")
    if hv_alive:
        bits.append("headset_view up")
    if not out:
        out.append(Finding("processes", OK,
                           "no strays" + (f" ({', '.join(bits)})" if bits else "")))
    return out


def check_ports(rig: dict, listeners: dict[int, dict] | None = None) -> list[Finding]:
    listeners = list_listeners() if listeners is None else listeners
    ours = re.compile("|".join(f"(?:{p})" for p in OURS.values()))
    # lsof's c-field is a short command name; accept the obvious runtimes too.
    benign_names = re.compile(r"^(python[\d.]*|Python|ffmpeg|uv)$")
    squat: list[Finding] = []
    for port, role in sorted(expected_ports(rig).items()):
        own = listeners.get(port)
        if own is None:
            continue
        cmd = own.get("cmd", "")
        if ours.search(cmd) or benign_names.match(cmd):
            continue                       # our tooling (or a python that is ours)
        squat.append(Finding(
            f"port {port}", FAIL,
            f"{role} held by '{cmd}' (pid {own['pid']}) — not our tooling; "
            f"inspect with: lsof -nP -iTCP:{port} ; kill it yourself if expected",
            data={"port": port, "pid": own["pid"], "cmd": cmd}))
    if not squat:
        return [Finding("ports", OK, "no foreign listeners on teleop ports")]
    return squat


def check_adb(rig: dict) -> list[Finding]:
    if not shutil.which("adb"):
        return [Finding("quest link", WARN,
                        "adb not found — install platform-tools or set up "
                        "`adb reverse` on another machine")]
    rc, out = _run(["adb", "get-state"], timeout=5)
    state = out.strip() if rc == 0 and out.strip() else \
        ("unauthorized" if "unauthorized" in out else "disconnected")
    if state != "device":
        hint = {"unauthorized": "put the headset ON and tap 'Allow USB debugging'",
                "disconnected": "plug the Quest USB cable in"}.get(state, state)
        return [Finding("quest link", WARN, f"adb: {state} — {hint}",
                        data={"state": state})]
    rc, out = _run(["adb", "reverse", "--list"], timeout=5)
    have = {int(a) for a, b in re.findall(r"tcp:(\d+)\s+tcp:(\d+)", out) if a == b}
    want = list(orbit_source._TUNNEL_PORTS)
    missing = [p for p in want if p not in have]
    if missing:
        def fix(ports=missing):
            ok = 0
            for p in ports:
                rc, _ = _run(["adb", "reverse", f"tcp:{p}", f"tcp:{p}"], timeout=5)
                ok += (rc == 0)
            return f"re-asserted {ok}/{len(ports)} reverse tunnels"
        return [Finding("quest link", WARN,
                        f"device connected, but {len(missing)}/{len(want)} reverse "
                        f"tunnels missing: {missing} (engine re-asserts on start; "
                        "fix matters for an already-running engine)",
                        fix_desc="adb reverse the missing ports", fix=fix,
                        data={"missing": missing})]
    return [Finding("quest link", OK,
                    f"device connected, {len(want)}/{len(want)} reverse tunnels up")]


def check_calibration(rig: dict, now: float | None = None) -> list[Finding]:
    raw = rig.get("mapping", {}).get("calib_file", "config/operator_calib.json")
    if not raw:
        return [Finding("calibration", OK, "disabled in rig")]
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return [Finding("calibration", OK,
                        "no persisted fit (normal: you calibrate in-session)")]
    res = load_calibration(path)
    if res is None:
        return [Finding("calibration", FAIL,
                        f"{path.name} exists but fails the load-time screen "
                        "(corrupt/out-of-range) — delete it or recalibrate",
                        data={"path": str(path)})]
    q = (res.meta or {}).get("quality") or {}
    stamp = (res.meta or {}).get("stamp")
    age_txt, stale = "age unknown", False
    if stamp:
        try:
            age = (now if now is not None else time.time()) - time.mktime(
                time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))
            age_txt, stale = f"{_fmt_age(age)} old", age > CALIB_STALE_S
        except ValueError:
            pass
    grade = q.get("grade", "ungraded")
    detail = (f"fit {grade.upper()}"
              + (f" (worst {q['worst_cm']} cm)" if q.get("worst_cm") is not None else "")
              + f", {age_txt}")
    status = OK
    if grade == "bad":
        status, detail = FAIL, detail + " — recalibrate before driving"
    elif grade in ("check", "ungraded") or stale:
        status, detail = WARN, detail + " — recalibrate this session"
    if bool(rig.get("vr", {}).get("require_calibration", True)):
        detail += " (require_calibration is on: live sessions recalibrate anyway)"
    return [Finding("calibration", status, detail,
                    data={"grade": grade, "stamp": stamp})]


def check_venv(venv: Path | None = None) -> list[Finding]:
    """The iCloud trap (proven on this machine): a venv inside an iCloud-synced
    folder gets UF_HIDDEN on its .pth files and CPython ≥3.12 silently skips
    them — every editable import breaks with ModuleNotFoundError."""
    venv = (REPO_ROOT / ".venv") if venv is None else venv
    if not venv.exists():
        return [Finding("venv", WARN, f"{venv} missing — run `uv sync`")]
    real = venv.resolve()
    icloudish = ("Mobile Documents" in str(real)) or ("com~apple~CloudDocs" in str(real))
    hidden = []
    for pth in real.glob("lib/python*/site-packages/*.pth"):
        try:
            if os.stat(pth).st_flags & stat_mod.UF_HIDDEN:
                hidden.append(pth.name)
        except (OSError, AttributeError):
            pass
    if icloudish or hidden:
        why = []
        if icloudish:
            why.append(f"venv resolves into iCloud ({real})")
        if hidden:
            why.append(f"UF_HIDDEN on {hidden} — CPython will skip them")
        return [Finding("venv", FAIL,
                        "; ".join(why) + " — move the venv outside iCloud: "
                        "uv venv ~/.venvs/bimanual && ln -sfn ~/.venvs/bimanual .venv && uv sync",
                        data={"hidden": hidden, "icloud": icloudish})]
    return [Finding("venv", OK, f"outside iCloud, no hidden .pth ({real})")]


def check_recordings() -> list[Finding]:
    rec = REPO_ROOT / "recordings"
    if not rec.exists():
        return [Finding("recordings", OK, "none yet")]
    files = list(rec.glob("*.npz"))
    size = sum(f.stat().st_size for f in files)
    status = WARN if size > 20e9 else OK
    return [Finding("recordings", status,
                    f"{len(files)} sessions, {size / 1e9:.1f} GB"
                    + (" — prune old ones" if status == WARN else ""))]


# --------------------------------------------------------------------------- #
def run_all(rig: dict | None = None) -> list[Finding]:
    rig = rig or load_rig()
    out: list[Finding] = []
    out += check_processes()
    out += check_ports(rig)
    out += check_adb(rig)
    out += check_calibration(rig)
    out += check_venv()
    out += check_recordings()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix", action="store_true",
                    help="repair what is safely ours (kill strays, re-assert tunnels)")
    ap.add_argument("--json", action="store_true", help="machine-readable findings")
    args = ap.parse_args()

    findings = run_all()
    if args.fix:
        for f in findings:
            if f.fix is not None and f.status in (WARN, FAIL):
                result = f.fix()
                f.detail += f"  → FIXED: {result}"
        # re-check so the exit code reflects the repaired state
        fixed_any = any(f.fix is not None and f.status in (WARN, FAIL) for f in findings)
        if fixed_any:
            time.sleep(0.5)
            findings = run_all()

    if args.json:
        print(json.dumps([{"name": f.name, "status": f.status, "detail": f.detail,
                           "fixable": f.fix is not None, **({"data": f.data} if f.data else {})}
                          for f in findings], indent=2))
    else:
        print(f"TELEOP PREFLIGHT — {time.strftime('%Y-%m-%d %H:%M:%S')}")
        width = max(len(f.name) for f in findings)
        for f in findings:
            print(f"  {_MARK[f.status]} {f.name:<{width}}  {f.detail}")
        bad = [f for f in findings if f.status == FAIL]
        warn = [f for f in findings if f.status == WARN]
        fixable = [f for f in findings if f.fix is not None and f.status in (WARN, FAIL)]
        if bad or warn:
            tail = f"{len(bad)} failure(s), {len(warn)} warning(s)."
            if fixable and not args.fix:
                tail += " Run with --fix to repair "
                tail += "(" + "; ".join(f.fix_desc for f in fixable if f.fix_desc) + ")."
            print(tail)
        else:
            print("All clear.")
    return 1 if any(f.status == FAIL for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
