#!/usr/bin/env python
"""Teleop dashboard — one browser page showing every piece of the puzzle, live.

Consumes the same newline-delimited TCP JSON `render.state` stream Unity uses
(no new schema, no extra deps) and serves a local page with:

  - stream/Quest status: connected, state age, loop Hz, per-side TRACKED/ENGAGED,
    calibration banner;
  - a drag-to-rotate 3D view: both arm link chains, achieved EE (dot) vs
    commanded target (ring), operator torso→wrist vectors;
  - numbers: per-joint angles (deg) with limit-margin highlighting, torso→wrist
    body vectors [right, up, forward], commanded-vs-achieved EE error.

    uv run python scripts/dashboard.py                 # http://127.0.0.1:8180
    uv run python scripts/dashboard.py --port 8181 --endpoint tcp://127.0.0.1:8102

Run alongside ANY teleop/jog process that has the Unity JSON bridge enabled
(run_teleop, run_hw with a render tee, scripts/jog_arms.py).
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np                                       # noqa: E402

from bimanual_teleop.config import load_rig             # noqa: E402


class MeshAssets:
    """Decimated YAM visual meshes + FK models so the browser draws the REAL robot
    geometry (same source as scripts/render_session.py's GIFs)."""

    def __init__(self, max_tris_per_link: int = 420):
        from bimanual_teleop.viz.yam_meshes import geom_transforms, load_arm_meshes, load_stand_meshes
        from bimanual_teleop.vr.frames import quat_to_R
        self._geom_transforms = geom_transforms
        rig = load_rig()
        self.models = {}
        self.base_T = {}
        self.geoms = {}
        for side in ("left", "right"):
            model, data, items = load_arm_meshes(side, max_tris_per_link=max_tris_per_link)
            T = np.eye(4)
            T[:3, :3] = quat_to_R(rig["arms"][side]["base_quat"])
            T[:3, 3] = rig["arms"][side]["base_pos"]
            self.models[side] = (model, data, items)
            self.base_T[side] = T
            self.geoms[side] = [it["tris"].reshape(-1).round(5).tolist() for it in items]
        self.geoms["stand"] = [t.reshape(-1).round(5).tolist()
                               for t in load_stand_meshes(rig["stand"]["pos"][2], 380)]
        # ORCA hands: the REAL model (sibling orcahand_description repo) when
        # available — geometry shipped once via /meshes, finger FK per state —
        # else the stylized parametric fallback.
        from bimanual_teleop.viz.yam_meshes import (
            load_orca_hand, orca_description_available, orca_q_from_degrees)
        self._q2R = quat_to_R
        self._orca_q = orca_q_from_degrees
        self.hand_models = {}
        self.hand_basis = {}
        if orca_description_available():
            self.hand_mode = "real"
            for side in ("left", "right"):
                model, data, items = load_orca_hand(side, max_tris_per_link // 3)
                self.hand_models[side] = (model, data, items)
                self.geoms[f"hand_{side}"] = [
                    {"v": it["tris"].reshape(-1).round(5).tolist(),
                     "c": [int(255 * v) for v in it["rgb"]]} for it in items]
        else:
            self.hand_mode = "parametric"
            from bimanual_teleop.arms.ik import ArmIK
            from bimanual_teleop.viz.hand_geom import hand_basis_for_side, orca_hand_tris_ee
            self._hand_tris = orca_hand_tris_ee
            for side in ("left", "right"):
                ik = ArmIK(rig, side)
                self.hand_basis[side] = hand_basis_for_side(ik, self.base_T[side][:3, :3], side)
        self._lock = threading.Lock()

    def hand_transforms(self, state: dict) -> dict:
        """REAL-hand mode: per-geom world transforms from streamed EE pose +
        17 ORCA joint angles."""
        out = {}
        hr = state.get("hand_render") or {}
        with self._lock:
            for side, (model, data, items) in self.hand_models.items():
                a = (state.get("arms") or {}).get(side) or {}
                h = hr.get(side) or {}
                if not a.get("ee_pos") or not a.get("ee_quat") or not h.get("q"):
                    continue
                T_ee = np.eye(4)
                T_ee[:3, :3] = self._q2R(a["ee_quat"])
                T_ee[:3, 3] = np.asarray(a["ee_pos"], dtype=float)
                joints = dict(zip(h.get("names", []), h["q"]))
                q = self._orca_q(model, joints, side)
                Ts = self._geom_transforms(model, data, items, q, T_ee)
                out[side] = [np.asarray(T).reshape(-1).round(6).tolist() for T in Ts]
        return out

    def hand_world(self, state: dict) -> dict:
        """World-frame articulated hand triangles from the streamed EE pose +
        17 ORCA joint angles (hand_render)."""
        out = {}
        hr = state.get("hand_render") or {}
        for side in ("left", "right"):
            a = (state.get("arms") or {}).get(side) or {}
            h = hr.get(side) or {}
            if not a.get("ee_pos") or not a.get("ee_quat") or not h.get("q"):
                continue
            joints = dict(zip(h.get("names", []), h["q"]))
            tris = self._hand_tris(joints, self.hand_basis[side], mirror=(side == "left"))
            R = self._q2R(a["ee_quat"])
            tris = tris @ R.T + np.asarray(a["ee_pos"], dtype=float)
            out[side] = tris.reshape(-1).round(5).tolist()
        return out

    def transforms(self, arms_state: dict) -> dict:
        """Per-geom world 4×4 (row-major, flattened) for the streamed joint state."""
        out = {}
        with self._lock:                       # pin data buffers are not thread-safe
            for side, (model, data, items) in self.models.items():
                q = (arms_state.get(side) or {}).get("q")
                if not q:
                    continue
                Ts = self._geom_transforms(model, data, items, q, self.base_T[side])
                out[side] = [np.asarray(T).reshape(-1).round(6).tolist() for T in Ts]
        return out


class StateFeed:
    """Reconnecting reader for the newline-JSON render stream; keeps the latest."""

    def __init__(self, endpoint: str):
        host, port_s = endpoint.removeprefix("tcp://").rsplit(":", 1)
        self.addr = (host, int(port_s))
        self.latest: dict | None = None
        self.rx_time: float = 0.0
        self.connected = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def snapshot(self) -> dict:
        age = (time.monotonic() - self.rx_time) if self.rx_time else None
        return {"connected": self.connected,
                "age": round(age, 3) if age is not None else None,
                "state": self.latest}

    def _run(self) -> None:
        while not self._stop:
            try:
                with socket.create_connection(self.addr, timeout=2.0) as sock:
                    sock.settimeout(None)   # stream is silent between previews — only EOF means the publisher is gone
                    self.connected = True
                    f = sock.makefile("r", encoding="utf-8")
                    while not self._stop:
                        line = f.readline()
                        if not line:
                            break
                        try:
                            self.latest = json.loads(line)
                            self.rx_time = time.monotonic()
                        except json.JSONDecodeError:
                            continue
            except OSError:
                pass
            self.connected = False
            time.sleep(0.5)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>bimanual-teleop dashboard</title>
<style>
 :root{--bg:#0e1116;--panel:#161a21;--ink:#dde3ea;--dim:#76808d;--blue:#6f9fe8;--orange:#e8854a;--gold:#d4af37;--green:#41d98d}
 body{background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,Segoe UI,sans-serif;margin:0}
 header{display:flex;gap:8px;align-items:center;padding:10px 16px;background:#11141a;border-bottom:1px solid #232936;flex-wrap:wrap}
 header b{font-size:15px;margin-right:8px}
 .chip{padding:4px 12px;border-radius:13px;background:#2a2f38;font-weight:600;font-size:13px}
 .ok{background:#1e5d3a}.bad{background:#7c2d2d}.warn{background:#7a6020}
 main{display:grid;grid-template-columns:minmax(900px,1fr) 350px;gap:14px;padding:14px;max-width:1760px}
 .panel{background:var(--panel);border:1px solid #232936;border-radius:12px;padding:10px 12px}
 .duo{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
 .ptitle{font-size:12.5px;font-weight:700;color:#9fb2c8;letter-spacing:.4px;margin:2px 0 6px}
 .ptitle span{color:var(--dim);font-weight:400}
 /* control-system pipeline: 4 stages L→R with arrow gutters between them */
 .pipe{display:grid;grid-template-columns:repeat(4,1fr);align-items:start;gap:0}
 .stage{display:flex;flex-direction:column;min-width:0}
 .stagehd{display:flex;align-items:center;gap:8px;margin:0 2px 6px}
 .stagenum{flex:0 0 auto;width:22px;height:22px;border-radius:50%;background:#2b3550;color:#cfe0ff;font-weight:800;font-size:13px;display:flex;align-items:center;justify-content:center}
 .stagettl{font-size:13px;font-weight:800;color:#cdd9e8;letter-spacing:.3px;line-height:1.1}
 .stagesub{font-size:11px;color:var(--dim);font-weight:500}
 .arrow{flex:0 0 auto;align-self:center;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#4b5566;font-size:26px;padding:0 6px;min-width:36px}
 .arrow.gapcol{min-width:74px;font-size:22px}
 .gapbars{display:flex;gap:2px;align-items:flex-end;height:34px;margin-top:4px}
 .gapbars .gb{width:6px;background:#2a3340;border-radius:2px 2px 0 0;min-height:2px}
 .gaplbl{font-size:9px;color:var(--dim);letter-spacing:.3px;margin-top:3px;text-align:center;font-weight:700}
 .readout{margin-top:7px;font-size:11.5px;line-height:1.45;font-variant-numeric:tabular-nums;min-height:54px}
 .readout .row{display:flex;justify-content:space-between;gap:6px;color:var(--dim)}
 .readout .row b{color:var(--ink);font-weight:600}
 #panelHW{position:relative}
 .placeholder{position:absolute;left:10px;right:10px;top:10px;height:360px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;color:#5a6472;font-weight:800;font-size:15px;letter-spacing:.4px;pointer-events:none;background:rgba(11,14,19,.55);border-radius:8px}
 .placeholder span{font-weight:500;font-size:11px;color:#454e5b;margin-top:5px}
 canvas{display:block;border-radius:8px;background:#0b0e13;cursor:grab;width:100%}
 h3{margin:2px 0 8px;font-size:14px} .armttl{display:flex;justify-content:space-between;align-items:baseline}
 .gauge{display:grid;grid-template-columns:24px 1fr 62px;gap:8px;align-items:center;margin:3px 0;font-variant-numeric:tabular-nums}
 .bar{position:relative;height:12px;background:#222833;border-radius:6px;overflow:hidden}
 .bar .zone{position:absolute;top:0;bottom:0;background:#3a2026}
 .bar .tick{position:absolute;top:-1px;bottom:-1px;width:3px;background:#aab7c9;border-radius:2px}
 .bar .tick.alert{background:#ff6b6b}
 .kv{display:flex;justify-content:space-between;color:var(--dim);font-size:12.5px;margin:2px 0}
 .kv b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
 .err-ok{color:var(--green)} .err-bad{color:#ff8a8a}
 .ctrlbar{display:flex;gap:9px;align-items:center;padding:9px 16px;background:#141925;border-bottom:1px solid #232936;flex-wrap:wrap}
 .btn{cursor:pointer;border:0;border-radius:8px;padding:8px 16px;font-weight:700;color:#fff}
 .btn.sm{padding:7px 11px;font-weight:600;font-size:13px}
 .btn.live{background:#1e5d3a}.btn.stop{background:#6a2626}
 .btn.kill{background:#b3261e;box-shadow:0 0 0 1px #e0584f inset}
 .btn.cal{background:#8a6d1a}.btn.play{background:#2b4a7a}.btn.ghost{background:#3a3f4b;color:#cdd5df}
 .btn:disabled{opacity:.4;cursor:not-allowed}
 .sel{background:#222833;color:#dde3ea;border:1px solid #353c4a;border-radius:8px;padding:7px 9px;max-width:240px}
 .meta{color:#8d97a5;font-size:12px;font-variant-numeric:tabular-nums}
 .slider{vertical-align:middle;width:120px;accent-color:#6f9fe8}
 .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;background:#0b0e13;border:1px solid #2a3340;border-radius:7px;color:#bcd0e6;padding:8px 10px}
</style></head><body>
<header><b>bimanual-teleop</b><span class=chip style="background:#2b3550">build __BUILD__</span>
 <span id=quest class=chip>QUEST …</span>
 <span id=conn class="chip bad">stream …</span><span id=hz class=chip>— Hz</span>
 <span id=L class="chip bad">LEFT —</span><span id=R class="chip bad">RIGHT —</span>
 <span id=calib class=chip style="display:none"></span>
 <span id=wsclamp class=chip style="display:none"></span>
 <span style="flex:1"></span>
 <button class=chip style="cursor:pointer;border:0" onclick="setView(2.48,0.24)">view: behind</button>
 <button class=chip style="cursor:pointer;border:0" onclick="setView(-0.66,0.24)">view: front</button>
 <button class=chip style="cursor:pointer;border:0" onclick="setView(VIEW_DEFAULT.yaw,VIEW_DEFAULT.pitch)">reset view</button>
 <span class=chip id=age>age —</span>
</header>
<div class=ctrlbar>
 <button id=btnLive class="btn live">&#9654; START LIVE</button>
 <select id=selClutch class=sel title="always: arms follow whenever tracked. gesture: arms follow ONLY while you hold the thumb-pinky pinch (deadman) — use for finger-only work so parked means parked">
  <option value=always selected>clutch: always</option>
  <option value=gesture>clutch: gesture</option>
 </select>
 <button id=btnCalib class="btn cal">&#8853; CALIBRATE</button>
 <button id=btnCalClear class="btn ghost sm" title="clear the applied neutral-pose fit (back to 1:1)" style="display:none">clear cal</button>
 <span style="width:6px"></span>
 <button id=btnStop class="btn stop" title="graceful stop of the dashboard's render engine (saves its recording)">&#9632; STOP</button>
 <button id=btnKill class="btn kill" title="SIGINT every teleop / jog / bring-up / replay process on this host — clean torque release. NOT a substitute for the physical e-stop.">&#9888; STOP ALL</button>
 <span id=ctrlStatus class=meta style="margin-left:6px">…</span>
 <span id=hint style="color:#e8b339;font-size:13px;font-weight:600;margin-left:4px"></span>
</div>
<div class=ctrlbar>
 <span class=meta style="font-weight:700;color:#9fb2c8;letter-spacing:.3px">REPLAY</span>
 <select id=selRec class=sel></select>
 <span id=recMeta class=meta>&mdash;</span>
 <span style="width:6px"></span>
 <label class=meta>speed <input id=spd class=slider type=range min=10 max=100 value=100 step=5></label>
 <span id=spdLbl class=meta style="width:30px;display:inline-block;font-weight:700;color:#bcd0e6">1.0&times;</span>
 <label class=meta><input type=checkbox id=chkLoop checked> loop</label>
 <button id=btnReplay class="btn play sm" title="preview on the dashboard — render only, no robot">&#9654; PREVIEW</button>
 <button id=btnAnalyze class="btn ghost sm" title="grade this recording against the mapping contracts (no robot)">analyze</button>
 <span id=anaOut class=meta></span>
 <span style="flex:1"></span>
 <button id=btnMetal class="btn ghost sm" title="show the run_hw command that drives the RIGHT ARM with this recording">metal cmd &#9662;</button>
</div>
<div id=metalRow class=ctrlbar style="display:none;padding-top:0">
 <code id=metalCmd class=mono style="flex:1;white-space:nowrap;overflow:auto"></code>
 <button id=btnCopyMetal class="btn ghost sm">copy</button>
</div>
<div class=ctrlbar style="border-top:1px solid #2a3340">
 <span class=meta style="font-weight:700;color:#ff9d57;letter-spacing:.3px">HARDWARE</span>
 <span class=meta style="color:#8d97a5">drives BOTH real arms &mdash; e-stop in hand</span>
 <button id=btnHome class="btn play sm" title="Drive the wired arms to the HOME / rest pose (rate-limited) and re-anchor rest so the engage gate passes. Arms MOVE — clear of people, e-stop in hand.">&#8962; RETURN HOME</button>
 <button id=btnHwTeleop class="btn play sm" title="LIVE Quest teleop driving BOTH real arms (run_hw --vr orbit, follows CONTINUOUSLY whenever tracked). Rest-gate + runtime guard active. Arms at rest, ORBIT running, e-stop in hand.">&#9881; TELEOP LIVE</button>
 <button id=btnLog class="btn ghost sm" title="show the run_hw / engine log — why a launch died (guard trips, gate failures, errors)">&#128203; LOG</button>
 <span style="flex:1"></span>
 <button id=btnRelease class="btn kill" title="EMERGENCY: SIGINT every mover on this host so run_hw / jog release torque — the arm goes LIMP. NOT a substitute for the physical e-stop.">&#9211; RELEASE TORQUE</button>
</div>
<div id=logRow style="display:none;padding:8px 16px;background:#0b0e13;border-bottom:1px solid #232936">
 <div id=logTrip style="color:#ff9a9a;font-weight:700;font-size:13px;margin-bottom:5px;display:none"></div>
 <pre id=engLog style="margin:0;max-height:200px;overflow:auto;font-size:11px;line-height:1.35;color:#9fb2c8;white-space:pre-wrap"></pre>
</div>
<div id=hwTrip style="display:none;padding:12px 16px;background:#3a1414;border-bottom:2px solid #ff5252;align-items:center;gap:14px">
 <span style="font-size:20px">&#9888;</span>
 <div style="flex:1">
  <div id=hwTripMsg style="font-weight:800;font-size:15px;color:#ff9a9a">RUNTIME SAFETY TRIP</div>
  <div class=meta style="margin-top:3px;color:#e6b0b0">torque was released — inspect the rig, then re-launch (hw_bringup --step rest if the pose drifted).</div>
 </div>
 <button id=btnTripAck class="btn ghost sm">dismiss</button>
</div>
<div id=calBanner style="display:none;padding:12px 16px;background:#2b2410;border-bottom:2px solid #d4af37;align-items:center;gap:14px">
 <span style="font-size:20px">&#129337;</span>
 <div style="flex:1">
  <div id=calMsg style="font-weight:700;font-size:15px;color:#f4d97a"></div>
  <div style="display:flex;gap:14px;align-items:center;margin-top:6px">
   <span id=calL class=chip>LEFT &mdash;</span><span id=calR class=chip>RIGHT &mdash;</span>
   <span style="flex:1;height:10px;background:#1a1610;border-radius:5px;overflow:hidden;display:block">
    <span id=calBar style="display:block;height:100%;width:0%;background:#d4af37;transition:width .15s"></span>
   </span>
  </div>
 </div>
</div>
<main>
 <div>
  <div id=replayBanner style="display:none;margin-bottom:12px;padding:10px 14px;border-radius:10px;background:#13233a;border:1px solid #2c4a78;align-items:center;gap:12px">
   <span id=rbIcon style="font-size:18px">&#9881;</span>
   <div style="flex:1">
    <div id=rbMsg style="font-weight:800;font-size:14px;color:#9fc2f4;letter-spacing:.3px"></div>
    <span style="display:block;margin-top:6px;height:8px;background:#0b1422;border-radius:4px;overflow:hidden">
     <span id=rbBar style="display:block;height:100%;width:0%;background:#6f9fe8;transition:width .15s"></span>
    </span>
   </div>
   <span id=rbRatio class=meta style="font-weight:700;color:#bcd0e6"></span>
  </div>
  <!-- CONTROL-SYSTEM PIPELINE: operator hands -> mapped EE target -> commanded
       joints -> (tracking gap) -> measured encoder pose. Four stages L→R, each a
       3D/visual panel + compact readout, with arrows between them. -->
  <div class=pipe>
   <div class=stage>
    <div class=stagehd><span class=stagenum>1</span><div><div class=stagettl>QUEST INPUT</div><div class=stagesub>hand tracking</div></div></div>
    <div class=panel><canvas id=cvH width=460 height=360></canvas>
     <div id=roH class=readout></div></div>
   </div>
   <div class=arrow>&#10142;</div>
   <div class=stage>
    <div class=stagehd><span class=stagenum>2</span><div><div class=stagettl>MAPPING &rarr; TARGET</div><div class=stagesub>Kai&#39;s retarget</div></div></div>
    <div class=panel><canvas id=cvM width=460 height=360></canvas>
     <div id=roM class=readout></div></div>
   </div>
   <div class=arrow>&#10142;</div>
   <div class=stage>
    <div class=stagehd><span class=stagenum>3</span><div><div class=stagettl>TARGET JOINTS</div><div class=stagesub>commanded q</div></div></div>
    <div class=panel><canvas id=cvR width=460 height=360></canvas>
     <div id=roR class=readout></div></div>
   </div>
   <div class="arrow gapcol" id=arrGap><span>&#10142;</span><div id=gapBars class=gapbars></div><div class=gaplbl>TRACKING GAP</div></div>
   <div class=stage>
    <div class=stagehd><span class=stagenum>4</span><div><div class=stagettl>REAL ROBOT</div><div class=stagesub>encoders</div></div></div>
    <div class=panel id=panelHW><canvas id=cvHW width=460 height=360></canvas>
     <div id=hwPlaceholder class=placeholder>no hardware<br><span>run TELEOP LIVE to drive the metal</span></div>
     <div id=roHW class=readout></div></div>
   </div>
  </div>
  <div class=panel style="margin-top:14px"><div class=ptitle>OVERLAY <span>— your hands mapped into robot world (gold) over the robot. drag = orbit, scroll = zoom</span></div>
   <canvas id=cvO width=952 height=410></canvas></div>
 </div>
 <div>
  <div class=panel id=cardSafety style="margin-bottom:14px;display:none"></div>
  <div class=panel id=cardR style="margin-bottom:14px"></div>
  <div class=panel id=cardL></div>
 </div>
</main>
<script>
const $=id=>document.getElementById(id);
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], dotp=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const nrm=a=>{const n=Math.hypot(a[0],a[1],a[2])||1;return[a[0]/n,a[1]/n,a[2]/n]};
const LIGHT=nrm([0.4,0.3,0.85]);
function Scene(canvas,scale,ctr){
 const s={cv:canvas,cx:canvas.getContext('2d'),scale,ctr,prims:[],drag:null};
 canvas.onmousedown=e=>{s.drag=[e.clientX,e.clientY];canvas.style.cursor='grabbing'};
 window.addEventListener('mouseup',()=>{s.drag=null;canvas.style.cursor='grab'});
 window.addEventListener('mousemove',e=>{if(!s.drag)return;VIEW.yaw-=(e.clientX-s.drag[0])*0.008;
  VIEW.pitch=Math.max(-1.35,Math.min(1.35,VIEW.pitch+(e.clientY-s.drag[1])*0.006));s.drag=[e.clientX,e.clientY]});
 canvas.onwheel=e=>{e.preventDefault();s.scale*=e.deltaY<0?1.1:0.9;s.scale=Math.max(120,Math.min(1200,s.scale))};
 return s;
}
function camOf(s){const cy=Math.cos(VIEW.yaw),sy=Math.sin(VIEW.yaw),cp=Math.cos(VIEW.pitch),sp=Math.sin(VIEW.pitch);
 // right = fwd × up (RIGHT-handed camera). The old up × fwd basis mirrored every
 // panel horizontally — anatomically the robot's left arm rendered where its
 // right should be, so the operator's left hand appeared to drive the right arm.
 const fwd=[cp*cy,cp*sy,sp];const right=nrm(cross(fwd,[0,0,1]));const up=cross(right,fwd);return{right,up,fwd}}
function P(s,cam,p){const q=sub(p,s.ctr);
 return{x:s.cv.width/2+dotp(q,cam.right)*s.scale,y:s.cv.height/2-dotp(q,cam.up)*s.scale,d:dotp(q,cam.fwd)}}
const rgb=c=>`rgb(${c[0]|0},${c[1]|0},${c[2]|0})`;
const dim=(c,d)=>{const f=Math.max(0.6,Math.min(1.12,1-d*0.3));return[c[0]*f,c[1]*f,c[2]*f]};
function seg(s,cam,a,b,c,w){const A=P(s,cam,a),B=P(s,cam,b);
 s.prims.push({d:(A.d+B.d)/2,f(){const x=s.cx;x.strokeStyle=rgb(dim(c,this.d));x.lineWidth=w;x.lineCap='round';
  x.beginPath();x.moveTo(A.x,A.y);x.lineTo(B.x,B.y);x.stroke()}})}
function dot(s,cam,p,c,r){const A=P(s,cam,p);
 s.prims.push({d:A.d,f(){const x=s.cx;x.fillStyle=rgb(dim(c,this.d));x.beginPath();x.arc(A.x,A.y,r,0,7);x.fill()}})}
function ring(s,cam,p,c,r){const A=P(s,cam,p);
 s.prims.push({d:A.d-0.001,f(){const x=s.cx;x.strokeStyle=rgb(c);x.lineWidth=2.4;x.beginPath();x.arc(A.x,A.y,r,0,7);x.stroke()}})}
function tri(s,cam,p0,p1,p2,base,alpha){
 const n=nrm(cross(sub(p1,p0),sub(p2,p0)));
 const lam=Math.abs(dotp(n,LIGHT))*0.65+0.35;
 const A=P(s,cam,p0),B=P(s,cam,p1),C=P(s,cam,p2);
 const col=[base[0]*lam,base[1]*lam,base[2]*lam];
 s.prims.push({d:(A.d+B.d+C.d)/3,f(){const x=s.cx;x.fillStyle=rgb(dim(col,this.d));if(alpha!=null)x.globalAlpha=alpha;
  x.beginPath();x.moveTo(A.x,A.y);x.lineTo(B.x,B.y);x.lineTo(C.x,C.y);x.closePath();x.fill();if(alpha!=null)x.globalAlpha=1}})}
function flush(s){s.prims.sort((a,b)=>b.d-a.d);for(const p of s.prims)p.f();s.prims=[]}
function clearCv(s){s.cx.clearRect(0,0,s.cv.width,s.cv.height)}
function grid(s,cam,z){for(let i=-4;i<=4;i++){seg(s,cam,[-1.1,i*0.25,z],[0.7,i*0.25,z],[32,38,49],1);
 seg(s,cam,[i*0.25-0.2,-1.05,z],[i*0.25-0.2,1.05,z],[32,38,49],1)}
 seg(s,cam,[-0.55,0,z],[-1.0,0,z],[90,200,140],3);                       // FRONT arrow (-X)
 {const A=P(s,cam,[-1.06,0,z]);
  s.prims.push({d:A.d,f(){s.cx.font='bold 12px sans-serif';s.cx.fillStyle='#5ac88c';s.cx.fillText('FRONT',A.x-18,A.y)}});}}
// ---- data ----
const FINGERS=[[[0,1,2,3,4],[212,105,158]],[[0,5,6,7,8,9],[74,144,217]],[[0,10,11,12,13,14],[42,161,152]],
 [[0,15,16,17,18,19],[202,165,32]],[[0,20,21,22,23,24],[166,108,201]]];
const ARM_RGB={left:[158,189,237],right:[235,143,90]},GOLD=[212,175,55];
const TRIAD=[[224,72,72],[60,190,80],[80,110,250]];
function quat2cols(q){const[w,x,y,z]=q;return[[1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)],
 [2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)],[2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)]]}
let MESH=null,RIG=null;
// ONE camera shared by all panels. Default = BEHIND the robot (over-the-shoulder
// embodiment): your left hand drives the arm on the LEFT of the screen.
const VIEW={yaw:2.48,pitch:0.24};
const VIEW_DEFAULT={yaw:2.48,pitch:0.24};
function setView(yaw,pitch){VIEW.yaw=yaw;VIEW.pitch=pitch;}
// All panels share VIEW (the GIF camera by default). Hands are drawn in the SAME
// world axes convention as the robot, so the three panels can never disagree.
const scH=Scene($('cvH'),430,[-0.28,0,0.10]);
const scM=Scene($('cvM'),300,[-0.1,-0.05,0.82]);    // stage 2: mapped EE target over the robot
const scR=Scene($('cvR'),300,[-0.1,-0.05,0.82]);    // stage 3: robot at COMMANDED joints
const scHW=Scene($('cvHW'),300,[-0.1,-0.05,0.82]);  // stage 4: robot at MEASURED encoder pose
const scO=Scene($('cvO'),330,[-0.12,-0.05,0.85]);
function meshInto(s,cam,T,verts,base,alpha){
 const R=[[T[0],T[1],T[2]],[T[4],T[5],T[6]],[T[8],T[9],T[10]]],t=[T[3],T[7],T[11]];
 const X=v=>[R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2]+t[0],
             R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2]+t[1],
             R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]+t[2]];
 for(let i=0;i<verts.length;i+=9)
  tri(s,cam,X([verts[i],verts[i+1],verts[i+2]]),X([verts[i+3],verts[i+4],verts[i+5]]),
      X([verts[i+6],verts[i+7],verts[i+8]]),base,alpha);
}
const EYE16=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
function robotInto(s,cam,st,meshT,alpha,handMesh,handT){
 let bases=[];
 if(MESH&&MESH.stand)for(const g of MESH.stand)meshInto(s,cam,EYE16,g,[88,98,112],alpha);
 for(const side of['left','right']){
  const a=st.arms[side];if(!a)continue;
  if(a.link_pos){const b=a.link_pos.slice(0,3);bases.push(b);
   if(!MESH||!MESH.stand)seg(s,cam,[b[0],b[1],0],b,[40,47,60],9)}
  if(MESH&&meshT&&meshT[side]){const Ts=meshT[side],gs=MESH[side];
   for(let g=0;g<gs.length&&g<Ts.length;g++)meshInto(s,cam,Ts[g],gs[g],ARM_RGB[side],alpha);}
  else if(a.link_pos){const Pn=[];for(let i=0;i<a.link_pos.length;i+=3)Pn.push(a.link_pos.slice(i,i+3));
   for(let i=1;i<Pn.length;i++)seg(s,cam,Pn[i-1],Pn[i],ARM_RGB[side],i<4?9:6.5)}
  if(a.ee_pos){dot(s,cam,a.ee_pos,[255,255,255],3.6);
   if(a.ee_quat){const C=quat2cols(a.ee_quat),L=0.09;
    for(let k=0;k<3;k++)seg(s,cam,a.ee_pos,[a.ee_pos[0]+C[k][0]*L,a.ee_pos[1]+C[k][1]*L,a.ee_pos[2]+C[k][2]*L],TRIAD[k],2.4)}}
  if(a.cmd_pos){ring(s,cam,a.cmd_pos,[65,217,141],7);if(a.ee_pos)seg(s,cam,a.ee_pos,a.cmd_pos,[65,217,141],1.4)}}
 if(handT)for(const side of['left','right']){
  const hg=MESH?MESH['hand_'+side]:null,Ts=handT[side];
  if(hg&&Ts)for(let g=0;g<hg.length&&g<Ts.length;g++)meshInto(s,cam,Ts[g],hg[g].v,hg[g].c,alpha);}
 else if(handMesh)for(const side of['left','right'])
  if(handMesh[side])meshInto(s,cam,EYE16,handMesh[side],[205,192,172],alpha);
 if(bases.length===2&&(!MESH||!MESH.stand))seg(s,cam,bases[0],bases[1],[40,47,60],9);
 return bases;
}
function handWorld(an,w){return[an[0]-w[2],an[1]+w[0],an[2]+w[1]]}      // body [r,u,f] -> world
function drawHands(st){
 clearCv(scH);const cam=camOf(scH);
 grid(scH,cam,-0.45);
 dot(scH,cam,[0,0,0],GOLD,5);                                            // torso proxy
 for(const side of['left','right']){
  const h=st.op&&st.op.hands?st.op.hands[side]:null;
  if(!h||!h.tracked||!h.wrist_body)continue;
  const w=h.wrist_body,o=[-w[2],w[0],w[1]];                               // body [r,u,f] -> world [-f,r,u]
  seg(scH,cam,[0,0,0],o,[120,104,40],1.6);
  {const A=P(scH,cam,[o[0],o[1],o[2]+0.10]);
   scH.prims.push({d:A.d-0.01,f(){scH.cx.font='bold 17px sans-serif';
    scH.cx.fillStyle=side==='left'?'#6f9fe8':'#e8854a';scH.cx.fillText(side==='left'?'L':'R',A.x-5,A.y)}});}
  if(h.lm_body){const L=[];for(let i=0;i<75;i+=3)
    L.push([-(w[2]+h.lm_body[i+2]),w[0]+h.lm_body[i],w[1]+h.lm_body[i+1]]);
   for(const[chain,c]of FINGERS){for(let k=1;k<chain.length;k++)seg(scH,cam,L[chain[k-1]],L[chain[k]],c,2.6);
    for(const j of chain)dot(scH,cam,L[j],c,2.2)}}
  else dot(scH,cam,o,GOLD,5);
 }
 flush(scH);
}
function drawRobot(st,meshT,hm,hT){clearCv(scR);const cam=camOf(scR);grid(scR,cam,0);robotInto(scR,cam,st,meshT,null,hm,hT);flush(scR)}
// Stage 2 — MAPPING → ROBOT TARGET: faint robot ghost + the commanded EE target
// (arms.{side}.cmd_pos/cmd_quat, world frame) drawn as a bold marker/triad. This
// is the operator→robot position-mapping output the IK is asked to reach.
function drawTarget(st,meshT){
 clearCv(scM);const cam=camOf(scM);grid(scM,cam,0);
 if(MESH&&MESH.stand)for(const g of MESH.stand)meshInto(scM,cam,EYE16,g,[60,67,80],0.5);
 for(const side of['left','right']){
  const a=st.arms?st.arms[side]:null;if(!a)continue;
  if(MESH&&meshT&&meshT[side]){const Ts=meshT[side],gs=MESH[side];     // faint ghost of where it is now
   for(let g=0;g<gs.length&&g<Ts.length;g++)meshInto(scM,cam,Ts[g],gs[g],ARM_RGB[side],0.28);}
  else if(a.link_pos){const Pn=[];for(let i=0;i<a.link_pos.length;i+=3)Pn.push(a.link_pos.slice(i,i+3));
   for(let i=1;i<Pn.length;i++)seg(scM,cam,Pn[i-1],Pn[i],dim(ARM_RGB[side],1.3),i<4?7:5)}
  if(a.cmd_pos){                                                       // the COMMANDED target
   ring(scM,cam,a.cmd_pos,[65,217,141],9);dot(scM,cam,a.cmd_pos,[65,217,141],3.2);
   if(a.cmd_quat){const C=quat2cols(a.cmd_quat),L=0.10;
    for(let k=0;k<3;k++)seg(scM,cam,a.cmd_pos,[a.cmd_pos[0]+C[k][0]*L,a.cmd_pos[1]+C[k][1]*L,a.cmd_pos[2]+C[k][2]*L],TRIAD[k],2.6)}
   if(a.ee_pos)seg(scM,cam,a.ee_pos,a.cmd_pos,[65,217,141],1.6);}       // achieved → target tie-line
  if(a.ee_pos)dot(scM,cam,a.ee_pos,[200,200,200],3);
 }
 flush(scM);
}
// Stage 4 — REAL ROBOT (encoders): the SAME YAM meshes FK'd at the MEASURED pose
// (status.hw.arms.{side}.measured → /state mesh_T_hw). Muted colours so it reads
// as "metal", and NO commanded ring (that lives in stage 2/3) — this is ground
// truth from the encoders, what the arm IS, not what it was told to do.
const HW_RGB={left:[120,150,196],right:[196,128,84]};
function drawRobotHW(st,meshHW){
 clearCv(scHW);const cam=camOf(scHW);grid(scHW,cam,0);
 if(MESH&&MESH.stand)for(const g of MESH.stand)meshInto(scHW,cam,EYE16,g,[78,88,100],null);
 for(const side of['left','right']){
  const Ts=meshHW?meshHW[side]:null,gs=MESH?MESH[side]:null;
  if(MESH&&Ts&&gs)for(let g=0;g<gs.length&&g<Ts.length;g++)meshInto(scHW,cam,Ts[g],gs[g],HW_RGB[side],null);
 }
 flush(scHW);
}
function drawOverlay(st,meshT,hm,hT){
 clearCv(scO);const cam=camOf(scO);grid(scO,cam,0);
 const bases=robotInto(scO,cam,st,meshT,0.85,hm,hT);
 if(bases.length===2&&st.op&&st.op.hands){
  const an=[(bases[0][0]+bases[1][0])/2-0.20,(bases[0][1]+bases[1][1])/2,(bases[0][2]+bases[1][2])/2-0.15];  // matches mapping.body_anchor_forward/drop
  dot(scO,cam,an,GOLD,5);
  for(const side of['left','right']){
   const h=st.op.hands[side];if(!h||!h.tracked||!h.wrist_body)continue;
   const w=h.wrist_body,o=handWorld(an,w);
   seg(scO,cam,an,o,[150,128,45],1.6);
   {const A=P(scO,cam,[o[0],o[1],o[2]+0.12]);
    scO.prims.push({d:A.d-0.01,f(){scO.cx.font='bold 17px sans-serif';
     scO.cx.fillStyle=side==='left'?'#6f9fe8':'#e8854a';scO.cx.fillText(side==='left'?'L':'R',A.x-5,A.y)}});}
   if(h.lm_body){const L=[];for(let i=0;i<75;i+=3)
     L.push(handWorld(an,[w[0]+h.lm_body[i],w[1]+h.lm_body[i+1],w[2]+h.lm_body[i+2]]));
    for(const[chain,c]of FINGERS){for(let k=1;k<chain.length;k++)seg(scO,cam,L[chain[k-1]],L[chain[k]],c,2.4);
     for(const j of chain)dot(scO,cam,L[j],c,2)}}
   else dot(scO,cam,o,GOLD,4.5);
  }}
 flush(scO);
}
// ---- pipeline compact readouts + gap/convergence indicators ----
function ro(rows){return rows.filter(Boolean).map(([k,v,c])=>
 '<div class=row><span>'+k+'</span><b'+(c?' style="color:'+c+'"':'')+'>'+v+'</b></div>').join('')}
function updPipeline(st,d){
 const arms=st.arms||{};
 // Stage 1 — QUEST INPUT: per-side tracked + torso→wrist vector magnitude.
 const oh=(st.op&&st.op.hands)||{};
 $('roH').innerHTML=ro([['left',(oh.left&&oh.left.tracked)?'tracked':'—',(oh.left&&oh.left.tracked)?'#6f9fe8':'#76808d'],
  ['right',(oh.right&&oh.right.tracked)?'tracked':'—',(oh.right&&oh.right.tracked)?'#e8854a':'#76808d']]);
 // Stage 2 — MAPPING → ROBOT TARGET: commanded EE + workspace-clamp warning.
 const m2=[];
 for(const side of['left','right']){const a=arms[side];if(!a)continue;
  if(a.cmd_pos)m2.push(['cmd '+side[0].toUpperCase(),a.cmd_pos.map(v=>v.toFixed(2)).join(', ')+' m']);
  if(a.clamp_dist!=null&&a.clamp_dist>0.005)
   m2.push(['WS CLAMP '+side[0].toUpperCase(),(a.clamp_dist*100).toFixed(1)+' cm',a.clamp_dist>0.02?'#ff8a8a':'#e8b339']);}
 $('roM').innerHTML=m2.length?ro(m2):'<div class=row><span>commanded EE</span><b>—</b></div>';
 // Stage 3 — TARGET ROBOT JOINTS: commanded q (deg), worst limit margin.
 const m3=[];
 for(const side of['left','right']){const a=arms[side];if(!a||!a.q)continue;
  const worst=a.margins?Math.min(...a.margins):1;
  m3.push(['q '+side[0].toUpperCase()+' (j1)',(a.q[0]*57.2958).toFixed(0)+'°',worst<0.12?'#ff8a8a':null]);}
 $('roR').innerHTML=m3.length?ro(m3):'<div class=row><span>commanded q</span><b>—</b></div>';
 // Stage 4 — REAL ROBOT + the TRACKING GAP between commanded & measured.
 updGapAndHW(st,d);
}
// Between stage 3 and 4: per-joint convergence error (gap / gap_max) as small bars,
// and the stage-4 readout / placeholder, all from status.hw (DEFENSIVE: absent on
// sim → muted placeholder, no throw).
// Per-joint tracking ceiling (rig safety.runtime.max_tracking_error): hardware
// telemetry sends limits.track as a PER-JOINT list ([0.6,0.6,0.6,1.4,3.0,3.0]) —
// dividing a scalar gap by the whole array yields NaN and silently kills the
// warning colours, so always pick the ceiling for THAT joint (0-based j).
function trackOf(lim,j){const t=lim&&lim.track;return Array.isArray(t)?(t[j]!=null?t[j]:0.35):(t!=null?t:0.35);}
function updGapAndHW(st,d){
 const hw=(st.status&&st.status.hw)||null;
 const ph=$('hwPlaceholder'),roHW=$('roHW'),gapWrap=$('arrGap'),gb=$('gapBars');
 const haveHW=!!(d&&d.mesh_T_hw&&(d.mesh_T_hw.left||d.mesh_T_hw.right));
 ph.style.display=haveHW?'none':'';
 if(!hw){gapWrap.style.opacity=0.3;gb.innerHTML='';roHW.innerHTML='';return}
 gapWrap.style.opacity=1;
 // gap bars: prefer the RIGHT arm (the wired/driven side); fall back to left.
 const arms=hw.arms||{},side=arms.right?'right':(arms.left?'left':null),a=side?arms[side]:null;
 const lim=hw.limits||{},wf=lim.warn_frac||0.8;
 if(a&&a.gap){let bars='';for(let i=0;i<a.gap.length;i++){
   const r=Math.max(0,Math.min(1.25,a.gap[i]/trackOf(lim,i))),hpx=Math.max(2,r*34);
   const col=r>1?'#ff8a8a':r>wf?'#e8b339':'#41d98d';
   bars+='<span class=gb title="j'+(i+1)+' '+(a.gap[i]*57.2958).toFixed(1)+'°" style="height:'+hpx.toFixed(0)+'px;background:'+col+'"></span>';}
  gb.innerHTML=bars;}
 else gb.innerHTML='';
 const m4=[];
 if(a&&a.gap_max!=null){const p=100*a.gap_max/trackOf(lim,(a.gap_joint||1)-1);
  m4.push(['gap (worst)',(a.gap_max*57.2958).toFixed(1)+'° j'+(a.gap_joint||'?'),p>100?'#ff8a8a':p>100*wf?'#e8b339':'#41d98d']);}
 if(a&&a.temp_max!=null)m4.push(['temp',a.temp_max.toFixed(0)+'°C']);
 if(!haveHW)m4.push(['encoders','no hardware','#76808d']);
 roHW.innerHTML=m4.length?ro(m4):'';
}
// CONVERGENCE / PLAYBACK banner — driven by the OPTIONAL status.hw.replay field
// (additive; sim runs without it never show the banner). Contract consumed:
//   replay = {phase:"converge"|"follow", scale:0..1, ratio, converged, timed_out}
// converge = arm auto-calibrating to the tape's start pose; follow = tracking the
// tape at `scale` of full speed. Every field read defensively.
function updReplayBanner(st){
 const hw=(st.status&&st.status.hw)||null;
 const rp=hw&&hw.replay?hw.replay:null;
 const bn=$('replayBanner');
 if(!rp){bn.style.display='none';return}
 bn.style.display='flex';
 const phase=rp.phase||'',scale=typeof rp.scale==='number'?rp.scale:0;
 const ratio=typeof rp.ratio==='number'?rp.ratio:(typeof rp.scale==='number'?rp.scale:0);
 const conv=phase==='converge';
 $('rbIcon').innerHTML=conv?'&#129517;':(rp.converged?'&#9654;':'&#9881;');
 $('rbMsg').textContent=conv
  ?('AUTO-CALIBRATING TO START'+(rp.timed_out?' — TIMED OUT (holding)':''))
  :('FOLLOWING x'+Math.round(scale*100)+'%'+(rp.converged?'':' (settling)'));
 $('rbMsg').style.color=rp.timed_out?'#ff9a9a':conv?'#f4d97a':'#9fc2f4';
 const pct=Math.max(0,Math.min(100,(conv?ratio:scale)*100));
 $('rbBar').style.width=pct.toFixed(0)+'%';
 $('rbBar').style.background=rp.timed_out?'#ff8a8a':conv?'#d4af37':'#6f9fe8';
 $('rbRatio').textContent=conv?(Math.round(pct)+'% to start'):('scale '+scale.toFixed(2));
}
function gaugeRow(i,q,lo,hi,margin){
 const span=hi-lo||1,pos=Math.max(0,Math.min(1,(q-lo)/span));
 const alert=margin<0.12,warnZ=0.25/span*100;
 return `<div class=gauge><span style="color:var(--dim)">j${i+1}</span>
  <span class=bar><span class=zone style="left:0;width:${warnZ}%"></span><span class=zone style="right:0;width:${warnZ}%"></span>
  <span class="tick ${alert?'alert':''}" style="left:calc(${(pos*100).toFixed(1)}% - 1px)"></span></span>
  <span style="text-align:right;${alert?'color:#ff8a8a':''}">${(q*57.2958).toFixed(1)}&deg;</span></div>`}
function card(side,s){
 const a=s.arms[side];if(!a)return'';
 const lim=RIG?RIG[side]:null;
 let h=`<div class=armttl><h3 style="color:${side==='left'?'var(--blue)':'var(--orange)'}">${side.toUpperCase()} ARM</h3>`;
 const st=s.status;h+=`<span style="font-size:12px;color:${st.engaged[side]?'var(--green)':'var(--dim)'}">${st.engaged[side]?'ENGAGED':'idle'}${st.tracked[side]?'':' · NOT TRACKED'}</span></div>`;
 for(let i=0;i<6;i++){const lo=lim?lim.lo[i]:-3.14,hi=lim?lim.hi[i]:3.14;h+=gaugeRow(i,a.q[i],lo,hi,a.margins?a.margins[i]:1)}
 const op=s.op&&s.op.hands?s.op.hands[side]:null;
 if(op&&op.wrist_body){const w=op.wrist_body;h+=`<div class=kv><span>your hand (torso→wrist)</span><b>R ${w[0].toFixed(2)} · U ${w[1].toFixed(2)} · F ${w[2].toFixed(2)} m</b></div>`}
 if(a.ee_pos)h+=`<div class=kv><span>EE world [x,y,z]</span><b>${a.ee_pos.map(v=>v.toFixed(2)).join(', ')} m</b></div>`;
 const wp=a.wrist_pos||a.ee_pos;
 if(a.cmd_pos&&wp){const e=Math.hypot(...[0,1,2].map(k=>a.cmd_pos[k]-wp[k]));
  h+=`<div class=kv><span>wrist target gap</span><b class="${e<0.05?'err-ok':'err-bad'}">${(e*100).toFixed(1)} cm</b></div>`}
 if(a.clamp_dist!=null&&a.clamp_dist>0.005)
  h+=`<div class=kv><span>target outside workspace</span><b class="${a.clamp_dist>0.02?'err-bad':'err-ok'}">${(a.clamp_dist*100).toFixed(1)} cm</b></div>`;
 return h}
function chip(id,cls,txt){const e=$(id);e.className='chip '+cls;e.textContent=txt}
async function control(params){try{const r=await fetch('/control?'+new URLSearchParams(params));updCtrl(await r.json())}catch(e){}}
let CTRL=null;
const speed=()=>$('spd').value/100;
function metalCmd(){
 const f=$('selRec').value; if(!f){$('metalCmd').textContent='— pick a recording —';return}
 const sp=speed(), s=sp<1?` --speed ${sp.toFixed(2)}`:'';
 $('metalCmd').textContent=`python -m bimanual_teleop.launch.run_hw --vr replay ${f} --clutch recorded${s} --rate-limit 0.5`;
}
async function refreshRec(){
 const f=$('selRec').value, m=$('recMeta');
 metalCmd();
 if(!f){m.textContent='—';return}
 try{const r=await(await fetch('/recinfo?file='+encodeURIComponent(f))).json();
  m.textContent=r.error?('⚠ '+r.error)
   :`${r.dur.toFixed(1)}s · ${r.frames}f · engaged ${(r.engaged*100).toFixed(0)}% · R ${(r.right*100).toFixed(0)}%`;
  m.style.color=r.error?'#e8b339':'#8d97a5';
 }catch(e){m.textContent=''}
}
function updCtrl(c){
 if(!c)return;
 CTRL=c;
 const el=$('ctrlStatus');
 el.textContent = c.error ? `⚠ ${c.error}`
   : c.running
   ? `● ${c.mode}${c.record?' — recording '+c.record:''}${c.uptime?'  ('+Math.floor(c.uptime/60)+':'+String(Math.floor(c.uptime%60)).padStart(2,'0')+')':''}`
   : `○ ${c.msg||'stopped'}`;
 el.style.color = c.error ? '#e8b339' : c.running ? '#41d98d' : '#9fb2c8';
 const sel=$('selRec');
 if(c.recordings && sel.options.length !== c.recordings.length){
  const cur=sel.value; sel.innerHTML='';
  for(const r of c.recordings){const o=document.createElement('option');o.value=r;o.textContent=r.split('/').pop();sel.appendChild(o)}
  sel.value = (cur && c.recordings.includes(cur)) ? cur : (c.recordings[0]||'');
  refreshRec();
 }
}
$('btnLive').onclick=()=>control({action:'start_live',clutch:$('selClutch').value});
$('btnStop').onclick=()=>control({action:'stop'});
$('btnKill').onclick=()=>control({action:'kill_all'});
$('btnReplay').onclick=()=>{const f=$('selRec').value;
 if(f)control({action:'start_replay',file:f,loop:$('chkLoop').checked?'1':'0',speed:speed().toFixed(2)})};
$('selRec').onchange=refreshRec;
$('spd').oninput=()=>{$('spdLbl').innerHTML=speed().toFixed(1)+'&times;';metalCmd()};
$('btnMetal').onclick=()=>{const r=$('metalRow');r.style.display=r.style.display==='none'?'flex':'none';metalCmd()};
$('btnCopyMetal').onclick=()=>{navigator.clipboard.writeText($('metalCmd').textContent);
 $('btnCopyMetal').textContent='copied';setTimeout(()=>$('btnCopyMetal').textContent='copy',1200)};
$('btnAnalyze').onclick=async()=>{const f=$('selRec').value;if(!f)return;
 const o=$('anaOut');o.textContent='analyzing…';o.style.color='#8d97a5';
 try{const r=await(await fetch('/analyze?file='+encodeURIComponent(f))).json();
  o.textContent=r.error?('⚠ '+r.error):r.verdict;
  o.style.color=r.error?'#e8b339':(r.ok?'#41d98d':'#ff8a8a');o.title=r.detail||'';
 }catch(e){o.textContent='⚠ analyze failed'}};
let CAL_ACTIVE=false;
let WSEMA={left:0,right:0};
$('btnCalib').onclick=()=>control({action:CAL_ACTIVE?'calibrate_cancel':'calibrate'});
$('btnCalClear').onclick=()=>control({action:'calibrate_clear'});
let HWTRIP_ACK='';
$('btnHwTeleop').onclick=()=>{
 if(!confirm('Start LIVE Quest teleop on BOTH real arms?\\n\\nThe arms follow your hands CONTINUOUSLY whenever tracked (no pinch). Both arms must be at rest and ORBIT running on the Quest. Keep a hand on the e-stop. Rest-gate + runtime guard + hard speed ceiling are active.'))return;
 control({action:'start_hw_teleop'})};
$('btnRelease').onclick=()=>{control({action:'release_torque'})};
$('btnHome').onclick=()=>{
 if(!confirm(`⌂ RETURN HOME?\n\nThe wired arms will MOVE (rate-limited) to the home/rest pose, then re-anchor rest so the next robot run can engage.\n\n• Area clear of people\n• Hand on the e-stop\n\nUse STOP ALL / RELEASE TORQUE to release torque.`)) return;
 const b=$('btnHome');b.textContent='homing…';b.disabled=true;
 control({action:'return_home'}).then(()=>{b.textContent='⌂ RETURN HOME';b.disabled=false})};
let LOGOPEN=false;
async function fetchLog(){
 try{const r=await(await fetch('/enginelog')).json();
  $('engLog').textContent=(r.lines||[]).join('\\n');
  const t=$('logTrip');
  if(r.trip){t.style.display='';t.textContent='⚠ '+r.trip}else t.style.display='none';
  $('engLog').scrollTop=$('engLog').scrollHeight;
  return r.trip||'';
 }catch(e){return''}
}
$('btnLog').onclick=()=>{LOGOPEN=!LOGOPEN;$('logRow').style.display=LOGOPEN?'block':'none';if(LOGOPEN)fetchLog()};
$('btnTripAck').onclick=()=>{const hw=LASTHW;HWTRIP_ACK=hw&&hw.trip?(hw.trip.kind+hw.trip.detail):'ack';$('hwTrip').style.display='none'};
let LASTHW=null;
function bar01(pct,col){return '<div class="bar" style="margin:3px 0 7px"><span class=tick style="left:'+Math.min(100,pct).toFixed(0)+'%;background:'+col+';width:4px"></span></div>'}
function updSafety(st){
 const hw=st&&st.status?st.status.hw:null; LASTHW=hw;
 const cd=$('cardSafety'), tb=$('hwTrip');
 if(!hw){cd.style.display='none';tb.style.display='none';return}
 cd.style.display='';
 const trip=hw.trip, lim=hw.limits||{}, wf=lim.warn_frac||0.8;
 const tripKey=trip?(trip.kind+trip.detail):'';
 if(trip&&tripKey!==HWTRIP_ACK){tb.style.display='flex';
  $('hwTripMsg').textContent='⚠ RUNTIME SAFETY TRIP ['+trip.kind+'] — '+trip.detail}
 else tb.style.display='none';
 let h='<div class=ptitle>SAFETY <span>— runtime guard '+(hw.enabled?'<b style="color:#41d98d">ON</b>':'<b style="color:#ff8a8a">OFF</b>')
  +(trip?' · <b style="color:#ff8a8a">TRIPPED: '+trip.kind+'</b>':'')+'</span></div>';
 h+='<div class=kv><span>hard speed ceiling</span><b>'+(hw.hard_max_joint_speed!=null?hw.hard_max_joint_speed+' rad/s':'—')+'</b></div>';
 h+='<div class=kv><span>shaper rate (clamped)</span><b>'+(hw.eff_rate!=null?hw.eff_rate+' rad/s':'—')+'</b></div>';
 for(const side of(hw.sides||[])){const a=(hw.arms||{})[side];if(!a)continue;
  h+='<div style="margin-top:9px;font-weight:700;color:#9fb2c8;font-size:12px">'+side.toUpperCase()+' arm</div>';
  if(a.gap_max!=null){const p=100*a.gap_max/trackOf(lim,(a.gap_joint||1)-1);
   h+='<div class=kv><span>tracking gap</span><b style="color:'+(p>100?'#ff8a8a':p>100*wf?'#e8b339':'#41d98d')+'">'+(a.gap_max*57.2958).toFixed(1)+'° (j'+a.gap_joint+')</b></div>'+bar01(p,p>100*wf?'#ff8a8a':'#41d98d')}
  if(a.temp_max!=null){const p=100*a.temp_max/(lim.temp||75);
   h+='<div class=kv><span>motor temp</span><b style="color:'+(p>100?'#ff8a8a':p>100*wf?'#e8b339':'#9fb2c8')+'">'+a.temp_max.toFixed(0)+'°C</b></div>'+bar01(p,p>100*wf?'#ff8a8a':'#e8b339')}
  if(a.effort_max!=null){const p=100*a.effort_max/(lim.current||8);
   h+='<div class=kv><span>motor effort</span><b style="color:'+(p>100*wf?'#ff8a8a':'#9fb2c8')+'">'+a.effort_max.toFixed(1)+'</b></div>'+bar01(p,p>80?'#ff8a8a':'#6f9fe8')}
 }
 cd.innerHTML=h;
}
function updCalib(st){
 const c=st&&st.status?st.status.calib:null, applied=st&&st.status?st.status.calib_applied:null;
 const locked=!!(st&&st.status&&st.status.follow_locked);
 const g=st&&st.status?st.status.guard:null, tripped=!!(g&&g.tripped);
 CAL_ACTIVE=!!(c&&c.active);
 const btn=$('btnCalib');
 btn.textContent=CAL_ACTIVE?'✕ CANCEL CAL':'⊕ CALIBRATE';
 btn.style.background=CAL_ACTIVE?'#7c2d2d':'#8a6d1a';
 const bn=$('calBanner');
 if(c&&c.active){
  bn.style.display='flex';
  $('calMsg').textContent=c.msg||'';
  $('calMsg').style.color='';
  $('calBar').style.width=((c.progress||0)*100).toFixed(0)+'%';
  for(const[side,id]of[['left','calL'],['right','calR']])
   chip(id,c[side]?'ok':'bad',(side==='left'?'LEFT ':'RIGHT ')+(c[side]?'✓ in view':'not tracked'));
 }else if(locked){
  bn.style.display='flex';
  $('calMsg').textContent=tripped
   ?'⚠ '+((c&&c.phase==='tripped'&&c.msg)||('TRACKING JUMPED — '+((g&&g.reason)||'anchor changed')+'. Recalibrate to resume.'))
   :'ARMS LOCKED — press ⊕ CALIBRATE and follow the 3 poses to enable control (required each session)';
  $('calMsg').style.color=tripped?'#ff8a8a':'';
  $('calBar').style.width='0%';
  chip('calL','warn','LEFT —');chip('calR','warn','RIGHT —');
 }else bn.style.display='none';
 // header chip: transient msgs (done/cancelled fade engine-side) or the applied fit
 const hc=$('calib');
 if(c&&c.phase==='tripped'&&!c.active){hc.style.display='';hc.className='chip bad';hc.textContent='⚠ TRACKING TRIP'+(g&&g.trips>1?' ×'+g.trips:'')}
 else if(c&&c.msg&&!c.active){hc.style.display='';hc.className='chip '+(c.phase==='done'?'ok':'warn');hc.textContent=c.msg}
 else if(applied&&applied.axis_scale){
  const q=applied.quality||null,gr=q?q.grade:null;
  hc.style.display='';hc.className='chip '+(gr==='bad'?'bad':gr==='check'?'warn':'ok');
  hc.title='body offset [r,u,f]: '+JSON.stringify(applied.body_offset)
    +(q&&q.reasons&&q.reasons.length?' · fit: '+q.reasons.join('; '):'');
  hc.textContent='CAL '+(gr==='bad'?'✗':gr==='check'?'⚠':'✓')+' lat ×'+applied.axis_scale[0].toFixed(2)
    +' / reach ×'+applied.axis_scale[2].toFixed(2)+(q&&q.worst_cm!=null?' · ±'+q.worst_cm+'cm':'')}
 else if(c&&c.msg){hc.style.display='';hc.className='chip warn';hc.textContent='calib: '+c.msg}
 else hc.style.display='none';
 $('btnCalClear').style.display=(applied&&!CAL_ACTIVE)?'':'none';
}
function hint(d){
 // First broken link in the chain wins: USB -> engine -> stream -> tracking.
 const live=!CTRL||!CTRL.running||(CTRL.mode||'').startsWith('LIVE');
 if(live&&d.quest==='unauthorized')return"→ put the headset ON and tap 'Allow USB debugging'";
 if(live&&d.quest==='disconnected')return'→ plug the Quest USB cable in';
 if(CTRL&&!CTRL.running)return'→ press START LIVE (Quest)';
 if(!d.connected)return CTRL&&CTRL.running?'→ engine starting…':'';
 const tr=d.state&&d.state.status&&d.state.status.tracked;
 if(live&&tr&&!tr.left&&!tr.right)return'→ open the ORBIT app on the Quest, WEAR it, controllers asleep, hands in view';
 return''}
let ctrlN=0;
async function tick(){
 try{
  if(!RIG){try{RIG=await(await fetch('/rig')).json()}catch(e){}}
  if(!MESH){try{MESH=await(await fetch('/meshes')).json();if(!MESH.left)MESH=null}catch(e){}}
  const d=await(await fetch('/state')).json();
  if(d.quest)chip('quest',d.quest==='device'?'ok':(d.quest==='no-adb'||d.quest==='checking'?'warn':'bad'),
   d.quest==='device'?'QUEST USB':d.quest==='unauthorized'?'QUEST UNAUTHORIZED':
   d.quest==='no-adb'?'adb missing':d.quest==='checking'?'QUEST …':'QUEST DISCONNECTED');
  chip('conn',d.connected?'ok':'bad',d.connected?'stream connected':'STREAM OFFLINE');
  chip('age',d.age!=null&&d.age<0.3?'ok':'warn','age '+(d.age==null?'—':d.age.toFixed(2)+'s'));
  $('hint').textContent=hint(d);
  const s=d.state;
  if(s&&s.status&&d.connected){
   chip('hz',s.status.hz>30?'ok':'warn',(s.status.hz||0).toFixed(0)+' Hz');
   for(const[side,id]of[['left','L'],['right','R']]){
    const tr=s.status.tracked[side],en=s.status.engaged[side];
    chip(id,tr?(en?'ok':'warn'):'bad',`${id==='L'?'LEFT':'RIGHT'} ${tr?(en?'tracked + engaged':'tracked'):'NO TRACKING'}`)}
   // sustained workspace clamping = the mapping/calibration is off (EMA ~1s)
   for(const side of['left','right']){const a=s.arms?s.arms[side]:null;
    WSEMA[side]=0.95*WSEMA[side]+0.05*((a&&a.clamp_dist>0.02)?1:0)}
   const wsworst=Math.max(WSEMA.left,WSEMA.right),wc=$('wsclamp');
   if(wsworst>0.5){wc.style.display='';wc.className='chip bad';
    wc.textContent='WS CLAMP '+(WSEMA.left>0.5?'L':'')+(WSEMA.right>0.5?'R':'')+' — mapping off? recalibrate'}
   else wc.style.display='none';
   updCalib(s);
   if(ctrlN%6===0)updSafety(s);          // telemetry panel: 8 Hz, not every frame
   drawHands(s);                                   // stage 1: Quest hands
   drawTarget(s,d.mesh_T);                          // stage 2: mapped EE target over robot
   drawRobot(s,d.mesh_T,d.hand_mesh,d.hand_T);      // stage 3: robot at COMMANDED joints
   drawRobotHW(s,d.mesh_T_hw);                      // stage 4: robot at MEASURED encoder pose
   updPipeline(s,d);                                // per-stage readouts + tracking-gap bars
   updReplayBanner(s);                              // convergence/playback banner (optional hw.replay)
   drawOverlay(s,d.mesh_T,d.hand_mesh,d.hand_T);
   $('cardL').innerHTML=card('left',s);$('cardR').innerHTML=card('right',s);
  }
  // status + engine-log polls are SLOW (pgrep/subprocess, file read) — fire them
  // WITHOUT await so they never block the render loop (that was the stutter/freeze).
  if(++ctrlN%20===1){
   fetch('/control?action=status').then(r=>r.json()).then(updCtrl).catch(()=>{});
   fetchLog();   // refresh the LOG panel only — do NOT write #hint here. hint(d)
                 // rewrites #hint every frame, so writing a (stale) log trip here
                 // fought it = the flicker. Live trips show in the hwTrip banner.
  }
 }catch(e){chip('conn','bad','dashboard error')}
 requestAnimationFrame(()=>setTimeout(tick,50));
}
tick();
</script></body></html>"""


def recording_info(path: Path) -> dict:
    """Cheap metadata for the recordings browser: duration, frames, engaged and
    right-tracked fractions. Fail-soft (corrupt .npz exist — e.g. truncated
    pre-atomic-save sessions)."""
    try:
        d = np.load(path, allow_pickle=False)
        t = np.asarray(d["t"], float)
        dur = float(t[-1] - t[0]) if t.size else 0.0
        eng = np.asarray(d["engaged"]) if "engaged" in d else None
        rt = np.asarray(d["right_tracked"]) if "right_tracked" in d else None
        return {"frames": int(t.size), "dur": dur,
                "engaged": float(eng.mean()) if eng is not None and eng.size else 0.0,
                "right": float(rt.mean()) if rt is not None and rt.size else 0.0}
    except Exception as e:
        return {"error": type(e).__name__}


def analyze_recording(path: Path) -> dict:
    """Run the offline contract grader (no robot) and surface its OVERALL verdict.
    Loads the IK model, so it takes a few seconds — on demand only."""
    try:
        r = subprocess.run([_sys.executable, "scripts/analyze_session.py", str(path)],
                           cwd=REPO_ROOT, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return {"error": "analyze timed out"}
    out = (r.stdout or "") + (r.stderr or "")
    verdict = next((ln.strip() for ln in reversed(out.splitlines()) if "OVERALL" in ln), "")
    if not verdict:
        return {"error": "no verdict — see out/engine.log"}
    return {"ok": "PASS" in verdict, "verdict": verdict, "detail": out[-1500:]}


def rig_info() -> dict:
    """Static per-side joint SOFT ranges for the dashboard gauges (falls back to
    hard limits if the IK model cannot be built)."""
    rig = load_rig()
    try:
        from bimanual_teleop.arms.ik import ArmIK
        out = {}
        for side in ("left", "right"):
            ik = ArmIK(rig, side)
            out[side] = {"lo": ik.soft_lo.tolist(), "hi": ik.soft_hi.tolist()}
        return out
    except Exception:
        lim = rig["arms"]["joint_limits"]
        return {side: {"lo": list(lim["lower"]), "hi": list(lim["upper"])}
                for side in ("left", "right")}


import glob
import re
import signal
import subprocess
import sys as _sys
from urllib.parse import parse_qs, urlparse

BUILD = time.strftime("%H:%M:%S")    # server start time, shown in the page header


class EngineManager:
    """The dashboard owns the teleop engine process: buttons instead of terminals.
    Starting anything first kills stray engine processes, so port collisions and
    zombie sessions cannot happen."""

    def __init__(self):
        self.proc = None
        self.mode = None
        self.record = None
        self.t0 = None
        self.adopted = False
        self.last_msg = "stopped"
        self._lock = threading.Lock()
        try:                                   # rig override for the control channel
            self.CONTROL_PORT = int(load_rig().get("vr", {}).get("control_port", 8201))
        except Exception:
            pass
        # Adopt a healthy engine that predates this dashboard (restarted mid-
        # session): an engine process exists and the render port answers.
        try:
            out = subprocess.run(["pgrep", "-fl", self.ENGINE_PATTERN],
                                 capture_output=True, text=True).stdout
            if out.strip():
                socket.create_connection(("127.0.0.1", 8102), timeout=0.5).close()
                m = re.search(r"--record (\S+)", out)
                self.adopted, self.mode, self.t0 = True, "LIVE", time.time()
                self.record = m.group(1) if m else None
                self.last_msg = "adopted an engine that was already running"
        except OSError:
            pass

    # Ports the engine must bind or it comes up as a husk: ORBIT ingest PULLs
    # (the ingest thread dies on EADDRINUSE), the render bridges (8101 zmq,
    # 8102 TCP JSON — 8102 is what this dashboard reads), and the engine
    # control channel (8201, the CALIBRATE button). 8099 (orbit viz) is
    # deliberately absent: the engine runs fine without it.
    ENGINE_PORTS = (8087, 8088, 8095, 8100, 8101, 8102, 8122, 8123, 8200, 8201)
    CONTROL_PORT = 8201                    # vr.control_port (engine command channel)
    # '[-]m' anchor: matches 'python -m bimanual_teleop.launch.run_teleop' (and
    # the uv wrapper) but not editors holding the source file open.
    ENGINE_PATTERN = r"[-]m bimanual_teleop\.launch\.run_teleop"

    @classmethod
    def _busy_ports(cls):
        busy = []
        for p in cls.ENGINE_PORTS:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # SO_REUSEADDR so a lingering TIME_WAIT socket from a just-stopped engine
            # does NOT read as "busy" (the engine binds with reuse too). Without this,
            # every launch right after a stop failed with "ports [8102] still busy".
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                busy.append(p)
            finally:
                s.close()
        return busy

    def _kill_strays(self):
        """Kill ANY engine process (ours or not) and wait until the engine ports
        are actually released — a graceful shutdown saves its recording first,
        which takes seconds. A fixed sleep here once spawned husks straight into
        EADDRINUSE. Returns the ports still busy ([] when clear to spawn)."""
        def alive():
            return subprocess.run(["pgrep", "-f", self.ENGINE_PATTERN],
                                  capture_output=True).returncode == 0
        if alive():
            subprocess.run(["pkill", "-INT", "-f", self.ENGINE_PATTERN], capture_output=True)
            deadline = time.time() + 12.0
            while alive() and time.time() < deadline:
                time.sleep(0.3)
            if alive():                       # wedged — there is no save left to lose
                subprocess.run(["pkill", "-9", "-f", self.ENGINE_PATTERN], capture_output=True)
                time.sleep(0.5)
        deadline = time.time() + 8.0
        busy = self._busy_ports()
        while busy and time.time() < deadline:
            time.sleep(0.4)
            busy = self._busy_ports()
        return busy

    def _spawn(self, args, mode, record, module="bimanual_teleop.launch.run_teleop"):
        (REPO_ROOT / "out").mkdir(exist_ok=True)
        log_path = REPO_ROOT / "out" / "engine.log"
        log = open(log_path, "ab")
        log.write(f"\n===== {time.strftime('%H:%M:%S')} dashboard spawn: {mode} ({module}) =====\n".encode())
        log.flush()
        scan_from = log_path.stat().st_size
        self.proc = subprocess.Popen([_sys.executable, "-m", module, *args],
                                     cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        self.mode, self.record, self.t0 = mode, record, time.time()
        # Health gate: "running" only once the render JSON port answers (that is
        # the stream this dashboard draws from). EADDRINUSE in the log or an
        # early exit means a husk — reap it and put the reason on the button row.
        err = None
        deadline = time.time() + 8.0
        while time.time() < deadline:
            if self.proc.poll() is not None:
                err = f"engine exited at startup (code {self.proc.returncode}) — see out/engine.log"
                break
            with open(log_path, "rb") as fh:
                fh.seek(scan_from)
                tail = fh.read()
            if b"Address already in use" in tail:
                err = "port conflict at startup — an old engine survived; press the button again"
                break
            try:
                socket.create_connection(("127.0.0.1", 8102), timeout=0.3).close()
                self.last_msg = f"{mode} running"
                return
            except OSError:
                time.sleep(0.3)
        if err is None:
            err = "render port 8102 never came up — see out/engine.log"
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc, self.mode, self.record, self.t0 = None, None, None, None
        self.last_msg = f"FAILED: {err}"

    def _start(self, args, mode, record):
        with self._lock:
            self._stop_inner()
            busy = self._kill_strays()
            if busy:
                self.last_msg = (f"FAILED: ports {busy} still busy after killing strays — "
                                 "wait a few seconds and press again")
                return self.status()
            self._spawn(args, mode, record)
            return self.status()

    def start_live(self, clutch: str = "always"):
        clutch = clutch if clutch in ("always", "gesture") else "always"
        rec = f"recordings/live_{time.strftime('%m%d_%H%M%S')}.npz"
        return self._start(["--vr", "orbit", "--clutch", clutch, "--record", rec],
                           f"LIVE ({clutch})", rec)

    def start_replay(self, file: str, loop: bool, speed: float = 1.0):
        args = ["--vr", "replay", file] + (["--loop"] if loop else [])
        if speed != 1.0:
            args += ["--speed", f"{speed:g}"]
        label = f"REPLAY {Path(file).name}"
        if speed != 1.0:
            label += f" @{speed:g}x"
        if loop:
            label += " (loop)"
        return self._start(args, label, None)

    # ---- HARDWARE launch (drives the REAL arm) -------------------------------- #
    # run_hw binds the same render ports as the dashboard engine, so a hardware
    # launch stops the render engine first, frees the ports, then takes over the
    # stream — the dashboard then shows the live METAL state + the runtime-guard
    # safety telemetry (status.hw). The rest-pose gate + runtime guard + hard speed
    # ceiling all apply; STOP ALL / RELEASE TORQUE remain the kill switch.
    def _start_hw(self, args, mode, record):
        with self._lock:
            self._stop_inner()                          # free 8101/8102 from the render engine
            for pat, _ in self.KILL_TARGETS:            # and from any stray hw process
                if self._matches(pat):
                    subprocess.run(["pkill", "-INT", "-f", pat], capture_output=True)
            deadline = time.time() + 10.0
            busy = self._busy_ports()
            while busy and time.time() < deadline:
                time.sleep(0.4)
                busy = self._busy_ports()
            if busy:
                self.last_msg = f"FAILED: render ports {busy} still busy — press STOP ALL, then retry"
                return self.status()
            self._spawn(args, mode, record, module="bimanual_teleop.launch.run_hw")
            return self.status()

    def start_hw_teleop(self):
        # Gesture (deadman pinch) clutch removed per operator request — TELEOP LIVE
        # follows CONTINUOUSLY whenever both hands are tracked (--clutch always).
        # STOP / RELEASE TORQUE / e-stop are the kill path.
        rec = f"recordings/hw_{time.strftime('%m%d_%H%M%S')}.npz"
        # BOTH arms (--sides override): the rest-pose gate catches a wrong-arm /
        # channel-flip per side (j6 differs by π) before any torque tracks.
        return self._start_hw(["--vr", "orbit", "--clutch", "always",
                               "--sides", "right,left", "--record", rec],
                              "TELEOP LIVE", rec)

    def return_home(self, side=None):
        """Drive the wired arms to the official HOME pose and RE-ANCHOR rest there
        (launch.return_home): moves the rig to one set position AND makes the
        engage gate pass (the guarded auto --step rest). Managed transient job —
        frees the bus first, spawns as self.proc so STOP / RELEASE TORQUE SIGINT
        it (the chain teardown releases torque). Rate-limited; a tangled/propped
        arm is NOT re-anchored (the map is left untouched)."""
        with self._lock:
            self._stop_inner()
            for pat, _ in self.KILL_TARGETS:
                if self._matches(pat):
                    subprocess.run(["pkill", "-INT", "-f", pat], capture_output=True)
            time.sleep(1.0)
            args = ["--side", side] if side in ("left", "right") else ["--sides", "right,left"]
            (REPO_ROOT / "out").mkdir(exist_ok=True)
            log_path = REPO_ROOT / "out" / "engine.log"
            log = open(log_path, "ab")
            log.write(f"\n===== {time.strftime('%H:%M:%S')} dashboard: RETURN HOME =====\n".encode())
            log.flush()
            self.proc = subprocess.Popen(
                [_sys.executable, "-m", "bimanual_teleop.launch.return_home", *args],
                cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            self.mode, self.record, self.t0 = "RETURN HOME", None, time.time()
            time.sleep(2.5)                       # catch an early crash (no map / bus busy)
            if self.proc.poll() is not None and self.proc.returncode != 0:
                self.last_msg = f"RETURN HOME failed (code {self.proc.returncode}) — see LOG"
                self.proc, self.mode, self.t0 = None, None, None
            else:
                self.last_msg = "RETURN HOME running — arms moving to rest"
            return self.status()

    # STOP ALL targets: everything that can move metal or hold the CAN bus. The
    # dashboard's own render engine (run_teleop) is stopped gracefully first via
    # _stop_inner (saves its recording); these get SIGINT so run_hw's finally
    # block releases torque, with SIGKILL only for the stubborn.
    KILL_TARGETS = (
        (r"scripts/jog_arms\.py", "jog_arms"),
        (r"scripts/jog_right\.py", "jog_right"),
        (r"scripts/hw_bringup\.py", "hw_bringup"),
        (r"scripts/probe_nudge\.py", "probe_nudge"),
        (r"scripts/test_pattern\.py", "test_pattern"),
        (r"bimanual_teleop\.launch\.run_hw", "run_hw"),
        (r"bimanual_teleop\.launch\.return_home", "return_home"),
    )

    def _matches(self, pat):
        return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0

    def kill_all(self):
        with self._lock:
            self._stop_inner()                          # graceful: render engine + its recording
            hit = [name for pat, name in self.KILL_TARGETS if self._matches(pat)]
            for pat, _ in self.KILL_TARGETS:
                if self._matches(pat):
                    subprocess.run(["pkill", "-INT", "-f", pat], capture_output=True)
            if hit:
                deadline = time.time() + 8.0
                while any(self._matches(p) for p, _ in self.KILL_TARGETS) and time.time() < deadline:
                    time.sleep(0.3)
                for pat, _ in self.KILL_TARGETS:        # escalate only the stubborn
                    if self._matches(pat):
                        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
                self.last_msg = f"STOP ALL — signalled {', '.join(hit)} (torque released)"
            else:
                self.last_msg = "STOP ALL — nothing else was running"
            return self.status()

    def _stop_inner(self):
        if self.adopted:
            self.adopted = False
            self._kill_strays()               # graceful INT first — saves its recording
        if self.proc is not None and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.record:
            self.last_msg = f"stopped — saved {self.record}"
        elif self.mode:
            self.last_msg = "stopped"
        self.proc, self.mode, self.record, self.t0 = None, None, None, None

    def stop(self):
        with self._lock:
            self._stop_inner()
            return self.status()

    def status(self):
        if self.adopted:
            alive = subprocess.run(["pgrep", "-f", self.ENGINE_PATTERN],
                                   capture_output=True).returncode == 0
            if not alive:
                self.adopted, self.last_msg = False, "adopted engine exited"
                self.mode, self.record, self.t0 = None, None, None
        else:
            alive = self.proc is not None and self.proc.poll() is None
            if self.proc is not None and not alive and self.mode is not None:
                self.last_msg = f"{self.mode} exited"
                self.mode, self.record, self.t0 = None, None, None
        # Both the curated committed tapes (replay_library/) AND any session
        # recordings/ — the dropdown was empty on a fresh clone (no recordings/ dir).
        recs = sorted(glob.glob(str(REPO_ROOT / "replay_library" / "*.npz"))
                      + glob.glob(str(REPO_ROOT / "recordings" / "*.npz")))
        return {"running": alive, "mode": self.mode, "record": self.record,
                "uptime": round(time.time() - self.t0, 1) if (alive and self.t0) else None,
                "msg": self.last_msg,
                "recordings": [str(Path(r).relative_to(REPO_ROOT)) for r in recs]}

    def engine_cmd(self, cmd: str):
        """Proxy a runtime command to the live engine's control channel
        (CALIBRATE etc.) — the engine keeps running, nothing is restarted."""
        try:
            from bimanual_teleop.control_server import send_command
            reply = send_command(cmd, self.CONTROL_PORT)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            return {"error": f"engine control unreachable ({e}) — is an engine running?",
                    **self.status()}
        out = self.status()
        out["calib_reply"] = reply
        return out

    def dispatch(self, query: dict):
        action = (query.get("action") or [""])[0]
        if action == "start_live":
            return self.start_live((query.get("clutch") or ["always"])[0])
        if action == "start_replay":
            f = (query.get("file") or [""])[0]
            if not f or not (REPO_ROOT / f).exists():
                return {"error": f"no such recording: {f}", **self.status()}
            try:
                sp = float((query.get("speed") or ["1"])[0])
            except ValueError:
                sp = 1.0
            return self.start_replay(f, (query.get("loop") or ["0"])[0] == "1", sp)
        if action == "stop":
            return self.stop()
        if action in ("kill_all", "release_torque"):    # both: SIGINT every mover → torque released
            return self.kill_all()
        if action == "start_hw_teleop":
            return self.start_hw_teleop()
        if action == "return_home":
            return self.return_home()
        if action in ("calibrate", "calibrate_cancel", "calibrate_clear"):
            return self.engine_cmd(action)
        return self.status()


class QuestMonitor:
    """Background `adb get-state` poller so the page can show the Quest USB link
    (device / unauthorized / disconnected / no-adb) independent of the engine."""

    def __init__(self):
        self.state = "checking"
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        import shutil
        while True:
            if not shutil.which("adb"):
                self.state = "no-adb"
            else:
                try:
                    r = subprocess.run(["adb", "get-state"], capture_output=True,
                                       timeout=3, text=True)
                    if r.returncode == 0 and r.stdout.strip():
                        self.state = r.stdout.strip()                 # "device"
                    else:
                        self.state = ("unauthorized" if "unauthorized" in (r.stderr or "")
                                      else "disconnected")
                except (subprocess.SubprocessError, OSError):
                    self.state = "disconnected"
            time.sleep(3.0)


def make_server(feed: StateFeed, host: str, port: int, rig: dict | None = None,
                meshes: "MeshAssets | None" = None,
                manager: "EngineManager | None" = None,
                quest: "QuestMonitor | None" = None) -> ThreadingHTTPServer:
    rig_body = json.dumps(rig or {}).encode()
    mesh_body = json.dumps(meshes.geoms if meshes else {}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):                                  # noqa: N802 (stdlib API)
            if self.path.startswith("/state"):
                snap = feed.snapshot()
                snap["quest"] = quest.state if quest else None
                if meshes is not None and snap.get("state"):
                    try:
                        snap["mesh_T"] = meshes.transforms(snap["state"].get("arms", {}))
                        if meshes.hand_mode == "real":
                            snap["hand_T"] = meshes.hand_transforms(snap["state"])
                        else:
                            snap["hand_mesh"] = meshes.hand_world(snap["state"])
                        # Pipeline stage 4 (REAL ROBOT): FK the same YAM meshes at the
                        # MEASURED encoder pose so the dashboard can draw the metal's
                        # actual joints next to the commanded ones. Only when run_hw is
                        # publishing runtime telemetry (status.hw); absent on the
                        # sim/render path, so stage 4 shows a "no hardware" placeholder.
                        # measured[6] is MODEL-space (HardwareSink maps motor→model
                        # before telemetry), so it feeds meshes.transforms() unchanged —
                        # the exact same FK that stage 3 uses for the commanded q.
                        hw = ((snap["state"].get("status") or {}).get("hw")) or None
                        if hw:
                            hw_arms = {}
                            for side, rec in (hw.get("arms") or {}).items():
                                meas = (rec or {}).get("measured")
                                if meas:
                                    hw_arms[side] = {"q": list(meas)}
                            if hw_arms:
                                snap["mesh_T_hw"] = meshes.transforms(hw_arms)
                    except Exception:
                        snap["mesh_T"] = {}
                body = json.dumps(snap).encode()
                ctype = "application/json"
            elif self.path.startswith("/meshes"):
                body = mesh_body
                ctype = "application/json"
            elif self.path.startswith("/rig"):
                body = rig_body
                ctype = "application/json"
            elif self.path.startswith("/control"):
                q = parse_qs(urlparse(self.path).query)
                out = manager.dispatch(q) if manager else {"error": "no manager"}
                body = json.dumps(out).encode()
                ctype = "application/json"
            elif self.path.startswith("/enginelog"):
                # Tail of the engine/run_hw log so the dashboard SHOWS why a launch
                # died (guard trips, gate failures, tracebacks) instead of hiding it.
                logp = REPO_ROOT / "out" / "engine.log"
                txt = ""
                try:
                    with open(logp, "rb") as fh:
                        fh.seek(0, 2)
                        size = fh.tell()
                        fh.seek(max(0, size - 6000))
                        txt = fh.read().decode("utf-8", "replace")
                except OSError:
                    txt = "(no engine.log yet)"
                lines = [ln for ln in txt.splitlines() if ln.strip()][-40:]
                trip = next((ln for ln in reversed(lines)
                             if "SAFETY TRIP" in ln or "GATE FAILED" in ln
                             or "Error" in ln or "error" in ln or "Traceback" in ln), "")
                out = {"lines": lines, "trip": trip.strip()}
                body = json.dumps(out).encode()
                ctype = "application/json"
            elif self.path.startswith("/recinfo") or self.path.startswith("/analyze"):
                q = parse_qs(urlparse(self.path).query)
                f = (q.get("file") or [""])[0]
                p = REPO_ROOT / f
                if not f or ".." in f or not p.exists():
                    out = {"error": "no such recording"}
                else:
                    out = (recording_info(p) if self.path.startswith("/recinfo")
                           else analyze_recording(p))
                body = json.dumps(out).encode()
                ctype = "application/json"
            else:
                body = PAGE.replace("__BUILD__", BUILD).encode()
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")   # stale tabs caused a "rollback" scare
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):                         # quiet
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=None,
                    help="render JSON stream (default: rig vr.unity_json_endpoint)")
    ap.add_argument("--port", type=int, default=8180, help="dashboard HTTP port")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    endpoint = args.endpoint or load_rig().get("vr", {}).get("unity_json_endpoint", "tcp://127.0.0.1:8102")
    feed = StateFeed(endpoint)
    feed.start()
    try:
        meshes = MeshAssets()
    except Exception as e:
        print(f"[dashboard] mesh view disabled ({e}); falling back to link lines")
        meshes = None
    quest = QuestMonitor()
    quest.start()
    srv = make_server(feed, args.host, args.port, rig=rig_info(), meshes=meshes,
                      manager=EngineManager(), quest=quest)
    print(f"[dashboard] http://{args.host}:{args.port}  ←  {endpoint}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop()
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
