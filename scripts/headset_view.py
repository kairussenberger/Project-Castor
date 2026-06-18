#!/usr/bin/env python
"""Stream the Mac screen (your dashboard) INTO the Quest's ORBIT app.

ORBIT renders a video panel in-headset and SUBs for it on tcp://127.0.0.1:10505
(via adb reverse) — normally fed by a robot camera. Without a stream the panel
is blank and the operator is blind to the dashboard while wearing the headset,
which makes solo operation miserable. This script captures the screen with
ffmpeg (hardware HEVC via VideoToolbox), packetizes it in ORBIT's wire format,
and publishes it — so the dashboard (banner prompts, TRACKED chips, robot view)
is visible inside VR.

Wire format (reverse-engineered from CameraOneStreamer.cs, verified constants):

    ZMQ PUB, one packet per HEVC access unit:
      'O''R''B''T' | version=1 (u8) | flags (u8: 1=keyframe, 2=config)
      | width u16 LE | height u16 LE | ptsUs u64 LE | payloadLen u32 LE
      | payload (Annex-B HEVC access unit)

The stream MUST be side-by-side stereo: StereoSbsRenderTexture.shader always
maps the left half of the texture to the left eye and the right half to the
right eye (both the legacy and the reprojection path). A mono frame puts a
DIFFERENT half of the screen in each eye — binocular rivalry, unreadable.
So the captured screen is duplicated into both halves (zero disparity → the
panel reads as a flat 2D screen). Header width/height = full SBS dims;
the reprojection config carries PER-EYE dims (half width).

Every frame is encoded all-intra with VPS/SPS/PPS repeated (dump_extra), so any
frame is a clean decoder entry point (the app's own config uses gop=1).

    uv run python scripts/headset_view.py                 # whole main screen
    uv run python scripts/headset_view.py --fps 30 --width 1280
    uv run python scripts/headset_view.py --list-screens  # pick a display

First run: grant the terminal Screen Recording permission
(System Settings → Privacy & Security → Screen Recording), then rerun.
Requires: ffmpeg (brew install ffmpeg). Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import json
import math
import select
import shutil
import struct
import subprocess
import sys
import time

MAGIC = b"ORBT"
VERSION = 1
FLAG_KEYFRAME = 0x01
FLAG_CONFIG = 0x02
PORT = 10505

# HEVC NAL unit types (nal_unit_header first byte >> 1, 6 bits)
NAL_AUD = 35
NAL_VPS, NAL_SPS, NAL_PPS = 32, 33, 34
IDR_TYPES = {19, 20}            # IDR_W_RADL, IDR_N_LP


def _adb_reverse() -> None:
    if not shutil.which("adb"):
        print(f"[headset-view] adb not found — run `adb reverse tcp:{PORT} tcp:{PORT}` yourself")
        return
    r = subprocess.run(["adb", "reverse", f"tcp:{PORT}", f"tcp:{PORT}"],
                       capture_output=True, timeout=5)
    print(f"[headset-view] adb reverse tcp:{PORT}: {'ok' if r.returncode == 0 else 'FAILED'}")


def _split_nals(buf: bytearray):
    """Yield (start, end, nal_type) for each Annex-B NAL in buf (complete ones)."""
    i = 0
    starts = []
    n = len(buf)
    while i < n - 3:
        if buf[i] == 0 and buf[i + 1] == 0 and (buf[i + 2] == 1 or
                                                (buf[i + 2] == 0 and i < n - 4 and buf[i + 3] == 1)):
            sc = 3 if buf[i + 2] == 1 else 4
            starts.append((i, i + sc))
            i += sc
        else:
            i += 1
    for k, (s, p) in enumerate(starts):
        e = starts[k + 1][0] if k + 1 < len(starts) else None
        if e is None:
            yield s, None, (buf[p] >> 1) & 0x3F
        else:
            yield s, e, (buf[p] >> 1) & 0x3F


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen", default=None,
                    help="avfoundation screen index (default: auto-detect 'Capture screen 0')")
    ap.add_argument("--list-screens", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1440,
                    help="PER-EYE encoded width (the SBS stream is twice this)")
    ap.add_argument("--height", type=int, default=810,
                    help="encoded height (screen is letterboxed to fit — the header must carry exact dims)")
    ap.add_argument("--bitrate", default="20M",
                    help="all-intra SBS needs headroom for legible dashboard text")
    ap.add_argument("--lavfi", default=None, help=argparse.SUPPRESS)   # self-test source (e.g. testsrc)
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found — brew install ffmpeg")
        return 1
    if args.list_screens:
        subprocess.run(["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                       stderr=None)
        return 0
    if args.screen is None:
        # Auto-detect: device indices shift with cameras (Desk View, iPhone…) —
        # a hard-coded index once streamed the Desk View camera into the headset.
        out = subprocess.run(["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                             capture_output=True, text=True).stderr
        import re
        m = re.search(r"\[(\d+)\] Capture screen 0", out)
        if not m:
            print("[headset-view] no 'Capture screen' device found — pass --screen N (see --list-screens)")
            return 1
        args.screen = m.group(1)
        print(f"[headset-view] auto-detected screen device index {args.screen}")

    # A wedged capture from a previous run (e.g. killed parent, orphaned ffmpeg)
    # holds the AVCapture session and starves new captures: zero frames, no
    # error. Reap our own stale pipelines (signature: avfoundation → raw hevc
    # on stdout) before starting.
    subprocess.run(["pkill", "-9", "-f", r"ffmpeg.*avfoundation.*-f hevc -$"],
                   capture_output=True)

    import zmq
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 4)            # latest-wins: never build latency
    pub.bind(f"tcp://127.0.0.1:{PORT}")
    _adb_reverse()

    # fps first: the avfoundation screen device's timestamps make ffmpeg's CFR
    # sync duplicate frames (~270/s measured) — drop the dups before encoding.
    # Then letterbox to one eye and duplicate into both SBS halves (docstring).
    vf = (f"fps={args.fps},"
          f"scale={args.width}:{args.height}:force_original_aspect_ratio=decrease,"
          f"pad={args.width}:{args.height}:(ow-iw)/2:(oh-ih)/2,"
          f"split=2[l][r];[l][r]hstack=inputs=2")
    if args.lavfi:
        src_args = ["-re", "-f", "lavfi", "-i", f"{args.lavfi}=rate={args.fps}"]
    else:
        # The screen device rejects ffmpeg's default yuv420p — request one of
        # its native formats explicitly (nv12), or the input fails to open.
        src_args = ["-f", "avfoundation", "-capture_cursor", "1",
                    "-pixel_format", "nv12",
                    "-framerate", str(args.fps), "-i", f"{args.screen}:none"]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *src_args,
           "-vf", vf,
           "-c:v", "hevc_videotoolbox", "-realtime", "1",
           "-b:v", args.bitrate, "-g", "1",                       # all-intra: gop 1
           "-bsf:v", "hevc_metadata=aud=insert,dump_extra=freq=keyframe",
           "-f", "hevc", "-"]
    print("[headset-view] " + " ".join(cmd))
    # bufsize=0 → raw pipe: select() sees exactly what read() will return.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=0)

    buf = bytearray()
    t0 = time.monotonic()
    sent = 0
    w_eye, h_out = args.width, args.height
    w_out = 2 * w_eye                        # full SBS stream width (header dims)
    last_log = 0.0
    last_cfg = -10.0

    def send_config(now: float) -> None:
        # Minimal valid reprojection payload (flat screen: pinhole intrinsics
        # from a chosen FOV; both eyes identical → plain 2D panel). The app
        # rejects any type other than this exact string (unexpected_type) and
        # validates PER-EYE dims, so width here is the half-frame width.
        hfov, vfov = 70.0, 70.0 * h_out / w_eye
        fx = w_eye / (2.0 * math.tan(math.radians(hfov) / 2.0))
        fy = h_out / (2.0 * math.tan(math.radians(vfov) / 2.0))
        eye = {"fx": fx, "fy": fy, "cx": w_eye / 2.0, "cy": h_out / 2.0,
               "width": w_eye, "height": h_out,
               "rectified_hfov_deg": hfov, "rectified_vfov_deg": vfov}
        payload = json.dumps({"type": "orbit_stereo_reprojection_config", "version": 1,
                              "backend": "headset_view", "profile": "flat",
                              "left": eye, "right": eye,
                              "generated_monotonic_us": int(now * 1e6)}).encode()
        pkt = MAGIC + struct.pack("<BBHHQI", VERSION, FLAG_CONFIG, w_out, h_out,
                                  int((now - t0) * 1e6), len(payload)) + payload
        pub.send(pkt)
    try:
        while True:
            # Without Screen Recording permission the capture device opens but
            # delivers ZERO frames forever (no error) — fail loudly instead.
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if not ready:
                if sent == 0 and time.monotonic() - t0 > 8.0:
                    print("\n[headset-view] no frames after 8s — this is almost always the "
                          "macOS Screen Recording permission.\nGrant it to your terminal app "
                          "(System Settings → Privacy & Security → Screen & System Audio "
                          "Recording),\nquit+reopen the terminal, and rerun.")
                    return 1
                continue
            chunk = proc.stdout.read(65536)
            if not chunk:
                err = proc.wait()
                print(f"\n[headset-view] ffmpeg exited ({err}). On first run grant the "
                      "terminal Screen Recording permission and rerun.")
                return 1 if err else 0
            buf.extend(chunk)
            # Access units are AUD-delimited (hevc_metadata=aud=insert): emit the
            # span between consecutive AUDs.
            cut = 0
            aud_positions = [s for s, e, ty in _split_nals(buf) if ty == NAL_AUD and e is not None]
            # need at least two AUDs to know one complete AU
            while len(aud_positions) >= 2:
                s0, s1 = aud_positions[0], aud_positions[1]
                au = bytes(buf[s0:s1])
                types = {ty for _, _, ty in _split_nals(bytearray(au))}
                # NOTE: FLAG_CONFIG is reserved for the JSON reprojection-config
                # packet — the app routes config packets AWAY from the decoder.
                flags = FLAG_KEYFRAME if types & IDR_TYPES else 0
                pts = int((time.monotonic() - t0) * 1e6)
                pkt = MAGIC + struct.pack("<BBHHQI", VERSION, flags, w_out, h_out,
                                          pts, len(au)) + au
                pub.send(pkt)
                sent += 1
                cut = s1
                aud_positions = aud_positions[1:]
            if cut:
                del buf[:cut]
            now = time.monotonic()
            if now - last_cfg > 2.0:                 # config keeps late joiners happy
                send_config(now)
                last_cfg = now
            if now - last_log > 5.0:
                last_log = now
                print(f"[headset-view] {sent} frames sent ({sent / max(now - t0, 1e-9):.1f} fps avg)",
                      flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        pub.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
