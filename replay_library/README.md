# Replay library — committed probe tapes

Curated teleop recordings (Quest 3 / ORBIT, 2026-06-12): trimmed, with each
session's neutral-pose calibration EMBEDDED in the npz (`calib_json` key) —
replay and analysis apply it automatically, so these tapes are meaningful on
any machine with no extra setup. Scores, per-tape verdicts, and known issues:
**docs/REPLAY_LIBRARY.md**.

    uv run python -m bimanual_teleop.launch.run_teleop --vr replay replay_library/roll_right_left.npz --viz
    uv run python scripts/analyze_session.py replay_library/roll_right_left.npz

Hardware day (ARCHITECTURE.md sim→real step 5) replays these in order:
roll_right_left → reach_box → wrist_swing (clap only after the crossing issue
closes). The `.calib.json` sidecars duplicate the embedded fits for provenance.
The untrimmed `.raw.npz` originals are NOT committed (local `recordings/`).
