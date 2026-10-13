# VLA closed-loop benchmark

`scripts/run_vla_benchmark.py` evaluates Easy, Medium, and Hard in one
persistent Isaac process.  Each level creates one seeded randomizer and consumes
sample indices `1..N` exactly once.  The standard seeds remain `7101`, `7102`,
and `7103` from `configs/benchmark.yaml`.

## Current v5 delta-policy quick start

On an 8 GB GPU, cap the VLA service at 35% so Isaac RTX rendering retains
enough memory for valid RGB frames:

```powershell
conda activate lerobot-vla
python scripts\serve_smolvla_windows_v044_v4.py `
  --checkpoint results\vla\training\smolvla_visual_fsm_delta_weighted_lora16_v5\checkpoints\027500\pretrained_model `
  --port 8766 `
  --execution-horizon 1 `
  --gpu-memory-fraction 0.35
```

Run strict delta-action evaluation in the Isaac Lab environment:

```powershell
conda activate isaaclab23
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\run_vla_benchmark.py `
  --headless `
  --levels easy medium hard `
  --episodes-per-level 4 `
  --condition-vla-on-visual-pose `
  --condition-vla-on-controller-phase `
  --vla-position-target-mode delta `
  --pure-vla `
  --require-vla `
  --vla-port 8766 `
  --execution-horizon 1 `
  --action-repeat 5 `
  --output-dir results\vla\closed_loop_benchmark\delta_weighted_repro
```

The camera initialization rejects uniform RGB, insufficient valid depth, and
clamp-saturated visual estimates. A rejected initialization is a perception
failure and no action is executed. See `vla_delta_weighted_v5_results.md` for
the measured offline and closed-loop results.

## Evaluation modes

For an RGB-D pose plus Scripted IK reference on the same fixed samples, no VLA
server is needed:

```powershell
python scripts\run_vla_benchmark.py `
  --headless `
  --levels easy medium hard `
  --episodes-per-level 4 `
  --scripted-only `
  --output-dir results\vla\closed_loop_benchmark\scripted_visual_repro
```

The v5 policy uses a 30-D state. The final component is the raw Scripted IK FSM
phase captured immediately before `scripted.compute()`. The phase flag
therefore requires `--condition-vla-on-visual-pose`.

For a strict pure-VLA measurement, add `--pure-vla` (an alias of
`--disable-scripted-fallback`).  A missing or rejected VLA action then fails the
episode; the Scripted IK action is never sent to the environment.  Omit that
flag for the safety-fallback hybrid mode.

The output directory contains:

- `vla_benchmark_episodes.csv`: one row per fixed validation sample;
- `vla_benchmark_levels.json`: Easy/Medium/Hard summaries;
- `vla_benchmark_report.json`: configuration snapshot and overall summary.

Every summary includes success rate, failure reasons, action source counts,
server/RPC latency, and `scripted_ik_fallback_fraction`.  Simulator object truth
is used only for success and failure metrics, never as a policy observation.
The config snapshot reports `state_dimension: 30`, both conditioning flags, and
`controller_phase_source: scripted_state_captured_before_compute`.

In `--scripted-only` reports, `scripted_ik_fallback_fraction=1.0` means every
executed action came from Scripted IK; it does not mean a VLA fallback occurred.
