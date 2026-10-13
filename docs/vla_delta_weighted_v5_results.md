# VLA delta-XYZ weighted training: v5 result

## Outcome

The requested v5 training run is complete. Position actions are now
current-end-effector-relative `delta xyz` targets, Z error is weighted more
heavily, and Hard/contact/lift frames receive both higher loss weight and higher
sampling probability. All previous datasets, checkpoints, and scripts remain
unchanged; v5 uses independent output paths.

The offline 2 cm gate passed, but the strict pure-VLA closed-loop benchmark did
not yet pass the 10 cm lift criterion. The result is useful because the final
camera checks and the 8 GB deployment setting separate perception failures from
policy failures.

## What changed

- Label: `clip_norm(FSM_waypoint_xyz - current_eef_xyz, 0.04 m)`.
- Runtime decode: `absolute_target_xyz = current_eef_xyz + predicted_delta_xyz`
  at each executed action.
- Policy state: 30-D visual state plus the pre-action controller phase.
- Loss: only the real eight action dimensions contribute; Z receives a 3.0
  multiplier.
- Per-frame loss weight: Hard 1.25, LIFT 1.5; the factors multiply.
- Weighted sampler: Hard 2.0; phase 2/3/4 are 1.25/1.5/3.0.
- Augmentation: photometric transforms only. Geometric image transforms remain
  disabled because delta directions are defined in the robot frame.

A Hard/LIFT frame therefore receives sampler weight `2 x 3 = 6` and loss
weight `1.25 x 1.5 = 1.875`: about 11.25 times the expected gradient exposure
of an Easy non-LIFT frame.

Key implementation files:

- `src/isaac_lab_data_engine/vla/dataset.py`
- `src/isaac_lab_data_engine/vla/hybrid_controller.py`
- `src/isaac_lab_data_engine/vla/weighted_smolvla_loss.py`
- `src/isaac_lab_data_engine/vla/sampling.py`
- `scripts/export_vla_visual_fsm_delta_v5.ps1`
- `scripts/train_smolvla_visual_fsm_delta_weighted_8gb_v6.ps1`

## Dataset and training

Dataset:
`exports/lerobot/pick_lift_vla_visual_fsm_delta_weighted_causal_clean_504_v5`

| Split | Episodes | Frames |
|---|---:|---:|
| Train | 402 | 9,906 |
| Validation | 51 | 1,256 |
| Test | 51 | 1,251 |

The sampler changes expected Hard coverage from 33.74% to 50.23% and LIFT
coverage from 13.33% to 27.93%. The effective sample size is 7,318.5 frames.

Training output:
`results/vla/training/smolvla_visual_fsm_delta_weighted_lora16_v5`

- fresh LoRA rank 16 from the local SmolVLA base;
- batch size 2, action chunk 4, 30,000 optimizer steps;
- checkpoints every 2,500 steps;
- best fixed-validation checkpoint: `027500`.

Reproduction commands use new output names because the scripts intentionally
refuse to overwrite an existing dataset or training run:

```powershell
conda activate lerobot-vla
cd C:\Users\yjh\Desktop\work_project\ai_BOT

powershell -ExecutionPolicy Bypass -File scripts\export_vla_visual_fsm_delta_v5.ps1 `
  -OutputRoot exports\lerobot\pick_lift_vla_visual_fsm_delta_weighted_causal_clean_504_v5_repro `
  -SamplingPlan results\vla\sampling_plans\smolvla_visual_fsm_delta_weighted_v5_repro_train.json

powershell -ExecutionPolicy Bypass -File scripts\train_smolvla_visual_fsm_delta_weighted_8gb_v6.ps1 `
  -DatasetRoot exports\lerobot\pick_lift_vla_visual_fsm_delta_weighted_causal_clean_504_v5_repro\train `
  -SamplingPlan results\vla\sampling_plans\smolvla_visual_fsm_delta_weighted_v5_repro_train.json `
  -RunName smolvla_visual_fsm_delta_weighted_lora16_v5_repro
```

These two PowerShell launchers contain the current machine's Conda Python path;
change `$PythonExe` before using them on another computer.

## Fixed validation result

Full episode-disjoint validation split, 1,256 frames:

| Metric | v4 absolute target | v5 delta target |
|---|---:|---:|
| Position MAE | 2.8002 cm | **1.1670 cm** |
| Position p90 | 5.2472 cm | **2.3290 cm** |
| X/Y/Z MAE | 0.9165 / 1.1890 / 1.9466 cm | **0.5749 / 0.5737 / 0.5781 cm** |
| Easy / Medium / Hard MAE | 2.0378 / 2.6804 / 3.6582 cm | **0.7402 / 1.3373 / 1.4075 cm** |

Overall position MAE improved by 58.3%; Z MAE improved by 70.3%. Orientation
MAE is 1.091 degrees, gripper-sign accuracy is 98.89%, and all predictions are
finite. The zero-delta baseline is 2.4692 cm, so the trained policy beats the
no-motion baseline.

Per-horizon position MAE is 1.1670, 1.4253, 1.6087, and 1.4734 cm for H0-H3.
Closed-loop evaluation therefore uses one predicted action at a time
(`--execution-horizon 1`) even though training uses a four-action chunk.

Validation artifact:
`results/vla/checkpoint_scans/smolvla_visual_fsm_delta_weighted_lora16_v5_validation_best_full1256_horizons.json`

Full fixed-validation scan for the selected checkpoint:

```powershell
conda activate lerobot-vla
python scripts\scan_smolvla_checkpoints.py `
  --checkpoint-root results\vla\training\smolvla_visual_fsm_delta_weighted_lora16_v5\checkpoints `
  --checkpoint-steps 27500 `
  --dataset-root exports\lerobot\pick_lift_vla_visual_fsm_delta_weighted_causal_clean_504_v5\validation `
  --repo-id local/isaac-pick-lift-vla-visual-fsm-delta-weighted-causal-clean-504-v5-validation `
  --samples 0 `
  --batch-size 2 `
  --position-target-mode delta `
  --threshold-position-mae-cm 2.0 `
  --output results\vla\checkpoint_scans\v5_validation_repro.json
```

## Closed-loop result

Both tests use the same 12 fixed samples: Easy/Medium/Hard, four episodes each.

| Controller | Easy | Medium | Hard | Overall |
|---|---:|---:|---:|---:|
| RGB-D pose + Scripted IK | 4/4 | 3/4 | 4/4 | **11/12 (91.67%)** |
| Pure v5 VLA, no fallback | 0/4 | 0/4 | 0/4 | **0/12** |

The Scripted IK failure is the Medium sample whose initial visual-position error
is 3.64 cm. The other successful samples are mostly around 0.6-0.8 cm.

For pure VLA, all 12 visual initializations are valid and all 720 RPC requests
succeed. Eleven episodes are approach failures. One Hard episode registers
grasp/held evidence and is classified as a lift failure, but still produces no
positive object lift. Initial visual error is 0.921 cm mean / 0.639 cm median;
the best end-effector-to-object distance is 4.257 cm and the mean is 13.730 cm.

Artifacts:

- Scripted IK:
  `results/vla/closed_loop_benchmark/stable_visual_scripted_ik_quality_retry_smoke12_v2`
- Pure VLA:
  `results/vla/closed_loop_benchmark/delta_weighted_027500_pure_h1_gpu35_quality_smoke12_v2`

## Camera and 8 GB GPU requirement

The benchmark now accepts a visual initialization only after five consecutive
valid estimates and rejects uniform or clamp-saturated RGB frames. Defaults are:

- RGB standard deviation at least 30;
- RGB range at least 128;
- finite positive depth fraction at least 0.40;
- three forced render/update passes per warmup step;
- at most three retries of the exact same randomized sample.

On the RTX 4060 Laptop 8 GB GPU, a VLA server memory fraction of 0.55 caused
Isaac to return uniform RGB while depth remained valid. The quality gate safely
reported `invalid_visual_initialization`. Limiting the VLA server to 0.35
restored valid RGB and is required for the commands below.

```powershell
# Terminal 1: local VLA service
conda activate lerobot-vla
cd C:\Users\yjh\Desktop\work_project\ai_BOT
python scripts\serve_smolvla_windows_v044_v4.py `
  --checkpoint results\vla\training\smolvla_visual_fsm_delta_weighted_lora16_v5\checkpoints\027500\pretrained_model `
  --host 127.0.0.1 `
  --port 8766 `
  --device cuda `
  --execution-horizon 1 `
  --gpu-memory-fraction 0.35
```

```powershell
# Terminal 2: strict pure-VLA benchmark
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\run_vla_benchmark.py `
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
  --headless `
  --device cuda:0 `
  --output-dir results\vla\closed_loop_benchmark\delta_weighted_repro
```

For the perception/controller reference, replace the VLA flags with
`--scripted-only`; no VLA server is required.

## Why offline MAE passed but closed loop failed

The label/runtime semantics are consistent. The dominant remaining issue is
the near-target phase-1 distribution:

- only 149 of 3,328 phase-1 training frames are within 1.5 cm of the waypoint;
- that is 1.50% of all train frames and about 1.20% after the current weighted
  sampler shifts probability toward Hard and LIFT;
- on 20 held-out near-target phase-1 frames, target delta norm is 1.08 cm but
  predicted norm is 2.53 cm, with 2.02 cm MAE and 1.82 cm lateral error;
- closed-loop episodes request phase 1 about 57 times, versus about 8.28 expert
  frames per episode in the offline dataset;
- diffusion sampling adds roughly 0.76-0.88 cm dispersion, comparable with the
  1.5 cm FSM transition threshold.

The next training round should therefore prioritize near-target phase-1
distance bins and DAgger recovery states, plus deterministic or multi-sample
median inference. More generic Hard/LIFT epochs alone are unlikely to solve the
observed approach oscillation.

## Verification

`97` simulator-free tests pass, covering delta labels/runtime decoding,
weighted loss, weighted sampling, checkpoint metrics, and benchmark reporting.
