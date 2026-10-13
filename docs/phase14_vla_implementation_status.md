# Phase 14: SmolVLA hybrid control implementation status

This phase adds a memory-conscious VLA route without replacing the existing Scripted IK, BC, DAgger, PPO, RGB-D pose, dataset, or benchmark pipelines.

## Safety and compatibility decisions

- The original project was copied to `../ai_BOT_original_20260923_before_vla` before VLA work started.
- Isaac Lab stays in `isaaclab23`; LeRobot and SmolVLA run in the separate `lerobot-vla` environment.
- The VLA process communicates with Isaac Lab through localhost RPC, avoiding dependency conflicts.
- The policy receives RGB plus the 25-D robot proprioceptive state. Simulator object pose is never part of the policy observation.
- Simulator object pose is allowed only in benchmark metrics.
- VLA outputs are guarded by workspace, step-size, quaternion, gripper-stage, and finite-value checks. Invalid actions fall back to Scripted IK.

## Verified local artifacts

- Usable LeRobot dataset: `exports/lerobot/pick_lift_vla_images_baseline_v1`
- Dataset size: 59 successful episodes, 1,623 frames
- Observation: one 240 x 320 RGB image plus 25-D proprioceptive state
- Action: 8-D absolute end-effector pose and gripper command
- Base model: `results/vla/models/smolvla_base`
- Model load: 450,046,176 parameters; about 0.86 GiB CUDA allocation before training activations
- LoRA smoke result: rank 8, 371,328 trainable parameters, batch size 1, one optimizer update in about 2.45 seconds, loss 0.964

## Authoritative commands for Python 3.11 / Windows

Validate the environment:

```powershell
conda activate lerobot-vla
cd C:\Users\yjh\Desktop\work_project\ai_BOT
python scripts\validate_vla_env_v2.py
```

Re-export the image-backed dataset if the Isaac data changes:

```powershell
python scripts\export_lerobot_vla_images_v044.py `
  --input_dir dataset\phase12_rgbd_hard_raw `
  --output_dir exports\lerobot\pick_lift_vla_images_baseline_v1 `
  --repo_id local/isaac-pick-lift-vla-images-baseline-v1
```

Run a 10-step checkpoint smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_smolvla_smoke_py311.ps1
```

Start the longer LoRA run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_smolvla_py311_8gb.ps1 `
  -Steps 5000 -SaveFreq 500 -RunName smolvla_lora_r8_v1
```

## Current limitation before claiming a trained VLA result

The current dataset contains only 59 complete successful Hard demonstrations. This is enough to verify the software and memory path, but not enough to make a credible robustness claim. The next data milestone should retain failures for analysis while collecting at least 500 balanced successful demonstrations across Easy, Medium, and Hard, including camera and physics randomization. The validation set must use held-out seeds and must not overlap training episodes.
