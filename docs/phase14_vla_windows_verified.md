# Phase 14 VLA: Windows-verified path

This document supersedes the early experimental VLA commands. No earlier file
was removed so that the project history and failure reproduction remain intact.

## Verified design

- Isaac Lab remains in `isaaclab23`.
- SmolVLA and LeRobot run in the independent `lerobot-vla` environment.
- The two processes exchange NumPy arrays only over localhost RPC.
- Policy input is one RGB frame plus 25-D robot proprioception; simulated object
  pose is excluded from the observation and used only for benchmark metrics.
- Policy action is an 8-D absolute end-effector pose plus gripper command.
- A safety guard clamps workspace and per-step motion, normalizes orientation,
  prevents premature closing, and falls back to Scripted IK on invalid output.

## Verified artifacts

- Dataset: `exports/lerobot/pick_lift_vla_images_baseline_v1`
- Content: 59 successful episodes and 1,623 aligned 240 x 320 RGB frames
- Base model: `results/vla/models/smolvla_base`
- LoRA: rank 8, 371,328 trainable parameters out of about 450M total
- 10 training updates complete on the local RTX 4060 8 GB configuration

## Windows checkpoint compatibility

LeRobot 0.4.4 normally updates `checkpoints/last` using a symbolic link. Default
Windows accounts cannot create that link. `scripts/lerobot_train_windows.py`
changes only this operation to a directory junction, which needs no elevated
token and preserves the checkpoint and resume directory layout.

## Commands

Validate the separate environment:

```powershell
conda activate lerobot-vla
cd C:\Users\yjh\Desktop\work_project\ai_BOT
python scripts\validate_vla_env_v2.py
```

Run the 10-step end-to-end training/checkpoint test:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_smolvla_smoke_py311_v2.ps1
```

Start the longer 8 GB LoRA run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_smolvla_py311_8gb_v2.ps1 `
  -Steps 5000 -SaveFreq 500 -RunName smolvla_lora_r8_v1
```

The present 59 episodes validate engineering correctness, not final policy
robustness. Before reporting VLA success rates, collect at least 500 balanced
successful demonstrations across Easy, Medium, and Hard and evaluate on held-out
random seeds that do not overlap training episodes.
