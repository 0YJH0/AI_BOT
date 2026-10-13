# SmolVLA on Windows: verified commands

Use this file as the authoritative Phase 14 entry point. Earlier `v044`, `v2`,
and `v3` files are retained as implementation history and must not be used for
the final PEFT inference path.

## 1. Validate the 8 GB VLA environment

```powershell
conda activate lerobot-vla
cd C:\Users\yjh\Desktop\work_project\ai_BOT
python scripts\validate_vla_env_v2.py
```

## 2. Export Isaac demonstrations without MP4 decoding

```powershell
python scripts\export_lerobot_vla_images_v044.py `
  --input_dir dataset\phase12_rgbd_hard_raw `
  --output_dir exports\lerobot\pick_lift_vla_images_baseline_v1 `
  --repo_id local/isaac-pick-lift-vla-images-baseline-v1
```

## 3. Verify training and checkpoint saving

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_smolvla_smoke_py311_v2.ps1
```

## 4. Train the LoRA adapter

Do this after collecting the balanced dataset described below:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_smolvla_py311_8gb_v2.ps1 `
  -Steps 5000 -SaveFreq 500 -RunName smolvla_lora_r8_v1
```

## 5. Start the verified PEFT inference service

Replace the checkpoint path with the selected benchmark checkpoint:

```powershell
conda activate lerobot-vla
python scripts\serve_smolvla_windows_v044_v4.py `
  --checkpoint results\vla\training\smolvla_lora_r8_v1\checkpoints\last\pretrained_model `
  --device cuda `
  --execution-horizon 8
```

## 6. Connect Isaac Lab hybrid control

Open a second PowerShell window:

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\run_hybrid_vla_pick_lift.py `
  --num_envs 1 `
  --max_steps 600 `
  --realtime `
  --keep_open
```

The hybrid controller executes VLA actions at 10 Hz and holds each command for
five 50 Hz physics/control ticks. Invalid or unsafe VLA output is rejected and
control falls back to Scripted IK.

## 7. Optional RPC-only check

With the service running on port 8766:

```powershell
python scripts\smoke_test_vla_rpc_v044.py --port 8766 --sample-index 0
```

Expected shapes are RGB `[240, 320, 3]`, state `[25]`, and actions `[8, 8]`.

## 8. Phase-conditioned 30-D checkpoint

The commands above remain valid for the earlier 25-D checkpoint.  For the new
30-D checkpoint, start the server with the training chunk size:

```powershell
conda activate lerobot-vla
python scripts\serve_smolvla_windows_v044_v4.py `
  --checkpoint <phase_conditioned_checkpoint_dir> `
  --device cuda `
  --execution-horizon 4
```

In the Isaac terminal, enable both visual-pose and controller-phase state:

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\run_hybrid_vla_pick_lift.py `
  --condition-vla-on-visual-pose `
  --condition-vla-on-controller-phase `
  --require-vla `
  --execution-horizon 4 `
  --realtime `
  --keep-open
```

The 30th value is the raw controller phase ID captured before the matching
`scripted.compute()` call.  The phase option cannot be enabled without visual
pose conditioning.  Pure-VLA Benchmark mode may still advance this FSM to form
the observation, but `--pure-vla` prevents every Scripted IK fallback action
from being executed.

## Data gate before the long run

The current image-backed dataset contains only 59 complete successful Hard
episodes (1,623 frames). It is sufficient for integration tests, not for a
credible policy result. Collect at least 500 balanced successful episodes:

- Easy: 100
- Medium: 150
- Hard: 250

Keep failed episodes for Failure Case statistics, but do not mix them into the
success-only behavior-cloning target without an explicit recovery label. Split
train/validation/test by randomization seed, not by adjacent frames.
