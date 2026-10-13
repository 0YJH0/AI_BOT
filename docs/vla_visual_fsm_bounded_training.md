# Visual/FSM/bounded SmolVLA training

This run starts a new LoRA adapter from the local SmolVLA base model. It does
not resume the earlier 5,000-step adapter, avoiding the LeRobot 0.4.4 PEFT
double-wrap path. The default schedule is 1,000 warmup steps followed by a
60,000-step decay, with checkpoints every 2,500 updates.

From the project root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_smolvla_visual_fsm_bounded_8gb_v5.ps1
```

Optional overrides include `-BatchSize`, `-Steps`, `-SaveFreq`, `-LoraRank`,
`-SchedulerWarmupSteps`, `-SchedulerDecaySteps`, `-RunName`, and
`-DatasetRoot`. A zero `-SchedulerDecaySteps` (the default) uses the selected
`-Steps` value so a shorter run does not silently inherit a mismatched decay
horizon. The default input dataset is
`exports\lerobot\pick_lift_vla_visual_fsm_bounded_causal_clean_504_v4\train` and
the default output is
`results\vla\training\smolvla_visual_fsm_bounded_lora16_v4`.

The script refuses to overwrite an existing run directory. It validates all
required paths before launching training.
