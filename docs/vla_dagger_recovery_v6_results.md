# Targeted 15 mm DAgger recovery fine-tuning: v6 result

## Outcome

This round added online DAgger recovery states around the phase-1 pre-grasp
transition and fine-tuned the existing v5 SmolVLA LoRA adapter without
overwriting any v5 data or checkpoint.

The selected checkpoint is `006000`. It preserves the original held-out task
distribution, and it improves the new recovery-set position MAE from 2.868 cm
to 2.055 cm (28.4%). It does **not** yet pass the strict recovery gate of
0.7 cm MAE / 1.2 cm p90, and a three-episode pure-VLA Isaac diagnostic remains
0/3 with approach failures. The result must therefore be reported as a useful
but incomplete recovery-policy improvement, not as a successful closed-loop
grasp policy.

## DAgger data

The collector stages the end effector in four distance bins around the visual
pre-grasp waypoint

```text
pregrasp_xyz = latched_visual_object_xyz + [0, 0, 0.12]
distance = norm(eef_xyz - pregrasp_xyz)
```

At the staged state, v5 executes one learner action. Scripted IK then labels
and executes a continuous recovery sequence. Selection and labels use the
latched RGB-D estimate; simulator object truth is used only for final metrics.

Raw data: `dataset/vla_dagger_recovery_pregrasp15mm_v6`

| Split | Episodes | Frames | Easy / Medium / Hard frames |
|---|---:|---:|---:|
| Train | 150 | 600 | 200 / 200 / 200 |
| Validation | 39 | 156 | 52 / 52 / 52 |

Across both splits there are 189 learner-visited frames in 756 continuous
frames. The audit found zero teacher-label reconstruction error, no train/val
domain overlap, minimum RGB standard deviation 64.44, minimum RGB range 229,
and minimum valid-depth fraction 0.454. In train, 594/600 frames are within
3 cm of pre-grasp and 507/600 are within 1 cm of the 1.5 cm threshold.

Audit artifact:
`results/vla/dagger_recovery_v6/data_audit.json`

## Export, replay, and weighting

Recovery export:
`exports/lerobot/pick_lift_vla_dagger_recovery_pregrasp15mm_756_v6`

The train split is merged with the immutable v5 train split:
`exports/lerobot/pick_lift_vla_visual_fsm_delta_weighted_dagger15mm_v6/train`

The merged set contains 552 episodes and 10,506 frames:

- v5 replay: 402 episodes / 9,906 frames;
- recovery: 150 episodes / 600 frames;
- source recovery fraction: 5.71%;
- expected recovery fraction after x4 sampling: 14.54%;
- expected Hard fraction: 50.20%;
- expected LIFT fraction: 23.87%.

Recovery frames have loss weight x2, Hard recovery frames x2.5, and the Z axis
has loss weight x3. Image augmentation is photometric only; geometric affine
augmentation remains disabled because delta directions are robot-frame values.

Sampling plan:
`results/vla/sampling_plans/v6_train_dagger15mm_sampling_plan.json`

## Fine-tuning

Initialization is the v5 `027500` rank-16 LoRA adapter. The run uses a fresh
optimizer and explicitly makes the existing single `default` adapter
trainable. Startup checks confirmed 742,656 trainable parameters in 74 tensors.

```powershell
conda activate lerobot-vla
cd C:\Users\yjh\Desktop\work_project\ai_BOT
powershell -ExecutionPolicy Bypass -File scripts\train_smolvla_dagger_recovery_8gb_v7.ps1
```

Configuration: batch 1, 10,000 steps, checkpoint every 1,000 steps, peak LR
`1e-5`, 500-step warmup, cosine decay to `1e-6`, AMP, frozen vision encoder,
and action chunk 4. The RTX 4060 Laptop 8 GB run completed in 44 min 06 s.

Training output:
`results/vla/training/smolvla_visual_fsm_delta_dagger15mm_lora16_v6`

## Held-out recovery scan

The original v5 `027500` baseline on all 156 held-out recovery frames is
2.868 cm MAE / 4.319 cm p90. Its zero-delta baseline is 1.708 cm.

| Step | MAE (cm) | p90 (cm) | X / Y / Z MAE (cm) |
|---:|---:|---:|---:|
| 1,000 | 2.385 | 3.857 | 1.321 / 1.000 / 1.233 |
| 2,000 | 2.385 | 3.993 | 1.346 / 0.963 / 1.238 |
| 3,000 | 2.224 | 3.746 | 1.271 / 0.913 / 1.126 |
| 4,000 | 2.129 | **3.519** | 1.258 / 0.889 / 1.023 |
| 5,000 | 2.117 | 3.597 | 1.280 / 0.884 / 0.977 |
| **6,000** | **2.055** | 3.524 | 1.265 / 0.874 / **0.910** |
| 7,000 | 2.134 | 3.613 | 1.267 / 0.887 / 1.020 |
| 8,000 | 2.116 | 3.586 | 1.258 / 0.888 / 1.005 |
| 9,000 | 2.091 | 3.560 | **1.256** / **0.874** / 0.981 |
| 10,000 | 2.110 | 3.566 | 1.260 / 0.877 / 1.006 |

Artifact:
`results/vla/checkpoint_scans/smolvla_visual_fsm_delta_dagger15mm_lora16_v6_recovery_full156.json`

The recovery MAE improves, but every checkpoint still performs worse than the
1.708 cm zero-delta baseline. Z improves strongly while lateral X/Y residuals
remain the dominant error. More generic training steps after 6k do not solve
that residual.

## Original-distribution retention

`006000` was evaluated on all 1,256 frames of the original episode-disjoint v5
validation split:

| Metric | v5 `027500` | v6 `006000` |
|---|---:|---:|
| Position MAE | 1.167 cm | 1.196 cm |
| Position p90 | 2.329 cm | 2.367 cm |
| Easy / Medium / Hard MAE | 0.740 / 1.337 / 1.408 cm | 0.777 / 1.358 / 1.437 cm |
| Gripper-sign accuracy | 98.89% | 99.28% |

The 0.029 cm MAE change is small, so the recovery fine-tune did not cause
material catastrophic forgetting.

Artifact:
`results/vla/checkpoint_scans/smolvla_visual_fsm_delta_dagger15mm_lora16_v6_validation_full1256_006000.json`

## Pure-VLA closed-loop diagnostic

The selected checkpoint was tested with one fixed episode per level, horizon
1, action repeat 5, 30-D visual/FSM conditioning, and no Scripted IK fallback.

| Level | Result | Initial visual error | Minimum EEF-object distance |
|---|---|---:|---:|
| Easy | approach failure | 0.637 cm | 16.789 cm |
| Medium | approach failure | 0.793 cm | 20.074 cm |
| Hard | approach failure | 0.615 cm | 28.021 cm |

All 180 RPC requests succeeded, all 897 executed actions came from VLA, and
the fallback fraction was zero. RGB and depth quality checks passed. This
isolates the failure to policy behavior rather than camera freshness, visual
initialization, RPC wiring, or scripted fallback.

Artifact:
`results/vla/closed_loop_benchmark/dagger15mm_v6_006000_pure_h1_smoke3`

## Reproduction commands

Checkpoint scans:

```powershell
conda activate lerobot-vla
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:HF_HUB_OFFLINE = "1"

python scripts\scan_smolvla_checkpoints.py `
  --checkpoint-root results\vla\training\smolvla_visual_fsm_delta_dagger15mm_lora16_v6\checkpoints `
  --checkpoint-steps 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 `
  --dataset-root exports\lerobot\pick_lift_vla_dagger_recovery_pregrasp15mm_756_v6\validation `
  --repo-id local/isaac-pick-lift-vla-dagger-recovery-pregrasp15mm-756-v6-validation `
  --samples 0 --batch-size 2 --position-target-mode delta `
  --threshold-position-mae-cm 0.7 --resume `
  --output results\vla\checkpoint_scans\smolvla_visual_fsm_delta_dagger15mm_lora16_v6_recovery_full156.json
```

For the three-episode diagnostic, start `006000/pretrained_model` with
`scripts/serve_smolvla_windows_v044_v4.py` on port 8766 and run:

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\run_vla_benchmark.py `
  --levels easy medium hard --episodes-per-level 1 --max-steps 300 `
  --condition-vla-on-visual-pose --condition-vla-on-controller-phase `
  --vla-position-target-mode delta --pure-vla --require-vla --vla-port 8766 `
  --execution-horizon 1 --action-repeat 5 --headless --device cuda:0 `
  --output-dir results\vla\closed_loop_benchmark\dagger15mm_v6_006000_pure_h1_smoke3_repro
```

## Next corrective experiment

Do not extend this run with more identical epochs. The next dataset/training
round should prioritize the actual learner-visited frame separately from the
three teacher-recovery frames, raise recovery sampling toward 40-50%, and
rebalance the already improved Z axis toward X/Y. It should also include
student-visited states along the full approach corridor, because the smoke run
never reaches the 1.5 cm shell. Only after the held-out recovery policy beats
the zero-delta baseline should the formal 12-episode Isaac benchmark be rerun.

## Verification

The simulator-free suite passes `104` tests. Adapter-only checkpoints are now
supported by the scanner and Windows RPC server through
`train_config.json["policy"]`; future Windows training checkpoints also
materialize `config.json` automatically.
