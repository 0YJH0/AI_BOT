# Hard/LIFT weighted VLA sampling

LeRobot 0.4.4's standard training CLI constructs its offline `DataLoader` with
`shuffle=True`; it does not expose `WeightedRandomSampler` through
`TrainPipelineConfig`.  The project therefore generates a separate, immutable
sampling-plan artifact for the training split.  Validation and test datasets
are never resampled.

Generate a plan after exporting the final delta-XYZ train split:

```powershell
C:\Users\yjh\anaconda3\envs\lerobot-vla\python.exe `
  scripts\build_vla_sampling_plan.py `
  --dataset-root exports\lerobot\<v5-dataset>\train `
  --output results\vla\sampling_plans\<v5-run>_train.json `
  --seed 20261009
```

The default multiplicative priorities are:

| Attribute | Multiplier |
|---|---:|
| Easy / Medium / Hard | 1.0 / 1.0 / 2.0 |
| REST | 1.0 |
| APPROACH_ABOVE_OBJECT | 1.0 |
| APPROACH_OBJECT | 1.25 |
| GRASP_OBJECT | 1.5 |
| LIFT_OBJECT | 3.0 |

Multipliers combine. A Hard/LIFT frame therefore has raw weight `2 x 3 = 6`.
All values can be overridden with `--easy-weight`, `--medium-weight`,
`--hard-weight`, and `--phase-{0..4}-weight`.

The JSON records the original, expected weighted, and deterministic preview
distributions. The adjacent `.weights.npy` contains one float64 weight per
original frame. Loading is rejected if the frame count, weight SHA-256, or
bound training manifest SHA-256 differs. The intended runtime is PyTorch
`WeightedRandomSampler(weights, num_samples, replacement=True, generator=...)`
with `shuffle=False` and the seed stored in the JSON.

For the current 9,906-frame v5 train split, the defaults change expected Hard
coverage from 33.74% to 50.23% and LIFT coverage from 13.33% to 27.93%. The
effective sample size is 7,318.5 frames, so every controller phase and every
difficulty remains represented while Hard contact/lift cases receive more
updates.
