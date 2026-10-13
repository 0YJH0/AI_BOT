# Phase 11：PPO 正式 10 cm 抬升

## 当前结论

策略按照与 Phase 7 相同的正式判据评测：物体中心高于桌面 10 cm、夹爪闭合、末端与物体距离小于 8 cm，并连续保持 10 个仿真步。每个难度独立运行 256 个 episode、最多 600 步。

| 策略 | Easy | Medium | Hard |
|---|---:|---:|---:|
| 旧 BC + PPO | 100.00% | 32.42% | 10.16% |
| 自适应 BC | 100.00% | 31.25% | 10.55% |
| 自适应 BC + 2 轮 DAgger | 100.00% | 77.73% | 73.83% |
| DAgger + 20 次保守 PPO | 100.00% | 77.73% | 57.42% |
| DAgger + 128-step PPO + BC auxiliary（当前最佳） | **100.00%** | **96.88%** | **89.45%** |

当前综合 Benchmark 最优 checkpoint：

`results/ppo/ik_abs_dagger2_ppo_bc_128step_1024env_100iter/model_100.pt`

256 episode/level 正式报告：

`results/ppo_evaluation_dagger2_ppo_bc_128step_100iter_final_256/benchmark_report.json`

新方案把 rollout 从 24 步提高到 128 步，并在 on-policy 状态上用冻结 DAgger actor 施加 BC 辅助约束。Hard 相比旧短 rollout PPO 提升 32.03 个百分点，相比 DAgger 初始化提升 15.63 个百分点。

## 本轮修改

1. 抬升目标不再使用固定中心命令 `(0.55, 0, z)`，而是固定在每个物体抓取前的 x/y 锚点，仅将 z 提高 0.15 m。
2. 删除固定目标位置观测及横向目标跟踪奖励，训练目标与正式“原地抬升 10 cm”判据一致。
3. 用几何关系和夹持反馈产生五阶段 one-hot：rest、approach、descend、grasp、lift；下降阶段会等待水平对齐，不再按固定时钟强制跳转。
4. 新增 6 维抓取状态：水平距离、垂直差、三维距离、夹爪平均位置、夹爪平均速度、接触/持有代理。
5. 接触代理要求夹爪位置位于 0.5–2.5 cm 且末端距离小于 8 cm，从而区分“夹住物体”和“空抓完全闭合”。
6. 策略观测为 57 维；旧的 53/56/59 维 checkpoint 与当前网络不兼容。
7. 新增采集诊断：达到正式 10 cm 的环境数、平均最大抬升、最大横向漂移和最高点横向偏移。
8. 两轮 DAgger 将教师标签扩展到策略实际访问的偏离状态，解决了离线 BC 误差很低但闭环超时严重的问题。
9. PPO rollout 从 24 步（0.48 s）增加到 128 步（2.56 s），一次更新可以覆盖接近、下降、闭合和抬升的大部分动作链。
10. 每轮 PPO 更新后，在本轮访问状态上最小化当前 actor 与冻结 DAgger teacher 的加权动作 MSE，抑制灾难性遗忘。
11. 修复 BC checkpoint 中 `1e-4` optimizer 学习率覆盖命令行 `1e-6` 的问题；正式实验不恢复空 BC optimizer，并使用 fixed schedule。
12. 每 20 iterations 保存 checkpoint，并用独立 Benchmark 按 Hard、Medium、Easy 的优先级自动选模。

## 数据与训练规模

- 初始并行示教：1024 个环境，Easy/Medium/Hard 共 675,840 samples。
- 初始 BC 验证集位置 MAE：2.00 mm；夹爪方向准确率：100%。
- 两轮 DAgger 后聚合数据：2,027,520 samples。
- 长时域 PPO 微调：1024 个环境、128 steps/env、100 iterations、13,107,200 transitions。
- PPO 学习率 `1e-6`（fixed）、动作噪声标准差 `0.005`、BC auxiliary coefficient `0.05`。
- 训练耗时 399.42 s，平均吞吐 32,815 transitions/s。

## 训练与评测命令

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"

python scripts\train_bc_warmstart.py --headless --device cuda:0 `
  --num-envs 1024 --horizon 220 --levels easy medium hard `
  --epochs 30 --batch-size 8192 --lift-delta-m 0.15 `
  --dagger-rounds 2 --dagger-epochs 10 `
  --dagger-initial-teacher-probability 0.5 `
  --output-dir results\ppo\bc_dagger2_adaptive_contact_vertical_ik_abs_1024env

python scripts\train_ppo.py --headless --device cuda:0 `
  --config configs\ppo_128step_bc.yaml --action-space ik_abs `
  --action-noise-std 0.005 --learning-rate 0.000001 `
  --learning-rate-schedule fixed `
  --resume-checkpoint results\ppo\bc_dagger2_adaptive_contact_vertical_ik_abs_1024env\model_bc.pt `
  --bc-teacher-checkpoint results\ppo\bc_dagger2_adaptive_contact_vertical_ik_abs_1024env\model_bc.pt `
  --bc-loss-coef 0.05 --bc-num-mini-batches 4 `
  --run-name ik_abs_dagger2_ppo_bc_128step_1024env_100iter

python scripts\benchmark_ppo_checkpoints.py `
  --run-dir results\ppo\ik_abs_dagger2_ppo_bc_128step_1024env_100iter `
  --iterations 20 40 60 80 100 --levels easy medium hard `
  --num-envs 128 --episodes-per-level 128 --max-steps 600 `
  --action-space ik_abs --device cuda:0

python scripts\evaluate_ppo.py --headless --device cuda:0 --action-space ik_abs `
  --checkpoint results\ppo\ik_abs_dagger2_ppo_bc_128step_1024env_100iter\model_100.pt `
  --num-envs 128 --episodes-per-level 256 --max-steps 600 `
  --output-dir results\ppo_evaluation_dagger2_ppo_bc_128step_100iter_final_256
```

若要复现 DAgger 基线，只需把评测命令中的 checkpoint 和输出目录替换为：

```powershell
--checkpoint results\ppo\bc_dagger2_adaptive_contact_vertical_ik_abs_1024env\model_bc.pt `
--output-dir results\ppo_evaluation_bc_dagger2_adaptive_contact_vertical
```

## 剩余瓶颈

- 最佳 PPO 的 Medium 仍有 8/256 个 lift failure；Hard 有 25 个 lift failure、1 个 object drop 和 1 个 timeout。
- 训练约第 80 iteration 后，官方 action-rate/joint-velocity curriculum 将权重提升到 `-0.1`；本次后期性能继续提高，但后续应单独消融该 curriculum。
- 当前接触信号是基于夹爪宽度和几何距离的代理量；真实接触力传感器可进一步提高长尾抓取判定质量。
