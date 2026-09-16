# Phase 11：PPO 正式 10 cm 抬升

## 当前结论

策略按照与 Phase 7 相同的正式判据评测：物体中心高于桌面 10 cm、夹爪闭合、末端与物体距离小于 8 cm，并连续保持 10 个仿真步。每个难度独立运行 256 个 episode、最多 600 步。

| 策略 | Easy | Medium | Hard |
|---|---:|---:|---:|
| 旧 BC + PPO | 100.00% | 32.42% | 10.16% |
| 自适应 BC | 100.00% | 31.25% | 10.55% |
| 自适应 BC + 2 轮 DAgger（当前最佳） | **100.00%** | **77.73%** | **73.83%** |
| DAgger + 20 次保守 PPO | 100.00% | 77.73% | 57.42% |

当前综合 Benchmark 最优 checkpoint：

`results/ppo/bc_dagger2_adaptive_contact_vertical_ik_abs_1024env/model_bc.pt`

完成 PPO 微调的 checkpoint：

`results/ppo/ik_abs_dagger2_adaptive_vertical_ppo_1024env_20iter/model_20.pt`

PPO 的 Hard 成功率比其 DAgger 初始化低，因此项目保留两者并如实报告，不用 PPO 结果覆盖当前最佳策略。

## 本轮修改

1. 抬升目标不再使用固定中心命令 `(0.55, 0, z)`，而是固定在每个物体抓取前的 x/y 锚点，仅将 z 提高 0.15 m。
2. 删除固定目标位置观测及横向目标跟踪奖励，训练目标与正式“原地抬升 10 cm”判据一致。
3. 用几何关系和夹持反馈产生五阶段 one-hot：rest、approach、descend、grasp、lift；下降阶段会等待水平对齐，不再按固定时钟强制跳转。
4. 新增 6 维抓取状态：水平距离、垂直差、三维距离、夹爪平均位置、夹爪平均速度、接触/持有代理。
5. 接触代理要求夹爪位置位于 0.5–2.5 cm 且末端距离小于 8 cm，从而区分“夹住物体”和“空抓完全闭合”。
6. 策略观测为 57 维；旧的 53/56/59 维 checkpoint 与当前网络不兼容。
7. 新增采集诊断：达到正式 10 cm 的环境数、平均最大抬升、最大横向漂移和最高点横向偏移。
8. 两轮 DAgger 将教师标签扩展到策略实际访问的偏离状态，解决了离线 BC 误差很低但闭环超时严重的问题。

## 数据与训练规模

- 初始并行示教：1024 个环境，Easy/Medium/Hard 共 675,840 samples。
- 初始 BC 验证集位置 MAE：2.00 mm；夹爪方向准确率：100%。
- 两轮 DAgger 后聚合数据：2,027,520 samples。
- PPO 微调：1024 个环境、20 iterations、491,520 transitions、学习率 `1e-6`、动作噪声标准差 `0.005`。

## 训练与评测命令

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"

python scripts\train_bc_warmstart.py --headless --device cuda:0 `
  --num-envs 1024 --horizon 220 --levels easy medium hard `
  --epochs 30 --batch-size 8192 --lift-delta-m 0.15 `
  --dagger-rounds 2 --dagger-epochs 10 `
  --dagger-initial-teacher-probability 0.5 `
  --output-dir results\ppo\bc_dagger2_adaptive_contact_vertical_ik_abs_1024env

python scripts\train_ppo.py --headless --device cuda:0 --action-space ik_abs `
  --num-envs 1024 --iterations 20 `
  --action-noise-std 0.005 --learning-rate 0.000001 `
  --resume-checkpoint results\ppo\bc_dagger2_adaptive_contact_vertical_ik_abs_1024env\model_bc.pt `
  --run-name ik_abs_dagger2_adaptive_vertical_ppo_1024env_20iter

python scripts\evaluate_ppo.py --headless --device cuda:0 --action-space ik_abs `
  --checkpoint results\ppo\ik_abs_dagger2_adaptive_vertical_ppo_1024env_20iter\model_20.pt `
  --num-envs 128 --episodes-per-level 256 --max-steps 600 `
  --output-dir results\ppo_evaluation_dagger2_adaptive_vertical_ppo_final
```

若要复现当前最佳 DAgger 策略，只需把评测命令中的 checkpoint 和输出目录替换为：

```powershell
--checkpoint results\ppo\bc_dagger2_adaptive_contact_vertical_ik_abs_1024env\model_bc.pt `
--output-dir results\ppo_evaluation_bc_dagger2_adaptive_contact_vertical
```

## 剩余瓶颈

- DAgger 的 Medium 57 个失败均为 lift failure；Hard 为 49 个 lift failure 和 18 个 timeout。
- PPO 的 Hard 退化说明纯 on-policy 更新仍会产生灾难性遗忘。下一步适合加入 PPO + BC auxiliary loss（或 DAPG）以及自动按正式 Benchmark 选择 checkpoint。
- 当前接触信号是基于夹爪宽度和几何距离的代理量；真实接触力传感器可进一步提高长尾抓取判定质量。
