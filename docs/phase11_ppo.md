# Phase 11：可选 PPO 基线

## 结论

Phase 11 已跑通 RSL-RL PPO 的训练、断点续训、检查点加载和 Easy / Medium / Hard 独立评测链路，但当前 PPO 策略没有通过项目的正式成功标准。该负结果被保留，不以训练 reward 代替任务成功率。

正式评测使用与 Phase 7 完全相同的判据：物体高于桌面 10 cm、夹爪闭合、末端与物体距离小于 8 cm，并连续保持 10 个仿真步。每档 256 个 episode，最大 600 步。

| 难度 | Scripted IK | PPO `model_500.pt` | PPO 主要失败 |
|---|---:|---:|---|
| Easy | 100.00% | 0.00% | 256 lift failure |
| Medium | 99.61% | 0.00% | 256 lift failure |
| Hard | 93.75% | 0.00% | 253 lift failure，3 object drop |

这说明 PPO 已能靠近物体并闭爪，但未学会把物体稳定提升到 10 cm。对求职展示而言，当前可真实表述为“建立了可复现的 PPO baseline 和同协议评测，并定位了接触阶段的策略瓶颈”，不能表述为“PPO 完成抓取”。

## 实现

- 后端：`rsl-rl-lib==3.1.2`，复用 Isaac Lab 2.3.2 的 `LiftCubePPORunnerCfg`。
- PPO 专用环境：`PickLiftPPOEnvCfg`。Scripted baseline 继续使用 absolute differential IK；PPO 使用官方配置所对应的 joint-position action space。
- 观测：53 维低维状态，包括关节、物体、目标、上一动作、末端位姿和相对位姿。
- Actor/Critic：`256 → 128 → 64` MLP，8 维动作。
- 训练并行度：1024 个环境，24 steps/env/update。
- 检查点：`results/ppo/jointpos_shaped_height_1024env_1000iter/model_500.pt`。
- 评测产物：`results/ppo_evaluation_shaped/`。

训练脚本支持中断恢复；Windows 上必须在 Kit 启动前首次导入 RSL-RL/tensordict，否则 `tensordict 0.14.2` 可能在原生扩展中触发 access violation。该顺序已固化在训练和评测脚本中。

## 训练诊断

最初误将官方 PPO config 与 IK-Abs action space 组合。Isaac Lab 本地注册信息表明，RSL-RL entry point 只绑定到 `Isaac-Lift-Cube-Franka-v0` 的 joint-position 环境，IK-Abs 环境没有 RL entry point。因此新增独立 PPO 环境，避免改变已验证的 scripted IK 链路。

无 shaping 的 joint-position 实验运行至约 iteration 700、1740 万 transitions，始终没有 lifting reward，策略收敛到“靠近但不抬升”。项目的 10 cm 正式标准显著严格于上游默认 lift reward，且接触后的抬升信号稀疏。

随后加入两个明确记录的训练 shaping term：

1. `grasp_shaping`：只在末端靠近物体时奖励闭爪；
2. `height_progress`：只奖励物体真实高度从初始中心 z=0.48 m 向正式目标 z=0.55 m 增加。

同时将训练期二值 lift gate 设为 z=0.50 m（真实抬升 2 cm），正式评测仍保持 z=0.55 m。最终 checkpoint 由约 1230 万累计 transitions 构成。Shaping 使三档评测都达到“held but not lifted”的 lift-failure 阶段，但没有转化为 10 cm 成功。

## 命令

新训练：

```powershell
python scripts\train_ppo.py --headless --iterations 1500 --num-envs 1024 `
  --run-name jointpos_shaped_height_1024env_1500iter
```

断点续训：

```powershell
python scripts\train_ppo.py --headless --iterations 500 --num-envs 1024 `
  --run-name resumed_run --resume-checkpoint results\ppo\some_run\model_999.pt
```

正式评测与对照图：

```powershell
python scripts\evaluate_ppo.py --headless
python scripts\plot_ppo_comparison.py
```

## 下一步（不计入已完成结果）

- 用 scripted IK 轨迹做行为克隆预训练，再用 PPO fine-tune；
- 引入接触传感器，区分“闭爪靠近”和“真实双指接触”；
- 采用 reach / grasp / lift 分阶段 curriculum；
- 先在上游原始 Franka Lift 环境复现官方 PPO 曲线，再逐项迁移项目几何和严格成功标准。
