# Phase 7：Benchmark

## 结论

Phase 7 已完成。项目现在具备可复现的 Easy / Medium / Hard 在线评测协议、GPU 向量化执行器、逐 episode 失败归因，以及 CSV/JSON 报告。正式评测共运行 768 个 episode；固定种子完整复跑后，逐 episode CSV 的 SHA-256 完全一致。

本阶段评测的是 Phase 2 的低维状态 scripted baseline，不是视觉策略。相机和光照参数会被采样并记录，但不会实例化 RTX 相机，也不会影响控制输入。这一边界在报告中显式记录，避免把视觉随机化误写成视觉鲁棒性结果。

## 运行命令

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\run_benchmark.py --headless --config configs\benchmark.yaml
```

可用参数包括 `--levels easy medium hard`、`--num-envs`、`--episodes_per_level`、`--max_steps` 和 `--output_dir`。默认配置使用 128 个并行环境，每个难度运行 256 个 episode，最大 300 步。

## 成功判定

一个 episode 必须连续 10 个仿真步同时满足以下条件才算成功：

- 物体高度至少达到桌面上方 0.10 m；
- 两个夹爪关节位置均值不大于 0.030；
- 末端执行器与物体的距离不大于 0.08 m。

成功时间取首次满足连续保持条件的仿真时间。该定义比“某一帧达到目标高度”更严格，可排除瞬时弹起或立即掉落。

## 难度定义

| 难度 | Seed | 物体 XY | 质量 | 静摩擦 | 其他随机化 |
|---|---:|---|---|---|---|
| Easy | 7101 | 固定 `(0.55, 0.00)` m | 固定 0.10 kg | 固定 0.70 | yaw、相机、光照固定 |
| Medium | 7102 | X `[0.48, 0.64]` m，Y `[-0.14, 0.14]` m | `[0.05, 0.25]` kg | 固定 0.70 | yaw `[-180°, 180°]` |
| Hard | 7103 | X `[0.36, 0.78]` m，Y `[-0.30, 0.30]` m | `[0.02, 0.45]` kg | `[0.08, 0.95]` | yaw、动摩擦比、恢复系数、相机与光照均随机 |

动态摩擦系数由静摩擦系数乘以配置中的随机比例生成。每个难度 YAML 自带 seed，加载时会与 `benchmark.yaml` 交叉校验。

## 正式结果

运行设备为 NVIDIA GeForce RTX 4060 Laptop GPU；Python 3.11.16、Isaac Lab 2.3.2、Isaac Sim 5.1.0.0、Torch 2.7.0+cu128。

| 难度 | Episodes | 成功 | 成功率 | 成功平均完成时间 | 平均奖励 | 失败构成 |
|---|---:|---:|---:|---:|---:|---|
| Easy | 256 | 256 | 100.00% | 2.400 s | 6.1790 | 无 |
| Medium | 256 | 255 | 99.61% | 2.423 s | 6.0610 | 1 lift failure |
| Hard | 256 | 240 | 93.75% | 2.453 s | 5.4654 | 15 lift failure，1 approach failure |

结果表明当前控制器在固定条件和中等位置/质量变化下非常稳定；Hard 的主要短板是抓住物体后未能达到抬升判据。参数与失败的相关性留给 Phase 8 Failure Analysis 做系统分析，Phase 7 不从 16 个失败样本直接推断因果。

第二次正式复跑记录的环境步吞吐量为 Easy 1797.31、Medium 3126.94、Hard 3681.82 env steps/s。吞吐受系统负载和一次性缓存影响，不属于固定种子复现判据；Phase 5 的专用吞吐基准仍是性能对比的权威结果。

## 失败分类

每条失败轨迹只归入一个互斥类别：

- `approach_failure`：控制器未进入有效闭爪阶段；
- `grasp_failure`：已闭爪，但从未同时满足“夹爪闭合且物体接近末端”；
- `lift_failure`：曾判定抓持物体，但从未达到抬升高度；
- `object_drop`：曾达到抬升高度，最终未保持到成功；
- `timeout`、`terminated`：环境截断或终止。

报告不伪造碰撞次数。当前低维环境没有配置 ContactSensor，因此 `collision_count_available=false`，JSON 中同时给出缺失原因。

## 输出文件

```text
results/benchmark/
├── benchmark_results.csv          # 每个难度一行的汇总指标
├── benchmark_episodes.csv         # 768 条逐 episode 指标与随机参数
├── benchmark_report.json          # 配置快照、版本、硬件、哈希与汇总结果
├── reproducibility.json           # 两次正式运行的一致性证明
└── reproducibility_run.log        # 第二次完整运行日志
```

配置快照包含主配置及三个难度配置的 SHA-256，因此后续可以判断结果是否来自完全相同的协议。

## 可复现性验证

在同一 GPU、软件版本和配置下连续运行两次正式 Benchmark：

```text
first  SHA-256: 22599FC7705B222ADD958A2421C005FE5C95E472427A79EB10A96D88526AEFD7
second SHA-256: 22599FC7705B222ADD958A2421C005FE5C95E472427A79EB10A96D88526AEFD7
```

两次 `benchmark_episodes.csv` 逐字节一致，包括成功状态、失败类型、完成步数、奖励、状态指标和采样的 domain parameters。报告生成时间、wall time 和吞吐量属于运行时元数据，不要求相同。

## 自动测试

```powershell
python -m pytest tests\test_benchmark.py -q
```

测试覆盖配置顺序与 seed 校验、失败分类、聚合指标和原子 CSV/JSON 报告写入。
