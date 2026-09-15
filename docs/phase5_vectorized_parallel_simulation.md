# Phase 5：Vectorized Parallel Simulation

## 结论

Phase 5 已完成。Pick-and-Lift 控制、成功判定、轨迹缓存和 Domain Randomization 已从单环境扩展到 Isaac Lab GPU vectorized environment。单次 `env.step()` 可以同时推进 N 个 Franka 场景，再把批量张量一次性传回 CPU 并拆分成可变长度 episode。

RTX 4060 Laptop GPU 已完成 1、16、64、128、256 环境的规定测试，并在显存允许的情况下继续验证到 8192 环境。另以 128 环境完成了真实随机化数据生成和逐 episode 校验。

## 并行数据采集

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\collect_data.py --headless --num-envs 128 --episodes 1 --max_steps 600 `
  --output_dir dataset\parallel_128 --randomization_config configs\randomization.yaml
```

参数含义：

- `--num-envs`：同一个 Isaac Lab scene 中并行推进的环境数；也兼容 `--num_envs`。
- `--episodes`：同步 rollout batch 数，不是最终文件数。
- 最终 episode 数：`num_envs × episodes`。
- 不传 `--randomization_config` 时生成固定场景 baseline。
- Phase 5 默认不启用相机，以隔离物理并行能力；RGB-D 批量导出属于 Phase 10。

## 并行 episode 生命周期

每个环境分别维护：

- scripted controller state 和 elapsed time；
- success streak、success/done mask；
- completion step 与 failure reason；
- 初始、最大和结束物体高度；
- 独立 `EpisodeBuffer` 与 Domain 参数。

每步先在 GPU 上向量化计算动作和成功条件，再对 `q/dq/eef_pose/object_pose/action/reward/state` 各执行一次批量 GPU→CPU 传输。已完成环境停止写入轨迹，并保持当前末端姿态，未完成环境继续运行。因此同一 batch 内 episode 长度可以不同。

每条 `metadata.json` 新增：

- `parallel_num_envs`
- `parallel_batch_index`
- `parallel_env_index`

这些字段也进入 `dataset_metadata.json` 的 episode 索引。

## 并行 Domain Randomization

物体位姿、质量、惯量、摩擦以及相机位姿均按环境独立采样和批量写入。每个环境仍保存自己的 `domain_params.json` 与 PhysX/USD 回读值。

当前场景只有一个 `/World/Light` DomeLight，因此同一 batch 共享一次光照采样；`lighting.shared_across_batch=true` 明确记录这一约束。不同 batch 会再次采样。这样避免把全局 USD 属性错误描述为每环境独立随机量。

## PhysX 大规模碰撞容量

Isaac Lab 官方 Lift task 将 `gpu_total_aggregate_pairs_capacity` 设为 16384，这对常规训练规模足够，但 6144 环境实测需要至少 18432。容量不足不会直接 OOM，而会丢失碰撞交互：控制器虽然进入 LIFT，物体却无法被可靠抓起。

项目按 `next_power_of_two(max(16384, 4 × num_envs))` 自动配置该容量。6144/8192 环境使用 32768，既消除碰撞丢失，又避免直接使用 Isaac Lab 通用默认值 `2**21` 带来的不必要性能开销。最终摘要会记录实际容量，便于复现实验。

## 吞吐基准

单个规模运行：

```powershell
python scripts\run_parallel_benchmark.py --headless --num-envs 128 `
  --max_steps 600 --warmup_steps 20
```

按原定规模完成 sweep：

```powershell
1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 6144, 8192 | ForEach-Object {
  python scripts\run_parallel_benchmark.py --headless --num-envs $_
}
```

脚本关闭相机和语义渲染，warm-up 后计时，只测物理、观测、控制器和成功判定，不含 Isaac Sim 启动及数据压缩写盘。每个规模的最新结果会更新到 `results/parallel_rollout.csv`。

指标定义：

- `simulation_fps`：每环境执行的 PhysX step 数除以墙钟时间；当前每个控制步对应 2 个 PhysX step。
- `environment_steps_per_second`：`num_envs × vector control steps / wall time`。
- `episodes_per_minute`：本批环境数按 rollout 墙钟时间折算的完成能力。
- `gpu_memory_used_mb`：rollout 结束时通过 CUDA `mem_get_info` 获取的整卡已用显存快照，不是峰值。

## GPU 实测结果

测试日期：2026-09-16。硬件为 RTX 4060 Laptop GPU 8 GB；Isaac Lab 2.3.2、Isaac Sim 5.1.0、Torch 2.7.0+cu128。固定场景、10 步 warm-up，每次抓取在 120 个控制步完成。

| 环境数 | Physics FPS | Environment steps/s | Episodes/min | 显存快照 | 成功率 |
|---:|---:|---:|---:|---:|---:|
| 1 | 68.56 | 34.28 | 17.14 | 2337.71 MB | 100% |
| 16 | 67.00 | 535.98 | 267.99 | 2301.71 MB | 100% |
| 64 | 69.20 | 2214.45 | 1107.23 | 2301.71 MB | 100% |
| 128 | 69.12 | 4423.88 | 2211.94 | 2301.71 MB | 100% |
| 256 | 72.10 | 9228.33 | 4614.16 | 2367.71 MB | 100% |
| 512 | 72.76 | 18626.62 | 9313.31 | 2435.71 MB | 100% |
| 1024 | 67.97 | 34801.15 | 17400.57 | 2505.71 MB | 100% |
| 2048 | 63.08 | 64592.50 | 32296.25 | 2779.71 MB | 100% |
| 4096 | 54.58 | 111777.91 | 55888.96 | 3183.71 MB | 100% |
| 6144 | 25.26 | 77596.94 | 38798.47 | 3641.71 MB | 100% |
| 8192 | 21.83 | 89432.57 | 44716.28 | 4045.71 MB | 100% |

1→2048 环境时聚合吞吐近似随环境数增长；4096 环境达到本轮最高吞吐约 111.8k steps/s。继续增加到 6144/8192 后，场景初始化和单步物理成本明显增加，聚合吞吐低于 4096。虽然 8192 环境仍只使用约 4.05 GB 显存并保持 100% 成功，但它不是本机当前任务的最佳吞吐点。

因此纯物理 rollout 推荐最多使用 4096 环境；实际轨迹生成考虑 CPU 缓冲和大量小文件写入，推荐从 128–512 环境开始。由于每个基准点只运行一次短 rollout，本表用于当前机器的工程验收；正式性能报告应增加重复次数并报告均值和标准差。

## 128 环境数据生成验收

启用完整 Phase 4 随机化、逐步轨迹缓存和 NPZ/JSON 输出后的实测结果：

- 写出 episode：128
- 成功 / 失败：128 / 0
- Domain sidecar：128
- Domain sample index：1–128，全部唯一
- 质量命令/PhysX 回读最大误差：`2.22e-8 kg`
- 有效环境转移：15453
- rollout 墙钟时间：4.4993 s
- rollout 吞吐：3434.55 environment-steps/s
- 每条轨迹校验：`EPISODE_VALIDATION_OK`

该吞吐低于纯仿真基准是预期结果，因为它包含逐步 GPU→CPU 数据复制和内存轨迹拆分；计时仍不包含最后的 NPZ 压缩写盘。
