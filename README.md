# Isaac Lab Robot Manipulation Data Engine

面向求职展示的机器人操作仿真、并行数据生成与自动评测项目。

当前完成阶段：Phase 0 至 Phase 11；Phase 11 已包含旧 joint-position 负基线，以及通过 Scripted IK 示范热启动和保守 PPO 微调得到的正式 10 cm 抬升策略。

## Phase 1 场景

场景包含：

- Franka Emika Panda 机械臂与夹爪
- 带碰撞和摩擦参数的桌面
- 带质量、碰撞和语义标签的红色立方体
- 固定 RGB-D 相机
- 末端执行器坐标传感器
- 地面与环境光

激活 Phase 0 创建的环境后运行：

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\run_env.py
```

无窗口验收命令：

```powershell
python scripts\run_env.py --headless --steps 120
```

保存一帧 RGB 和归一化深度调试图：

```powershell
python scripts\run_env.py --headless --steps 120 --save_camera
```

Phase 1 不包含抓取控制、奖励函数、Domain Randomization 或 RL；这些内容将在后续阶段逐步加入。

## Phase 2 Pick-and-Lift

Phase 2 使用绝对 Differential IK 和向量化 scripted state machine 执行：

`rest -> approach above -> descend -> grasp -> lift`

无窗口运行：

```powershell
python scripts\run_pick_lift.py --headless --num_envs 1 --max_steps 600 --save_camera
```

实时窗口调试，并在成功后保持抓取姿态：

```powershell
python scripts\run_pick_lift.py --num_envs 1 --max_steps 600 --realtime --keep_open
```

更轻量的慢速 GUI 演示（不加载 RGB-D 传感器）：

```powershell
python scripts\run_pick_lift.py --num_envs 1 --max_steps 600 --realtime --realtime_speed 0.4 --rendering_mode performance
```

需要在 GUI 调试时同时读取 RGB-D，可额外传入 `--enable_cameras`。

Phase 2 的实现、观测定义和实测结果见 `docs/phase2_pick_and_lift.md`。

## Phase 3 单环境数据采集

采集一个完整的状态轨迹 episode：

```powershell
python scripts\collect_data.py --headless --episodes 1 --max_steps 600
```

输出位于 `dataset/episode_XXXXXX/`，包括压缩的 `trajectory.npz` 和自描述的 `metadata.json`；数据集根目录同时维护 `dataset_metadata.json`。成功与失败 episode 都会保存并自动回读校验。

Phase 3 的数据 schema、shape 和实测结果见 `docs/phase3_single_environment_collection.md`。

## Phase 4 Domain Randomization

在每个 episode reset 后随机化物体初始位置/朝向、质量、摩擦，相机外参以及环境光，并将采样值和仿真实际回读值保存到 `domain_params.json`：

```powershell
python scripts\collect_data.py --headless --episodes 3 --max_steps 600 `
  --output_dir dataset\randomized --randomization_config configs\randomization.yaml
```

如需同时创建 RGB-D 相机并验证随机外参确实写入 USD 场景，添加 `--enable_cameras`。Phase 4 仍只保存低维轨迹；图像数据将在 Phase 10 接入。

随机范围、可复现机制、数据格式与 GPU 实测结果见 `docs/phase4_domain_randomization.md`。

## Phase 5 Vectorized Parallel Simulation

使用 128 个 GPU 并行环境生成一批随机化轨迹：

```powershell
python scripts\collect_data.py --headless --num-envs 128 --episodes 1 --max_steps 600 `
  --output_dir dataset\parallel_128 --randomization_config configs\randomization.yaml
```

`--episodes` 表示并行 batch 数，因此上述命令写出 128 个独立 episode。每条轨迹拥有自己的成功状态、长度、随机参数和 manifest 索引。

运行单个规模的纯仿真吞吐基准：

```powershell
python scripts\run_parallel_benchmark.py --headless --num-envs 128
```

结果自动更新到 `results/parallel_rollout.csv`。并行设计、PhysX 容量修正、指标定义和 1–8192 环境实测数据见 `docs/phase5_vectorized_parallel_simulation.md`。

## Phase 6 Dataset Export

将 Raw NPZ/JSON episodes 导出为可训练的分片 HDF5 数据集：

```powershell
python scripts\export_dataset.py `
  --input-dir dataset\phase5_validation_128 `
  --output-dir exports\pick_lift_v1 `
  --episodes-per-shard 64
```

导出器生成 `index.csv`、固定 Train/Validation/Test split、数据集统计、shard SHA-256 和逐 tensor 源数据对比报告。中断后可添加 `--resume`，已完成的数据集则只重新校验，不会覆盖。

读取训练 split：

```python
from isaac_lab_data_engine.data_collection.hdf5_dataset import HDF5EpisodeDataset

dataset = HDF5EpisodeDataset("exports/pick_lift_v1", split="train")
sample = dataset[0]
q = sample["trajectory"]["q"]
```

格式、完整性策略和 128-episode 实测结果见 `docs/phase6_dataset_export.md`。

## Phase 7 Benchmark

使用固定 seed 对 scripted Pick-and-Lift baseline 运行 Easy / Medium / Hard 在线评测：

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\run_benchmark.py --headless --config configs\benchmark.yaml
```

默认每个难度运行 256 个 episode，共 768 条结果。正式结果为 Easy 100.00%、Medium 99.61%、Hard 93.75%；Hard 的 16 次失败由 15 次 lift failure 和 1 次 approach failure 构成。完整复跑后逐 episode CSV 的 SHA-256 完全一致。

汇总、逐 episode 数据和自描述报告位于 `results/benchmark/`。评测协议、难度范围、失败定义及复现证明见 `docs/phase7_benchmark.md`。

## Phase 8 Failure Analysis

对 Hard Benchmark 的失败率与摩擦、物体距离、质量和朝向进行自动分析：

```powershell
python scripts\analyze_failure.py
```

分析器输出分箱 CSV、Wilson 95% 区间、失败明细、JSON 报告和 PNG 图。当前 256 条 Hard episode 中有 16 次失败；最远距离分箱失败率 20.69%，最低质量分箱 15.79%，正向大 yaw 分箱 14.89%，最低摩擦分箱 11.36%。这些结果是带不确定性的描述性关联，将作为 Phase 9 failure-aware sampling 的输入。

统计定义、完整结果和局限性见 `docs/phase8_failure_analysis.md`。

## Phase 9 Failure-aware Sampling

运行完整的评测驱动数据生成闭环：

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\run_data_loop.py
```

系统从 Phase 8 的真实失败联合参数分布附近采样 70%，同时保留 30% Uniform Hard exploration。在同样的 256 episode 预算下，失败/困难样本从 16 条增加到 46 条，产出提高 2.875 倍；最远距离高风险区间覆盖率从 11.33% 提高到 32.81%。Round 2 固定种子完整复跑的逐 episode SHA-256 完全一致。

采样算法、审计机制、真实 GPU 结果和正确解释见 `docs/phase9_failure_aware_sampling.md`。

## Phase 10 Camera RGB-D Data

同步采集 50 Hz 机器人轨迹和可配置频率的 RGB-D/标定/6DoF 标签：

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\collect_data.py --headless --enable_cameras `
  --num-envs 1 --episodes 1 --camera_frame_interval 5 `
  --output_dir dataset\phase10_rgbd_validation `
  --randomization_config configs\randomization_medium.yaml
```

真实验证 episode 包含 120 条状态和 25 帧 320×240 RGB-D，并同步保存 K、world→camera 外参、camera/object/end-effector 世界位姿和 object-in-camera 6DoF。多模态数据已接入 HDF5 exporter 和 Reader，逐 tensor round-trip 校验通过。

完整 schema、坐标系约定、真实图像结果和性能边界见 `docs/phase10_rgbd_data.md`。

## Phase 11 Optional PPO

复用 Isaac Lab 官方 RSL-RL PPO 配置。当前方案使用 57 维状态观测、absolute-IK 动作、自适应几何/夹持阶段、原地垂直抬升示教和两轮 DAgger，并使用 Phase 7 相同协议独立评测：

```powershell
python scripts\train_bc_warmstart.py --headless --num-envs 1024 --horizon 220 `
  --epochs 30 --lift-delta-m 0.15 --dagger-rounds 2 --dagger-epochs 10 `
  --output-dir results\ppo\bc_dagger2_adaptive_contact_vertical_ik_abs_1024env
python scripts\train_ppo.py --headless --action-space ik_abs --num-envs 1024 `
  --iterations 20 --action-noise-std 0.005 --learning-rate 0.000001 `
  --resume-checkpoint results\ppo\bc_dagger2_adaptive_contact_vertical_ik_abs_1024env\model_bc.pt `
  --run-name ik_abs_dagger2_adaptive_vertical_ppo_1024env_20iter
python scripts\evaluate_ppo.py --headless --action-space ik_abs `
  --checkpoint results\ppo\ik_abs_dagger2_adaptive_vertical_ppo_1024env_20iter\model_20.pt `
  --episodes-per-level 256 --max-steps 600 `
  --output-dir results\ppo_evaluation_dagger2_adaptive_vertical_ppo_final
python scripts\plot_ppo_comparison.py
```

正式判据为“桌面上方 10 cm、闭爪、物体仍被持有、连续 10 帧”。当前最佳 DAgger checkpoint 为 Easy 100.00%、Medium 77.73%、Hard 73.83%；完成 20 次微调的 PPO checkpoint 为 100.00%、77.73%、57.42%。旧 PPO 的 100.00% / 32.42% / 10.16% 和 joint-position 三档 0% 仍作为消融基线保留。训练策略、动作空间、失败分析和完整结果见 `docs/phase11_ppo.md`。
