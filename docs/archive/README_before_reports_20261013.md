# Isaac Lab 机器人操作仿真、并行数据生成与视觉控制

这是一个面向机器人算法与仿真岗位展示的 Isaac Lab 项目。项目围绕 Franka Panda 的 Pick-and-Lift 任务，构建了从物理仿真、Domain Randomization、GPU 并行数据生成，到 BC/DAgger/PPO 训练、分级 Benchmark、Failure Analysis，再到 RGB-D 视觉感知接入控制策略的完整流程。

当前项目已实现：

- 基于 Isaac Lab / Isaac Sim 的 Franka 抓取与原地垂直抬升任务；
- 物体位置、朝向、质量、摩擦，相机位姿和光照随机化；
- 向量化 GPU 并行 rollout 与 50 Hz 状态轨迹采集；
- 10 Hz RGB-D、相机标定和 6DoF 标签同步采集；
- NPZ/JSON 原始 episode 到分片 HDF5 数据集导出；
- Scripted IK、Behavior Cloning、DAgger 和 PPO 策略；
- Easy / Medium / Hard 分级 Benchmark 与 Failure Case 分析；
- RGB-D 关键点与深度残差网络；
- `RGB-D → 目标三维位置 → Scripted IK → 10 cm 抬升`视觉闭环；
- 将视觉预测替换 57 维学习策略中的物体真值观测。

## 当前结果

| 模块 | 结果 |
| --- | --- |
| Scripted IK Benchmark | Easy 100.00%，Medium 99.61%，Hard 93.75% |
| BC + 2 轮 DAgger | Easy 100.00%，Medium 77.73%，Hard 73.83% |
| 128-step PPO + BC auxiliary | Easy 100.00%，Medium 96.88%，Hard 89.45% |
| RGB-D 测试集三维误差 | 中位数 1.53 cm，平均 2.73 cm，90.80% 帧在 5 cm 内 |
| 视觉驱动 Scripted IK | Hard 随机场景 121 步完成，最大抬升 13.38 cm |
| 自动化测试 | 33 passed |

视觉闭环结果是单场景功能验收，不代表视觉 Benchmark 成功率。视觉观测已经接入 DAgger/PPO actor，但现有 actor 使用无噪声仿真状态训练，单 episode 视觉冒烟测试发生 timeout；后续需要进行 Visual DAgger 或带视觉噪声的策略微调。

## 系统流程

```text
Isaac Sim GPU Physics
        │
        ├── Scripted IK ──► 状态/RGB-D 数据采集 ──► HDF5 Dataset
        │                                           │
        │                                           ├── BC / DAgger ──► PPO
        │                                           │
        │                                           └── RGB-D 视觉模型
        │                                                     │
        └── Easy / Medium / Hard Benchmark ◄── 控制策略 ◄─────┘
                              │
                              └── Failure Analysis ──► Failure-aware Sampling
```

## 技术栈与验证环境

| 组件 | 版本 |
| --- | --- |
| 操作系统 | Windows 11 |
| Python | 3.11.16 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.2 |
| PyTorch | 2.7.0+cu128 |
| RSL-RL | 3.1.2 |
| h5py | 3.14.0 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8 GB VRAM |

项目固定使用 `h5py==3.14.0`，以匹配 Isaac Sim 5.1 GUI 插件使用的 HDF5 版本。

## 目录结构

```text
ai_BOT/
├── configs/                         # Domain Randomization、Benchmark、PPO 配置
├── dataset/                         # 原始 NPZ/JSON episode
├── exports/                         # 分片 HDF5 数据集
├── results/                         # checkpoint、Benchmark、图表和评测报告
├── scripts/                         # 采集、训练、评测与可视化入口
├── src/isaac_lab_data_engine/
│   ├── controllers/                 # Scripted IK 状态机
│   ├── data_collection/             # episode writer、HDF5 exporter/reader
│   ├── envs/                        # Isaac Lab 环境、观测和奖励
│   ├── benchmark/                   # 分级评测与报告
│   ├── analysis/                    # Failure Case 分析
│   ├── randomization/               # Domain Randomization
│   └── vision/                      # RGB-D 模型、数据集和策略观测适配器
├── tests/                           # 不启动仿真的单元测试
└── docs/                            # 各 Phase 的设计与结果
```

## 1. 环境准备

当前机器已经创建好 Conda 环境 `isaaclab23`。每次打开新的 PowerShell 后先执行：

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"
```

如果项目包尚未以 editable 模式安装：

```powershell
python -m pip install -e ".[ppo]"
```

快速检查 PyTorch 和 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

运行全部不依赖 Isaac Sim 窗口的测试：

```powershell
python -m pytest tests -q
```

> 本文命令面向 PowerShell。反引号只能作为 PowerShell 的换行续写符，并且必须放在行尾；在 Anaconda Prompt/CMD 中请把多行命令合并为一行，不要输入反引号。

## 2. 最快运行方式

### 2.1 查看状态真值驱动的 Scripted IK 抓取

```powershell
python scripts\run_pick_lift.py --num_envs 1 --max_steps 600 --realtime --keep_open
```

### 2.2 查看视觉模型驱动的抓取

```powershell
python scripts\run_visual_pick_lift.py --device cuda:0 --num-envs 1 --max-steps 600 --realtime --keep-open
```

该脚本会明确输出：

```text
model: RGBDKeypointPositionNet
controller_object_source: visual_prediction_only
simulator_object_truth: metrics_only
```

也就是说，物体真值只用于计算误差和成功判据，不参与控制动作生成。成功后窗口保持打开，关闭 Isaac Sim 即可结束。

无窗口验收：

```powershell
python scripts\run_visual_pick_lift.py --headless --device cuda:0 --num-envs 1 --max-steps 600
```

## 3. 从零复现完整流程

下面按项目 Phase 顺序执行。训练或导出脚本通常不覆盖已有的非空输出目录，重复实验时应使用新的目录名。

### Phase 1：验证场景

启动 Franka、桌面、方块、相机和灯光场景：

```powershell
python scripts\run_env.py
```

无窗口运行 120 步并保存相机图像：

```powershell
python scripts\run_env.py --headless --steps 120 --save_camera
```

### Phase 2：Scripted IK Pick-and-Lift

```powershell
python scripts\run_pick_lift.py --headless --num_envs 1 --max_steps 600 --save_camera
```

控制状态机为：

```text
rest → approach above → descend → grasp → lift
```

正式成功标准为：物体中心高于桌面 10 cm、夹爪闭合、物体仍靠近末端，并连续保持 10 个控制步。

### Phase 3：采集单环境状态轨迹

```powershell
python scripts\collect_data.py --headless --num-envs 1 --episodes 1 --max_steps 600 --output_dir dataset\state_single_rerun
```

每个 episode 输出：

- `trajectory.npz`：关节位置/速度、末端位姿、物体位姿、动作、奖励和时间戳；
- `metadata.json`：成功状态、失败类型、步数和数据 schema；
- `dataset_metadata.json`：数据集级 manifest。

### Phase 4：Domain Randomization

```powershell
python scripts\collect_data.py --headless --num-envs 8 --episodes 4 --max_steps 600 `
  --randomization_config configs\randomization_hard.yaml `
  --output_dir dataset\randomized_hard_rerun
```

Hard 配置随机化物体 x/y/yaw、质量、静动摩擦、恢复系数、相机位姿和光照。

### Phase 5：GPU 并行仿真与吞吐测试

采集一批 128 环境轨迹：

```powershell
python scripts\collect_data.py --headless --num-envs 128 --episodes 1 --max_steps 600 `
  --randomization_config configs\randomization_hard.yaml `
  --output_dir dataset\parallel_128_rerun
```

单独测量并行仿真吞吐：

```powershell
python scripts\run_parallel_benchmark.py --headless --num-envs 128 --max_steps 600 `
  --randomization_config configs\randomization_hard.yaml
```

### Phase 6：导出 HDF5 数据集

```powershell
python scripts\export_dataset.py `
  --input-dir dataset\parallel_128_rerun `
  --output-dir exports\pick_lift_state_rerun `
  --episodes-per-shard 32
```

导出结果包含分片 HDF5、`index.csv`、固定 Train/Validation/Test split、统计信息、SHA-256 和逐 tensor 回读校验。

### Phase 7：Easy / Medium / Hard Benchmark

先用少量 episode 做冒烟测试：

```powershell
python scripts\run_benchmark.py --headless --config configs\benchmark.yaml `
  --num-envs 8 --episodes_per_level 16 --max_steps 600 `
  --output_dir results\benchmark_smoke_rerun
```

正式评测：

```powershell
python scripts\run_benchmark.py --headless --config configs\benchmark.yaml
```

默认每个难度 256 个 episode，输出汇总 CSV、逐 episode CSV 和 JSON 报告。

### Phase 8：Failure Case 分析

```powershell
python scripts\analyze_failure.py `
  --input results\benchmark\benchmark_episodes.csv `
  --output-dir results\failure_analysis `
  --difficulty hard
```

分析内容包括失败类型、距离/质量/摩擦/yaw 分箱、Wilson 置信区间、失败样本明细和图表。

### Phase 9：Failure-aware Sampling

完整执行“失败分析 → 高风险区域重采样 → 新一轮 Benchmark → 两轮对比”：

```powershell
python scripts\run_data_loop.py --num-envs 128 --episodes 256
```

如果前两轮结果已经存在，只重新生成对比报告：

```powershell
python scripts\run_data_loop.py --reuse-existing
```

### Phase 10：RGB-D 数据采集与导出

当前视觉训练数据使用 8 个带 RTX 相机的并行环境、8 个同步 batch，共 64 个 episode：

```powershell
python scripts\collect_data.py --headless --device cuda:0 --enable_cameras `
  --num-envs 8 --episodes 8 --max_steps 180 --camera_frame_interval 5 `
  --randomization_config configs\randomization_hard.yaml `
  --output_dir dataset\rgbd_hard_rerun
```

状态控制频率为 50 Hz，`camera_frame_interval=5` 对应约 10 Hz RGB-D。

导出视觉 HDF5：

```powershell
python scripts\export_dataset.py `
  --input-dir dataset\rgbd_hard_rerun `
  --output-dir exports\pick_lift_rgbd_hard_rerun `
  --episodes-per-shard 16
```

可视化一个原始 RGB-D episode 的相机、物体和末端轨迹：

```powershell
python scripts\visualize_rgbd_episode.py `
  --episode-dir dataset\rgbd_hard_rerun\episode_000001 `
  --output results\rgbd_hard_rerun_trajectory.png
```

### Phase 11：BC、DAgger 和 PPO

#### 11.1 收集 Scripted IK 示范并训练 BC + DAgger

推荐先用小规模参数验证命令：

```powershell
python scripts\train_bc_warmstart.py --headless --device cuda:0 `
  --num-envs 64 --horizon 220 --levels easy medium hard `
  --epochs 3 --batch-size 2048 --lift-delta-m 0.15 `
  --dagger-rounds 1 --dagger-epochs 2 `
  --output-dir results\ppo\bc_dagger_smoke_rerun
```

正式训练配置：

```powershell
python scripts\train_bc_warmstart.py --headless --device cuda:0 `
  --num-envs 1024 --horizon 220 --levels easy medium hard `
  --epochs 30 --batch-size 8192 --lift-delta-m 0.15 `
  --dagger-rounds 2 --dagger-epochs 10 `
  --dagger-initial-teacher-probability 0.5 `
  --output-dir results\ppo\bc_dagger2_rerun
```

训练数据含 57 维策略观测和 8 维绝对 IK 动作。两轮 DAgger 会在 student 实际访问的偏离状态上重新请求 Scripted IK 教师动作，降低纯离线 BC 的闭环分布偏移。

#### 11.2 评测 BC/DAgger checkpoint

```powershell
python scripts\evaluate_ppo.py --headless --device cuda:0 --action-space ik_abs `
  --checkpoint results\ppo\bc_dagger2_rerun\model_bc.pt `
  --num-envs 128 --episodes-per-level 256 --max-steps 600 `
  --output-dir results\ppo_evaluation_bc_dagger2_rerun
```

#### 11.3 从 DAgger checkpoint 继续 PPO 微调

```powershell
python scripts\train_ppo.py --headless --device cuda:0 `
  --config configs\ppo_128step_bc.yaml --action-space ik_abs `
  --action-noise-std 0.005 --learning-rate 0.000001 `
  --learning-rate-schedule fixed `
  --resume-checkpoint results\ppo\bc_dagger2_adaptive_contact_vertical_ik_abs_1024env\model_bc.pt `
  --bc-teacher-checkpoint results\ppo\bc_dagger2_adaptive_contact_vertical_ik_abs_1024env\model_bc.pt `
  --bc-loss-coef 0.05 --bc-num-mini-batches 4 `
  --run-name ik_abs_dagger2_ppo_bc_128step_1024env_100iter
```

配置使用 1024 个环境、128 steps/env 和 100 iterations，共 13,107,200 transitions。BC 辅助项在 PPO 实际访问的状态上约束当前 actor 不要偏离冻结 DAgger teacher；BC warm start 默认不恢复其空 optimizer，避免 checkpoint 中的旧学习率覆盖本次设置。

#### 11.4 评测 PPO checkpoint

```powershell
python scripts\benchmark_ppo_checkpoints.py `
  --run-dir results\ppo\ik_abs_dagger2_ppo_bc_128step_1024env_100iter `
  --iterations 20 40 60 80 100 --levels easy medium hard `
  --num-envs 128 --episodes-per-level 128 --max-steps 600 `
  --action-space ik_abs --device cuda:0
```

脚本自动按 Hard、Medium、Easy 成功率选择 checkpoint。当前最佳 `model_100.pt` 在 256 episode/level 正式复核中达到 Easy 100%、Medium 96.88%、Hard 89.45%。

### Phase 12：训练视觉模型并接入控制

#### 12.1 训练 RGB-D 关键点模型

使用已导出的视觉数据：

```powershell
python scripts\train_visual_keypoint.py `
  --dataset-dir exports\pick_lift_rgbd_hard_rerun `
  --output-dir results\vision\rgbd_keypoint_rerun `
  --epochs 50 --batch-size 128 --device cuda:0
```

网络执行：

```text
RGB-D → 2D 物体中心热力图
      → RTX 表面深度 + 网络深度残差
      → 相机内参反投影
      → camera xyz
```

当前正式 checkpoint：

```text
results/vision/rgbd_keypoint_depth_residual_hard_v2/best_model.pt
```

#### 12.2 视觉模型接入 Scripted IK

```powershell
python scripts\run_visual_pick_lift.py --headless --device cuda:0 `
  --checkpoint results\vision\rgbd_keypoint_rerun\best_model.pt `
  --randomization-config configs\randomization_hard.yaml `
  --num-envs 1 --max-steps 600
```

GUI 演示：

```powershell
python scripts\run_visual_pick_lift.py --device cuda:0 `
  --checkpoint results\vision\rgbd_keypoint_rerun\best_model.pt `
  --num-envs 1 --max-steps 600 --realtime --keep-open
```

视觉控制使用已知工作空间掩码、REST 阶段多帧 EMA 和遮挡前目标锁定。仿真物体真值只用于输出定位误差和判断是否完成 10 cm 抬升。

#### 12.3 将视觉观测接入 DAgger/PPO actor

```powershell
python scripts\evaluate_ppo.py --headless --device cuda:0 `
  --checkpoint results\ppo\ik_abs_dagger2_ppo_bc_128step_1024env_100iter\model_100.pt `
  --agent-config results\ppo\ik_abs_dagger2_ppo_bc_128step_1024env_100iter\agent_config.yaml `
  --action-space ik_abs `
  --visual-checkpoint results\vision\rgbd_keypoint_depth_residual_hard_v2\best_model.pt `
  --num-envs 8 --episodes-per-level 32 --max-steps 600 `
  --output-dir results\grasp_mode_comparison_32\visual_ppo
```

`VisualPolicyObservationAdapter` 会用视觉预测重新计算以下 57 维观测项：

- object position；
- object pose 中的位置，未知朝向使用单位四元数；
- object position in end-effector frame；
- grasp state features；
- adaptive manipulation phase。

手指位置和速度直接保留机器人本体感知值；检测到夹持后，以末端执行器位移传播被遮挡物体的位置，使 PPO 能在不读取仿真物体真值的情况下观察抬升进度。视觉图像只在线推理，不参与本轮 PPO 训练，也不保存图像文件。

#### 12.4 三种抓取方案对比

纯视觉（RGB-D + Scripted IK）：

```powershell
python scripts\run_benchmark.py --headless --device cuda:0 `
  --visual-checkpoint results\vision\rgbd_keypoint_depth_residual_hard_v2\best_model.pt `
  --num-envs 8 --episodes_per_level 32 --max_steps 600 `
  --output_dir results\grasp_mode_comparison_32\visual_scripted_ik
```

状态 PPO 不带 `--visual-checkpoint`；视觉 + PPO 使用 12.3 的命令。三份报告完成后生成统一结果：

```powershell
python scripts\compare_grasp_modes.py `
  --visual-ik-report results\grasp_mode_comparison_32\visual_scripted_ik\benchmark_report.json `
  --state-ppo-report results\grasp_mode_comparison_32\state_ppo\benchmark_report.json `
  --visual-ppo-report results\grasp_mode_comparison_32\visual_ppo\benchmark_report.json `
  --output-dir results\grasp_mode_comparison_32\summary
```

32 episode/level 的当前结果：

| 控制方案 | Easy | Medium | Hard |
| --- | ---: | ---: | ---: |
| RGB-D 位姿 + Scripted IK | 100.00% | 28.13% | 40.63% |
| 仿真状态 + PPO | 100.00% | 96.88% | 93.75% |
| RGB-D 位姿 + PPO | 100.00% | 6.25% | 15.63% |

纯视觉 IK 的成功样本初始位姿误差均值在 Medium/Hard 分别为 1.51 cm/0.97 cm，失败样本分别为 4.47 cm/4.24 cm，说明当前主要瓶颈是视觉域随机化下的定位误差。视觉 + PPO 还受到状态训练到视觉测试的观测分布偏移影响；本轮按要求未用图像重新训练 PPO。

## 4. 已有 checkpoint 和结果

| 内容 | 路径 |
| --- | --- |
| DAgger 基线 | `results/ppo/bc_dagger2_adaptive_contact_vertical_ik_abs_1024env/model_bc.pt` |
| 当前最佳 PPO | `results/ppo/ik_abs_dagger2_ppo_bc_128step_1024env_100iter/model_100.pt` |
| PPO 最终 Benchmark | `results/ppo_evaluation_dagger2_ppo_bc_128step_100iter_final_256/benchmark_report.json` |
| RGB-D 视觉模型 | `results/vision/rgbd_keypoint_depth_residual_hard_v2/best_model.pt` |
| 视觉训练报告 | `results/vision/rgbd_keypoint_depth_residual_hard_v2/summary.json` |
| 视觉闭环报告 | `results/vision/visual_control_smoke_summary.json` |
| 三方案对比报告 | `results/grasp_mode_comparison_32/summary/grasp_mode_comparison.json` |
| Scripted IK Benchmark | `results/benchmark/` |
| Failure Analysis | `results/failure_analysis/` |

### VLA：1.5 cm 预抓取恢复数据补充（2026-10-10）

已完成严格恢复集 `dataset/vla_dagger_recovery_pregrasp15mm_tight_v7`：360 条连续轨迹、1,440 帧，每条 4 帧且包含 1 帧 learner-visited 状态。使用 v6 最佳 checkpoint `006000` 执行一次 VLA 动作，再由视觉 Scripted IK 提供相对末端的 `Δxyz` 恢复标签。预抓取点定义为锁定的 RGB-D 目标位置加 `[0, 0, 0.12]`；起始状态和 learner-visited 状态距该点均为 1–2 cm。

| 分区 | Easy | Medium | Hard | 总帧数 |
| --- | ---: | ---: | ---: | ---: |
| Train | 400 | 400 | 400 | 1,200 |
| Validation | 80 | 80 | 80 | 240 |

审计结果位于 `results/vla/dagger_recovery_v7/data_audit.json`：标签重算误差 0，无训练/验证场景键重叠，与旧 v6 场景键重叠 0；最低 RGB 标准差 62.93、动态范围 229、深度有效比例 0.4313。场景键使用 `(randomization_seed, sample_index)`。独立 LeRobot 导出位于 `exports/lerobot/pick_lift_vla_dagger_recovery_pregrasp15mm_tight_v7`。

保留原训练集并加入新 Train 分区后，`exports/lerobot/pick_lift_vla_visual_fsm_delta_tight_dagger15mm_v7/train` 包含 852 条轨迹、11,706 帧。恢复采样倍率为 4，全部恢复帧预计占抽样的 33.79%；Hard 与 LIFT 预计分别占 50.15% 和 18.49%。新恢复帧损失权重为 Easy/Medium ×2、Hard ×2.5；learner-visited 帧额外 ×2，Z 轴继续 ×3。采样计划为 `results/vla/sampling_plans/v7_train_tight_dagger15mm_sampling_plan.json`。

本轮微调已完成，共 6,000 steps、每 500 steps 保存，从 v6 `006000` 的原 LoRA adapter 开始并使用新 optimizer，学习率从 `5e-6` 衰减到 `5e-7`。已有训练目录不会被覆盖；复现实验需传入新的 `-RunName`。命令可直接用于 PowerShell 或 Anaconda Prompt：

训练入口会从 adapter-only checkpoint 的 `train_config.json.policy` 补齐缺失的 `config.json`，并在启动前实际解析策略配置；原 adapter 权重保持不变。已有失败日志会保留，重试写入带时间戳的新日志。已完成独立的两步 GPU 训练验证，结果位于 `results/vla/training/smolvla_tight_recovery_v7_startup_smoke_20261010`。Windows 上的 `torchcodec` 缺失提示表示自动使用 PyAV，可继续训练。

```text
conda activate lerobot-vla
cd C:\Users\yjh\Desktop\work_project\ai_BOT
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_smolvla_tight_recovery_8gb_v8.ps1 -ValidateOnly
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_smolvla_tight_recovery_8gb_v8.ps1 -RunName smolvla_tight_recovery_repro
```

训练后对新恢复验证集执行固定扫描：

```text
python scripts\scan_smolvla_checkpoints.py --checkpoint-root results\vla\training\smolvla_visual_fsm_delta_tight_dagger15mm_lora16_v7\checkpoints --dataset-root exports\lerobot\pick_lift_vla_dagger_recovery_pregrasp15mm_tight_v7\validation --repo-id local/isaac-pick-lift-vla-dagger-recovery-pregrasp15mm-tight-v7-validation --samples 0 --position-target-mode delta --threshold-position-mae-cm 0.7 --output results\vla\checkpoint_scans\smolvla_tight_recovery_v7_full240.json
```

这批样本针对 phase 1 的局部恢复，筛选会拒绝执行后离开 1–2 cm 的状态。它尚不能证明完整抓取改善，也不覆盖从远处进入预抓取点的所有偏移；微调后仍需检查原验证集的性能保持、零动作基线及 Isaac 闭环结果。

2026-10-10 已完成 500～6,000 的 12 个 checkpoint 固定扫描，使用全部 240 帧独立恢复验证数据、固定推理噪声种子 `20261008`。这里的位置误差是预测 `Δxyz` 与专家动作标签之间的平均欧氏距离，不等于闭环抓取成功率。

| 同一恢复验证集 | 平均位置误差（cm） | p90（cm） |
| --- | ---: | ---: |
| 微调前 v6 `006000` | 1.842 | 2.898 |
| 本轮最佳 v7 `001500` | 1.788 | 2.764 |
| 本轮最终 v7 `006000` | 1.793 | 2.769 |
| 零 `Δxyz` 基线 | 1.439 | — |

最佳与最终 checkpoint 差异很小；当前恢复误差仍高于零动作基线，未通过 0.7 cm 恢复门槛。单独评估 60 个 learner-visited 状态，误差从 2.078 cm 降至 2.031 cm，p90 从 2.968 cm 变为 2.994 cm。原任务独立验证集 1,256 帧的误差从 1.196 cm 降至 1.174 cm，未发现明显遗忘。

测试报告：`results/vla/checkpoint_scans/smolvla_tight_recovery_v7_full240.json`；同数据 v6 对照：`results/vla/checkpoint_scans/smolvla_v6_006000_on_tight_v7_full240.json`；learner 子集：`results/vla/dagger_recovery_v7/learner_visited_validation_comparison.json`；原任务保持测试：`results/vla/checkpoint_scans/smolvla_tight_v7_001500_original_validation_full1256.json`。

最佳 `001500` 随后完成 Isaac 闭环诊断：Easy / Medium / Hard 各 1 次、固定场景种子 7101 / 7102 / 7103、每次 300 步；CPU VLA 推理种子 `20261010`，GPU Isaac，execution horizon 1、action repeat 5、禁止 Scripted IK 回退。结果 0/3，均为 `approach_failure`，未闭合夹爪或抬升；最近末端到物体距离分别为 22.18 / 35.19 / 36.16 cm。180 次 VLA 请求均无错误，实际动作全部来自 VLA（保留安全限幅）。初始 RGB-D 位姿误差分别为 0.65 / 0.81 / 0.62 cm，相机质量门与连续稳定性检查均通过。该测试提示接近阶段控制仍未跑通，不能仅用局部恢复验证误差推断抓取性能；3 次诊断也不能作为正式统计成功率。报告位于 `results/vla/closed_loop_benchmark/tight_dagger15mm_v7_001500_pure_h1_smoke3/vla_benchmark_report.json`。

注意上述“微调前 v6”仍是已经微调的策略，并非原始预训练 SmolVLA。当前未找到原始基础模型的零微调闭环抓取报告；其配置为 6 维状态、三路相机、6 维动作，与本项目的 30 维状态、单路相机、8 维末端动作不同，需要先核对动作语义、单位与归一化并对齐接口，才能建立有效的未微调基线。

可视化当前最佳 VLA 闭环（在项目目录执行，PowerShell / CMD 均可）：

```text
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\demo_smolvla_tight_v7.ps1 -Level easy
```

此入口现默认启动 GPU VLA `001500` 服务（PyTorch 分配上限为整卡的 35%，不是对所有进程的显存预留）和 GPU Isaac GUI，使用 30 维视觉/阶段条件输入与 `Δxyz` 动作，禁止 Scripted IK 回退；固定场景运行 300 步后保存报告，自动退出窗口并停止本次服务，避免残留的保留窗口阻碍下一轮实验。显式 `-KeepOpen` 才保留最终画面；`-VlaDevice cpu` 可复现原 CPU 路径。`-Level medium` / `-Level hard` 可切换难度；每次结果和日志保存至新的带时间戳目录，原结果不覆盖。

2026-10-10 GPU 300 / 600 步对照已完成：相同 `001500` adapter SHA256、Easy 场景键 `easy:7101:000001`、物体/物理/相机/光照参数、控制参数及推理种子 `20261010`，每轮重启服务和 Isaac。未修改权重、精度设置、动作重复次数或成功判据。

| GPU VLA 控制预算 | 平均推理延迟 | 仿真时间 / 实际控制耗时（不含加载） | 最近末端—物体距离 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 300 步 | 497.6 ms | 6 s / 44.87 s | 28.40 cm | 接近失败 |
| 600 步 | 523.5 ms | 12 s / 91.89 s | 22.13 cm | 接近失败 |

共 180 次推理无错误，动作全部来自 VLA，均未夹持或抬升。VLA PyTorch 峰值分配约 906 MiB、缓存预留 934 MiB；约每秒一次的整卡 `nvidia-smi` 采样，300 / 600 步最低空闲分别为 899 / 995 MiB，未出现 OOM。采样可能漏掉瞬时峰值，当前配置有余量但仍偏紧；Windows WDDM 下 CUDA `mem_get_info` 与整卡专用显存读数可能不同，整卡余量以 `nvidia-smi` 为准。GPU 推理较原 GUI CPU 的 2.63 s 约快 5 倍，但仍未达到每次动作 0.1 s 的实时节奏。

本对照仅覆盖一个固定 Easy 场景，不能代表分级成功率。增加步数使距离缩小，但不足以完成抓取；当前失败不能解释为 RPC 超时。相同种子不保证 CPU/CUDA 噪声相同，也不保证两次 RGB 渲染逐像素一致，不能把 CPU/GPU 轨迹差异直接解释为策略优劣。汇总与一致性核验：`results/vla/dagger_recovery_v7/gpu_300_600_comparison_20261010.json`。

```text
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\demo_smolvla_tight_v7.ps1 -Level easy -VlaDevice cuda -MaxSteps 300 -NoKeepOpen
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\demo_smolvla_tight_v7.ps1 -Level easy -VlaDevice cuda -MaxSteps 600 -NoKeepOpen
```

采集时使用 CPU VLA 推理、GPU Isaac 渲染和 RGB-D 质量门，避免 8 GB 显存竞争。相机就绪使用 180 秒真实时间上限；下一轮启动时，RGB-D 合格但视觉位姿持续无效 30 秒会提前拒绝该场景。本次正式数据使用的是原 180 秒等待策略。

## 5. 常见问题

### 命令提示 `unrecognized arguments: \``

反引号是 PowerShell 的续行符，不是脚本参数。它必须放在 PowerShell 行尾；如果使用 Anaconda Prompt/CMD，请将整条命令写在一行。

### GUI 首次加载很慢

Isaac Sim 需要初始化 Kit、RTX shader、USD 资源和相机扩展。第一次通常最慢；后续启动会利用缓存。纯状态训练和大规模并行 rollout 默认关闭相机，以减少 RTX 开销。

### 8 GB 显存能否使用 1024 个并行环境

无相机的低维状态环境已经完成 1024 并行 BC/DAgger/PPO 实验。带 RGB-D 相机时显存和渲染开销显著增加，当前视觉数据采集使用 8 个相机环境，不应直接套用 1024 环境配置。

### 为什么视觉 DAgger/PPO 目前不如视觉 Scripted IK

现有 actor 在精确仿真物体状态上训练，而部署时输入包含视觉误差、遮挡、缺失朝向和时间相关噪声，产生 observation distribution shift。Scripted IK 对厘米级目标误差更容易加入显式约束；学习策略需要 Visual DAgger、观测噪声训练或端到端视觉策略微调。

## 真值教师、视觉学生：重新采集完整示范（v9）

此次流程停止并删除了旧 v8 视觉教师采集数据及其试采集/导出；更早的 v7 数据、模型和诊断日志保留。新采集不混入旧轨迹，不对旧轨迹事后重标。

教师从仿真真值生成 IK 抓取动作，学生仍使用 RGB、机器人状态、视觉估计位置/有效性和阶段编号，不接收真实目标坐标。原始 NPZ 单独保留教师目标和物体真值作为审计字段，LeRobot 模型特征不导出这些字段。训练阶段编号来自教师反馈；部署阶段编号来自视觉反馈，仍需检查阶段分布偏移。

保持 4 cm 的末端位置增量限幅、5 个 50 Hz 控制步保持、工作空间/夹爪安全限制。采集当场记录安全过滤后实际发送给 IK 的动作，将其相对同一时刻末端位置表示为 `Δxyz`；另存教师原始提案及 50 Hz 实际命令以验证因果对齐。只有实际抬升至少 10 cm、连续成功至少 10 步、五阶段齐全且 RGB-D 合格的完整轨迹才进入训练集。

默认 Easy / Medium / Hard 各收集 200 条成功轨迹，各划分 170/15/15 条 Train/Validation/Test。难度均衡采样，Z/抬升加权，使用不改变几何标签的图像增强。以下命令已包含自动导出、逐帧标签审计、6000 步 LoRA 微调和每 500 步的固定验证集扫描；使用新 RunTag 复现，不覆盖已有结果。

```text
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_vla_execution_aligned_v8.ps1 -RunTag truth_teacher_repro -Teacher truth
```

验证集动作位置误差通过约 2 cm 门槛后，自动运行独立固定种子的 Easy/Medium/Hard 视觉学生闭环评测，每级 10 次、每次 600 步，要求实际抬升 10 cm。此时不使用真值教师、真值阶段判断、真值夹爪距离或 Scripted IK 回退；真值只用于计算评测指标。Hard 包含相机、光照、物体/物理参数随机化；这是组合随机场景评测，不能单独归因于相机视角变化，也不等于真实机器人泛化。教师示范的成功率与离线动作误差均不能替代 VLA 闭环成功率。

当前正式批次的状态文件：`results/vla/execution_aligned_20261011_truth_v9/pipeline_status.json`；原始数据：`dataset/vla_execution_aligned_full_20261011_truth_v9`；导出：`exports/lerobot/pick_lift_execution_aligned_20261011_truth_v9`。目录仅在对应阶段启动后创建。

PowerShell 查看采集进度：

```powershell
Get-Content dataset\vla_execution_aligned_full_20261011_truth_v9\collection_summary.json
Get-Content results\vla\execution_aligned_20261011_truth_v9\collection.log -Tail 15 -Wait
```

## 原始 SmolVLA + 新 LoRA 对照实验（v10）

与已经完成的 v9 续训权重对比：从本地原始开源 SmolVLA 创建新 rank-16 LoRA，训练前核验基础参数冻结、37 个 B 矩阵零初始化、742,656 个可训练参数。正式训练不使用试跑权重，也不加载旧 LoRA；保留所有旧权重、数据与结果。

仅使用 truth-v9 审计通过的 510 条训练轨迹（27,338 帧），45 条验证轨迹与 45 条测试轨迹不参与梯度更新。保持 6,000 次更新、batch 1、训练种子 1000、同一采样计划、图像增强、Z 轴加权和学习率设置。每 500 步保存；两版各扫描 500/3000/6000，在相同 240 个验证帧上比较，主要比较固定 6000 步而不是事后挑选不同最佳点。

在项目目录执行（使用新 RunTag 复现实验，不覆盖已存在的运行）：

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_smolvla_fresh_truth_comparison_v10.ps1 -RunTag 20261011_v10

两版最终权重均通过约 2 cm 的验证门槛后，自动串行运行 Easy/Medium/Hard 每级 3 个相同场景，600 控制步、5 步保持、实际抬升 10 cm；不使用真值控制或 Scripted IK 回退。比较程序核对模型哈希、验证索引、训练配置和逐场景随机化参数。3 次/级只是小批量配对诊断，不是稳定成功率证明；相同种子也不保证训练或 RGB 渲染逐位一致。若门槛未通过，保存离线比较并停止进入闭环。

查看进度（PowerShell，Ctrl+C 仅停止查看日志）：

    Get-Content results\vla\fresh_lora_comparison_20261011_v10\pipeline_status.json
    Get-Content results\vla\training_logs\smolvla_execution_aligned_fresh_lora16_20261011_v10.log -Tail 15 -Wait

新权重保存于 results/vla/training/smolvla_execution_aligned_fresh_lora16_20261011_v10/checkpoints；对比报告保存于 results/vla/fresh_lora_comparison_20261011_v10/comparison.json。报告区分离线完成和闭环完成，不能把专家示范成功率当作学生模型结果。

### v10 未通过离线门槛时的诊断性闭环对照

默认自动训练流程仍保留 2 cm 门槛。仅在用户明确要求直接比较闭环时，以下入口记录豁免理由并对两版固定 6000 步权重进行测试；不会重新训练、修改验证结果或声称门槛通过。每版每级默认 3 次，共 18 次，保留此前离线报告，并在新目录中核对场景参数与模型哈希后输出闭环比较。

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_smolvla_paired_closed_loop_v10.ps1 -RunTag 20261011_v10 -EpisodesPerLevel 3

新 RunTag 可复现实验而不覆盖旧目录。上述正式诊断运行的状态位于 results/vla/fresh_lora_closed_loop_20261011_v10/pipeline_status.json；结果位于同目录 comparison.json。阶段 COMPLETE 前不能把离线报告当作闭环实验已经完成。只有本次启动的 Isaac/VLA 子进程会在结束时被关闭。

诊断对照使用独立端口（默认旧版 8768、新版 8769），并等待本次子进程退出。若某一版已完成而流程中断，可使用以下命令恢复；仅复用权重哈希和回合数核验通过的完整结果，不覆盖已保存的测试轨迹：

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_smolvla_paired_closed_loop_v10.ps1 -RunTag 20261011_v10 -EpisodesPerLevel 3 -Resume

### 固定权重的阶段／姿态诊断（v11）

使用旧版 v9 的 6000 步权重进行 2×2 对照：legacy FSM／滞回夹持反馈 FSM，分别配合模型姿态／固定俯视姿态。默认每组 Easy/Medium/Hard 各 3 次，共 36 次；相同场景参数、初始推理种子和成功判据，不重新训练，不修改旧数据或模型。固定姿态仅替换阶段 1–4 的四元数，同次预测的 Δxyz 和夹爪值不变；闭环轨迹分叉后，后续预测不保证相同。

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_smolvla_phase_ablation_v11.ps1 -RunTag 20261011_v11 -EpisodesPerLevel 3

新阶段机制是组合干预：2.5 cm 进入／3.5 cm 退出阈值、滞回与持续判定，加上非空夹爪宽度、速度和视觉距离组成的夹持代理；不读取物体真值，也不单纯靠超时推进。比较无法单独归因于其中一个改动，也不是稳定统计成功率证明。每个控制决策记录预抓取点距离、阶段切换、原始动作、保持目标、安全过滤和夹爪拦截；不额外保存图像。

    Get-Content results\vla\phase_ablation_20261011_v11\pipeline_status.json

逐场景记录在各组结果目录的 phase_diagnostics 中。完成后生成 results/vla/phase_ablation_20261011_v11/phase_ablation_report.json，核对权重、场景和干预范围后汇总配对成功／失败变化、phase 1 滞留、预抓取距离和夹爪拦截。结果须人工审阅后再决定是否统一训练／部署阶段判定及重新设计接口，不能仅凭推进了阶段就宣称抓取成功。

### 视觉初始化修复与 A 组补跑（v12）

v11 的 A 组 Medium/Hard 六回合未进入控制，而是视觉连续有效帧门未通过。修复改为真实仿真步推进、每步一次相机采样，并让质量检查与估计使用同一份 RGB-D 快照；重复帧不能填满连续窗口，新的无效帧仍会清空窗口。没有降低 RGB 标准差/动态范围、深度有效率或位姿稳定性门槛。

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\rerun_phase_ablation_a_v12.ps1 -RunTag 20261011_v12

已补跑六回合，均在 14 个预热控制步后取得 5 个连续有效帧；Hard 第 3 回合纯 VLA 抬升 13.21 cm。保留原 Easy 的 A 组结果后为 Easy 1/3、Medium 0/3、Hard 1/3，共 2/9。补充报告为 results/vla/phase_ablation_a_repair_20261011_v12/repaired_a_summary.json；原 v11 报告不覆盖。这不是重跑全部四组，不能声称像素、逐请求噪声或初始化协议完全一致。

阶段 2 审计在同目录 phase2_audit.json：4590 个训练帧全部为负 Z 位移、夹爪打开；归一化统计与训练导出一致，输入/输出动作统计一致。45 条验证轨迹的观察回放显示教师与部署阶段仍不一致；回放并不等于 VLA 闭环抓取。

### 相对视觉位置与 4 维动作实验（v13）

独立新接口使用 29 维状态：9 个关节位置、9 个速度、7 维末端位姿、3 维“锁定视觉估计位置减当前末端位置”、视觉有效位。相对向量使用机器人基座坐标轴，不是工具局部轴。模型不再输入教师阶段编号，也不输入真实物体位置；阶段只用于训练采样和诊断。输出为 Δxyz 与夹爪四维；控制适配层添加固定俯视四元数，再经原 4 cm 限幅、5 个控制步保持和夹爪安全规则执行。

流程先扫描旧模型分阶段验证误差，再用真值教师验证新动作执行链路（每级两个成功轨迹；不是 VLA 成功率），随后从原校验过的 600 条轨迹导出新特征版本。RGB 字节不改变，原执行动作的 xyz/夹爪直接选取、不重标；原 REST 姿态与新固定姿态不完全相同，故设置教师执行门。分区保持 train 510 / validation 45 / test 45，pilot 不混入训练，test 不用于选权重。

从原始开源 SmolVLA 新建 rank-16 LoRA，避免直接复用旧 8 维动作通道的适配权重；默认训练 6000 optimizer steps（不是 epochs），chunk=1 对齐每次只执行一个预测的部署路径，重算新输入/输出统计，保留光度增强和均衡采样。新实验显式采用 per_sample 损失归约，使 batch=1 时 Hard/LIFT 权重也能影响梯度；旧训练入口默认仍采用原归约。它同时改变接口、初始化、chunk 与损失归约，是组合实验，不是单因素因果证明。

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_vla_relative4d_v13.ps1 -RunTag 20261011_v13

流程拒绝覆盖同名目录。查看进度（PowerShell）：

    Get-Content results\vla\relative4d_20261011_v13\pipeline_status.json
    Get-Content results\vla\relative4d_20261011_v13_stdout.log -Tail 20 -Wait

若训练启动前的阶段失败，可用同一入口加 -Resume 复用完整教师、导出与验证结果；保留失败状态和日志快照，不支持覆盖已有训练目录或自动恢复 optimizer。当前 v13 已从修复后的采样步骤恢复，活动外层日志为 results/vla/relative4d_20261011_v13_resume_stdout.log；训练开始后直接跟踪训练日志：

    Get-Content results\vla\training_logs\smolvla_relative4d_fresh_lora16_20261011_v13.log -Tail 20 -Wait

新数据：exports/lerobot/pick_lift_relative4d_20261011_v13；新权重：results/vla/training/smolvla_relative4d_fresh_lora16_20261011_v13；训练日志在 results/vla/training_logs 的同名 .log。保存 interface_contract.json 并在部署时检查，防止同为 29 维但语义不同的状态混用。验证扫描固定 240 个 validation 帧；位置动作误差未达到 2 cm 时自动停止，不启动 Isaac 学生 Benchmark。通过后执行 Easy/Medium/Hard 各 3 回合纯 VLA 测试，要求真实抬升 10 cm，无专家回退。

### 30k 闭环诊断与闭合／抬升增强（v14）

v13 已训练至总计 30000 steps，固定验证 240 帧的位置动作误差为 2.332 cm、夹爪符号准确率 85.42%。仍未通过 2 cm 门槛。经用户明确要求，对 030000 权重单独运行 Easy/Medium 各 3 个回合（600 控制步、GPU 推理、无专家回退），报告保留门槛覆盖原因：

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\diagnose_vla_relative4d_30k_v14.ps1

结果位于 results/vla/relative4d_30k_diagnostic_20261012_v14。phase_diagnostics 是逐控制步记录，阶段不进入 29 维模型输入；夹爪 latch 表示安全控制器保持闭合，不能作为物体已夹住的证据。可用 scripts/summarize_vla_close_lift_diagnostic.py 加具体 benchmark 目录生成轨迹摘要。

完成这 6 个回合后，增强微调入口为：

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_vla_close_lift_v14.ps1

该入口要求诊断报告完整且模型 hash 一致。从 30k adapter 初始化独立 v14 训练，使用新的 AdamW 优化器，额外训练 6000 steps，学习率 3e-6→1e-6；v14 的 2000/4000/6000 checkpoint 对应累计 32k/34k/36k 参数更新。闭合和抬升采样权重分别从 1.5/2 提高为 3/4，保留其他阶段和 Easy/Medium/Hard 总采样质量均衡；夹爪维度 index=3 的 flow-matching 损失权重为 2，Z 权重为 3。没有新增图像、重标动作或增加阶段模型输入。

训练后自动扫描相同的 240 个 validation 帧与固定噪声种子，报告逐阶段误差和夹爪准确率。采样和损失同时调整，因此属于组合改进实验，不能单独归因于夹爪加权。原始 v13 权重、数据、验证报告继续保留。

    Get-Content results\vla\relative4d_close_lift_20261012_v14\pipeline_status.json
    Get-Content results\vla\training_logs\smolvla_relative4d_close_lift_lora16_20261012_v14.log -Tail 10 -Wait

对增强后的 006000 权重（30k 基线再微调 6k）执行相同 Easy/Medium 六回合诊断：

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\diagnose_vla_close_lift_v14.ps1

输出目录为 results/vla/relative4d_close_lift_paired_20261012_v14。此项是用户明确要求的 2 cm 门槛未通过后的诊断测试。对比使用完整且视觉初始化有效的基线 visual_only_benchmark_20261012_175444_556；不完整运行的日志保留但不计入成功率。

scripts/compare_vla_close_lift_pair.py 接受 --baseline、--candidate、--output，核对物体位姿、质量、摩擦、相机、照明、场景种子、控制参数和推理种子一致，再生成逐回合 JSON。提前闭合的主指标定义为：首次执行闭合前/当次、距初始视觉目标超过 2.5 cm 的闭合策略请求；按 10 Hz 请求计数，不重复计入 5 个保持步，也不把抬升时继续保持闭合算成新的提前闭合。另记录首次执行闭合距离、首次平均指节位置小于 3 cm 的距离，以及真实物体最大/最终抬升高度。2.5 cm 是几何诊断阈值，同时给出 1.5/5 cm 的敏感性指标，不能替代实际接触或抓取成功证据。

### v9 成功版本回归基线（v15）

后续改动以 `smolvla_execution_aligned_full_lora16_20261011_truth_v9/checkpoints/006000` 为历史成功基线：30 维状态（含视觉目标绝对坐标及阶段）、8 维动作（Δxyz、四元数、夹爪）、预测 chunk=4，但每次仅执行一个预测并保持 5 个控制步。历史 Easy 小测试为 1/1、后续对比为 2/3；不是已建立稳定成功率。

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_vla_v9_regression_v15.ps1

每次创建独立 `results/vla/v9_regression_<时间戳>_v15` 目录，不覆盖旧模型和报告。先保存 `baseline_lock.json`，校验成功权重 SHA256，并记录基础权重、适配器、归一化处理器、配置和执行代码的哈希；随后以当前修复后的连续有效帧初始化流程运行 Easy/Medium 各 3 回合，600 控制步、真实抬升至少 10 cm、无专家接管。原始相机初始化缺陷不作为应当恢复的行为。

目录内 `pipeline_status.json` 查看进度；结束后 `regression_report.json` 包含逐回合首次闭合、提前闭合请求、真实抬升、阶段切换，以及历史改动台账。异常初始化或通信错误不当作策略失败纳入有效对比。该入口不启动训练。

2026-10-12 回归实测已完成：`results/vla/v9_regression_20261012_191554_831_v15/regression_report.json`。v9 Easy 为 **2/3**，Medium 为 **0/3**；两个成功回合首次执行闭合距离分别为 **2.43 / 1.83 cm**，物体最大真实抬升 **11.98 / 13.33 cm**。六回合均有效初始化，无通信错误、无专家接管。相同六个场景的 v13 30k 和 v14 36k 均为 0/6；这确认了成功基线仍可复现，但不能把多因素版本差异归因于四维接口或 chunk 单一因素。新增锁定与场景一致性测试连同闭合统计测试共 5 项通过。

| 后续修改 | 现有证据 | 归因限制 |
|---|---|---|
| 旧 adapter 续训 → 原始模型新 LoRA | v10 保留旧接口的对比 | 初始化及其历史训练不同，不等于接口消融 |
| legacy → 滞回/夹持反馈阶段机制 | v11 同权重 2×2 诊断 | A 组曾初始化失败，v12 补测不是严格四组同步重跑 |
| 模型姿态 → 固定俯视姿态 | v11 同权重对照 | v11 仅阶段 1–4 覆盖，v13 所有阶段固定 |
| 绝对目标 → 相对目标、移除阶段输入 | v13 与其他修改一起引入 | 尚未独立消融，需配套训练与部署 |
| 8 维 → 4 维动作，chunk 4 → 1 | v13 同时改变 | 不能从新版权重失败直接判断哪项有害 |
| 闭合/抬升采样与夹爪损失加强 | v14 同场景六回合对比 | 采样、损失及优化器同时改变，不是单因素证据 |

训练消融必须保持初始化、数据划分、训练预算、随机种子和执行频率一致。预测 chunk 长度与执行 horizon 分开控制；不能仅在推理时改 `chunk_size`、重排状态语义或删除输入维度，就宣称完成接口适配对比。下一步从同权重执行规则对照开始，训练消融再逐项展开；本回归入口不自动启动这些训练。

### v9 单因素：首次闭合距离阈值（v16）

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_vla_v9_guard_ablation_v16.ps1

同一 v9/006000 权重串行运行 A（原阈值 7.5 cm）和 B（收紧到 2.5 cm），各 Easy/Medium 3 回合，共 12 回合。唯一干预变量是 `grasp_enable_distance_m`；两组都保留原闭合锁定、legacy 阶段、模型姿态、30/8 维接口、chunk=4、horizon=1、动作保持 5 步、600 步上限。禁止将阈值放宽至原值以上；通用入口默认仍为 7.5 cm。

输出独立 `results/vla/v9_guard_ablation_<时间戳>_v16`，其中 `pipeline_status.json` 查看进度，`comparison.json` 为最终逐回合对比。每组必须完成六个有效初始化且无通信错误、无专家接管的回合；比较脚本核对相同权重和全部场景参数，并拒绝除阈值与服务端口之外的第二项配置变化。原始脚本备份保存在 `results/vla/v9_guard_ablation_v16_source_backup`。

评测包括成功率、首次实际闭合距离、模型提前闭合请求、被拦截请求、最大/最终真实抬升。收紧阈值自然会减少“远处被接受的闭合”，这本身不等于模型改善；须看真实抬升成功是否提升。两组共享全局推理种子，但没有强制逐请求共用扩散噪声，GPU 图像/物理也不保证逐位相同。本实验不更新权重、不自动修改部署默认值。

2026-10-12 实测：`results/vla/v9_guard_ablation_20261012_193040_913_v16/comparison.json`，12 回合全部有效、无通信错误、无专家接管；权重与场景一致、除闭合阈值和服务端口外配置完全一致。两组均 Easy 2/3、Medium 0/3。首次被接受闭合距视觉目标超过 2.5 cm 的回合从 4 降到 0，但从未闭合的回合从 1 增到 4；成功回合仍为 Easy 1/2。独立提前闭合请求为 27→184（只统计首次接受闭合前/当次，未闭合回合统计至结束），增加主要反映策略继续请求而被门控阻拦、统计窗口变长，并不表示模型权重变差。结论：本组小样本没有显示仅收紧阈值的抓取成功收益；保留原默认值，不启动额外训练。13 项相关单元测试通过。

### v9 单因素微调：chunk=4 对照 chunk=1（v17）

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_vla_chunk_ablation_v17.ps1

两组都从未修改的 v9/006000 adapter 初始化，新建优化器，各额外训练 6000 steps、batch=1、seed=1000、同一 v9 train 数据及加权采样计划、原 normalized 样本损失归约、Z 权重 3。学习率 1e-5→1e-6，warmup=300；唯一训练变量为预测 `chunk_size=4/1`，两组 `n_action_steps=1`。不是续接原 optimizer/RNG/scheduler 状态，也不是从 chunk4 新结果再训练 chunk1。更长 chunk 包含更多未来动作监督，所以预算按优化器更新数及输入帧数相同定义，不声称计算量或标签总数相同。

串行流程：chunk4 训练→固定 240 帧扫描→chunk1 训练→相同帧扫描→训练合同核对→两组最终 6000 checkpoint 各 6 个 Easy/Medium 闭环回合→对比。保存并扫描 2000/4000/6000，主要比较第一个预测动作的误差；闭环 checkpoint 预先固定为 6000，不按各组最优验证结果挑选。即使离线 2 cm 门槛未过，用户授权的单因素诊断仍可执行，但报告保留未过门槛事实。

每次创建 `results/vla/v9_chunk_ablation_<时间戳>_v17`：`pipeline_status.json` 查看当前阶段，`chunk4_initialization.json` / `chunk1_initialization.json` 检查训练前实际加载的 chunk、接口及可训练权重哈希；`input_lock.json` 锁定输入权重、train/validation 文件、采样和代码；`training_comparison.json` 核对配置、归一化及固定验证集一致；`comparison.json` 为最终闭环对比。模型目录和日志名见 `experiment_plan.json`。两组数据索引采样器使用相同的独立 RNG；不同 chunk 的噪声形状不同，不保证所有随机张量逐位相同。原脚本备份位于 `results/vla/v9_chunk_ablation_v17_source_backup`，旧入口默认行为不变。

2026-10-12 v9 历史 loss 审查：`results/vla/v9_loss_review_20261012_v17/v9_loss_curve.png` 及 `loss_review.json`。1000-step 均值为 0.589、0.448、0.462、0.431、0.437、0.418，后期波动明显；固定验证位置误差 500/3000/6000 steps 为 1.724/1.451/1.424 cm。6000 是本段优化器更新数，不是 epoch；batch1 对 27338 帧约 0.219 nominal epochs，且该权重已有 v7 训练历史。不能仅依据这一比值或最后一次 loss=0.345 判断还需大量训练。

## 6. 详细文档

- `docs/phase1_scene.md`：场景资产与坐标系；
- `docs/phase2_pick_and_lift.md`：Scripted IK；
- `docs/phase3_single_environment_collection.md`：单环境采集；
- `docs/phase4_domain_randomization.md`：随机化参数；
- `docs/phase5_vectorized_parallel_simulation.md`：GPU 并行；
- `docs/phase6_dataset_export.md`：HDF5 schema；
- `docs/phase7_benchmark.md`：分级 Benchmark；
- `docs/phase8_failure_analysis.md`：失败分析；
- `docs/phase9_failure_aware_sampling.md`：闭环采样；
- `docs/phase10_rgbd_data.md`：RGB-D 数据；
- `docs/phase11_ppo.md`：BC、DAgger、PPO；
- `docs/phase12_visual_perception_control.md`：视觉模型与控制接入。
- `docs/phase13_grasp_mode_comparison.md`：纯视觉、状态 PPO 与视觉 + PPO 对比。
- `docs/vla_delta_weighted_v5_results.md`：相对末端 Δxyz、Z/Hard/LIFT 加权训练与最终验证结果。
- `docs/vla_closed_loop_benchmark.md`：VLA 服务及固定 Easy/Medium/Hard 闭环复现命令。
- `docs/vla_dagger_recovery_v6_results.md`：1.5 cm 预抓取 DAgger 恢复采集、LoRA 微调、checkpoint 扫描与闭环诊断结果。
