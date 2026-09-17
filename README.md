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
| DAgger 初始化后 PPO 微调 | Easy 100.00%，Medium 77.73%，Hard 57.42% |
| RGB-D 测试集三维误差 | 中位数 1.53 cm，平均 2.73 cm，90.80% 帧在 5 cm 内 |
| 视觉驱动 Scripted IK | Hard 随机场景 121 步完成，最大抬升 13.38 cm |
| 自动化测试 | 32 passed |

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
python scripts\train_ppo.py --headless --device cuda:0 --action-space ik_abs `
  --num-envs 1024 --iterations 20 `
  --action-noise-std 0.005 --learning-rate 0.000001 `
  --resume-checkpoint results\ppo\bc_dagger2_rerun\model_bc.pt `
  --run-name ik_abs_dagger2_ppo_rerun
```

`--iterations` 表示额外 PPO 更新次数。输出目录由 `configs/ppo.yaml` 中的 `output_root` 和 `--run-name` 共同决定。

#### 11.4 评测 PPO checkpoint

```powershell
python scripts\evaluate_ppo.py --headless --device cuda:0 --action-space ik_abs `
  --checkpoint results\ppo\ik_abs_dagger2_ppo_rerun\model_20.pt `
  --num-envs 128 --episodes-per-level 256 --max-steps 600 `
  --output-dir results\ppo_evaluation_ik_abs_dagger2_ppo_rerun
```

不要只看训练 reward；最终模型选择必须依据独立 Easy/Medium/Hard Benchmark。当前 20 轮 PPO 在 Hard 上由 73.83% 下降到 57.42%，说明存在 on-policy catastrophic forgetting，因此项目同时保留最佳 DAgger 和 PPO checkpoint。

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
  --checkpoint results\ppo\bc_dagger2_adaptive_contact_vertical_ik_abs_1024env\model_bc.pt `
  --action-space ik_abs `
  --visual-checkpoint results\vision\rgbd_keypoint_depth_residual_hard_v2\best_model.pt `
  --levels hard --num-envs 1 --episodes-per-level 1 --max-steps 600 `
  --output-dir results\visual_policy_evaluation\dagger_hard_rerun
```

`VisualPolicyObservationAdapter` 会用视觉预测重新计算以下 57 维观测项：

- object position；
- object pose 中的位置，未知朝向使用单位四元数；
- object position in end-effector frame；
- grasp state features；
- adaptive manipulation phase。

当前这里只证明视觉观测已经进入 actor。要提高视觉学习策略的成功率，需要采集视觉预测下的 student-visited states，执行 Visual DAgger 再训练。

## 4. 已有 checkpoint 和结果

| 内容 | 路径 |
| --- | --- |
| 当前最佳 DAgger | `results/ppo/bc_dagger2_adaptive_contact_vertical_ik_abs_1024env/model_bc.pt` |
| PPO 微调模型 | `results/ppo/ik_abs_dagger2_adaptive_vertical_ppo_1024env_20iter/model_20.pt` |
| RGB-D 视觉模型 | `results/vision/rgbd_keypoint_depth_residual_hard_v2/best_model.pt` |
| 视觉训练报告 | `results/vision/rgbd_keypoint_depth_residual_hard_v2/summary.json` |
| 视觉闭环报告 | `results/vision/visual_control_smoke_summary.json` |
| Scripted IK Benchmark | `results/benchmark/` |
| Failure Analysis | `results/failure_analysis/` |

## 5. 常见问题

### 命令提示 `unrecognized arguments: \``

反引号是 PowerShell 的续行符，不是脚本参数。它必须放在 PowerShell 行尾；如果使用 Anaconda Prompt/CMD，请将整条命令写在一行。

### GUI 首次加载很慢

Isaac Sim 需要初始化 Kit、RTX shader、USD 资源和相机扩展。第一次通常最慢；后续启动会利用缓存。纯状态训练和大规模并行 rollout 默认关闭相机，以减少 RTX 开销。

### 8 GB 显存能否使用 1024 个并行环境

无相机的低维状态环境已经完成 1024 并行 BC/DAgger/PPO 实验。带 RGB-D 相机时显存和渲染开销显著增加，当前视觉数据采集使用 8 个相机环境，不应直接套用 1024 环境配置。

### 为什么视觉 DAgger/PPO 目前不如视觉 Scripted IK

现有 actor 在精确仿真物体状态上训练，而部署时输入包含视觉误差、遮挡、缺失朝向和时间相关噪声，产生 observation distribution shift。Scripted IK 对厘米级目标误差更容易加入显式约束；学习策略需要 Visual DAgger、观测噪声训练或端到端视觉策略微调。

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
