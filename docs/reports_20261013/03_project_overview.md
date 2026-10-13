# 项目总述：机器人操作仿真、数据引擎与策略闭环评测

版本：2026-10-13｜项目：ai_BOT｜结果口径：已保存的本地实验，不外推到真实机器人。

## 1. 项目做什么，以及做到了什么

本项目以 Franka 机械臂抓取桌面红色方块为载体，把 Isaac Lab / Isaac Sim 物理仿真、并行轨迹生成、RGB-D 数据集导出、BC / DAgger / PPO、RGB-D 位姿感知与 SmolVLA 微调接入同一套评测流程。目标不是仅制作一段抓取动画，而是建立能够追踪“哪批数据、哪个模型、哪些场景、怎样执行、为何成功或失败”的工程闭环。

已经实现：状态与图像同步采集、HDF5 与 LeRobot 数据出口；状态策略和视觉策略的分支控制；可复现的 Easy / Medium / Hard 分级；单场景可视化与共享 VLA 模型的多场景批量推理；按场景保存成功率、抬升高度、闭合距离、相机有效性和运行资源记录。

尚未实现或不能据现有结果宣称：可靠的通用 VLA 抓取、真实机器人迁移、多任务语言泛化、Hard 场景高成功率的纯 RGB VLA，以及跨机器长期稳定的 16 环境实时运行。当前 VLA 仍显著弱于具有目标真值输入的状态策略。

## 2. 系统结构与执行链

```text
configs：任务参数 / 随机化范围 / 分级标准
  ↓
envs + randomization：机器人、物体、相机、物理世界
  ├─ 真值 + Scripted IK → 专家动作 → 完整成功示范
  ├─ 状态观测 → BC / DAgger / PPO → IK / 关节执行
  └─ RGB-D → 位姿估计 → 视觉 IK 或 SmolVLA 条件输入
                         ↑ RGB + 语言 + 自身状态
                         ↓ 动作块，经安全限制后执行
data_collection → HDF5 / LeRobot → 训练 / 固定验证集
benchmark → 逐回合指标 → failure_analysis → 下一轮数据或实验
```

Isaac Sim 提供场景、渲染、物理等运行基础，Isaac Lab 在其上组织机器人环境、观测、动作、奖励和并行训练。应用层代码负责本项目的任务定义、数据契约、模型连接与评价，不把平台能力等同于本项目独立实现。

VLA 推理与 Isaac 运行在两个 Python 环境中，通过本地 RPC 交换批量观测和动作。这样隔离 Isaac 依赖与 LeRobot / Transformers / PEFT 依赖，但也增加端口、超时、协议和生命周期管理的要求。

## 3. 代码地图：每部分负责什么

以下路径均相对仓库根目录；核心包前缀为 `src/isaac_lab_data_engine/`。

| 模块 | 主要文件 | 实现功能 |
| --- | --- | --- |
| 场景与环境 | envs/phase1_scene.py；pick_lift_env_cfg.py | 桌面、Franka、方块、相机、仿真步长、环境复制及控制接口 |
| 观测与奖励 | envs/observations.py；ppo_rewards.py；ppo_reward_math.py | 策略状态拼接、抓取相关奖励及可独立测试的数值逻辑 |
| 教师控制 | controllers/scripted_pick_lift.py | 接近、预抓取、闭合、抬升阶段及目标末端位姿；由 IK 转为关节控制 |
| 域随机化 | randomization/domain_randomizer.py | 采样位姿、质量、摩擦、相机和光照；应用到指定环境并保存样本 |
| 坐标系 | geometry/transforms.py | 相机、世界、机器人坐标之间的变换约定 |
| 原始采集 | data_collection/episode_writer.py | 写入逐回合状态、动作、奖励、时间和相机数据 |
| 数据出口 | data_collection/dataset_exporter.py；hdf5_dataset.py | HDF5 分片、索引、划分、校验与回读 |
| 视觉感知 | vision/rgbd_dataset.py；rgbd_pose.py；policy_adapter.py | RGB-D 数据、目标位置估计及替换状态策略目标观测的适配 |
| 基准评测 | benchmark/config.py；runner.py；reporting.py | 读取等级、运行场景、成功判断、逐回合与汇总报告 |
| 失败分析 | analysis/failure_analysis.py | 汇总典型失败类型，而非仅输出平均成功率 |
| 数据闭环 | data_engine/failure_aware_sampling.py；round_comparison.py | 根据失败分布组织采样及实验轮次比较 |
| PPO 辅助训练 | training/bc_auxiliary.py | PPO 更新中加入专家动作监督，约束策略偏离 |
| VLA 通信 | vla/rpc.py；checkpoint.py | 请求/响应、批量环境标识、模型配置和权重定位 |
| VLA 数据契约 | vla/dataset.py；full_trajectory.py；teacher.py；label_replay.py | 训练样本构造、实时执行标签、教师与学生观测隔离、标签回放诊断 |
| VLA 运行安全 | vla/visual_initialization.py；hybrid_controller.py；batch_execution.py | 连续有效帧、动作保护、各环境动作块与保持步计数 |
| VLA 训练诊断 | vla/sampling.py；weighted_smolvla_loss.py；phase_diagnostics.py | 样本权重、阶段/坐标分量损失、阶段和夹爪诊断 |
| 接口实验 | vla/relative_interface.py；joint_interface.py；future_joint_labels.py | 相对状态、4维动作、关节目标及未来状态标签等独立实验分支 |

### 3.1 可直接查找的运行入口

| 入口脚本 | 用途 |
| --- | --- |
| scripts/run_env.py | 基础环境启动与场景检查 |
| scripts/run_pick_lift.py | 无学习权重的 Scripted IK 抓取演示 |
| scripts/collect_data.py；export_dataset.py | 轨迹采集与 HDF5 导出 |
| scripts/run_benchmark.py；analyze_failure.py | 状态基准测试与失败报告 |
| scripts/train_bc_warmstart.py；train_ppo.py；evaluate_ppo.py | BC 初始化、PPO 训练与评测 |
| scripts/train_visual_keypoint.py；run_visual_pick_lift.py | RGB-D 感知训练与视觉位姿驱动的 IK 演示 |
| scripts/lerobot_train_windows.py | Windows / LeRobot 训练兼容入口 |
| scripts/serve_smolvla_windows_v044_v4.py | 加载基础模型和 LoRA，启动 VLA 推理服务 |
| scripts/run_vla_benchmark.py | 单环境 VLA、视觉对照及详细诊断 |
| scripts/run_vla_parallel_probe.py | 共享模型的批量相机、动作执行和容量/闭环测试 |
| scripts/run_vla_medium64_parallel_comparison.ps1 | 四组 Medium 对照实验的本机编排 |

`configs/` 保存配置，`tests/` 保存测试，`docs/` 保存阶段设计，`dataset/` 保存原始示范，`exports/` 保存训练格式，`results/` 保存权重与评测。大量结果和数据不随普通 Git 克隆分发。历史 v9、v17、v18 等脚本是实验分支，不是要求新用户依次运行的必经步骤。

## 4. 数据与接口：训练到底学什么

### 4.1 状态轨迹与图像

状态控制轨迹为 50 Hz，保存机器人关节位置/速度、末端位姿、目标位姿、动作、奖励、时间及控制阶段。RGB-D 以 10 Hz 采集，通过 frame_step / 时间戳对应状态时间轴；这不是每个 50 Hz 控制步都有一张新图像。

相机数据包含 RGB、以米为单位的深度、内参和外参。深度中的无效/无命中像素需要显式过滤。导出器构造 HDF5 分片、索引与训练/验证/测试划分，在回读校验后标记完成；原始数据仍保留。

### 4.2 真值教师与视觉学生

完整示范采集使用真实物体位置指导教师 IK，但学生输入仍为 RGB、语言、自身状态及该分支的视觉估计/阶段。真实物体坐标用于教师和评价，不直接拼入学生模型输入。教师有特权信息不等于学生有同样信息。

采集和部署都采用当前末端相对位移、4 cm 位移上限、每个动作保持 5 个控制步的链路，直接记录实际执行标签。这避免把旧轨迹在不一致的控制语义下事后重标后直接当作可靠示范。

v9 完整数据集为 600 条成功轨迹：Easy、Medium、Hard 各 200 条；总尝试 610 次，拒绝 10 次。导出后为训练 510 条 / 27,338 帧，验证 45 条 / 2,438 帧，测试 45 条 / 2,399 帧，共 32,175 个观测时刻。图像编码在 LeRobot Parquet 的 image 字段中，不是必须存在单独的 PNG 文件夹。

语言任务为固定指令 “Pick up the red cube and lift it 10 centimeters.”。这具备语言输入通道，但没有证明多指令语义泛化。

### 4.3 当前 v17 / v18 评测接口

| 内容 | 当前主评测分支 |
| --- | --- |
| 视觉 | RGB；另有 RGB-D 位姿估计作为结构化条件 |
| 文本 | 固定抓取并抬升指令 |
| 状态 | 30维：9关节位置 + 9关节速度 + 7末端位姿 + 3视觉目标坐标 + 1有效标志 + 1阶段 |
| 动作 | 8维：Δxyz + 目标四元数 wxyz + 夹爪指令 |
| 模型内部容量 | state/action 均填充到32维；外部8维不意味着重建为8维模型主干 |
| 归一化 | STATE / ACTION 使用训练统计的 MEAN_STD；视觉按处理器路径处理 |
| chunk_size | 4：预测四个后续动作 |
| 执行长度 H | 对照1、2、4；执行后再获取观测和预测 |

位移在每个子动作开始时相对当前末端解码，再保持 5 个控制步。H=2 对应约 0.2 秒仿真时间才重规划一次，不是实际墙钟时间必然达到 5 Hz。仿真会等待同步推理；模型耗时会降低运行速度，不应简单等同于“机器人漏掉动作”。

仅 RGB＋语言＋自身状态的分支也存在，但不是下面 64 回合主结果采用的接口。`--pure-vla` 在当前脚本里表示关闭 Scripted IK 回退，不能据此删去“视觉估计与阶段条件”的描述。

### 4.4 BC、DAgger、PPO、VLA 的区别

BC 监督拟合教师动作。DAgger 让学习策略访问自身会遇到的状态，再取得专家标签补入训练，针对分布偏移。PPO 在环境里采样奖励，通过优势估计与截断策略目标更新 actor / critic；本项目还可叠加 BC 辅助项。VLA 分支使用图像、语言和状态条件进行动作序列监督微调，不是把 PPO 奖励直接迁移成 VLA 训练目标。

已核验的长 rollout PPO 使用57维状态观测和8维绝对 IK 动作，actor / critic 为 256、128、64 隐藏层的 ELU MLP；PPO 的 γ=0.98、λ=0.95、clip=0.2，每轮5个优化 epoch、4个 minibatch，BC辅助系数0.05。奖励结合接近、夹爪闭合、抬升及高度进度；该抓取版本关闭不适合当前目标的横向目标搬运奖励项。PPO学到的是从状态到控制目标的反馈策略，而不是视觉识别器或现成轨迹播放器。

## 5. 分级 Benchmark 的设计

| 参数 | Easy | Medium | Hard |
| --- | --- | --- | --- |
| 物体 X / m | 0.55 | 0.48–0.64 | 0.36–0.78 |
| 物体 Y / m | 0 | −0.14–0.14 | −0.30–0.30 |
| yaw / 度 | 0 | −180–180 | −180–180 |
| 质量 / kg | 0.10 | 0.05–0.25 | 0.02–0.45 |
| 静摩擦系数 | 0.70 | 0.70 | 0.08–0.95 |
| 相机位置扰动 | 无 | 无 | XY各±0.15 m，Z±0.12 m |
| 光照强度 | 2500 | 2500 | 1000–4000 |
| 随机种子 | 7101 | 7102 | 7103 |

范围来自 `configs/randomization_*.yaml`。Hard 还变化动态摩擦比例、恢复系数、相机朝向目标和光色。不同初始条件是正常的任务分布测试，不意味着任意变化都应零样本成功。必须区分训练覆盖范围内插值与新相机/新物体等分布外泛化。

注意：历史状态策略评测未开启相机，因而该报告中记录的 Hard 相机参数并不构成视觉鲁棒性实验。最新 16 环境四组测试明确是 Medium，相机固定，不是 Hard。

## 6. 量化成果与证据边界

### 6.1 历史状态控制结果

| 方法 | Easy | Medium | Hard | 回合数 |
| --- | --- | --- | --- | --- |
| Scripted IK | 100% | 99.61% | 93.75% | 每等级256，合计768 |
| PPO + BC，128步 rollout | 100% | 96.88% | 89.45% | 每等级256，合计768 |

Scripted IK 分子为 256 / 255 / 240；PPO 为 256 / 248 / 229。PPO 使用目标真值状态，并非图像策略。历史两类成功判据要求物体中心高于桌面10 cm，并结合夹持、距离与持续10步条件，不能直接改写成“相对初始高度实际抬升10 cm”。两者最大步数也分别为300与600，因此不是完全等预算性能比较。

所引用 PPO 训练为 1024 个并行环境 × 每环境128步 × 100轮，共 13,107,200 transitions；日志训练耗时约399.4秒，约32,815 transitions/s。它是状态输入、无VLA渲染负担的训练吞吐，不能套用于相机或VLA。早期“约1230万、Hard 57.42%”属于另一个历史实验，不能与此结果混写。

### 6.2 最新 VLA Medium 成对场景测试

统一：16普通相机环境、4波、每组64回合、600控制步、chunk=4、repeat=5、相同场景manifest；输入包含视觉估计和阶段，无真值目标输入，无Scripted IK回退。成功额外要求实际抬升至少10 cm。

| 权重 / 连续执行长度 | 成功 | 成功率 | 平均首次闭合真值距离 | 平均最大抬升 | 测试耗时 |
| --- | --- | --- | --- | --- | --- |
| v18 r32 / H=1 | 10/64 | 15.63% | 3.95 cm | 2.19 cm | 633.6 s |
| v17 r16 / H=1 | 8/64 | 12.50% | 4.36 cm | 1.90 cm | 611.9 s |
| v17 r16 / H=2 | 14/64 | 21.88% | 4.47 cm | 2.81 cm | 416.0 s |
| v17 r16 / H=4 | 7/64 | 10.94% | 4.32 cm | 1.53 cm | 309.7 s |

本次最佳是 v17 / H=2，比同权重 H=1 高9.375个百分点、测试段耗时低约32%。这是该固定样本上的观察，不足以称为普遍最佳或统计显著。H=1与H=2共同成功5例，仅H=1成功3例，仅H=2成功9例；两组共同失败47例，仍需针对失败状态改进。

首次闭合均值只在出现闭合请求的回合统计，四组样本数为63、58、55、54。该距离不是指尖到方块表面的间隙；2.5 cm也不是几何上必然不可抓的边界。全部未成功回合在汇总中记为max_steps，不能都归因为“抓住后不抬升”。

r32和r16训练历史不同，不能将这一表格用作只改变LoRA秩的因果实验。VLA与上节PPO的输入、等级、步数、成功判据不同，更不能直接拿百分比证明网络结构优劣。

### 6.3 并行与显存

普通Camera的16环境容量测试完成2波、32回合，3,840次相机检查没有无效帧；显存峰值7,631 MiB / 8,188 MiB，剩余557 MiB。最新四组正式批量测试也完成，无无效相机帧，峰值最高7,612 MiB。

结论应写为“当前机器、当前配置下，普通Camera 16环境已通过测试”，而不是“硬件最多16个环境”或“16环境实时50 Hz”。容量吞吐约54.67个环境控制步/秒，包含完成回合的保持槽位；这是负载指标，不等于有效任务吞吐。

TiledCamera 曾出现RGB全黑，普通Camera对照恢复；其确切根因尚未确认。长时间等待无效帧并不能证明只是算力不足，也不能凭一次对照断言上游库必有缺陷。

## 7. 已知限制与后续优先级

第一，统一旧新成功判据，再做公平重测，而不是修改历史数字。第二，冻结可复现的v17/H=2与v9回归基线，记录首次闭合、闭合后Z动作、夹持保持和实际抬升。第三，根据失败轨迹补充接近到闭合、闭合到抬升的恢复样本，以单因素实验验证采样或损失权重。第四，完成社区权重清单、路径配置和许可证后，再声称可独立开源复现。

继续提高LoRA秩、延长训练或改为关节输出都可以成为实验，但不应在动作语义、训练部署分布或渲染有效性未确认时默认有效。

## 8. 证据索引与阅读路线

最新主结果：`results/vla/medium64_parallel16_20261013_160323_916/comparison_summary.json`。

普通相机容量：`results/vla/parallel_capacity_20261013_155045_631_standard16/capacity_summary.json`。

状态IK：`results/benchmark/benchmark_report.json`。长rollout PPO：`results/ppo_evaluation_dagger2_ppo_bc_128step_100iter_final_256/benchmark_report.json`。训练计数：`results/ppo/ik_abs_dagger2_ppo_bc_128step_1024env_100iter/training_summary.json`。

完整采集：`dataset/vla_execution_aligned_full_20261011_truth_v9/collection_summary.json`。划分：`exports/lerobot/pick_lift_execution_aligned_20261011_truth_v9/*/meta/info.json`。

可携带证据摘要见同目录 `evidence_snapshot.json`；完整逐回合数据仍以原始结果目录为准。
