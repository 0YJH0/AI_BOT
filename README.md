# ai_BOT：Isaac Lab 机器人抓取、数据生成与学习策略评测

以 Franka 抓取桌面红色方块为任务，连接物理仿真、并行数据生成、BC / DAgger / PPO、RGB-D 感知、SmolVLA 微调和闭环 Benchmark。

本仓库是研究与工程实验项目，不是已验证的通用抓取产品。以下说明更新于 **2026-10-13**，命令使用 **Windows PowerShell**，在仓库根目录执行。每条 Python 命令均写成一行，避免把 PowerShell 反引号续行符粘贴到 CMD。

## 1. 先选择你要演示的内容

| Demo | 控制依据 | 是否需要本项目训练权重 | 推荐起点 |
| --- | --- | --- | --- |
| Scripted IK | 仿真目标真值 + 阶段控制 + IK | 否 | 首次安装、验证场景和物理 |
| 视觉 IK | RGB-D 估计目标位置 + 阶段控制 + IK | 需要视觉权重 | 展示视觉感知接入控制 |
| SmolVLA | RGB、语言、自身状态；本示例另含视觉估计和阶段 | 需要基础模型、LoRA和视觉权重 | 展示模型闭环，不保证每回合成功 |

“关闭专家回退”不等于“仅图像与语言输入”。下面 v17 示例是30维条件状态、8维动作接口，不能与仓库中 proprio-only、relative4d、joint-action 实验分支混用。

## 2. 环境准备

已运行的本机组合：

| 环境 | 版本与职责 |
| --- | --- |
| isaaclab23 | Python 3.11、Isaac Sim 5.1.0、Isaac Lab 2.3.2、Torch 2.7.0+cu128 |
| lerobot-vla | Python 3.11、LeRobot 0.4.4、PEFT 0.18.1、Transformers 4.57.6、Torch 2.11.0+cu128 |
| 硬件 | Windows、RTX 4060 Laptop 8 GB；这是实测机器，不是普适最低配置保证 |

先按 [Isaac Lab 官方安装说明](https://isaac-sim.github.io/IsaacLab/v2.3.1/source/setup/installation/pip_installation.html) 建立对应5.1 / 2.3系列环境并确认Isaac启动。该链接为2.3.1文档，本机实际安装为2.3.2；不同版本请核对兼容性。Isaac首次运行可能下载资产、构建缓存，冷启动明显慢于后续运行。

在已能导入Isaac Lab的环境中安装本项目：

~~~powershell
conda activate isaaclab23
python -m pip install -e .
~~~

PPO可选依赖：

~~~powershell
python -m pip install -e ".[ppo]"
~~~

使用者阅读并同意NVIDIA EULA后，才在当前终端设置：

~~~powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"
~~~

不要将Isaac和LeRobot的依赖直接合并到一个环境。当前 requirements-vla.txt 的版本范围与已验证的LeRobot 0.4.4服务入口不一致；不要把“安装该文件”当作已验收的VLA一键安装方案。scripts/setup_vla_env_py311.ps1 可作为历史依赖参考，但包含本机路径，使用前需审阅。

## 3. Demo A：先运行无学习权重的抓取

~~~powershell
conda activate isaaclab23
python scripts\run_pick_lift.py --num_envs 1 --max_steps 600 --realtime --keep_open
~~~

你应看到机械臂接近、闭合和抬升的控制过程。此处使用真实物体状态，是场景与专家控制演示，不是视觉或VLA推理。--keep_open 在动作结束后保留GUI；关闭窗口或在启动终端Ctrl+C结束。不要关闭与其他任务共用的终端或批量结束所有python.exe。

无GUI运行：

~~~powershell
python scripts\run_pick_lift.py --headless --num_envs 1 --max_steps 600
~~~

## 4. Demo B：RGB-D 估计位姿 + Scripted IK

需要本地视觉权重：

~~~text
results/vision/rgbd_keypoint_supplement_160x120_v3/best_model.pt
~~~

显式选择Medium配置，避免脚本默认Hard造成误解：

~~~powershell
python scripts\run_visual_pick_lift.py --checkpoint results\vision\rgbd_keypoint_supplement_160x120_v3\best_model.pt --randomization-config configs\randomization_medium.yaml --device cuda:0 --num-envs 1 --max-steps 600 --realtime --keep-open
~~~

模型从RGB-D估计位置，控制器根据估计位姿抓取。看到物体并不代表传感器帧有效；如果初始化失败，先检查相机与估计日志，不要当成抓取策略失败。

## 5. Demo C：SmolVLA 单环境闭环

### 5.1 需要哪些文件

本指南采用当前64回合Medium对照中表现最好的v17 r16、chunk=4、连续执行2个动作配置；它不是每次都能成功。

~~~text
results/vla/models/smolvla_base/
results/vla/training/smolvla_v9_chunk4_20261012_195615_610_v17/checkpoints/006000/pretrained_model/
results/vision/rgbd_keypoint_supplement_160x120_v3/best_model.pt
~~~

LoRA目录应保留完整配置、adapter权重和预处理/后处理器，不能只复制一个safetensors文件。当前adapter_config.json中的base_model_name_or_path可能指向作者电脑绝对路径；迁移时只在checkpoint副本中将其改为你实际基础模型路径，并保留原版。基础模型来自 [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)，迁移应保持原基础模型版本；下载最新版本不自动等于历史实验版本。

**当前仓库未提供统一的项目权重下载入口。** 普通克隆通常不包含results、dataset和exports。缺少上述文件时只能先跑Demo A；不能用未微调的基础权重替代后仍称为v17复现。

### 5.2 终端A：启动模型服务

在仓库根目录，激活单独的LeRobot环境：

~~~powershell
conda activate lerobot-vla
python scripts\serve_smolvla_windows_v044_v4.py --checkpoint results\vla\training\smolvla_v9_chunk4_20261012_195615_610_v17\checkpoints\006000\pretrained_model --device cuda --port 8765 --execution-horizon 4 --seed 20261011 --gpu-memory-fraction 0.35
~~~

等待服务完成模型加载并进入监听，再启动终端B。此处允许服务返回4个动作，实际执行几个由仿真端决定。显存配额只约束该进程的相关Torch分配，不是整机显存隔离保证。请关闭不必要的GPU应用。

### 5.3 终端B：启动可视化仿真

~~~powershell
conda activate isaaclab23
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$DemoOut = "results/demo_vla_" + (Get-Date -Format "yyyyMMdd_HHmmss")
python scripts\run_vla_benchmark.py --levels medium --episodes-per-level 1 --max-steps 600 --device cuda:0 --vla-port 8765 --vla-timeout 120 --execution-horizon 2 --action-repeat 5 --vla-position-target-mode delta --condition-vla-on-visual-pose --condition-vla-on-controller-phase --minimum-true-lift-m 0.10 --pure-vla --require-vla --realtime --keep-open --output-dir $DemoOut
~~~

默认视觉checkpoint就是5.1节所列路径。要用其他视觉权重，应显式传 --vision-checkpoint 并记录变化。

预期：GUI显示抓取过程，控制台写回合结果，输出目录保存评测记录。无专家回退时失败就应如实记录失败。--require-vla 遇到服务错误会中止；--pure-vla 禁止专家回退，但仍保留视觉/阶段条件与安全限制。任务结束后先结束仿真，再在终端A用Ctrl+C结束推理服务。

单环境GUI与16环境批量运行的随机调用顺序可能不同，不保证逐回合复现完全相同结果。需要统计复现时使用固定manifest和下面批量流程。

## 6. 多场景批量验证：先4个，再按显存扩大

服务保持终端A的v17配置，终端B使用普通相机。先运行4环境、1波容量与闭环烟测：

~~~powershell
$ProbeOut = "results/demo_parallel4_" + (Get-Date -Format "yyyyMMdd_HHmmss")
python scripts\run_vla_parallel_probe.py --parallel-envs 4 --waves 1 --camera-mode standard --strict-benchmark --headless --rendering_mode performance --device cuda:0 --max-steps 600 --vla-port 8765 --vla-timeout 120 --execution-horizon 2 --action-repeat 5 --vla-position-target-mode delta --condition-vla-on-visual-pose --condition-vla-on-controller-phase --minimum-true-lift-m 0.10 --pure-vla --require-vla --output-dir $ProbeOut
~~~

该脚本当前固定选择Medium，支持legacy 30维delta接口，不是任意任务通用并行入口。确认相机全部ready、帧非黑、无RPC错误后，才把 --parallel-envs 改为16、--waves改为4，得到64回合。输出的scene_manifest.json、camera_wave*.json、parallel_report.json用于核对场景、初始化和逐回合结果。

本机16普通相机曾通过，但显存仅剩约557 MiB容量余量；其他GPU应用、分辨率或依赖变化均可能导致失败。16不是推荐给所有8GB机器的默认值。不要省略 --camera-mode standard：脚本默认TiledCamera，而当前配置曾出现Tiled黑帧问题。

四组自动编排脚本为 scripts/run_vla_medium64_parallel_comparison.ps1，包含作者的Conda和模型路径；在其他机器使用前应先修改路径配置，不建议直接盲跑。

## 7. 从采集到训练再到测试的路线

1. 用Demo A验证物理、IK和终止条件。
2. 用collect_data采样轨迹；检查相机有效性、时间戳和执行标签。
3. 用export_dataset导出HDF5并回读；VLA分支使用独立的LeRobot导出与审计。
4. 状态策略先BC，再按需要DAgger / PPO；视觉模型单独训练；SmolVLA使用独立环境和明确接口微调。
5. 固定验证集扫描checkpoint；不只看总loss，还看关键阶段、夹爪、Z误差。
6. 单环境闭环诊断，再相同manifest批量Benchmark；保存分母、失败和协议，禁止只挑成功回合。

小规模数据入口示例（不代表复现600条VLA训练集）：

~~~powershell
$RawOut = "dataset/demo_state_" + (Get-Date -Format "yyyyMMdd_HHmmss")
python scripts\collect_data.py --headless --num-envs 1 --episodes 3 --max_steps 600 --output_dir $RawOut
python scripts\export_dataset.py --input_dir $RawOut --output_dir ($RawOut + "_hdf5") --format hdf5
~~~

关键训练入口：

| 工作 | 入口 / 资料 |
| --- | --- |
| BC / DAgger | scripts/train_bc_warmstart.py |
| PPO | scripts/train_ppo.py；scripts/evaluate_ppo.py |
| RGB-D感知 | scripts/train_visual_keypoint.py |
| VLA训练 | scripts/lerobot_train_windows.py |
| 执行对齐VLA导出 | scripts/export_vla_execution_aligned.py；audit_vla_execution_aligned_export.py |
| 完整历史训练命令 | [归档README](docs/archive/README_before_reports_20261013.md) |

历史训练命令依赖相应数据、权重和本机路径，不是新环境的一键复现承诺。本次文档整理没有重新启动训练；若复现训练，应新建run名和输出目录，保留旧权重。

## 8. 已核验结果

| 实验 | 结果 | 关键限制 |
| --- | --- | --- |
| Scripted IK Hard | 240/256，93.75% | 目标真值；物体中心高于桌面10cm |
| PPO + BC，128步rollout，Hard | 229/256，89.45% | 目标真值；与上项同类高度判据；最大步数不同 |
| v18 r32 / H=1，Medium | 10/64，15.63% | 视觉/阶段条件；实际抬升10cm |
| v17 r16 / H=1，Medium | 8/64，12.50% | 同上 |
| v17 r16 / H=2，Medium | 14/64，21.88% | 本次固定样本最佳，非普遍最优结论 |
| v17 r16 / H=4，Medium | 7/64，10.94% | 更少推理调用不等于更高成功率 |

PPO训练计数为1024×128×100=13,107,200 transitions。v9完整示范集为600条成功轨迹，Easy/Medium/Hard各200，导出划分510/45/45。VLA结果不是RGB-only结果；状态策略与VLA成功率不具有直接公平可比性。r32与r16训练历史不同，不是纯秩消融。

最新四组原始结果位于 results/vla/medium64_parallel16_20261013_160323_916/。已随报告保存关键证据摘要及原始JSON哈希。

## 9. 常见问题

- **config.json找不到**：先检查checkpoint目录层级和完整性；torchcodec回退pyav通常不是致命原因。
- **端口已占用**：用 Get-NetTCPConnection -LocalPort 8765 查看进程；识别后处理本项目残留服务，或让两个终端共同改用空闲端口。不结束不明进程。
- **图像全黑**：分别检查RGB标准差、深度有效率、新帧编号和连续有效窗口。单纯延长等待不保证解决；当前批量示例显式使用普通Camera。
- **窗口像卡住**：先看是否在等待同步推理或初始化，以及日志是否报RPC超时。仿真50Hz不意味着墙钟50Hz。
- **提前闭合**：区分模型请求、保护器过滤、实际夹爪状态；检查动作归一化、阶段与标签，不直接归因于模型小。
- **输出不再刷新**：Get-Content -Wait会持续等待同一个文件，不自动切换下一组日志；先看pipeline_status.json的stage/current_arm。
- **6000steps不是6000epoch**：等效遍历次数取决于训练帧数、batch、累积及采样方式。
- **16环境显存不足**：退回4或1环境；同一模型共享权重仍需要各环境渲染与特征内存。

不依赖启动Isaac的基础逻辑测试（需安装pytest并可导入本项目）：

~~~powershell
python -m pytest tests/test_vla_batch_execution.py tests/test_vla_batch_rpc.py tests/test_vla_visual_initialization.py
~~~

## 10. 项目报告

| 报告 | Markdown | PDF |
| --- | --- | --- |
| 挑战点：难点、解决与边界 | [阅读](docs/reports_20261013/01_project_challenges.md) | [下载](output/pdf/01_project_challenges.pdf) |
| 问题报告：通信、初始化、数据与评测 | [阅读](docs/reports_20261013/02_project_issue_report.md) | [下载](output/pdf/02_project_issue_report.pdf) |
| 总述：结构、功能与量化结果 | [阅读](docs/reports_20261013/03_project_overview.md) | [下载](output/pdf/03_project_overview.pdf) |
| 1小时介绍：讲稿、演示与问答 | [阅读](docs/reports_20261013/04_project_one_hour_talk.md) | [下载](output/pdf/04_project_one_hour_talk.pdf) |

[证据摘要](docs/reports_20261013/evidence_snapshot.json) · [报告说明](docs/reports_20261013/README.md) · [历史README完整归档](docs/archive/README_before_reports_20261013.md)

正式面向社区发布前，仍需项目维护者确认代码许可证、数据/权重分发许可、权重下载与基础模型版本清单，并在干净机器验收安装。当前不擅自声明MIT等许可证，也不将本机测试成功等同于任意机器开箱即用。
