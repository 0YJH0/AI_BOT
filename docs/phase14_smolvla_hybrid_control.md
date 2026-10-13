# Phase 14：8GB 显存下的 SmolVLA 混合抓取

## 已实现的结构

本阶段没有修改或删除原有 PPO、DAgger、Scripted IK 和视觉抓取代码，而是新增一条可选链路：

```text
RGB + 25维机器人本体状态 + 语言指令
                 |
             SmolVLA 450M
                 |
          8维绝对IK动作块
                 |
   工作空间/步长/四元数/夹爪安全约束
                 |
              Isaac Lab

RGB-D目标位置 -------------------> 动作校验
RGB-D + Scripted IK -------------> 异常自动回退
```

VLA 的25维状态只包含9维关节位置、9维关节速度和7维末端位姿，不包含仿真器物体真实位姿。物体真值只用于最终成功率统计。

## 原项目副本

VLA 开发前的完整项目位于：

```text
C:\Users\yjh\Desktop\work_project\ai_BOT_original_20260923_before_vla
```

源码、配置、数据集、导出结果和训练结果均已核对一致。未复制的只有系统拒绝访问的四个临时 pytest 缓存目录。

## 1. 收集完整成功演示

静态视觉样本不能训练动作模型。VLA 数据必须包含从接近、下降、夹持到抬升10cm的完整成功轨迹。

在 `isaaclab23` 环境分别采集三个难度，建议目标为 Easy 100、Medium 150、Hard 250：

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"

python scripts\collect_data.py --num_envs 1 --episodes 100 --max_steps 600 --enable_cameras --camera_frame_interval 5 --randomization_config configs\randomization_easy.yaml --output_dir dataset\vla_easy
python scripts\collect_data.py --num_envs 1 --episodes 150 --max_steps 600 --enable_cameras --camera_frame_interval 5 --randomization_config configs\randomization_medium.yaml --output_dir dataset\vla_medium
python scripts\collect_data.py --num_envs 1 --episodes 250 --max_steps 600 --enable_cameras --camera_frame_interval 5 --randomization_config configs\randomization_hard.yaml --output_dir dataset\vla_hard
```

导出器默认只接收 `success=true` 且至少有两帧相机数据的轨迹。

## 2. 创建独立 VLA 环境

不要将 LeRobot 安装到 `isaaclab23`：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_vla_env.ps1
conda activate lerobot-vla
```

脚本安装 CUDA PyTorch、LeRobot 0.6.x、SmolVLA 与 PEFT。若本机驱动不支持脚本中的 CUDA 12.8 wheel，应按 PyTorch 官方安装器替换对应命令。

## 3. 转为本地 LeRobot v3

先做只读检查：

```powershell
python scripts\export_lerobot_vla.py --source dataset\vla_easy --source dataset\vla_medium --source dataset\vla_hard --dry-run
```

正式导出使用一个新目录，程序拒绝覆盖已有数据：

```powershell
python scripts\export_lerobot_vla.py --source dataset\vla_easy --source dataset\vla_medium --source dataset\vla_hard --output-root exports\lerobot\pick_lift_vla_v1 --repo-id local/isaac-pick-lift-vla-v1
```

## 4. 8GB 显存微调

训练时关闭 Isaac Sim。第一轮使用 batch size 1、LoRA rank 16 和5000步，成功后再增加到10000步：

```powershell
lerobot-train `
  --policy.path=lerobot/smolvla_base `
  --policy.output_features=null `
  --policy.input_features=null `
  --dataset.repo_id=local/isaac-pick-lift-vla-v1 `
  --dataset.root=exports/lerobot/pick_lift_vla_v1 `
  --output_dir=results/vla/smolvla_lora_r16 `
  --job_name=isaac_pick_lift_smolvla `
  --policy.device=cuda `
  --policy.push_to_hub=false `
  --wandb.enable=false `
  --batch_size=1 `
  --steps=5000 `
  --save_freq=1000 `
  --peft.method_type=LORA `
  --peft.r=16 `
  --peft.lora_alpha=16
```

如果 batch size 1 仍然显存不足，先将 LoRA rank 降为8；仍不足时在云端训练，再将本地 checkpoint 下载到 `results/vla`。不要在 Isaac Sim 运行期间训练。

## 5. 启动推理服务

在 `lerobot-vla` 终端运行：

```powershell
python scripts\serve_smolvla.py --checkpoint results\vla\smolvla_lora_r16\checkpoints\last\pretrained_model --execution-horizon 8 --gpu-memory-fraction 0.45
```

服务只绑定 `127.0.0.1`。模型返回8个10Hz动作，Isaac 端每个动作保持5个控制周期，因此物理控制仍为50Hz。

## 6. 可视化混合抓取

另开 `isaaclab23` 终端：

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\run_hybrid_vla_pick_lift.py --device cuda:0 --realtime --keep-open --require-vla
```

去掉 `--require-vla` 后，服务连接失败、VLA动作出现 NaN、视觉位置无效或四元数非法时会自动使用 RGB-D + Scripted IK。结果写入 `results/vla/hybrid_vla_smoke_summary.json`，其中分别统计 VLA 和回退动作数量，便于证明实际是否使用了 VLA。

## 7. Benchmark

第一轮不应只看混合系统总成功率，还应分别比较：

1. RGB-D + Scripted IK；
2. 纯 SmolVLA（保留安全限幅，但禁止 Scripted IK 回退）；
3. SmolVLA + RGB-D校验 + Scripted IK回退；
4. 状态 PPO。

每个难度至少评测100个固定种子，并报告成功率、VLA动作占比、回退率、平均推理延迟、最大抬升高度和失败类型。
