# VLA Quick Start（Windows / Python 3.11 / 8GB显存）

> 当前机器应使用本页的 `setup_vla_env_py311.ps1`。它复用已下载的 CUDA PyTorch 2.11，并安装支持 Python 3.11 的 LeRobot 0.4.4。不要再次运行早期的 `setup_vla_env.ps1`，因为 LeRobot 0.6.1 要求 Python 3.12。

## 环境

```powershell
cd C:\Users\yjh\Desktop\work_project\ai_BOT
powershell -ExecutionPolicy Bypass -File scripts\setup_vla_env_py311.ps1
```

## 数据检查

现有 `phase12_rgbd_hard_raw` 中自动筛得59条完整成功演示、1623个10Hz视觉动作样本：

```powershell
conda activate lerobot-vla
python scripts\export_lerobot_vla.py --source dataset\phase12_rgbd_hard_raw --dry-run
```

建议继续用 `scripts\collect_data.py --enable_cameras` 采集 Easy 100、Medium 150、Hard 250 条完整成功演示。静态视觉补充样本不会进入 VLA 动作训练。

## 控制链路

```text
SmolVLA动作块 -> 安全过滤 -> Isaac绝对IK控制
                      ^
RGB-D位置校验 ---------+---- 异常时回退Scripted IK
```

完整的数据导出、LoRA训练、推理服务和可视化命令见 `docs/phase14_smolvla_hybrid_control.md`；其中环境创建命令以本页为准。
