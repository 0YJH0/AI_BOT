# 当前机器的 VLA 命令（LeRobot 0.4.4）

本机已验证环境组合：Python 3.11、LeRobot 0.4.4、PyTorch 2.11.0+cu128、RTX 4060 Laptop。

与通用 Phase 14 文档相比，数据导出和推理服务请使用带 `_v044` 后缀的兼容入口：

```powershell
conda activate lerobot-vla
cd C:\Users\yjh\Desktop\work_project\ai_BOT

python scripts\validate_vla_env.py
python scripts\export_lerobot_vla_v044.py --source dataset\phase12_rgbd_hard_raw --dry-run
python scripts\export_lerobot_vla_v044.py --source dataset\phase12_rgbd_hard_raw --output-root exports\lerobot\pick_lift_vla_baseline --repo-id local/isaac-pick-lift-vla-baseline
python scripts\serve_smolvla_v044.py --checkpoint results\vla\smolvla_lora_r16\checkpoints\last\pretrained_model --execution-horizon 8 --gpu-memory-fraction 0.45
```

Isaac 端命令不依赖 LeRobot 版本，仍使用：

```powershell
conda activate isaaclab23
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\run_hybrid_vla_pick_lift.py --device cuda:0 --realtime --keep-open --require-vla
```
