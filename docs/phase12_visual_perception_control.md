# Phase 12：RGB-D 视觉感知与控制策略接入

## 目标

Phase 12 将此前“只采集、不使用”的 RGB-D 数据接入控制链路：

`RGB-D → 视觉关键点 → 相机坐标 xyz → 世界/机器人坐标 → 控制策略 → 10 cm 抬升评测`

控制器不再读取物体真值位姿。仿真中的物体真值只用于计算视觉误差和最终成功率。

## 数据集

- 配置：Hard Domain Randomization
- 并行相机环境：8
- episode：64
- 状态 transitions：8,080
- RGB-D 帧：1,828（320×240）
- episode 级划分：Train 51 / Validation 7 / Test 6
- 对应帧数：Train 1,456 / Validation 209 / Test 163
- HDF5：`exports/pick_lift_rgbd_hard_v1`

每帧包含 RGB、米制深度、相机内参、相机世界位姿、物体相机坐标标签和末端位姿。划分按 episode 完成，避免同一条轨迹的相邻帧同时进入训练集和测试集。

## 模型

最终模型是轻量 RGB-D 关键点网络 `RGBDKeypointPositionNet`：

1. 四通道 RGB-D 输入缩放到 80×60。
2. CNN 输出物体中心热力图，学习二维关键点。
3. RTX 深度提供可见表面的米制距离，网络只回归表面到方块中心的深度残差。
4. 利用相机内参执行针孔反投影，恢复 ROS optical camera frame 中的 xyz。
5. 使用相机外参和机器人根位姿，将 xyz 转换到 IK/学习策略使用的 robot-root frame。

第一版直接回归 xyz 出现训练集记忆，验证平均误差约 22.18 cm，已被弃用。几何关键点版本显著降低了 episode 间泛化误差。

最终 checkpoint：

`results/vision/rgbd_keypoint_depth_residual_hard_v2/best_model.pt`

测试集指标：

- 关键点平均误差：0.44 个缩放像素
- 三维误差中位数：1.53 cm
- 三维平均误差：2.73 cm
- 2 cm 内帧比例：72.39%
- 5 cm 内帧比例：90.80%
- P95：12.09 cm（主要来自夹爪遮挡帧）

## 遮挡与安全约束

红色方块与 Franka 夹爪端部外观接近，机械臂靠近后还会遮挡方块。在线控制采用以下处理：

- 将已知 Hard 任务物体工作空间投影为深度有效掩码；
- 在线使用掩码内热力图峰值，避免多峰 soft-argmax 产生不存在的平均位置；
- REST 阶段进行多帧 EMA，机械臂开始接近后锁定静止桌面目标；
- 依据已知桌面高度和 5 cm 方块尺寸，将抓取前中心高度约束为 3 cm。

这些约束来自任务定义，不读取当前物体真值。

## 闭环结果

`run_visual_pick_lift.py` 已在一个 Hard 随机场景中完成真实闭环验收：

- 初始视觉定位误差：0.93 cm
- 融合期间平均定位误差：1.02 cm
- 网络推理平均耗时：22.07 ms
- 完成步数：121
- 最大抬升：13.39 cm
- 正式持续 10 cm 成功判据：通过

Scripted IK 路径已经跑通。57 维 `VisualPolicyObservationAdapter` 还会用视觉位置重新计算 object position、object pose、object-in-EEF、grasp features 和 adaptive phase，并已接入 `evaluate_ppo.py` 的 RSL-RL TensorDict 接口。

DAgger actor 的单 episode Hard 接线冒烟测试已完整执行，但该次结果为 timeout。原因是现有 DAgger/PPO 使用无噪声仿真物体状态训练，视觉误差与无物体朝向输入造成 observation distribution shift。下一阶段应使用视觉观测重放进行 DAgger/PPO 微调，而不能把“接口跑通”等同于“学习策略视觉闭环成功”。

## 复现命令

激活环境并接受 EULA：

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"
```

重新训练视觉模型：

```powershell
python scripts\train_visual_keypoint.py --dataset-dir exports\pick_lift_rgbd_hard_v1 --output-dir results\vision\rgbd_keypoint_depth_residual_hard_v3 --epochs 50 --batch-size 128 --device cuda:0
```

无窗口验收：

```powershell
python scripts\run_visual_pick_lift.py --headless --device cuda:0 --num-envs 1 --max-steps 600
```

GUI 实时观察：

```powershell
python scripts\run_visual_pick_lift.py --device cuda:0 --num-envs 1 --max-steps 600 --realtime --keep-open
```

将视觉观测接入 DAgger actor：

```powershell
python scripts\evaluate_ppo.py --headless --device cuda:0 --checkpoint results\ppo\bc_dagger2_adaptive_contact_vertical_ik_abs_1024env\model_bc.pt --action-space ik_abs --visual-checkpoint results\vision\rgbd_keypoint_depth_residual_hard_v2\best_model.pt --levels hard --num-envs 1 --episodes-per-level 1 --max-steps 600 --output-dir results\visual_policy_evaluation\dagger_hard_smoke
```
