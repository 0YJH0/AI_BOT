# Phase 13：三种抓取方案对比

## 对比目标

本阶段不使用图像重新训练 PPO，也不导出新的图像数据。RGB-D 模型只在测试开始阶段在线估计物体 XYZ；三种方案使用同一套 Easy / Medium / Hard 随机化和“持续 10 步抬升 10 cm”成功判据。

| 方案 | 控制器输入 | 控制器 |
| --- | --- | --- |
| 纯视觉抓取 | RGB-D 估计的物体 XYZ + 机器人本体状态 | Scripted IK |
| PPO 抓取 | 仿真物体状态 + 机器人本体状态 | PPO actor |
| 视觉 + PPO | RGB-D 估计的物体 XYZ + 机器人本体状态 | 同一个 PPO actor |

视觉 + PPO 不读取物体真值作为 actor 输入。遮挡前锁存 RGB-D 位置；手指闭合并接近目标后，使用末端执行器本体感知位移传播物体位置。仿真真值仅用于成功判定和离线误差分析。

## 关键修复

`VisualPolicyObservationAdapter` 现在：

1. 保留 grasp-state 中真实的手指绝对位置与速度，不再错误使用 `joint_pos_rel`；
2. 用视觉 XYZ 重算 object position、object pose、object-in-EEF、距离、held proxy 和 phase；
3. 抓取后用末端执行器平移量传播被遮挡物体的位置，使 actor 能观察抬升进度。

## 结果

测试规模为 32 episode/level，共 96 episode/方案。

| 方案 | Easy | Medium | Hard |
| --- | ---: | ---: | ---: |
| RGB-D + Scripted IK | 32/32（100.00%） | 9/32（28.13%） | 13/32（40.63%） |
| State + PPO | 32/32（100.00%） | 31/32（96.88%） | 30/32（93.75%） |
| RGB-D + PPO | 32/32（100.00%） | 2/32（6.25%） | 5/32（15.63%） |

纯视觉 IK 的失败类型：Medium 为 15 次 approach、1 次 grasp、7 次 lift；Hard 为 11 次 approach、2 次 grasp、6 次 lift。视觉 + PPO 的失败主要是 lift failure：Medium 28 次，Hard 27 次。

视觉误差与纯视觉 IK 成败存在明显相关性：

| 难度 | 成功样本平均误差 | 失败样本平均误差 |
| --- | ---: | ---: |
| Medium | 1.51 cm | 4.47 cm |
| Hard | 0.97 cm | 4.24 cm |

Hard 的本轮成功率高于 Medium 不是难度倒置，而是 32 个确定性随机样本下视觉误差分布不同；样本量较小，不应据此声称 Hard 普遍更容易。

## 结论

状态 PPO 说明控制策略与 10 cm 抬升能力已经建立。视觉接入后的主要瓶颈是相机/光照随机化下的 XYZ 定位误差，以及 PPO 只在精确状态上训练导致的 observation distribution shift。下一步若仍坚持不使用图像训练 PPO，优先方案是：在低维状态训练中注入与实测视觉误差相匹配的 XYZ 噪声和时间延迟，再进行 PPO/DAgger 微调；这样只训练 57 维观测，不存储图像。

## 结果文件

- `results/grasp_mode_comparison_32/visual_scripted_ik/benchmark_report.json`
- `results/grasp_mode_comparison_32/state_ppo/benchmark_report.json`
- `results/grasp_mode_comparison_32/visual_ppo/benchmark_report.json`
- `results/grasp_mode_comparison_32/summary/grasp_mode_comparison.csv`
- `results/grasp_mode_comparison_32/summary/grasp_mode_comparison.json`
