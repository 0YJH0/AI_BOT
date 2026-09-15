# Phase 0：环境检查与验收记录

验收日期：2026-09-15  
项目工作目录：`C:\Users\yjh\Desktop\work_project\ai_BOT`  
Conda 环境：`isaaclab23`  
环境路径：`C:\Users\yjh\anaconda3\envs\isaaclab23`

## 结论

Phase 0 核心环境验收通过，可以进入 Phase 1。

当前机器能够启动 Isaac Sim 和 Isaac Lab 的 headless GPU 仿真，完成 World/SimulationContext 初始化、reset、物理步进和 CUDA 张量计算。由于显存和驱动版本低于 Isaac Sim 5.1 官方测试配置，后续需要采用渐进式环境数量测试，并在正式开发前升级显卡驱动。

## 已安装版本

| 组件 | 版本 |
| --- | --- |
| Windows | Windows 11 Home China 25H2，Build 26200 |
| Conda | 24.11.3 |
| Python | 3.11.16 |
| pip | 26.2.1 |
| Git | 2.51.0.windows.1 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | 2.3.2 |
| PyTorch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| CUDA runtime（PyTorch） | 12.8 |
| ONNX | 1.18.0 |
| transformers | 4.57.1 |
| huggingface-hub | 0.36.0 |
| tokenizers | 0.22.1 |

为避免 2026 年的新依赖与 Isaac Sim 5.1 的固定依赖冲突，额外锁定：

- `click==8.1.7`
- `typing_extensions==4.12.2`
- `onnx==1.18.0`
- `wheel==0.44.0`
- `setuptools==75.8.0`

最终执行 `python -m pip check`，结果为 `No broken requirements found.`。

## 硬件检查

| 项目 | 本机结果 | 判断 |
| --- | --- | --- |
| CPU | Intel Core i9-14900HX，24 核 / 32 线程 | 满足开发需求 |
| RAM | 31.8 GB | 达到官方最低档，运行时需控制后台应用 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU | 支持 RTX/CUDA，但低于官方最低参考 GPU |
| VRAM | 8 GB | 低于官方 16 GB 最低参考值 |
| NVIDIA 驱动 | 576.02 | 低于 Isaac Sim 5.1 官方测试的 Windows 580.88 |
| C 盘可用空间 | 约 106.1 GB | 当前足够；需持续关注缓存和数据集增长 |

## 已通过测试

### 1. PyTorch CUDA

- `torch.cuda.is_available()` 返回 `True`。
- CUDA 设备识别为 `NVIDIA GeForce RTX 4060 Laptop GPU`。
- CUDA 张量计算成功。

### 2. Isaac Sim

- 使用 headless 模式启动 `SimulationApp`。
- 成功创建 `World` 和默认地面。
- 成功 reset 并运行 10 个物理步。
- 同一进程内 CUDA 张量和为 `1048576.0`。
- 测试进程退出码为 0。

### 3. Isaac Lab

- `AppLauncher` 使用 `cuda:0` 成功启动。
- 成功创建 GPU `SimulationContext`。
- 成功 reset 并运行 10 个物理步。
- 按 `SimulationContext.clear_instance()`、`app.close()` 顺序清理。
- 测试进程退出码为 0。

## 运行方式

打开 Anaconda Prompt 或已经初始化 Conda 的 PowerShell：

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"
```

`OMNI_KIT_ACCEPT_EULA` 表示当前用户已经确认接受 NVIDIA Omniverse EULA；新终端中需要重新设置，除非以后添加 Conda 激活脚本。

## 已知限制与后续动作

### 1. 显卡驱动需要升级

当前驱动 576.02 已能通过最小 GPU 仿真测试，但 Isaac Sim 5.1 官方在 Windows 上测试的是 580.88。进入 Phase 1 前建议通过 NVIDIA App 升级到 580.88 或更新的稳定 Production/Studio 驱动。

### 2. Windows GUI 的 HDF5 兼容版本

项目固定使用 `h5py==3.14.0`，其 wheel 内置 HDF5 1.14.6，与 Isaac Sim 5.1 GUI/RTX 传感器插件一致。较新的 `h5py 3.16.0` 内置 HDF5 2.0.0，会在 GUI 启动时造成 DLL 符号冲突；无头 Vulkan 模式下不一定暴露这个问题。

### 3. Windows 长路径未启用

注册表 `LongPathsEnabled` 当前为 `0`。Isaac Sim 的深层扩展路径曾导致 pip 安装失败，本次通过临时短盘符路径完成安装。建议使用管理员 PowerShell 执行：

```powershell
Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Type DWord -Value 1
```

执行后重启 Windows，再确认值为 `1`。当前会话没有管理员令牌，因此没有修改系统注册表。

### 4. 显存策略

8 GB VRAM 不适合直接假定 `128/256` 个带 RGB-D 相机的并行环境可运行。后续按照以下顺序实测：

1. 无相机：`1 -> 16 -> 64 -> 128`。
2. 记录显存、steps/s 和稳定性。
3. RGB-D 单独从较小环境数开始，必要时降低分辨率与采样频率。
4. 只有真实测试通过后，才在 README 和简历中写入吞吐量或最大环境数。

### 5. 官方源码下载

Git clone 两次被网络重置，随后 codeload 下载在约 15 MB 处中断。不完整文件保留为：

`_deps\IsaacLab-v2.3.2.zip.partial`

这不影响当前环境：官方 `isaaclab==2.3.2` PyPI 包已经安装，且包含 Isaac Lab 源码模块与测试。完整仓库可在 Phase 1 开始前使用可续传工具补充，用于查看官方脚本和示例，但不会直接修改上游源码。

## Phase 0 验收状态

- [x] 创建独立 Conda 环境
- [x] 安装与验证 CUDA PyTorch
- [x] 安装 Isaac Sim 5.1
- [x] 安装 Isaac Lab 2.3.2
- [x] 消除 Python 依赖冲突
- [x] Isaac Sim GPU/物理冒烟测试
- [x] Isaac Lab GPU/物理冒烟测试
- [ ] 升级 NVIDIA 驱动（建议项）
- [ ] 启用 Windows 长路径（建议项）
- [ ] 补全 Isaac Lab 官方仓库（非阻塞项）
