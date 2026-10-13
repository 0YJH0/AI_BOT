# 项目报告集（2026-10-13）

本报告集依据当前代码、配置与保存的实验结果编写，不重新启动训练或仿真。

1. [项目挑战点](01_project_challenges.md)：遇到什么难点、怎样定位、采取了什么解决办法以及未解决边界。
2. [项目问题报告](02_project_issue_report.md)：通信、端口、初始化、相机、标签、归一化、成功判据和开源复现问题。
3. [项目总述](03_project_overview.md)：代码职责、执行链、数据接口、分级设计与可核验数字。
4. [1小时介绍](04_project_one_hour_talk.md)：严格60分钟安排，包含口述、代码导读、现场演示、结果解读和问答。

PDF对应保存在 [output/pdf](../../output/pdf/)。社区演示入口见 [根README](../../README.md)；旧README完整保留在 [归档](../archive/README_before_reports_20261013.md)。

`evidence_snapshot.json` 保存选定原始JSON的相对路径、SHA256及内容快照，便于结果目录未分发时查看关键证据。它不是完整数据或逐回合日志的替代品。

四篇的共同口径：历史状态策略的桌面高度阈值，不等同于最新VLA实际抬升10cm；v17/v18当前主结果有视觉估计与阶段输入；16环境是当前配置已验证规模；最新四组为Medium各64回合，不是Hard。

Markdown是可编辑源文件。`scripts/build_project_reports.py` 将其生成PDF并做文字/渲染检查。PDF构建不需要Isaac或VLA运行环境，仅需要reportlab、pypdf、pymupdf、Pillow及中文字体。具体构建参数见该脚本的 `--help`。

本次质量检查：四份PDF均提取中文文本、逐页渲染并检查边界；根README的命令参数做静态核对，19个本地链接检查通过，相关批量执行/RPC/视觉初始化基础测试11项通过。未重新运行GUI、训练或全量Benchmark。系统未安装Poppler，尝试安装的pdftopng包在Windows构建失败，因此使用PyMuPDF逐页渲染；如本地已有Poppler，可用 `pdftoppm -png 报告.pdf 页面前缀` 独立复核。
