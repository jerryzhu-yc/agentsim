# M1 GUI 实施计划（PySide6 可视化 + 实时观测面板）

## Context

M0 已收官并开源（71 测试全绿）。按 Roadmap 与 M0 规格文档定位（"M1: PySide6 GUI 只读 + 控制命令接内核；GUI 只是给人类开的窗口"），下一步交付 M1.0 GUI 骨架：加载世界、3D 实时视口、观测面板、run/reset 控制。

**核心架构故事**：GUI 是 MCP 客户端（spawn `python -m agentsim mcp` 子进程为后端），与 AI Agent 同构——"GUI 也只是个客户端"。

**已验证的环境事实**：mujoco 3.15 离屏渲染 OK（`Renderer.update_scene().render()`）；PySide6 未装（用 PySide6-Essentials，~150MB）。

**关键前提修正**（设计发现）：`serve()` 单线程逐行循环——run 阻塞期间无法应答 get_observation，GUI 实时看过程必须先做 MCP 两项小增强。

## 实施步骤

### 步骤 1：MCP/内核两项增强（其余全依赖它）

1. **物体 quat**：`agentsim/kernel/sim.py` `_obs_msg` 的 objects 循环加 `data.xquat[bid]`（wxyz，供渲染写 freejoint 7 维）
2. **run 后台线程化**：
   - `Simulation.run()` 加可选 `on_tick` 回调（每 `_advance` 后调用，约 5 行）
   - `agentsim/mcp_server.py`：run 工具在后台线程执行；线程安全 obs 快照（on_tick 里写 dict + 锁）；run 期间 get_observation 返回快照；stdout 写出加锁；并发 run 以 WorldError 拒绝
   - 测试：run 期间 get_observation 可响应且 tick 在涨；物体 quat 在（tests/test_mcp.py 追加，沿用会话层模式）

### 步骤 2：MCP 客户端（不依赖 Qt 的部分可单测）

新文件 `agentsim/gui/mcp_client.py`：
- QProcess 托管后端（`python -m agentsim mcp`），readyRead 信号驱动
- 行协议编解码：输入累积缓冲按 `\n` 切分防半行；响应**按 id 匹配**（run 的响应会晚于其后请求，不能按序）
- 信号 `toolFinished(id, isError, text)`，不做阻塞 Future
- 纯 Python 的 Codec/PendingMap 部分无头单测（tests/test_gui.py 新建）

### 步骤 3：渲染视口（先脱离 GUI 渲出一帧）

新文件 `agentsim/gui/render_view.py`：
- 本地 `build_world(spec)` 编译只读渲染模型（不起 Simulation 进程，零 ZMQ 开销）
- `apply_observation(obs)`：机器人各关节按 joint_names 对齐 qpos（`RobotIndex.joints` 的 jid → `jnt_qposadr`，宽度按 jnt_type：hinge/slide=1、ball=4、free=7）；物体按 body 的 `body_jntadr` 找 freejoint → `jnt_qposadr` 写 7 维 [xyz, quat(wxyz)] → `mj_forward`
- QTimer 30Hz 节流：仅 dirty 时发 get_observation；上一帧未回则跳过防堆积
- `Renderer` 离屏渲染 → QImage(Format_RGB888)；持久 MjvCamera + 鼠标改 az/el/dist
- 无头单测：build_world → apply → render 断言帧形状（不建 QApplication）

### 步骤 4：主窗口组装

新文件 `agentsim/gui/panels.py`、`agentsim/gui/state.py`、`agentsim/gui/app.py`、`agentsim/gui/__init__.py`：
- UI 状态机（纯逻辑可测）：EMPTY（仅 Load）→ IDLE（全可用）→ RUNNING（全禁用，realtime 世界可看过程）→ FINISHED（Load/Reset 可用；**点 Run 自动 reset 再 run** 的便利设计）
- 布局：QMainWindow 水平分割。左 3D 视口；右侧竖排：世界信息（load_world 返回值）/ 任务状态（tick/time/success/reward 大字）/ 观测树（QTreeWidget：逐关节 qpos/ctrl + 物体 xyz+quat）/ 控制组（Load 文件对话框、Reset、Run 终态、Run N ticks 带 SpinBox、状态标签）

### 步骤 5：CLI 接入 + 手动验收

- `agentsim/cli.py` 加 `gui` 子命令（lazy import PySide6，保持核心零依赖）
- requirements.txt 加 `PySide6-Essentials`（注释标注仅 GUI 需要）
- 手动验收清单：pick_cube 全速跑看终态、realtime 版看过程、reset 重跑、坏 YAML 报错可读

## 测试与验证

- 内核/MCP 增强：tests/test_mcp.py 追加（run 期间 get_observation 响应 + quat）
- GUI 包：QT_QPA_PLATFORM=offscreen 冒烟 import；Codec/PendingMap/状态机纯 Python 单测；render_view 无头渲一帧断言
- 全量回归：`python -m pytest tests/ -q` 全绿
- 视觉效果：手动验收（用户跑 `python -m agentsim gui`）

## 关键文件

- 改：`agentsim/kernel/sim.py`（on_tick + xquat）、`agentsim/mcp_server.py`（run 线程化 + 快照）、`agentsim/cli.py`（gui 子命令）、`requirements.txt`
- 新：`agentsim/gui/{__init__,mcp_client,render_view,panels,state,app}.py`、`tests/test_gui.py`
- 复用：`world/builder.py` 的 `build_world`/`NameIndex`（渲染模型编译与索引）、test_mcp.py 的 stdio 子进程测试模式

## 风险与缓解

- PySide6 体积 → 装 Essentials 避开 WebEngine
- 帧率 → 640×480@30Hz 已验证无压力；未回帧跳过防堆积
- realtime run 无法中途取消 → M1 接受（pick_cube ≈15s），stop_run 留 M1.2
- 前后端状态不一致 → GUI 独占后端进程；每次状态转换强拉一次 obs 对齐，显示以 obs 为准
