# AgentSim

**AI 第一用户、人类第二用户的轻量机器人仿真平台。**

AgentSim 把"世界"做成代码（World as Code：一份 YAML 即一个完整仿真世界），把"控制器"做成独立进程（Webots 风格的 `robot.step()` 循环），并通过标准 MCP（Model Context Protocol）把整个仿真内核暴露给 AI Agent——Agent 可以读世界、改世界、重跑、观察结果、再调整，形成完整的感知-决策闭环。

```
        ┌──────────────────────────────────────────────┐
        │                AI Agent (MCP)                 │
        │   read_world → 改 YAML → load_world → run     │
        └───────────────┬──────────────────────────────┘
                        │ stdio JSON-RPC (MCP)
        ┌───────────────▼──────────────────────────────┐
        │           MCP 工具层（8 个工具）              │
        │  load_world / run / get_observation / ...    │
        └───────────────┬──────────────────────────────┘
                        │
        ┌───────────────▼──────────────────────────────┐
        │        仿真内核（ZMQ ROUTER + 栅栏同步）       │
        │   World as Code: YAML → MjSpec → MuJoCo       │
        └───┬──────────────────────────────┬───────────┘
            │ spawn + REQ/REP 锁步          │
   ┌────────▼────────┐            ┌─────────▼────────┐
   │  控制器进程 #1   │            │   控制器进程 #N   │
   │ while robot.step│            │  （独立 OS 进程） │
   └─────────────────┘            └──────────────────┘
```

## 特性

- **World as Code**：一份 YAML 描述物理、机器人、任务、奖励，字段语义与 MJCF 对齐（`size`=半尺寸、`pos`=中心），有 MuJoCo 经验即可直接上手；错误信息带字段路径、期望值、实际值，人和 AI 都能直接看懂
- **进程级控制器隔离**：每个机器人的控制器是独立 OS 进程，通过 ZMQ REQ/ROUTER 与内核栅栏同步；控制器崩溃、超时都不会死锁内核
- **MCP 原生接入**：零依赖自实现的 stdio JSON-RPC MCP 服务器，任何支持 MCP 的 AI Agent（Claude、Trae 等）都能直接驱动仿真
- **Supervisor 能力**：save/load 内存快照（物理状态逐位恢复）、teleport、reset——快照包含 `mjSTATE_FULLPHYSICS | mjSTATE_WARMSTART`，保证轨迹级可复现
- **场景生成器**：5 个确定性模板（tabletop / push / maze / race / empty），同一命令 + 同一参数 = 逐字节相同的输出
- **零重量依赖**：核心只依赖 mujoco / numpy / pyyaml / pyzmq，MCP 层不依赖任何 SDK

## 安装

要求 Python 3.10+。

```bash
git clone https://github.com/jerryzhu-yc/agentsim.git
cd agentsim
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 快速开始

### 1. 跑一个内置任务（人类用户）

```bash
python -m agentsim run worlds/pick_cube.yaml
```

输出示例（机械臂侧扫推 cube 入成功区的端到端任务）：

```
世界 'pick_cube'：1 物体，1 机器人
运行 238 ticks（墙钟 0.112s，仿真 15.232s）
物理状态: 正常
任务: 成功（reward 0.0092）
  cube 最终位置: (+0.510, +0.376, +0.050)
```

加 `--json` 得到机器可读输出，加 `--ticks N` 限定最大步数。

### 2. 让 AI Agent 驱动仿真（MCP）

```bash
python -m agentsim mcp
```

在支持 MCP 的客户端里把它注册为 stdio 服务器，Agent 即可使用 8 个工具完成闭环：

| 工具 | 说明 |
|---|---|
| `load_world` | 加载世界（`yaml_path` 或 `yaml_text` 二选一），整体替换当前世界 |
| `run` | 推进仿真并返回摘要（ticks / success / reward / 物体终位）；`ticks=N` 只推进 N 步 |
| `get_observation` | 当前完整观察量：qpos / qvel / ctrl / body 位置 + 任务状态 |
| `get_robot_manifest` | 机器人设备清单：执行器、关节限位、body 名 |
| `save_state` / `load_state` | 内存快照，物理状态逐位恢复 |
| `reset` | 世界回到 YAML 初态（重编译 + 控制器进程重启） |
| `read_world` | 回读当前世界的 YAML 原文 |

Agent 的标准闭环：`read_world` → 修改 YAML → `load_world(yaml_text)` → `run` → `get_observation` → 观察成败 → 调整 → 重试。

> 语义提示：`run` 是一次性 episode——返回时本轮控制器已收尾退出；继续推进请 `reset` 后重跑，或改世界后重新 `load_world`。

### 3. 从模板生成场景

```bash
python -m agentsim scenegen maze --rows 4 --cols 4 --out worlds/my_maze.yaml
python -m agentsim scenegen race --lanes 3 --out worlds/my_race.yaml
python -m agentsim scenegen tabletop --objects 2 --out worlds/my_table.yaml
python -m agentsim run worlds/my_maze.yaml
```

| 模板 | 说明 | 主要参数 |
|---|---|---|
| `tabletop` | 桌面抓取/搬运：抬升区 + 若干 cube | `--objects`（1~3） |
| `push` | 推式任务（同 worlds/pick_cube.yaml 布局） | - |
| `maze` | 差速车走蛇形迷宫 | `--rows` `--cols`（2~8） |
| `race` | 多车竞速 | `--lanes`（2~3） |
| `empty` | 空世界（配 arm / diff_car / 无机器人） | `--robot` |

## 世界文件格式

一个完整的世界长这样（`worlds/pick_cube.yaml`，有删节）：

```yaml
name: pick_cube
timestep: 0.002        # 物理步长（秒）
ctrl_period: 32        # 一个控制 tick 的物理步数
gravity: [0, 0, -9.81]

scene:
  floor:
    size: [2.0, 2.0]      # 半尺寸
    friction: 0.3         # 滑动摩擦（MuJoCo 语义：接触对取 max，需双方都低才能滑）
  lights:
    - {pos: [0, 0, 3]}
  objects:
    - name: cube
      type: box            # box | sphere | cylinder
      size: [0.03, 0.035, 0.05]   # MJCF 半尺寸
      pos: [0.635, 0.15, 0.05]
      mass: 0.05
      friction: 0.3
      color: "#F59E0B"

robots:
  - name: arm
    model: ../models/demo_arm.xml   # MJCF 路径（相对本文件）
    pos: [0, 0, 0]
    controller: ../controllers/pick_place.py   # 控制器脚本（相对本文件）

task:
  success:                  # 成功判定：body 位置落入 zone
    body: cube
    zone: {min: [0.35, 0.25, 0.0], max: [0.65, 0.50, 0.12]}
  reward:                   # 距离/zone/接触 三种奖励项
    - distance: {a: cube, b: wrist, weight: -0.01}
    - zone: {body: cube, zone: {min: [...], max: [...]}, weight: 0.01}
  max_ticks: 5000

runtime:
  realtime: false           # 实时模式（true = 按墙钟限速）
  controller_timeout: 100   # 控制器响应超时（ms）
  timeout_policy: last_ctrl # last_ctrl=用上次控制量补位 | pause=原地等待
```

命名引用规则：物体名全局唯一；机器人 body 自动带机器人前缀（`arm/wrist`）。当机器人名与其 body 名不同时，任务里引用必须写全名（如 `car_a/car`）；同名时可用短名。

## 控制器 API（Webots 风格）

控制器是普通 Python 脚本，唯一要 import 的东西是 `Robot`：

```python
from agentsim.controller import Robot

robot = Robot()                     # 自动从 --sock/--robot 参数连接内核
while robot.step():                 # 推进一 tick，返回观测；内核叫停时返回 False
    lo, hi = robot.range("a1")      # 执行器控制范围
    robot.set("a1", 0.5)            # 按名字写控制量
    obs = robot.obs                 # {"tick","qpos","qvel","bodies","objects",...}
    cube = obs["objects"]["cube"]    # 场景物体全局可见
    snap = robot.supervise("save", tag="mid")   # 快照/读档/teleport
```

控制器与内核是独立进程锁步（每个 step 请求有且只有一个回复）。控制器可以无限等待内核开跑（Agent 的"思考时间"不会杀死控制器）；内核会在 run 结束时通知控制器收尾。

## 测试

```bash
python -m pytest tests/ -q
# 71 passed —— 覆盖 schema/builder/内核/多机器人/Supervisor 逐位重现/MCP/场景生成
```

其中包含真实 Agent 压测回归：控制器在 LLM 思考时间下存活、崩溃控制器在统计中可见、每轮 run 零进程泄漏。

## 项目结构

```
agentsim/
├── agentsim/
│   ├── kernel/sim.py      # 仿真内核：ZMQ ROUTER + 栅栏同步推进
│   ├── controller.py      # 控制器侧 Robot 客户端（用户唯一 API）
│   ├── world/schema.py    # YAML → dataclass，前置校验
│   ├── world/builder.py   # spec → MjSpec → MuJoCo 模型编译
│   ├── supervisor.py      # 快照/teleport（逐位可复现）
│   ├── mcp_server.py      # MCP stdio 服务器（零依赖 JSON-RPC）
│   ├── scene_gen.py       # 5 个确定性场景模板
│   └── cli.py             # run / mcp / scenegen 命令行
├── models/                # 机器人 MJCF（demo_arm 三指臂、diff_car 差速车）
├── controllers/           # 示例控制器（pick_place 侧扫推式相位机等）
├── worlds/                # 示例世界（pick_cube、twin_arms）
└── tests/                 # 71 项测试
```

## Roadmap

- [x] M0 仿真内核（物理 / 进程隔离 / Supervisor / MCP / 场景生成，71 项测试）
- [ ] M1 GUI（PySide6 可视化 + 实时观测面板）—— [实施方案](docs/M1_GUI实施计划.md)：GUI 作为 MCP 客户端（与 AI Agent 同构）+ MuJoCo 离屏渲染，内核已备好 `run(on_tick=)` 观测钩子
- [ ] M1 心跳机制（内核崩溃时控制器可感知退出）
- [ ] 更多机器人模型与传感器（关节力矩、触觉、相机）
- [ ] RL 训练接口（gym.Env 封装）

## 已知取舍（M0）

- 内核进程被强杀（kill -9）时控制器可能成为孤儿阻塞——正常退出路径（EOF / close / reset）都会收尾控制器
- 无渲染：M0 是无头仿真，可视化留给 M1 GUI
- MCP 实现为协议子集（initialize / ping / tools），覆盖主流客户端所需

---

如果你在做一个用 AI 操控机器人仿真的项目，欢迎 issue / PR。
