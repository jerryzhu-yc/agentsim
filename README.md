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
        │           MCP 工具层（9 个工具）              │
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
- **MCP 原生接入**：零依赖自实现的 stdio JSON-RPC MCP 服务器，任何支持 MCP 的 AI Agent（Claude、Trae 等）都能直接驱动仿真，包括让 Agent 用 `screenshot` 截取仿真画面"亲眼"看世界
- **传感器系统**：YAML 里给机器人声明 encoder / IMU / 触觉 / 关节力矩 / 相机五类传感器，编译进 MuJoCo 模型、读数进观测流，GUI 面板实时显示；传感器是只读派生量，不改变物理轨迹（bit 级守护测试保证）
- **机器人模型库**：MuJoCo Menagerie 官方模型一键接入——预配置 14 个著名机器人（Unitree Go1/Go2/H1/G1、ANYmal C、Spot、Franka Panda、UR5e/UR10e、xArm7、Shadow Hand、ALOHA 等），GUI 点"模型库"选模型即生成世界 + 控制器模板并自动打开编辑器，改完控制律点 Run 即跑；AI Agent 走同一条 `zoo_create` 工具
- **夹爪装配工作台**：内置 OnRobot 2FG7 平行夹爪，GUI"装备"页点选即可给 UR5e / UR10e / xArm7 / Panda 四臂挂爪/拆爪，并可视化挂点、关节、body 清单；同一页表单式添加五类传感器（encoder/IMU/触觉/关节力矩/相机），全部改 YAML 自动重载，不手写配置。夹爪安装位姿（mount）对四臂统一标定，与官方参考场景逐位吻合；自带"码垛分区"场景模板（UR5e+2FG7，A 区取料→B 区码放）
- **RL 训练接口**：一个 YAML 世界 = 一个确定性 gymnasium.Env——动作空间 = 执行器拼接、观测 = qpos/qvel/传感器、reset 逐位一致（同 YAML 跑两遍轨迹逐 bit 相同，eval 可复现），单进程直驱不起 ZMQ，训练吞吐优先
- **Supervisor 能力**：save/load 内存快照（物理状态逐位恢复）、teleport、reset——快照包含 `mjSTATE_FULLPHYSICS | mjSTATE_WARMSTART`，保证轨迹级可复现
- **场景生成器**：6 个确定性模板（tabletop / push / maze / race / pallet / empty），同一命令 + 同一参数 = 逐字节相同的输出
- **零重量依赖**：核心只依赖 mujoco / numpy / pyyaml / pyzmq，MCP 层不依赖任何 SDK

## 安装

要求 Python 3.10+。

```bash
git clone https://github.com/jerryzhu-yc/agentsim.git
cd agentsim
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Windows 部署

AgentSim 全链路 Windows 兼容：内核用 `tcp://127.0.0.1` 随机端口（无 `ipc://`，Windows 不支持）；控制器子进程走 `subprocess.Popen` + `sys.executable`（无 fork）；渲染后端代码已平台自适应——**Linux/macOS 默认 EGL，Windows 默认 wgl**（`mcp_server.py` / `gui/__init__.py` 里的 `setdefault`，用户可用环境变量 `MUJOCO_GL` 覆盖）。零配置，照下面做即可：

**1. 装 Python 3.10+**（python.org 安装包勾选 "Add python.exe to PATH"，或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)）

**2. 克隆 + 装依赖**（PowerShell 或 CMD）：

```powershell
git clone https://github.com/jerryzhu-yc/agentsim.git
cd agentsim
python -m venv .venv
.venv\Scripts\activate          # PowerShell 若报执行策略错误见下方 FAQ
pip install -r requirements.txt
```

所有依赖（mujoco / PySide6 / pyzmq / numpy）在 Windows 均有官方预编译 wheel，无需编译器。

**3. 验证安装**：

```powershell
python -m pytest tests/ -q      # 全绿即环境就绪（含 GUI/传感器/RL 全量回归）
python -m agentsim run worlds/pick_cube.yaml     # 跑一个内置任务
python -m agentsim gui                            # 桌面工作台
```

**4. 桌面快捷方式（可选，推荐）**：

仓库根目录带两个批处理（UTF-8 中文输出，自动定位 Python，无需手动激活 venv）：

```bat
create_desktop_shortcut.bat    # 双击一次：在桌面生成 AgentSim 图标（agentsim.ico）
start.bat                      # 双击即启动 GUI（快捷方式即指向它）
```

`start.bat` 按 `AGENTSIM_PYTHON` 环境变量 → 仓库内 `.venv` → PATH 里的 `python` → `py -3` 的顺序定位解释器；GUI 异常退出时窗口保留并提示装依赖，正常关闭时自动收窗。之后桌面双击 AgentSim 图标即可进入工作台，不用再开终端。

**Windows 常见问题（FAQ）**：

| 现象 | 处理 |
|---|---|
| PowerShell 激活 venv 报 "禁止运行脚本" | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 后重试，或改用 CMD |
| 首次运行弹 Windows 防火墙提示 | 点"允许"——内核 ZMQ 只绑 `127.0.0.1` 本机回环，不对外通信 |
| 远程桌面（RDP）断开后 GUI 崩溃 | RDP 会话的 GPU 上下文随会话注销销毁；用控制台会话（`qwinsta` 抢 session 1）或无 GUI 的 `python -m agentsim run` |
| 渲染报 `wgl` 相关错误（无 GPU 的服务器） | Windows Server 无 OpenGL 驱动，装 [Mesa3D 离屏驱动](https://mesa3d.org/)（或设 `MUJOCO_GL=osmesa`），或改用 Linux 容器跑无头测试 |
| `mujoco-menagerie` 模型首次下载慢/失败 | 模型资产由 `mujoco-menagerie` 包从 GitHub 下载到 `~/.cache/mujoco_menagerie/`；网络受限时配代理（`HTTPS_PROXY` 环境变量），或先在有网机器上下载后拷贝缓存目录 |
| 中文/空格路径 | 完全支持（pathlib 全程处理），但建议路径不含空格避免个别工具链问题 |

**可选依赖**（都是 `pip install`，Windows 同样直接装）：

```powershell
pip install stable-baselines3   # RL 训练示例（examples/train_ppo_go2.py）
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

在支持 MCP 的客户端里把它注册为 stdio 服务器，Agent 即可使用 9 个工具完成闭环：

| 工具 | 说明 |
|---|---|
| `load_world` | 加载世界（`yaml_path` 或 `yaml_text` 二选一），整体替换当前世界 |
| `run` | 推进仿真并返回摘要（ticks / success / reward / 物体终位）；`ticks=N` 只推进 N 步 |
| `get_observation` | 当前完整观察量：qpos / qvel / ctrl / body 位置 + 传感器读数 + 任务状态 |
| `get_robot_manifest` | 机器人设备清单：执行器、关节限位、挂点 site、body 名、传感器、相机、当前夹爪（含开合范围）+ 可用夹爪库目录 |
| `screenshot` | 把当前状态渲染成 PNG：缺省自由机位返回 image content，`camera` 指定已声明相机，`path` 写盘 |
| `save_state` / `load_state` | 内存快照，物理状态逐位恢复 |
| `reset` | 世界回到 YAML 初态（重编译 + 控制器进程重启） |
| `read_world` | 回读当前世界的 YAML 原文 |
| `save_world` | 当前世界（含 GUI/Agent 全部修改）保存为 YAML 文件；跨目录保存时相对资源路径自动改绝对，任何位置重开都有效 |
| `zoo_create` | 从模型库（MuJoCo Menagerie）生成可编辑项目：世界 YAML + 控制器模板（含执行器清单与 API 注释）；`key='list'` 只列清单 |

Agent 的标准闭环：`read_world` → 修改 YAML → `load_world(yaml_text)` → `run` → `get_observation` → 观察成败 → 调整 → 重试。

> 语义提示：`run` 是一次性 episode——返回时本轮控制器已收尾退出；继续推进请 `reset` 后重跑，或改世界后重新 `load_world`。

### 3. 桌面 GUI 实时可视化

```bash
pip install PySide6-Essentials   # GUI 可选依赖
python -m agentsim gui
```

> Windows 用户可免终端：双击仓库根的 `start.bat`，或先跑一次 `create_desktop_shortcut.bat` 生成桌面图标（见 [Windows 部署](#windows-部署)）。

GUI 本身就是一个 MCP 客户端（与 AI Agent 同构）：左键加载世界 YAML，后端经 stdio JSON-RPC 驱动同一个 `agentsim.mcp` 服务器——Run 时仿真跑在服务器后台线程，GUI 以 30Hz 拉取观测快照并渲染 MuJoCo 3D 画面 + 关节/物体面板。窗口里可以：

- **模型库**：选 Menagerie 官方模型（首次联网下载一次）→ 自动生成世界 + 控制器模板 → 自动加载并打开编辑器——点击-打开-编程闭环
- **装备页**：选中机器人后点选即可装配/移除 OnRobot 2FG7 夹爪（UR5e/UR10e/xArm7/Panda），并按类型表单添加/删除五类传感器——所有改动写回世界 YAML 并自动重载，无需手写
- **新建场景**（文件 → 新建场景，Ctrl+N）：选参数化模板（含码垛分区 pallet）→ 设参数 → 保存即生成并加载
- **保存世界**（Ctrl+S）：拖动/位姿/装备/添加机器人等全部修改写回 YAML 文件——改完的世界可以留到下次（内存世界会提示另存位置；跨目录保存时相对资源路径自动改绝对）
- **最近打开**（文件菜单）：最近 8 个世界一键重开（跨会话持久化）
- **添加物体 / 删除节点**（场景菜单）：往场景加方块/圆柱/球（类型/尺寸/颜色，放在相机注视点）；场景树右键或 Del 键删除选中的机器人/物体——搭场景不用手写 YAML
- **Load**：选择世界 YAML（解析在本地，坏文件直接报错）
- **Run / Run N**：跑一整个 episode（结束后可再 Run，自动 reset）
- **停止**：run 进行中随时可点（F6）——提前收尾当前 run，控制器正常退出、已跑轨迹保留，之后照常 Reset / 继续 Run
- **实时勾选框**：勾选=按墙钟实时播放动画（1 仿真秒≈1 墙钟秒，MCP `run` 的 `realtime` 参数）；取消=全速推进瞬间出结果（跑批验证）
- **Reset**：世界回初态；渲染视角支持鼠标拖拽旋转、滚轮缩放、**F 键聚焦选中对象**

> run 期间 `get_observation` 返回 33ms 节流的运动学快照（含物体位置 + 四元数），不影响物理轨迹。

### 4. 从模板生成场景

```bash
python -m agentsim scenegen maze --rows 4 --cols 4 --out worlds/my_maze.yaml
python -m agentsim scenegen race --lanes 3 --out worlds/my_race.yaml
python -m agentsim scenegen tabletop --objects 2 --out worlds/my_table.yaml
python -m agentsim scenegen pallet --cubes 3 --out worlds/pallet.yaml
python -m agentsim run worlds/my_maze.yaml
```

| 模板 | 说明 | 主要参数 |
|---|---|---|
| `tabletop` | 桌面抓取/搬运：抬升区 + 若干 cube | `--objects`（1~3） |
| `push` | 推式任务（同 worlds/pick_cube.yaml 布局） | - |
| `maze` | 差速车走蛇形迷宫 | `--rows` `--cols`（2~8） |
| `race` | 多车竞速 | `--lanes`（2~3） |
| `pallet` | 码垛分区：UR5e+2FG7 把 A 区 N 个 cube 逐个码到 B 区目标格（首次需联网下载一次 UR5e） | `--cubes`（1~6） |
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
    pos: [0, 0, 0.05]               # root 绝对位姿（模型自带 pos 被吸收）
    controller: ../controllers/pick_place.py   # 控制器脚本（相对本文件）
    gripper:                         # 可选：末端夹爪（库 key 或自定义 MJCF 路径）
      model: onrobot_2fg7            # OnRobot 2FG7，适配四臂（见下）
    sensors:                         # 可选：五类传感器，编译进 MuJoCo
      - {name: j1_enc, type: encoder, joint: j1}          # 关节位置
      - {name: j1_tau, type: joint_torque, joint: j1}     # 关节约束力
      - {name: wrist_imu, type: imu, body: wrist, pos: [0, 0, 0.02]}   # 6 维
      - {name: fl_touch, type: tactile, body: finger_l, pos: [0, 0, 0]}
      - {name: gripper_cam, type: camera, body: wrist,
         pos: [0, 0.06, 0.10], quat: [0.7071, 0.7071, 0, 0]}

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

传感器说明：

- **类型与挂点**：`encoder` / `joint_torque` 挂 `joint`；`imu` / `tactile` / `camera` 挂 `body`（`pos` 为 body 局部系挂点偏移，`quat` 仅 camera 用于朝向）
- **读数进观测流**：obs 里 `sensors` 键为 `{名字: 读数}`，标量（encoder/joint_torque/tactile）是 float，IMU 是 6 维列表 `[陀螺 xyz, 加速度 xyz]`；相机不产生数值读数，只作 `screenshot` 的 viewpoint
- **tactile 的 `radius`**：MuJoCo 3.x 的 touch 只统计接触点落在以挂点为球心、`radius` 为半径的球内的接触力——`radius` 要罩住挂点 body 的预期接触区，默认 0.06（约指尖/指板整面大小）
- **实现映射**：encoder→JOINTPOS、joint_torque→JOINTACTFRC、imu→GYRO+ACCELEROMETER、tactile→TOUCH 标量（不是需要 plugin 的 mjSENS_TACTILE 网格阵列）；传感器是只读派生量，加不加传感器物理轨迹逐位一致（有 bit 级守护测试）

## 末端夹爪装配（OnRobot 2FG7）

给机器人挂夹爪有两条等价路径：

1. **GUI（推荐）**：加载世界 → 右侧"装备"页 → 选机器人 → 夹爪下拉选 `OnRobot 2FG7` → "应用夹爪"；传感器也在同页表单里点选添加。改动写回世界 YAML 并自动重载。
2. **YAML**：机器人下加 `gripper` 字段（见上例）。

```yaml
gripper:
  model: onrobot_2fg7      # 库 key；也可写自定义 MJCF（相对世界文件的路径）
  site: attachment_site    # 可选：挂点 site，缺省用该夹爪标定的 attachment_site
  pos: [0, 0, 0]           # 可选：覆盖标定安装平移（米，挂点局部系）
  quat: [1, 0, 0, 0]       # 可选：覆盖标定安装朝向（wxyz）
```

要点：

- **适配机型**：`ur5e.xml` / `ur10e.xml` / `xarm7_nohand.xml` / `panda_nohand.xml` 四臂。安装位姿（mount）用同一套标定，以官方 OnRobot workbench 参考场景为黄金标准逐位校准（挂点 site → 夹爪 TCP 变换四臂一致）；未标定的机器人会报清晰错误。
- **设备命名**：挂爪后夹爪设备带 `gripper/` 前缀并入该机器人设备索引——执行器 `gripper/gripper_act`、两滑动关节 `gripper/left_slide` / `gripper/right_slide`、body `gripper/...`。传感器可引用夹爪手指 body（如 `body: gripper/left_finger`），指尖预置了 `gripper/left_pad_site` / `gripper/right_pad_site`，TCP 为 `gripper/gripper_tcp`。
- **开合控制**：夹爪是一个 position 执行器，`ctrlrange 0~0.038`（0 全开、0.038 闭合，两指对向耦合）：

```python
robot.set("gripper/gripper_act", 0.0)    # 张开
robot.set("gripper/gripper_act", 0.038)  # 闭合抓取
```

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
    touch = robot.sensor("fl_touch")           # 传感器读数（世界 YAML 里声明过）
    imu = robot.sensor("wrist_imu")            # 6 维 IMU 返回 [gx,gy,gz,ax,ay,az]
    snap = robot.supervise("save", tag="mid")   # 快照/读档/teleport
```

控制器与内核是独立进程锁步（每个 step 请求有且只有一个回复）。控制器可以无限等待内核开跑（Agent 的"思考时间"不会杀死控制器）；内核会在 run 结束时通知控制器收尾。

## RL 训练接口（gymnasium）

一个 YAML 世界 = 一个确定性 gym 环境。`reset` 重建 `MjData`，同 YAML 跑两遍轨迹逐 bit 相同——eval 可复现是 AgentSim 的差异化卖点：

```python
from agentsim.rl_env import AgentSimEnv

env = AgentSimEnv("worlds/pick_cube.yaml")     # YAML 路径 / dict / WorldSpec
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
# action  = 所有机器人执行器按声明顺序拼接（Box，界 = 各自 ctrlrange）
# obs     = 各机器人 qpos + qvel + 传感器读数拼接（float32）
# reward  = 世界 task.reward 的当步加权值；terminated = task.success；
#           truncated = tick 到 max_ticks
# info    = {"tick", "time", "raw": {机器人: {qpos/qvel/ctrl/sensors}}}（float64）
```

设计要点：

- **单进程直驱**：不走控制器子进程、不起 ZMQ，训练吞吐优先；MCP/内核零改动
- **语义一处实现**：任务评估与观测组装和内核共用 `kernel/task.py`，两路不会漂移
- **官方合规**：过 `gymnasium.utils.env_checker.check_env`；`render_mode="rgb_array"` 返回 (600,800,3) 截图（自由机位），SB3/CleanRL 等标准算法库可直接接
- **确定性守护测试**：同动作序列两个 episode 逐 tick qpos `array_equal`（禁 allclose）

## 机器人模型库（MuJoCo Menagerie）

预配置 14 个著名机器人（`agentsim/zoo.py`）：Unitree Go1 / Go2 / H1 / G1、ANYmal C、Boston Dynamics Spot、Franka Emika Panda、UR5e / UR10e、xArm7、Shadow Hand、ALOHA、Google Robot、Stretch 3——全部来自 DeepMind 官方精选模型库，经 MjSpec attach 编译进 AgentSim 世界（浮动基座四足/人形实测落地稳定）。

**GUI 闭环**（`python -m agentsim gui`）：点"模型库"按钮 → 选模型（首次使用联网下载一次，之后永久离线）→ 自动生成世界 + 控制器模板 → 自动加载并打开编辑器 → 改完控制律点 Run 重跑。

**Agent / CLI 闭环**（与 GUI 同一条 MCP 工具）：

```python
r = tools.zoo_create(key="unitree_go2")   # 或 key="list" 先列清单
# -> {"world": "~/.agentsim/zoo/unitree_go2/world.yaml",
#     "controller": ".../controller.py", "actuators": [...12 个...]}
tools.load_world(yaml_path=r["world"])
tools.run(ticks=200)
```

生成的 `controller.py` 模板带完整执行器清单与 API 注释——AI Agent 可以直接读清单后改写控制律，人类用户用任意编辑器改写。依赖官方包 `mujoco-menagerie`（`pip install mujoco-menagerie`），模型资产缓存在 `~/.cache/mujoco_menagerie/`。

> 已知语义：attach 时子模型的 MuJoCo option（impratio / cone 等）被父世界值覆盖——接触求解参数以世界 YAML 为准（`worlds/zoo` 生成模板已按各模型类目调好 floor 摩擦）。

## 桌面 GUI 智能工作台（M5）

GUI（`python -m agentsim gui`）从可视化扩展为完整的机器人编程工作台——左右四块面板（场景树/属性/代码/Agent），配合模型库的运动演示与位姿编辑，覆盖"打开/写代码/调位姿/跑/让 AI 来"全闭环。

### 1. 运动演示一键播放

模型库窗口选中任意机器人 → "生成演示并运行"按钮，自动生成内置运动演示并加载世界 → 自动 Run。14 个模型全部预配置运动（见 `agentsim/zoo.py`）：

- **四足小跑（trot）**：Unitree Go1 / Go2、ANYmal C、Boston Dynamics Spot——按对角腿相位（FL-RR / FR-RL）交替摆动；CPG 控制律 `freq`/`amp`/`lag` 在 `zoo.py` 顶部 `_trot` 工厂生成，落地即稳定行走
- **机械臂扫掠**：Franka Panda / UR5e / UR10e / xArm7——7（或 6）关节正弦相位机，base 绕 z 旋转 + 末端周期摆动
- **人形摆臂**：Unitree H1 / G1——髋膝肩肘相位差摆动模拟踏步
- **差速车**：两轮差速车走蛇形前进
- **静止**：Shadow Hand / ALOHA / Google Robot / Stretch 3——多指/双臂姿态展示

> 四足稳定行走的根因：MuJoCo `attach` 时父世界（默认 Euler 积分器）的 option 会覆盖子模型原值——Spot 原生 `solref=0.004 1` 超硬接触用 `implicitfast`，被 Euler 覆盖后"硬接触+显式积分"会数值震荡弹跳。`WorldSpec` 已加 `integrator` 字段，四足 demo 世界统一设 `implicitfast`。详见 `zoo.py` 中各模型的根因注释。

### 2. 机器人/物体位姿编辑（属性面板 + 3D 拖动）

在场景树选中机器人**或物体**节点 → 右侧"属性"面板出现位姿组：3 个 xyz 双精度 SpinBox（步进 0.01m，范围 ±20m）+ 3 个 roll/pitch/yaw 角度框（步进 5°，范围 ±360°）→ "应用位姿"按钮：

- 欧拉角与四元数双向转换（`panels.quat_to_euler` / `euler_to_quat`，ZYX 内旋序）
- 单位朝向（1,0,0,0）时 YAML 里 `quat` 字段省略（不写就是默认）
- 应用流程：UI → `read_world` MCP 工具拿当前 YAML → 改 `pos`/`quat` 字段 → `load_world` 重载 → 自动重选该节点
- **跑完演示（FINISHED）后也能直接编辑**——提交走 `load_world` 全量重载，世界自动回到 YAML 初态；仅运行中（RUNNING）禁止编辑并给出提示

3D 视口内的拖动（编辑态）：

- **左键拖动**：水平面（XY）内移动选中对象
- **Ctrl+左键拖动**：沿 Z 轴调高度（向上拖=升）
- 拖动时 HUD 左上实时显示选中对象的**三维世界坐标**，松手写入世界 YAML 并重载

机器人模型自带 root body 的 pos/quat 会被 `build_world._attach_robot` 清零——`spec.pos`/`spec.quat` 就是机器人 root 的**绝对世界位姿**。例如 `demo_arm.xml` 的 base 自带 0.05m 高度，世界 YAML 里要写 `pos: [0, 0, 0.05]`（吸收内部偏移）。

### 3. GUI 内置代码编辑器（代码页）

GUI 右侧第四个 tab"代码"：一个 Python 语法高亮编辑器（`PyHighlighter`——QSyntaxHighlighter 零依赖近似实现，字符串/注释/关键字/数字四类着色），下拉选当前世界的机器人 → 加载其控制器文件全文 → 编辑 → "保存并运行"按钮或 `Ctrl+S` 快捷键：

- 保存路径：世界 YAML `base_dir` + 机器人 `controller` 字段
- 保存后流程：写文件 → `read_world` 重载世界 → 自动 Run（`_pending_run` 状态标记 + 重载完成后 `if self.state.can("run"): self._do_run(...)`）
- 没绑控制器的机器人显示"没有绑定控制器"提示

可以同时打开任意编辑器改 `.py` 后回到 GUI 点 Run——GUI 内编辑只是更顺手。

### 4. Agent 控制台（Agent 页）

GUI 右侧第五个 tab"Agent"：聊天式下达任务给 AI（OpenAI 兼容 chat/completions API，支持本地 Ollama / OpenAI / DeepSeek / 智谱等任何 OpenAI 兼容服务）。点"配置"填三项：

- `base_url`：例如 `https://api.openai.com/v1` 或 `http://localhost:11434/v1`
- `api_key`：本地服务可空（Ollama），公开服务必填；密码模式输入
- `model`：例如 `gpt-4o-mini`、`deepseek-chat`、`glm-4`

配置保存到 `~/.agentsim/agent.json`，`chmod 0600` 保护 api_key。配好后聊天下达任务（Enter 发送，Shift+Enter 换行）：

- Agent 拿到的是 OpenAI function calling 格式工具清单（`bridge.to_openai_tools`）——MCP 的 11 个工具（含 `stop` 停止仿真）+ 2 个本地文件工具（`write_file`/`read_file`，限定在世界目录内防越界）
- 工具循环：Agent 返回文本 → 显示在聊天框；返回 tool_calls → MCP 同步等回复（QEventLoop + 30s QTimer 兜底）→ 显示工具调用与结果 → 继续下一轮直到 Agent 不再调工具
- 系统提示词包含：身份（机器人仿真助手）、当前世界名、所有工具清单（MCP + 本地）、建议工作流（read_world → load_world → run → get_observation）

让 Agent 干的事例：生成一个抓取场景、写一个控制器让臂推 cube、跑起来看结果——Agent 直接读 MCP 工具结果自循环完成。

### 5. PPO 训练示例

`examples/train_ppo_go2.py`：用 stable-baselines3 PPO 在 Go2 上训练行走策略（demo CPG 已经能让 Go2 走 10m，这里做 RL 微调）。sb3 是可选依赖——不装时脚本会给出 `pip install stable-baselines3` 提示后退出，不在回归路径里。

```bash
pip install stable-baselines3
~/python examples/train_ppo_go2.py
# 200k 步训练，输出 models/ppo_go2.zip
```

奖励设计：`forward（5× dz_root）+ 高度罚（-3 × |z - 0.30|）+ 能耗正则（-0.01 × Σ a²）+ 跌倒终止（rootz < 0.15）`。世界用 `zoo.generate("unitree_go2")` 生成 + 删 controller（env 直驱）+ 保留 `integrator=implicitfast` / `impratio=100` / `ctrl_period=8`——四足稳定的关键参数。

## 测试

```bash
python -m pytest tests/ -q
# 206 passed —— 覆盖 schema/builder/内核/多机器人/Supervisor 逐位重现/
#              MCP（含 screenshot/zoo_create/stop/save_world 往返）/
#              传感器（含 bit 级轨迹守护）/
#              RL 环境（含逐位一致性守护）/控制器心跳（含跨平台死亡探测）/
#              握手鲁棒性（超时终止/握手期停止/stderr 直通）/模型库/场景生成/
#              GUI（含深色主题守护、浮动基座 root 位姿同步、三维拖动/
#              FINISHED 可编辑/添加物体对话框）/ M5 运动演示 /
#              位姿 quat 链路 / 代码编辑器 / Agent 控制台后端 /
#              M6 2FG7 夹爪（四臂装配/索引/物理开合/触觉挂手指）/ 装备面板 /
#              码垛分区 pallet / ur10e 入库
```

其中包含真实 Agent 压测回归：控制器在 LLM 思考时间下存活、崩溃控制器在统计中可见、每轮 run 零进程泄漏。M5 新增 13 项测试覆盖 demo 生成（全覆盖 + go2 600 tick 行走 >1m + panda 关节活动度）、quat 链路（attach + 欧拉↔四元数 roundtrip + 属性面板应用）、PyHighlighter 四类着色 + CodePanel 保存信号、Agent 后端（config 0600 + chat_url 兼容 + MCP↔OpenAI 工具转换 + 本地文件越界防护）。停止功能另加 5 项：内核 `stop_check` 三态（带控制器提前收尾不泄漏 / 无控制器空跑可停 / 恒 False 行为不变）、MCP stop 工具（run 中停止 + 无 run 无操作）、GUI 状态机（停止仅 RUNNING 可用）。

## 项目结构

```
agentsim/
├── agentsim/
│   ├── kernel/sim.py      # 仿真内核：ZMQ ROUTER + 栅栏同步推进
│   ├── kernel/task.py    # 任务评估 + 观测组装纯函数（内核与 RL env 共用）
│   ├── rl_env.py          # gymnasium.Env 包装（RL 训练接口）
│   ├── zoo.py              # 机器人模型库（Menagerie 14 个预配置模型 + 运动演示生成器）
│   ├── grippers.py         # 夹爪库（OnRobot 2FG7：注册表/资产/四臂 mount 标定）
│   ├── controller.py      # 控制器侧 Robot 客户端（用户唯一 API）
│   ├── world/schema.py    # YAML → dataclass，前置校验（含 gripper/integrator/quat）
│   ├── world/builder.py   # spec → MjSpec → MuJoCo 模型编译（机器人+夹爪统一装配）
│   ├── supervisor.py      # 快照/teleport（逐位可复现）
│   ├── mcp_server.py      # MCP stdio 服务器（零依赖 JSON-RPC）
│   ├── render.py          # 离屏渲染 + 手写 PNG 编码（screenshot 实现层）
│   ├── agent/             # GUI Agent 控制台后端（config/bridge/local_tools/worker）
│   ├── gui/               # PySide6 桌面工作台（app/render_view/panels[含装备页]/mcp_client）
│   ├── scene_gen.py       # 6 个确定性场景模板（含码垛分区 pallet）
│   └── cli.py             # run / mcp / gui / scenegen 命令行
├── models/                # 机器人 MJCF（demo_arm 三指臂、diff_car）+ grippers/2FG7
├── controllers/           # 示例控制器（pick_place 侧扫推式相位机等）
├── worlds/                # 示例世界（pick_cube、twin_arms）
├── examples/             # PPO 训练示例（train_ppo_go2.py，sb3 可选依赖）
├── start.bat / create_desktop_shortcut.bat / agentsim.ico
│                            # Windows 双击启动 + 桌面快捷方式（项目图标）
└── tests/                 # 206 项测试（含 M6 夹爪装配/装备面板/码垛场景/ur10e）
```

## Roadmap

- [x] M0 仿真内核（物理 / 进程隔离 / Supervisor / MCP / 场景生成）
- [x] M1 GUI（PySide6 可视化 + 实时观测面板）—— [实施方案](docs/M1_GUI实施计划.md)：GUI 作为 MCP 客户端（与 AI Agent 同构）+ MuJoCo 离屏渲染；run 后台线程化，30Hz 观测快照（`python -m agentsim gui`）
- [x] M2 传感器系统 + MCP screenshot —— 五类传感器（encoder / IMU / 触觉 / 关节力矩 / 相机）YAML 声明进观测流，bit 级守护保证不改变物理轨迹；`screenshot` 工具让 Agent 截取仿真画面（自由机位或声明的相机机位）
- [x] M3 RL 训练接口 —— `AgentSimEnv`（gymnasium.Env）：YAML 世界即确定性环境，动作/观测/奖励语义与内核一处实现（`kernel/task.py`），同 YAML 两遍轨迹逐位一致
- [x] M1 心跳机制 —— 控制器 `_recv` 100ms poller 轮询 + `getppid` 孤儿检测：内核进程一死控制器必被 init/subreaper 收养（父 pid 变化），随即 `RuntimeError("内核已退出")` 退出；思考时间无限等待语义不变
- [x] M4 机器人模型库 —— MuJoCo Menagerie 官方模型一键接入（14 个预配置著名机器人，M6 新增 UR10e），`zoo_create` 生成世界 + 控制器模板，GUI 点击-打开-编程闭环
- [x] M5 运动演示与智能工作台 —— 14 个模型一键运动演示（四足 trot / 机械臂扫掠 / 人形摆臂）+ GUI 位姿编辑面板（xyz+欧拉角）+ 内置 Python 代码编辑器（保存并运行）+ Agent 控制台（OpenAI 兼容 chat API 调用，AI 直接生成场景/写代码/运行验证）+ PPO 训练示例（`examples/train_ppo_go2.py`）
- [x] M6 夹具/传感器装配工作台 —— OnRobot 2FG7 夹爪入库（site-attach 到 UR5e/UR10e/xArm7/Panda，mount 对四臂统一标定并与官方参考场景逐位吻合）+ GUI"装备"页一键挂爪/拆爪与五类传感器表单式增删（自动改 YAML 重载）+ 码垛分区场景模板 pallet（UR5e+2FG7，A 区取料→B 区码放，IK 验证全部取放点可达）+ GUI"新建场景"对话框 + UR10e 入库

## 已知取舍（M0）

- ~~内核进程被强杀（kill -9）时控制器可能成为孤儿阻塞~~ **已由 M1 心跳机制补全**（`tests/test_heartbeat.py`：强杀内核后控制器 ≤2s 退出）
- 无渲染：M0 是无头仿真，可视化留给 M1 GUI
- MCP 实现为协议子集（initialize / ping / tools），覆盖主流客户端所需

---

如果你在做一个用 AI 操控机器人仿真的项目，欢迎 issue / PR。
