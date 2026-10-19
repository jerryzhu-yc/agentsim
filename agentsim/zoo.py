"""机器人模型库：MuJoCo Menagerie 官方模型一键接入（M4）。

预配置 13 个著名机器人（四足/人形/机械臂/灵巧手/双臂/移动操作）。
generate() 生成可编辑的世界 YAML + 控制器模板——用户改控制律即编程，
GUI/CLI/MCP zoo_create 三路同构。

依赖：pip install mujoco-menagerie（官方包；首次用某模型时下载一次到
~/.cache/mujoco_menagerie/，之后永久离线可用）。
"""
from __future__ import annotations

import pathlib

import yaml

from agentsim.errors import WorldError

# 预配置清单：key = menagerie key；xml = 纯机器人入口（不带 menagerie 的
# floor/light——场景由 AgentSim 世界提供）；pos = attach 初始高度
# （浮动基座要离地一点落下；固定基座全零）
MODELS: dict[str, dict] = {
    # ---- 四足机器人 ----
    "unitree_go1": {
        "title": "Unitree Go1", "cat": "四足", "xml": "go1.xml",
        "pos": [0, 0, 0.5], "desc": "宇树 Go1，12 关节（hip/thigh/calf × 4 腿）"},
    "unitree_go2": {
        "title": "Unitree Go2", "cat": "四足", "xml": "go2.xml",
        "pos": [0, 0, 0.5], "desc": "宇树 Go2，12 关节（hip/thigh/calf × 4 腿）"},
    "anybotics_anymal_c": {
        "title": "ANYmal C", "cat": "四足", "xml": "anymal_c.xml",
        "pos": [0, 0, 0.6], "desc": "ANYbotics 工业四足，12 关节"},
    "boston_dynamics_spot": {
        "title": "Boston Dynamics Spot", "cat": "四足", "xml": "spot.xml",
        "pos": [0, 0, 0.6], "desc": "Spot 四足（带 arm 变体在 menagerie 里）"},
    # ---- 人形机器人 ----
    "unitree_h1": {
        "title": "Unitree H1", "cat": "人形", "xml": "h1.xml",
        "pos": [0, 0, 1.3], "desc": "宇树人形，19 自由度"},
    "unitree_g1": {
        "title": "Unitree G1", "cat": "人形", "xml": "g1.xml",
        "pos": [0, 0, 0.8], "desc": "宇树 G1 人形，29 自由度"},
    # ---- 机械臂 ----
    "franka_emika_panda": {
        "title": "Franka Emika Panda", "cat": "机械臂", "xml": "panda_nohand.xml",
        "pos": [0, 0, 0], "desc": "Franka Panda 7 轴臂（含手变体 panda.xml）"},
    "universal_robots_ur5e": {
        "title": "Universal Robots UR5e", "cat": "机械臂", "xml": "ur5e.xml",
        "pos": [0, 0, 0], "desc": "UR5e 6 轴协作臂"},
    "ufactory_xarm7": {
        "title": "UFACTORY xArm7", "cat": "机械臂", "xml": "xarm7_nohand.xml",
        "pos": [0, 0, 0], "desc": "xArm7 7 轴臂（含手变体 xarm7.xml）"},
    # ---- 灵巧手 ----
    "shadow_hand": {
        "title": "Shadow Hand E3M5", "cat": "灵巧手", "xml": "left_hand.xml",
        "pos": [0, 0, 0.3], "desc": "Shadow 灵巧手 20 关节（浮动基座）"},
    # ---- 双臂 ----
    "aloha": {
        "title": "ALOHA", "cat": "双臂", "xml": "aloha.xml",
        "pos": [0, 0, 0], "desc": "ALOHA 双臂遥操作平台，16 自由度"},
    # ---- 移动操作 ----
    "google_robot": {
        "title": "Google Robot", "cat": "移动操作", "xml": "robot.xml",
        "pos": [0, 0, 0], "desc": "Everyday Robots 移动操作平台"},
    "hello_robot_stretch_3": {
        "title": "Hello Robot Stretch 3", "cat": "移动操作", "xml": "stretch.xml",
        "pos": [0, 0, 0], "desc": "Stretch 3 家庭服务移动机器人"},
}


def ensure_model(key: str) -> pathlib.Path:
    """确保模型 XML 在本地缓存（未下载则触发一次下载），返回其路径。"""
    if key not in MODELS:
        raise WorldError(f"模型库没有 '{key}'，可用: {sorted(MODELS)}")
    try:
        import mujoco_menagerie as mm
    except ImportError:
        raise WorldError(
            "模型库依赖 mujoco-menagerie，请安装：pip install mujoco-menagerie"
        ) from None
    entry = mm.get(key)
    xml_name = MODELS[key]["xml"]
    try:
        hits = [f for f in entry.files() if f.name == xml_name and f.exists()]
    except Exception:
        hits = []
    if not hits:                      # files() 未落盘时走正式下载路径
        entry.model()                 # 下载 + 编译（一次，之后永久离线）
        hits = [f for f in entry.files() if f.name == xml_name]
    if not hits:
        raise WorldError(f"menagerie 包里没有 '{key}' 的 {xml_name}")
    return hits[0]


def generate(key: str, out_dir: str | pathlib.Path | None = None) -> dict:
    """生成可编辑的模型项目（world.yaml + controller.py 模板）。

    默认输出 ~/.agentsim/zoo/<key>/；返回 {model, world, controller}。
    """
    xml = ensure_model(key)
    meta = MODELS[key]
    out = pathlib.Path(out_dir) if out_dir else (
        pathlib.Path.home() / ".agentsim" / "zoo" / key)
    out.mkdir(parents=True, exist_ok=True)

    # 编译一次拿权威执行器清单（attach 后的名字，与 hello 下发完全一致）
    from agentsim.world.builder import build_world
    from agentsim.world.schema import parse_world
    world = {
        "name": key, "timestep": 0.002, "ctrl_period": 32,
        "gravity": [0, 0, -9.81],
        "scene": {"floor": {"size": [4.0, 4.0], "texture": "grid",
                            "friction": 0.8},
                  "lights": [{"pos": [0, 0, 3]}]},
        "robots": [{"name": key, "pos": meta["pos"],
                    "model": str(xml), "controller": "controller.py"}],
        "task": {"max_ticks": 3000},
        "runtime": {"realtime": False, "controller_timeout": 100,
                    "timeout_policy": "last_ctrl"},
    }
    spec = parse_world(world)
    _model, index = build_world(spec)         # 编译校验 + 索引
    actuators = list(index.robots[key].actuators)

    world_path = out / "world.yaml"
    with open(world_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(world, f, allow_unicode=True, sort_keys=False,
                       default_flow_style=None)

    ctrl_path = out / "controller.py"
    ctrl_path.write_text(_controller_template(meta, actuators),
                         encoding="utf-8")

    return {"model": key, "title": meta["title"], "world": str(world_path),
            "controller": str(ctrl_path), "actuators": actuators}


def _controller_template(meta: dict, actuators: list[str]) -> str:
    """控制器模板：清单 + 注释化的 API 演示，用户改控制律。"""
    act_lines = ",\n#   ".join(", ".join(actuators[i:i + 4])
                              for i in range(0, len(actuators), 4))
    return f'''"""{meta['title']}（{meta['cat']}）—— {meta['desc']}

编辑这个文件写你的控制律，然后在 GUI 点 Run 重跑（或 CLI 重新 run）。

可用执行器（{len(actuators)} 个，{meta['cat']}）：
#   {act_lines}

常用 API（agentsim.controller.Robot）：
#   robot.set("执行器名", 目标值)     写目标（下个 tick 生效）
#   robot.set_all([v1, v2, ...])      按上面顺序一次写全部
#   robot.range("执行器名")           查范围 (lo, hi)
#   obs = robot.obs                   最近观测 {{"qpos"/"qvel"/"bodies"/"sensors"}}
#   robot.step() 返回 False           内核叫停（到 max_ticks），退出循环
"""
from agentsim.controller import Robot

robot = Robot()

while robot.step():
    # —— 示例控制律：所有执行器回零目标（改成你的）——
    for name in robot.actuators:
        lo, hi = robot.range(name)
        robot.set(name, 0.0)
    # 例：正弦摆动第一个执行器
    # robot.set(robot.actuators[0], 0.5 * (robot.tick % 100 < 50))
'''
