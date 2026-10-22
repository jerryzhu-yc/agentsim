"""场景模板生成器（M0.7 A 档）：确定性参数化模板 -> 世界 YAML。

定位（方案 5.7）：AI/用户只拿参数化模板，不生成代码——输出空间压到
YAML 子集；同一命令 + 同一输出位置永远产出同一世界（可进 pytest）。

用法：
    python -m agentsim scenegen maze --rows 4 --cols 4 --out worlds/maze.yaml
    python -m agentsim scenegen tabletop --objects 2

路径解析：模板里 model/controller 路径按目标位置（--out 的目录；缺省
当前目录）计算相对路径，产出的 YAML 放哪都能跑。
"""
from __future__ import annotations

import os
import pathlib

import yaml

from agentsim.errors import WorldError

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CONTROLLERS = ROOT / "controllers"

_CELL = 0.5      # 迷宫单元格边长（米）


# ---------------------------------------------------------------- 组装辅助

def _rel(target: pathlib.Path, base: pathlib.Path) -> str:
    """target 相对 base 的路径（写进 YAML，让产物在保存位置直接可用）"""
    return os.path.relpath(target.resolve(), base.resolve())


def _robot(name: str, model_file: str, base: pathlib.Path,
           pos: tuple, controller: str | None = None,
           gripper: str | None = None) -> dict:
    """组装 robot 段。model_file：仓库 models/ 下文件名，或 zoo://key
    模型库引用（load_world 时解析成 menagerie 缓存绝对路径，产物
    跨机器可移植，首次使用自动下载）。"""
    if model_file.startswith("zoo://"):
        model = model_file
    else:
        model = _rel(MODELS / model_file, base)
    r = {"name": name, "model": model,
         "pos": [round(float(v), 4) for v in pos]}
    if controller:
        r["controller"] = _rel(CONTROLLERS / controller, base)
    if gripper:
        r["gripper"] = {"model": gripper}
    return r


def _obj(name: str, size: tuple, pos: tuple, color: str,
         mass: float | None = None, friction: float | None = None,
         static: bool = False) -> dict:
    o = {"name": name, "type": "box", "size": [round(float(s), 4) for s in size],
         "pos": [round(float(v), 4) for v in pos], "color": color}
    if static:
        o["static"] = True
    else:
        o["mass"] = mass
    if friction is not None:
        o["friction"] = friction
    return o


def _base_world(name: str, objects: list, robots: list,
                task: dict | None = None) -> dict:
    w = {
        "name": name,
        "timestep": 0.002,
        "ctrl_period": 32,
        "gravity": [0, 0, -9.81],
        "scene": {
            "floor": {"size": [2.0, 2.0], "texture": "grid"},
            "lights": [{"pos": [0, 0, 3]}],
            "objects": objects,
        },
        "robots": robots,
    }
    if task is not None:
        w["task"] = task
    return w


def _check_range(pname, value, lo, hi):
    if not (lo <= value <= hi):
        raise WorldError(
            f"模板参数 '{pname}' 超出范围 [{lo}, {hi}]，实际是: {value}")


# ---------------------------------------------------------------- 模板

def _tpl_tabletop(p, base):
    """机械臂桌面抓取：桌面 + N 个 cube + 抬升成功区"""
    n = p["objects"]
    _check_range("objects", n, 1, 3)
    table = _obj("table", (0.20, 0.20, 0.075), (0.50, 0, 0.075),
                 "#6B7280", static=True)
    cubes = [_obj(f"cube_{i}", (0.03, 0.03, 0.03), (0.35 + 0.15 * i, 0, 0.18),
                  "#F59E0B", mass=0.05, friction=0.6) for i in range(n)]
    zone = {"min": [0.40, -0.10, 0.35], "max": [0.60, 0.10, 0.50]}
    task = {
        "success": {"body": "cube_0", "zone": zone},
        "reward": [
            {"distance": {"a": "cube_0", "b": "arm/link7", "weight": -0.01}},
            {"zone": {"body": "cube_0", "zone": zone, "weight": 0.01}},
        ],
        "max_ticks": 5000,
    }
    w = _base_world(f"tabletop_{n}", [table] + cubes,
                    [_robot("arm", "zoo://franka_emika_panda", base, (0, 0, 0),
                            gripper="onrobot_2fg7")], task)
    # panda 高刚度位置伺服（kp=4500）依赖隐式积分（同 worlds/pick_cube.yaml）
    w["integrator"] = "implicitfast"
    w["impratio"] = 10
    return w


def _tpl_push(p, base):
    """地面推物：panda+2FG7 把低摩擦 cube 沿 +y 推进成功区（无绑定控制器，
    控制律留给 Agent/用户写）"""
    cube = _obj("cube", (0.03, 0.035, 0.05), (0.55, 0.15, 0.05),
                "#F59E0B", mass=0.05, friction=0.3)
    zone = {"min": [0.40, 0.26, 0.0], "max": [0.65, 0.50, 0.14]}
    task = {
        "success": {"body": "cube", "zone": zone},
        "reward": [
            {"distance": {"a": "cube", "b": "arm/link7", "weight": -0.01}},
            {"zone": {"body": "cube", "zone": zone, "weight": 0.01}},
        ],
        "max_ticks": 5000,
    }
    w = _base_world("push_cube", [cube],
                    [_robot("arm", "zoo://franka_emika_panda", base, (0, 0, 0),
                            gripper="onrobot_2fg7")], task)
    w["scene"]["floor"]["friction"] = 0.3
    # panda 高刚度位置伺服（kp=4500）依赖隐式积分（同 worlds/pick_cube.yaml）
    w["integrator"] = "implicitfast"
    w["impratio"] = 10
    return w


def _maze_walls(rows: int, cols: int) -> list:
    """蛇形迷宫墙：行间隔墙缺口左右交替 + 外圈边界。确定性，无随机。"""
    s, t, h = _CELL, 0.015, 0.06
    walls = []

    def seg(x0, x1, y):
        if x1 - x0 > 1e-6:
            walls.append(_obj(f"wall_x{len(walls)}", ((x1 - x0) / 2, t, h),
                              ((x0 + x1) / 2, y, h), "#6B7280", static=True))

    # 行间隔墙（row i 与 i+1 之间，缺口列交替：偶数行在右、奇数行在左）
    for i in range(rows - 1):
        gap = cols - 1 if i % 2 == 0 else 0
        seg(0, gap * s, (i + 1) * s)
        seg((gap + 1) * s, cols * s, (i + 1) * s)
    # 外圈
    walls.append(_obj("wall_n", (cols * s / 2 + t, t, h),
                      (cols * s / 2, rows * s, h), "#6B7280", static=True))
    walls.append(_obj("wall_s", (cols * s / 2 + t, t, h),
                      (cols * s / 2, 0, h), "#6B7280", static=True))
    walls.append(_obj("wall_w", (t, rows * s / 2 + t, h),
                      (0, rows * s / 2, h), "#6B7280", static=True))
    walls.append(_obj("wall_e", (t, rows * s / 2 + t, h),
                      (cols * s, rows * s / 2, h), "#6B7280", static=True))
    return walls


def _tpl_maze(p, base):
    """差速车蛇形迷宫：从 (0,0) 格走到左上目标格"""
    rows, cols = p["rows"], p["cols"]
    _check_range("rows", rows, 2, 8)
    _check_range("cols", cols, 2, 8)
    s = _CELL
    goal = {"min": [0.05, (rows - 1) * s + 0.05, 0.0],
            "max": [s - 0.05, rows * s - 0.05, 0.3]}
    task = {
        "success": {"body": "car", "zone": goal},
        "reward": [{"zone": {"body": "car", "zone": goal, "weight": 0.01}}],
        "max_ticks": 20000,
    }
    w = _base_world(f"maze_{rows}x{cols}", _maze_walls(rows, cols),
                    [_robot("car", "diff_car.xml", base,
                            (s / 2, s / 2, 0))], task)
    w["scene"]["floor"]["size"] = [cols * s + 1, rows * s + 1]
    return w


def _tpl_race(p, base):
    """双车竞速：并行车道 + 终点区（success 看车 A，谁先到由控制器决定）"""
    lanes = p["lanes"]
    _check_range("lanes", lanes, 2, 3)
    lane_w, length = 0.6, 4.0
    walls = []
    for i in range(1, lanes):      # 车道隔墙
        walls.append(_obj(f"divider_{i}", (length / 2, 0.015, 0.06),
                          (length / 2, i * lane_w, 0.06), "#6B7280",
                          static=True))
    for side_y in (0.0, lanes * lane_w):   # 上下边界
        walls.append(_obj(f"bound_{len(walls)}", (length / 2 + 0.03, 0.015, 0.06),
                          (length / 2, side_y, 0.06), "#6B7280", static=True))
    cars = [_robot(f"car_{chr(97 + i)}", "diff_car.xml", base,
                   (0.3, lane_w / 2 + i * lane_w, 0))
            for i in range(lanes)]
    zone = {"min": [3.55, 0.05, 0.0], "max": [3.90, lanes * lane_w - 0.05, 0.3]}
    task = {
        "success": {"body": "car_a/car", "zone": zone},
        "reward": [{"zone": {"body": "car_a/car", "zone": zone,
                             "weight": 0.01}}],
        "max_ticks": 20000,
    }
    w = _base_world(f"race_{lanes}", walls, cars, task)
    w["scene"]["floor"]["size"] = [length + 1, lanes * lane_w + 1]
    return w


def _tpl_empty(p, base):
    """空世界：地面 + 一个机器人（或裸地面），任务留给 Agent 写"""
    kind = p["robot"]
    if kind not in ("arm", "diff_car", "none"):
        raise WorldError(f"参数 robot 需要是 arm/diff_car/none，实际是: {kind!r}")
    robots = []
    if kind == "arm":
        robots.append(_robot("arm", "zoo://franka_emika_panda", base,
                             (0, 0, 0), gripper="onrobot_2fg7"))
    elif kind == "diff_car":
        robots.append(_robot("car", "diff_car.xml", base, (0, 0, 0.045)))
    w = _base_world(f"empty_{kind}", [], robots)
    if kind == "arm":
        # panda 高刚度位置伺服（kp=4500）依赖隐式积分（同 worlds/pick_cube.yaml）
        w["integrator"] = "implicitfast"
        w["impratio"] = 10
    return w


def _tpl_pallet(p, base):
    """码垛分区：UR5e + 2FG7 把 A 区一排 cube 逐个码到 B 区对应目标格。

    布局（臂装在桌后缘，坐标系对齐官方 workbench 参考场景）：
    - 桌面顶 z=0.75，臂固定基座在桌后缘 (0,0,0.75)
    - A 区（取料）x=0.62，cube 沿 y 居中排开
    - B 区（放料）x=0.30，每个 cube 一个目标 zone（绿色垫板标记）
    """
    n = p["cubes"]
    _check_range("cubes", n, 1, 6)

    top_z = 0.75
    half, cs, gap = 0.025, 0.025, 0.11      # cube 半边长 / 目标格间距
    x_a, x_b = 0.62, 0.30

    def yi(i):
        return round((i - (n - 1) / 2.0) * gap, 4)

    objects = [
        # 工作台（顶 z=0.75）
        _obj("table", (0.40, 0.40, 0.03), (0.45, 0, top_z - 0.03),
             "#6B7280", static=True),
    ]
    zones = []
    for i in range(n):
        y = yi(i)
        # A 区待抓 cube
        objects.append(_obj(f"cube_{i}", (half, half, half),
                            (x_a, y, top_z + cs), "#F59E0B",
                            mass=0.05, friction=1.0))
        # B 区目标垫板（静态薄板，纯视觉标记）
        objects.append(_obj(f"pad_{i}", (0.045, 0.045, 0.002),
                            (x_b, y, top_z + 0.002), "#10B981",
                            static=True))
        zones.append({"min": [x_b - 0.05, y - 0.05, top_z],
                      "max": [x_b + 0.05, y + 0.05, top_z + 0.15]})

    task = {
        "success": {"body": "cube_0", "zone": zones[0]},
        "reward": [{"zone": {"body": f"cube_{i}", "zone": zones[i],
                             "weight": 0.01}} for i in range(n)],
        "max_ticks": 8000,
    }
    arm = {
        "name": "arm",
        "model": "zoo://universal_robots_ur5e",
        "pos": [0, 0, top_z],
        "gripper": {"model": "onrobot_2fg7"},
    }
    w = _base_world(f"pallet_{n}", objects, [arm], task)
    w["integrator"] = "implicitfast"
    w["impratio"] = 100
    w["scene"]["floor"]["size"] = [1.5, 1.5]
    return w


# name -> (说明, {参数: (类型, 默认, 范围/取值说明)}, 构建函数)
TEMPLATES = {
    "tabletop": ("机械臂桌面抓取：桌面 + N 个 cube + 抬升成功区",
                 {"objects": (int, 2, "桌面 cube 数 1~3")}, _tpl_tabletop),
    "push": ("地面推物：panda+2FG7 把低摩擦 cube 推进成功区（控制器留给 Agent 写）",
             {}, _tpl_push),
    "maze": ("差速车蛇形迷宫：蛇形通道走到左上目标格",
             {"rows": (int, 3, "行数 2~8"),
              "cols": (int, 4, "列数 2~8")}, _tpl_maze),
    "race": ("多车竞速：并行车道 + 终点区",
             {"lanes": (int, 2, "车道数 2~3")}, _tpl_race),
    "pallet": ("码垛分区：UR5e+2FG7 把 A 区 cube 逐个码到 B 区目标格",
               {"cubes": (int, 3, "cube 数 1~6")}, _tpl_pallet),
    "empty": ("空世界：地面 + 可选机器人，任务留给 Agent",
              {"robot": (str, "arm", "arm | diff_car | none")}, _tpl_empty),
}


def generate(name: str, base, params: dict | None = None) -> str:
    """生成模板世界的 YAML 文本。base：产物保存目录（相对路径基准）。"""
    if name not in TEMPLATES:
        raise WorldError(f"未知模板 '{name}'，可用: {sorted(TEMPLATES)}")
    _, _, build = TEMPLATES[name]
    p = {k: spec[1] for k, spec in TEMPLATES[name][1].items()}
    if params:
        for k, v in params.items():
            if k not in p:
                raise WorldError(f"模板 '{name}' 没有参数 '{k}'，"
                                 f"可用: {sorted(p)}")
            p[k] = v
    world = build(p, pathlib.Path(base))
    return yaml.dump(world, sort_keys=False, allow_unicode=True,
                     default_flow_style=None)
