"""世界文件 schema：YAML -> dataclass，全部校验在前置完成，错误信息可读。

设计原则：
- 字段语义与 MJCF 对齐（size=半尺寸、pos=中心），AI 生成世界可迁移 MuJoCo 知识
- 一切皆命名引用：名字在此校验唯一，跨命名空间引用在 builder 编译后解析
- 报错带位置（字段路径）、期望、实际值，人和 AI 都能直接看懂
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field

import yaml

from agentsim.errors import WorldError

# 几何类型 -> size 需要的分量数（MJCF 半尺寸语义）
_GEOM_ARITY = {"box": 3, "sphere": 1, "cylinder": 2}
_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_TIMEOUT_POLICIES = {"last_ctrl", "pause"}


# ---------------------------------------------------------------- dataclass 定义

@dataclass
class FloorSpec:
    size: tuple[float, float] = (2.0, 2.0)
    texture: str = "grid"
    friction: float = 1.0             # 地面滑动摩擦（MuJoCo 语义，接触对取 max）


@dataclass
class LightSpec:
    pos: tuple[float, float, float] = (0.0, 0.0, 3.0)


@dataclass
class ObjectSpec:
    name: str
    type: str                       # box | sphere | cylinder
    size: tuple[float, ...]         # MJCF 半尺寸
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] | None = None
    mass: float | None = None
    color: str = "#9CA3AF"
    static: bool = False            # true = 无 freejoint 的固定场景体（台/墙）
    friction: float = 1.0           # 滑动摩擦（MuJoCo 语义，接触对取 max）


@dataclass
class RobotSpec:
    name: str
    model: str                      # MJCF 路径（相对世界文件）
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    controller: str | None = None   # 控制器脚本路径（相对世界文件）
    args: dict = field(default_factory=dict)


@dataclass
class ZoneSpec:
    min: tuple[float, float, float]
    max: tuple[float, float, float]


@dataclass
class SuccessSpec:
    body: str
    zone: ZoneSpec


@dataclass
class DistanceReward:
    """|body_a - body_b| 距离，weight 乘以距离累加"""
    a: str
    b: str
    weight: float = -1.0


@dataclass
class ZoneReward:
    """body 在 zone 内时每 tick 加 weight"""
    body: str
    zone: ZoneSpec
    weight: float = 1.0


@dataclass
class ContactReward:
    """body_a 与 body_b 接触时每 tick 加 weight"""
    a: str
    b: str
    weight: float = 1.0


@dataclass
class TaskSpec:
    success: SuccessSpec | None = None
    reward: list = field(default_factory=list)   # Distance/Zone/ContactReward
    max_ticks: int | None = None


@dataclass
class RuntimeSpec:
    realtime: bool = False
    controller_timeout: float = 100.0            # ms
    timeout_policy: str = "last_ctrl"           # last_ctrl | pause


@dataclass
class SceneSpec:
    floor: FloorSpec = field(default_factory=FloorSpec)
    lights: list[LightSpec] = field(default_factory=list)
    objects: list[ObjectSpec] = field(default_factory=list)


@dataclass
class WorldSpec:
    name: str
    timestep: float = 0.002
    ctrl_period: int = 32
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    scene: SceneSpec = field(default_factory=SceneSpec)
    robots: list[RobotSpec] = field(default_factory=list)
    task: TaskSpec = field(default_factory=TaskSpec)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    base_dir: pathlib.Path = pathlib.Path(".")   # 世界文件所在目录，路径解析基准
                                              # （dataclass eq 忽略不了它，测试请勿比较整个 spec）


# ---------------------------------------------------------------- 解析辅助

def _err(path: str, msg: str) -> WorldError:
    return WorldError(f"[{path}] {msg}")


def _req(d: dict, key: str, path: str):
    if key not in d:
        raise _err(path, f"缺少必填字段 '{key}'，现有字段: {sorted(d.keys())}")
    return d[key]


def _num(v, path: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _err(path, f"需要数字，实际是 {type(v).__name__}: {v!r}")
    return float(v)


def _vec(v, n: int, path: str) -> tuple:
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise _err(path, f"需要 {n} 个数字的列表，实际是: {v!r}")
    return tuple(_num(x, f"{path}[{i}]") for i, x in enumerate(v))


def _str(v, path: str, allow_empty: bool = False) -> str:
    if not isinstance(v, str) or (not allow_empty and not v.strip()):
        raise _err(path, f"需要非空字符串，实际是: {v!r}")
    return v


def _name_unique(name: str, seen: dict, path: str):
    if name in seen:
        raise _err(path, f"名字 '{name}' 重复（已用于 {seen[name]}），物体和机器人共用一个命名空间")
    seen[name] = path


# ---------------------------------------------------------------- 各段解析

def _parse_scene(d, path: str) -> SceneSpec:
    floor_d = d.get("floor", {})
    if not isinstance(floor_d, dict):
        raise _err(f"{path}.floor", f"需要字典，实际是: {floor_d!r}")
    size = _vec(floor_d.get("size", [2.0, 2.0]), 2, f"{path}.floor.size")
    if size[0] <= 0 or size[1] <= 0:
        raise _err(f"{path}.floor.size", f"半尺寸必须为正，实际是: {size}")
    floor = FloorSpec(size=size, texture=_str(floor_d.get("texture", "grid"),
                                              f"{path}.floor.texture"))
    floor_friction = _num(floor_d.get("friction", 1.0), f"{path}.floor.friction")
    if floor_friction <= 0:
        raise _err(f"{path}.floor.friction", f"摩擦系数必须为正，实际是: {floor_friction}")
    floor.friction = floor_friction

    lights = []
    for i, ld in enumerate(d.get("lights", [])):
        lights.append(LightSpec(pos=_vec(_req(ld, "pos", f"{path}.lights[{i}].pos"),
                                         3, f"{path}.lights[{i}].pos")))

    objects = []
    for i, od in enumerate(d.get("objects", [])):
        p = f"{path}.objects[{i}]"
        name = _str(_req(od, "name", p), f"{p}.name")
        gtype = _str(_req(od, "type", p), f"{p}.type")
        if gtype not in _GEOM_ARITY:
            raise _err(f"{p}.type", f"未知几何类型 '{gtype}'，支持: {sorted(_GEOM_ARITY)}")
        size = _vec(_req(od, "size", p), _GEOM_ARITY[gtype], f"{p}.size")
        if any(s <= 0 for s in size):
            raise _err(f"{p}.size", f"半尺寸必须为正，实际是: {size}")
        pos = _vec(od.get("pos", [0, 0, 0]), 3, f"{p}.pos")
        mass = od.get("mass")
        if mass is not None and _num(mass, f"{p}.mass") <= 0:
            raise _err(f"{p}.mass", f"质量必须为正，实际是: {mass}")
        color = _str(od.get("color", "#9CA3AF"), f"{p}.color")
        if not _COLOR_RE.match(color):
            raise _err(f"{p}.color", f"颜色需要 '#RRGGBB' 格式，实际是: {color!r}")
        quat = od.get("quat")
        if quat is not None:
            quat = _vec(quat, 4, f"{p}.quat")
        static = od.get("static", False)
        if not isinstance(static, bool):
            raise _err(f"{p}.static", f"需要 true/false，实际是: {static!r}")
        friction = _num(od.get("friction", 1.0), f"{p}.friction")
        if friction <= 0:
            raise _err(f"{p}.friction", f"摩擦系数必须为正，实际是: {friction}")
        objects.append(ObjectSpec(name=name, type=gtype, size=size, pos=pos,
                                  quat=quat, mass=mass, color=color, static=static,
                                  friction=friction))
    return SceneSpec(floor=floor, lights=lights, objects=objects)


def _parse_robots(d, path: str, seen: dict) -> list[RobotSpec]:
    robots = []
    for i, rd in enumerate(d.get("robots", [])):
        p = f"{path}.robots[{i}]"
        name = _str(_req(rd, "name", p), f"{p}.name")
        _name_unique(name, seen, f"{p}.name")
        model = _str(_req(rd, "model", p), f"{p}.model")
        pos = _vec(rd.get("pos", [0, 0, 0]), 3, f"{p}.pos")
        controller = rd.get("controller")
        if controller is not None:
            controller = _str(controller, f"{p}.controller")
        args = rd.get("args", {})
        if not isinstance(args, dict):
            raise _err(f"{p}.args", f"需要字典，实际是: {args!r}")
        robots.append(RobotSpec(name=name, model=model, pos=pos,
                                controller=controller, args=args))
    return robots


def _parse_zone(d, path: str) -> ZoneSpec:
    lo = _vec(_req(d, "min", path), 3, f"{path}.min")
    hi = _vec(_req(d, "max", path), 3, f"{path}.max")
    if any(l >= h for l, h in zip(lo, hi)):
        raise _err(path, f"zone 的 min 必须 < max（分量逐个比较），实际 min={lo}, max={hi}")
    return ZoneSpec(min=lo, max=hi)


def _parse_task(d, path: str) -> TaskSpec:
    td = d.get("task", {})
    if not isinstance(td, dict):
        raise _err(f"{path}.task", f"需要字典，实际是: {td!r}")

    success = None
    if "success" in td:
        p = f"{path}.task.success"
        sd = td["success"]
        body = _str(_req(sd, "body", p), f"{p}.body")
        zone = _parse_zone(_req(sd, "zone", p), f"{p}.zone")
        success = SuccessSpec(body=body, zone=zone)

    reward = []
    for i, item in enumerate(td.get("reward", [])):
        p = f"{path}.task.reward[{i}]"
        if not isinstance(item, dict) or len(item) != 1:
            raise _err(p, f"每条 reward 必须是单键字典，如 {{distance: {{...}}}}，实际是: {item!r}")
        kind, rd = next(iter(item.items()))
        if kind == "distance":
            reward.append(DistanceReward(
                a=_str(_req(rd, "a", p), f"{p}.a"),
                b=_str(_req(rd, "b", p), f"{p}.b"),
                weight=_num(rd.get("weight", -1.0), f"{p}.weight")))
        elif kind == "zone":
            reward.append(ZoneReward(
                body=_str(_req(rd, "body", p), f"{p}.body"),
                zone=_parse_zone(_req(rd, "zone", p), f"{p}.zone"),
                weight=_num(rd.get("weight", 1.0), f"{p}.weight")))
        elif kind == "contact":
            reward.append(ContactReward(
                a=_str(_req(rd, "a", p), f"{p}.a"),
                b=_str(_req(rd, "b", p), f"{p}.b"),
                weight=_num(rd.get("weight", 1.0), f"{p}.weight")))
        else:
            raise _err(p, f"未知 reward 类型 '{kind}'，支持: distance, zone, contact")

    max_ticks = td.get("max_ticks")
    if max_ticks is not None:
        max_ticks = int(_num(max_ticks, f"{path}.task.max_ticks"))
        if max_ticks <= 0:
            raise _err(f"{path}.task.max_ticks", f"必须为正整数，实际是: {max_ticks}")
    return TaskSpec(success=success, reward=reward, max_ticks=max_ticks)


def _parse_runtime(d, path: str) -> RuntimeSpec:
    rd = d.get("runtime", {})
    if not isinstance(rd, dict):
        raise _err(f"{path}.runtime", f"需要字典，实际是: {rd!r}")
    realtime = rd.get("realtime", False)
    if not isinstance(realtime, bool):
        raise _err(f"{path}.runtime.realtime", f"需要 true/false，实际是: {realtime!r}")
    timeout = _num(rd.get("controller_timeout", 100.0),
                    f"{path}.runtime.controller_timeout")
    if timeout <= 0:
        raise _err(f"{path}.runtime.controller_timeout", f"必须为正数(ms)，实际是: {timeout}")
    policy = _str(rd.get("timeout_policy", "last_ctrl"), f"{path}.runtime.timeout_policy")
    if policy not in _TIMEOUT_POLICIES:
        raise _err(f"{path}.runtime.timeout_policy",
                   f"未知策略 '{policy}'，支持: {sorted(_TIMEOUT_POLICIES)}")
    return RuntimeSpec(realtime=realtime, controller_timeout=timeout, timeout_policy=policy)


# ---------------------------------------------------------------- 入口

def parse_world(d: dict, path: str = "world") -> WorldSpec:
    """从已解析的 YAML 字典构建 WorldSpec（校验全部在此完成）。"""
    if not isinstance(d, dict):
        raise _err(path, f"顶层需要字典（映射），实际是: {type(d).__name__}")

    name = _str(_req(d, "name", path), f"{path}.name")
    timestep = _num(d.get("timestep", 0.002), f"{path}.timestep")
    if timestep <= 0:
        raise _err(f"{path}.timestep", f"必须为正数(秒)，实际是: {timestep}")
    ctrl_period = int(_num(d.get("ctrl_period", 32), f"{path}.ctrl_period"))
    if ctrl_period < 1:
        raise _err(f"{path}.ctrl_period", f"必须为 >=1 的整数，实际是: {ctrl_period}")
    gravity = _vec(d.get("gravity", [0, 0, -9.81]), 3, f"{path}.gravity")

    scene = _parse_scene(d.get("scene", {}), f"{path}.scene")
    seen: dict[str, str] = {}
    for i, od in enumerate(scene.objects):
        _name_unique(od.name, seen, f"{path}.scene.objects[{i}].name")
    robots = _parse_robots(d, path, seen)
    task = _parse_task(d, path)
    runtime = _parse_runtime(d, path)

    # 任务引用体校验：至少能命中物体名或机器人名（机器人内部 body 名编译后才能校验，
    # builder.resolve_task_refs 会做最终核对并给出候选列表）
    known = set(seen)
    refs = []
    if task.success:
        refs.append(("task.success.body", task.success.body))
    for i, r in enumerate(task.reward):
        if isinstance(r, DistanceReward):
            refs += [(f"task.reward[{i}].a", r.a), (f"task.reward[{i}].b", r.b)]
        elif isinstance(r, ZoneReward):
            refs.append((f"task.reward[{i}].body", r.body))
        elif isinstance(r, ContactReward):
            refs += [(f"task.reward[{i}].a", r.a), (f"task.reward[{i}].b", r.b)]
    if not robots:  # 无机器人时引用不可能命中前缀路径，可提前失败
        for ref_path, ref in refs:
            if ref not in known:
                raise _err(f"{path}.{ref_path}",
                           f"引用的 body '{ref}' 不存在，可用名字: {sorted(known)}")

    return WorldSpec(name=name, timestep=timestep, ctrl_period=ctrl_period,
                     gravity=gravity, scene=scene, robots=robots,
                     task=task, runtime=runtime)


def load_world(yaml_path: str | pathlib.Path) -> WorldSpec:
    """从 YAML 文件加载世界。路径（model/controller）按世界文件所在目录解析。"""
    yaml_path = pathlib.Path(yaml_path)
    if not yaml_path.exists():
        raise WorldError(f"世界文件不存在: {yaml_path}")
    try:
        d = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise WorldError(f"[{yaml_path}] YAML 语法错误: {e}") from e
    if d is None:
        raise WorldError(f"[{yaml_path}] 世界文件为空")
    spec = parse_world(d, str(yaml_path))
    spec.base_dir = yaml_path.resolve().parent
    return spec


def hex_to_rgba(color: str) -> list[float]:
    """'#RRGGBB' -> MuJoCo rgba（不透明）"""
    r = int(color[1:3], 16) / 255.0
    g = int(color[3:5], 16) / 255.0
    b = int(color[5:7], 16) / 255.0
    return [round(r, 4), round(g, 4), round(b, 4), 1.0]
