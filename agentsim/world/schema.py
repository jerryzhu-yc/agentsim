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
# 积分器名 -> mjtIntegrator 枚举值；implicitfast 是 MuJoCo 官方推荐的
# 稳定默认（比 Euler 略慢），高刚度接触模型（solref 0.004 量级）必须用它
_INTEGRATORS = {"euler": 0, "rk4": 1, "implicit": 2, "implicitfast": 3}
# 传感器类型 -> 挂点要求：joint 系只引用关节，body 系只引用 body
_SENSOR_TYPES = {"encoder", "imu", "tactile", "joint_torque", "camera"}
_JOINT_SENSORS = {"encoder", "joint_torque"}
_BODY_SENSORS = {"imu", "tactile", "camera"}


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
class SensorSpec:
    """机器人传感器声明（编译进 MuJoCo，观测流带读数）。

    类型 -> 挂点要求：
    - encoder / joint_torque：引用 joint（读数 = 关节位置 / 关节约束力）
    - imu / tactile：引用 body（builder 自动在 body 上建挂点 site；
      imu = 陀螺 3 + 加速度计 3 共 6 维；tactile 用 mjSENS_TOUCH 标量接触力，
      不是 MuJoCo 的 mjSENS_TACTILE 网格阵列）
    - camera：引用 body（仅声明 mj 相机 viewpoint，供 screenshot 按名渲染；
      不进 obs 数值流——每 tick 渲染图像太重）

    tactile 的 radius：MuJoCo 3.x 的 touch 只统计接触点落在以挂点 site 为球心、
    site size 为半径的球内的接触力（site size 默认 0 → 恒 0 的坑由此而来）。
    radius 应覆盖挂点 body 的预期接触区，默认 0.06（罩住典型指尖/指板整面）。
    """
    name: str
    type: str                       # encoder | imu | tactile | joint_torque | camera
    joint: str | None = None        # encoder/joint_torque 必填
    body: str | None = None          # imu/tactile/camera 必填
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)   # body 局部系挂点偏移
    quat: tuple[float, float, float, float] | None = None  # 仅 camera
    radius: float | None = None     # 仅 tactile：touch 接触过滤球半径(米)


@dataclass
class GripperSpec:
    """末端夹爪声明：经 site-attach 挂到机器人腕部挂点。

    model：夹爪库 key（如 onrobot_2fg7）或 MJCF 路径（相对世界文件）。
    site：挂到机器人的哪个 body site，缺省由夹爪库给定（attachment_site）。
    pos/quat：一般不写——builder 用 grippers.py 里按机器人模型标定的 mount
    变换；显式写则覆盖 mount 表（非标准臂/手动微调）。
    """
    model: str
    site: str | None = None
    pos: tuple[float, float, float] | None = None
    quat: tuple[float, float, float, float] | None = None


@dataclass
class RobotSpec:
    name: str
    model: str                      # MJCF 路径（相对世界文件）
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] | None = None   # root 绝对朝向
                                    # （wxyz；模型自带 quat 被清零吸收进此处）
    controller: str | None = None   # 控制器脚本路径（相对世界文件）
    args: dict = field(default_factory=dict)
    sensors: list[SensorSpec] = field(default_factory=list)
    gripper: GripperSpec | None = None


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
    impratio: float = 1.0         # 接触求解摩擦/法向刚度比：大质量机器人
                                  # （spot 30kg/anymal 50kg）需要 50-100，
                                  # 否则 pyramidal 锥下摩擦误差被质量放大
                                  # （attach 时 world 值优先于模型原值）
    integrator: int = 0           # 积分器（0=euler 1=rk4 2=implicit
                                  # 3=implicitfast）：超硬接触模型（如 spot
                                  # solref=0.004）被 Euler 显式积分会数值
                                  # 震荡弹跳，必须 implicit 系（attach 时
                                  # world 值优先于模型原值）
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


def _parse_sensors(rd: dict, p: str) -> list[SensorSpec]:
    """robot 段下的 sensors 列表。joint/body 是否真实存在由 builder 核对。"""
    sensors = []
    seen_names: dict[str, str] = {}
    for i, sd in enumerate(rd.get("sensors", [])):
        sp = f"{p}.sensors[{i}]"
        if not isinstance(sd, dict):
            raise _err(sp, f"需要字典，实际是: {sd!r}")
        name = _str(_req(sd, "name", sp), f"{sp}.name")
        if "/" in name:
            raise _err(f"{sp}.name", f"名字里不能含 '/'（会与机器人前缀冲突）: {name!r}")
        if name in seen_names:
            raise _err(f"{sp}.name", f"名字 '{name}' 重复（已用于 {seen_names[name]}）")
        seen_names[name] = sp
        stype = _str(_req(sd, "type", sp), f"{sp}.type")
        if stype not in _SENSOR_TYPES:
            raise _err(f"{sp}.type", f"未知传感器类型 '{stype}'，支持: {sorted(_SENSOR_TYPES)}")
        joint = sd.get("joint")
        body = sd.get("body")
        if stype in _JOINT_SENSORS:
            if joint is None:
                raise _err(sp, f"类型 '{stype}' 需要 joint 字段")
            if body is not None:
                raise _err(f"{sp}.body", f"类型 '{stype}' 不使用 body（挂点是 joint）")
            joint = _str(joint, f"{sp}.joint")
        else:
            if body is None:
                raise _err(sp, f"类型 '{stype}' 需要 body 字段")
            if joint is not None:
                raise _err(f"{sp}.joint", f"类型 '{stype}' 不使用 joint（挂点是 body）")
            body = _str(body, f"{sp}.body")
        pos = _vec(sd.get("pos", [0, 0, 0]), 3, f"{sp}.pos")
        quat = None
        if "quat" in sd:
            quat = _vec(sd["quat"], 4, f"{sp}.quat")
            if stype != "camera":
                raise _err(f"{sp}.quat", f"只有 camera 用 quat，类型 '{stype}' 请删掉")
        radius = None
        if "radius" in sd:
            if stype != "tactile":
                raise _err(f"{sp}.radius", f"只有 tactile 用 radius，类型 '{stype}' 请删掉")
            radius = _num(sd["radius"], f"{sp}.radius")
            if radius <= 0:
                raise _err(f"{sp}.radius", f"必须为正数(米)，实际是: {radius}")
        if stype == "tactile" and radius is None:
            radius = 0.06
        sensors.append(SensorSpec(name=name, type=stype, joint=joint, body=body,
                                  pos=pos, quat=quat, radius=radius))
    return sensors


def _parse_robots(d, path: str, seen: dict) -> list[RobotSpec]:
    robots = []
    for i, rd in enumerate(d.get("robots", [])):
        p = f"{path}.robots[{i}]"
        name = _str(_req(rd, "name", p), f"{p}.name")
        _name_unique(name, seen, f"{p}.name")
        model = _str(_req(rd, "model", p), f"{p}.model")
        pos = _vec(rd.get("pos", [0, 0, 0]), 3, f"{p}.pos")
        quat = rd.get("quat")
        if quat is not None:
            quat = _vec(quat, 4, f"{p}.quat")
        controller = rd.get("controller")
        if controller is not None:
            controller = _str(controller, f"{p}.controller")
        args = rd.get("args", {})
        if not isinstance(args, dict):
            raise _err(f"{p}.args", f"需要字典，实际是: {args!r}")
        sensors = _parse_sensors(rd, p)
        gripper = _parse_gripper(rd.get("gripper"), f"{p}.gripper")
        robots.append(RobotSpec(name=name, model=model, pos=pos, quat=quat,
                                controller=controller, args=args,
                                sensors=sensors, gripper=gripper))
    return robots


def _parse_gripper(gd, path: str) -> "GripperSpec | None":
    if gd is None:
        return None
    if not isinstance(gd, dict):
        raise _err(path, f"需要字典（model/site/pos/quat），实际是: {gd!r}")
    model = _str(_req(gd, "model", path), f"{path}.model")
    site = gd.get("site")
    if site is not None:
        site = _str(site, f"{path}.site")
    gpos = gd.get("pos")
    gquat = gd.get("quat")
    if gpos is not None:
        gpos = _vec(gpos, 3, f"{path}.pos")
    if gquat is not None:
        gquat = _vec(gquat, 4, f"{path}.quat")
    return GripperSpec(model=model, site=site, pos=gpos, quat=gquat)


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
    impratio = _num(d.get("impratio", 1.0), f"{path}.impratio")
    if impratio <= 0:
        raise _err(f"{path}.impratio", f"必须为正数，实际是: {impratio}")
    integ_name = str(d.get("integrator", "euler")).lower()
    if integ_name not in _INTEGRATORS:
        raise _err(f"{path}.integrator",
                   f"必须是 {sorted(_INTEGRATORS)} 之一，实际是: {d.get('integrator')!r}")
    integrator = _INTEGRATORS[integ_name]

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
                     gravity=gravity, impratio=impratio, integrator=integrator,
                     scene=scene, robots=robots, task=task, runtime=runtime)


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
