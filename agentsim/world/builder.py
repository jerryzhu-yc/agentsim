"""世界组装器：WorldSpec -> MjSpec -> MjModel + NameIndex。

流程（方案 5.3 节）：
1. 建 floor/light/objects
2. 每个机器人 attach 到独立 frame（前缀 "<name>/" 隔离命名空间）
3. compile 出 MjModel
4. NameIndex 把 YAML 名字解析成 MuJoCo id，此后内核全部用名字操作

已实测的 API 事实（mujoco 3.15.0）：
- world.attach(child, prefix=..., frame=...) 要求 frame 或 site 之一，返回 MjsFrame
- 前缀对 child 内全部名字（body/joint/actuator/geom）生效
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import mujoco

from agentsim.errors import WorldError
from agentsim.world.schema import (
    WorldSpec, RobotSpec, TaskSpec, DistanceReward, ZoneReward, ContactReward,
    hex_to_rgba,
)

_GEOM_TYPE_MAP = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
}


# ---------------------------------------------------------------- 名字索引

@dataclass
class RobotIndex:
    """单个机器人的设备索引（名字均为无前缀原始名）"""
    name: str
    actuators: dict[str, int] = field(default_factory=dict)   # 执行器名 -> id
    joints: dict[str, int] = field(default_factory=dict)     # 关节名 -> id
    bodies: dict[str, int] = field(default_factory=dict)     # body 名 -> id


@dataclass
class NameIndex:
    """编译后的全局名字索引：内核此后只用名字，不碰裸 id"""
    objects: dict[str, int] = field(default_factory=dict)     # YAML 物体名 -> body id
    robots: dict[str, RobotIndex] = field(default_factory=dict)
    bodies: dict[str, int] = field(default_factory=dict)     # 全部 body（含前缀名）-> id

    def resolve_body(self, ref: str) -> int:
        """按 物体名 -> 完整前缀名 -> 任意机器人前缀 依次解析 body 引用。"""
        if ref in self.objects:
            return self.objects[ref]
        if ref in self.bodies:
            return self.bodies[ref]
        for robot in self.robots.values():
            pref = f"{robot.name}/{ref}"
            if pref in self.bodies:
                return self.bodies[pref]
        raise WorldError(
            f"引用的 body '{ref}' 不存在。"
            f"可用: 物体 {sorted(self.objects)}，"
            f"机器人内部 body {sorted(set(self.bodies) - set(self.objects))}"
        )

    def available_names(self) -> list[str]:
        return sorted(set(self.bodies))


# ---------------------------------------------------------------- 任务引用解析

@dataclass
class ResolvedTask:
    """schema 的 TaskSpec 编译后版本：所有 body 引用已解析成 id"""
    success_body: int | None = None
    success_min: tuple = ()
    success_max: tuple = ()
    rewards: list = field(default_factory=list)   # (kind, id_a, id_b/zone, weight)
    max_ticks: int | None = None


def resolve_task(task: TaskSpec, index: NameIndex) -> ResolvedTask:
    """把 TaskSpec 里的名字引用解析成 MuJoCo id，错误时列出全部可用名字。"""
    rt = ResolvedTask(max_ticks=task.max_ticks)
    if task.success is not None:
        rt.success_body = index.resolve_body(task.success.body)
        rt.success_min = tuple(task.success.zone.min)
        rt.success_max = tuple(task.success.zone.max)
    for r in task.reward:
        if isinstance(r, DistanceReward):
            rt.rewards.append(("distance", index.resolve_body(r.a),
                               index.resolve_body(r.b), r.weight))
        elif isinstance(r, ZoneReward):
            # zone reward 的 body 用 id，zone 存 (min,max) 元组
            rt.rewards.append(("zone", index.resolve_body(r.body),
                               (tuple(r.zone.min), tuple(r.zone.max)), r.weight))
        elif isinstance(r, ContactReward):
            rt.rewards.append(("contact", index.resolve_body(r.a),
                               index.resolve_body(r.b), r.weight))
    return rt


# ---------------------------------------------------------------- 组装

def _resolve_path(base: pathlib.Path, rel: str, what: str) -> pathlib.Path:
    p = pathlib.Path(rel)
    if not p.is_absolute():
        p = base / p
    if not p.exists():
        raise WorldError(f"{what} 文件不存在: {p}（世界文件目录: {base}）")
    return p


def _build_scene(spec: WorldSpec, mj: mujoco.MjSpec):
    sc = spec.scene
    # 地面
    mj.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[sc.floor.size[0], sc.floor.size[1], 0.1],
        rgba=[0.45, 0.47, 0.52, 1.0],
        friction=[sc.floor.friction, 0.005, 0.0001],
    )
    # 灯光
    if not sc.lights:
        mj.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
    for i, light in enumerate(sc.lights):
        mj.worldbody.add_light(name=f"light_{i}", pos=list(light.pos), dir=[0, 0, -1])
    # 物体：独立 body + freejoint（可被抓取/瞬移）；static 物体无 freejoint
    for od in sc.objects:
        body = mj.worldbody.add_body(name=od.name, pos=list(od.pos),
                                     quat=list(od.quat) if od.quat else None)
        if not od.static:
            body.add_joint(name=f"{od.name}_free", type=mujoco.mjtJoint.mjJNT_FREE)
        kwargs = dict(name=f"{od.name}_geom", type=_GEOM_TYPE_MAP[od.type],
                      size=list(od.size), rgba=hex_to_rgba(od.color),
                      friction=[od.friction, 0.005, 0.0001])
        if od.mass is not None:
            kwargs["mass"] = od.mass
        body.add_geom(**kwargs)


def _attach_robot(spec: WorldSpec, robot: RobotSpec, mj: mujoco.MjSpec) -> None:
    model_path = _resolve_path(spec.base_dir, robot.model, "机器人模型")
    try:
        child = mujoco.MjSpec.from_file(str(model_path))
    except Exception as e:
        raise WorldError(f"机器人 '{robot.name}' 模型解析失败 ({model_path}): {e}") from e

    frame = mj.worldbody.add_frame(name=f"{robot.name}_frame", pos=list(robot.pos))
    try:
        mj.attach(child, prefix=f"{robot.name}/", frame=frame)
    except Exception as e:
        raise WorldError(f"机器人 '{robot.name}' 挂载失败: {e}") from e


def _build_index(model: mujoco.MjModel, spec: WorldSpec) -> NameIndex:
    index = NameIndex()
    # 物体
    for od in spec.scene.objects:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, od.name)
        if bid < 0:
            raise WorldError(f"编译后找不到物体 '{od.name}' 的 body（编译器吞了名字？）")
        index.objects[od.name] = bid
    # 机器人设备
    for rd in spec.robots:
        prefix = f"{rd.name}/"
        ridx = RobotIndex(name=rd.name)
        for act_name in _child_actuator_names(spec, rd):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                     f"{prefix}{act_name}")
            if aid < 0:
                raise WorldError(
                    f"编译后找不到机器人 '{rd.name}' 的执行器 '{act_name}'"
                    f"（期望全名 '{prefix}{act_name}'）")
            ridx.actuators[act_name] = aid
        for jnt_name in _child_joint_names(spec, rd):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                     f"{prefix}{jnt_name}")
            if jid < 0:
                raise WorldError(
                    f"编译后找不到机器人 '{rd.name}' 的关节 '{jnt_name}'"
                    f"（期望全名 '{prefix}{jnt_name}'）")
            ridx.joints[jnt_name] = jid
        index.robots[rd.name] = ridx
    # 全部 body（含前缀）
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name:
            continue
        index.bodies[name] = i
        # 归入所属机器人的 bodies（去掉前缀）
        for rd in spec.robots:
            if name.startswith(f"{rd.name}/"):
                index.robots[rd.name].bodies[name[len(rd.name) + 1:]] = i
    return index


def _child_actuator_names(spec: WorldSpec, rd: RobotSpec) -> list[str]:
    """从机器人的 MJCF 里读取执行器原始名（无前缀），供索引解析。"""
    model_path = _resolve_path(spec.base_dir, rd.model, "机器人模型")
    child = mujoco.MjSpec.from_file(str(model_path))
    names = [a.name for a in child.actuators if a.name]
    if not names:
        raise WorldError(
            f"机器人 '{rd.name}' 模型里没有任何命名的执行器 ({model_path})，"
            f"无法建立设备索引")
    return names


def _child_joint_names(spec: WorldSpec, rd: RobotSpec) -> list[str]:
    """从机器人的 MJCF 里读取关节原始名（无前缀）。"""
    model_path = _resolve_path(spec.base_dir, rd.model, "机器人模型")
    child = mujoco.MjSpec.from_file(str(model_path))
    return [j.name for j in child.joints if j.name]


def build_world(spec: WorldSpec) -> tuple[mujoco.MjModel, NameIndex]:
    """WorldSpec -> (MjModel, NameIndex)。schema 校验必须已完成（load_world 保证）。"""
    mj = mujoco.MjSpec()
    mj.modelname = spec.name
    mj.option.timestep = spec.timestep
    mj.option.gravity = list(spec.gravity)
    # 离屏帧缓冲调大：GUI 渲染视口可达 ~1600px，默认 640 会抛异常
    mj.visual.global_.offwidth = 1920
    mj.visual.global_.offheight = 1440

    _build_scene(spec, mj)
    for rd in spec.robots:
        _attach_robot(spec, rd, mj)

    try:
        model = mj.compile()
    except Exception as e:
        raise WorldError(f"世界 '{spec.name}' 编译失败: {e}") from e

    index = _build_index(model, spec)

    # 任务引用最终核对（此前只有无机器人场景能提前校验）
    resolve_task(spec.task, index)

    return model, index
