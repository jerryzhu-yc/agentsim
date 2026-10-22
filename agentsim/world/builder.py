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
from agentsim.grippers import (GRIPPER_PREFIX, GRIPPERS, ensure_gripper,
                               gripper_info, mount_for)
from agentsim.world.schema import (
    WorldSpec, RobotSpec, SensorSpec, TaskSpec, DistanceReward, ZoneReward,
    ContactReward, hex_to_rgba,
)

_GEOM_TYPE_MAP = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
}


# ---------------------------------------------------------------- 名字索引

@dataclass
class SensorIndex:
    """单个传感器的读数定位：obs 按顺序拼接 parts 得到该传感器全量读数。

    encoder/joint_torque/tactile 为单 part；imu 为 [gyro(3), accel(3)] 两个 part。
    """
    name: str
    kind: str                       # YAML 类型
    parts: list[tuple[int, int]] = field(default_factory=list)  # (sensor_adr, dim)

    @property
    def dim(self) -> int:
        return sum(d for _, d in self.parts)


@dataclass
class RobotIndex:
    """单个机器人的设备索引（名字均为无前缀原始名）"""
    name: str
    actuators: dict[str, int] = field(default_factory=dict)   # 执行器名 -> id
    joints: dict[str, int] = field(default_factory=dict)     # 关节名 -> id
    bodies: dict[str, int] = field(default_factory=dict)     # body 名 -> id
    sensors: dict[str, SensorIndex] = field(default_factory=dict)  # YAML 传感器名 -> 索引
    cameras: dict[str, int] = field(default_factory=dict)     # 相机名 -> camera id


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
    # 白天背景：内置渐变 skybox（上方浅天蓝，地平方向近白）。
    # MuJoCo 3.13 classic 渲染器清屏色硬编码为黑色，只有 skybox 能给出非黑背景。
    mj.add_texture(
        name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.72, 0.82, 0.92], rgb2=[0.97, 0.98, 1.0],
        width=512, height=3072,
    )
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


def _add_child_sensors(robot: RobotSpec, child: mujoco.MjSpec) -> None:
    """attach 前往 child spec 注入传感器/site/相机（前缀随后自动生效）。

    命名约定（attach 后的全名 = "<robot>/<名字>"）：
    - encoder/joint_torque/tactile：MuJoCo 传感器名 = YAML 传感器名
    - imu：拆成 "<名>_gyro"(3) + "<名>_accel"(3)，obs 拼成 6 维
    - imu/tactile 的挂点 site："<名>_site"
    - camera：相机名 = YAML 传感器名（不加 MuJoCo sensor）
    """
    if not robot.sensors:
        return
    joint_names = {j.name for j in child.joints}
    body_names = {b.name for b in child.bodies}
    site_names = {s.name for s in child.sites}
    sensor_names = {s.name for s in child.sensors}
    camera_names = {c.name for c in child.cameras}

    def _find_body(s: SensorSpec):
        if s.body not in body_names:
            raise WorldError(
                f"机器人 '{robot.name}' 的传感器 '{s.name}' 引用的 body "
                f"'{s.body}' 不在模型里，可用: {sorted(body_names)}")
        return child.body(s.body)

    for s in robot.sensors:
        if s.name in sensor_names or s.name in camera_names:
            raise WorldError(
                f"机器人 '{robot.name}' 的传感器 '{s.name}' 与模型里既有"
                f" sensor/camera 名冲突")
        if s.type in ("encoder", "joint_torque"):
            if s.joint not in joint_names:
                raise WorldError(
                    f"机器人 '{robot.name}' 的传感器 '{s.name}' 引用的关节 "
                    f"'{s.joint}' 不在模型里，可用: {sorted(joint_names)}")
            stype = (mujoco.mjtSensor.mjSENS_JOINTPOS if s.type == "encoder"
                     else mujoco.mjtSensor.mjSENS_JOINTACTFRC)
            child.add_sensor(name=s.name, type=stype,
                             objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=s.joint)
        elif s.type == "imu":
            site_name = f"{s.name}_site"
            if site_name in site_names:
                raise WorldError(
                    f"机器人 '{robot.name}' 的挂点 site '{site_name}' 与模型里"
                    f"既有 site 名冲突")
            _find_body(s).add_site(name=site_name, pos=list(s.pos))
            child.add_sensor(name=f"{s.name}_gyro",
                             type=mujoco.mjtSensor.mjSENS_GYRO,
                             objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site_name)
            child.add_sensor(name=f"{s.name}_accel",
                             type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                             objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site_name)
        elif s.type == "tactile":
            site_name = f"{s.name}_site"
            if site_name in site_names:
                raise WorldError(
                    f"机器人 '{robot.name}' 的挂点 site '{site_name}' 与模型里"
                    f"既有 site 名冲突")
            # size 即 touch 的接触过滤球半径：MuJoCo 3.x 只统计接触点落在
            # 以 site 为球心、size 为半径的球内的接触力（size=0 → 恒 0）
            _find_body(s).add_site(name=site_name, pos=list(s.pos),
                                   size=[s.radius] * 3)
            child.add_sensor(name=s.name, type=mujoco.mjtSensor.mjSENS_TOUCH,
                             objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site_name)
        elif s.type == "camera":
            kwargs = dict(name=s.name, pos=list(s.pos))
            if s.quat is not None:
                kwargs["quat"] = list(s.quat)
            _find_body(s).add_camera(**kwargs)


def _ensure_actuator_names(child: mujoco.MjSpec) -> list[str]:
    """给无名执行器补名（google_robot 等模型的 <position joint=.../> 不写 name）。

    名字取传输目标（关节名），冲突按 actuatorN 兜底。attach 前执行——
    名字随前缀存活，索引与 hello 清单才引用得到。返回全部执行器名
    （与 attach 后的前缀名一一对应，去掉前缀即这里的名字）。
    """
    taken = {a.name for a in child.actuators if a.name}
    names = []
    for i, a in enumerate(child.actuators):
        if not a.name:
            base = a.target or f"actuator{i}"
            name, k = base, 1
            while name in taken:
                name = f"{base}_{k}"
                k += 1
            a.name = name
            taken.add(name)
        names.append(a.name)
    return names


def _load_gripper_xml(gmodel: str, spec: WorldSpec) -> pathlib.Path:
    """夹爪 model 字段 -> MJCF 路径：夹爪库 key 优先，否则按世界相对路径。"""
    if gmodel in GRIPPERS:
        return ensure_gripper(gmodel)
    return _resolve_path(spec.base_dir, gmodel, "夹爪模型")


def _assemble_robot_child(spec: WorldSpec,
                          robot: RobotSpec) -> tuple[mujoco.MjSpec, list[str]]:
    """装配单台机器人的合并 MjSpec（机器人本体 + 可选夹爪 + 传感器）。

    统一入口：attach 到世界（_attach_robot）与建设备索引
    （_child_actuator_names/_child_joint_names）都调它，保证三处看到的
    是同一份装配结果，不再各自重新 load。

    顺序很关键：先挂夹爪（site-attach 到腕部），再注入传感器——这样触觉
    传感器可以引用夹爪手指 body（body 名以 "gripper/" 开头）。
    返回 (合并后的 child spec, 全部执行器短名列表)。
    """
    model_path = _resolve_path(spec.base_dir, robot.model, "机器人模型")
    try:
        child = _load_robot_spec(model_path)
    except WorldError:
        raise
    except Exception as e:
        raise WorldError(
            f"机器人 '{robot.name}' 模型解析失败 ({model_path}): {e}") from e

    # ---- 夹爪 site-attach 到机器人腕部挂点 ----
    if robot.gripper is not None:
        g = robot.gripper
        grip_path = _load_gripper_xml(g.model, spec)
        try:
            grip = mujoco.MjSpec.from_file(str(grip_path))
        except Exception as e:
            raise WorldError(
                f"机器人 '{robot.name}' 的夹爪解析失败 ({grip_path}): {e}") from e
        if g.site is not None:
            site_name = g.site
        elif g.model in GRIPPERS:
            site_name = gripper_info(g.model)["site"]
        else:                              # 自定义 MJCF 路径：menagerie 惯例挂点名
            site_name = "attachment_site"
        site_names = {s.name for s in child.sites}
        if site_name not in site_names:
            raise WorldError(
                f"机器人 '{robot.name}' 模型里没有挂点 site '{site_name}'，"
                f"可用: {sorted(site_names)}（夹爪 {g.model}）")
        if g.model in GRIPPERS:
            xf = mount_for(g.model, robot.model, g.pos, g.quat)
        elif g.pos is None or g.quat is None:
            raise WorldError(
                f"自定义夹爪 '{g.model}' 不在夹爪库、无标定 mount，必须在 "
                f"gripper 下同时显式写 pos 与 quat 指定相对挂点 '{site_name}' "
                f"的安装变换（库内夹爪则自动标定）")
        else:
            xf = {"pos": list(g.pos), "quat": list(g.quat)}
        try:
            grip_frame = child.attach(grip, site=site_name,
                                      prefix=GRIPPER_PREFIX)
        except Exception as e:
            raise WorldError(
                f"机器人 '{robot.name}' 挂载夹爪 '{g.model}' 失败: {e}") from e
        grip_frame.pos[:] = list(xf["pos"])
        grip_frame.quat[:] = list(xf["quat"])

    _add_child_sensors(robot, child)          # 在合并 spec 上注入（触觉可挂手指）
    act_names = _ensure_actuator_names(child)  # 无名执行器补名（attach 前改名才存活）

    # root body 自带 pos/quat 清零：spec.pos/quat 必须是机器人 root 的绝对世界
    # 位姿，否则 menagerie 四足（trunk 自带 0.445 高度）会被叠加出生到 0.9m
    # 高空——GUI 拖动回写位姿也因此不幂等
    for b in child.bodies:
        if b.parent is child.worldbody:
            b.pos[:] = [0.0, 0.0, 0.0]
            b.quat[:] = [1.0, 0.0, 0.0, 0.0]
    return child, act_names


def _load_robot_spec(model_path: pathlib.Path) -> mujoco.MjSpec:
    """读机器人模型：MJCF 直接读；URDF 做导入适配。

    URDF 与 SolidWorks 工作流（sw2urdf 插件导出）的差异由这里吸收：
    - mesh 路径：MuJoCo 的 from_string 无文件目录上下文，先把全部
      mesh filename 重写为基于 urdf 所在目录的绝对路径；package://NAME/REST
      按 REST / NAME/REST 两级就近查找（sw2urdf 的导出目录惯例）
    - 执行器：URDF 没有 <actuator>（transmission 少有人写），每个关节
      自动补一个同名 motor（力矩控制，agentsim 的 ctrl 直写语义）

    注意：MuJoCo 的 URDF 解析只用 <collision>，<visual> 全部忽略——
    SolidWorks 导出时确保零件有碰撞几何（sw2urdf 默认会生成）。

    Windows 适配：
    - 编码：sw2urdf 在中文 Windows 上可能写出 GBK 编码的 URDF，
      utf-8 优先、GBK 兜底（中文用户路径/注释常见）
    - 绝对路径：/x（POSIX）、C:\\x / C:/x（盘符）、\\\\srv\\x（UNC）
      均不参与相对改写
    - 重写结果统一正斜杠（.as_posix()）：XML 属性 / YAML / 跨平台
      三处都安全，MuJoCo 在 Windows 上也接受正斜杠
    """
    if model_path.suffix.lower() != ".urdf":
        return mujoco.MjSpec.from_file(str(model_path))

    import re
    try:
        text = model_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = model_path.read_text(encoding="gbk", errors="replace")
    base = model_path.parent

    def _resolve(m):
        fn = m.group(1)
        # 跨平台绝对路径不参与改写（POSIX / 盘符 / UNC）
        if re.match(r"^([A-Za-z]:[\\/]|\\\\|/)", fn):
            return fn.replace("\\", "/")
        fn = re.sub(r"^(package|file)://", "", fn)  # sw2urdf 惯例前缀
        fn = fn.replace("\\", "/")                  # 相对路径 \ 也当分隔符
        # package://NAME/REST：先按原相对结构找，再按 NAME 目录下找
        cands = [base / fn]
        parts = fn.split("/", 1)
        if len(parts) == 2 and parts[1]:
            cands.append(base / parts[0] / parts[1])
        for c in cands:
            if c.is_file():
                return c.resolve().as_posix()
        return fn                                   # 找不到保持原样（报错可读）

    text = re.sub(r'filename="([^"]+)"',
                  lambda m: f'filename="{_resolve(m)}"', text)
    try:
        child = mujoco.MjSpec.from_string(text)
    except Exception as e:
        raise WorldError(
            f"URDF 解析失败 ({model_path}): {e}\n"
            f"（提示：MuJoCo 只读取 URDF 的 <collision>；mesh 仅支持 "
            f"STL/OBJ）") from e
    # URDF 无执行器：逐关节补 motor（URDF 解析器已把 fixed joint 合并，
    # joints 列表里全是可动关节）
    for j in child.joints:
        child.add_actuator(name=j.name,
                           trntype=int(mujoco.mjtTrn.mjTRN_JOINT),
                           target=j.name)
    return child


def _attach_robot(spec: WorldSpec, robot: RobotSpec, mj: mujoco.MjSpec) -> None:
    child, _ = _assemble_robot_child(spec, robot)
    frame = mj.worldbody.add_frame(
        name=f"{robot.name}_frame", pos=list(robot.pos),
        quat=list(robot.quat) if robot.quat else None)
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
        # 传感器（imu 拆成 gyro/accel 两个 MuJoCo sensor，按 part 记录读数区间）
        for s in rd.sensors:
            if s.type == "camera":
                cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                                         f"{prefix}{s.name}")
                if cid < 0:
                    raise WorldError(
                        f"编译后找不到机器人 '{rd.name}' 的相机 '{s.name}'"
                        f"（期望全名 '{prefix}{s.name}'）")
                ridx.cameras[s.name] = cid
                continue
            mj_names = ([f"{prefix}{s.name}"] if s.type != "imu"
                        else [f"{prefix}{s.name}_gyro", f"{prefix}{s.name}_accel"])
            parts = []
            for full in mj_names:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, full)
                if sid < 0:
                    raise WorldError(
                        f"编译后找不到机器人 '{rd.name}' 的传感器 '{full}'"
                        f"（编译器吞了名字？）")
                parts.append((int(model.sensor_adr[sid]), int(model.sensor_dim[sid])))
            ridx.sensors[s.name] = SensorIndex(name=s.name, kind=s.type, parts=parts)
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
    """读取装配后（含夹爪）全部执行器名（机器人前缀之外的短名），供索引解析。

    与 _attach_robot 走同一个 _assemble_robot_child，补名/夹爪 prefix
    确定性一致，夹爪执行器（gripper/gripper_act）自动并入。
    """
    _, names = _assemble_robot_child(spec, rd)
    if not names:
        raise WorldError(
            f"机器人 '{rd.name}' 模型里没有任何执行器 ({rd.model})，"
            f"无法建立设备索引")
    return names


def _child_joint_names(spec: WorldSpec, rd: RobotSpec) -> list[str]:
    """读取装配后（含夹爪）全部关节名（机器人前缀之外的短名）。"""
    child, _ = _assemble_robot_child(spec, rd)
    return [j.name for j in child.joints if j.name]


def build_world(spec: WorldSpec) -> tuple[mujoco.MjModel, NameIndex]:
    """WorldSpec -> (MjModel, NameIndex)。schema 校验必须已完成（load_world 保证）。"""
    mj = mujoco.MjSpec()
    mj.modelname = spec.name
    mj.option.timestep = spec.timestep
    mj.option.gravity = list(spec.gravity)
    # attach 时 parent（世界）的 option 优先：大质量机器人需要高
    # impratio（如 spot/anymal 50-100），在此设置才会覆盖模型原值
    mj.option.impratio = spec.impratio
    # 积分器同理被 attach 覆盖：menagerie 高刚度模型（spot 的 solref=0.004）
    # 原生用 implicitfast，被世界默认 Euler 覆盖后"硬接触 + 显式积分"会
    # 数值震荡弹跳（躯干周期性跳起、关节跟踪却良好的根因）
    mj.option.integrator = spec.integrator
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
