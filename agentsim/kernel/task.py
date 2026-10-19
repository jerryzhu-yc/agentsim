"""任务评估与观测组装（纯函数层）。

内核 Simulation 与 RL 环境（rl_env.py）共用这里——两个消费方、
一份语义：success/reward 的定义、qpos/qvel/sensors 的组装规则只允许有
一处实现（两份实现迟早漂移，reward 是任务语义）。
"""
from __future__ import annotations

import mujoco
import numpy as np

from agentsim.world.builder import ResolvedTask, RobotIndex

# 关节类型 -> qpos/qvel 占用宽度（hinge/slide 等默认 1）。
# qvel 宽度与 qpos 不同：free 关节 qpos 7（位置3+四元数4）但 qvel 6（角速度3+线速度3），
# ball 关节 qpos 4（四元数）但 qvel 3（角速度）——此前共用一张表会让带命名
# 自由关节的机器人（spot/g1 等）观测 qvel 错位一位。
_QPOS_WIDTH = {int(mujoco.mjtJoint.mjJNT_FREE): 7,
               int(mujoco.mjtJoint.mjJNT_BALL): 4}
_QVEL_WIDTH = {int(mujoco.mjtJoint.mjJNT_FREE): 6,
               int(mujoco.mjtJoint.mjJNT_BALL): 3}


def obs_layout(model: mujoco.MjModel, ridx: RobotIndex) -> dict[str, tuple[int, int, int]]:
    """观测拼接布局：{关节名: (qpos 偏移, qvel 偏移, qpos 宽度)}。

    robot_obs 按此布局拼接；zoo demo 生成器注入控制器用的偏移也取自这里——
    一份布局，两个消费方，杜绝错位。
    """
    qp = qv = 0
    out = {}
    for jname, jid in ridx.joints.items():
        w = _QPOS_WIDTH.get(int(model.jnt_type[jid]), 1)
        out[jname] = (qp, qv, w)
        qp += w
        qv += _QVEL_WIDTH.get(int(model.jnt_type[jid]), 1)
    return out


def joint_width(model: mujoco.MjModel, ridx: RobotIndex) -> tuple[int, int]:
    """单机器人 qpos/qvel 拼接宽度（观测空间构造用）。"""
    qpos_w = qvel_w = 0
    for jid in ridx.joints.values():
        qpos_w += _QPOS_WIDTH.get(int(model.jnt_type[jid]), 1)
        qvel_w += _QVEL_WIDTH.get(int(model.jnt_type[jid]), 1)
    return qpos_w, qvel_w


def robot_obs(ridx: RobotIndex, model: mujoco.MjModel,
              data: mujoco.MjData) -> tuple[list[float], list[float]]:
    """单机器人 qpos/qvel（按 RobotIndex.joints 声明顺序拼接）。"""
    lay = obs_layout(model, ridx)
    qpos, qvel = [], []
    for jname, jid in ridx.joints.items():
        w = lay[jname][2]
        vw = _QVEL_WIDTH.get(int(model.jnt_type[jid]), 1)
        adr = model.jnt_qposadr[jid]
        qpos += list(map(float, data.qpos[adr:adr + w]))
        vadr = model.jnt_dofadr[jid]
        qvel += list(map(float, data.qvel[vadr:vadr + vw]))
    return qpos, qvel


def task_success(task: ResolvedTask, data: mujoco.MjData) -> bool | None:
    """终态任务判定：成功 body 的 xpos 落在 zone 内。无任务返回 None。"""
    if task.success_body is None:
        return None
    p = data.xpos[task.success_body]
    lo = np.asarray(task.success_min)
    hi = np.asarray(task.success_max)
    return bool(np.all(p >= lo) and np.all(p <= hi))


def task_reward(task: ResolvedTask, model: mujoco.MjModel,
                data: mujoco.MjData) -> float:
    """终态奖励值（分项求和）。"""
    total = 0.0
    for kind, ia, ib, w in task.rewards:
        if kind == "distance":
            total += w * float(np.linalg.norm(
                data.xpos[ia] - data.xpos[ib]))
        elif kind == "zone":
            p = data.xpos[ia]
            lo, hi = np.asarray(ib[0]), np.asarray(ib[1])
            total += w * float(np.all(p >= lo) and np.all(p <= hi))
        elif kind == "contact":
            # a/b 是 body id，contact 里是 geom id：经 geom_bodyid 对齐
            pair = {ia, ib}
            hit = any(
                {model.geom_bodyid[data.contact[i].geom1],
                 model.geom_bodyid[data.contact[i].geom2]}
                == pair
                for i in range(data.ncon))
            total += w * float(hit)
    return total


def sensors_obs(ridx: RobotIndex, data: mujoco.MjData) -> dict:
    """单机器人全部传感器读数 {名: 标量 or 列表}。

    dim=1 出标量（encoder/joint_torque/tactile），多维出列表
    （imu 按 gyro+accel 两个 part 顺序拼接成 6 维）。
    """
    sensors = {}
    for sname, six in ridx.sensors.items():
        vals = []
        for adr, dim in six.parts:
            vals += list(map(float, data.sensordata[adr:adr + dim]))
        sensors[sname] = vals[0] if six.dim == 1 else vals
    return sensors
