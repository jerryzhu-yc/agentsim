"""Supervisor 操作：save_state / load_state / reset / teleport。

快照 = (FULLPHYSICS + WARMSTART) + ctrl + tick。

三个刻意的实现决策（都被 T2/T1 的逐位断言保护着）：
1. mjSTATE_FULLPHYSICS 实际只有 time+qpos+qvel+act（实测 24 != 35），
   不含 warmstart——warmstart 影响迭代求解器收敛路径，漏存会导致
   读档后轨迹微小偏移并被逐步放大（T2 实测踩过）。必须显式并上
   mjSTATE_WARMSTART。
2. FULLPHYSICS 不含 data.ctrl，必须单独存——否则读档后第一步就分岔。
3. load 故意不调 mj_forward：forward 会改写 warmstart，破坏逐位一致。
   派生量在下一个物理步自动刷新。
"""
from __future__ import annotations

import mujoco
import numpy as np

from agentsim.errors import WorldError

INIT_TAG = "__init__"          # 保留：reset 用的初始快照
_SNAP_SPEC = (mujoco.mjtState.mjSTATE_FULLPHYSICS
              | mujoco.mjtState.mjSTATE_WARMSTART)


def _check_tag(tag) -> None:
    if not isinstance(tag, str) or not tag:
        raise WorldError(f"快照名需要非空字符串，实际: {tag!r}")
    if tag.startswith("__"):
        raise WorldError(f"快照名不允许 '__' 前缀（保留），实际: {tag!r}")


def list_tags(storage) -> list[str]:
    return sorted(t for t in storage if not t.startswith("__"))


def _snapshot(model, data, tick) -> tuple:
    n = mujoco.mj_stateSize(model, _SNAP_SPEC)
    buf = np.zeros(n)
    mujoco.mj_getState(model, data, buf, _SNAP_SPEC)
    return (buf.tobytes(), tuple(float(c) for c in data.ctrl), int(tick))


def save_state(model, data, storage, tag, tick, internal: bool = False) -> None:
    if not internal:
        _check_tag(tag)
    storage[tag] = _snapshot(model, data, tick)


def load_state(model, data, storage, tag) -> int:
    """读档，返回快照里的 tick（调用方要恢复自己的 tick 计数）。

    注：load 不做 '__' 前缀校验——save 已挡住用户创建保留名，
    而 reset 要读 '__init__'。
    """
    snap = storage.get(tag)
    if snap is None:
        raise WorldError(f"快照 '{tag}' 不存在；可用: {list_tags(storage)}")
    state_bytes, ctrl, tick = snap
    buf = np.frombuffer(state_bytes, dtype=np.float64)
    mujoco.mj_setState(model, data, buf, _SNAP_SPEC)
    data.ctrl[:] = ctrl
    return tick


def teleport(model, data, index, name, pos) -> None:
    """把物体瞬移到 pos（保持姿态为单位四元数，清零该物体速度）。"""
    if name in index.robots:
        raise WorldError(
            f"瞬移失败：'{name}' 是机器人，只有场景物体可以瞬移；"
            f"可用物体: {sorted(index.objects)}")
    if name not in index.objects:
        raise WorldError(
            f"瞬移失败：'{name}' 不存在；可用物体: {sorted(index.objects)}")
    if not isinstance(pos, (list, tuple)) or len(pos) != 3:
        raise WorldError(f"瞬移 pos 需要 3 个数字，实际: {pos!r}")

    bid = index.objects[name]
    free = -1
    for k in range(model.body_jntnum[bid]):
        jid = model.body_jntadr[bid] + k
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            free = jid
    if free < 0:
        raise WorldError(f"物体 '{name}' 没有 freejoint，无法瞬移")

    adr = model.jnt_qposadr[free]
    data.qpos[adr:adr + 3] = [float(v) for v in pos]
    data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
    vadr = model.jnt_dofadr[free]
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)   # 刷新派生量，让位置立即可读
