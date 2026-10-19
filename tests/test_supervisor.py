"""M0.4 Supervisor 测试（红先行，按方案 6 用例设计）。

验收核心（T2）：save -> 扰动(teleport+异控制律) -> load，
load 后的轨迹与从未扰动的参考分支逐位一致（array_equal，不是 allclose）。

反空转护栏：扰动分支在 load 前存了 "perturbed" 快照，
必须断言它与 "snap" 不同——否则扰动没生效，测试空转也绿。
"""
import pathlib

import mujoco
import numpy as np
import pytest

from agentsim.errors import WorldError
from agentsim.kernel.sim import Simulation
from agentsim.world.schema import parse_world

ROOT = pathlib.Path(__file__).parent.parent
ARM_MODEL = str(ROOT / "models" / "demo_arm.xml")

# 确定性控制律（与 test_multi 一致，两侧数值必须完全相同）
F = "[0.05*(t%5), 0.03*(t%7), -0.02*(t%3), -0.005*(t%4), 0.005*(t%4)]"
G = "[-0.04*(t%6), 0.02*(t%5), 0.03*(t%4), -0.008*(t%3), 0.008*(t%3)]"


def _world(controller: str | None = None, with_robot: bool = True) -> dict:
    robots = []
    if with_robot:
        robots.append({"name": "a", "model": ARM_MODEL, "pos": [0, 0, 0.05],
                       "controller": controller})
    return {
        "name": "supervisor_test",
        "scene": {"objects": [
            {"name": "cube", "type": "box", "size": [0.05, 0.05, 0.05],
             "pos": [0.3, 0, 0.051], "mass": 0.1, "color": "#F59E0B"}]},
        "robots": robots,
        "runtime": {"controller_timeout": 100.0},
    }


def _step(sim: Simulation, ticks: int, ctrl_fn=None) -> None:
    """进程内手动推进（无控制器路径），与内核 _advance 同语义。"""
    ids = (list(sim.index.robots["a"].actuators.values())
           if "a" in sim.index.robots else [])
    for t in range(ticks):
        if ctrl_fn is not None and ids:
            sim.data.ctrl[ids] = ctrl_fn(t)
        for _ in range(sim.spec.ctrl_period):
            mujoco.mj_step(sim.model, sim.data)


class TestInProcess:
    """进程内 API：save/load/reset/teleport"""

    def test_T1_state_bytes_roundtrip(self):
        """存档->继续跑(分岔)->读档->再存档：字节级一致"""
        sim = Simulation(parse_world(_world()))
        _step(sim, 5)
        sim.save_state("s1")
        q_before = sim.data.qpos.copy()
        _step(sim, 10)                      # 制造分岔（cube 落地弹跳）
        assert not np.array_equal(sim.data.qpos, q_before)   # 确实分岔了
        sim.load_state("s1")
        sim.save_state("s2")
        assert sim.get_state("s1") == sim.get_state("s2")
        assert np.array_equal(sim.data.qpos, q_before)
        sim.close()

    def test_T3_teleport_object(self):
        """瞬移：位置立即生效、自由落体稳定；瞬移机器人/幽灵名 -> 可读错误"""
        sim = Simulation(parse_world(_world()))
        sim.teleport("cube", [1.0, -0.8, 0.3])
        assert np.allclose(sim.data.body("cube").xpos, [1.0, -0.8, 0.3])
        _step(sim, 100)
        assert np.all(np.isfinite(sim.data.qpos))
        z = sim.data.body("cube").xpos[2]
        assert 0.049 < z < 0.052, "cube 应落回地面（半尺寸 0.05）"
        # 机器人名：没有 freejoint，应报可读错误
        with pytest.raises(WorldError) as e:
            sim.teleport("a", [0, 0, 0.5])
        assert "a" in str(e.value) and "物体" in str(e.value)
        # 幽灵名：报错要列出可用物体
        with pytest.raises(WorldError) as e:
            sim.teleport("ghost", [0, 0, 0.5])
        assert "cube" in str(e.value)
        sim.close()

    def test_T4_reset_to_initial(self):
        """reset 回初始：状态==新编译模型；同控制律重跑逐位一致"""
        spec = parse_world(_world())
        sim = Simulation(spec)
        f = eval(f"lambda t: {F}")
        _step(sim, 8, ctrl_fn=f)
        q1 = sim.data.qpos.copy()

        sim.reset()
        # 与全新模型逐位一致
        from agentsim.world.builder import build_world
        m0, _ = build_world(parse_world(_world()))
        fresh = mujoco.MjData(m0)
        assert np.array_equal(sim.data.qpos, fresh.qpos)
        # 同控制律重跑 -> 与第一次逐位一致
        _step(sim, 8, ctrl_fn=f)
        assert np.array_equal(sim.data.qpos, q1)
        sim.close()

    def test_T5_snapshot_reproducible_across_runs(self):
        """同一世界两次运行，同 tick 存档：快照字节相同"""
        snaps = []
        for _ in range(2):
            sim = Simulation(parse_world(_world()))
            _step(sim, 7)
            sim.save_state("s")
            snaps.append(sim.get_state("s"))
            sim.close()
        assert snaps[0] == snaps[1]

    def test_T6_unknown_tag_error_lists_candidates(self):
        """load 不存在的 tag：错误信息列出可用 tag"""
        sim = Simulation(parse_world(_world()))
        sim.save_state("snap")
        with pytest.raises(WorldError) as e:
            sim.load_state("nope")
        msg = str(e.value)
        assert "nope" in msg and "snap" in msg
        sim.close()


class TestEndToEnd:
    """T2 验收主用例：save -> 扰动 -> load，经真实控制器 IPC 全链路"""

    def _run_with_controller(self, tmp_path, name: str, code: str,
                              max_ticks: int):
        path = tmp_path / f"{name}.py"
        path.write_text(code, encoding="utf-8")
        spec = parse_world(_world(controller=str(path)))
        sim = Simulation(spec)
        stats = sim.run(max_ticks=max_ticks)
        return sim, stats

    def test_T2_save_perturb_load_bitwise(self, tmp_path):
        T, K, N = 8, 4, 10          # 8 tick 后存档，扰动 4 tick，读档后再跑 10

        ctrl_a = f"""from agentsim.controller import Robot

def f(t):
    return {F}

robot = Robot()
t = 0
while True:
    robot.set_all(f(t))
    if not robot.step():
        break
    t += 1
    if t == {T}:
        robot.supervise("save", tag="snap")
"""
        ctrl_b = f"""from agentsim.controller import Robot

def f(t):
    return {F}

def g(t):
    return {G}

robot = Robot()
T, K = {T}, {K}
t = 0
while True:
    if t <= T:
        robot.set_all(f(t))
    elif t < T + K:
        robot.set_all(g(t - T))            # 扰动段：迭代 T+1..T+K-1
    else:
        robot.set_all(f(robot.tick))      # 读档后按内核 tick 对齐参考段
        # （load 在迭代 T+K-1 末尾触发，此时 t 已=T+K，必须走 else 分支，
        #   否则 g 会多打一拍，整个 f 序列错位——off-by-one 实测踩过）
    if not robot.step():
        break
    t += 1
    if t == T:
        robot.supervise("save", tag="snap")
    elif t == T + 1:
        robot.supervise("teleport", body="cube", pos=[1.2, -0.8, 0.3])
    elif t == T + K:
        robot.supervise("save", tag="perturbed")
        robot.supervise("load", tag="snap")
"""

        sim_a, stats_a = self._run_with_controller(tmp_path, "ref", ctrl_a,
                                                   max_ticks=T + N)
        q_ref = sim_a.data.qpos.copy()
        assert stats_a["ticks"] == T + N and stats_a["crashed"] == []
        sim_a.close()

        sim_b, stats_b = self._run_with_controller(tmp_path, "ptb", ctrl_b,
                                                   max_ticks=T + N)
        q_load = sim_b.data.qpos.copy()
        assert stats_b["ticks"] == T + N and stats_b["crashed"] == []
        assert stats_b["lag"] == 0, "补位非零说明时序不可控，验收不干净"

        # 反空转护栏：扰动分支在 load 前确实分岔了
        snap = sim_b.get_state("snap")
        perturbed = sim_b.get_state("perturbed")
        assert snap != perturbed, "扰动没有生效（teleport/g 控制律无效？）"

        # 主断言：读档后轨迹与参考分支逐位一致
        assert np.array_equal(q_load, q_ref), (
            f"读档后轨迹不一致: max|diff|="
            f"{np.max(np.abs(q_load - q_ref))}")
        sim_b.close()
