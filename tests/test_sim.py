"""M0.2 内核测试：多进程通信、栅栏推进、超时与崩溃策略。

控制器脚本由测试动态生成（tmp_path），走真实子进程 + ZMQ 路径。
"""
import pathlib
import subprocess

import mujoco
import numpy as np

from agentsim.kernel.sim import Simulation
from agentsim.world.builder import build_world
from agentsim.world.schema import load_world, parse_world

ROOT = pathlib.Path(__file__).parent.parent
ARM_MODEL = str(ROOT / "models" / "demo_arm.xml")
WORLD = ROOT / "worlds" / "pick_cube.yaml"


def _world_dict(controller: str | None, **runtime) -> dict:
    d = {
        "name": "sim_test",
        "scene": {"objects": [
            {"name": "cube", "type": "box", "size": [0.05, 0.05, 0.05],
             "pos": [0.3, 0, 0.051], "mass": 0.1, "color": "#F59E0B"}]},
        "robots": [{"name": "arm", "model": ARM_MODEL, "pos": [0, 0, 0.05],
                    "controller": controller}],
        "runtime": runtime or {"controller_timeout": 100.0},
    }
    return d


def _write_controller(tmp_path, code: str) -> str:
    p = tmp_path / "ctrl.py"
    p.write_text(code, encoding="utf-8")
    return str(p)


class TestAcceptance:
    def test_random_arm_1000_ticks(self):
        """验收：random_arm 跑 1000 tick，无死锁、无 NaN、无崩溃"""
        spec = load_world(WORLD)
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=1000)
        # 任务成功会提前终止运行（内核语义），两种结束方式都算通过
        assert stats["ticks"] == 1000 or stats["success"]
        assert not stats["nan"]
        assert stats["crashed"] == []
        assert stats["wall_time"] < 30        # 死锁保护网


class TestCtrlPlumbing:
    def test_ctrl_matches_reference(self, tmp_path):
        """控制量一致性：内核推进的轨迹与进程内参考实现逐位一致"""
        ctrl_path = _write_controller(tmp_path, f"""
from agentsim.controller import Robot
robot = Robot()
robot.set_all({[0.5, 0.3, -0.2, -0.01, 0.01]})
while robot.step():
    pass
robot.close()
""")
        spec = parse_world(_world_dict(ctrl_path))
        with Simulation(spec) as sim:
            sim.run(max_ticks=20)
            kernel_qpos = sim.data.qpos.copy()

        # 进程内参考：同样模型、同样控制量、同样步数
        ref_model, _ = build_world(parse_world(_world_dict(None)))
        ref_data = mujoco.MjData(ref_model)
        spec2 = parse_world(_world_dict(None))
        for _ in range(20):
            ref_data.ctrl[:] = [0.5, 0.3, -0.2, -0.01, 0.01]
            for _ in range(spec2.ctrl_period):
                mujoco.mj_step(ref_model, ref_data)

        assert np.allclose(kernel_qpos, ref_data.qpos), \
            "内核路径与参考实现的物理轨迹不一致"
        assert np.all(np.isfinite(kernel_qpos))

    def test_obs_contains_names_and_state(self, tmp_path):
        """观测里带关节名/位置/body 位置，控制器侧可读"""
        ctrl_path = _write_controller(tmp_path, """
from agentsim.controller import Robot
robot = Robot()
obs = robot.step()
assert "qpos" in obs and "bodies" in obs, f"观测缺字段: {obs.keys()}"
assert "wrist" in obs["bodies"], f"body 视图缺 wrist: {obs['bodies']}"
while robot.step():
    pass
robot.close()
""")
        spec = parse_world(_world_dict(ctrl_path))
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=5)
        assert stats["ticks"] == 5
        assert stats["crashed"] == []       # 断言没炸 = 控制器拿到了合法观测


class TestTimeoutPolicy:
    def test_slow_controller_last_ctrl(self, tmp_path):
        """慢控制器（0.1s/tick）+ 50ms 超时 + last_ctrl：内核补位推进，正常跑完"""
        ctrl_path = _write_controller(tmp_path, """
import time
from agentsim.controller import Robot
robot = Robot()
while robot.step():
    time.sleep(0.1)
robot.close()
""")
        d = _world_dict(ctrl_path, controller_timeout=50.0)
        spec = parse_world(d)
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=10)
        assert stats["ticks"] == 10
        assert stats["lag"] > 0              # 确实发生了超时补位
        assert not stats["nan"]


class TestCrash:
    def test_crash_marks_inactive_no_deadlock(self, tmp_path):
        """控制器跑 2 tick 后 os._exit：内核标记崩溃并优雅结束，不死锁"""
        ctrl_path = _write_controller(tmp_path, """
import os
from agentsim.controller import Robot
robot = Robot()
robot.step()
robot.step()
os._exit(1)
""")
        spec = parse_world(_world_dict(ctrl_path))
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=100)
        assert "arm" in stats["crashed"]
        assert stats["ticks"] >= 2           # 死掉前推进过
        assert not stats["nan"]


class TestNoController:
    def test_world_without_controllers_autoruns(self):
        """无控制器世界：空跑指定 tick（scene_gen 冒烟路径）"""
        spec = parse_world(_world_dict(None))
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=50)
        assert stats["ticks"] == 50
        assert not stats["nan"]


class TestStop:
    """stop_check：GUI 停止按钮 / MCP stop 工具的内核层。"""

    def test_stop_check_premature_finish_with_controller(self, tmp_path):
        """带控制器的世界：stop_check 为真时提前收尾——控制器进程随 stop
        正常退出（不泄漏、不算崩溃），已跑 tick 数小于上限"""
        ctrl_path = _write_controller(tmp_path, """
from agentsim.controller import Robot
robot = Robot()
while robot.step():
    robot.set("a1", 0.3)
""")
        spec = parse_world(_world_dict(ctrl_path))
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=500, stop_check=lambda: sim.tick >= 5)
        assert 5 <= stats["ticks"] < 500     # 提前收尾且至少推进了几 tick
        assert stats["crashed"] == []        # 停止收尾不是崩溃
        assert not stats["nan"]
        proc = sim._procs["arm"]             # 控制器毫秒级退出（无泄漏）
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            raise AssertionError("停止后控制器未在 3s 内退出（泄漏）")

    def test_stop_check_no_controller_world(self):
        """无控制器世界：stop_check 同样生效（空跑循环也要能停）"""
        spec = parse_world(_world_dict(None))
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=10000, stop_check=lambda: sim.tick >= 7)
        assert 7 <= stats["ticks"] < 10000
        assert not stats["nan"]

    def test_stop_check_never_true_runs_to_end(self):
        """stop_check 恒 False：行为与不传完全一致（跑满）"""
        spec = parse_world(_world_dict(None))
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=20, stop_check=lambda: False)
        assert stats["ticks"] == 20
