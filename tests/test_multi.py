"""M0.3 多机器人栅栏测试。

核心命题：不管几个控制器、什么步长、有没有控制器死掉，
内核推进的物理轨迹必须与进程内参考实现逐位一致，且可重现。

参考实现 = 同样的世界（controllers=None）+ 手动按 tick 写同样的控制量。
控制器用确定性控制律（无随机），保证两侧输入完全相同。
"""
import pathlib

import mujoco
import numpy as np

from agentsim.kernel.sim import Simulation
from agentsim.world.builder import build_world
from agentsim.world.schema import parse_world

ROOT = pathlib.Path(__file__).parent.parent
ARM_MODEL = str(ROOT / "models" / "demo_arm.xml")

# 确定性控制律（测试与控制器脚本共用同一表达式，两侧数值必须一致）
CTRL_A = "[0.05*(t%5), 0.03*(t%7), -0.02*(t%3), -0.005*(t%4), 0.005*(t%4)]"
CTRL_B = "[-0.04*(t%6), 0.02*(t%5), 0.03*(t%4), -0.008*(t%3), 0.008*(t%3)]"


def _two_robot_dict(ctrl_a: str | None, ctrl_b: str | None,
                     timeout: float = 100.0) -> dict:
    return {
        "name": "multi_test",
        "scene": {"objects": [
            {"name": "cube", "type": "box", "size": [0.05, 0.05, 0.05],
             "pos": [0.3, 0, 0.051], "mass": 0.1, "color": "#F59E0B"}]},
        "robots": [
            {"name": "a", "model": ARM_MODEL, "pos": [0, 0, 0.05],
             "controller": ctrl_a},
            {"name": "b", "model": ARM_MODEL, "pos": [0.6, 0, 0.05],
             "controller": ctrl_b},
        ],
        "runtime": {"controller_timeout": timeout},
    }


def _run_kernel(d: dict, ticks: int) -> tuple[np.ndarray, dict]:
    spec = parse_world(d)
    with Simulation(spec) as sim:
        stats = sim.run(max_ticks=ticks)
        return sim.data.qpos.copy(), stats


def _run_reference(d: dict, ticks: int, ctrl_fn: dict) -> np.ndarray:
    """进程内参考：ctrl_fn[机器人名](tick) -> 控制量列表"""
    spec = parse_world(d)
    model, index = build_world(spec)
    data = mujoco.MjData(model)
    for t in range(ticks):
        for rname, fn in ctrl_fn.items():
            ids = list(index.robots[rname].actuators.values())
            data.ctrl[ids] = fn(t)
        for _ in range(spec.ctrl_period):
            mujoco.mj_step(model, data)
    return data.qpos.copy()


def _ctrl_code(ctrl_expr: str, n: int = 1, stop_after: int | None = None,
               hard_exit: bool = False) -> str:
    """生成确定性控制器脚本源码。

    语义：每次请求前写 f(t) 的控制量再 step(n)；t 每次加 n。
    这样全局 tick i 上应用的控制量 = f(t)，与参考实现逐 tick 对齐。
    """
    exit_block = ""
    if stop_after is not None:
        action = ("import os; os._exit(1)" if hard_exit
                  else "robot.close(); raise SystemExit(0)")
        exit_block = (f"    if t >= {stop_after}:\n"
                      f"        {action}\n")
    return f"""from agentsim.controller import Robot

robot = Robot()
t = 0
while True:
    robot.set_all({ctrl_expr})
    if not robot.step({n}):
        break
    t += {n}
{exit_block}robot.close()
"""


class TestTwoRobots:
    def test_trajectory_matches_reference(self, tmp_path):
        """双机器人 + 各自控制律：内核轨迹与参考实现逐位一致"""
        ca = tmp_path / "ctrl_a.py"
        cb = tmp_path / "ctrl_b.py"
        ca.write_text(_ctrl_code(CTRL_A), encoding="utf-8")
        cb.write_text(_ctrl_code(CTRL_B), encoding="utf-8")

        qpos, stats = _run_kernel(_two_robot_dict(str(ca), str(cb)), 15)
        assert stats["ticks"] == 15
        assert stats["crashed"] == []

        ref = _run_reference(_two_robot_dict(None, None), 15, {
            "a": lambda t: [0.05*(t % 5), 0.03*(t % 7), -0.02*(t % 3),
                            -0.005*(t % 4), 0.005*(t % 4)],
            "b": lambda t: [-0.04*(t % 6), 0.02*(t % 5), 0.03*(t % 4),
                            -0.008*(t % 3), 0.008*(t % 3)],
        })
        assert np.allclose(qpos, ref), "双机器人轨迹与参考实现不一致"

    def test_reproducible_across_runs(self, tmp_path):
        """可重现性：同一世界跑两次，轨迹完全相同（AI 迭代的根基）"""
        ca = tmp_path / "ctrl_a.py"
        ca.write_text(_ctrl_code(CTRL_A), encoding="utf-8")
        d = _two_robot_dict(str(ca), None)   # 只有 a 有控制器，b 空控制

        q1, _ = _run_kernel(d, 10)
        q2, _ = _run_kernel(d, 10)
        assert np.array_equal(q1, q2)


class TestMixedStepRates:
    def test_step_n2_barrier(self, tmp_path):
        """混合步长：b 每次 step(2)，a 每次 step(1)，栅栏正确合并"""
        ca = tmp_path / "ctrl_a.py"
        cb = tmp_path / "ctrl_b.py"
        ca.write_text(_ctrl_code(CTRL_A, n=1), encoding="utf-8")
        cb.write_text(_ctrl_code(CTRL_B, n=2), encoding="utf-8")

        ticks = 16
        qpos, stats = _run_kernel(_two_robot_dict(str(ca), str(cb)), ticks)
        assert stats["ticks"] == ticks
        assert stats["crashed"] == []

        def f_a(t):
            return [0.05*(t % 5), 0.03*(t % 7), -0.02*(t % 3),
                    -0.005*(t % 4), 0.005*(t % 4)]

        def f_b(t):     # b 的控制量在它的每个 2-tick 跨度内保持不变
            k = (t // 2) * 2
            return [-0.04*(k % 6), 0.02*(k % 5), 0.03*(k % 4),
                    -0.008*(k % 3), 0.008*(k % 3)]

        ref = _run_reference(_two_robot_dict(None, None), ticks,
                             {"a": f_a, "b": f_b})
        assert np.allclose(qpos, ref), "混合步长轨迹与参考实现不一致"


class TestKillAndBye:
    def test_hard_kill_other_continues(self, tmp_path):
        """a 跑 3 tick 后 os._exit：b 继续推进到上限，a 冻结在最后控制量"""
        ca = tmp_path / "ctrl_a.py"
        cb = tmp_path / "ctrl_b.py"
        ca.write_text(_ctrl_code(CTRL_A, stop_after=3, hard_exit=True),
                      encoding="utf-8")
        cb.write_text(_ctrl_code(CTRL_B), encoding="utf-8")

        ticks = 12
        qpos, stats = _run_kernel(_two_robot_dict(str(ca), str(cb)), ticks)
        assert stats["ticks"] == ticks          # b 跑满
        assert "a" in stats["crashed"]
        assert "b" not in stats["crashed"]

        f_a = lambda t: [0.05*(t % 5), 0.03*(t % 7), -0.02*(t % 3),
                         -0.005*(t % 4), 0.005*(t % 4)]
        f_b = lambda t: [-0.04*(t % 6), 0.02*(t % 5), 0.03*(t % 4),
                         -0.008*(t % 3), 0.008*(t % 3)]

        def f_a_frozen(t):                     # t>=3 后 a 的控制量冻结在 f_a(2)
            return f_a(t) if t < 3 else f_a(2)

        ref = _run_reference(_two_robot_dict(None, None), ticks,
                             {"a": f_a_frozen, "b": f_b})
        assert np.allclose(qpos, ref), "崩溃后 b 的轨迹与参考实现不一致"

    def test_clean_bye_other_continues(self, tmp_path):
        """a 跑 3 tick 后正常退出（bye）：b 继续推进，a 记为 done 而非 crashed"""
        ca = tmp_path / "ctrl_a.py"
        cb = tmp_path / "ctrl_b.py"
        ca.write_text(_ctrl_code(CTRL_A, stop_after=3, hard_exit=False),
                      encoding="utf-8")
        cb.write_text(_ctrl_code(CTRL_B), encoding="utf-8")

        ticks = 10
        qpos, stats = _run_kernel(_two_robot_dict(str(ca), str(cb)), ticks)
        assert stats["ticks"] == ticks
        assert "a" in stats["done"]
        assert stats["crashed"] == []

        f_a = lambda t: [0.05*(t % 5), 0.03*(t % 7), -0.02*(t % 3),
                         -0.005*(t % 4), 0.005*(t % 4)]
        f_b = lambda t: [-0.04*(t % 6), 0.02*(t % 5), 0.03*(t % 4),
                         -0.008*(t % 3), 0.008*(t % 3)]

        def f_a_frozen(t):
            return f_a(t) if t < 3 else f_a(2)

        ref = _run_reference(_two_robot_dict(None, None), ticks,
                             {"a": f_a_frozen, "b": f_b})
        assert np.allclose(qpos, ref)
