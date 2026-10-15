"""M0.5 验收：端到端抓取（pick_cube.yaml + pick_place.py 全链路）。

验收标准（M0 方案 5.6 节）：
- 相位机控制器抓起 cube、搬运、释放，内核判定 success
- cube 终态在成功区（释放方位的地面 zone）
- 全链路逐位可重现
"""
import pathlib

import numpy as np

from agentsim.kernel.sim import Simulation
from agentsim.world.schema import load_world

ROOT = pathlib.Path(__file__).parent.parent
WORLD = ROOT / "worlds" / "pick_cube.yaml"
ZONE_LO, ZONE_HI = (0.46, 0.29, 0.0), (0.61, 0.44, 0.06)


def run_once():
    spec = load_world(WORLD)
    with Simulation(spec) as sim:
        stats = sim.run()
        cube = sim.data.xpos[sim.index.objects["cube"]].copy()
        qpos = sim.data.qpos.copy()
    return stats, cube, qpos


class TestPick:
    def test_end_to_end_success(self):
        """抓取→搬运→释放全链路：任务 success，且相位机正常退出未被截断"""
        stats, cube, _ = run_once()
        assert stats["ticks"] < 5000, "相位机没有在 max_ticks 内正常退出"
        assert stats["success"] is True, f"任务未成功，cube 终态: {cube}"
        assert not stats["nan"]
        assert stats["crashed"] == []

    def test_cube_lands_in_zone(self):
        """cube 终态落在成功区内（释放方位的地面 zone）"""
        _, cube, _ = run_once()
        for i in range(3):
            assert ZONE_LO[i] < cube[i] < ZONE_HI[i], \
                f"cube 分量 {i} 越界: {cube}"

    def test_bitwise_reproducible(self):
        """同一世界跑两遍：终态 cube 位置与全量 qpos 逐位一致"""
        _, c1, q1 = run_once()
        _, c2, q2 = run_once()
        assert np.array_equal(c1, c2), f"cube 终态不一致: {c1} vs {c2}"
        assert np.array_equal(q1, q2), "全量 qpos 终态不一致"
