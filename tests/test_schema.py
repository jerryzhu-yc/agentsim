"""schema 校验测试：合法世界全字段解析 + 各种坏输入报错可读。"""
import pytest

from agentsim.errors import WorldError
from agentsim.world.schema import load_world, parse_world, hex_to_rgba

import pathlib

ROOT = pathlib.Path(__file__).parent.parent
WORLD = ROOT / "worlds" / "pick_cube.yaml"


def _base() -> dict:
    """最小合法世界"""
    return {
        "name": "t",
        "scene": {"objects": [
            {"name": "ball", "type": "sphere", "size": [0.1], "pos": [0, 0, 0.1]}
        ]},
    }


class TestValidWorld:
    def test_pick_cube_loads(self):
        spec = load_world(WORLD)
        assert spec.name == "pick_cube"
        assert spec.timestep == 0.002
        assert spec.ctrl_period == 32
        assert spec.gravity == (0.0, 0.0, -9.81)

        cube = spec.scene.objects[0]
        assert cube.name == "cube"
        assert cube.type == "box"
        assert cube.size == (0.03, 0.035, 0.05)
        assert cube.mass == 0.05
        assert cube.friction == 0.3
        assert spec.scene.floor.friction == 0.3

        arm = spec.robots[0]
        assert arm.name == "arm"
        assert arm.model.endswith("models/demo_arm.xml")
        assert arm.controller.endswith("pick_place.py")
        assert arm.args == {}

        assert spec.task.success.body == "cube"
        assert spec.task.max_ticks == 5000
        assert spec.runtime.timeout_policy == "last_ctrl"
        assert spec.base_dir == WORLD.resolve().parent

    def test_minimal_world(self):
        spec = parse_world(_base())
        assert spec.scene.floor.size == (2.0, 2.0)
        assert spec.robots == []
        assert spec.runtime.realtime is False


class TestSchemaErrors:
    def _err(self, d):
        with pytest.raises(WorldError) as e:
            parse_world(d)
        return str(e.value)

    def test_missing_name(self):
        msg = self._err({})
        assert "缺少必填字段" in msg and "name" in msg

    def test_duplicate_names(self):
        d = _base()
        d["robots"] = [{"name": "ball", "model": "x.xml"}]
        assert "重复" in self._err(d)

    def test_bad_geom_type(self):
        d = _base()
        d["scene"]["objects"][0]["type"] = "torus"
        assert "torus" in self._err(d)

    def test_bad_size_arity(self):
        d = _base()
        d["scene"]["objects"][0]["size"] = [0.1, 0.2]
        assert "1 个数字" in self._err(d)

    def test_negative_size(self):
        d = _base()
        d["scene"]["objects"][0]["size"] = [-0.1]
        assert "必须为正" in self._err(d)

    def test_bad_color(self):
        d = _base()
        d["scene"]["objects"][0]["color"] = "red"
        assert "#RRGGBB" in self._err(d)

    def test_bad_zone(self):
        d = _base()
        d["task"] = {"success": {"body": "ball",
                                 "zone": {"min": [0, 0, 0], "max": [0, 0, 1]}}}
        assert "min 必须 < max" in self._err(d)

    def test_unknown_task_body_without_robots(self):
        d = _base()
        d["task"] = {"success": {"body": "ghost",
                                 "zone": {"min": [0, 0, 0], "max": [1, 1, 1]}}}
        msg = self._err(d)
        assert "ghost" in msg and "ball" in msg  # 错误里要列出可用名字

    def test_unknown_reward_kind(self):
        d = _base()
        d["task"] = {"reward": [{"velocity": {"a": "ball"}}]}
        assert "velocity" in self._err(d)

    def test_bad_timeout_policy(self):
        d = _base()
        d["runtime"] = {"timeout_policy": "explode"}
        assert "explode" in self._err(d)

    def test_bad_ctrl_period(self):
        d = _base()
        d["ctrl_period"] = 0
        assert "ctrl_period" in self._err(d)

    def test_missing_world_file(self):
        with pytest.raises(WorldError) as e:
            load_world("不存在.yaml")
        assert "不存在" in str(e.value)


class TestHexColor:
    def test_hex_to_rgba(self):
        assert hex_to_rgba("#FF8000") == [1.0, 0.502, 0.0, 1.0]
