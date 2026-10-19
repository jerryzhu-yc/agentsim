"""builder 测试：编译正确性、名字索引、任务解析、物理冒烟。"""
import pathlib

import mujoco
import numpy as np
import pytest

from agentsim.errors import WorldError
from agentsim.world.builder import build_world, resolve_task
from agentsim.world.schema import load_world, parse_world

ROOT = pathlib.Path(__file__).parent.parent
WORLD = ROOT / "worlds" / "pick_cube.yaml"


@pytest.fixture(scope="module")
def built():
    spec = load_world(WORLD)
    model, index = build_world(spec)
    return spec, model, index


class TestCompile:
    def test_model_counts(self, built):
        _, model, _ = built
        # worldbody + cube + (base, link1, link2, wrist, finger_l, finger_r)
        assert model.nbody == 1 + 1 + 6
        # cube freejoint + 3 hinge + 2 slide
        assert model.njnt == 6
        assert model.nu == 5      # demo_arm 的 5 个执行器

    def test_options_applied(self, built):
        spec, model, _ = built
        assert model.opt.timestep == spec.timestep
        assert np.allclose(model.opt.gravity, [0, 0, -9.81])

    def test_cube_color_and_mass(self, built):
        _, model, _ = built
        cube_geom = model.geom("cube_geom")
        assert np.allclose(cube_geom.rgba[:3], [0xF5 / 255, 0x9E / 255, 0x0B / 255],
                           atol=0.01)


class TestNameIndex:
    def test_objects(self, built):
        _, model, index = built
        assert index.objects == {"cube": mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "cube")}

    def test_robot_actuators(self, built):
        _, _, index = built
        arm = index.robots["arm"]
        assert set(arm.actuators) == {"a1", "a2", "a3", "a_l", "a_r"}
        # id 是全前缀名的 id
        assert arm.actuators["a1"] == 0

    def test_robot_bodies_unprefixed(self, built):
        _, _, index = built
        arm = index.robots["arm"]
        assert "wrist" in arm.bodies
        assert "finger_l" in arm.bodies

    def test_resolve_body_all_ways(self, built):
        _, _, index = built
        assert index.resolve_body("cube") == index.objects["cube"]
        assert index.resolve_body("arm/wrist") in index.bodies.values()
        # 无前缀引用机器人内部 body（自动补 arm/）
        assert index.resolve_body("wrist") == index.bodies["arm/wrist"]

    def test_resolve_body_error_lists_candidates(self, built):
        _, _, index = built
        with pytest.raises(WorldError) as e:
            index.resolve_body("ghost")
        msg = str(e.value)
        assert "ghost" in msg and "cube" in msg and "arm/wrist" in msg


class TestTaskResolution:
    def test_success_and_reward_resolve(self, built):
        spec, _, index = built
        rt = resolve_task(spec.task, index)
        assert rt.success_body == index.objects["cube"]
        assert rt.success_min == (0.35, 0.25, 0.0)
        assert rt.success_max == (0.65, 0.50, 0.12)
        assert len(rt.rewards) == 2
        kind, a, b, w = rt.rewards[0]
        assert kind == "distance"
        assert a == index.objects["cube"]
        assert b == index.bodies["arm/wrist"]
        assert w == -0.01
        kind, a, zone, w = rt.rewards[1]
        assert kind == "zone"
        assert a == index.objects["cube"]
        assert zone == ((0.35, 0.25, 0.0), (0.65, 0.50, 0.12))
        assert w == 0.01
        assert rt.max_ticks == 5000

    def test_bad_task_body(self, built):
        spec, _, index = built
        from dataclasses import replace
        from agentsim.world.schema import SuccessSpec, ZoneSpec
        bad = replace(spec.task, success=SuccessSpec(
            "ghost", ZoneSpec((0, 0, 0), (1, 1, 1))))
        with pytest.raises(WorldError):
            resolve_task(bad, index)


class TestRobotPlacement:
    def test_attach_pos(self, built):
        """pos = root 绝对位姿（模型自带 pos 被清零吸收进 spec.pos）"""
        _, model, _ = built
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert np.allclose(data.body("arm/base").xpos, [0, 0, 0.05])

    def test_offset_attach(self):
        """机器人放到非原点，前缀 body 位置正确"""
        d = {
            "name": "offset",
            "scene": {"objects": [
                {"name": "ball", "type": "sphere", "size": [0.05],
                 "pos": [0, 0, 0.05]}]},
            "robots": [{"name": "r1",
                        "model": str(ROOT / "models" / "demo_arm.xml"),
                        "pos": [0.5, 0.25, 0.05]}],
        }
        spec = parse_world(d)
        model, index = build_world(spec)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert np.allclose(data.body("r1/base").xpos, [0.5, 0.25, 0.05])

    def test_attach_quat(self):
        """P2/T4：robot.quat（wxyz）传到 root——绕 z +90° 后 R 首行翻转"""
        d = {
            "name": "quaternion",
            "robots": [{"name": "r1",
                        "model": str(ROOT / "models" / "demo_arm.xml"),
                        "pos": [0, 0, 0.05],
                        "quat": [0.7071068, 0, 0, 0.7071068]}],
        }
        spec = parse_world(d)
        model, index = build_world(spec)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        b = data.body("r1/base")
        assert np.allclose(b.xquat, [0.7071068, 0, 0, 0.7071068], atol=1e-3)
        assert np.allclose(b.xmat[:3], [0, -1, 0], atol=1e-3)   # R_z(90°) 首行
        # 默认 None：不写 quat 保持单位朝向
        d["robots"][0].pop("quat")
        model2, _ = build_world(parse_world(d))
        data2 = mujoco.MjData(model2)
        mujoco.mj_forward(model2, data2)
        assert np.allclose(data2.body("r1/base").xquat, [1, 0, 0, 0], atol=1e-6)


class TestSmoke:
    def test_1000_step_stable(self, built):
        """物理冒烟：1000 步无 NaN，方块安稳落在地面上"""
        _, model, _ = built
        data = mujoco.MjData(model)
        for _ in range(1000):
            mujoco.mj_step(model, data)
        assert np.all(np.isfinite(data.qpos))
        cube_z = data.body("cube").xpos[2]
        assert 0.049 < cube_z < 0.052   # 半尺寸 0.05，静止在地面

    def test_two_robots_same_model_no_clash(self):
        """同模型双机器人：前缀隔离，名字不冲突"""
        d = {
            "name": "twin",
            "scene": {"objects": [
                {"name": "ball", "type": "sphere", "size": [0.05],
                 "pos": [0, 0, 0.05]}]},
            "robots": [
                {"name": "a", "model": str(ROOT / "models" / "demo_arm.xml"),
                 "pos": [0, 0, 0.05]},
                {"name": "b", "model": str(ROOT / "models" / "demo_arm.xml"),
                 "pos": [0.6, 0, 0.05]},
            ],
        }
        spec = parse_world(d)
        model, index = build_world(spec)
        assert set(index.robots["a"].actuators) == set(index.robots["b"].actuators)
        assert index.robots["a"].actuators["a1"] != index.robots["b"].actuators["a1"]
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert np.allclose(data.body("b/base").xpos, [0.6, 0, 0.05])
