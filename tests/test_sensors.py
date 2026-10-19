"""M2 传感器系统测试：schema 校验 -> builder 编译 -> 观测流 -> bit 级守护。"""
import pathlib

import mujoco
import numpy as np
import pytest
import yaml

from agentsim.errors import WorldError
from agentsim.kernel.sim import Simulation
from agentsim.world.builder import build_world
from agentsim.world.schema import parse_world

ROOT = pathlib.Path(__file__).parent.parent
WORLD = ROOT / "worlds" / "pick_cube.yaml"

# demo_arm 的五类传感器全套（挂在 pick_cube 的 arm 上）
_FULL_SENSORS = [
    {"name": "j1_enc", "type": "encoder", "joint": "j1"},
    {"name": "j1_tau", "type": "joint_torque", "joint": "j1"},
    {"name": "wrist_imu", "type": "imu", "body": "wrist", "pos": [0, 0, 0.02]},
    {"name": "fl_touch", "type": "tactile", "body": "finger_l", "pos": [0, 0, 0.05]},
    {"name": "gripper_cam", "type": "camera", "body": "wrist",
     "pos": [0, 0.06, 0.10], "quat": [0.7071, 0.7071, 0, 0]},
]


def _arm_world(sensors):
    """带传感器的最小机械臂世界（模型路径同 pick_cube，基准 worlds/ 目录）。"""
    return {
        "name": "sensors_t",
        "scene": {"objects": [
            {"name": "cube", "type": "box", "size": [0.03, 0.035, 0.05],
             "pos": [0.5, 0.2, 0.05], "mass": 0.05}
        ]},
        "robots": [{
            "name": "arm", "model": "../models/demo_arm.xml",
            "pos": [0.35, 0.15, 0.1], "sensors": sensors,
        }],
    }


class TestSensorSchema:                        # ------------------------------ T1
    def test_full_set_parses(self):
        spec = parse_world(_arm_world(_FULL_SENSORS))
        arm = spec.robots[0]
        assert [s.name for s in arm.sensors] == \
            ["j1_enc", "j1_tau", "wrist_imu", "fl_touch", "gripper_cam"]
        assert arm.sensors[2].pos == (0.0, 0.0, 0.02)
        assert arm.sensors[4].quat == (0.7071, 0.7071, 0.0, 0.0)

    def test_unknown_type(self):
        w = _arm_world([{"name": "s", "type": "lidar", "joint": "j1"}])
        with pytest.raises(WorldError, match="未知传感器类型 'lidar'"):
            parse_world(w)

    def test_encoder_requires_joint(self):
        with pytest.raises(WorldError, match="需要 joint 字段"):
            parse_world(_arm_world([{"name": "s", "type": "encoder"}]))

    def test_joint_sensor_rejects_body(self):
        w = _arm_world([{"name": "s", "type": "encoder", "joint": "j1",
                         "body": "wrist"}])
        with pytest.raises(WorldError, match="不使用 body"):
            parse_world(w)

    def test_body_sensor_rejects_joint(self):
        w = _arm_world([{"name": "s", "type": "imu", "body": "wrist",
                         "joint": "j1"}])
        with pytest.raises(WorldError, match="不使用 joint"):
            parse_world(w)

    def test_body_sensor_requires_body(self):
        with pytest.raises(WorldError, match="需要 body 字段"):
            parse_world(_arm_world([{"name": "s", "type": "tactile"}]))

    def test_duplicate_names(self):
        w = _arm_world([{"name": "s", "type": "encoder", "joint": "j1"},
                        {"name": "s", "type": "encoder", "joint": "j2"}])
        with pytest.raises(WorldError, match="'s' 重复"):
            parse_world(w)

    def test_slash_in_name(self):
        w = _arm_world([{"name": "a/b", "type": "encoder", "joint": "j1"}])
        with pytest.raises(WorldError, match="不能含 '/'"):
            parse_world(w)

    def test_quat_only_for_camera(self):
        w = _arm_world([{"name": "s", "type": "imu", "body": "wrist",
                         "quat": [1, 0, 0, 0]}])
        with pytest.raises(WorldError, match="只有 camera 用 quat"):
            parse_world(w)

    def test_radius_only_for_tactile(self):
        w = _arm_world([{"name": "s", "type": "imu", "body": "wrist",
                         "radius": 0.05}])
        with pytest.raises(WorldError, match="只有 tactile 用 radius"):
            parse_world(w)

    def test_radius_must_be_positive(self):
        w = _arm_world([{"name": "s", "type": "tactile", "body": "finger_l",
                         "radius": 0}])
        with pytest.raises(WorldError, match="必须为正数"):
            parse_world(w)

    def test_tactile_default_radius(self):
        """未写 radius 的 tactile 默认 0.06（touch 接触过滤球半径）。"""
        spec = parse_world(_arm_world([{"name": "s", "type": "tactile",
                                       "body": "finger_l"}]))
        assert spec.robots[0].sensors[0].radius == 0.06

    def test_no_sensors_is_empty_list(self):
        spec = parse_world(_arm_world([]))
        assert spec.robots[0].sensors == []


@pytest.fixture(scope="module")
def built():
    spec = parse_world(_arm_world(_FULL_SENSORS))
    spec.base_dir = ROOT / "worlds"
    return spec, *build_world(spec)


class TestSensorBuild:                        # ------------------------------ T2

    def test_model_counts(self, built):
        _, model, _ = built
        # j1_enc + j1_tau + wrist_imu(gyro+accel) + fl_touch = 5 个 sensor
        assert model.nsensor == 5
        assert model.nsite == 2                    # wrist_imu_site + fl_touch_site
        assert model.ncam == 1                     # gripper_cam

    def test_sensor_index_and_dims(self, built):
        _, model, index = built
        arm = index.robots["arm"]
        assert set(arm.sensors) == {"j1_enc", "j1_tau", "wrist_imu", "fl_touch"}
        assert arm.sensors["j1_enc"].dim == 1
        assert arm.sensors["j1_tau"].dim == 1
        assert arm.sensors["fl_touch"].dim == 1
        assert arm.sensors["wrist_imu"].dim == 6   # gyro 3 + accel 3
        # adr 与编译后 sensordata 布局一致
        enc = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "arm/j1_enc")
        assert arm.sensors["j1_enc"].parts[0] == \
            (int(model.sensor_adr[enc]), int(model.sensor_dim[enc]))

    def test_camera_index(self, built):
        _, model, index = built
        arm = index.robots["arm"]
        assert arm.cameras["gripper_cam"] == \
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "arm/gripper_cam")

    def test_bad_joint_reference_lists_candidates(self):
        w = _arm_world([{"name": "s", "type": "encoder", "joint": "j9"}])
        spec = parse_world(w)
        spec.base_dir = ROOT / "worlds"
        with pytest.raises(WorldError, match="可用.*j1.*j2"):
            build_world(spec)

    def test_bad_body_reference_lists_candidates(self):
        w = _arm_world([{"name": "s", "type": "imu", "body": "hand"}])
        spec = parse_world(w)
        spec.base_dir = ROOT / "worlds"
        with pytest.raises(WorldError, match="可用.*wrist"):
            build_world(spec)

    def test_imu_gyro_accel_contiguous(self, built):
        """imu 的 gyro/accel 两 part 在 sensordata 里必须连续（obs 拼接的前提）。"""
        _, _, index = built
        parts = index.robots["arm"].sensors["wrist_imu"].parts
        assert parts[0][0] + parts[0][1] == parts[1][0]

    def test_sensordata_sane_at_rest(self, built):
        """静止一步后：accel z≈-9.81（世界系重力），gyro≈0。"""
        _, model, _ = built
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        _, _, index = built
        adr = index.robots["arm"].sensors["wrist_imu"].parts[0][0]
        v = data.sensordata[adr:adr + 6]
        assert abs(v[2]) < 1e-6                     # 静止无角速度
        assert abs(v[5] - 9.81) < 0.01              # 重力（MuJoCo 传感器 z 朝上）


# ---------------------------------------------------------------- 观测流（T3/T4/T7）

class TestObsFlow:
    def test_obs_msg_has_sensors(self):
        """无控制器世界推进后，obs 消息带传感器读数（T3+T4 内核路径）。"""
        spec = parse_world(_arm_world(_FULL_SENSORS))
        spec.base_dir = ROOT / "worlds"
        with Simulation(spec) as sim:
            sim.run(max_ticks=1)
            msg = sim._obs_msg("arm")
        assert "sensors" in msg
        s = msg["sensors"]
        assert set(s) == {"j1_enc", "j1_tau", "wrist_imu", "fl_touch"}
        assert s["j1_enc"] == msg["qpos"][0]         # encoder == j1 关节位置
        imu = s["wrist_imu"]
        assert len(imu) == 6
        assert abs(imu[2]) < 1e-6                    # 静止 gyro z
        assert abs(imu[5] - 9.81) < 0.01            # 静止 accel z（重力）
        assert s["fl_touch"] == 0.0                 # 静止无接触
        assert isinstance(s["j1_tau"], float)       # 声明过就一定有读数

    def test_no_sensors_key_without_declaration(self):
        """未声明传感器的机器人 obs 消息不出现 sensors 键（T7 向后兼容）。"""
        spec = parse_world(_arm_world([]))
        spec.base_dir = ROOT / "worlds"
        with Simulation(spec) as sim:
            sim.run(max_ticks=1)
            assert "sensors" not in sim._obs_msg("arm")


class TestControllerProtocol:
    def test_hello_manifest_and_sensor_api(self, tmp_path):
        """真实 ZMQ 协议端到端：hello 清单 + sensor() 便捷读取（控制器进程）。"""
        ctrl = tmp_path / "assert_sensors.py"
        ctrl.write_text(
            "from agentsim.controller import Robot\n"
            "r = Robot()\n"
            "names = {s['name'] for s in r.sensors}\n"
            "assert names == {'j1_enc', 'j1_tau', 'wrist_imu', 'fl_touch'}, names\n"
            "dims = {s['name']: s['dim'] for s in r.sensors}\n"
            "assert dims['wrist_imu'] == 6 and dims['j1_enc'] == 1\n"
            "obs = r.step()\n"
            "assert 'sensors' in obs\n"
            "imu = r.sensor('wrist_imu')\n"
            "assert abs(imu[5] - 9.81) < 0.05, imu\n"
            "assert abs(r.sensor('j1_enc')) < 1e-9\n"
            "assert r.sensor('fl_touch') == 0.0\n",
            encoding="utf-8")
        spec = parse_world(_arm_world(_FULL_SENSORS[:4]))    # 不带相机
        spec.robots[0].controller = str(ctrl)
        spec.base_dir = ROOT / "worlds"
        with Simulation(spec) as sim:
            stats = sim.run(max_ticks=5)
        # 控制器断言失败会以非零退出码计入 crashed——这里必须干净
        assert stats["crashed"] == []
        assert stats["ticks"] >= 1


# ---------------------------------------------------------------- 触觉与守护（T5/T6）

_SENSORS_ON_ARM = """    sensors:
      - {name: j1_enc, type: encoder, joint: j1}
      - {name: j1_tau, type: joint_torque, joint: j1}
      - {name: wrist_imu, type: imu, body: wrist, pos: [0, 0, 0.02]}
      - {name: fl_touch, type: tactile, body: finger_l, pos: [0, 0, 0.05]}
      - {name: gripper_cam, type: camera, body: wrist, pos: [0, 0.06, 0.10], quat: [0.7071, 0.7071, 0, 0]}"""


def _pick_world(sensors_yaml: str | None):
    """pick_cube + arm 段可选注入 sensors（文本注入，与 Agent 改 YAML 同路径）。"""
    text = WORLD.read_text(encoding="utf-8")
    anchor = "    controller: ../controllers/pick_place.py   # 侧扫推式相位机"
    if sensors_yaml:
        text = text.replace(anchor, anchor + "\n" + sensors_yaml)
    spec = parse_world(yaml.safe_load(text))
    spec.base_dir = ROOT / "worlds"
    return spec


class TestTactileAndGuard:
    def test_tactile_fires_during_push(self):
        """pick_cube 推挤段指板触觉 > 0（T5）。

        接触审计实证：推 cube 的指板是 finger_r（finger_l 全程不碰 cube）。
        """
        sensors = """    sensors:
      - {name: fr_touch, type: tactile, body: finger_r, pos: [0, 0, 0]}"""
        spec = _pick_world(sensors)
        touch_max = 0.0
        with Simulation(spec) as sim:
            adr, dim = sim.index.robots["arm"].sensors["fr_touch"].parts[0]

            def grab():                       # on_tick：run 线程内 data 一致可读
                nonlocal touch_max
                v = float(sim.data.sensordata[adr:adr + dim][0])
                touch_max = max(touch_max, v)

            stats = sim.run(on_tick=grab)
        assert stats["success"] is True
        assert touch_max > 0.0, "推挤全程指板触觉从未触发"

    def test_sensors_do_not_change_trajectory(self):
        """bit 级守护（T6）：加全套传感器前后，pick_cube 轨迹逐位一致。"""
        results = []
        for sensors_yaml in (None, _SENSORS_ON_ARM):
            spec = _pick_world(sensors_yaml)
            with Simulation(spec) as sim:
                stats = sim.run()
                results.append((stats, sim.data.qpos.copy()))
        (s0, q0), (s1, q1) = results
        assert s0["ticks"] == s1["ticks"]
        assert s0["success"] == s1["success"]
        assert s0["reward"] == s1["reward"]
        # 严格逐位比较（allclose 会掩盖 warmstart 级漂移——项目硬约束）
        assert np.array_equal(q0, q1), \
            "加传感器改变了物理轨迹（传感器必须是只读派生量）"
