"""M6 P1：2FG7 夹爪 site-attach 到四台 menagerie 机械臂。

覆盖：
- 四臂（ur5e/ur10e/xarm7_nohand/panda_nohand）挂爪均编译成功
- 夹爪 actuator/joint/body 并入机器人设备索引（gripper/ 前缀）
- 物理：ctrl=close 后两 slide 实际闭合（qpos>0.015），open 回 0
- 触觉传感器可引用夹爪手指 body（先挂爪再注传感器的顺序保证）
- mount 未标定的机器人给清晰错误
"""
from __future__ import annotations

import glob

import mujoco
import numpy as np
import pytest

from agentsim.errors import WorldError
from agentsim.grippers import GRIPPERS, GRIPPER_PREFIX
from agentsim.world.builder import build_world
from agentsim.world.schema import parse_world


def _menagerie_xml(pattern: str) -> str:
    hits = glob.glob(
        f"/home/yuancheng-zhu/.cache/mujoco_menagerie/models/{pattern}")
    if not hits:
        pytest.skip(f"menagerie 模型未下载: {pattern}")
    return hits[0]


ARMS = [
    ("ur5e", "universal_robots_ur5e-*/ur5e.xml"),
    ("ur10e", "universal_robots_ur10e-*/ur10e.xml"),
    ("xarm7", "ufactory_xarm7-*/xarm7_nohand.xml"),
    ("panda", "franka_emika_panda-*/panda_nohand.xml"),
]


def _world_dict(arm_xml: str, gripper=True, sensors=()) -> dict:
    robot = {"name": "arm", "model": arm_xml, "pos": [0, 0, 0]}
    if gripper:
        robot["gripper"] = {"model": "onrobot_2fg7"}
    if sensors:
        robot["sensors"] = list(sensors)
    return {"name": "gripper_test",
            "timestep": 0.002, "ctrl_period": 8,
            "integrator": "implicitfast", "impratio": 100,
            "scene": {"floor": {"size": [2.0, 2.0]}},
            "robots": [robot]}


@pytest.mark.parametrize("label,pattern", ARMS)
def test_gripper_attaches_to_four_arms(label, pattern):
    """四臂挂 2FG7 均编译成功，且夹爪设备并入索引。"""
    xml = _menagerie_xml(pattern)
    spec = parse_world(_world_dict(xml))
    model, index = build_world(spec)
    ridx = index.robots["arm"]

    assert f"{GRIPPER_PREFIX}gripper_act" in ridx.actuators
    assert f"{GRIPPER_PREFIX}left_slide" in ridx.joints
    assert f"{GRIPPER_PREFIX}right_slide" in ridx.joints
    # body 全名（机器人前缀 + 夹爪前缀）
    assert f"arm/{GRIPPER_PREFIX}gripper_base" in index.bodies
    # tendon 跨嵌套 attach 存活
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON,
                            "arm/gripper/gripper_tendon")
    assert tid >= 0


@pytest.mark.parametrize("label,pattern", ARMS)
def test_gripper_actuator_count(label, pattern):
    """挂爪后执行器/关节数 = 裸臂 + 夹爪各 1 个 actuator / 2 个 slide。"""
    xml = _menagerie_xml(pattern)
    bare_spec = parse_world(_world_dict(xml, gripper=False))
    bare_model, _ = build_world(bare_spec)
    full_spec = parse_world(_world_dict(xml, gripper=True))
    full_model, _ = build_world(full_spec)
    assert full_model.nu == bare_model.nu + 1
    assert full_model.njnt == bare_model.njnt + 2


def test_gripper_closes_and_opens():
    """ur5e + 2FG7：ctrl=0.038 闭合（两 slide qpos>0.015），回 0 张开。"""
    xml = _menagerie_xml("universal_robots_ur5e-*/ur5e.xml")
    spec = parse_world(_world_dict(xml))
    model, index = build_world(spec)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ridx = index.robots["arm"]
    grip_act = ridx.actuators[f"{GRIPPER_PREFIX}gripper_act"]
    lid = ridx.joints[f"{GRIPPER_PREFIX}left_slide"]
    rid = ridx.joints[f"{GRIPPER_PREFIX}right_slide"]

    def slide_qpos():
        return (float(data.qpos[model.jnt_qposadr[lid]]),
                float(data.qpos[model.jnt_qposadr[rid]]))

    # 非夹爪执行器目标设为当前关节角（保持臂姿态，隔离夹爪运动）
    for short, aid in ridx.actuators.items():
        if not short.startswith(GRIPPER_PREFIX):
            jid = model.actuator_trnid[aid, 0]
            data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]

    l0, r0 = slide_qpos()
    assert abs(l0) < 1e-6 and abs(r0) < 1e-6     # 出生全开

    data.ctrl[grip_act] = GRIPPERS["onrobot_2fg7"]["close"]
    for _ in range(400):
        mujoco.mj_step(model, data)
    lc, rc = slide_qpos()
    assert lc > 0.015 and rc > 0.015, (lc, rc)    # 实际闭合到位

    data.ctrl[grip_act] = GRIPPERS["onrobot_2fg7"]["open"]
    for _ in range(400):
        mujoco.mj_step(model, data)
    lo, ro = slide_qpos()
    assert lo < 0.005 and ro < 0.005, (lo, ro)    # 重新张开


def test_tactile_sensor_on_gripper_finger():
    """触觉传感器可挂到夹爪手指 body（先挂爪、后注传感器）。"""
    xml = _menagerie_xml("universal_robots_ur5e-*/ur5e.xml")
    sensors = [{"name": "lf_touch", "type": "tactile",
                "body": f"{GRIPPER_PREFIX}left_finger", "radius": 0.03}]
    spec = parse_world(_world_dict(xml, sensors=sensors))
    model, index = build_world(spec)
    assert "lf_touch" in index.robots["arm"].sensors
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                            "arm/lf_touch")
    assert sid >= 0


def test_unknown_mount_robot_raises():
    """夹爪挂到未标定的机器人（且未写 pos/quat）→ 清晰 WorldError。"""
    from agentsim.grippers import mount_for
    with pytest.raises(WorldError, match="标定安装位姿"):
        mount_for("onrobot_2fg7", "/some/path/acme_arm.xml")
    # 显式覆盖 pos/quat 则放行（非标准臂手动指定）
    xf = mount_for("onrobot_2fg7", "/some/path/acme_arm.xml",
                   override_pos=[0, 0, 0.1],
                   override_quat=[1, 0, 0, 0])
    assert xf["pos"] == [0, 0, 0.1]


def test_no_attachment_site_raises():
    """机器人模型压根没有挂点 site（demo_arm）→ 错误列出可用 site。"""
    import pathlib
    demo_arm = pathlib.Path(__file__).resolve().parents[1] / "models" / "demo_arm.xml"
    spec = parse_world(_world_dict(str(demo_arm)))
    with pytest.raises(WorldError, match="没有挂点 site"):
        build_world(spec)


def test_missing_site_raises():
    """挂点 site 不在机器人模型里 → 错误列出可用 site。"""
    xml = _menagerie_xml("universal_robots_ur5e-*/ur5e.xml")
    d = _world_dict(xml)
    d["robots"][0]["gripper"]["site"] = "no_such_site"
    spec = parse_world(d)
    with pytest.raises(WorldError, match="没有挂点 site"):
        build_world(spec)


def test_manifest_reports_gripper_sites_and_catalog():
    """M6 P2：get_robot_manifest 返回夹爪信息、挂点清单与夹爪库目录。"""
    import yaml as _yaml
    from agentsim.mcp_server import AgentSimSession

    xml = _menagerie_xml("universal_robots_ur5e-*/ur5e.xml")
    wd = _world_dict(xml, gripper=True, sensors=[
        {"name": "pad_touch", "type": "tactile",
         "body": "gripper/left_finger", "radius": 0.02}])
    text = _yaml.safe_dump(wd, sort_keys=False)

    s = AgentSimSession()
    try:
        s.call_tool("load_world", {"yaml_text": text, "base_dir": "/tmp"})
        m = s.call_tool("get_robot_manifest", {})
        arm = m["robots"][0]
        # 夹爪信息
        assert arm["gripper"] is not None
        assert arm["gripper"]["model"] == "onrobot_2fg7"
        assert arm["gripper"]["info"]["close"] == GRIPPERS["onrobot_2fg7"]["close"]
        assert arm["gripper"]["info"]["actuator"] == "gripper/gripper_act"
        # 挂点含机器人自带 attachment_site 与夹爪内 site
        assert "attachment_site" in arm["sites"]
        assert "gripper/gripper_tcp" in arm["sites"]
        # 夹爪执行器在清单里
        assert "gripper/gripper_act" in {a["name"] for a in arm["actuators"]}
        # 触觉传感器出现在清单
        assert {x["name"] for x in arm["sensors"]} == {"pad_touch"}
        # 夹爪库目录
        cat = {g["key"]: g for g in m["gripper_catalog"]}
        assert "onrobot_2fg7" in cat
        assert "ur5e.xml" in cat["onrobot_2fg7"]["supported_robots"]
    finally:
        s.close()


def test_manifest_gripper_none_when_bare():
    """裸臂 manifest 的 gripper 为 None。"""
    import yaml as _yaml
    from agentsim.mcp_server import AgentSimSession

    xml = _menagerie_xml("universal_robots_ur5e-*/ur5e.xml")
    text = _yaml.safe_dump(_world_dict(xml, gripper=False), sort_keys=False)
    s = AgentSimSession()
    try:
        s.call_tool("load_world", {"yaml_text": text, "base_dir": "/tmp"})
        arm = s.call_tool("get_robot_manifest", {})["robots"][0]
        assert arm["gripper"] is None
        assert "attachment_site" in arm["sites"]
    finally:
        s.close()
