"""M6 P1：2FG7 夹爪 site-attach 到四台 menagerie 机械臂。

覆盖：
- 四臂（ur5e/ur10e/xarm7_nohand/panda_nohand）挂爪均编译成功
- 夹爪 actuator/joint/body 并入机器人设备索引（gripper/ 前缀）
- 物理：ctrl=close 后两 slide 实际闭合（qpos>0.015），open 回 0
- 触觉传感器可引用夹爪手指 body（先挂爪再注传感器的顺序保证）
- mount 未标定的机器人给清晰错误
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from agentsim.errors import WorldError
from agentsim.grippers import GRIPPERS, GRIPPER_PREFIX
from agentsim.world.builder import build_world
from agentsim.world.schema import parse_world
from agentsim.zoo import ensure_model


def _menagerie_xml(key: str) -> str:
    """模型库取臂 XML（ensure_model 跨平台解析缓存目录，Windows 同样可用）。"""
    try:
        return str(ensure_model(key))
    except Exception as e:                       # 未下载/无网络等
        pytest.skip(f"menagerie 模型不可用: {e}")


ARMS = [
    ("ur5e", "universal_robots_ur5e"),
    ("ur10e", "universal_robots_ur10e"),
    ("xarm7", "ufactory_xarm7"),
    ("panda", "franka_emika_panda"),
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


@pytest.mark.parametrize("label,key", ARMS)
def test_gripper_attaches_to_four_arms(label, key):
    """四臂挂 2FG7 均编译成功，且夹爪设备并入索引。"""
    xml = _menagerie_xml(key)
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


@pytest.mark.parametrize("label,key", ARMS)
def test_gripper_actuator_count(label, key):
    """挂爪后执行器/关节数 = 裸臂 + 夹爪各 1 个 actuator / 2 个 slide。"""
    xml = _menagerie_xml(key)
    bare_spec = parse_world(_world_dict(xml, gripper=False))
    bare_model, _ = build_world(bare_spec)
    full_spec = parse_world(_world_dict(xml, gripper=True))
    full_model, _ = build_world(full_spec)
    assert full_model.nu == bare_model.nu + 1
    assert full_model.njnt == bare_model.njnt + 2


def test_gripper_closes_and_opens():
    """ur5e + 2FG7：ctrl=0.038 闭合（两 slide qpos>0.015），回 0 张开。"""
    xml = _menagerie_xml("universal_robots_ur5e")
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
    xml = _menagerie_xml("universal_robots_ur5e")
    sensors = [{"name": "lf_touch", "type": "tactile",
                "body": f"{GRIPPER_PREFIX}left_finger", "radius": 0.03}]
    spec = parse_world(_world_dict(xml, sensors=sensors))
    model, index = build_world(spec)
    assert "lf_touch" in index.robots["arm"].sensors
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR,
                            "arm/lf_touch")
    assert sid >= 0


@pytest.mark.parametrize("label,key", ARMS)
def test_gripper_mount_flushes_on_flange(label, key):
    """2FG7 后法兰面与机器人法兰贴合（无悬浮间隙、无明显穿插）。

    历史回归：旧 mount 误用 Rot_x(180°) 且 z=0.1494/y=-0.028，夹爪被转到
    site ±y 方向垂在腕部壳体下方，与法兰留约 9cm 间隙。这里把"贴合"固化为
    几何断言：夹爪 base 后端面与腕部法兰面在法兰半径内轴向间距≈0。
    """
    xml = _menagerie_xml(key)
    spec = parse_world(_world_dict(xml))
    model, _ = build_world(spec)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                            "arm/attachment_site")
    sp = data.site_xpos[sid]
    sq = data.site_xmat[sid].reshape(3, 3)

    def to_site(points):
        return (points - sp) @ sq

    def geom_points(gid, n=30000):
        """geom 表面点集（mesh 取真实顶点；圆柱/胶囊实心采样），世界系。"""
        rng = np.random.default_rng(0)
        mid = model.geom_dataid[gid]
        if mid >= 0:
            v = model.mesh_vert[
                model.mesh_vertadr[mid]:
                model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            return (data.geom_xmat[gid].reshape(3, 3) @ v.T).T \
                + data.geom_xpos[gid]
        ty = int(model.geom_type[gid])
        sz = model.geom_size[gid]
        a = rng.uniform(0, 2 * np.pi, n)
        if ty == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            z = rng.uniform(-sz[1], sz[1], n)
            q = np.c_[sz[0] * np.cos(a), sz[0] * np.sin(a), z]
        elif ty == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            r, hh = sz[0], sz[1]
            z = rng.uniform(-(hh + r), hh + r, n)
            t = np.clip(np.abs(z) - hh, 0, r)
            rad = np.sqrt(np.clip(r * r - t * t, 0, None))
            q = np.c_[rad * np.cos(a), rad * np.sin(a), z]
        else:
            return None
        return (data.geom_xmat[gid].reshape(3, 3) @ q.T).T \
            + data.geom_xpos[gid]

    # 腕部：site 所在 body 及其父 body（panda 挂点是空 body，法兰在 link7）
    wrist_bodies = {model.site_bodyid[sid],
                    model.body_parentid[model.site_bodyid[sid]]}
    wrist_z, grip_z = [], []
    for gid in range(model.ngeom):
        bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               model.geom_bodyid[gid]) or ""
        pts = geom_points(gid)
        if pts is None:
            continue
        s = to_site(pts)
        r = np.hypot(s[:, 0], s[:, 1])
        if model.geom_bodyid[gid] in wrist_bodies:
            wrist_z.append(s[r < 0.030, 2])
        if "/gripper/gripper_base" in bn:
            grip_z.append(s[r < 0.030, 2])
    wrist_front = np.max(np.concatenate(wrist_z))
    grip_back = np.min(np.concatenate(grip_z))
    gap = grip_back - wrist_front
    assert -0.003 <= gap <= 0.004, (
        f"{label}: 夹爪后端面与法兰面轴向间距 {gap:.4f}m，应≈0")


@pytest.mark.parametrize("label,key", ARMS)
def test_gripper_fingers_point_along_flange_axis(label, key):
    """指尖沿挂点 +z（法兰法向）伸出、与法兰同轴、开合沿 site x。

    旧 Rot_x(180°) 标定下指尖沿 site ±y（UR home 时竖直向下），是本 bug 的
    朝向特征；该断言确保不再回退。
    """
    xml = _menagerie_xml(key)
    spec = parse_world(_world_dict(xml))
    model, _ = build_world(spec)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                            "arm/attachment_site")
    sp = data.site_xpos[sid]
    sq = data.site_xmat[sid].reshape(3, 3)

    def site_pos(site_name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return (data.site_xpos[i] - sp) @ sq

    lp = site_pos("arm/gripper/left_pad_site")
    rp = site_pos("arm/gripper/right_pad_site")
    mid_xy = (lp + rp) / 2
    # 指尖朝 site +z 伸出（远离法兰 >10cm），而非沿 ±y 垂下
    assert lp[2] > 0.10 and rp[2] > 0.10, (label, lp, rp)
    assert abs(mid_xy[0]) < 0.005 and abs(mid_xy[1]) < 0.005, (label, mid_xy)
    # 两指开合沿 site x（分离约 25mm），z 同高
    assert abs(lp[0] - rp[0]) > 0.020, (label, lp, rp)
    assert abs(lp[2] - rp[2]) < 0.005 and abs(lp[1] - rp[1]) < 0.005


def test_mount_table_per_arm_values():
    """mount 表逐臂标定值黄金断言（防回退到错误的共用值）。"""
    from agentsim.grippers import mount_for
    assert mount_for("onrobot_2fg7", "x/ur5e.xml")["pos"] == [0, 0, 0.1094]
    assert mount_for("onrobot_2fg7", "x/ur10e.xml")["pos"] == [0, 0, 0.1264]
    assert mount_for("onrobot_2fg7",
                     "x/xarm7_nohand.xml")["pos"] == [0, 0, 0.1094]
    assert mount_for("onrobot_2fg7",
                     "x/panda_nohand.xml")["pos"] == [0, 0, 0.1092]
    # 朝向统一 Rot_x(-90°)，wxyz
    for arm in ("ur5e.xml", "ur10e.xml", "xarm7_nohand.xml",
                "panda_nohand.xml"):
        q = mount_for("onrobot_2fg7", f"x/{arm}")["quat"]
        assert abs(q[0] - 0.7071) < 1e-3 and abs(q[1] + 0.7071) < 1e-3
        assert q[2] == 0 and q[3] == 0


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
    xml = _menagerie_xml("universal_robots_ur5e")
    d = _world_dict(xml)
    d["robots"][0]["gripper"]["site"] = "no_such_site"
    spec = parse_world(d)
    with pytest.raises(WorldError, match="没有挂点 site"):
        build_world(spec)


def test_manifest_reports_gripper_sites_and_catalog():
    """M6 P2：get_robot_manifest 返回夹爪信息、挂点清单与夹爪库目录。"""
    import yaml as _yaml
    from agentsim.mcp_server import AgentSimSession

    xml = _menagerie_xml("universal_robots_ur5e")
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

    xml = _menagerie_xml("universal_robots_ur5e")
    text = _yaml.safe_dump(_world_dict(xml, gripper=False), sort_keys=False)
    s = AgentSimSession()
    try:
        s.call_tool("load_world", {"yaml_text": text, "base_dir": "/tmp"})
        arm = s.call_tool("get_robot_manifest", {})["robots"][0]
        assert arm["gripper"] is None
        assert "attachment_site" in arm["sites"]
    finally:
        s.close()
