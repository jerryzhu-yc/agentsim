"""URDF 导入测试：SolidWorks（sw2urdf）工作流 + 通用 URDF。

覆盖（均在 mujoco 3.15 实验确证的行为上）：
- URDF 无 <actuator> -> 每个可动关节自动补同名 motor
- mesh 相对路径按 urdf 所在目录解析（from_string 无目录上下文）
- package://NAME/REST 前缀剥除 + 两级就近查找
- fixed joint 在 URDF 解析时已被合并（joints 里不出现）
- 解析失败给出可读的 WorldError（含 collision-only 提示）
"""
import pathlib
import struct

import mujoco
import pytest
import yaml

from agentsim.errors import WorldError
from agentsim.kernel.sim import Simulation
from agentsim.world.builder import _load_robot_spec, build_world
from agentsim.world.schema import load_world, parse_world

# 两连杆铰接臂：base(-z 轴 box，最低点 -0.02) + j1 revolute + arm
_URDF = """<?xml version="1.0"?>
<robot name="mybot">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual><geometry><box size="0.1 0.1 0.04"/></geometry></visual>
    <collision><geometry><box size="0.1 0.1 0.04"/></geometry></collision>
  </link>
  <link name="arm">
    <inertial>
      <origin xyz="0.1 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.001"/>
    </inertial>
    <visual><origin xyz="0.1 0 0"/><geometry><box size="0.2 0.03 0.03"/></geometry></visual>
    <collision><origin xyz="0.1 0 0"/><geometry><box size="0.2 0.03 0.03"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <origin xyz="0 0 0.03" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="2"/>
  </joint>
</robot>
"""


def _stl(path: pathlib.Path) -> None:
    """程序生成单三角形二进制 STL（免依赖外部 mesh 文件）。"""
    tri = struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 0.1, 0, 0, 0.1, 0, 0.1, 0)
    path.write_bytes(b"\0" * 80 + struct.pack("<I", 1) + tri)


def _world_yaml(model: str, pos=(0, 0, 0.02)) -> str:
    return yaml.safe_dump({
        "name": "urdf_world",
        "floor": {},
        "scene": {"objects": []},
        "robots": [{"name": "mybot", "model": model, "pos": list(pos)}],
    }, allow_unicode=True, sort_keys=False)


def test_urdf_motor_autocreate(tmp_path):
    """URDF 无执行器 -> 每个可动关节自动补同名 motor（j1 一个）。"""
    p = tmp_path / "mybot.urdf"
    p.write_text(_URDF, encoding="utf-8")
    spec = _load_robot_spec(p)
    assert [a.name for a in spec.actuators] == ["j1"]
    # motor 都挂在对应关节上
    j = {j.name: j for j in spec.joints}
    assert "j1" in j                       # fixed joint（若有）已被解析器合并
    assert spec.actuators[0].target == "j1"


def test_urdf_build_and_sim(tmp_path):
    """URDF 机器人走完整链：build_world -> 设备索引 -> 冒烟 50 tick。"""
    p = tmp_path / "mybot.urdf"
    p.write_text(_URDF, encoding="utf-8")
    wf = tmp_path / "w.yaml"
    wf.write_text(_world_yaml(str(p)), encoding="utf-8")
    spec = load_world(wf)
    _model, index = build_world(spec)
    ridx = index.robots["mybot"]
    assert sorted(ridx.actuators) == ["j1"]
    assert sorted(ridx.joints) == ["j1"]
    with Simulation(spec) as sim:
        stats = sim.run(max_ticks=50)
    assert stats["ticks"] == 50
    assert not stats["nan"]
    assert stats["crashed"] == []


def test_urdf_mesh_relative_and_package(tmp_path):
    """mesh 相对路径按 urdf 目录解析；package:// 前缀剥除后两级查找。"""
    _stl(tmp_path / "meshes_test.stl")
    # 相对路径（同目录）
    rel = tmp_path / "rel.urdf"
    rel.write_text(_URDF.replace(
        '<visual><geometry><box size="0.1 0.1 0.04"/></geometry></visual>\n'
        '    <collision><geometry><box size="0.1 0.1 0.04"/></geometry></collision>',
        '<collision><geometry><mesh filename="meshes_test.stl" '
        'scale="1 1 1"/></geometry></collision>'), encoding="utf-8")
    spec = _load_robot_spec(rel)
    assert any(g.type == mujoco.mjtGeom.mjGEOM_MESH for g in spec.geoms)

    # package://NAME/REST：先按 REST 找再按 NAME/REST 找
    pkg = tmp_path / "pkg.urdf"
    pkg.write_text(_URDF.replace(
        '<visual><geometry><box size="0.1 0.1 0.04"/></geometry></visual>\n'
        '    <collision><geometry><box size="0.1 0.1 0.04"/></geometry></collision>',
        '<collision><geometry><mesh '
        'filename="package://mybot/meshes_test.stl" '
        'scale="1 1 1"/></geometry></collision>'), encoding="utf-8")
    spec = _load_robot_spec(pkg)
    assert any(g.type == mujoco.mjtGeom.mjGEOM_MESH for g in spec.geoms)


def test_urdf_bad_file_readable_error(tmp_path):
    """坏 URDF -> WorldError 且提示 MuJoCo 只读 <collision>。"""
    p = tmp_path / "bad.urdf"
    p.write_text("<robot name='x'><link name='l'></robot>", encoding="utf-8")
    with pytest.raises(WorldError, match="URDF 解析失败"):
        _load_robot_spec(p)


def test_urdf_missing_mesh_reported(tmp_path):
    """mesh 文件不存在 -> build_world 报错里带原文件名（懒加载到编译才触发）。"""
    p = tmp_path / "nope.urdf"
    p.write_text(_URDF.replace(
        '<visual><geometry><box size="0.1 0.1 0.04"/></geometry></visual>\n'
        '    <collision><geometry><box size="0.1 0.1 0.04"/></geometry></collision>',
        '<collision><geometry><mesh filename="ghost.stl" '
        'scale="1 1 1"/></geometry></collision>'), encoding="utf-8")
    wf = tmp_path / "w.yaml"
    wf.write_text(_world_yaml(str(p)), encoding="utf-8")
    spec = load_world(wf)
    with pytest.raises(WorldError, match="ghost.stl"):
        build_world(spec)


def test_mjcf_still_direct_load(tmp_path):
    """MJCF 不做任何改写：from_file 原样读（保留既有 xml 导入行为）。"""
    xml = (tmp_path / "bot.xml")
    xml.write_text(
        "<mujoco><worldbody>"
        '<body name="b" pos="0 0 0.1"><geom type="box" size=".1 .1 .02"/>'
        '<joint name="j" type="slide" axis="0 0 1"/>'
        "</body></worldbody>"
        '<actuator><motor name="m" joint="j"/></actuator></mujoco>',
        encoding="utf-8")
    spec = _load_robot_spec(xml)
    assert [a.name for a in spec.actuators] == ["m"]  # 不额外补 motor


def test_world_yaml_with_urdf_robot(tmp_path):
    """世界 YAML 的 robots[].model 指向 .urdf 直接可用（GUI 导入链路）。"""
    p = tmp_path / "mybot.urdf"
    p.write_text(_URDF, encoding="utf-8")
    spec = parse_world(yaml.safe_load(_world_yaml(str(p))))
    assert spec.robots[0].model == str(p)


# ================================================================ Windows 适配

def _mesh_urdf(filename: str) -> str:
    """base link 只有 mesh collision 的 URDF（懒加载：文件不存在也不报错）。"""
    return _URDF.replace(
        '<visual><geometry><box size="0.1 0.1 0.04"/></geometry></visual>\n'
        '    <collision><geometry><box size="0.1 0.1 0.04"/></geometry></collision>',
        f'<collision><geometry><mesh filename="{filename}" '
        f'scale="1 1 1"/></geometry></collision>')


def test_windows_drive_letter_path_kept(tmp_path):
    """盘符绝对路径（C:/x、C:\\x）不与 urdf 目录拼接，原样保留。"""
    for fn in ["C:/meshes/x.stl", "C:\\meshes\\x.stl"]:
        p = tmp_path / "t.urdf"
        p.write_text(_mesh_urdf(fn), encoding="utf-8")
        spec = _load_robot_spec(p)
        assert spec.meshes[0].file == "C:/meshes/x.stl", spec.meshes[0].file


def test_windows_posix_slash_resolution(tmp_path):
    """解析成功的 mesh 路径统一正斜杠（as_posix，Windows 反斜杠安全）。"""
    _stl(tmp_path / "meshes_test.stl")
    p = tmp_path / "t.urdf"
    p.write_text(_mesh_urdf("meshes_test.stl"), encoding="utf-8")
    spec = _load_robot_spec(p)
    assert spec.meshes[0].file == (tmp_path / "meshes_test.stl").as_posix()
    assert "\\" not in spec.meshes[0].file


def test_gbk_encoded_urdf_readable(tmp_path):
    """sw2urdf 在中文 Windows 可能写出 GBK：utf-8 失败自动回退 GBK。"""
    p = tmp_path / "gbk.urdf"
    p.write_bytes(_mesh_urdf("meshes_test.stl")
                  .replace('name="mybot"', 'name="我的机器人"')
                  .encode("gbk"))
    spec = _load_robot_spec(p)               # 不抛 UnicodeDecodeError 即可
    assert [a.name for a in spec.actuators] == ["j1"]
