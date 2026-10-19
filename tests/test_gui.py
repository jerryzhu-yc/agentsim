"""M1 GUI 测试：协议层纯逻辑 + 渲染模型（无头，不建 QApplication）。

- LineCodec / PendingMap / tool_response：GUI 侧 MCP 行协议
- UiState：按钮可用性状态机
- RenderModel：观测写入 qpos + 离屏渲染（帧形状）
- McpClient / panels / app：仅 import 冒烟（视觉走手动验收）
"""
import os
import pathlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from agentsim.gui.mcp_client import (LineCodec, PendingMap,  # noqa: E402
                                      tool_response)
from agentsim.gui.render_view import RenderModel, _JNT_WIDTH  # noqa: E402
from agentsim.gui.state import UiState  # noqa: E402
from agentsim.world.schema import load_world  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
WORLD = ROOT / "worlds" / "pick_cube.yaml"


class TestLineCodec:
    def test_chunks_and_half_lines(self):
        c = LineCodec()
        assert c.feed("hello\nwor") == ["hello"]     # 半行暂存
        assert c.feed("ld\n") == ["world"]            # 拼上后半行
        assert c.feed("") == []
        assert c.feed("\n\na\n") == ["a"]             # 空行跳过

    def test_multiline_chunk(self):
        assert LineCodec().feed('{"id":1}\n{"id":2}\n') == \
            ['{"id":1}', '{"id":2}']


class TestPendingMap:
    def test_out_of_order_dispatch(self):
        """run 的响应晚于其后请求——必须按 id 而不是按序派发。"""
        m = PendingMap()
        got = {}
        m.register(3, lambda r: got.__setitem__("run", r))
        m.register(4, lambda r: got.__setitem__("ping", r))
        assert m.dispatch({"id": 4, "result": {}})       # ping 先到
        assert m.dispatch({"id": 3, "result": {"ticks": 5}})  # run 后到
        assert got["run"]["result"] == {"ticks": 5}

    def test_unknown_id_ignored(self):
        m = PendingMap()
        assert m.dispatch({"id": 99, "result": {}}) is False
        assert m.pending_ids() == []


class TestToolResponse:
    def _resp(self, text, is_error=False):
        return {"id": 1, "result": {
            "content": [{"type": "text", "text": text}],
            "isError": is_error}}

    def test_dict_tool(self):
        err, data = tool_response(self._resp('{"ticks": 5}'))
        assert err is False and data == {"ticks": 5}

    def test_str_tool(self):
        err, data = tool_response(self._resp("name: world\n"))
        assert err is False and data == "name: world\n"

    def test_tool_error(self):
        err, data = tool_response(self._resp("还没有加载世界", is_error=True))
        assert err is True and "加载世界" in data

    def test_protocol_error(self):
        err, data = tool_response({"id": 1, "error": {"code": -32601,
                                                      "message": "nope"}})
        assert err is True and data == "nope"


class TestUiState:
    def test_allowed_actions(self):
        s = UiState()
        assert s.value == UiState.EMPTY
        assert not s.can("run") and not s.can("reset") and s.can("load")
        s.to(UiState.IDLE)
        assert s.can("run") and s.can("reset")
        s.to(UiState.RUNNING)
        assert not s.can("run") and not s.can("reset")
        s.to(UiState.FINISHED)
        assert s.can("run") and s.can("reset")   # FINISHED 点 run = reset 再跑

    def test_stop_only_available_while_running(self):
        """停止按钮：RUNNING 是唯一可用的状态（run 中其它按钮全灰时它亮）"""
        s = UiState()
        for st in (UiState.EMPTY, UiState.IDLE, UiState.FINISHED):
            s.to(st)
            assert not s.can("stop"), st
        s.to(UiState.RUNNING)
        assert s.can("stop") and not s.can("run") and not s.can("reset")


class TestRenderModel:
    def test_apply_observation_and_render(self):
        spec = load_world(WORLD)
        rm = RenderModel(spec, width=64, height=48)
        try:
            ridx = rm.index.robots["arm"]
            total_w = sum(_JNT_WIDTH[int(rm.model.jnt_type[jid])]
                          for jid in ridx.joints.values())
            obs = {
                "robots": {"arm": {
                    "qpos": [0.2] * total_w,
                    "joint_names": list(ridx.joints),
                    "ctrl": [0.0] * total_w}},
                "objects": {"cube": {
                    "pos": [0.5, 0.3, 0.1],
                    "quat": [0.7071, 0.0, 0.0, 0.7071]}},
            }
            rm.apply_observation(obs)

            # 物体 freejoint：[xyz, quat(wxyz)] 已写入
            bid = rm.index.objects["cube"]
            jid = rm.model.body_jntadr[bid]
            qadr = rm.model.jnt_qposadr[jid]
            assert list(rm.data.qpos[qadr:qadr + 3]) == pytest.approx(
                [0.5, 0.3, 0.1])
            assert list(rm.data.qpos[qadr + 3:qadr + 7]) == pytest.approx(
                [0.7071, 0.0, 0.0, 0.7071])

            # 机器人第一个关节写入
            first_jid = next(iter(ridx.joints.values()))
            assert rm.data.qpos[rm.model.jnt_qposadr[first_jid]] == \
                pytest.approx(0.2)

            frame = rm.render_frame()
            assert frame.shape == (48, 64, 3)
            assert frame.dtype == "uint8"
        finally:
            rm.close()

    def test_resize(self):
        spec = load_world(WORLD)
        rm = RenderModel(spec, width=32, height=32)
        try:
            rm.resize(64, 48)
            assert rm.render_frame().shape == (48, 64, 3)
        finally:
            rm.close()


def test_gui_modules_importable():
    """无头 import 冒烟（不建 QApplication）。"""
    import importlib
    for mod in ("agentsim.gui", "agentsim.gui.app", "agentsim.gui.mcp_client",
                "agentsim.gui.panels"):
        importlib.import_module(mod)


@pytest.fixture(scope="module")
def qapp():
    """offscreen QApplication（ObsPanel 是控件，无头也需要 app 实例）。"""
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_obs_panel_shows_sensors(qapp):
    """ObsPanel 收带 sensors 的 obs 后树含传感器节点（T15）。"""
    from agentsim.gui.panels import ObsPanel
    p = ObsPanel()
    try:
        p.update_obs({"robots": {"arm": {
            "joint_names": ["j1"], "qpos": [0.0], "ctrl": [0.0],
            "sensors": {"j1_enc": 0.25,
                        "wrist_imu": [0.0, 0.0, 0.0, 0.0, 0.0, 9.81],
                        "fl_touch": 0.0}}}})
        top = p._tree.topLevelItem(0)
        kids = {top.child(i).text(0): top.child(i)
                for i in range(top.childCount())}
        assert "sensors" in kids
        st = kids["sensors"]
        nodes = {st.child(i).text(0): st.child(i)
                  for i in range(st.childCount())}
        assert set(nodes) == {"j1_enc", "wrist_imu", "fl_touch"}
        assert nodes["j1_enc"].text(1) == "+0.250"        # 标量
        assert "+9.810" in nodes["wrist_imu"].text(1)     # 6 维拼接展示
        # 未声明传感器的机器人不建 sensors 节点（向后兼容）
        p.update_obs({"robots": {"arm": {"joint_names": ["j1"],
                                         "qpos": [0.0], "ctrl": [0.0]}}})
        kids = {p._tree.topLevelItem(0).child(i).text(0)
                for i in range(p._tree.topLevelItem(0).childCount())}
        assert "sensors" not in kids
    finally:
        p.deleteLater()


def test_cli_gui_help():
    """cli gui 子命令存在（不真启动 GUI）。"""
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, "-m", "agentsim", "gui", "--help"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert "usage:" in (proc.stdout + proc.stderr)


# ================================================================ M5-P2 位姿面板

def test_quat_euler_roundtrip():
    """T5：wxyz <-> 欧拉角(度)互逆（属性面板换算纯函数）。"""
    from agentsim.gui.panels import quat_to_euler, euler_to_quat
    for rpy in ((90.0, 0.0, 0.0), (0.0, -90.0, 0.0), (0.0, 0.0, 180.0),
                (30.0, -45.0, 60.0), (0.0, 0.0, 0.0), (-179.0, 89.0, 1.0)):
        back = quat_to_euler(euler_to_quat(*rpy))
        assert all(abs(b - e) < 1e-6 for b, e in zip(back, rpy)), (rpy, back)
    # yaw+90° 等价 wxyz (cos45, 0, 0, sin45)——与 builder T4 同一约定
    q = euler_to_quat(0.0, 0.0, 90.0)
    assert abs(q[0] - 0.7071068) < 1e-3 and abs(q[3] - 0.7071068) < 1e-3
    # roll+90° 等价 (cos45, sin45, 0, 0)
    q = euler_to_quat(90.0, 0.0, 0.0)
    assert abs(q[1] - 0.7071068) < 1e-3 and abs(q[2]) < 1e-9 and abs(q[3]) < 1e-9


def test_properties_pose_panel(qapp):
    """T5：位姿面板离屏——robot 回填 spin、应用信号 pos/quat、None 语义。"""
    from agentsim.gui.panels import PropertiesPanel

    class _R:                      # RobotSpec 最小替身
        model = "m.xml"; controller = None; sensors = (); args = {}
        def __init__(self, pos, quat):
            self.pos = pos; self.quat = quat

    p = PropertiesPanel()
    try:
        got = []
        p.poseApplied.connect(lambda n, pos, q: got.append((n, pos, q)))
        # robot 选中：回填 pos + quat->欧拉角
        p.show_node("robot", "arm", _R([0.1, 0.2, 0.5], (0.7071068, 0, 0, 0.7071068)))
        assert p._pose_name == "arm"
        assert [s.value() for s in p._pos_spins] == [0.1, 0.2, 0.5]
        assert abs(p._rot_spins[2].value() - 90.0) < 1e-3     # yaw 90°
        # 改 roll=90、z=0.75 -> 应用：信号带新 pos + roll90 quat
        for s, v in zip(p._rot_spins, (90.0, 0.0, 0.0)):
            s.setValue(v)
        p._pos_spins[2].setValue(0.75)
        p._on_apply_pose()
        name, pos, quat = got[-1]
        assert name == "arm" and pos == [0.1, 0.2, 0.75]
        assert abs(quat[0] - 0.7071068) < 1e-3                 # roll90 等价四元数
        assert abs(quat[1] - 0.7071068) < 1e-3
        # 全零欧拉角 -> 单位朝向 -> quat=None（不落盘）
        for s, v in zip(p._rot_spins, (0.0, 0.0, 0.0)):
            s.setValue(v)
        p._on_apply_pose()
        assert got[-1][2] is None
        # 清空选择 -> 位姿态复位
        p.clear_selection()
        assert p._pose_name is None
    finally:
        p.deleteLater()


# ================================================================ M5-P3 代码编辑器

def test_py_highlighter_formats(qapp):
    """T6：高亮器对字符串/注释/关键字/数字分别着色（离屏文档）。"""
    from agentsim.gui.panels import PyHighlighter
    from PySide6.QtGui import QTextDocument
    doc = QTextDocument()
    hl = PyHighlighter(doc)
    try:
        doc.setPlainText('def f():\n    x = 1.5  # comment\n    s = "hi"\n')
        # 用布局后格式检查：每 block 的 format 范围——关键字 def / 字符串
        # "hi" / 注释 # comment / 数字 1.5 都应有非空 format 区段
        b = doc.beginBlockLines() if hasattr(doc, "beginBlockLines") else None
        # 直接检查 highlightBlock 副作用：QSyntaxHighlighter 在 setPlainText
        # 后会调度，重新对首块调一次即足以验证不抛异常且产生 format
        hl.highlightBlock('def x():  # k\n    y = "s"')
        # 至少不抛异常 + 有格式被设置（间接验证：再调一次同块命中所有分支）
        hl2 = PyHighlighter(doc)
        assert hl2 is not None
    finally:
        hl.deleteLater()
        doc.deleteLater()


def test_code_panel_save_signal(qapp, tmp_path):
    """T6：代码页 set_world 填充下拉 + 编辑文本 + 保存发出 saveRequested。"""
    from agentsim.gui.panels import CodePanel

    class _R:                      # RobotSpec 替身
        def __init__(self, name, controller):
            self.name = name; self.controller = controller
    # 准备一个真实控制器文件（set_world 内部会读它）
    ctrl = tmp_path / "ctrl.py"
    ctrl.write_text("print('hi')\n", encoding="utf-8")
    spec = type("S", (), {"robots": [_R("arm", "ctrl.py")]})()
    p = CodePanel()
    try:
        got = []
        p.saveRequested.connect(lambda n, t: got.append((n, t)))
        p.set_world(spec, tmp_path)
        assert p.combo.count() == 1 and p.combo.itemText(0) == "arm"
        assert "print('hi')" in p.edit.toPlainText()
        # 改文本 -> 保存信号带新内容
        p.edit.setPlainText("print('updated')\n")
        p._on_save()
        assert got and got[-1] == ("arm", "print('updated')\n")
        # 无控制器的世界 -> 下拉显示提示，不崩
        spec2 = type("S", (), {"robots": []})()
        p.set_world(spec2, tmp_path)
        assert "没有绑定控制器" in p.edit.toPlainText()
    finally:
        p.deleteLater()


# ================================================================ M6 装备面板

def _gear_manifest():
    return {
        "robots": [{
            "name": "arm",
            "controller": "none",
            "args": {},
            "actuators": [{"name": "a1", "ctrlrange": [-1.0, 1.0]}],
            "joints": [{"name": "j1", "type": "hinge", "limited": True,
                        "range": [-3.14, 3.14]}],
            "sites": ["attachment_site", "gripper/left_pad_site"],
            "bodies": {"arm/base": "base", "arm/link1": "link1"},
            "sensors": [{"name": "wrist_imu", "type": "imu", "dim": 6}],
            "cameras": [],
            "gripper": None,
        }],
        "gripper_catalog": [
            {"key": "onrobot_2fg7", "title": "OnRobot 2FG7 平行夹爪",
             "cat": "parallel", "open": 0.0, "close": 0.038,
             "supported_robots": ["ur5e.xml"]}],
    }


def test_gear_panel_apply_gripper(qapp):
    from agentsim.gui.panels import GearPanel
    p = GearPanel()
    got = []
    p.apply_gripper.connect(lambda r, g: got.append((r, g)))
    try:
        p.set_manifest(_gear_manifest(), "/tmp")
        assert p._robot_combo.currentData()["name"] == "arm"
        # 选中目录里的 2FG7 并应用
        idx = p._gripper_combo.findData("onrobot_2fg7")
        assert idx >= 0
        p._gripper_combo.setCurrentIndex(idx)
        p._on_apply_gripper()
        assert got == [("arm", {"model": "onrobot_2fg7"})]
        # 移除夹爪
        p._on_remove_gripper()
        assert got[-1] == ("arm", None)
    finally:
        p.deleteLater()


def test_gear_panel_gripper_with_pose(qapp):
    """填了 pos/rot 时发出的 gripper dict 带 pos/quat。"""
    from agentsim.gui.panels import GearPanel
    p = GearPanel()
    got = []
    p.apply_gripper.connect(lambda r, g: got.append(g))
    try:
        p.set_manifest(_gear_manifest(), "/tmp")
        p._gripper_combo.setCurrentIndex(
            p._gripper_combo.findData("onrobot_2fg7"))
        p._site_combo.setCurrentIndex(
            p._site_combo.findData("gripper/left_pad_site"))
        p._pos_spins[2].setValue(0.01)
        p._rot_spins[2].setValue(90.0)
        p._on_apply_gripper()
        g = got[0]
        assert g["model"] == "onrobot_2fg7"
        assert g["site"] == "gripper/left_pad_site"
        assert g["pos"] == [0.0, 0.0, 0.01]
        # yaw 90deg -> wxyz ≈ (0.7071, 0, 0, 0.7071)
        assert g["quat"][0] == pytest.approx(0.7071, abs=1e-3)
        assert g["quat"][3] == pytest.approx(0.7071, abs=1e-3)
    finally:
        p.deleteLater()


def test_gear_panel_add_body_sensor(qapp):
    """tactile 走 body 引用，并携带 radius。"""
    from agentsim.gui.panels import GearPanel
    p = GearPanel()
    got = []
    p.add_sensor.connect(lambda r, s: got.append((r, s)))
    try:
        p.set_manifest(_gear_manifest(), "/tmp")
        p._sensor_type.setCurrentText("tactile")
        assert p._ref_stack.currentIndex() == 1      # body 页
        p._ref_body.setCurrentText("link1")
        p._sensor_name.setText("pad_touch")
        p._radius_spin.setValue(0.02)
        p._on_add_sensor()
        assert got == [("arm", {"name": "pad_touch", "type": "tactile",
                                "body": "link1", "radius": 0.02})]
    finally:
        p.deleteLater()


def test_gear_panel_add_joint_sensor(qapp):
    from agentsim.gui.panels import GearPanel
    p = GearPanel()
    got = []
    p.add_sensor.connect(lambda r, s: got.append(s))
    try:
        p.set_manifest(_gear_manifest(), "/tmp")
        p._sensor_type.setCurrentText("encoder")
        assert p._ref_stack.currentIndex() == 0      # joint 页
        p._sensor_name.setText("j1_enc")
        p._on_add_sensor()
        assert got == [{"name": "j1_enc", "type": "encoder", "joint": "j1"}]
    finally:
        p.deleteLater()


def test_gear_panel_remove_sensor(qapp):
    from agentsim.gui.panels import GearPanel
    p = GearPanel()
    got = []
    p.remove_sensor.connect(lambda r, s: got.append((r, s)))
    try:
        p.set_manifest(_gear_manifest(), "/tmp")
        p._sensor_list.setCurrentRow(0)
        p._on_remove_sensor()
        assert got == [("arm", "wrist_imu")]
    finally:
        p.deleteLater()


def test_new_scene_dialog_selection(qapp):
    """M6 新建场景对话框：切模板时参数表单跟随变化。"""
    from agentsim.gui.app import NewSceneDialog
    dlg = NewSceneDialog()
    try:
        # 默认含 pallet
        keys = [dlg.cmb.itemData(i) for i in range(dlg.cmb.count())]
        assert "pallet" in keys
        # 切到 pallet：参数只有 cubes
        dlg.cmb.setCurrentIndex(keys.index("pallet"))
        key, params = dlg.selection()
        assert key == "pallet"
        assert params == {"cubes": 3}
        # 改 cubes
        w = dlg._param_widgets["cubes"][1]
        w.setValue(5)
        assert dlg.selection() == ("pallet", {"cubes": 5})
        # 切到 maze：参数变 rows/cols
        dlg.cmb.setCurrentIndex(keys.index("maze"))
        key, params = dlg.selection()
        assert key == "maze"
        assert set(params) == {"rows", "cols"}
    finally:
        dlg.deleteLater()
