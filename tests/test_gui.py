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
