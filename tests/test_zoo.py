"""M4 验收：机器人模型库（Menagerie 接入）。

网络保护：go2 已缓存才跑（generate 内部 build_world 已验编译，端到端只测
生成的世界能经 MCP 会话层跑起来零 crash）。
"""
import os
import pathlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from agentsim.errors import WorldError  # noqa: E402

mm = pytest.importorskip("mujoco_menagerie",
                         reason="未安装 mujoco-menagerie")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _cached(key: str, xml: str) -> bool:
    try:
        return any(f.name == xml and f.exists() for f in mm.get(key).files())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _cached("unitree_go2", "go2.xml"),
    reason="go2 未缓存（首次使用需网络下载）")


def test_ensure_model_cached_and_unknown():
    """T1：缓存命中返回存在的 xml；未知 key 报错列可用名。"""
    from agentsim.zoo import ensure_model, MODELS
    p = ensure_model("unitree_go2")
    assert p.name == "go2.xml" and p.exists()
    assert "unitree_go2" in MODELS
    with pytest.raises(WorldError, match="可用"):
        ensure_model("no_such_robot")


def test_generate_products(tmp_path):
    """T2：产物完整——world.yaml 可编译、控制器模板含执行器与 API 示例。"""
    from agentsim.zoo import generate
    from agentsim.world.schema import parse_world
    from agentsim.world.builder import build_world
    r = generate("unitree_go2", out_dir=tmp_path / "go2")
    world_p = pathlib.Path(r["world"])
    ctrl_p = pathlib.Path(r["controller"])
    assert world_p.exists() and ctrl_p.exists()
    # 世界 YAML 落盘后独立可编译（model 绝对路径）
    import yaml
    with open(world_p, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    spec = parse_world(doc)
    spec.base_dir = world_p.parent
    _model, index = build_world(spec)
    assert len(index.robots["unitree_go2"].actuators) == 12
    # 控制器模板：执行器清单 + set/range/obs 示例 + 修改引导
    text = ctrl_p.read_text(encoding="utf-8")
    for name in ("FL_hip", "RR_calf"):
        assert name in text
    for api in ("robot.set(", "robot.range(", "robot.step()", "robot.obs"):
        assert api in text
    assert "改成你的" in text


def test_generated_world_runs_end_to_end(tmp_path):
    """T3：生成的世界经 MCP 会话层 load_world + run，零 crash、清单正确。"""
    from agentsim.zoo import generate
    from agentsim.mcp_server import AgentSimSession
    r = generate("unitree_go2", out_dir=tmp_path / "go2")
    s = AgentSimSession()
    try:
        s.call_tool("load_world", {"yaml_path": r["world"]})
        stats = s.call_tool("run", {"ticks": 40})
        assert stats["ticks"] == 40
        assert stats["crashed"] == []          # 模板控制器不得 crash
    finally:
        s.close()


def test_zoo_create_list_via_session():
    """T4a：MCP zoo_create(key='list') 下发预配置清单（GUI/Agent 同构入口）。"""
    from agentsim.mcp_server import AgentSimSession
    s = AgentSimSession()
    try:
        r = s.call_tool("zoo_create", {"key": "list"})
        keys = [m["key"] for m in r["models"]]
        assert "unitree_go2" in keys and "franka_emika_panda" in keys
        assert all("title" in m and "category" in m for m in r["models"])
        with pytest.raises(WorldError):
            s.call_tool("zoo_create", {"key": "no_such"})
    finally:
        s.close()


def test_zoo_dialog_offscreen(qapp):
    """T4b：ZooDialog 离屏实例化、清单非空、selected_key 可取（GUI 闭环）。"""
    from agentsim.gui.app import ZooDialog
    dlg = ZooDialog()
    try:
        assert dlg.listw.count() >= 13
        dlg.listw.setCurrentRow(1)
        assert dlg.selected_key        # 任一 key 可取即 UI 链路通
    finally:
        dlg.deleteLater()
