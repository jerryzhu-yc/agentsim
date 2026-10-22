"""M0.6 验收：MCP 工具层（方案 5.6）。

两层验证：
1. 会话层（AgentSimSession）：Agent 闭环 read_world -> 改 YAML ->
   load_world -> run -> get_observation，外加 manifest / save / load / reset
2. 协议层（真实子进程）：stdio JSON-RPC 的 initialize / tools/list /
   tools/call 线协议
"""
import base64
import json
import pathlib
import struct
import subprocess
import sys
import threading

import pytest
import yaml

from agentsim.errors import WorldError
from agentsim.mcp_server import AgentSimSession   # 导入即 setdefault MUJOCO_GL=egl


def _have_offscreen_glx() -> bool:
    """离屏渲染可用性探测（EGL 由 mcp_server 导入时默认）。"""
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_string("<mujoco/>")
        r = mujoco.Renderer(m, 32, 32)
        r.render()
        r.close()
        return True
    except Exception:
        return False


_OSSCREEN = _have_offscreen_glx()
ROOT = pathlib.Path(__file__).parent.parent
WORLD = ROOT / "worlds" / "pick_cube.yaml"


class TestAgentLoop:
    def test_closed_loop_end_to_end(self):
        """5.6 节闭环：load_world -> read_world -> manifest -> run -> observation"""
        s = AgentSimSession()
        try:
            info = s.call_tool("load_world", {"yaml_path": str(WORLD)})
            assert info["world"] == "pick_cube"
            assert info["robots"] == ["arm"]
            assert info["objects"] == ["table", "cube"]

            text = s.call_tool("read_world", {})
            assert "name: pick_cube" in text

            m = s.call_tool("get_robot_manifest", {})
            arm = m["robots"][0]
            assert arm["name"] == "arm"
            assert [a["name"] for a in arm["actuators"]] == \
                ["actuator1", "actuator2", "actuator3", "actuator4",
                 "actuator5", "actuator6", "actuator7",
                 "gripper/gripper_act"]
            assert arm["actuators"][0]["ctrlrange"] == [-2.8973, 2.8973]
            assert {j["name"] for j in arm["joints"]} >= \
                {"joint1", "joint2", "joint3"}

            stats = s.call_tool("run", {})
            assert stats["success"] is True
            assert stats["objects"]["cube"][2] > 0.80   # 被夹进空中成功区

            obs = s.call_tool("get_observation", {})
            assert obs["task"]["success"] is True
            assert len(obs["robots"]["arm"]["qpos"]) == 9
        finally:
            s.close()

    def test_edit_world_reload(self):
        """Agent 改 YAML（去控制器 + 成功区下移到桌面）-> yaml_text 重载"""
        text = WORLD.read_text(encoding="utf-8")
        text = text.replace("controller: ../controllers/pick_place.py", "controller:")
        # 空中 zone 改到桌面：cube 静置桌上即在区内（run 收尾才判终态，
        # 空中 zone 下 cube 坠落 0.64s 早已出区）
        text = text.replace(
            "zone: {min: [0.49, -0.06, 0.80], max: [0.61, 0.06, 0.92]}",
            "zone: {min: [0.49, -0.06, 0.68], max: [0.61, 0.06, 0.80]}")
        text = text.replace("max_ticks: 5000", "max_ticks: 10")
        s = AgentSimSession()
        try:
            s.call_tool("load_world", {"yaml_text": text,
                                       "base_dir": str(ROOT / "worlds")})
            stats = s.call_tool("run", {})
            assert stats["ticks"] == 10
            assert stats["success"] is True
        finally:
            s.close()

    def test_save_world_roundtrip(self, tmp_path):
        """save_world：GUI 拖动/位姿修改（yaml_text 链）后落盘，重开文件
        修改仍在——"改了半天白改"的守护。"""
        s = AgentSimSession()
        try:
            s.call_tool("load_world", {"yaml_path": str(WORLD)})
            # 模拟 GUI 编辑：改 cube 位置（去 controller——保存到 tmp 后
            # 相对路径失效，无控制器也能 load/run）
            d = yaml.safe_load(s.call_tool("read_world", {}))
            for r in d.get("robots") or []:
                r["controller"] = None
            for o in d["scene"]["objects"]:
                if o["name"] == "cube":
                    o["pos"] = [0.55, 0.02, 0.71]
            new_text = yaml.safe_dump(d, allow_unicode=True, sort_keys=False)
            s.call_tool("load_world", {"yaml_text": new_text,
                                       "base_dir": str(ROOT / "worlds")})
            # 保存回原文件路径
            out = tmp_path / "saved.yaml"
            r = s.call_tool("save_world", {"path": str(out)})
            assert r["saved"] == str(out) and out.exists()
            # 重开保存的文件：修改还在
            s.call_tool("load_world", {"yaml_path": str(out)})
            obs = s.call_tool("get_observation", {})
            assert obs["objects"]["cube"]["pos"][:3] == [0.55, 0.02, 0.71]
            # world_path 追踪：yaml_text 重载不换来源文件，缺省保存回原文件
            s.call_tool("load_world", {"yaml_path": str(WORLD)})
            s.call_tool("load_world", {"yaml_text": new_text,
                                       "base_dir": str(ROOT / "worlds")})
            out2 = tmp_path / "origin.yaml"
            r2 = s.call_tool("save_world", {"path": str(out2)})
            assert r2["saved"] == str(out2)      # 显式 path 优先
        finally:
            s.close()

    def test_state_and_reset(self):
        s = AgentSimSession()
        try:
            s.call_tool("load_world", {"yaml_path": str(WORLD)})
            stats = s.call_tool("run", {"ticks": 5})
            assert stats["ticks"] == 5

            r = s.call_tool("save_state", {"tag": "mid"})
            assert r["tick"] == 5 and "mid" in r["tags"]
            s.call_tool("load_state", {"tag": "mid"})
            assert s.sim.tick == 5

            s.call_tool("reset", {})          # 重编译 + 控制器重启
            assert s.sim.tick == 0
        finally:
            s.close()

    def test_controller_survives_agent_think_time(self):
        """真实 Agent 在 load_world 与 run 之间有数秒以上"思考时间"，
        控制器必须活着等内核开跑（压测实测：旧 5s RCVTIMEO 下必死）。"""
        import time
        s = AgentSimSession()
        try:
            s.call_tool("load_world", {"yaml_path": str(WORLD)})
            time.sleep(6)                     # 超过旧的 5s hello 超时
            stats = s.call_tool("run", {})
            assert stats["ticks"] > 0, "控制器在 Agent 思考期间超时死亡"
            assert stats["success"] is True
        finally:
            s.close()

    def test_crashed_controller_visible_in_stats(self):
        """未握手即死的控制器（脚本自身错误）必须在 run 统计里上报，
        不能静默返回 ticks=0（压测实测踩过）。"""
        import tempfile
        bad = pathlib.Path(tempfile.mkdtemp()) / "bad_ctrl.py"
        bad.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
        text = WORLD.read_text(encoding="utf-8")
        text = text.replace(
            "controller: ../controllers/pick_place.py",
            f"controller: {bad}")
        s = AgentSimSession()
        try:
            s.call_tool("load_world", {"yaml_text": text,
                                       "base_dir": str(ROOT / "worlds")})
            import time
            time.sleep(1)                      # 等控制器进程死透
            stats = s.call_tool("run", {})
            assert stats["crashed"] == ["arm"], stats
            assert stats["ticks"] == 0
        finally:
            s.close()

    def test_controller_exits_promptly_after_run(self):
        """控制器 bye 后内核必须回 bye_ack，进程毫秒级退出——
        否则 recv 永久阻塞，每轮 run 泄漏一个控制器进程（压测实测踩过）。"""
        import subprocess
        s = AgentSimSession()
        try:
            s.call_tool("load_world", {"yaml_path": str(WORLD)})
            stats = s.call_tool("run", {})
            assert stats["done"] == ["arm"]
            proc = s.sim._procs["arm"]
            try:
                proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                raise AssertionError("控制器在 run 结束 1.5s 后仍未退出（泄漏）")
        finally:
            s.close()

    def test_partial_run_releases_controller(self):
        """run(ticks=N) 中途截断后控制器必须收到 stop 退出——
        stop 窗口竞态（step 在 run 返回后才到）曾致永久阻塞（压测实测）。"""
        import subprocess
        s = AgentSimSession()
        try:
            s.call_tool("load_world", {"yaml_path": str(WORLD)})
            stats = s.call_tool("run", {"ticks": 50})   # 截断时控制器还在相位机中
            assert stats["ticks"] == 50
            proc = s.sim._procs["arm"]
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                raise AssertionError("run 截断后控制器未在 3s 内退出（窗口泄漏）")
        finally:
            s.close()

    def test_observation_has_object_quat(self):
        """get_observation 顶层 objects 带 pos+quat（GUI 渲染写 freejoint 需要）。"""
        s = AgentSimSession()
        try:
            s.call_tool("load_world", {"yaml_path": str(WORLD)})
            obs = s.call_tool("get_observation", {})
            cube = obs["objects"]["cube"]
            assert len(cube["pos"]) == 3
            q = cube["quat"]
            assert len(q) == 4
            assert abs(sum(x * x for x in q) - 1.0) < 1e-6   # 单位四元数
        finally:
            s.close()

    def test_run_streaming_observation(self):
        """run 后台线程化：run 进行中 get_observation 返回实时快照（tick 在涨），
        不等 run 结束、不死锁——GUI 实时可视化的地基。"""
        import threading
        import time
        text = WORLD.read_text(encoding="utf-8")
        text = text.replace("controller: ../controllers/pick_place.py", "controller:")
        text = text.replace("max_ticks: 5000", "max_ticks: 50000")
        s = AgentSimSession()
        t = None
        try:
            s.call_tool("load_world", {"yaml_text": text,
                                       "base_dir": str(ROOT / "worlds")})
            done = {}

            def runner():
                done["stats"] = s.call_tool("run", {})

            t = threading.Thread(target=runner)
            t.start()
            time.sleep(1.0)                     # run 进行中（全速也要跑数秒）
            obs = s.call_tool("get_observation", {})
            assert obs.get("running") is True, obs
            assert 0 < obs["tick"] < 50000, obs
            t.join(timeout=60)
            assert done["stats"]["ticks"] == 50000
        finally:
            if t is not None and t.is_alive():
                t.join(timeout=60)
            s.close()

    def test_stop_tool_stops_running_sim(self):
        """stop 工具：run 进行中调 stop -> 立即返回且 run 提前收尾，
        stats 带 stopped=true（停止不是崩溃）。"""
        import threading
        import time
        text = WORLD.read_text(encoding="utf-8")
        text = text.replace("controller: ../controllers/pick_place.py", "controller:")
        text = text.replace("max_ticks: 5000", "max_ticks: 100000")
        s = AgentSimSession()
        t = None
        try:
            s.call_tool("load_world", {"yaml_text": text,
                                       "base_dir": str(ROOT / "worlds")})
            done = {}

            def runner():
                done["stats"] = s.call_tool("run", {})

            t = threading.Thread(target=runner)
            t.start()
            time.sleep(1.0)                     # run 进行中

            r = s.call_tool("stop", {})          # 立即返回
            assert r["stopped"] is True, r
            t.join(timeout=30)
            stats = done["stats"]
            assert stats["stopped"] is True     # run 的响应带停止标记
            assert stats["ticks"] < 100000       # 确实提前收尾
            assert stats["crashed"] == []        # 停止不是崩溃

            # 无控制器世界没有进程可泄漏；再验证"没有 run 时 stop 无操作"
            r2 = s.call_tool("stop", {})
            assert r2["stopped"] is False
        finally:
            if t is not None and t.is_alive():
                t.join(timeout=30)
            s.close()

    def test_stdio_async_run_interleaved_response(self):
        """stdio 层：run 响应推迟到跑完，期间 ping 立即应答且先到
        （GUI 的请求乱序到达必须按 id 匹配）。"""
        import subprocess
        import sys as _sys
        import time
        root = pathlib.Path(__file__).resolve().parents[1]
        proc = subprocess.Popen(
            [_sys.executable, "-m", "agentsim", "mcp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=str(root), text=True)
        try:
            def send(obj):
                proc.stdin.write(json.dumps(obj) + "\n")
                proc.stdin.flush()

            def read_until(pred, timeout=30):
                deadline = time.time() + timeout
                while time.time() < deadline:
                    line = proc.stdout.readline()
                    if not line:
                        raise AssertionError("MCP 服务器提前退出")
                    resp = json.loads(line)
                    if pred(resp):
                        return resp
                raise AssertionError("等待响应超时")

            send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                  "params": {}})
            read_until(lambda r: r["id"] == 1)
            send({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                  "params": {"name": "load_world",
                             "arguments": {"yaml_path": str(WORLD)}}})
            r = read_until(lambda r: r["id"] == 2)
            assert not r["result"]["isError"]

            # run（响应会推迟）紧跟 ping（应立即先回）
            send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                  "params": {"name": "run", "arguments": {}}})
            send({"jsonrpc": "2.0", "id": 4, "method": "ping"})
            r_ping = read_until(lambda r: r["id"] == 4)
            assert r_ping["result"] == {}
            r_run = read_until(lambda r: r["id"] == 3)
            stats = json.loads(r_run["result"]["content"][0]["text"])
            assert stats["ticks"] > 0 and "success" in stats
        finally:
            proc.stdin.close()
            proc.wait(timeout=15)


class TestStdioProtocol:
    def test_jsonrpc_over_stdio(self):
        proc = subprocess.Popen(
            [sys.executable, "-m", "agentsim", "mcp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, cwd=str(ROOT))

        def rpc(method, params=None, req_id=1):
            proc.stdin.write(json.dumps(
                {"jsonrpc": "2.0", "id": req_id, "method": method,
                 "params": params if params is not None else {}}) + "\n")
            return json.loads(proc.stdout.readline())

        try:
            r = rpc("initialize", {"protocolVersion": "2024-11-05"})
            assert r["result"]["serverInfo"]["name"] == "agentsim"

            r = rpc("tools/list")
            names = {t["name"] for t in r["result"]["tools"]}
            assert {"load_world", "run", "get_observation", "get_robot_manifest",
                    "save_state", "load_state", "reset", "read_world",
                    "screenshot"} <= names

            r = rpc("tools/call", {"name": "load_world",
                                   "arguments": {"yaml_path": str(WORLD)}})
            assert r["result"]["isError"] is False

            r = rpc("tools/call", {"name": "run", "arguments": {"ticks": 3}})
            assert json.loads(r["result"]["content"][0]["text"])["ticks"] == 3

            r = rpc("tools/call", {"name": "nope"})       # 工具级错误
            assert r["result"]["isError"] is True

            r = rpc("bogus/method")                        # 协议级错误
            assert r["error"]["code"] == -32601
        finally:
            proc.stdin.close()
            proc.wait(timeout=15)

    @pytest.mark.skipif(not _OSSCREEN, reason="本环境无离屏渲染（EGL）")
    def test_stdio_screenshot_image_content(self):
        """stdio 线协议层：screenshot 的 image content 原样透传给 Agent。"""
        proc = subprocess.Popen(
            [sys.executable, "-m", "agentsim", "mcp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, cwd=str(ROOT))

        def rpc(method, params=None, req_id=1):
            proc.stdin.write(json.dumps(
                {"jsonrpc": "2.0", "id": req_id, "method": method,
                 "params": params if params is not None else {}}) + "\n")
            return json.loads(proc.stdout.readline())

        try:
            rpc("initialize", {"protocolVersion": "2024-11-05"})
            r = rpc("tools/call", {"name": "load_world",
                                   "arguments": {"yaml_path": str(WORLD)}})
            assert r["result"]["isError"] is False
            r = rpc("tools/call", {"name": "screenshot",
                                   "arguments": {"width": 32, "height": 32}})
            assert r["result"]["isError"] is False
            block = r["result"]["content"][0]
            assert block["type"] == "image"
            assert block["mimeType"] == "image/png"
            assert _png_size(base64.b64decode(block["data"])) == (32, 32)
        finally:
            proc.stdin.close()
            proc.wait(timeout=15)


# ---------------------------------------------------------------- screenshot（T9-T13）

def _world_with_sensors():
    """pick_cube + arm 传感器全套（含相机），Agent 改 YAML 的同一注入路径。"""
    text = WORLD.read_text(encoding="utf-8")
    anchor = "    controller: ../controllers/pick_place.py"
    sensors = (
        "\n    sensors:\n"
        "      - {name: j1_enc, type: encoder, joint: joint1}\n"
        "      - {name: wrist_imu, type: imu, body: link7, pos: [0, 0, 0.02]}\n"
        "      - {name: fl_touch, type: tactile,"
        " body: gripper/left_finger, pos: [0, 0, 0]}\n"
        "      - {name: gripper_cam, type: camera, body: link7,"
        " pos: [0, 0.06, 0.10], quat: [0.7071, 0.7071, 0, 0]}")
    return text.replace(anchor, anchor + sensors)


def _png_size(data: bytes) -> tuple[int, int]:
    """从 PNG bytes 里解 IHDR 的宽高。"""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])


@pytest.mark.skipif(not _OSSCREEN, reason="本环境无离屏渲染（EGL）")
class TestScreenshot:
    def _session(self):
        s = AgentSimSession()
        s.call_tool("load_world", {"yaml_text": _world_with_sensors(),
                                   "base_dir": str(ROOT / "worlds")})
        return s

    def test_inline_image_content(self):
        """无 path：返回 image content，base64 解码后是 PNG 且尺寸正确（T9）。"""
        s = self._session()
        try:
            r = s.call_tool("screenshot", {"width": 64, "height": 48})
            assert r["isError"] is False
            block = r["content"][0]
            assert block["type"] == "image"
            assert block["mimeType"] == "image/png"
            png = base64.b64decode(block["data"])
            assert _png_size(png) == (64, 48)
        finally:
            s.close()

    def test_camera_view_differs_from_free(self):
        """相机机位（manifest 短名）与自由机位截图不同（T10）。"""
        s = self._session()
        try:
            free = s.call_tool("screenshot", {"width": 96, "height": 72})
            cam = s.call_tool("screenshot", {"width": 96, "height": 72,
                                             "camera": "gripper_cam"})
            assert free["content"][0]["data"] != cam["content"][0]["data"]
            # 不存在的相机：报错并给出可用名
            with pytest.raises(WorldError, match="可用"):
                s.call_tool("screenshot", {"camera": "no_such_cam"})
        finally:
            s.close()

    def test_path_mode_writes_png_file(self, tmp_path):
        """path 模式写盘并可解 PNG 头（T11）。"""
        p = tmp_path / "shot.png"
        s = self._session()
        try:
            r = s.call_tool("screenshot", {"width": 32, "height": 24,
                                           "path": str(p)})
            assert r["path"] == str(p)
            assert r["width"] == 32 and r["height"] == 24
            assert r["bytes"] > 0
            assert _png_size(p.read_bytes()) == (32, 24)
        finally:
            s.close()

    def test_rejected_while_run_in_progress(self):
        """run 进行中调用 screenshot 必须 isError（快照与渲染有竞态，T12）。

        _require_idle 只看 _run_thread 存活状态——注入一个活跃线程与真实
        run 等价，免去真实 run 的启动窗口竞态（轮询 is_running 也可能错过）。
        """
        s = AgentSimSession()
        try:
            s.call_tool("load_world", {"yaml_path": str(WORLD)})
            stop = threading.Event()

            def hold():                 # 挂住直到测试放行，保证线程全程存活
                stop.wait(timeout=30)

            s._run_thread = threading.Thread(target=hold, daemon=True)
            s._run_thread.start()
            with pytest.raises(WorldError, match="run 正在进行中"):
                s.call_tool("screenshot", {})
            stop.set()
            s._run_thread.join(timeout=5)
        finally:
            s.close()

    def test_manifest_lists_sensors_and_cameras(self):
        """manifest 含传感器（名/类型/维度）与相机名（T13）。"""
        s = self._session()
        try:
            m = s.call_tool("get_robot_manifest", {})
            arm = m["robots"][0]
            dims = {x["name"]: x["dim"] for x in arm["sensors"]}
            types = {x["name"]: x["type"] for x in arm["sensors"]}
            assert dims["j1_enc"] == 1 and types["j1_enc"] == "encoder"
            assert dims["wrist_imu"] == 6 and types["wrist_imu"] == "imu"
            assert types["fl_touch"] == "tactile"
            assert arm["cameras"] == ["gripper_cam"]
        finally:
            s.close()

    def test_renderer_cache_and_world_reload(self):
        """同尺寸复用缓存 Renderer；load_world 换世界后旧 Renderer 失效可重截。"""
        s = self._session()
        try:
            s.call_tool("screenshot", {"width": 40, "height": 30})
            rend = s._renderer
            s.call_tool("screenshot", {"width": 40, "height": 30})
            assert s._renderer is rend            # 尺寸不变不重建
            s.call_tool("screenshot", {"width": 50, "height": 30})
            assert s._renderer is not rend         # 尺寸变则重建
            s.call_tool("load_world", {"yaml_path": str(WORLD)})
            assert s._renderer is None             # 换世界销毁
            r = s.call_tool("screenshot", {"width": 40, "height": 30})
            assert r["content"][0]["type"] == "image"
        finally:
            s.close()
