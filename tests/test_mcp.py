"""M0.6 验收：MCP 工具层（方案 5.6）。

两层验证：
1. 会话层（AgentSimSession）：Agent 闭环 read_world -> 改 YAML ->
   load_world -> run -> get_observation，外加 manifest / save / load / reset
2. 协议层（真实子进程）：stdio JSON-RPC 的 initialize / tools/list /
   tools/call 线协议
"""
import json
import pathlib
import subprocess
import sys

from agentsim.mcp_server import AgentSimSession

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
            assert info["objects"] == ["cube"]

            text = s.call_tool("read_world", {})
            assert "name: pick_cube" in text

            m = s.call_tool("get_robot_manifest", {})
            arm = m["robots"][0]
            assert arm["name"] == "arm"
            assert [a["name"] for a in arm["actuators"]] == \
                ["a1", "a2", "a3", "a_l", "a_r"]
            assert arm["actuators"][0]["ctrlrange"] == [-3.14, 3.14]
            assert {j["name"] for j in arm["joints"]} >= {"j1", "j2", "j3"}

            stats = s.call_tool("run", {})
            assert stats["success"] is True
            assert stats["objects"]["cube"][1] > 0.25   # 被推进成功区

            obs = s.call_tool("get_observation", {})
            assert obs["task"]["success"] is True
            assert len(obs["robots"]["arm"]["qpos"]) == 5
        finally:
            s.close()

    def test_edit_world_reload(self):
        """Agent 改 YAML（去控制器 + cube 挪进成功区）-> yaml_text 重载"""
        text = WORLD.read_text(encoding="utf-8")
        text = text.replace("controller: ../controllers/pick_place.py", "controller:")
        text = text.replace("pos: [0.635, 0.15, 0.05]", "pos: [0.40, 0.30, 0.05]")
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
                    "save_state", "load_state", "reset", "read_world"} <= names

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
