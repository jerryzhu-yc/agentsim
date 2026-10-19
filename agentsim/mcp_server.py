"""MCP 工具层：把内核能力经 stdio JSON-RPC 暴露给 AI Agent（方案 5.6 节）。

零依赖实现：环境里没有 mcp 包，而协议子集很小，自己写更稳——
- stdio 传输，换行分隔的 JSON-RPC 2.0
- 只实现 MCP 必需方法：initialize / notifications/initialized / ping /
  tools/list / tools/call

工具即内核薄封装（AI Agent 的标准闭环）：
    read_world -> 改 YAML -> load_world -> run -> get_observation -> 循环

语义说明：
- load_world 会整体替换当前世界（旧仿真的控制器进程先收尾再退出）
- reset 按方案 5.4 语义：重编译 + 控制器进程重启，世界回到 YAML 初态
- save_state/load_state 用内核内存快照（tag），与 M0.4 逐位重现测试同一路径
"""
from __future__ import annotations

import json
import pathlib
import sys
import threading
import time

import mujoco
import yaml

from agentsim import supervisor
from agentsim.errors import WorldError
from agentsim.kernel.sim import Simulation
from agentsim.world.schema import load_world as _load_world_file
from agentsim.world.schema import parse_world

_PROTOCOL = "2024-11-05"

# run 的响应被推迟（后台线程跑完才写回）时 dispatch 返回此哨兵，
# serve 主循环见到它不写 stdout（响应由 run 线程经 Writer 写）。
_DEFERRED = object()

# 观测快照节流：GUI 30Hz 拉观测，每 tick 生成快照没必要
_SNAP_INTERVAL = 0.033

_JTYPE_NAME = {int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
               int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
               int(mujoco.mjtJoint.mjJNT_FREE): "free",
               int(mujoco.mjtJoint.mjJNT_BALL): "ball"}

# ---------------------------------------------------------------- 工具清单

_TOOLS = [
    {
        "name": "load_world",
        "description": "加载世界（yaml_path 与 yaml_text 二选一），整体替换当前世界。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "yaml_path": {"type": "string",
                              "description": "世界 YAML 文件路径"},
                "yaml_text": {"type": "string",
                              "description": "世界 YAML 全文（Agent 改完直接投喂）"},
                "base_dir": {"type": "string",
                             "description": "相对路径（model/controller）的解析基准目录；"
                                            "yaml_path 缺省取文件所在目录，"
                                            "yaml_text 缺省取当前目录"},
            },
        },
    },
    {
        "name": "run",
        "description": "推进仿真并返回摘要（ticks/success/reward/物体终位）。"
                       "缺省跑到终态（控制器退出/世界 max_ticks）；ticks=N 只推进 N 步。"
                       "run 在后台线程执行：期间可以继续 get_observation（返回实时快照，"
                       "GUI 实时可视化的基础）；本条响应会等 run 结束才返回。"
                       "注意：run 返回时本轮控制器已收尾退出，继续推进请 reset 后重跑，"
                       "或改世界后重新 load_world。同一时刻只允许一个 run。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticks": {"type": "integer", "minimum": 1,
                          "description": "本段推进的 tick 数上限"},
                "realtime": {"type": "boolean",
                             "description": "按墙钟实时节流（看动画用，1 仿真秒≈1 墙钟秒）；"
                                            "缺省 false 全速推进"},
            },
        },
    },
    {
        "name": "get_observation",
        "description": "当前完整观察量：各机器人 qpos/qvel/ctrl/body 位置 + "
                       "场景物体位置 + 任务状态（success/reward）。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_robot_manifest",
        "description": "机器人设备清单：执行器名字与控制范围、关节类型与限位、"
                       "body 名（带前缀的全名与短名都在）。写控制器前先看这个。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "save_state",
        "description": "存内存快照（tag），与逐位重现测试同一路径。",
        "inputSchema": {
            "type": "object",
            "required": ["tag"],
            "properties": {"tag": {"type": "string",
                                   "description": "快照名（不可用 '__' 前缀）"}},
        },
    },
    {
        "name": "load_state",
        "description": "读内存快照，物理状态逐位恢复。注意：已退出的控制器进程"
                       "不会复活，需要重跑请用 reset。",
        "inputSchema": {
            "type": "object",
            "required": ["tag"],
            "properties": {"tag": {"type": "string"}},
        },
    },
    {
        "name": "reset",
        "description": "世界回到 YAML 初态：重编译 + 控制器进程重启。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_world",
        "description": "回读当前世界的 YAML 原文（Agent 修改世界前先看）。",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------- 工具实现

class AgentSimSession:
    """MCP 工具的内核侧状态：当前世界 + 当前仿真。"""

    def __init__(self):
        self.spec = None
        self.sim: Simulation | None = None
        self.yaml_text: str | None = None
        # run 后台线程化（GUI 实时观测的基础）
        self._run_thread: threading.Thread | None = None
        self._snap = None                 # run 期间最新观测快照（run 线程写）
        self._snap_t = 0.0
        self._snap_lock = threading.Lock()
        self._run_req = None              # (req_id, writer)：run 结束后写延迟响应
        self._run_error: BaseException | None = None
        self._last_stats: dict | None = None

    # ------------------------------------------------------------ 工具入口

    def call_tool(self, name: str, args: dict):
        """执行工具，返回可直接 JSON 化的结果。WorldError 原样抛给协议层。"""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise WorldError(f"未知工具 '{name}'，可用: "
                             f"{sorted(t['name'] for t in _TOOLS)}")
        return handler(args or {})

    @property
    def is_running(self) -> bool:
        return self._run_thread is not None and self._run_thread.is_alive()

    def close(self):
        # run 还在跑（如 realtime 世界）：给它一点时间收尾，等不及就随进程退出
        if self.is_running and threading.current_thread() is not self._run_thread:
            self._run_thread.join(timeout=5)
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    # ------------------------------------------------------------ 各工具

    def _tool_load_world(self, args: dict):
        self._require_idle()
        path, text = args.get("yaml_path"), args.get("yaml_text")
        if (path is None) == (text is None):
            raise WorldError("load_world 需要 yaml_path 或 yaml_text 恰好一个")
        if path is not None:
            spec = _load_world_file(path)          # base_dir = 文件所在目录
            yaml_text = pathlib.Path(path).read_text(encoding="utf-8")
        else:
            try:
                d = yaml.safe_load(text)
            except yaml.YAMLError as e:
                raise WorldError(f"YAML 语法错误: {e}") from e
            spec = parse_world(d)
            base = args.get("base_dir") or "."
            spec.base_dir = pathlib.Path(base).resolve()
            yaml_text = text

        self.close()
        self.spec, self.yaml_text = spec, yaml_text
        self.sim = Simulation(spec)
        return {
            "world": spec.name,
            "objects": [o.name for o in spec.scene.objects],
            "robots": [r.name for r in spec.robots],
            "task_max_ticks": spec.task.max_ticks,
        }

    def _tool_run(self, args: dict):
        """会话层同步语义：起后台线程并等它跑完（旧 API 行为不变）。

        stdio 协议层（dispatch）走 start_run(block=False)：响应推迟到
        run 结束，期间 serve 循环继续应答 get_observation 等请求。
        """
        return self.start_run(args, block=True)

    def start_run(self, args: dict, req_id=None, writer=None, block: bool = False):
        """启动后台 run。返回 _DEFERRED（异步）或 stats dict（block=True）。

        WorldError（未加载世界/并发 run）同步抛出。
        """
        sim = self._require_sim()
        if self.is_running:
            raise WorldError("已有 run 在进行中，等它结束（或完成后 reset）")
        ticks = args.get("ticks")
        max_ticks = (sim.tick + int(ticks)) if ticks is not None else None
        realtime = bool(args.get("realtime", False))
        self._run_error = None
        self._run_req = (req_id, writer) if req_id is not None else None
        self._snap = None
        self._snap_t = 0.0

        def job():
            try:
                stats = self._run_job(max_ticks, realtime)
            except BaseException as e:      # 线程里异常不能无声吞掉
                self._run_error = e
                stats = {"error": f"{type(e).__name__}: {e}"}
            self._write_run_response(stats)

        self._run_thread = threading.Thread(target=job, daemon=True,
                                            name="agentsim-run")
        self._run_thread.start()
        if block:
            self._run_thread.join()
            if self._run_error is not None:
                raise self._run_error
            return self._last_stats
        return _DEFERRED

    def _run_job(self, max_ticks, realtime=False):
        sim = self.sim               # 线程里重取（load_world 会换实例）
        stats = dict(sim.run(max_ticks=max_ticks, on_tick=self._on_tick,
                             realtime=realtime))
        stats["objects"] = self._objects_pos(sim)
        self._last_stats = stats
        return stats

    def _on_tick(self):
        """内核每个 tick 回调（run 线程内，data 状态一致可安全读）。"""
        now = time.perf_counter()
        if now - self._snap_t < _SNAP_INTERVAL:
            return
        self._snap_t = now
        with self._snap_lock:
            self._snap = self._obs_payload()

    def _write_run_response(self, stats):
        if self._run_req is not None:
            req_id, writer = self._run_req
            self._run_req = None
            if writer is not None:
                is_err = "error" in stats
                writer.write(_rpc_result(req_id, {
                    "content": [{"type": "text",
                                 "text": json.dumps(stats, ensure_ascii=False)}],
                    "isError": is_err}))

    def _objects_pos(self, sim) -> dict:
        return {name: [round(float(x), 4) for x in sim.data.xpos[bid]]
                for name, bid in sim.index.objects.items()}

    def _obs_payload(self) -> dict:
        """完整观测（含物体 pos+quat）。调用方保证此刻没有并发 mj_step。"""
        sim = self.sim
        # xpos/xquat 是派生量：刚 load（还没跑过 step）或 load_state 恢复后
        # 都是过期/零值。mj_kinematics 只重算运动学，不碰 warmstart，
        # 对 bit 级轨迹重现无影响。
        mujoco.mj_kinematics(sim.model, sim.data)
        robots = {}
        for rname in sim.index.robots:
            obs = sim._obs_msg(rname)
            obs.pop("op", None)
            robots[rname] = obs
        objects = {}
        for name, bid in sim.index.objects.items():
            objects[name] = {
                "pos": [round(float(x), 4) for x in sim.data.xpos[bid]],
                "quat": [round(float(x), 4) for x in sim.data.xquat[bid]],
            }
        return {
            "tick": sim.tick,
            "time": round(float(sim.data.time), 4),
            "running": self.is_running,
            "robots": robots,
            "objects": objects,
            "task": {"success": sim._task_success(),
                     "reward": round(sim._task_reward(), 4)},
        }

    def _tool_get_observation(self, args: dict):
        sim = self._require_sim()
        if self.is_running:
            # run 期间 data 正被 run 线程 mj_step，直接读有竞态——返回快照
            with self._snap_lock:
                snap = self._snap
            if snap is not None:
                return snap
            return {"running": True, "tick": sim.tick,
                    "note": "首帧快照未生成（run 刚启动）"}
        return self._obs_payload()

    def _tool_get_robot_manifest(self, args: dict):
        sim = self._require_sim()
        m = sim.model
        robots = []
        for rd in self.spec.robots:
            ridx = sim.index.robots[rd.name]
            actuators = []
            for a_name, aid in ridx.actuators.items():
                lo, hi = m.actuator_ctrlrange[aid]
                actuators.append({"name": a_name,
                                  "ctrlrange": [float(lo), float(hi)]})
            joints = []
            for j_name, jid in ridx.joints.items():
                joints.append({
                    "name": j_name,
                    "type": _JTYPE_NAME.get(int(m.jnt_type[jid]), "?"),
                    "limited": bool(m.jnt_limited[jid]),
                    "range": [float(x) for x in m.jnt_range[jid]],
                })
            robots.append({
                "name": rd.name,
                "controller": rd.controller,
                "args": rd.args,
                "actuators": actuators,
                "joints": joints,
                "bodies": {f"{rd.name}/{short}": short for short in ridx.bodies},
            })
        return {
            "world": self.spec.name,
            "timestep": self.spec.timestep,
            "ctrl_period": self.spec.ctrl_period,
            "robots": robots,
            "objects": [{"name": o.name, "type": o.type, "pos": list(o.pos)}
                        for o in self.spec.scene.objects],
        }

    def _tool_save_state(self, args: dict):
        self._require_idle()
        sim = self._require_sim()
        tag = args.get("tag")
        if not tag:
            raise WorldError("save_state 缺少必填参数 tag")
        sim.save_state(tag)
        return {"tag": tag, "tick": sim.tick,
                "tags": supervisor.list_tags(sim._snaps)}

    def _tool_load_state(self, args: dict):
        self._require_idle()
        sim = self._require_sim()
        tag = args.get("tag")
        if not tag:
            raise WorldError("load_state 缺少必填参数 tag")
        sim.load_state(tag)
        return {"tag": tag, "tick": sim.tick,
                "tags": supervisor.list_tags(sim._snaps)}

    def _tool_reset(self, args: dict):
        self._require_idle()
        if self.spec is None:
            raise WorldError("还没有加载世界（先 load_world）")
        self.close()
        self.sim = Simulation(self.spec)     # 重编译 + 控制器进程重启
        return {"world": self.spec.name, "tick": 0}

    def _tool_read_world(self, args: dict):
        if self.yaml_text is None:
            raise WorldError("还没有加载世界（先 load_world）")
        return self.yaml_text

    # ------------------------------------------------------------ 辅助

    def _require_idle(self):
        """run 进行中禁止换世界/动状态（否则 run 线程会用着已 close 的 sim）。"""
        if self.is_running:
            raise WorldError("run 正在进行中，此操作需要等它结束")

    def _require_sim(self) -> Simulation:
        if self.sim is None:
            raise WorldError("还没有加载世界（先 load_world）")
        return self.sim


# ---------------------------------------------------------------- JSON-RPC / MCP 协议

class _Writer:
    """线程安全写出器：serve 线程与 run 线程共用 stdout。"""

    def __init__(self, out):
        self._out = out
        self._lock = threading.Lock()

    def write(self, obj) -> None:
        with self._lock:
            self._out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._out.flush()


def _rpc_result(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _rpc_error(req_id, code: int, message: str):
    return {"jsonrpc": "2.0", "id": req_id,
            "error": {"code": code, "message": message}}


def dispatch(session: AgentSimSession, method: str, params: dict, req_id,
             writer: _Writer | None = None):
    """单个 JSON-RPC 请求 -> 响应 dict（通知返回 None，延迟响应返回 _DEFERRED）。"""
    if req_id is None:            # 通知：静默接受
        return None
    if method == "initialize":
        version = (params or {}).get("protocolVersion") or _PROTOCOL
        return _rpc_result(req_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "agentsim", "version": "0.1.0"},
        })
    if method == "ping":
        return _rpc_result(req_id, {})
    if method == "tools/list":
        return _rpc_result(req_id, {"tools": _TOOLS})
    if method == "tools/call":
        params = params or {}
        name = params.get("name", "")
        args = params.get("arguments") or {}
        if name == "run":
            # run 后台线程化：响应推迟到跑完（GUI 实时观测的基础）。
            # 校验类错误（未加载世界/并发 run）仍同步返回。
            try:
                return session.start_run(args, req_id, writer, block=False)
            except WorldError as e:
                return _rpc_result(req_id, {
                    "content": [{"type": "text", "text": str(e)}],
                    "isError": True})
        try:
            result = session.call_tool(name, args)
        except WorldError as e:
            return _rpc_result(req_id, {
                "content": [{"type": "text", "text": str(e)}],
                "isError": True})
        text = result if isinstance(result, str) else json.dumps(
            result, ensure_ascii=False)
        return _rpc_result(req_id, {
            "content": [{"type": "text", "text": text}], "isError": False})
    return _rpc_error(req_id, -32601, f"未知方法: {method}")


def serve(stdin=sys.stdin, stdout=sys.stdout) -> None:
    """MCP stdio 主循环：每行一个 JSON-RPC 消息。"""
    session = AgentSimSession()
    writer = _Writer(stdout)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            resp = _rpc_error(None, -32700, f"解析错误: {e}")
        else:
            method = req.get("method", "") if isinstance(req, dict) else ""
            if not method:
                resp = _rpc_error(req.get("id") if isinstance(req, dict) else None,
                                  -32600, "非法请求：缺少 method")
            else:
                try:
                    resp = dispatch(session, method, req.get("params"),
                                    req.get("id"), writer)
                except Exception as e:      # 内部错误兜底：进程绝不能死
                    resp = _rpc_error(req.get("id"), -32603,
                                      f"内部错误: {type(e).__name__}: {e}")
        if resp is not None and resp is not _DEFERRED:
            writer.write(resp)
    session.close()


if __name__ == "__main__":
    serve()
