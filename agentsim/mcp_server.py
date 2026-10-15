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

import mujoco
import yaml

from agentsim import supervisor
from agentsim.errors import WorldError
from agentsim.kernel.sim import Simulation
from agentsim.world.schema import load_world as _load_world_file
from agentsim.world.schema import parse_world

_PROTOCOL = "2024-11-05"

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
        "description": "全速推进仿真并返回摘要（ticks/success/reward/物体终位）。"
                       "缺省跑到终态（控制器退出/世界 max_ticks）；ticks=N 只推进 N 步。"
                       "注意：run 返回时本轮控制器已收尾退出，继续推进请 reset 后重跑，"
                       "或改世界后重新 load_world。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticks": {"type": "integer", "minimum": 1,
                          "description": "本段推进的 tick 数上限"},
                "until_done": {"type": "boolean",
                               "description": "一直跑到终态（默认）"},
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

    # ------------------------------------------------------------ 工具入口

    def call_tool(self, name: str, args: dict):
        """执行工具，返回可直接 JSON 化的结果。WorldError 原样抛给协议层。"""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise WorldError(f"未知工具 '{name}'，可用: "
                             f"{sorted(t['name'] for t in _TOOLS)}")
        return handler(args or {})

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    # ------------------------------------------------------------ 各工具

    def _tool_load_world(self, args: dict):
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
        sim = self._require_sim()
        ticks = args.get("ticks")
        if ticks is not None:
            stats = sim.run(max_ticks=sim.tick + int(ticks))
        else:
            stats = sim.run()          # until_done / 缺省：跑到终态
        stats = dict(stats)
        stats["objects"] = {name: [round(float(x), 4) for x in sim.data.xpos[bid]]
                            for name, bid in self.sim.index.objects.items()}
        return stats

    def _tool_get_observation(self, args: dict):
        sim = self._require_sim()
        robots = {}
        for rname in sim.index.robots:
            obs = sim._obs_msg(rname)
            obs.pop("op", None)
            robots[rname] = obs
        return {
            "tick": sim.tick,
            "time": round(float(sim.data.time), 4),
            "robots": robots,
            "task": {"success": sim._task_success(),
                     "reward": round(sim._task_reward(), 4)},
        }

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
        sim = self._require_sim()
        tag = args.get("tag")
        if not tag:
            raise WorldError("save_state 缺少必填参数 tag")
        sim.save_state(tag)
        return {"tag": tag, "tick": sim.tick,
                "tags": supervisor.list_tags(sim._snaps)}

    def _tool_load_state(self, args: dict):
        sim = self._require_sim()
        tag = args.get("tag")
        if not tag:
            raise WorldError("load_state 缺少必填参数 tag")
        sim.load_state(tag)
        return {"tag": tag, "tick": sim.tick,
                "tags": supervisor.list_tags(sim._snaps)}

    def _tool_reset(self, args: dict):
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

    def _require_sim(self) -> Simulation:
        if self.sim is None:
            raise WorldError("还没有加载世界（先 load_world）")
        return self.sim


# ---------------------------------------------------------------- JSON-RPC / MCP 协议

def _rpc_result(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _rpc_error(req_id, code: int, message: str):
    return {"jsonrpc": "2.0", "id": req_id,
            "error": {"code": code, "message": message}}


def dispatch(session: AgentSimSession, method: str, params: dict, req_id):
    """单个 JSON-RPC 请求 -> 响应 dict（通知返回 None）。"""
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
                                    req.get("id"))
                except Exception as e:      # 内部错误兜底：进程绝不能死
                    resp = _rpc_error(req.get("id"), -32603,
                                      f"内部错误: {type(e).__name__}: {e}")
        if resp is not None:
            stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            stdout.flush()
    session.close()


if __name__ == "__main__":
    serve()
