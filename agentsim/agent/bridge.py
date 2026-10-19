"""MCP manifest -> OpenAI tools 转换 + system prompt 注入（纯函数，无 IO）。

OpenAI function calling 的 schema 与 MCP 的 inputSchema 同源（都是 JSON Schema
子集），转换只需字段重命名：inputSchema -> parameters。
"""
from __future__ import annotations

import json


def to_openai_tools(mcp_tools: list[dict]) -> list[dict]:
    """MCP tools 清单 -> OpenAI tools 数组（function calling 格式）。

    [{"type":"function","function":{"name":..,"description":..,"parameters":..}}]
    """
    out = []
    for t in mcp_tools:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
            },
        })
    return out


def local_tools_manifest() -> list[dict]:
    """本地文件工具的 manifest（注入 system prompt + OpenAI tools 数组
    都用——它们不经 MCP，由 AgentWorker 直接 dispatch）。"""
    return [
        {"name": "write_file",
         "description": "在当前世界目录写文件（场景 YAML / 控制器 .py 等）。"
                        "参数 path（相对当前世界目录）、content（文件全文）。",
         "inputSchema": {"type": "object",
                         "properties": {
                             "path": {"type": "string"},
                             "content": {"type": "string"}},
                         "required": ["path", "content"]}},
        {"name": "read_file",
         "description": "读当前世界目录下文件全文（含 AI 自己刚写的代码）。",
         "inputSchema": {"type": "object",
                         "properties": {"path": {"type": "string"}},
                         "required": ["path"]}},
    ]


def system_prompt(world_name: str | None, mcp_tools: list[dict]) -> str:
    """组装系统提示：身份 + 当前世界 + 工具清单（MCP 10 + 本地 2）。

    LLM 不能直接调工具——它回 tool_calls 由 worker 调，故 system prompt 里
    描述每项工具的用途，让 LLM 知道何时调。
    """
    lines = [
        "你是 AgentSim（机器人仿真平台）的 AI 助手。",
        "用户用自然语言描述目标，你调用工具生成场景、写控制律、运行验证。",
        "",
    ]
    if world_name:
        lines.append(f"当前已加载世界：{world_name}")
    lines.append("")
    lines.append("可用工具（function calling 调用）：")
    for t in mcp_tools:
        lines.append(f"- {t['name']}：{t.get('description', '').split(chr(10))[0]}")
    for t in local_tools_manifest():
        lines.append(f"- {t['name']}：{t['description']}")
    lines.append("")
    lines.append("工作流建议：")
    lines.append("- 生成机器人世界：zoo_create(key, demo=true) -> load_world -> run")
    lines.append("- 改场景/位姿：read_world 看现状 -> write_file 写新 YAML -> "
                 "load_world -> run")
    lines.append("- 改控制律：write_file 改控制器 .py -> load_world -> run")
    lines.append("- 截图自检：screenshot 看当前状态")
    return "\n".join(lines)


def extract_tool_calls(resp_json: dict) -> list[dict]:
    """从 OpenAI 响应抽 tool_calls（统一格式）。

    返回 [{name, arguments}]，arguments 为 dict（已 JSON 解码）；
    无 tool_calls 或异常时返回空列表。
    """
    try:
        choices = resp_json.get("choices") or []
        if not choices:
            return []
        msg = choices[0].get("message", {})
        calls = msg.get("tool_calls") or []
    except (TypeError, KeyError):
        return []
    out = []
    for c in calls:
        fn = c.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        args_raw = fn.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            args = {}
        out.append({"name": name, "arguments": args})
    return out


def content_text(resp_json: dict) -> str:
    """抽响应的文本部分（无 tool_calls 时给用户看）。"""
    try:
        return resp_json["choices"][0]["message"].get("content") or ""
    except (TypeError, KeyError, IndexError):
        return ""
