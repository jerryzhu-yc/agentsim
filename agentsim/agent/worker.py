"""AgentWorker：HTTP + 工具循环 QThread。

UI 非阻塞：
- send(user_text)：把消息塞队列，循环驱动 OpenAI 调用
- 工具调用：本地工具直发 dispatch_local，MCP 工具发 McpClient.call_tool
- 完成或异常：发 finished/completed 信号回 UI
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from PySide6.QtCore import QThread, Signal, Slot

from agentsim.agent.bridge import (content_text, extract_tool_calls,
                                    local_tools_manifest, system_prompt,
                                    to_openai_tools)
from agentsim.agent.config import AgentConfig
from agentsim.agent.local_tools import LocalFiles, dispatch_local


class AgentWorker(QThread):
    """OpenAI 兼容 chat/completions + 工具桥接（非阻塞后台线程）。"""

    # 信号：用户消息 / 工具调用日志 / 出错 / 一轮对话结束
    assistantText = Signal(str)             # LLM 文本回复（无 tool_calls 时）
    toolCall = Signal(str, dict)            # (工具名, args)
    toolResult = Signal(str, object)        # (工具名, 结果 dict 或 错误字串)
    error = Signal(str)
    finishedRound = Signal()                # 一轮对话结束（无更多 tool_calls）

    def __init__(self, client, parent=None):
        """client：McpClient（call_tool 异步发 MCP）。本地文件工具内置。"""
        super().__init__(parent)
        self._client = client                # McpClient
        self._config: AgentConfig = AgentConfig()
        self._files = LocalFiles()
        self._tools_manifest: list[dict] = []
        self._messages: list[dict] = []
        self._pending: list[tuple[str, dict]] = []  # 待发 McpClient 调用 [(name, args, cb)]
        # 简易消息队列：UI -> send() 入队，run() 主循环拉取
        self._queue: list[str] = []
        self._running = True

    def configure(self, cfg: AgentConfig, mcp_tools: list[dict]) -> None:
        """更新配置与工具清单（运行中也可调，下一轮生效）。"""
        self._config = cfg
        self._tools_manifest = list(mcp_tools)

    def set_world_dir(self, path) -> None:
        """加载/切换世界时调（LocalFiles 根目录）。"""
        import pathlib
        self._files.set_root(pathlib.Path(path) if path else None)

    def set_history(self, world_name: str | None) -> None:
        """加载新世界后重置对话历史（system prompt 重注）。"""
        self._messages = [{
            "role": "system",
            "content": system_prompt(world_name, self._tools_manifest),
        }]

    @Slot(str)
    def send(self, text: str) -> None:
        """UI 发用户文本（队列驱动，避免主线程直接调 OpenAI）。"""
        self._queue.append(text)

    def stop(self) -> None:
        self._running = False
        self._queue.append("__stop__")     # 唤醒可能在 wait 的循环

    # ------------------------------------------------------------ 主循环

    def run(self) -> None:
        import time
        while self._running:
            if not self._queue:
                time.sleep(0.05)
                continue
            text = self._queue.pop(0)
            if text == "__stop__":
                break
            if not self._config.usable:
                self.error.emit("未配置（点 配置 填 base_url/model/api_key）")
                continue
            try:
                self._messages.append({"role": "user", "content": text})
                self._run_round()
            except Exception as e:
                self.error.emit(f"对话异常: {e}")
        self._running = False

    def _run_round(self) -> None:
        """一轮对话：HTTP -> tool_calls? -> 分发 -> 回灌 -> 再 HTTP，直到无。"""
        max_iters = 8          # 防死循环
        for _ in range(max_iters):
            resp = self._post_chat()
            if resp is None:
                return        # 出错已发 error 信号
            text = content_text(resp)
            if text:
                self.assistantText.emit(text)
                self._messages.append({"role": "assistant", "content": text})
            calls = extract_tool_calls(resp)
            if not calls:
                self.finishedRound.emit()
                return
            # 把 tool_calls 也记进 assistant 消息（OpenAI 要求回灌 tool 角色）
            self._messages.append(self._assistant_with_tool_calls(resp, calls))
            for c in calls:
                self.toolCall.emit(c["name"], c["arguments"])
                result = self._dispatch_tool(c["name"], c["arguments"])
                self.toolResult.emit(c["name"], result)
                # tool 消息：role=tool，content=JSON 字符串，tool_call_id 关联
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": c.get("id", c["name"]),
                    "content": (json.dumps(result, ensure_ascii=False)
                                if isinstance(result, dict) else str(result)),
                })
        self.finishedRound.emit()

    def _assistant_with_tool_calls(self, resp: dict, calls: list[dict]) -> dict:
        """构造 OpenAI 期望的 assistant 消息（带 tool_calls 字段）。"""
        try:
            msg = dict(resp["choices"][0]["message"])
        except (TypeError, KeyError, IndexError):
            msg = {"role": "assistant", "content": ""}
        msg["role"] = "assistant"
        return msg

    def _dispatch_tool(self, name: str, args: dict):
        """本地工具直发；MCP 工具同步阻塞 McpClient.call_tool（在 worker 线程）。

        McpClient.call_tool 本是异步（发 JSON-RPC 给子进程），但本 worker 在
        QThread，可同步等回复（QEventLoop）：复用 GUI 工具调用链。
        """
        local = dispatch_local(self._files, name, args)
        if local is not None:
            return local
        # MCP 工具：同步阻塞调
        return self._call_mcp_sync(name, args)

    def _call_mcp_sync(self, name: str, args: dict):
        """同步等 McpClient.call_tool 回复（在 worker 线程用 QEventLoop）。"""
        from PySide6.QtCore import QEventLoop, QTimer
        loop = QEventLoop()
        result_box: list = []
        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(lambda: (result_box.append(
            {"error": "timeout"}), loop.quit()))

        def cb(resp):
            result_box.append(resp)
            loop.quit()

        timeout.start(30000)         # 30s 超时（run 等长工具也够）
        self._client.call_tool(name, args, cb)
        loop.exec()
        return result_box[0] if result_box else {"error": "no response"}

    def _post_chat(self) -> dict | None:
        """POST chat/completions；返回响应 JSON（出错发 error 返回 None）。"""
        tools = to_openai_tools(self._tools_manifest)
        # 本地工具也注入 OpenAI tools（让 LLM 知道参数 schema）
        for t in local_tools_manifest():
            tools.append({"type": "function", "function": {
                "name": t["name"], "description": t["description"],
                "parameters": t["inputSchema"]}})
        body = {
            "model": self._config.model,
            "messages": self._messages,
            "tools": tools,
            "stream": False,
        }
        body.update(self._config.extra)
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._config.chat_url(), data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self._config.api_key:
            req.add_header("Authorization", f"Bearer {self._config.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")[:500]
            except Exception:
                pass
            self.error.emit(f"HTTP {e.code}: {body or e.reason}")
            return None
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            self.error.emit(f"网络/解析错误: {e}")
            return None
