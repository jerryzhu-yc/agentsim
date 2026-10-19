"""GUI 侧 MCP 客户端：用 QProcess 托管 `python -m agentsim mcp` 后端。

分层（下层不依赖 Qt，可无头单测）：
- LineCodec：字节流 -> 完整行（防半行，FIFO/管道分块到达都能拼）
- PendingMap：id -> 回调登记，响应按 id 派发（run 的响应会晚于其后
  请求的响应——服务器端 run 是延迟响应，绝不能按序配对）
- McpClient(QObject)：QProcess 生命周期 + 行协议编解码 + 回调派发

GUI 与 AI Agent 同构：都只是 MCP 客户端，走同一套工具。
"""
from __future__ import annotations

import json
import sys

from PySide6.QtCore import QObject, QProcess, Signal


# ---------------------------------------------------------------- 纯逻辑层

class LineCodec:
    """把陆续到达的字节块拼成完整行。"""

    def __init__(self):
        self._buf = ""

    def feed(self, chunk: str) -> list[str]:
        self._buf += chunk
        parts = self._buf.split("\n")
        self._buf = parts.pop()          # 尾段：不完整行（或空串）
        return [ln for ln in parts if ln.strip()]


class PendingMap:
    """请求-响应按 id 关联：响应乱序到达也能派发到正确的回调。"""

    def __init__(self):
        self._cbs: dict = {}

    def register(self, rid: int, cb) -> None:
        self._cbs[rid] = cb

    def dispatch(self, resp: dict) -> bool:
        """派发一个响应。返回 False 表示没有登记的 id（如服务器自发消息）。"""
        rid = resp.get("id")
        cb = self._cbs.pop(rid, None)
        if cb is None:
            return False
        cb(resp)
        return True

    def pending_ids(self) -> list[int]:
        return list(self._cbs)


def tool_response(resp: dict):
    """把 JSON-RPC 响应解析成 (is_error, data)。

    data：run/get_observation 等 dict 工具解析为 dict；read_world 等
    str 工具原样返回；协议层错误返回 message 字符串。
    """
    if "error" in resp:
        return True, str(resp["error"].get("message", resp["error"]))
    result = resp.get("result", {})
    if result.get("isError"):
        return True, result.get("content", [{}])[0].get("text", "")
    text = result.get("content", [{}])[0].get("text", "")
    try:
        return False, json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, text


# ---------------------------------------------------------------- Qt 客户端

class McpClient(QObject):
    """QProcess 托管的 MCP 后端客户端。

    用法：
        client = McpClient(self)
        client.start()
        client.call_tool("load_world", {"yaml_path": "..."},
                         on_done=lambda err, data: ...)
    """

    backendDied = Signal(str)            # 后端异常退出（msg）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._codec = LineCodec()
        self._pending = PendingMap()
        self._next_id = 0

    # ------------------------------------------------------------ 生命周期

    def start(self) -> None:
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._proc.readyRead.connect(self._on_ready_read)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(
            lambda err: self.backendDied.emit(f"后端进程错误: {err}"))
        # cwd=项目根：`python -m agentsim mcp` 能找到 agentsim 包
        root = self._package_root()
        self._proc.setWorkingDirectory(str(root))
        self._proc.start(sys.executable, ["-m", "agentsim", "mcp"])

    @staticmethod
    def _package_root():
        import pathlib
        return pathlib.Path(__file__).resolve().parents[2]

    def shutdown(self) -> None:
        """优雅退出：关 stdin（EOF）让服务器收尾，等不到再杀。"""
        if self._proc is None:
            return
        if self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.closeWriteChannel()
            if not self._proc.waitForFinished(5000):
                self._proc.kill()
                self._proc.waitForFinished(2000)
        self._proc = None

    def is_alive(self) -> bool:
        return (self._proc is not None
                and self._proc.state() != QProcess.ProcessState.NotRunning)

    # ------------------------------------------------------------ 请求

    def request(self, method: str, params: dict | None, cb) -> int:
        """发一个 JSON-RPC 请求，响应到达时 cb(resp_dict)。返回请求 id。"""
        self._next_id += 1
        rid = self._next_id
        self._pending.register(rid, cb)
        req = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            req["params"] = params
        self._write(req)
        return rid

    def call_tool(self, name: str, args: dict | None, on_done) -> int:
        """tools/call 封装。on_done(is_error, data)——见 tool_response。"""
        def cb(resp):
            err, data = tool_response(resp)
            on_done(err, data)
        return self.request("tools/call", {"name": name, "arguments": args or {}}, cb)

    # ------------------------------------------------------------ 内部

    def _write(self, obj: dict) -> None:
        if self._proc is None:
            return
        data = json.dumps(obj, ensure_ascii=False) + "\n"
        self._proc.write(data.encode("utf-8"))

    def _on_ready_read(self) -> None:
        chunk = bytes(self._proc.readAllStandardOutput()).decode("utf-8",
                                                                 errors="replace")
        for line in self._codec.feed(chunk):
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue                  # 服务器输出的坏行直接忽略
            if isinstance(resp, dict):
                self._pending.dispatch(resp)

    def _on_finished(self, code, status) -> None:
        if code != 0 or status != QProcess.ExitStatus.NormalExit:
            self.backendDied.emit(f"后端进程退出 code={code} status={status}")
