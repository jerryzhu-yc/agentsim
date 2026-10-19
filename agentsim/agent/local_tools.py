"""本地文件工具：write_file / read_file，范围限定当前世界目录。

AI Agent 自己写场景 YAML / 控制器代码后，再调 load_world + run 验证——
与 GUI 用户的工作流同构。无 ZMQ 中转：直接读写磁盘，回复给 LLM。
"""
from __future__ import annotations

import pathlib

from agentsim.errors import WorldError


class LocalFiles:
    """管理本地文件工具的根目录（= 当前世界 YAML 所在目录）。

    路径解析：
    - 绝对路径：必须在 root 内（防止 AI 越界写 ~/.bashrc 等）
    - 相对路径：按 root 拼接
    """

    def __init__(self, root: pathlib.Path | None = None):
        self._root = root.resolve() if root else None

    @property
    def root(self) -> pathlib.Path | None:
        return self._root

    def set_root(self, root: pathlib.Path | None) -> None:
        self._root = root.resolve() if root else None

    def _resolve(self, path: str) -> pathlib.Path:
        if self._root is None:
            raise WorldError("LocalFiles", "未设置世界目录，无法读写文件")
        p = pathlib.Path(path)
        full = p if p.is_absolute() else (self._root / p)
        full = full.resolve()
        try:
            full.relative_to(self._root)        # 越界抛 ValueError
        except ValueError:
            raise WorldError(
                "LocalFiles",
                f"路径 '{path}' 越界：只能在 {self._root} 内读写") from None
        return full

    def write_file(self, path: str, content: str) -> dict:
        full = self._resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        try:
            rel = full.relative_to(self._root)
        except ValueError:
            rel = full
        return {"path": str(rel), "bytes": len(content.encode("utf-8")),
                "abs": str(full)}

    def read_file(self, path: str) -> dict:
        full = self._resolve(path)
        try:
            content = full.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise WorldError(
                "LocalFiles",
                f"文件不存在: {path}（在 {self._root} 下）") from None
        return {"path": path, "content": content,
                "bytes": len(content.encode("utf-8"))}


_LOCAL_DISPATCH = {"write_file", "read_file"}


def dispatch_local(files: LocalFiles, name: str, args: dict) -> dict | None:
    """分发本地工具调用。非本地工具名返回 None（让 worker 走 MCP）。"""
    if name not in _LOCAL_DISPATCH:
        return None
    if name == "write_file":
        return files.write_file(args.get("path", ""), args.get("content", ""))
    if name == "read_file":
        return files.read_file(args.get("path", ""))
    return None
