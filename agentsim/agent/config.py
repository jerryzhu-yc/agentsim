"""Agent 配置持久化：~/.agentsim/agent.json（chmod 600 护 api_key）。

OpenAI 兼容端点覆盖 OpenAI/DeepSeek/GLM/Ollama/vLLM 全部：一个 base_url +
api_key + model 名即可。空 api_key 视为本地 Ollama 等无需鉴权的服务。
"""
from __future__ import annotations

import json
import os
import pathlib
import stat

CONFIG_DIR = pathlib.Path.home() / ".agentsim"
CONFIG_PATH = CONFIG_DIR / "agent.json"


class AgentConfig:
    """OpenAI 兼容 chat/completions 配置（同 DeepSeek/GLM/Ollama 字段）。"""

    def __init__(self, base_url: str = "", api_key: str = "",
                 model: str = "", extra: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.extra = dict(extra) if extra else {}

    @property
    def usable(self) -> bool:
        """缺 base_url/model 不可用；api_key 空允许（本地 Ollama 等）。"""
        return bool(self.base_url and self.model)

    def chat_url(self) -> str:
        if not self.base_url:
            return ""
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url.rstrip("/") + "/chat/completions"

    def to_dict(self) -> dict:
        d = {"base_url": self.base_url, "model": self.model}
        if self.api_key:
            d["api_key"] = self.api_key
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AgentConfig":
        return cls(base_url=d.get("base_url", ""),
                   api_key=d.get("api_key", ""),
                   model=d.get("model", ""),
                   extra=d.get("extra"))


def load_config() -> AgentConfig:
    """从 ~/.agentsim/agent.json 读配置（无文件返回空配置）。"""
    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return AgentConfig()
    try:
        return AgentConfig.from_dict(json.loads(text))
    except (json.JSONDecodeError, TypeError):
        return AgentConfig()


def save_config(cfg: AgentConfig) -> None:
    """写 ~/.agentsim/agent.json（目录与文件都 chmod 600 护 api_key）。"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8")
    # 显式收紧权限（Windows 忽略 chmod，但 Linux 上护住 api_key）
    try:
        os.chmod(CONFIG_PATH,
                 stat.S_IRUSR | stat.S_IWUSR)        # 0600
    except OSError:
        pass
