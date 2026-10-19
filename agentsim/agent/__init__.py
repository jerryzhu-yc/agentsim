"""M5-P4 Agent 控制台：OpenAI 兼容 chat/completions + 工具桥接。

设计：
- config.py：AgentConfig 持久化 ~/.agentsim/agent.json（chmod 600 护密钥）
- bridge.py：MCP manifest -> OpenAI tools 转换 + system prompt 注入
- local_tools.py：write_file / read_file（限定当前世界目录，AI 生成场景/
  控制器后自己 load_world+run 验证，与 GUI 同款工作流）
- worker.py：QThread 跑 HTTP + 工具循环，发信号回 UI（不阻塞）
"""
from agentsim.agent.config import AgentConfig, load_config, save_config
from agentsim.agent.bridge import (local_tools_manifest, to_openai_tools,
                                    system_prompt)
from agentsim.agent.local_tools import LocalFiles, dispatch_local
from agentsim.agent.worker import AgentWorker

__all__ = ["AgentConfig", "load_config", "save_config",
           "local_tools_manifest", "to_openai_tools", "system_prompt",
           "LocalFiles", "dispatch_local", "AgentWorker"]
