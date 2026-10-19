"""M5-P4 验收：Agent 控制台桥接（纯函数 + 配置持久化，不打真 API）。"""
import os
import stat

import pytest


# ================================================================ T7-a 配置持久化

def test_config_roundtrip(tmp_path, monkeypatch):
    """配置写读往返；目录与文件权限 0600 护 api_key。"""
    from agentsim.agent import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "CONFIG_DIR", tmp_path / ".agentsim")
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / ".agentsim" / "agent.json")

    cfg = cfg_mod.AgentConfig(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test-123456",
        model="deepseek-chat")
    assert cfg.usable and cfg.chat_url().endswith("/chat/completions")
    cfg_mod.save_config(cfg)
    p = tmp_path / ".agentsim" / "agent.json"
    assert p.exists()
    # Linux 上权限必须 0600（护 api_key；Windows chmod 静默忽略）
    if os.name == "posix":
        mode = stat.S_IMODE(p.stat().st_mode)
        assert mode == 0o600
    loaded = cfg_mod.load_config()
    assert loaded.base_url == cfg.base_url
    assert loaded.api_key == cfg.api_key
    assert loaded.model == cfg.model
    # 空 api_key 允许（本地 Ollama 等）
    cfg2 = cfg_mod.AgentConfig(base_url="http://localhost:11434/v1", model="llama3")
    assert cfg2.usable and not cfg2.api_key
    # 缺 base_url 不可用
    assert not cfg_mod.AgentConfig(model="x").usable
    # 已有 /chat/completions 后缀不重复
    cfg3 = cfg_mod.AgentConfig(base_url="https://x/v1/chat/completions", model="m")
    assert cfg3.chat_url() == "https://x/v1/chat/completions"


# ================================================================ T7-b 工具 manifest 转换

def test_to_openai_tools_shape():
    """MCP manifest -> OpenAI tools 数组（function calling 格式）。"""
    from agentsim.agent.bridge import to_openai_tools
    mcp = [{"name": "run", "description": "运行仿真",
            "inputSchema": {"type": "object",
                            "properties": {"ticks": {"type": "integer"}}}}]
    tools = to_openai_tools(mcp)
    assert tools == [{"type": "function",
                      "function": {"name": "run", "description": "运行仿真",
                                   "parameters": mcp[0]["inputSchema"]}}]


def test_local_tools_manifest_fields():
    """本地工具 write_file/read_file 的参数 schema 完整。"""
    from agentsim.agent.bridge import local_tools_manifest
    names = {t["name"] for t in local_tools_manifest()}
    assert names == {"write_file", "read_file"}
    wf = next(t for t in local_tools_manifest() if t["name"] == "write_file")
    assert set(wf["inputSchema"]["properties"]) == {"path", "content"}
    assert set(wf["inputSchema"]["required"]) == {"path", "content"}


def test_system_prompt_lists_all_tools():
    """系统 prompt 列出 MCP 10 + 本地 2，且含当前世界名 + 工作流建议。"""
    from agentsim.agent.bridge import system_prompt, local_tools_manifest
    mcp_tools = [{"name": "zoo_create", "description": "生成世界"},
                 {"name": "load_world", "description": "加载世界"},
                 {"name": "run", "description": "运行"}]
    s = system_prompt("my_world", mcp_tools)
    assert "my_world" in s
    for t in mcp_tools:
        assert t["name"] in s
    for t in local_tools_manifest():
        assert t["name"] in s
    assert "zoo_create" in s and "write_file" in s
    assert "工作流" in s


# ================================================================ T7-c tool_calls 解析

def test_extract_tool_calls_and_content():
    """OpenAI 响应解析：tool_calls 与 content 文本双向覆盖。"""
    from agentsim.agent.bridge import extract_tool_calls, content_text
    # 有 tool_calls
    resp = {"choices": [{"message": {
        "content": "好的，先加载世界",
        "tool_calls": [{"id": "c1", "function": {
            "name": "load_world",
            "arguments": '{"yaml_path": "/tmp/w.yaml"}'}}]}}]}
    calls = extract_tool_calls(resp)
    assert len(calls) == 1
    assert calls[0]["name"] == "load_world"
    assert calls[0]["arguments"] == {"yaml_path": "/tmp/w.yaml"}
    assert content_text(resp) == "好的，先加载世界"
    # 无 tool_calls：仅 content
    resp2 = {"choices": [{"message": {"content": "已处理完"}}]}
    assert extract_tool_calls(resp2) == []
    assert content_text(resp2) == "已处理完"
    # 异常结构不崩
    assert extract_tool_calls({}) == []
    assert content_text({}) == ""
    # arguments 不是 JSON 字符串（dict 直接）也兼容
    resp3 = {"choices": [{"message": {"tool_calls": [{"function": {
        "name": "run", "arguments": {"ticks": 10}}}]}},]}
    assert extract_tool_calls(resp3)[0]["arguments"] == {"ticks": 10}


# ================================================================ T7-d 本地文件工具

def test_local_files_in_world_dir(tmp_path):
    """write_file/read_file 限定世界目录：相对/绝对路径都解析到根内。"""
    from agentsim.agent.local_tools import LocalFiles, dispatch_local
    from agentsim.errors import WorldError
    root = tmp_path / "world_dir"
    root.mkdir()
    f = LocalFiles(root)
    # 相对路径写文件
    r = f.write_file("scene.yaml", "name: test\n")
    assert (root / "scene.yaml").read_text() == "name: test\n"
    assert r["bytes"] == len("name: test\n")
    # 读文件
    r = f.read_file("scene.yaml")
    assert r["content"] == "name: test\n"
    # 绝对路径在根内允许
    abs_p = root / "sub" / "ctrl.py"
    r = f.write_file(str(abs_p), "print('x')\n")
    assert abs_p.exists()
    # 越界写：抛 WorldError
    with pytest.raises(WorldError):
        f.write_file(str(tmp_path / "outside.yaml"), "x")
    # 未设置 root：抛错
    f2 = LocalFiles()
    with pytest.raises(WorldError):
        f2.write_file("a", "b")
    # dispatch_local 分发：本地工具名走本地，否则返回 None
    assert dispatch_local(f, "write_file", {"path": "a.txt", "content": "x"}).get("bytes") == 1
    assert dispatch_local(f, "read_file", {"path": "scene.yaml"}).get("content") == "name: test\n"
    assert dispatch_local(f, "run", {}) is None      # MCP 工具走 MCP


def test_local_files_read_nonexistent(tmp_path):
    """读不存在文件：抛 WorldError 列出根目录。"""
    from agentsim.agent.local_tools import LocalFiles
    from agentsim.errors import WorldError
    f = LocalFiles(tmp_path)
    with pytest.raises(WorldError) as e:
        f.read_file("ghost.yaml")
    assert "ghost.yaml" in str(e.value)
