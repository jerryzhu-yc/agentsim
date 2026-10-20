"""场景生成 Agent 测试：闭环纯本地可测（HTTP 打桩，不打真 API）。

覆盖：
- extract_yaml：围栏剥离 / 无围栏容错
- system prompt：schema 说明 + zoo 模型清单注入
- generate_scene：成功路径 / 校验失败重试 / 重试耗尽 / 未配置
- resolve_zoo_models：zoo:// 替换 + 未知 key 报错
"""
import pytest
import yaml

from agentsim.agent import AgentConfig
from agentsim.agent.scene_agent import (SceneGenError, extract_yaml,
                                        generate_scene, resolve_zoo_models,
                                        scene_system_prompt)
from agentsim.errors import WorldError


# ---------------------------------------------------------------- extract_yaml

def test_extract_yaml_fenced():
    text = "好的，这是场景：\n```yaml\nname: t\nfloor: {}\n```\n如需调整请说。"
    assert extract_yaml(text) == "name: t\nfloor: {}"


def test_extract_yaml_bare_fallback():
    """无围栏：从首个 name: 行截到文末（容错 LLM 不守规矩）。"""
    text = "以下是 YAML：\nname: t\nfloor: {}\n"
    assert extract_yaml(text) == "name: t\nfloor: {}"


# ---------------------------------------------------------------- prompt

def test_system_prompt_has_schema_and_zoo():
    s = scene_system_prompt()
    assert "zoo://" in s                       # 模型引用约定
    assert "unitree_go2" in s                  # 模型库清单已注入
    assert "static" in s and "半尺寸" in s     # schema 关键字段
    assert "只输出一个 YAML" in s               # 生成规则


# ---------------------------------------------------------------- resolve_zoo

def test_resolve_zoo_models(monkeypatch, tmp_path):
    d = {"robots": [
        {"name": "r1", "model": "zoo://unitree_go2", "pos": [0, 0, 0.4]},
        {"name": "r2", "model": "/abs/path/bot.xml"},
    ]}
    seen = []
    monkeypatch.setattr("agentsim.zoo.ensure_model",
                        lambda key: (seen.append(key),
                                     str(tmp_path / "go2.xml"))[1])
    resolve_zoo_models(d, on_download=lambda k: seen.append(f"dl:{k}"))
    assert d["robots"][0]["model"] == str(tmp_path / "go2.xml")
    assert d["robots"][1]["model"] == "/abs/path/bot.xml"   # 非 zoo:// 不动
    assert seen == ["dl:unitree_go2", "unitree_go2"]        # 回调先于下载


def test_resolve_zoo_models_unknown_key():
    d = {"robots": [{"name": "r", "model": "zoo://nope_bot"}]}
    with pytest.raises(WorldError, match="nope_bot"):
        resolve_zoo_models(d)


# ---------------------------------------------------------------- generate_scene

_GOOD = """```yaml
name: ai_scene
floor: {size: [3, 3]}
scene:
  objects:
    - {name: cube, type: box, size: [0.05, 0.05, 0.05], pos: [0.5, 0, 0.05],
       color: "#EF4444"}
robots: []
```"""

_BAD = """```yaml
name: bad_scene
scene:
  objects:
    - {name: cube, type: hexagon, size: [1, 1]}
```"""


class _FakeHTTP:
    """按序吐响应，记录每次收到的完整 messages（校验错误回喂用）。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.all_messages: list[list[dict]] = []

    def __call__(self, cfg, messages):
        self.all_messages.append([dict(m) for m in messages])
        return self.responses.pop(0)


def test_generate_scene_success(monkeypatch):
    fake = _FakeHTTP([{"choices": [{"message": {"content": _GOOD}}]}])
    monkeypatch.setattr("agentsim.agent.scene_agent._post_chat", fake)
    cfg = AgentConfig(base_url="http://x/v1", model="m")
    text, _note = generate_scene("放一个红方块", cfg)
    d = yaml.safe_load(text)
    assert d["name"] == "ai_scene"
    assert d["scene"]["objects"][0]["color"] == "#EF4444"
    assert len(fake.all_messages) == 1          # 一次成功，无重试


def test_generate_scene_retry_on_bad_yaml(monkeypatch):
    """第一次坏 YAML -> 错误回喂 -> 第二次好 -> 成功。"""
    fake = _FakeHTTP([
        {"choices": [{"message": {"content": _BAD}}]},
        {"choices": [{"message": {"content": _GOOD}}]},
    ])
    monkeypatch.setattr("agentsim.agent.scene_agent._post_chat", fake)
    cfg = AgentConfig(base_url="http://x/v1", model="m")
    text, _ = generate_scene("先错后对", cfg)
    assert yaml.safe_load(text)["name"] == "ai_scene"
    assert len(fake.all_messages) == 2
    # 第二轮 messages：含 assistant 坏 YAML + user 错误反馈（回喂闭环）
    second = fake.all_messages[1]
    roles = [m["role"] for m in second]
    assert "assistant" in roles
    feedback = next(m["content"] for m in second if m["role"] == "user"
                   and "校验失败" in m["content"])
    assert "hexagon" in feedback                # 错误信息带具体原因


def test_generate_scene_exhausted(monkeypatch):
    """一直坏 -> 重试耗尽抛 SceneGenError（含末次错误信息）。"""
    fake = _FakeHTTP([{"choices": [{"message": {"content": _BAD}}]}] * 5)
    monkeypatch.setattr("agentsim.agent.scene_agent._post_chat", fake)
    cfg = AgentConfig(base_url="http://x/v1", model="m")
    with pytest.raises(SceneGenError, match="重试"):
        generate_scene("永远失败", cfg, max_retries=3)
    assert len(fake.all_messages) == 3


def test_generate_scene_unconfigured():
    with pytest.raises(SceneGenError, match="未配置"):
        generate_scene("x", AgentConfig())


def test_generate_scene_zoo_model_resolved(monkeypatch, tmp_path):
    """LLM 写 zoo://key -> 本地替换为绝对路径后再校验（闭环含下载回调）。"""
    d_text = """```yaml
name: with_bot
floor: {}
robots:
  - {name: bot, model: "zoo://unitree_go2", pos: [0, 0, 0.4]}
```"""
    monkeypatch.setattr("agentsim.zoo.ensure_model",
                        lambda key: str(tmp_path / "go2.xml"))
    # 校验需要真实可编译模型：给一个最小 MJCF（带执行器）
    (tmp_path / "go2.xml").write_text(
        "<mujoco><worldbody>"
        '<body name="b" pos="0 0 0.3"><geom type="box" size=".1 .1 .02"/>'
        '<joint name="j" type="slide" axis="0 0 1"/>'
        "</body></worldbody>"
        '<actuator><motor name="m" joint="j"/></actuator></mujoco>',
        encoding="utf-8")
    monkeypatch.setattr("agentsim.agent.scene_agent._post_chat",
                        _FakeHTTP([{"choices": [{"message": {
                            "content": d_text}}]}]))
    cfg = AgentConfig(base_url="http://x/v1", model="m")
    text, _ = generate_scene("一只机器狗", cfg)
    out = yaml.safe_load(text)
    assert out["robots"][0]["model"] == str(tmp_path / "go2.xml")
