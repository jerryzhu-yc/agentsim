"""场景生成 Agent：自然语言需求 -> 世界 YAML（生成-校验-重试闭环）。

与通用 Agent 控制台（worker.py 的工具循环）不同，这里是"单轮生成 +
本地校验 + 错误回喂"模式：LLM 只做一件事——按 schema 写 YAML；校验
（parse_world + build_world）在本地完成，失败把可读错误喂回去重来，
最多 max_retries 次。闭环全部在本地可测（不打真 API 也全覆盖）。

机器人模型引用：LLM 写 `zoo://key`（模型库 key），生成后本地替换成
menagerie 缓存绝对路径（ensure_model，首次联网下载一次）——LLM 不必
知道任何文件路径。
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

import yaml

from agentsim.agent.config import AgentConfig
from agentsim.errors import WorldError

# ---------------------------------------------------------------- prompt

_SCHEMA_DOC = """世界 YAML schema（字段语义与 MJCF 对齐）：

顶层：
  name: string（必填，世界名）
  timestep: 秒（默认 0.002）  ctrl_period: 每 N 个 tick 一次控制（默认 32）
  gravity: [x,y,z]（默认 [0,0,-9.81]）
  impratio: 摩擦/法向刚度比（大质量机器人 50-100，默认 1.0）
  integrator: euler|rk4|implicit|implicitfast（默认 euler；硬接触用 implicitfast）

scene.floor: {size: [半长,半宽]（默认 2x2）, texture: grid, friction: 1.0}
scene.lights: [{pos: [x,y,z]}]（可选，默认自带一盏）
scene.objects: 物体列表，字段：
  name: string（必填，全场景唯一）
  type: box|sphere|cylinder（必填）
  size: 半尺寸列表（box 3 个=半长半宽半高；sphere 1 个=半径；cylinder 2 个=半径半高）
  pos: [x,y,z] 中心位置（默认 0,0,0；注意 z 要 >= 半高 才不陷地）
  quat: [w,x,y,z]（可选，wxyz）
  mass: kg（可选，默认按尺寸估）
  color: "#RRGGBB"（默认 #9CA3AF）
  static: true/false（true=固定场景体如台子/墙/斜坡，无 freejoint 不可推动）
  friction: 滑动摩擦（默认 1.0）

robots: 机器人列表，字段：
  name: string（必填，唯一）
  model: 模型引用（必填）——写 "zoo://<key>"，key 取下方模型库清单；
         用户明确给了模型文件路径时写绝对路径（.xml MJCF / .urdf URDF）
  pos: [x,y,z] root 世界位置（默认 0,0,0）
  quat: [w,x,y,z]（可选）
  controller: 控制器 .py 路径（可选；没有控制器机器人不动，纯场景摆放可不写）
  sensors: 传感器列表（可选）[{name, type: encoder|imu|tactile|joint_torque|camera,
            joint 或 body（encoder/joint_torque 用 joint，其余用 body）， pos}]
  gripper: {model: 夹爪 key（onrobot_2fg7|robotiq_2f85|wsg50）}（可选，仅机械臂）

task: 任务（可选）
  success: {body: 物体或机器人名, zone: {min: [x,y,z], max: [x,y,z]}}（min 逐分量 < max）
  reward: 列表，每条单键字典：
    {distance: {a, b, weight（默认 -1）}}
    {zone: {body, zone: {min,max}, weight（默认 1）}}
    {contact: {a, b, weight（默认 1）}}
  max_ticks: 正整数（可选）

runtime: {realtime: bool, controller_timeout: ms, timeout_policy: last_ctrl|pause}（少用）
"""

_RULES = """生成规则：
1. 只输出一个 YAML 文档（可用 ```yaml 围栏包裹），不要任何解释文字。
2. 拿不准的字段全部省略（用默认值），保持 YAML 尽量短。
3. 物理常识：物体 z >= 半高（贴地）；机器人物体不要重叠；台子/墙/坡用
   static: true；小物体质量 0.1-1kg，机器人附近别摆太挤。
4. 任务 reward/success 引用的 body 必须是物体名或机器人名（真实存在）。
5. floor.size 要罩住全部物体和机器人（留 0.5m 余量）。
"""


def _zoo_catalog() -> str:
    """模型库清单（注入 prompt）：key -> 标题[类别] 描述。"""
    from agentsim.zoo import MODELS
    lines = ["可用机器人模型（robots[].model 写 zoo://key）："]
    for key, m in MODELS.items():
        lines.append(f"- zoo://{key}：{m['title']} [{m['cat']}]——{m['desc']}")
    return "\n".join(lines)


def scene_system_prompt() -> str:
    return ("你是 AgentSim 机器人仿真平台的场景生成器。用户用自然语言"
            "描述需求，你输出对应的世界 YAML（唯一交付物）。\n\n"
            + _SCHEMA_DOC + "\n" + _zoo_catalog() + "\n\n" + _RULES)


# ---------------------------------------------------------------- 纯函数

def extract_yaml(text: str) -> str:
    """从 LLM 回复提取 YAML：剥 ```yaml / ``` 围栏与围栏外杂文字。

    有围栏取第一块围栏内容；无围栏时取首个 `name:` 行到文末（容错）。
    """
    m = re.search(r"```(?:ya?ml)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    idx = text.find("name:")
    return text[idx:].strip() if idx >= 0 else text.strip()


def resolve_zoo_models(d: dict, on_download=None) -> dict:
    """把 robots[].model 的 zoo://key 替换成 menagerie 缓存绝对路径。

    on_download(key)：下载前回调（首次联网下载可能数秒~数分钟）。
    就地修改并返回 d（链式）。未知 key 抛 WorldError（可读）。
    """
    from agentsim.zoo import MODELS, ensure_model
    for rd in d.get("robots") or []:
        model = rd.get("model", "")
        if not str(model).startswith("zoo://"):
            continue
        key = str(model)[len("zoo://"):]
        if key not in MODELS:
            raise WorldError(
                f"AI 引用了未知模型库 key '{key}'，可用: {sorted(MODELS)}")
        if on_download:
            on_download(key)
        rd["model"] = str(ensure_model(key))
    return d


# ---------------------------------------------------------------- 生成循环

class SceneGenError(Exception):
    """生成失败（重试耗尽/网络错误），message 面向用户可读。"""


def generate_scene(requirement: str, cfg: AgentConfig,
                   max_retries: int = 3,
                   on_download=None) -> tuple[str, str]:
    """需求文本 -> (世界 YAML 文本, LLM 附带说明)。

    闭环：LLM 生成 -> 提取 YAML -> zoo:// 替换 -> parse + build 校验
    -> 失败把错误喂回重试。全部尝试失败抛 SceneGenError（含末次错误）。
    on_download(key)：首次下载模型前的回调（进度提示）。
    """
    if not cfg.usable:
        raise SceneGenError("未配置 LLM（先在 Agent 面板配置 base_url/model）")
    messages = [
        {"role": "system", "content": scene_system_prompt()},
        {"role": "user", "content": requirement},
    ]
    last_err = ""
    note = ""
    for _ in range(max_retries):
        resp = _post_chat(cfg, messages)
        note = (resp.get("choices") or [{}])[0].get("message", {}).get(
            "content", "") or ""
        text = extract_yaml(note)
        try:
            d = yaml.safe_load(text)
            if not isinstance(d, dict):
                raise WorldError(f"顶层不是映射: {type(d).__name__}")
            resolve_zoo_models(d, on_download)
            _validate(d)
            out = yaml.safe_dump(d, allow_unicode=True, sort_keys=False)
            return out, note
        except Exception as e:
            last_err = f"{e}"
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": f"生成的 YAML 校验失败：{e}\n"
                           f"请修正后重新输出完整 YAML（仍然只输出 YAML）。",
            })
    raise SceneGenError(f"重试 {max_retries} 次仍失败，末次错误: {last_err}")


def _validate(d: dict) -> None:
    """schema + 编译双重校验（编译能抓住 parse 抓不住的引用/物理错误）。"""
    from agentsim.world.builder import build_world
    from agentsim.world.schema import parse_world
    spec = parse_world(d)          # base_dir 默认 "."，model 均为绝对路径
    build_world(spec)


def _post_chat(cfg: AgentConfig, messages: list[dict]) -> dict:
    """POST chat/completions（无 tools，纯文本生成）。出错抛 SceneGenError。"""
    body = json.dumps({"model": cfg.model, "messages": messages,
                       "stream": False, "temperature": 0.2},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(cfg.chat_url(), data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if cfg.api_key:
        req.add_header("Authorization", f"Bearer {cfg.api_key}")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:300]
        except Exception:
            pass
        raise SceneGenError(f"LLM 服务 HTTP {e.code}: {detail or e.reason}") from e
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        raise SceneGenError(f"LLM 服务不可达/响应异常: {e}") from e
