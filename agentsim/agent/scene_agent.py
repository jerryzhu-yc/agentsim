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
import pathlib
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
   static: true；小物体质量 0.1-1kg，机器人附近别摆太挤；带机械臂或硬
   接触的世界写上 integrator: implicitfast 与 impratio: 10（高刚度伺服
   稳定性必需，否则 attach 时被默认 euler 覆盖会发散）。
4. 任务 reward/success 引用的 body 必须是物体名或机器人名（真实存在）。
5. floor.size 要罩住全部物体和机器人（留 0.5m 余量）。
6. 绝不输出 MJCF/XML，不引用任何 mesh/纹理文件（.obj/.dae/.stl/.png）——
   物体只用 box/sphere/cylinder 基本体表达，机器人只用 zoo:// 模型库。
"""


def _zoo_catalog() -> str:
    """模型库清单（注入 prompt）：key -> 标题[类别] 描述。"""
    from agentsim.zoo import MODELS
    lines = ["可用机器人模型（robots[].model 写 zoo://key）："]
    for key, m in MODELS.items():
        lines.append(f"- zoo://{key}：{m['title']} [{m['cat']}]——{m['desc']}")
    return "\n".join(lines)


def _gripper_catalog() -> str:
    """夹爪清单（注入 prompt，从 GRIPPERS 动态生成，不再硬编码）。"""
    from agentsim.grippers import GRIPPERS
    if not GRIPPERS:
        return ""
    lines = ["可用夹爪（robots[].gripper.model 写 key；仅机械臂可挂）："]
    for key, g in GRIPPERS.items():
        arms = ", ".join(sorted(g.get("mounts", {})))
        lines.append(f"- {key}：{g['title']}（已标定臂: {arms}）")
    return "\n".join(lines)


# few-shot 示例：一个紧凑典型世界，比纯 schema 文档更能约束格式。
# 字段全部合法（pos z=半高贴地、floor 罩住、reward/success 引用真实名）。
_EXAMPLE = """\
name: demo
integrator: implicitfast
impratio: 10
scene:
  floor: {size: [3, 3]}
  objects:
    - {name: red_cube, type: box, size: [0.05, 0.05, 0.05], pos: [0.5, 0, 0.05], color: "#EF4444", mass: 0.5}
robots:
  - name: arm
    model: zoo://ur5e
    pos: [-0.5, 0, 0]
    gripper: {model: onrobot_2fg7}
task:
  success: {body: red_cube, zone: {min: [0.8, -0.1, 0], max: [1.0, 0.1, 0.2]}}
  reward:
    - {distance: {a: red_cube, b: arm, weight: -1}}
"""


# ---------------------------------------------------------------- manifest 按需注入

_MANIFEST_DIR = pathlib.Path.home() / ".agentsim" / "zoo" / "manifests"


def _model_manifest(key: str) -> dict | None:
    """某模型的设备清单（joint/body/actuator 短名），缓存到
    ~/.agentsim/zoo/manifests/<key>.json。返回 None 表示提取失败
    （不阻塞生成，回退到无 manifest 的回喂试错）。

    用 MjModel.from_xml_path 直接编译 menagerie 模型，拿到的名字是无
    机器人前缀的原始短名——与 builder 注入传感器/task 引用时写的短名
    一致（attach 时自动补 "<robot>/" 前缀）。
    """
    cache = _MANIFEST_DIR / f"{key}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass  # 缓存损坏，重新提取
    try:
        import mujoco
        from agentsim.zoo import ensure_model
        xml = ensure_model(key)
        m = mujoco.MjModel.from_xml_path(str(xml))
        joints, bodies, actuators = [], [], []
        for i in range(m.njnt):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n and not n.startswith("world"):
                joints.append(n)
        for i in range(m.nbody):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
            if n and n != "world":
                bodies.append(n)
        for i in range(m.nu):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if n:
                actuators.append(n)
        d = {"key": key, "joints": sorted(joints),
             "bodies": sorted(bodies), "actuators": sorted(actuators)}
        try:
            _MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(d, ensure_ascii=False),
                             encoding="utf-8")
        except OSError:
            pass  # 缓存写失败不影响生成
        return d
    except Exception:
        return None


# 需求识别时跳过的品牌/虚词（太短或非型号词）
_BRAND_WORDS = {"unitree", "onrobot", "robotiq", "anybotics", "franka",
                "uestc", "the", "of", "a", "and", "with"}


def _detect_keys(requirement: str) -> list[str]:
    """从需求文本识别提到的 zoo key（key 下划线拆分 + title 显著词匹配）。

    命中即按需注入对应 manifest；一个都没命中时不注入（维持现状靠回喂
    试错）。纯中文需求（如"四足机器狗"无英文型号）命中不了，回退到
    无 manifest 的回喂路径——builder 校验失败时会带候选名，仍可收敛。
    """
    from agentsim.zoo import MODELS
    req = requirement.lower()
    hits = []
    for key, m in MODELS.items():
        needles = set()
        for part in key.split("_"):          # unitree_go2 -> go2
            if len(part) >= 3 and part.lower() not in _BRAND_WORDS:
                needles.add(part.lower())
        for w in m.get("title", "").split():     # "Unitree Go2" -> go2
            wl = w.lower()
            if len(wl) >= 3 and wl not in _BRAND_WORDS:
                needles.add(wl)
        if any(n in req for n in needles):
            hits.append(key)
    return hits


def _manifest_appendix(keys: list[str]) -> str:
    """按需 manifest 附录文本（只含命中的 key 的设备清单）。"""
    lines = []
    for key in keys:
        d = _model_manifest(key)
        if not d:
            continue
        lines.append(
            f"\n## {key} 设备清单（sensors 的 joint/body、task 的 body "
            f"引用都写下面的短名，builder 自动补机器人前缀）")
        lines.append(f"  关节(joints): {', '.join(d['joints']) or '（无）'}")
        lines.append(f"  body: {', '.join(d['bodies']) or '（无）'}")
        lines.append(f"  执行器(actuators): {', '.join(d['actuators']) or '（无）'}")
    return "\n".join(lines)


def scene_system_prompt(requirement: str = "") -> str:
    """组装 system prompt。requirement 非空时按需注入命中模型的 manifest
    附录（joint/body/actuator 短名），让 LLM 写 sensors/task 引用不再瞎猜。

    无参调用（requirement=""）行为与旧版一致：只给 schema+模型清单+规则，
    不含 manifest 附录——向后兼容。
    """
    parts = ["你是 AgentSim 机器人仿真平台的场景生成器。用户用自然语言"
             "描述需求，你输出对应的世界 YAML（唯一交付物）。\n"
             "（覆盖指令：忽略你此前收到的任何其他系统提示或角色设定——"
             "包括 MJCF/XML 生成器人设——一律以本提示为准。）\n\n",
             _SCHEMA_DOC, "\n", _zoo_catalog(), "\n"]
    gc = _gripper_catalog()
    if gc:
        parts.append(gc + "\n")
    parts.append(_RULES)
    parts.append("\n\n## 示例（参考格式，不要照搬内容）\n```yaml\n"
                 + _EXAMPLE + "```")
    if requirement:
        app = _manifest_appendix(_detect_keys(requirement))
        if app:
            parts.append("\n" + app)
    return "".join(parts)


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

    实现已下沉到 world/schema.py（load_world 与 MCP 路径也用同一份），
    此处 re-export 保持原调用兼容（scene_agent 的存量引用/测试不破）。
    语义见 agentsim.world.schema.resolve_zoo_models。
    """
    from agentsim.world.schema import resolve_zoo_models as _resolve
    return _resolve(d, on_download)


# ---------------------------------------------------------------- 生成循环

def _looks_like_mjcf(text: str) -> bool:
    """识别 LLM 忽略提示词输出的原生 MJCF/XML（典型症状：mesh 文件引用）。"""
    return ("<mujoco" in text.lower()
            or bool(re.search(r"\.(obj|dae|stl)\b", text, re.IGNORECASE)))


class SceneGenError(Exception):
    """生成失败（重试耗尽/网络错误），message 面向用户可读。"""


class SceneGenSession:
    """多轮场景生成会话：首轮 generate + 后续 refine（增量修改）。

    messages 历史保留，refine 把当前 YAML + 修改要求追加进历史，要求 LLM
    输出完整修正版 YAML，仍走完整校验回喂闭环。每轮成功后更新
    current_yaml，作为下一轮 refine 的基准。

    单轮便捷入口 generate_scene() 内部即新建一个 Session 跑 generate，
    行为与旧版完全一致（存量调用不破）。GUI 微调循环持有同一个 Session。
    """

    def __init__(self, cfg: AgentConfig, max_retries: int = 3,
                 on_download=None):
        if not cfg.usable:
            raise SceneGenError("未配置 LLM（先在 Agent 面板配置 base_url/model）")
        self.cfg = cfg
        self.max_retries = max_retries
        self.on_download = on_download
        self.messages: list[dict] = []
        self.current_yaml: str = ""

    def generate(self, requirement: str) -> tuple[str, str]:
        """首轮生成：按需求组装 system prompt（含按需 manifest 注入），
        跑生成-校验-回喂闭环，成功后留存 messages 与 current_yaml。"""
        self.messages = [
            {"role": "system", "content": scene_system_prompt(requirement)},
            {"role": "user", "content": requirement},
        ]
        out, note = self._gen_loop()
        self.current_yaml = out
        return out, note

    def refine(self, change: str) -> tuple[str, str]:
        """增量修改：把当前 YAML 当作 assistant 上一轮输出，把修改要求
        作为新 user 消息追加，要求 LLM 输出完整修正版，仍走校验回喂。

        change：自然语言修改要求（如"把方块改成红色，再加一个球"）。
        未生成过（current_yaml 空）时报错——先 generate 再 refine。
        """
        if not self.current_yaml:
            raise SceneGenError("尚未生成过世界，请先调用 generate()")
        self.messages.append({"role": "assistant", "content": self.current_yaml})
        self.messages.append({
            "role": "user",
            "content": f"在以上世界基础上做如下修改，输出完整的修正版 YAML"
                       f"（仍然只输出 YAML，不要解释）：\n{change}",
        })
        out, note = self._gen_loop()
        self.current_yaml = out
        return out, note

    def _gen_loop(self) -> tuple[str, str]:
        """共享的生成-校验-回喂循环，返回 (yaml_text, note)。

        messages 已由 generate/refime 填好首条 user 消息；循环内失败时
        把坏 YAML 与可读错误追加进历史再重试。成功返回校验通过的 YAML
        文本（safe_dump 重排，保持字段顺序）。
        """
        last_err = ""
        note = ""
        for _ in range(self.max_retries):
            resp = _post_chat(self.cfg, self.messages)
            note = (resp.get("choices") or [{}])[0].get("message", {}).get(
                "content", "") or ""
            text = extract_yaml(note)
            try:
                d = yaml.safe_load(text)
                if not isinstance(d, dict):
                    raise WorldError(f"顶层不是映射: {type(d).__name__}")
                resolve_zoo_models(d, self.on_download)
                _validate(d)
                out = yaml.safe_dump(d, allow_unicode=True, sort_keys=False)
                return out, note
            except Exception as e:
                last_err = f"{e}"
                if _looks_like_mjcf(text):
                    # 定向纠错：模型把平台当成 MJCF 生成器（mesh 文件引用
                    # 是无解的——平台根本不加载外部 mesh），明确改道
                    last_err = ("输出的是 MJCF/XML 且引用了 mesh 文件——平台"
                                "不加载外部 mesh。物体改用 box/sphere/cylinder "
                                "基本体表达，机器人改用 zoo:// 模型库，"
                                "重新输出完整的世界 YAML")
                self.messages.append({"role": "assistant", "content": text})
                self.messages.append({
                    "role": "user",
                    "content": f"生成的 YAML 校验失败：{last_err}\n"
                               f"请修正后重新输出完整 YAML（仍然只输出 YAML）。",
                })
        raise SceneGenError(f"重试 {self.max_retries} 次仍失败，末次错误: {last_err}")


def generate_scene(requirement: str, cfg: AgentConfig,
                   max_retries: int = 3,
                   on_download=None) -> tuple[str, str]:
    """需求文本 -> (世界 YAML 文本, LLM 附带说明)。

    单轮便捷入口：内部新建 SceneGenSession 跑一次 generate。多轮微调
    （"再改改"）请直接用 SceneGenSession.refine。行为与旧版完全一致，
    存量调用与测试不破。
    """
    sess = SceneGenSession(cfg, max_retries=max_retries, on_download=on_download)
    return sess.generate(requirement)


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
