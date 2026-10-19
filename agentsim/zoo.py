"""机器人模型库：MuJoCo Menagerie 官方模型一键接入（M4）。

预配置 13 个著名机器人（四足/人形/机械臂/灵巧手/双臂/移动操作）。
generate() 生成可编辑的世界 YAML + 控制器模板——用户改控制律即编程，
GUI/CLI/MCP zoo_create 三路同构。

依赖：pip install mujoco-menagerie（官方包；首次用某模型时下载一次到
~/.cache/mujoco_menagerie/，之后永久离线可用）。
"""
from __future__ import annotations

import math
import pathlib

import yaml

from agentsim.errors import WorldError


# ================================================================ 演示动作配置
# 两种步态：
# - trot：四足对角小跑。PD 力矩或位置目标双模式（编译期检测），对角腿同相
# - osc：绕"出生姿态"振荡。sweep=全部执行器摆（臂/手）；wave=摆子集其余保持（人形）
# 参数出处：go2 为实测值（600 tick / 9.6s 走 6.6m 不倒）；其余为官方 home
# keyframe + 同族缩放，逐模型实测校准。

def _trot(legs, kinds, home, amp, freq=2.0, settle=40, lag=None,
          kp=None, kd=None, ramp=150, hind_flip=False, desc="对角小跑"):
    """四足 trot 配置。

    legs: 腿名列表（[0]/[3] 同相，[1]/[2] 反相）；kinds: {kind: 执行器后缀}；
    home/amp: {kind: 标量} 或 {kind: {腿名: 标量}}（anymal 前后腿镜像）；
    kp/kd: 力矩模式 PD（位置执行器内部已含 PD，用不到）；
    ramp: 出生姿态渐变到站姿的 tick 数——出生 qpos0 与站姿差大时必须
    渐入（spot 直腿 -> hy 1.04 满目标直接 kp500 爆发发射，实测 31m）；
    hind_flip: 镜像构型（前后腿安装方向相反，如 anymal）：后腿相位
    再翻 π 且滞后取负——同相位的摆动会让前后腿推进互相抵消（原地
    踏步），实测 anymal 无翻转 0.43m / 翻转后 1.83m 稳定行走。
    """
    return {"kind": "trot", "legs": legs, "kinds": kinds, "home": home,
            "amp": amp, "freq": freq, "settle": settle, "ramp": ramp,
            "hind_flip": hind_flip,
            "lag": lag or {"hip": 0.0, "thigh": 0.0, "calf": 0.5},
            "kp": kp or {"hip": 25.0, "thigh": 60.0, "calf": 60.0},
            "kd": kd or {"hip": 0.5, "thigh": 1.0, "calf": 1.0}, "desc": desc}


def _balance(wave_re, amp, stance, pitch_acts, roll_acts, pelvis, ankles,
             bx=(8.0, 1.5), by=(6.0, 1.0), freq=0.6, settle=30,
             kp=120.0, kd=3.0, desc="站立平衡摆臂"):
    """人形站立平衡演示：微屈站姿 + 踝策略反馈。

    人形圆脚/窄脚静态不稳定，挥臂必倒——demo 用骨盆相对踝中点的水平
    偏移（运动学倾角）做 PD 反馈，修正踝 pitch/roll 目标。
    stance: {执行器名: 站姿目标}（微屈膝，未列的执行器保持出生姿态）；
    pitch_acts/roll_acts: [(执行器名, 符号)]——前向倾角修正 pitch、侧向修正
    roll，符号编译时按模型实测（方向反了就翻）；
    pelvis/ankles: obs["bodies"] 里的 body 名（倾角基准）；
    bx/by: (P, D) 反馈增益；kp/kd: 力矩模式腿部保持 PD（位置模式忽略）。
    """
    return {"kind": "balance", "wave_re": wave_re, "amp": amp, "freq": freq,
            "settle": settle, "stance": stance,
            "pitch_acts": pitch_acts, "roll_acts": roll_acts,
            "pelvis": pelvis, "ankles": ankles,
            "bx": bx, "by": by, "kp": kp, "kd": kd, "desc": desc}


def _osc(sweep_frac=None, wave_re=None, amp=0.0, freq=0.4, settle=10,
         kp=80.0, kd=2.0, desc="关节扫掠"):
    """振荡演示配置：sweep_frac 给 sweep（全部摆，幅值=占范围比例）；
    wave_re 给 wave（匹配的执行器摆，其余保持在出生姿态）。
    """
    cfg = {"kind": "osc", "freq": freq, "settle": settle,
           "kp": kp, "kd": kd, "desc": desc}
    if wave_re is not None:
        cfg["wave_re"] = wave_re
        cfg["amp"] = amp
    else:
        cfg["sweep_frac"] = sweep_frac if sweep_frac is not None else 0.35
    return cfg


# 预配置清单：key = menagerie key；xml = 纯机器人入口（不带 menagerie 的
# floor/light——场景由 AgentSim 世界提供）；pos = attach 初始高度
# （浮动基座要离地一点落下；固定基座全零）；demo = 运动演示配置
MODELS: dict[str, dict] = {
    # ---- 四足机器人 ----
    "unitree_go1": {
        "title": "Unitree Go1", "cat": "四足", "xml": "go1.xml",
        "pos": [0, 0, 0.5], "desc": "宇树 Go1，12 关节（hip/thigh/calf × 4 腿）",
        # 位置执行器（kp=100 内置 PD）：小摆幅 + 膝滞后大，900 tick 走 10.8m
        # ramp=1：出生即全目标（空中收腿、落地即站姿）——渐入反而半直腿
        # 落地失稳（实测 RAMP=150 下 rootz 0.06 趴地）
        "demo": _trot(["FL", "FR", "RL", "RR"],
                      {"hip": "_hip", "thigh": "_thigh", "calf": "_calf"},
                      {"hip": 0.0, "thigh": 0.9, "calf": -1.8},
                      {"hip": 0.0, "thigh": 0.2, "calf": 0.3},
                      freq=2.4, settle=60, ramp=1,
                      lag={"hip": 0.0, "thigh": 0.0, "calf": 0.9})},
    "unitree_go2": {
        "title": "Unitree Go2", "cat": "四足", "xml": "go2.xml",
        "pos": [0, 0, 0.5], "desc": "宇树 Go2，12 关节（hip/thigh/calf × 4 腿）",
        # ramp=1 同 go1：出生即全目标，落地即站姿。
        # implicitfast 下 freq2.0 趴地（rootz 0.06/1.07m）、freq2.4 最优
        # （rootz 0.28/6.50m）——积分器变更后网格重扫的实测值
        "demo": _trot(["FL", "FR", "RL", "RR"],
                      {"hip": "_hip", "thigh": "_thigh", "calf": "_calf"},
                      {"hip": 0.0, "thigh": 0.9, "calf": -1.8},
                      {"hip": 0.0, "thigh": 0.35, "calf": 0.55},
                      freq=2.4, ramp=1)},
    "anybotics_anymal_c": {
        "title": "ANYmal C", "cat": "四足", "xml": "anymal_c.xml",
        # 出生直腿贴 keyframe 会穿地 3cm（几何实测），抬高 5cm 悬空落下
        "pos": [0, 0, 0.65], "desc": "ANYbotics 工业四足，12 关节",
        # 原生执行器 kp100/kd0 压不住 50kg 机身（跟踪误差实测 ~1.0 rad，
        # 全参数网格都不走）——demo 补丁加 4 倍刚度 + 阻尼（同 h1 路径）。
        # hind_flip：前后腿镜像安装，后腿相位翻 π + 滞后取负才与 go1 同构
        # （不翻：前后腿推进抵消原地踏步；翻后稳定行走）。
        # implicitfast 下重扫：freq1.6/lag0.9 最优（rootz 0.51/1.75m），
        # Euler 时代的 freq2.0/lag0.7 趴行（rootz 0.11）
        "demo": {**_trot(["LF", "RF", "LH", "RH"],
                         {"hip": "_HAA", "thigh": "_HFE", "calf": "_KFE"},
                         # 官方 standing keyframe：前后腿镜像（HFE ±30°，KFE ∓45°）
                         {"hip": 0.0,
                          "thigh": {"LF": 0.5236, "RF": 0.5236,
                                    "LH": -0.5236, "RH": -0.5236},
                          "calf": {"LF": -0.7854, "RF": -0.7854,
                                   "LH": 0.7854, "RH": 0.7854}},
                         {"hip": 0.0, "thigh": 0.35, "calf": 0.50},
                         freq=1.6, settle=60, hind_flip=True,
                         lag={"hip": 0.0, "thigh": 0.0, "calf": 0.9}),
                "act_patch": {"kp": [("", 400.0)], "kd": [("", 8.0)]}}},
    "boston_dynamics_spot": {
        "title": "Boston Dynamics Spot", "cat": "四足", "xml": "spot.xml",
        # qpos0 原生即半蹲（hy 1.04/kn -1.8，躯干 0.75 脚悬空 5.7cm）。
        # 实测坑：①出生 0.50 压脚进地 19cm → 弹射 31m；②步态塌弹
        # （rootz 0.15-0.86，kp/kd/amp/freq/出生高度 30+ 组合无解）——
        # 根因是原生 solref=0.004 超硬接触被 attach 的默认 Euler 覆盖
        # 原生 implicitfast（"硬接触+显式积分=数值震荡"），修 integrator
        # 后 rootz 0.47 稳定行走 3.16m；出生贴原生姿态 0.75 是最优起点
        "pos": [0, 0, 0.75], "desc": "Spot 四足（带 arm 变体在 menagerie 里）",
        "demo": {**_trot(["fl", "fr", "hl", "hr"],
                         {"hip": "_hx", "thigh": "_hy", "calf": "_kn"},
                         # home = keyframe 站姿（出生即此姿态，RAMP 无操作）
                         {"hip": 0.0, "thigh": 1.04, "calf": -1.8},
                         {"hip": 0.0, "thigh": 0.15, "calf": 0.25},
                         freq=1.6, settle=100, ramp=1),
                 # kp/kd 原生值（执行器不动，声明只为触发 demo_model.xml
                 # 路径以备后续补丁）
                 "act_patch": {"kp": [("", 500.0)], "kd": [("", 40.0)]}}},
    # ---- 人形机器人 ----
    "unitree_h1": {
        "title": "Unitree H1", "cat": "人形", "xml": "h1.xml",
        "pos": [0, 0, 1.06], "desc": "宇树人形，19 自由度",
        # menagerie h1 是球脚 + 力矩电机（踝 ±40Nm）：外置 62.5Hz 采样力矩
        # PD 无法在踝 ±40Nm 的窄窗口内站立（v10 实测出生 1mm 落差也单调
        # 后仰发散）。demo 模型补丁双管齐下：脚底加平垫（真机出厂即平垫
        # 脚）+ 执行器 position 化（内部 PD 每 timestep 计算，g1 已验证
        # 的站立路径）；真机 h1 站立/行走需 RL（同 PPO 示例管线）
        "demo": {**_balance(r"shoulder_pitch|elbow", amp=0.25,
                            # 直腿站立：h1 运动学与 g1 相反，屈髋/屈膝都让
                            # COM 后移（静力学扫描实测），微屈站姿反而更
                            # 不稳；腿部静矩由 range 锁位硬扛（站立模式）
                            stance={},
                            # 纯髋策略微调：腿被锁位 ±0.03，反馈只在锁内
                            # 微调对抗扰动；踝不参与（任何大幅转动都翘
                            # 平底破坏贴地）
                            pitch_acts=[("left_hip_pitch", -1.0),
                                        ("right_hip_pitch", -1.0)],
                            roll_acts=[("left_hip_roll", 1.0),
                                       ("right_hip_roll", 1.0)],
                            pelvis="pelvis",
                            ankles=["left_ankle_link", "right_ankle_link"],
                            bx=(2.0, 0.15), by=(1.5, 0.1),
                            freq=0.6, settle=160,
                            desc="站立摆臂（腿锁位站立模式 + 髋微调）"),
               "soles": {"bodies": ["left_ankle_link", "right_ankle_link"],
                         "size": [0.09, 0.04, 0.008],
                         # 平底后移让 COM 投影居中：qpos0 直腿时 COM x=+0.016
                         # 落在原支撑面 [-0.001,+0.179] 的后边缘 17mm 内，
                         # 落地冲击即倒；平底中心对齐 COM 后前后余量各 ~0.07
                         "pos": [-0.023, 0.0, -0.077]},
               # 踝 kp 加大 + 腿关节 range 锁位 + 固定基座：h1 的 COM 后置
               # 踝 0.023m 产生 10.6Nm 恒后仰静矩，实测（v10-v15 十五轮）
               # 锁位刚体 + COM 居中平底仍匀速后滑（接触求解层面的蠕动
               # 无法收敛），固定基座是站立展示的标准做法；真机站立/行走
               # 需 RL（见 examples/train_ppo_go2.py 同款管线）
               "act_patch": {"kp": [("ankle", 300.0), ("knee", 200.0),
                                    ("hip", 200.0), ("torso", 100.0),
                                    ("", 60.0)],
                             "kd": [("ankle", 8.0), ("knee", 5.0),
                                    ("hip", 5.0), ("torso", 4.0),
                                    ("", 2.0)],
                             "freeze_root": True}}},
    "unitree_g1": {
        "title": "Unitree G1", "cat": "人形", "xml": "g1.xml",
        "pos": [0, 0, 0.8], "desc": "宇树 G1 人形，29 自由度",
        # 位置执行器（内部 PD）：微屈站姿 + 踝 pitch/roll 目标修正
        "demo": _balance(r"shoulder_pitch|elbow", amp=0.25,
                         stance={"left_hip_pitch_joint": -0.2,
                                 "right_hip_pitch_joint": -0.2,
                                 "left_knee_joint": 0.4, "right_knee_joint": 0.4,
                                 "left_ankle_pitch_joint": -0.2,
                                 "right_ankle_pitch_joint": -0.2},
                         pitch_acts=[("left_ankle_pitch_joint", 1.0),
                                     ("right_ankle_pitch_joint", 1.0)],
                         roll_acts=[("left_ankle_roll_joint", 1.0),
                                    ("right_ankle_roll_joint", 1.0)],
                         pelvis="pelvis",
                         ankles=["left_ankle_roll_link", "right_ankle_roll_link"],
                         bx=(2.0, 0.4), by=(1.5, 0.3),
                         freq=0.6, settle=30, desc="站立平衡 + 摆臂（踝策略）")},
    # ---- 机械臂 ----
    "franka_emika_panda": {
        "title": "Franka Emika Panda", "cat": "机械臂", "xml": "panda_nohand.xml",
        "pos": [0, 0, 0], "desc": "Franka Panda 7 轴臂（含手变体 panda.xml）",
        "demo": _osc(sweep_frac=0.4)},
    "universal_robots_ur5e": {
        "title": "Universal Robots UR5e", "cat": "机械臂", "xml": "ur5e.xml",
        "pos": [0, 0, 0], "desc": "UR5e 6 轴协作臂",
        "demo": _osc(sweep_frac=0.4)},
    "ufactory_xarm7": {
        "title": "UFACTORY xArm7", "cat": "机械臂", "xml": "xarm7_nohand.xml",
        "pos": [0, 0, 0], "desc": "xArm7 7 轴臂（含手变体 xarm7.xml）",
        "demo": _osc(sweep_frac=0.4)},
    # ---- 灵巧手 ----
    "shadow_hand": {
        "title": "Shadow Hand E3M5", "cat": "灵巧手", "xml": "left_hand.xml",
        "pos": [0, 0, 0.3], "desc": "Shadow 灵巧手 20 关节（浮动基座）",
        "demo": _osc(sweep_frac=0.5, freq=0.6, desc="手指波浪开合")},
    # ---- 双臂 ----
    "aloha": {
        "title": "ALOHA", "cat": "双臂", "xml": "aloha.xml",
        "pos": [0, 0, 0], "desc": "ALOHA 双臂遥操作平台，16 自由度",
        "demo": _osc(sweep_frac=0.35, freq=0.5, desc="双臂交替摆动")},
    # ---- 移动操作 ----
    "google_robot": {
        "title": "Google Robot", "cat": "移动操作", "xml": "robot.xml",
        "pos": [0, 0, 0], "desc": "Everyday Robots 移动操作平台",
        "demo": _osc(sweep_frac=0.35)},
    "hello_robot_stretch_3": {
        "title": "Hello Robot Stretch 3", "cat": "移动操作", "xml": "stretch.xml",
        "pos": [0, 0, 0], "desc": "Stretch 3 家庭服务移动机器人",
        "demo": _osc(sweep_frac=0.3, freq=0.5, desc="底盘行进 + 升降/臂展示")},
}


def ensure_model(key: str) -> pathlib.Path:
    """确保模型 XML 在本地缓存（未下载则触发一次下载），返回其路径。"""
    if key not in MODELS:
        raise WorldError(f"模型库没有 '{key}'，可用: {sorted(MODELS)}")
    try:
        import mujoco_menagerie as mm
    except ImportError:
        raise WorldError(
            "模型库依赖 mujoco-menagerie，请安装：pip install mujoco-menagerie"
        ) from None
    entry = mm.get(key)
    xml_name = MODELS[key]["xml"]

    def _hits():
        try:
            return [f for f in entry.files() if f.name == xml_name and f.exists()]
        except Exception:
            return []

    hits = _hits()
    # files() 清单可能滞后于解压目录（旧版包不列 nohand/mjx 变体），
    # 直接查解压目录兜底
    if not hits and (entry.path() / xml_name).exists():
        hits = [entry.path() / xml_name]
    if not hits:                      # 本地没有，走正式下载路径
        entry.model()                 # 下载 + 编译（一次，之后永久离线）
        hits = _hits() or ([entry.path() / xml_name]
                           if (entry.path() / xml_name).exists() else [])
    if not hits:
        raise WorldError(f"menagerie 包里没有 '{key}' 的 {xml_name}")
    return hits[0]


def _write_demo_model(src_xml: pathlib.Path, out: pathlib.Path,
                      soles: dict | None = None,
                      act_patch: dict | None = None) -> pathlib.Path:
    """demo 专用模型补丁（不改 menagerie 原文件），写 demo_model.xml 到生成目录。

    soles：给指定 body 脚底加平底碰撞 geom（h1 球脚静态站不住，真机出厂
    是平垫脚）。mesh 等资产路径 to_xml 后是相对 assets/，替换成绝对路径
    保证可加载。
    act_patch：motor -> position（gainprm/biasprm 内部 PD，mujoco 每个物理
    步计算，等效真机 RL 策略的力矩刚度）——外置采样力矩 PD 无法在窄力矩
    窗口（h1 踝 ±40Nm）内稳定站立；ctrlrange 换成关节角度范围。
    freeze_root（act_patch 的字段）：删 root freejoint 变固定基座——
    站立展示类 demo 的标准做法（行走/真浮基运动留给 RL 示例）。
    """
    import mujoco
    spec = mujoco.MjSpec.from_file(str(src_xml))
    if act_patch and act_patch.get("freeze_root"):
        for b in spec.bodies:
            if b.parent is spec.worldbody:
                for j in list(b.joints):
                    spec.delete(j)
    if soles:
        for body_name in soles["bodies"]:
            b = spec.body(body_name)
            if b is None:
                raise WorldError(f"demo 模型补丁：找不到 body '{body_name}'")
            b.add_geom(name=f"demo_sole_{body_name}",
                       type=mujoco.mjtGeom.mjGEOM_BOX, size=list(soles["size"]),
                       pos=list(soles["pos"]), rgba=[0.3, 0.3, 0.35, 0.6])
    if act_patch:
        _patch_actuators(spec, act_patch)
        # qpos0：[(关节子串, ref)]——改 hinge ref（qpos0 出生角），出生
        # 即站姿避免空中收腿（spot 直腿 0.75 落地：腿收不完砸地趴/直腿
        # 反弹 1.06m，网格无解）；ref 只对 hinge 生效，freejoint 不动
        for sub, val in act_patch.get("qpos0", ()):
            for j in spec.joints:
                if sub in j.name and j.type == mujoco.mjtJoint.mjJNT_HINGE:
                    j.ref = float(val)
    text = spec.to_xml()
    assets = src_xml.parent / "assets"
    if assets.is_dir():
        text = text.replace('="assets/', f'="{assets}/')
    path = out / "demo_model.xml"
    path.write_text(text, encoding="utf-8")
    return path


def _patch_actuators(spec, patch: dict) -> None:
    """执行器 position 化：gainprm=[kp]，biasprm=[0,-kp,-kd]（MJCF 标准
    位置执行器写法）。kp/kd 是 (子串匹配, 值) 列表，最后一个子串空串
    作兜底；省略 kp/kd 则不动执行器（只用 qpos0/range 的纯姿态补丁）。
    ctrlrange 换成所驱动关节的角度范围。
    range_patch：[(子串, 锁位半宽)]——匹配执行器驱动的关节 range 收窄到
    [-w, +w]，模拟真机站立位的关节刚性锁定（静矩由限位承担，不靠
    执行器刚度硬扛），仅限 demo 模型（站立展示不动腿）。
    """
    import mujoco

    def pick(rules, name):
        for sub, v in rules:
            if sub in name:
                return v
        return rules[-1][1]

    for a in spec.actuators:
        if "kp" in patch:
            kp, kd = pick(patch["kp"], a.name), pick(patch["kd"], a.name)
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.gainprm[0:3] = [kp, 0.0, 0.0]
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm[0:3] = [0.0, -kp, -kd]
        joint = (spec.joint(a.target)
                 if a.trntype == mujoco.mjtTrn.mjTRN_JOINT else None)
        if joint is None:
            continue
        if joint.limited:
            a.ctrlrange[0:2] = list(joint.range)
        a.ctrllimited = True
        for sub, w in patch.get("range", ()):   # 关节锁位（demo 站立模式）
            if sub in a.name:
                joint.range = [-w, w]
                joint.limited = True
                a.ctrlrange[0:2] = [-w, w]
                break


def generate(key: str, out_dir: str | pathlib.Path | None = None,
             demo: bool = False) -> dict:
    """生成可编辑的模型项目（world.yaml + 控制器）。

    demo=False：controller.py 模板（用户改控制律）；demo=True：另生成
    demo.py 运动演示且 world.yaml 指向它（controller.py 模板仍附上），
    ctrl_period 降到 8 提高控制频率（四足步态需要 62.5Hz）。
    默认输出 ~/.agentsim/zoo/<key>/；返回 {model, world, controller, demo?}。
    """
    xml = ensure_model(key)
    meta = MODELS[key]
    out = pathlib.Path(out_dir) if out_dir else (
        pathlib.Path.home() / ".agentsim" / "zoo" / key)
    out.mkdir(parents=True, exist_ok=True)

    # demo 专用模型补丁（如 h1 加平底 geom + 执行器 position 化）：
    # 生成 demo_model.xml，world.yaml 指向它；不改 menagerie 原文件
    if demo and (meta["demo"].get("soles") or meta["demo"].get("act_patch")):
        xml = _write_demo_model(xml, out, meta["demo"].get("soles"),
                                meta["demo"].get("act_patch"))

    # 编译一次拿权威执行器清单（attach 后的名字，与 hello 下发完全一致）
    from agentsim.world.builder import build_world
    from agentsim.world.schema import parse_world
    world = {
        "name": key, "timestep": 0.002,
        "ctrl_period": 8 if demo else 32,
        "gravity": [0, 0, -9.81],
        # demo 世界给高摩擦求解刚度：menagerie 四足原生 impratio 50-100，
        # 大质量机身在默认 1 下 pyramidal 摩擦误差被放大（spot 趴行根因）
        "impratio": 100,
        # implicitfast：menagerie 高刚度模型原生积分器（spot solref=0.004
        # 超硬接触），Euler 显式积分会数值震荡弹跳——attach 用世界值
        # 覆盖模型原值，故必须显式声明
        "integrator": "implicitfast",
        "scene": {"floor": {"size": [4.0, 4.0], "texture": "grid",
                            "friction": 0.8},
                  "lights": [{"pos": [0, 0, 3]}]},
        "robots": [{"name": key, "pos": meta["pos"],
                    "model": str(xml),
                    "controller": "demo.py" if demo else "controller.py"}],
        "task": {"max_ticks": 3000},
        "runtime": {"realtime": False, "controller_timeout": 100,
                    "timeout_policy": "last_ctrl"},
    }
    spec = parse_world(world)
    model, index = build_world(spec)          # 编译校验 + 索引
    actuators = list(index.robots[key].actuators)

    world_path = out / "world.yaml"
    with open(world_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(world, f, allow_unicode=True, sort_keys=False,
                       default_flow_style=None)

    ctrl_path = out / "controller.py"
    ctrl_path.write_text(_controller_template(meta, actuators),
                         encoding="utf-8")

    result = {"model": key, "title": meta["title"], "world": str(world_path),
              "controller": str(ctrl_path), "actuators": actuators}
    if demo:
        demo_path = out / "demo.py"
        dt = world["ctrl_period"] * world["timestep"]   # 1 控制周期秒数
        demo_path.write_text(_demo_source(key, meta, model, index, dt),
                             encoding="utf-8")
        result["demo"] = str(demo_path)
    return result


def _controller_template(meta: dict, actuators: list[str]) -> str:
    """控制器模板：清单 + 注释化的 API 演示，用户改控制律。"""
    act_lines = ",\n#   ".join(", ".join(actuators[i:i + 4])
                              for i in range(0, len(actuators), 4))
    return f'''"""{meta['title']}（{meta['cat']}）—— {meta['desc']}

编辑这个文件写你的控制律，然后在 GUI 点 Run 重跑（或 CLI 重新 run）。

可用执行器（{len(actuators)} 个，{meta['cat']}）：
#   {act_lines}

常用 API（agentsim.controller.Robot）：
#   robot.set("执行器名", 目标值)     写目标（下个 tick 生效）
#   robot.set_all([v1, v2, ...])      按上面顺序一次写全部
#   robot.range("执行器名")           查范围 (lo, hi)
#   obs = robot.obs                   最近观测 {{"qpos"/"qvel"/"bodies"/"sensors"}}
#   robot.step() 返回 False           内核叫停（到 max_ticks），退出循环
"""
from agentsim.controller import Robot

robot = Robot()

while robot.step():
    # —— 示例控制律：所有执行器回零目标（改成你的）——
    for name in robot.actuators:
        lo, hi = robot.range(name)
        robot.set(name, 0.0)
    # 例：正弦摆动第一个执行器
    # robot.set(robot.actuators[0], 0.5 * (robot.tick % 100 < 50))
'''


# ================================================================ 运动演示生成
# ACTS 表从编译后的世界模型提取（执行器名/观测偏移/模式/范围与内核 hello
# 完全同一来源），生成后 demo.py 自包含、可自由改参数。

def _demo_source(key: str, meta: dict, model, index, dt: float) -> str:
    demo = meta["demo"]
    rows = _act_rows(model, index.robots[key], key)
    if demo["kind"] == "trot":
        return _trot_source(meta, demo, rows, dt)
    if demo["kind"] == "balance":
        return _balance_source(meta, demo, rows, dt)
    return _osc_source(meta, demo, rows, dt)


def _act_rows(model, ridx, robot_name: str) -> list[dict]:
    """从编译后的模型提取执行器表：观测偏移（与 kernel.robot_obs 同一布局）、
    控制模式（torque/position/velocity，由 biastype/biasprm 判定）、gear、范围。
    """
    import mujoco

    from agentsim.kernel.task import obs_layout
    lay = obs_layout(model, ridx)
    prefix = f"{robot_name}/"
    rows = []
    for name, aid in ridx.actuators.items():
        if int(model.actuator_trntype[aid]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            jid = int(model.actuator_trnid[aid][0])
            jfull = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            jname = jfull[len(prefix):] if jfull.startswith(prefix) else jfull
            qi, vi, _ = lay[jname]
            bt = int(model.actuator_biastype[aid])
            bp = [float(x) for x in model.actuator_biasprm[aid]]
            if bt == int(mujoco.mjtBias.mjBIAS_NONE):
                mode = "torque"                       # ctrl = 力矩（go2/h1）
            elif bp[1] == 0.0 and bp[2] != 0.0:
                mode = "velocity"                      # ctrl = 目标速度（轮子）
            else:
                mode = "position"                     # ctrl = 目标角度（内部含 PD）
        else:
            qi = vi = 0
            mode = "raw"                              # 无关节传输，摆控制量本身
        rows.append({"name": name, "qi": qi, "vi": vi, "mode": mode,
                     "gear": float(model.actuator_gear[aid][0]),
                     "lo": float(model.actuator_ctrlrange[aid][0]),
                     "hi": float(model.actuator_ctrlrange[aid][1])})
    return rows


def _leg_kind(name: str, legs: list, kinds: dict):
    """从执行器名解析 (腿, 关节类)：FL_thigh -> (FL, thigh)。
    对不上返回 (None, None)——该执行器保持出生姿态不动。"""
    leg = next((l for l in legs if name.startswith(l + "_")), None)
    kind = next((k for k, suf in kinds.items() if name.endswith(suf)), None)
    return leg, kind


def _home_of(spec, leg):
    """home 配置项解析：标量或 {腿名: 标量}（anymal 前后腿镜像）。"""
    if isinstance(spec, dict):
        return float(spec.get(leg, 0.0))
    return float(spec)


def _fmt_acts(rows: list[dict]) -> str:
    """ACTS 表渲染：一行一个执行器，参数齐全可读。"""
    keys = list(rows[0]) if rows else []
    order = ["name", "qi", "vi", "mode", "home", "amp", "phase", "kind",
             "lag", "kp", "kd", "gear", "lo", "hi"]
    keys = [k for k in order if k in keys] + [k for k in keys if k not in order]
    lines = []
    for r in rows:
        body = ", ".join(f"{k!r}: {r[k]!r}" for k in keys)
        lines.append(f"    {{{body}}},")
    return "[\n" + "\n".join(lines) + "\n]"


def _trot_source(meta: dict, demo: dict, rows: list[dict], dt: float) -> str:
    legs = demo["legs"]
    phase_of = {legs[0]: 0.0, legs[3]: 0.0, legs[1]: math.pi, legs[2]: math.pi}
    hind = {legs[2], legs[3]} if demo.get("hind_flip") else set()
    out = []
    for r in rows:
        leg, kind = _leg_kind(r["name"], legs, demo["kinds"])
        home = None
        kp = kd = amp = phase = lg = 0.0
        if kind is not None:
            home = _home_of(demo["home"][kind], leg)
            amp = float(demo["amp"].get(kind, 0.0))
            phase = phase_of.get(leg, 0.0)
            if leg in hind:                      # 镜像构型：后腿翻相
                phase += math.pi
            lg = float(demo["lag"].get(kind, 0.0))
            if leg in hind:                       # …且滞后方向也镜像
                lg = -lg
            kp = float(demo["kp"].get(kind, 0.0))
            kd = float(demo["kd"].get(kind, 0.0))
        out.append({**r, "home": home, "amp": amp, "phase": phase,
                    "lag": lg, "kind": kind or "hold", "kp": kp, "kd": kd})
    acts = _fmt_acts(out)
    return f'''"""{meta['title']} 运动演示 —— {demo['desc']}

CPG 开环对角小跑：q*(t) = home + amp·sin(2πf·t + 相位)，对角腿（{legs[0]}/{legs[3]}
与 {legs[1]}/{legs[2]}）互为反相；先站稳 SETTLE tick 再起摆。
力矩执行器用 PD 跟踪（τ = kp·(q*−q) − kd·qd），位置执行器直接发目标角。

可调参数：FREQ（步频 Hz）、SETTLE（先站稳的 tick 数）、ACTS 里的 amp/相位/
lag（膝滞后大腿，模拟推进弧线；镜像构型后腿已取负，改符号会破坏对称）。
想写自己的控制律：改 world.yaml 的 controller 字段为 controller.py（模板已附）。
想上 RL 步态：见项目 examples/train_ppo_go2.py（M3 AgentSimEnv + PPO 训练）。
"""
import math

from agentsim.controller import Robot

DT = {dt!r}                 # 1 控制周期 = ctrl_period × timestep
FREQ = {demo['freq']!r}
SETTLE = {demo['settle']!r}
RAMP = {demo['ramp']!r}     # 出生姿态渐变到站姿的 tick 数（防出发射）
ACTS = {acts}

robot = Robot()
HOME = None                 # 首 tick 记录出生姿态（home=None 的执行器用）

while robot.step():
    obs = robot.obs
    q, qd = obs["qpos"], obs["qvel"]
    if HOME is None:
        HOME = {{a["name"]: q[a["qi"]] for a in ACTS}}
    t = (robot.tick - SETTLE) * DT
    frac = min(1.0, robot.tick / RAMP)
    for a in ACTS:
        h = a["home"] if a["home"] is not None else HOME[a["name"]]
        des = HOME[a["name"]] + (h - HOME[a["name"]]) * frac   # 渐变到站姿
        if t >= 0.0:
            des += a["amp"] * math.sin(
                2.0 * math.pi * FREQ * t + a["phase"] + a["lag"])
        des = max(a["lo"], min(a["hi"], des))
        if a["mode"] == "torque":
            tau = a["kp"] * (des - q[a["qi"]]) - a["kd"] * qd[a["vi"]]
            robot.set(a["name"], max(a["lo"], min(a["hi"], tau / a["gear"])))
        else:
            robot.set(a["name"], des)
'''


def _osc_source(meta: dict, demo: dict, rows: list[dict], dt: float) -> str:
    import re
    wave_re = demo.get("wave_re")
    wave = re.compile(wave_re) if wave_re else None
    out = []
    n = len(rows)
    for i, r in enumerate(rows):
        if wave is not None:
            amp = float(demo["amp"]) if wave.search(r["name"]) else 0.0
            phase = 0.0                               # 摆动子集同相
        else:
            amp = float(demo["sweep_frac"]) * (r["hi"] - r["lo"]) / 2.0
            phase = 2.0 * math.pi * i / max(n, 1)      # 行波：各关节错相
        out.append({**r, "home": None, "amp": amp, "phase": phase,
                    "kind": "osc", "kp": float(demo["kp"]),
                    "kd": float(demo["kd"])})
    acts = _fmt_acts(out)
    desc = demo.get("desc", "关节扫掠")
    mode_note = ("摆动子集（shoulder_pitch/elbow）绕出生姿态摆动，其余关节保持——"
                 if wave is not None else "各关节绕出生姿态错相摆动——")
    return f'''"""{meta['title']} 运动演示 —— {desc}

{mode_note}首 tick 的关节位置即摆动中心（出生姿态）；力矩执行器 PD 跟踪，
位置执行器直接发目标角，速度执行器（轮子）摆目标速度。

可调参数：FREQ（摆动频率 Hz）、SETTLE（保持出生姿态的 tick 数）、
ACTS 里的 amp（摆幅）与 phase（相位）。写自己的控制律改 world.yaml 的
controller 字段为 controller.py（模板已附）。
"""
import math

from agentsim.controller import Robot

DT = {dt!r}                 # 1 控制周期 = ctrl_period × timestep
FREQ = {demo['freq']!r}
SETTLE = {demo['settle']!r}
ACTS = {acts}

robot = Robot()
HOME = None                 # 首 tick 记录出生姿态（速度执行器记 0）

while robot.step():
    obs = robot.obs
    q, qd = obs["qpos"], obs["qvel"]
    if HOME is None:
        HOME = {{a["name"]: (q[a["qi"]] if a["mode"] != "velocity" else 0.0)
                 for a in ACTS}}
    t = (robot.tick - SETTLE) * DT
    for a in ACTS:
        h = HOME[a["name"]]
        des = h if t < 0.0 else h + a["amp"] * math.sin(
            2.0 * math.pi * FREQ * t + a["phase"])
        des = max(a["lo"], min(a["hi"], des))
        if a["mode"] == "torque":
            tau = a["kp"] * (des - q[a["qi"]]) - a["kd"] * qd[a["vi"]]
            robot.set(a["name"], max(a["lo"], min(a["hi"], tau / a["gear"])))
        else:
            robot.set(a["name"], des)
'''


def _balance_source(meta: dict, demo: dict, rows: list[dict], dt: float) -> str:
    import re
    wave_re = re.compile(demo.get("wave_re") or r"$^")   # 不给则永不匹配
    acts = _fmt_acts(rows)
    wave = {r["name"]: (float(demo["amp"]) if wave_re.search(r["name"]) else 0.0)
            for r in rows}
    wave_lit = "\n".join(f"    {k!r}: {v!r}," for k, v in wave.items())
    stance_lit = "\n".join(f"    {k!r}: {v!r}," for k, v in demo["stance"].items())
    pitch_lit = ", ".join(f"({k!r}, {s!r})" for k, s in demo["pitch_acts"])
    roll_lit = ", ".join(f"({k!r}, {s!r})" for k, s in demo["roll_acts"])
    return f'''"""{meta['title']} 运动演示 —— {demo['desc']}

人形窄支撑面静态不稳定，光靠保持关节目标必倒——本演示加踝策略反馈：
骨盆相对两踝中点的水平偏移（运动学倾角代理）做 PD，实时修正踝 pitch /
roll 目标。位置执行器直接发目标角；力矩执行器 PD 跟踪 + 平衡力矩叠加。

可调参数：SETTLE（站稳 tick 数）、AMP/FREQ（摆臂）、BP/BP_D（前后
平衡增益）、BR/BR_D（侧向）、PITCH_ACTS/ROLL_ACTS 里的符号（修正方向
反了就翻符号）。想上 RL 行走：见 examples/train_ppo_go2.py 的训练管线。
"""
import math

from agentsim.controller import Robot

DT = {dt!r}                 # 1 控制周期 = ctrl_period × timestep
FREQ = {demo['freq']!r}
SETTLE = {demo['settle']!r}
RAMP = 150                  # 从出生姿态渐变到站姿的 tick 数（防出发射）
STANCE = {{
{stance_lit}
}}                          # 微屈站姿（未列出的执行器保持出生姿态）
WAVE = {{
{wave_lit}
}}                          # 摆臂子集：执行器名 -> 摆幅
PITCH_ACTS = [{pitch_lit}]  # 前后倾角修正（符号 = 修正方向）
ROLL_ACTS = [{roll_lit}]    # 侧向倾角修正
PELVIS = {demo['pelvis']!r}
ANKLES = {demo['ankles']!r}   # obs["bodies"] 里的踝 body 名
BP, BP_D = {demo['bx'][0]!r}, {demo['bx'][1]!r}   # 前后 (P, D)
BR, BR_D = {demo['by'][0]!r}, {demo['by'][1]!r}   # 侧向 (P, D)
KP, KD = {demo['kp']!r}, {demo['kd']!r}           # 力矩模式保持 PD
ACTS = {acts}

robot = Robot()
HOME = None                 # 出生姿态
LX0 = LY0 = None            # 倾角基准（首 tick 的骨盆-踝中点偏移）
PLX = PLY = 0.0             # 上 tick 倾角（算变化率用）

while robot.step():
    obs = robot.obs
    q, qd = obs["qpos"], obs["qvel"]
    B = obs["bodies"]
    if HOME is None:
        HOME = {{a["name"]: q[a["qi"]] for a in ACTS}}
    ax = (B[ANKLES[0]][0] + B[ANKLES[1]][0]) / 2
    ay = (B[ANKLES[0]][1] + B[ANKLES[1]][1]) / 2
    lx, ly = B[PELVIS][0] - ax, B[PELVIS][1] - ay
    if LX0 is None:
        LX0, LY0 = lx, ly
    lx -= LX0
    ly -= LY0
    rlx = (lx - PLX) / DT
    rly = (ly - PLY) / DT
    PLX, PLY = lx, ly
    # 倾角 -> 各执行器修正量（pitch 管前后、roll 管侧向）
    corr = {{}}
    for name, s in PITCH_ACTS:
        corr[name] = corr.get(name, 0.0) + s * (BP * lx + BP_D * rlx)
    for name, s in ROLL_ACTS:
        corr[name] = corr.get(name, 0.0) + s * (BR * ly + BR_D * rly)
    t = (robot.tick - SETTLE) * DT
    frac = min(1.0, robot.tick / RAMP)
    for a in ACTS:
        n = a["name"]
        des = STANCE.get(n)
        if des is not None:
            des = HOME[n] + (des - HOME[n]) * frac
        else:
            des = HOME[n] + WAVE.get(n, 0.0) * (
                0.0 if t < 0.0 else math.sin(2.0 * math.pi * FREQ * t))
        des += corr.get(n, 0.0)
        des = max(a["lo"], min(a["hi"], des))
        if a["mode"] == "torque":
            tau = KP * (des - q[a["qi"]]) - KD * qd[a["vi"]]
            robot.set(n, max(a["lo"], min(a["hi"], tau / a["gear"])))
        else:
            robot.set(n, des)
'''
