"""夹爪库：末端执行器（gripper/end-effector）注册表与装配解析。

与 zoo.py（机器人库）同构，但夹爪不是独立世界实体——它通过 MjSpec 的
site-attach 挂到某台机器人腕部的挂点 site（如 menagerie 机械臂标配的
attachment_site），其关节/执行器/tendon 经嵌套 attach 后全名变为
"<robot>/gripper/..."，自动并入该机器人的设备索引，控制器直接
robot.set("gripper/gripper_act", v) 开合。

mount 标定（见 models/grippers/onrobot_2fg7.xml 顶部注释）：
夹爪 root 在自包含 XML 中归零，相对挂点 site 的安装变换（法兰盘厚度 +
轴向翻转）全部记录在下表 mounts 里。四台 menagerie 机械臂（ur5e/ur10e/
xarm7_nohand/panda_nohand）的 attachment_site 局部轴约定一致，实测同一
套 (pos=[0,-0.028,0.1494], quat=[0,1,0,0]) 即可让夹爪 TCP 落在挂点
+z 0.1494m、夹爪正向与官方 ur10e+2FG7 参考场景逐位吻合。
"""
from __future__ import annotations

import pathlib

from agentsim.errors import WorldError

ROOT = pathlib.Path(__file__).resolve().parents[1]   # 仓库根（含 models/）
GRIPPER_DIR = ROOT / "models" / "grippers"

# 内层 attach 的固定前缀：夹爪 body/joint/actuator 去掉机器人前缀后都以
# "gripper/" 开头，避免与机器人本体同名设备冲突
GRIPPER_PREFIX = "gripper/"

#: 2FG7 统一安装变换（site 局部系；四臂通用，实测值，见模块 docstring）
_MOUNT_2FG7 = {"pos": [0.0, -0.028, 0.1494],
               "quat": [0.0, 1.0, 0.0, 0.0]}     # Rot_x(180°)，wxyz

GRIPPERS: dict[str, dict] = {
    "onrobot_2fg7": {
        "title": "OnRobot 2FG7 平行夹爪",
        "cat": "平行夹爪",
        "xml": "onrobot_2fg7.xml",
        "site": "attachment_site",          # 缺省挂点（可在 YAML 覆盖）
        "actuator": "gripper_act",           # 开合执行器（夹爪内短名）
        "open": 0.0,                         # ctrl=0   全开
        "close": 0.038,                      # ctrl=0.038 闭合
        "joints": ("left_slide", "right_slide"),
        # key = 机器人模型文件 basename；未列出的机器人报清晰错误而非静默错挂
        "mounts": {
            "ur5e.xml": _MOUNT_2FG7,
            "ur10e.xml": _MOUNT_2FG7,
            "xarm7_nohand.xml": _MOUNT_2FG7,
            "panda_nohand.xml": _MOUNT_2FG7,
        },
    },
}


def ensure_gripper(key: str) -> pathlib.Path:
    """返回夹爪入口 MJCF 的绝对路径（夹爪资产随仓库，无需下载）。"""
    if key not in GRIPPERS:
        raise WorldError(
            f"夹爪库没有 '{key}'，可用: {sorted(GRIPPERS)}")
    path = GRIPPER_DIR / GRIPPERS[key]["xml"]
    if not path.exists():
        raise WorldError(f"夹爪 '{key}' 的模型文件缺失: {path}")
    return path


def gripper_info(key: str) -> dict:
    if key not in GRIPPERS:
        raise WorldError(f"夹爪库没有 '{key}'，可用: {sorted(GRIPPERS)}")
    return GRIPPERS[key]


def mount_for(key: str, robot_model: str,
              override_pos=None, override_quat=None) -> dict:
    """查夹爪挂到指定机器人模型上的安装变换。

    robot_model：机器人 MJCF 路径（用 basename 匹配 mount 表）。
    override_*：YAML gripper.pos/quat 显式覆盖（高级用户/非标准臂），
    覆盖时绕过 mount 表，于是任意带挂点 site 的机器人也能手动挂。
    """
    info = gripper_info(key)
    if override_pos is not None or override_quat is not None:
        return {"pos": list(override_pos) if override_pos is not None
                else [0.0, 0.0, 0.0],
                "quat": list(override_quat) if override_quat is not None
                else [1.0, 0.0, 0.0, 0.0]}
    base = pathlib.Path(robot_model).name
    if base not in info["mounts"]:
        raise WorldError(
            f"夹爪 '{key}' 没有为机器人模型 '{base}' 标定安装位姿，"
            f"已标定: {sorted(info['mounts'])}。可在世界 YAML 的 gripper 下"
            f"显式写 pos/quat 手动指定，或在 grippers.py mount 表补一条。")
    return dict(info["mounts"][base])
