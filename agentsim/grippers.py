"""夹爪库：末端执行器（gripper/end-effector）注册表与装配解析。

与 zoo.py（机器人库）同构，但夹爪不是独立世界实体——它通过 MjSpec 的
site-attach 挂到某台机器人腕部的挂点 site（如 menagerie 机械臂标配的
attachment_site），其关节/执行器/tendon 经嵌套 attach 后全名变为
"<robot>/gripper/..."，自动并入该机器人的设备索引，控制器直接
robot.set("gripper/gripper_act", v) 开合。

mount 标定（见 models/grippers/onrobot_2fg7.xml 顶部注释）：
夹爪 root 在自包含 XML 中归零，相对挂点 site 的安装变换（贴合量 +
轴向翻转）全部记录在下表 mounts 里。关键事实：base STL 经 mesh 内置
旋转后，其后法兰对接面在 gripper_base body 系 +y=0.1094 处、手指沿
body -y 伸出（开合沿 ±x）；四台 menagerie 机械臂挂点的工具轴均为
site +z（UR 的 site y 沿腕部壳体、非法兰法向）。故安装朝向统一用
Rot_x(-90°)（wxyz=[0.7071,-0.7071,0,0]）：手指转向 site +z（法兰
法向）、base 后端面转向 site -z 贴合机器人法兰；x=y=0 保证两法兰
同轴。z 分量逐臂按"法兰面在 site 系的 z + 0.1094"几何标定
（ur5e/xarm7/panda 法兰面≈0，ur10e=0.017），不可共用一值。旧值
误写 Rot_x(180°) 且 z=0.1494、y=-0.028，夹爪被转到 site ±y 方向
垂在腕部壳体下方、并与法兰留有约 9cm 间隙。
"""
from __future__ import annotations

import pathlib

from agentsim.errors import WorldError

ROOT = pathlib.Path(__file__).resolve().parents[1]   # 仓库根（含 models/）
GRIPPER_DIR = ROOT / "models" / "grippers"

# 内层 attach 的固定前缀：夹爪 body/joint/actuator 去掉机器人前缀后都以
# "gripper/" 开头，避免与机器人本体同名设备冲突
GRIPPER_PREFIX = "gripper/"

#: 2FG7 安装朝向（site 局部系，wxyz）：Rot_x(-90°)，把沿 body -y 伸出的
#: 手指转到 site +z（法兰法向），四臂挂点轴约定一致，共用此朝向
_QUAT_2FG7 = [0.7071067811865476, -0.7071067811865475, 0.0, 0.0]

#: 2FG7 逐臂安装位姿（site 局部系；pos 的 z = 该臂法兰面 z + base 后端面
#: 0.1094，几何标定，见模块 docstring；x=y=0 保持两法兰同轴）
_MOUNT_UR5E = {"pos": [0.0, 0.0, 0.1094], "quat": _QUAT_2FG7}
_MOUNT_UR10E = {"pos": [0.0, 0.0, 0.1264], "quat": _QUAT_2FG7}
_MOUNT_XARM7 = {"pos": [0.0, 0.0, 0.1094], "quat": _QUAT_2FG7}
_MOUNT_PANDA = {"pos": [0.0, 0.0, 0.1092], "quat": _QUAT_2FG7}

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
            "ur5e.xml": _MOUNT_UR5E,
            "ur10e.xml": _MOUNT_UR10E,
            "xarm7_nohand.xml": _MOUNT_XARM7,
            "panda_nohand.xml": _MOUNT_PANDA,
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
