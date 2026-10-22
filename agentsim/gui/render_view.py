"""3D 视口：本地只读渲染模型 + MuJoCo 离屏渲染 + 专业 HUD 叠加。

分层：
- RenderModel（纯 Python，可无头单测）：本地 build_world 编译一份只读
  渲染模型（零 ZMQ 开销），apply_observation 把 MCP 观测写入 qpos，
  render_frame 离屏渲染出 RGB ndarray。另提供拾取（mjv_select）、
  射线-地面求交（拖动定位）、选中高亮、body 预览移动。
- RenderView(QWidget)：持有 RenderModel，鼠标控制相机 + 场景交互：
    左键点中对象 -> 选中并拖动移动（编辑态）；点空白 -> 旋转视角
    中键拖动      -> 俯仰（改变 z 向视角）
    右键/Shift+左键-> 平移；滚轮 -> 缩放
    模型库列表拖到视口 -> 发 dropModelRequested（app 添加机器人）

为什么本地再编译一份模型：渲染需要完整的 MjModel（几何/纹理），MCP
协议里传不了；观测（qpos）就是两个模型间唯一需要同步的状态。

相机射线数学（与 /tmp 定标实验一致，mujoco 3.x orbit 相机）：
  forward = (-sin(az)cos(el), -cos(az)cos(el), sin(el))
  cam_pos = lookat - dist * forward
  right   = (sin(az), -cos(az), 0)；up = cross(right, forward)
  射线方向 = fwd + right*(2*relx-1)*tan(fov/2)*aspect + up*(1-2*rely)*tan(fov/2)
  自由相机垂直 fov 恒为 45°（mjvScene.camera 默认）。
"""
from __future__ import annotations

import math
import time

import mujoco
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QListWidget, QWidget

from agentsim.gui import theme
from agentsim.world.schema import WorldSpec
from agentsim.world.builder import build_world

# 关节类型 -> qpos 维数
_JNT_WIDTH = {
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_BALL): 4,
    int(mujoco.mjtJoint.mjJNT_FREE): 7,
}

# 相机预设： (方位角°, 俯仰角°, 距离倍率)
_CAM_PRESETS = {
    "等轴视图": (135.0, -30.0, 1.0),
    "前视图": (90.0, 0.0, 0.85),
    "后视图": (-90.0, 0.0, 0.85),
    "左视图": (180.0, 0.0, 0.85),
    "右视图": (0.0, 0.0, 0.85),
    "顶视图": (90.0, -89.9, 0.9),
}

_FOV_Y_DEG = 45.0        # MuJoCo 自由相机垂直 fov（定标实验确认）


def _hex_rgb(color: str) -> tuple[float, float, float]:
    """'#RRGGBB' -> 0..1 浮点三元组。"""
    return (int(color[1:3], 16) / 255.0,
            int(color[3:5], 16) / 255.0,
            int(color[5:7], 16) / 255.0)


class RenderModel:
    """只读渲染模型：观测 -> qpos -> 离屏帧。不依赖 Qt。

    拾取/拖动索引（build_world 的 NameIndex -> 视口交互用）：
      object_bodies: body id -> 物体名
      robot_bodies : body id -> 机器人名（任意部件都归属其机器人）
      robot_roots  : 机器人名 -> 顶层 body id 列表（拖动整台机器人）
    """

    def __init__(self, spec: WorldSpec, width: int = 800, height: int = 600):
        self.model, self.index = build_world(spec)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.cam = mujoco.MjvCamera()
        # 空场景/大地面：默认取景对准地面中心并按地面尺寸拉远，
        # 否则按机械臂作业位取景（近景，利于桌面抓取）
        floor = spec.scene.floor.size
        if not spec.robots and not spec.scene.objects:
            self._default_lookat = np.array([0.0, 0.0, 0.0])
            self._default_distance = max(floor) * 1.6
        else:
            self._default_lookat = np.array([0.4, 0.15, 0.25])
            self._default_distance = 1.8
        self.cam.lookat[:] = self._default_lookat
        self.cam.distance = self._default_distance
        self.cam.azimuth = 135.0
        self.cam.elevation = -30.0

        # 可视化选项（接触点 / 接触力 / 质心 / 坐标系）
        self.opt = mujoco.MjvOption()
        self.show_wireframe = False

        # ---- 拾取/拖动索引
        self.object_bodies: dict[int, str] = {}
        self.robot_bodies: dict[int, str] = {}
        self.robot_roots: dict[str, list[int]] = {}
        for oname, bid in self.index.objects.items():
            self.object_bodies[int(bid)] = oname
        for rname, ridx in self.index.robots.items():
            self.robot_roots[rname] = []
            for bid in ridx.bodies.values():
                bid = int(bid)
                self.robot_bodies[bid] = rname
                if int(self.model.body_parentid[bid]) == 0:
                    self.robot_roots[rname].append(bid)

        # 选中高亮：body id 集合（render_frame 时向强调色着色）
        self.highlight_bodies: set[int] = set()
        # 选中目标 (kind, name)：XYZ 移动轴锚点用（与高亮集合同步设置）
        self.highlight_target: tuple[str, str] = ("", "")
        # 拖动预览：begin_move 记录初值，apply_move 施加 delta
        self._move_orig: list[tuple] = []

        self._width, self._height = width, height
        self._renderer = mujoco.Renderer(self.model, height, width)

    # ------------------------------------------------------------ 观测写入

    def apply_observation(self, obs: dict) -> None:
        """把 MCP get_observation 的 payload 写进 qpos（机器人+物体）。"""
        for rname, robs in (obs.get("robots") or {}).items():
            ridx = self.index.robots.get(rname)
            if ridx is None:
                continue
            # 浮动基座躯干 [x,y,z,qw,qx,qy,qz]：menagerie 的匿名 free joint
            # 不在 ridx.joints 里，必须单独写——否则四足/人形躯干被锁在
            # 出生位（只有腿动），视觉上就是"原地悬空奔跑"
            root = robs.get("root")
            if root is not None:
                for bid in ridx.bodies.values():
                    if int(self.model.body_parentid[bid]) != 0:
                        continue
                    n = int(self.model.body_jntnum[bid])
                    for k in range(n):
                        jid = int(self.model.body_jntadr[bid]) + k
                        if int(self.model.jnt_type[jid]) == \
                                int(mujoco.mjtJoint.mjJNT_FREE):
                            qadr = int(self.model.jnt_qposadr[jid])
                            self.data.qpos[qadr:qadr + 7] = root
            qpos = robs.get("qpos") or []
            if not qpos:
                continue
            src = 0
            for _jname, jid in ridx.joints.items():
                w = _JNT_WIDTH.get(int(self.model.jnt_type[jid]), 1)
                qadr = self.model.jnt_qposadr[jid]
                self.data.qpos[qadr:qadr + w] = qpos[src:src + w]
                src += w
        for oname, oinfo in (obs.get("objects") or {}).items():
            bid = self.index.objects.get(oname)
            if bid is None:
                continue
            jid = self.model.body_jntadr[bid]
            if jid < 0:
                continue                      # 固定场景体没有 freejoint
            qadr = self.model.jnt_qposadr[jid]
            pos, quat = oinfo.get("pos"), oinfo.get("quat")
            if pos is not None:
                self.data.qpos[qadr:qadr + 3] = pos
            if quat is not None:               # freejoint: [x y z qw qx qy qz]
                self.data.qpos[qadr + 3:qadr + 7] = quat
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------ 相机

    def set_camera_preset(self, name: str) -> None:
        az, el, dist_scale = _CAM_PRESETS[name]
        self.cam.azimuth = az
        self.cam.elevation = el
        self.cam.distance = self._default_distance * dist_scale
        self.cam.lookat[:] = self._default_lookat

    def reset_camera(self) -> None:
        self.set_camera_preset("等轴视图")

    def _cam_basis(self):
        """轨道相机基：(cam_pos, forward, right, up)（见模块 docstring）。"""
        c = self.cam
        az, el = math.radians(c.azimuth), math.radians(c.elevation)
        fwd = np.array([-math.sin(az) * math.cos(el),
                        -math.cos(az) * math.cos(el),
                        math.sin(el)])
        cam_pos = np.asarray(c.lookat, dtype=float) - c.distance * fwd
        right = np.array([math.sin(az), -math.cos(az), 0.0])
        up = np.cross(right, fwd)
        return cam_pos, fwd, right, up

    def ray_to_zplane(self, relx: float, rely: float, z0: float):
        """屏幕点 (relx, rely) 的射线与高度 z=z0 水平面求交，无交点返回 None。"""
        cam_pos, d = self.ray_dir(relx, rely, normalize=False)
        if d[2] >= -1e-9:                      # 朝上/平行：永远到不了 z0 之下
            return None
        t = (z0 - cam_pos[2]) / d[2]
        if t <= 0:
            return None
        return cam_pos + t * d

    def ray_dir(self, relx: float, rely: float, normalize: bool = True):
        """屏幕点 -> (射线起点 cam_pos, 射线方向)。相机基见模块 docstring。"""
        cam_pos, fwd, right, up = self._cam_basis()
        tan_half = math.tan(math.radians(_FOV_Y_DEG / 2))
        aspect = self._width / max(self._height, 1)
        d = (fwd
             + right * (2.0 * relx - 1.0) * tan_half * aspect
             + up * (1.0 - 2.0 * rely) * tan_half)
        if normalize:
            d = d / np.linalg.norm(d)
        return np.array(cam_pos, dtype=float), d

    # ------------------------------------------------------------ XYZ 移动轴

    def gizmo_axes(self):
        """当前选中目标的移动轴几何；无选中返回 None。

        返回 (kind, name, anchor_world, length)；锚点为 root body 世界坐标。
        """
        if not self.highlight_target:
            return None
        kind, name = self.highlight_target
        a = self.node_world_pos(kind, name)
        if a is None:
            return None
        length = max(0.08, self.cam.distance * 0.16)
        return kind, name, np.asarray(a, dtype=float), length

    def world_to_screen(self, p):
        """世界点 -> (relx, rely)；在相机后方返回 None（0=顶/左，同渲染帧）。"""
        cam_pos, fwd, right, up = self._cam_basis()
        v = np.asarray(p, dtype=float) - cam_pos
        vf = float(np.dot(v, fwd))
        if vf <= 1e-6:
            return None
        tan_half = math.tan(math.radians(_FOV_Y_DEG / 2))
        aspect = self._width / max(self._height, 1)
        relx = 0.5 + float(np.dot(v, right)) / (vf * tan_half * aspect) * 0.5
        rely = 0.5 - float(np.dot(v, up)) / (vf * tan_half) * 0.5
        return relx, rely

    def gizmo_hit(self, px: float, py: float, tol_px: float = 11.0):
        """像素坐标拾取移动轴：返回轴索引 0=X/1=Y/2=Z，未命中 None。

        屏幕点到轴投影线段的像素距离小于 tol_px 即命中（含轴端点外
        一小段余量，方便点中箭头）。
        """
        g = self.gizmo_axes()
        if g is None:
            return None
        _, _, anchor, length = g
        hit, best = None, tol_px
        for axis in range(3):
            e = np.zeros(3); e[axis] = 1.0
            sa = self.world_to_screen(anchor)
            sb = self.world_to_screen(anchor + e * length * 1.12)
            if sa is None or sb is None:
                continue
            ax0, ay0 = sa[0] * self._width, sa[1] * self._height
            ax1, ay1 = sb[0] * self._width, sb[1] * self._height
            vx, vy = ax1 - ax0, ay1 - ay0
            ln2 = vx * vx + vy * vy
            if ln2 < 1e-9:
                continue
            t = max(0.0, min(1.0,
                             ((px - ax0) * vx + (py - ay0) * vy) / ln2))
            cx, cy = ax0 + t * vx, ay0 + t * vy
            dist = math.hypot(px - cx, py - cy)
            if dist < best:
                best, hit = dist, axis
        return hit

    def ray_axis_param(self, relx: float, rely: float,
                       anchor, axis: int):
        """屏幕点射线与"过 anchor 沿世界轴的无限直线"的最近点参数 s。

        最近点 = anchor + s*e_axis；拖动时 s 的变化量即沿轴位移（米）。
        射线与轴平行（退化）返回 None。
        """
        o, d = self.ray_dir(relx, rely, normalize=True)
        e = np.zeros(3); e[axis] = 1.0
        a = np.asarray(anchor, dtype=float)
        b = float(np.dot(d, e))
        denom = 1.0 - b * b
        if denom < 1e-9:
            return None
        w0 = o - a
        # 射线 o+t d 与轴 a+s e 最近点：对 (t,s) 求导解 2x2 得
        # s = (e·w0 - (d·e)(d·w0)) / (1 - (d·e)^2)
        s = (float(np.dot(e, w0)) - b * float(np.dot(d, w0))) / denom
        return float(s)

    # ------------------------------------------------------------ 拾取/选中

    def pick(self, relx: float, rely: float) -> tuple[int, np.ndarray | None]:
        """屏幕点拾取（mjv_select）。返回 (body_id, 命中点)；未命中 (0, None)。

        入参 rely 与渲染帧一致（0=顶）；mjv_select 的 y 原点在左下角，
        传入前需翻转（实测：渲染在 rely=0.64 的物体，mjv_select 在 0.36 命中）。
        """
        self._renderer.update_scene(self.data, camera=self.cam,
                                    scene_option=self.opt)
        scn = self._renderer._scene
        selpnt = np.zeros(3)
        geomid = np.zeros(1, np.int32)
        flexid = np.zeros(1, np.int32)
        skinid = np.zeros(1, np.int32)
        bid = mujoco.mjv_select(self.model, self.data, self.opt,
                                self._width / max(self._height, 1),
                                float(relx), float(1.0 - rely), scn,
                                selpnt, geomid, flexid, skinid)
        if int(bid) <= 0:
            return 0, None
        return int(bid), selpnt.copy()

    def resolve_body(self, bid: int) -> tuple[str | None, str | None]:
        """body id -> ('object'|'robot', 名字)；不认识的返回 (None, None)。"""
        if bid in self.object_bodies:
            return "object", self.object_bodies[bid]
        if bid in self.robot_bodies:
            return "robot", self.robot_bodies[bid]
        return None, None

    def set_selection(self, kind: str, name: str) -> None:
        """设置选中高亮（scene 树联动，1.设置高亮集合 2.记录移动轴目标）。"""
        hl: set[int] = set()
        if kind == "object":
            bid = self.index.objects.get(name)
            if bid is not None:
                hl.add(int(bid))
        elif kind == "robot":
            hl.update(self.robot_roots.get(name, []))
        self.highlight_bodies = hl
        self.highlight_target = (kind, name)

    # ------------------------------------------------------------ 拖动预览

    def begin_move(self, bids) -> None:
        """记录要拖动的 body 集的初始位姿（freejoint 存 qpos，固定体存 body_pos）。"""
        self._move_orig = []
        for bid in bids:
            jid = int(self.model.body_jntadr[bid])
            is_free = (jid >= 0
                       and int(self.model.jnt_type[jid])
                       == int(mujoco.mjtJoint.mjJNT_FREE))
            if is_free:
                adr = int(self.model.jnt_qposadr[jid])
                self._move_orig.append(
                    (int(bid), True, adr,
                     np.array(self.data.qpos[adr:adr + 3])))
            else:
                self._move_orig.append(
                    (int(bid), False, -1,
                     np.array(self.model.body_pos[bid])))

    def apply_move(self, dx: float, dy: float, dz: float = 0.0) -> None:
        """把 begin_move 后的当前拖动量 (dx, dy, dz) 施加到记录的 bodies。"""
        if not self._move_orig:
            return
        for bid, is_free, adr, orig in self._move_orig:
            if is_free:
                self.data.qpos[adr] = orig[0] + dx
                self.data.qpos[adr + 1] = orig[1] + dy
                self.data.qpos[adr + 2] = orig[2] + dz
            else:
                self.model.body_pos[bid][0] = orig[0] + dx
                self.model.body_pos[bid][1] = orig[1] + dy
                self.model.body_pos[bid][2] = orig[2] + dz
        mujoco.mj_forward(self.model, self.data)

    def node_world_pos(self, kind: str, name: str):
        """对象/机器人 root body 的当前世界坐标（HUD 三维坐标显示用；
        拖动预览会实时改 qpos/body_pos -> mj_forward 后这里读到实时值）。

        world/sensor 等节点没有世界坐标，返回 None（否则 'world' 会按
        名字误命中同名机器人，HUD/gizmo 拿到错误锚点）。
        """
        if kind == "object":
            bid = self.index.objects.get(name)
            return (tuple(self.data.xpos[int(bid)])
                    if bid is not None else None)
        if kind != "robot":
            return None
        roots = self.robot_roots.get(name) or []
        if not roots:
            return None
        x = self.data.xpos[roots[0]]
        return (float(x[0]), float(x[1]), float(x[2]))

    # ------------------------------------------------------------ 渲染

    def resize(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return
        if (width, height) == (self._width, self._height):
            return
        self._width, self._height = width, height
        self._renderer.close()
        self._renderer = mujoco.Renderer(self.model, height, width)

    def render_frame(self) -> np.ndarray:
        """离屏渲染一帧 RGB（H, W, 3）uint8。"""
        self._renderer.update_scene(self.data, camera=self.cam,
                                    scene_option=self.opt)
        scene = getattr(self._renderer, "_scene", None)
        if scene is not None:
            # 线框是渲染标志（在 mjvScene 上，不在 mjvOption 上）
            scene.flags[int(mujoco.mjtRndFlag.mjRND_WIREFRAME)] = \
                self.show_wireframe
            # 选中高亮：把命中 body 的 geom 颜色向强调色混合
            if self.highlight_bodies:
                accent = _hex_rgb(theme.C_ACCENT)
                for i in range(scene.ngeom):
                    g = scene.geoms[i]
                    if (int(g.objtype) != int(mujoco.mjtObj.mjOBJ_GEOM)
                            or int(self.model.geom_bodyid[int(g.objid)])
                            not in self.highlight_bodies):
                        continue
                    r, gg, b, a = (float(g.rgba[0]), float(g.rgba[1]),
                                   float(g.rgba[2]), float(g.rgba[3]))
                    g.rgba[0] = 0.55 * accent[0] + 0.45 * r
                    g.rgba[1] = 0.55 * accent[1] + 0.45 * gg
                    g.rgba[2] = 0.55 * accent[2] + 0.45 * b
                    g.rgba[3] = a
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()


def frame_to_qimage(frame: np.ndarray) -> QImage:
    h, w, _ = frame.shape
    return QImage(frame.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)


class RenderView(QWidget):
    """Qt 薄壳：显示 RenderModel 的帧，鼠标控制相机 + 场景交互。

    鼠标（Webots 惯例）：
      左键点中对象   选中并拖动移动（编辑态才可拖；发送 nodeClicked）
      Ctrl+左键点中  选中并沿 Z 轴拖动（调高度）
      左键点空白     旋转视角
      中键拖动        俯仰（z 向视角）
      右键/Shift+左键 平移
      滚轮            缩放
    拖放：模型库 QListWidget 的条目可拖入视口 -> dropModelRequested

    信号：
      nodeClicked(kind, name)            3D 点击选中（场景树联动）
      bodyMoved(kind, name, dx, dy, dz)  拖动过程（状态栏提示用）
      bodyDropped(kind, name, dx, dy, dz)拖动松手（app 提交 YAML）
      dropModelRequested(key, relx, rely)      模型库拖放添加机器人
    """

    nodeClicked = Signal(str, str)
    bodyMoved = Signal(str, str, float, float, float)
    bodyDropped = Signal(str, str, float, float, float)
    dropModelRequested = Signal(str, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rm: RenderModel | None = None
        self._frame: QImage | None = None
        self._drag = None
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        # HUD 数据
        self._hud_state = "未加载"
        self._hud_world = ""
        self._fps_ema = 0.0
        self._last_render_t = 0.0
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # 编辑态（IDLE 才允许拖动；app 在状态切换时设置）
        self.edit_allowed = False
        # 选中保持：重建模型（load_world 后）时恢复高亮
        self._selection = ("", "")
        # XYZ 移动轴：悬停轴 / 拖动状态（axis, kind, name, s0, moved）
        self._gizmo_hover = None
        self._gizmo_axis = None

    # ------------------------------------------------------------ 模型接入

    def set_world(self, spec: WorldSpec) -> None:
        cam = None
        if self.rm is not None:                 # 保留机位，提交 YAML 后不跳变
            c = self.rm.cam
            cam = (np.array(c.lookat), c.azimuth, c.elevation, c.distance)
            self.rm.close()
        self.rm = RenderModel(spec, self.width(), max(self.height(), 1))
        if cam is not None:
            self.rm.cam.lookat[:] = cam[0]
            self.rm.cam.azimuth = cam[1]
            self.rm.cam.elevation = cam[2]
            self.rm.cam.distance = cam[3]
        self._hud_world = spec.name
        kind, name = self._selection            # 恢复选中（名字在新模型里仍有效）
        if kind:
            self.rm.set_selection(kind, name)
        self.refresh()

    def set_selection(self, kind: str, name: str) -> None:
        """场景树 -> 3D 高亮（单向，不发信号避免回环）。"""
        self._selection = (kind, name)
        if self.rm is not None:
            self.rm.set_selection(kind, name)
            self.refresh()

    def set_hud_state(self, text: str) -> None:
        self._hud_state = text
        self.update()

    def apply_observation(self, obs: dict) -> None:
        if self.rm is not None:
            self.rm.apply_observation(obs)

    def refresh(self) -> None:
        """重新渲染并重绘（观测更新后调用）。"""
        if self.rm is None:
            self._frame = None
            self.update()
            return
        self.rm.resize(self.width(), max(self.height(), 1))
        t0 = time.perf_counter()
        self._frame = frame_to_qimage(self.rm.render_frame())
        dt = time.perf_counter() - t0
        if dt > 0:
            inst = 1.0 / dt
            self._fps_ema = inst if self._fps_ema == 0 else \
                0.9 * self._fps_ema + 0.1 * inst
        self.update()

    # ------------------------------------------------------------ 可视化开关

    def set_vis_flag(self, key: str, enabled: bool) -> None:
        if self.rm is None:
            return
        if key == "wireframe":
            self.rm.show_wireframe = enabled
        elif key == "contacts":
            self.rm.opt.flags[int(mujoco.mjtVisFlag.mjVIS_CONTACTPOINT)] = enabled
        elif key == "contact_force":
            self.rm.opt.flags[int(mujoco.mjtVisFlag.mjVIS_CONTACTFORCE)] = enabled
        elif key == "com":
            self.rm.opt.flags[int(mujoco.mjtVisFlag.mjVIS_COM)] = enabled
        elif key == "frames":
            self.rm.opt.frame = int(mujoco.mjtFrame.mjFRAME_BODY) if enabled \
                else int(mujoco.mjtFrame.mjFRAME_NONE)
        self.refresh()

    def set_camera_preset(self, name: str) -> None:
        if self.rm is not None:
            self.rm.set_camera_preset(name)
            self.refresh()

    # ------------------------------------------------------------ 绘制

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#101113"))
        if self._frame is not None:
            p.drawImage(self.rect(), self._frame)
            self._draw_hud(p)
            self._draw_gizmo(p)
        else:
            p.setPen(QPen(QColor(theme.C_TEXT_DIM)))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "加载世界后此处显示 3D 视口\n\n"
                       "文件 → 打开世界，或从模型库新建\n"
                       "模型库里的机器人可以直接拖进来")

    def _draw_hud(self, p: QPainter) -> None:
        """专业仿真器风格的半透明信息叠加。"""
        font = QFont()
        font.setFamilies(["Cascadia Mono", "Consolas", "Noto Sans Mono",
                         "DejaVu Sans Mono"])
        font.setPointSize(9)
        p.setFont(font)
        fm = p.fontMetrics()

        def chip(x, y, lines, accent=None):
            """在 (x,y) 处画一块半透明信息条，返回结束 y。"""
            pad = 6
            w = max(fm.horizontalAdvance(t) for t in lines) + pad * 2
            h = fm.height() * len(lines) + pad * 2 - 2
            p.fillRect(x, y, w, h, QColor(16, 17, 19, 185))
            if accent is not None:
                p.fillRect(x, y, 3, h, QColor(accent))
            p.setPen(QPen(QColor(theme.HUD_TEXT)))
            for i, t in enumerate(lines):
                p.drawText(x + pad, y + pad + fm.ascent()
                           + i * fm.height(), t)
            return y + h + 6

        # 左上：世界名 + 运行状态（编辑态提示）
        state_line = f"状态: {self._hud_state}"
        if self.edit_allowed:
            state_line += "  [编辑模式]"
        y = chip(10, 10,
                 [f"世界: {self._hud_world}", state_line],
                 accent=theme.C_RUNNING if self._hud_state == "运行中"
                 else theme.C_ACCENT)

        # 选中对象的三维坐标（拖动中实时变化——用户拖动定位的直接参照）
        if self.rm is not None and self._selection[0]:
            wpos = self.rm.node_world_pos(*self._selection)
            if wpos is not None:
                chip(10, y, [f"选中 {self._selection[1]}  "
                             f"pos=({wpos[0]:+.3f}, {wpos[1]:+.3f}, "
                             f"{wpos[2]:+.3f})"],
                     accent=theme.C_OK)

        # 左下：相机参数 + FPS
        if self.rm is not None:
            c = self.rm.cam
            chip(10, self.height() - 58,
                 [f"az {c.azimuth:+7.1f}  el {c.elevation:+6.1f}  "
                  f"dist {c.distance:.2f}",
                  f"渲染 {self._fps_ema:5.1f} FPS"])

        # 右下：鼠标操作提示
        help_lines = ["左键 选中/拖动   拖 XYZ 轴单轴移动   Ctrl+左键 调高度(Z)",
                      "中键 俯仰   右键 平移   滚轮 缩放   F 聚焦   模型库可拖入"]
        w = max(fm.horizontalAdvance(t) for t in help_lines) + 12
        x = self.width() - w - 10
        y = self.height() - 12 - fm.height() * 2 - 8
        p.fillRect(x, y, w, fm.height() * 2 + 10, QColor(16, 17, 19, 160))
        p.setPen(QPen(QColor(theme.C_TEXT_DIM)))
        for i, t in enumerate(help_lines):
            p.drawText(x + 6, y + 6 + fm.ascent() + i * fm.height(), t)

    # ------------------------------------------------------------ 鼠标交互

    _AXIS_COLORS = ("#EF4444", "#22C55E", "#3B82F6")   # X 红 Y 绿 Z 蓝
    _AXIS_LABELS = ("X", "Y", "Z")

    def _draw_gizmo(self, p: QPainter) -> None:
        """选中机器人/物体时叠加三维 XYZ 移动轴（Blender 风格）。"""
        if (self.rm is None or not self._selection[0]
                or not self.edit_allowed):
            return
        g = self.rm.gizmo_axes()
        if g is None:
            return
        kind, name, anchor, length = g
        sa = self.rm.world_to_screen(anchor)
        if sa is None:
            return
        active = getattr(self, "_gizmo_axis", None)
        hover = getattr(self, "_gizmo_hover", None)
        font = QFont(); font.setBold(True); font.setPointSize(9)
        p.setFont(font)
        for axis in range(3):
            e = np.zeros(3); e[axis] = 1.0
            sb = self.rm.world_to_screen(anchor + e * length)
            if sb is None:
                continue
            x0, y0 = sa[0] * self.width(), sa[1] * self.height()
            x1, y1 = sb[0] * self.width(), sb[1] * self.height()
            col = QColor(self._AXIS_COLORS[axis])
            if active == axis:
                col = QColor("#FBBF24")          # 拖动中：琥珀色
            elif hover == axis:
                col = col.lighter(150)
            pen = QPen(col, 3.5 if active == axis else 2.5)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(int(x0), int(y0), int(x1), int(y1))
            # 箭头：端点两侧短线
            ang = math.atan2(y1 - y0, x1 - x0)
            ah = 9.0
            for da in (math.radians(28), -math.radians(28)):
                p.drawLine(int(x1), int(y1),
                           int(x1 - ah * math.cos(ang - da)),
                           int(y1 - ah * math.sin(ang - da)))
            # 轴标签（端点外 12px）
            lx = x1 + 13 * math.cos(ang)
            ly = y1 + 13 * math.sin(ang) + 4
            p.drawText(int(lx), int(ly), self._AXIS_LABELS[axis])

    def _relxy(self, e):
        pos = e.position()
        return (pos.x() / max(self.width(), 1),
                pos.y() / max(self.height(), 1))

    def _is_pan(self, e) -> bool:
        return (e.button() == Qt.MouseButton.RightButton
                or bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier))

    def _select(self, kind: str, name: str) -> None:
        self._selection = (kind, name)
        if self.rm is not None:
            self.rm.set_selection(kind, name)
        self.nodeClicked.emit(kind, name)

    def mousePressEvent(self, e) -> None:
        if self.rm is None:
            return
        btn = e.button()
        if btn == Qt.MouseButton.MiddleButton:
            # 中键按住拖动 -> 俯仰角（z 向视角）
            self._drag = ("elev", e.position().y(), self.rm.cam.elevation)
            return
        if self._is_pan(e):
            self._drag = ("pan", e.position().x(), e.position().y())
            return
        # 左键：编辑态优先拾取 XYZ 移动轴（命中则沿单轴拖动），否则再选物体
        relx, rely = self._relxy(e)
        if self.edit_allowed and self._selection[0]:
            axis = self.rm.gizmo_hit(e.position().x(), e.position().y())
            if axis is not None:
                g = self.rm.gizmo_axes()
                kind, name = self._selection
                bids = ([self.rm.index.objects[name]] if kind == "object"
                        else self.rm.robot_roots.get(name, []))
                if bids:
                    self.rm.begin_move(bids)
                    s0 = self.rm.ray_axis_param(relx, rely, g[2], axis)
                    if s0 is not None:
                        self._gizmo_axis = axis
                        self._drag = ("gizmo", axis, kind, name,
                                      np.array(g[2]), s0, False)
                        self.refresh()
                        return
        # 左键：先拾取——点中可拖对象则选中（编辑态进入拖动），否则旋转相机
        bid, selpnt = self.rm.pick(relx, rely)
        kind, name = self.rm.resolve_body(bid) if bid > 0 else (None, None)
        if kind is not None:
            self._select(kind, name)
            if self.edit_allowed:
                bids = ([self.rm.index.objects[name]] if kind == "object"
                        else self.rm.robot_roots.get(name, []))
                if bids:
                    self.rm.begin_move(bids)
                    if bool(e.modifiers() & Qt.KeyboardModifier.
                            ControlModifier):
                        # Ctrl+左键：Z 轴模式（调高度）——向上拖=升
                        self._drag = ("bodyz", kind, name,
                                      e.position().y(), 0.0, False)
                    else:
                        self._drag = ("body", kind, name, selpnt, 0.0, 0.0,
                                      False)
            return
        self._drag = ("rotate", e.position().x(), e.position().y(),
                      self.rm.cam.azimuth, self.rm.cam.elevation)

    def mouseMoveEvent(self, e) -> None:
        if self._drag is None or self.rm is None:
            self._update_gizmo_hover(e)
            return
        mode = self._drag[0]
        if mode == "rotate":
            _, x0, y0, az0, el0 = self._drag
            dx = e.position().x() - x0
            dy = e.position().y() - y0
            self.rm.cam.azimuth = az0 - dx * 0.4
            self.rm.cam.elevation = max(-89.0, min(89.0, el0 + dy * 0.4))
        elif mode == "elev":
            _, y0, el0 = self._drag
            dy = e.position().y() - y0
            self.rm.cam.elevation = max(-89.0, min(89.0, el0 + dy * 0.4))
        elif mode == "gizmo":
            # 沿单轴拖动：当前射线与轴直线最近点参数 - 起点参数 = 轴向位移
            _, axis, kind, name, anchor0, s0, _moved = self._drag
            relx, rely = self._relxy(e)
            s = self.rm.ray_axis_param(relx, rely, anchor0, axis)
            if s is not None:
                ds = s - s0
                delta = [0.0, 0.0, 0.0]
                delta[axis] = ds
                self.rm.apply_move(*delta)
                self._drag = ("gizmo", axis, kind, name, anchor0, s0, True)
                self.bodyMoved.emit(kind, name, *delta)
        elif mode == "body":
            _, kind, name, selpnt, dx, dy, _moved = self._drag
            relx, rely = self._relxy(e)
            p = self.rm.ray_to_zplane(relx, rely, float(selpnt[2]))
            if p is not None:
                ndx, ndy = float(p[0] - selpnt[0]), float(p[1] - selpnt[1])
                if (ndx, ndy) != (dx, dy):
                    self.rm.apply_move(ndx, ndy)
                    self._drag = ("body", kind, name, selpnt, ndx, ndy, True)
                    self.bodyMoved.emit(kind, name, ndx, ndy, 0.0)
        elif mode == "bodyz":
            # Z 轴拖动：屏幕 y 位移 -> 世界 dz（向上拖=升，与缩放距离成比例）
            _, kind, name, y0, dz, _moved = self._drag
            ndz = -(e.position().y() - y0) * self.rm.cam.distance * 0.0015
            if ndz != dz:
                self.rm.apply_move(0.0, 0.0, ndz)
                self._drag = ("bodyz", kind, name, y0, ndz, True)
                self.bodyMoved.emit(kind, name, 0.0, 0.0, ndz)
        else:                                   # pan
            _, x0, y0 = self._drag
            dx = e.position().x() - x0
            dy = e.position().y() - y0
            self._drag = ("pan", e.position().x(), e.position().y())
            # 与 MuJoCo 官方 viewer 相同的平移公式
            scale = self.rm.cam.distance * 0.0015
            rot = math.radians(self.rm.cam.azimuth)
            self.rm.cam.lookat[0] += (math.cos(rot) * dx
                                      + math.sin(rot) * dy) * scale
            self.rm.cam.lookat[1] += (-math.sin(rot) * dx
                                      + math.cos(rot) * dy) * scale
        self.refresh()

    def mouseReleaseEvent(self, _e) -> None:
        if self._drag is not None and self._drag[0] == "body" \
                and self._drag[6]:
            _, kind, name, _selpnt, dx, dy, _moved = self._drag
            self.bodyDropped.emit(kind, name, dx, dy, 0.0)
        elif self._drag is not None and self._drag[0] == "bodyz" \
                and self._drag[5]:
            _, kind, name, _y0, dz, _moved = self._drag
            self.bodyDropped.emit(kind, name, 0.0, 0.0, dz)
        elif self._drag is not None and self._drag[0] == "gizmo" \
                and self._drag[6]:
            _, axis, kind, name, anchor0, s0, _moved = self._drag
            relx = _e.position().x() / max(self.width(), 1)
            rely = _e.position().y() / max(self.height(), 1)
            s = self.rm.ray_axis_param(relx, rely, anchor0, axis)
            ds = (s - s0) if s is not None else 0.0
            delta = [0.0, 0.0, 0.0]
            delta[axis] = ds
            self.bodyDropped.emit(kind, name, *delta)
        self._gizmo_axis = None
        self._drag = None
        self.refresh()

    def _update_gizmo_hover(self, e) -> None:
        """无拖动时：鼠标悬停在轴上则高亮该轴并换光标。"""
        if self.rm is None or not self.edit_allowed or not self._selection[0]:
            if self._gizmo_hover is not None:
                self._gizmo_hover = None
                self.unsetCursor()
                self.update()
            return
        axis = self.rm.gizmo_hit(e.position().x(), e.position().y())
        if axis != self._gizmo_hover:
            self._gizmo_hover = axis
            self.setCursor(Qt.CursorShape.SizeAllCursor
                           if axis is not None
                           else Qt.CursorShape.ArrowCursor)
            self.update()

    def wheelEvent(self, e) -> None:
        if self.rm is None:
            return
        factor = 1.1 if e.angleDelta().y() > 0 else 1 / 1.1
        self.rm.cam.distance = max(0.2, self.rm.cam.distance * factor)
        self.refresh()

    def keyPressEvent(self, e) -> None:
        """F = 聚焦选中对象（相机注视点移过去；Webots/Blender 惯例）。"""
        if (e.key() == Qt.Key.Key_F and self.rm is not None
                and self._selection[0]):
            p = self.rm.node_world_pos(*self._selection)
            if p is not None:
                self.rm.cam.lookat[:3] = [p[0], p[1], p[2]]
                self.refresh()
                return
        super().keyPressEvent(e)

    # ------------------------------------------------------------ 拖放（模型库）

    def dragEnterEvent(self, e) -> None:
        if isinstance(e.source(), QListWidget):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e) -> None:
        src = e.source()
        if not isinstance(src, QListWidget):
            return
        it = src.currentItem()
        key = it.data(Qt.ItemDataRole.UserRole) if it is not None else None
        if not key:
            return
        e.acceptProposedAction()
        pos = e.position()
        relx = pos.x() / max(self.width(), 1)
        rely = pos.y() / max(self.height(), 1)
        self.dropModelRequested.emit(str(key), relx, rely)
