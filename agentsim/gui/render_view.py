"""3D 视口：本地只读渲染模型 + MuJoCo 离屏渲染。

分层：
- RenderModel（纯 Python，可无头单测）：本地 build_world 编译一份只读
  渲染模型（零 ZMQ 开销），apply_observation 把 MCP 观测写入 qpos，
  render_frame 离屏渲染出 RGB ndarray
- RenderView(QWidget)：持有 RenderModel，鼠标控制相机，paintEvent 显示

为什么本地再编译一份模型：渲染需要完整的 MjModel（几何/纹理），MCP
协议里传不了；观测（qpos）就是两个模型间唯一需要同步的状态。
"""
from __future__ import annotations

import mujoco
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

from agentsim.world.schema import WorldSpec
from agentsim.world.builder import build_world

# 关节类型 -> qpos 维数
_JNT_WIDTH = {
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_BALL): 4,
    int(mujoco.mjtJoint.mjJNT_FREE): 7,
}


class RenderModel:
    """只读渲染模型：观测 -> qpos -> 离屏帧。不依赖 Qt。"""

    def __init__(self, spec: WorldSpec, width: int = 800, height: int = 600):
        self.model, self.index = build_world(spec)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.4, 0.15, 0.25]
        self.cam.distance = 1.8
        self.cam.azimuth = 135.0
        self.cam.elevation = -30.0

        self._width, self._height = width, height
        self._renderer = mujoco.Renderer(self.model, height, width)

    # ------------------------------------------------------------ 观测写入

    def apply_observation(self, obs: dict) -> None:
        """把 MCP get_observation 的 payload 写进 qpos（机器人+物体）。"""
        for rname, robs in (obs.get("robots") or {}).items():
            ridx = self.index.robots.get(rname)
            qpos = robs.get("qpos") or []
            if ridx is None or not qpos:
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
        self._renderer.update_scene(self.data, camera=self.cam)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()


def frame_to_qimage(frame: np.ndarray) -> QImage:
    h, w, _ = frame.shape
    return QImage(frame.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)


class RenderView(QWidget):
    """Qt 薄壳：显示 RenderModel 的帧，鼠标控制相机。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rm: RenderModel | None = None
        self._frame: QImage | None = None
        self._drag = None
        self.setMinimumSize(400, 300)

    # ------------------------------------------------------------ 模型接入

    def set_world(self, spec: WorldSpec) -> None:
        if self.rm is not None:
            self.rm.close()
        self.rm = RenderModel(spec, self.width(), max(self.height(), 1))
        self.refresh()

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
        self._frame = frame_to_qimage(self.rm.render_frame())
        self.update()

    # ------------------------------------------------------------ 绘制/交互

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), p.background())
        if self._frame is not None:
            p.drawImage(self.rect(), self._frame)
        else:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "加载世界后此处显示 3D 视口")

    def mousePressEvent(self, e) -> None:
        self._drag = (e.position().x(), e.position().y(),
                      self.rm.cam.azimuth if self.rm else 0,
                      self.rm.cam.elevation if self.rm else 0)

    def mouseMoveEvent(self, e) -> None:
        if self._drag is None or self.rm is None:
            return
        x0, y0, az0, el0 = self._drag
        dx = e.position().x() - x0
        dy = e.position().y() - y0
        self.rm.cam.azimuth = az0 - dx * 0.4
        self.rm.cam.elevation = max(-89.0, min(89.0, el0 + dy * 0.4))
        self.refresh()

    def mouseReleaseEvent(self, _e) -> None:
        self._drag = None

    def wheelEvent(self, e) -> None:
        if self.rm is None:
            return
        factor = 1.1 if e.angleDelta().y() > 0 else 1 / 1.1
        self.rm.cam.distance = max(0.2, self.rm.cam.distance * factor)
        self.refresh()
