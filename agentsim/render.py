"""离屏渲染：当前物理状态 -> RGB 数组 / PNG bytes（MCP screenshot 实现层）。

纯 MuJoCo + stdlib，不 import PySide6——mcp_server 进程和测试也要能截图。
PNG 用 zlib 手写（签名/IHDR/IDAT/IEND），避免再引 Pillow。
"""
from __future__ import annotations

import struct
import zlib

import mujoco
import numpy as np

# 自由机位默认参数：与 GUI RenderModel 初始机位一致，截图与 GUI 所见对齐
_FREE_LOOKAT = (0.4, 0.15, 0.25)
_FREE_DISTANCE = 1.8
_FREE_AZIMUTH = 135.0
_FREE_ELEVATION = -30.0


def render_offscreen(model: mujoco.MjModel, data: mujoco.MjData,
                     width: int, height: int,
                     camera: str | None = None,
                     renderer: mujoco.Renderer | None = None) -> np.ndarray:
    """渲染当前状态，返回 (height, width, 3) uint8 RGB。

    - 先 mj_kinematics：刚 load / load_state 后 xpos 等派生量是过期/零值
    - camera=None：自由机位（GUI 同款默认视角）；str：模型里的相机名
      （manifest 短名需先解析成带机器人前缀的全名）
    - renderer：传入则复用（session 按 (w,h) 缓存），不传入则自建自毁
    """
    mujoco.mj_kinematics(model, data)      # 只重算运动学，不碰 warmstart
    own = renderer is None
    if own:
        renderer = mujoco.Renderer(model, height, width)
    try:
        if camera is None:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = _FREE_LOOKAT
            cam.distance = _FREE_DISTANCE
            cam.azimuth = _FREE_AZIMUTH
            cam.elevation = _FREE_ELEVATION
            renderer.update_scene(data, camera=cam)
        else:
            renderer.update_scene(data, camera=camera)
        return renderer.render()
    finally:
        if own:
            renderer.close()


def encode_png(rgb: np.ndarray) -> bytes:
    """(h, w, 3) uint8 -> PNG bytes。8bit truecolor（colortype 2），无交织。

    PNG 结构：签名 + 各 chunk（len + type + data + crc32）。
    IDAT 前每行加 filter byte 0（None 滤波），再 zlib 压缩。
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"需要 (h, w, 3) RGB 数组，实际: {rgb.shape}")
    h, w = rgb.shape[:2]

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (struct.pack(">I", len(body)) + tag + body +
                struct.pack(">I", zlib.crc32(tag + body)))

    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    # IHDR：宽、高、位深 8、colortype 2(RGB)、压缩 0、滤波 0、交织 0
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", ihdr) +
            chunk(b"IDAT", zlib.compress(raw, 6)) +
            chunk(b"IEND", b""))
