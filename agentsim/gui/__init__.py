"""AgentSim GUI（M1）：`python -m agentsim gui` 启动。

GUI 是 MCP 客户端（与 AI Agent 同构）：spawn `python -m agentsim mcp`
子进程为仿真后端，自己只做渲染与面板。
"""
from __future__ import annotations

import os
import pathlib
import sys

# 离屏渲染后端：Linux 默认 EGL（无头/远程友好），Windows 默认 wgl
# （桌面 Windows 无 EGL 用户态库，wgl 是官方支持路径）。
# 必须在 import mujoco 之前设置；用户可用环境变量覆盖。
os.environ.setdefault("MUJOCO_GL",
                      "wgl" if sys.platform == "win32" else "egl")


def mount_ime_plugin() -> None:
    """Linux 下自动挂载本地编译的 fcitx5 输入法插件（修复中文输入）。

    pip 装的 PySide6 只自带 compose/ibus 输入法插件——fcitx5 桌面
    （中文等 CJK 输入）接不进来；系统 Qt 的 fcitx 插件又因私有 ABI
    必须与运行时 Qt 严格同版本而无法借用。解法：把针对 PySide6 捆绑
    Qt 编译的 fcitx5 插件装到
    ~/.local/share/agentsim/qt-plugins/platforminputcontexts/（构建
    方法见 README「Linux 中文输入」），GUI 启动时把该目录并入
    QT_PLUGIN_PATH，并在 QT_IM_MODULE 未定时默认 fcitx。
    必须在 QApplication 创建之前调用。
    """
    if sys.platform != "linux":
        return
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return          # 无显示环境（无头/CI），挂了也没人用
    base = (pathlib.Path.home() / ".local/share/agentsim/qt-plugins"
            / "platforminputcontexts")
    if not base.is_dir() or not any(base.glob("libfcitx5*.so")):
        return          # 用户未安装本地插件（见 README）
    old = os.environ.get("QT_PLUGIN_PATH", "")
    paths = [p for p in old.split(os.pathsep) if p]
    if str(base.parent) not in paths:
        os.environ["QT_PLUGIN_PATH"] = os.pathsep.join(
            [str(base.parent), *paths])
    os.environ.setdefault("QT_IM_MODULE", "fcitx")


def main(argv=None) -> int:
    mount_ime_plugin()   # fcitx5 中文输入：必须在 QApplication 之前挂载
    from PySide6.QtWidgets import QApplication

    from agentsim.gui.app import MainWindow

    app = QApplication(argv if argv is not None else sys.argv)
    from agentsim.gui import theme
    theme.apply_dark_palette(app)      # 深色基调打底（占位符/禁用态等 palette 通道）
    app.setStyleSheet(theme.APP_QSS)   # QSS 覆盖主要控件
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
