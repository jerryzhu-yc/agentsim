"""AgentSim GUI（M1）：`python -m agentsim gui` 启动。

GUI 是 MCP 客户端（与 AI Agent 同构）：spawn `python -m agentsim mcp`
子进程为仿真后端，自己只做渲染与面板。
"""
from __future__ import annotations

import os
import sys

# 离屏渲染后端：Linux 默认 EGL（无头/远程友好），Windows 默认 wgl
# （桌面 Windows 无 EGL 用户态库，wgl 是官方支持路径）。
# 必须在 import mujoco 之前设置；用户可用环境变量覆盖。
os.environ.setdefault("MUJOCO_GL",
                      "wgl" if sys.platform == "win32" else "egl")


def main(argv=None) -> int:
    from PySide6.QtWidgets import QApplication

    from agentsim.gui.app import MainWindow

    app = QApplication(argv if argv is not None else sys.argv)
    from agentsim.gui import theme
    app.setStyleSheet(theme.APP_QSS)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
