"""专业机器人仿真平台风格主题（参考 Webots / MuJoCo Sim 的深色工作台）。

只提供 QSS 与少量配色常量，不依赖任何控件实例。
"""
from __future__ import annotations

# 调色板（深色 IDE 风格）
C_BG = "#1e1f22"          # 窗口底色
C_BG_ALT = "#2b2d30"      # 面板 / 停靠区
C_BG_INPUT = "#3c3f44"    # 输入框 / spinbox
C_BG_HOVER = "#36393f"    # 悬停
C_BG_PRESSED = "#40434a"
C_BORDER = "#393b40"
C_TEXT = "#dfe1e5"
C_TEXT_DIM = "#9aa0a8"
C_ACCENT = "#4a90d9"      # 主强调色（蓝）
C_ACCENT_HOVER = "#5a9fe6"
C_OK = "#59a869"          # 运行/成功 绿
C_WARN = "#d9a441"        # 暂停/警告 琥珀
C_ERR = "#d96a5f"         # 错误 红
C_RUNNING = "#59a869"

# 3D 视口叠加层文字颜色
HUD_TEXT = "#e8eaed"

APP_QSS = f"""
* {{
    font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 12px;
    color: {C_TEXT};
}}

QMainWindow, QDialog {{
    background: {C_BG};
}}

/* ---------------- 菜单栏 ---------------- */
QMenuBar {{
    background: {C_BG};
    border-bottom: 1px solid {C_BORDER};
    padding: 1px;
}}
QMenuBar::item {{
    background: transparent;
    padding: 5px 12px;
    border-radius: 3px;
}}
QMenuBar::item:selected {{ background: {C_BG_HOVER}; }}
QMenu {{
    background: {C_BG_ALT};
    border: 1px solid {C_BORDER};
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 24px 6px 14px;
    border-radius: 4px;
}}
QMenu::item:selected {{ background: {C_ACCENT}; color: white; }}
QMenu::separator {{
    height: 1px; background: {C_BORDER}; margin: 4px 8px;
}}

/* ---------------- 工具栏 ---------------- */
QToolBar {{
    background: {C_BG_ALT};
    border: none;
    border-bottom: 1px solid {C_BORDER};
    spacing: 3px;
    padding: 3px;
}}
QToolBar QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 8px;
    color: {C_TEXT};
    min-height: 22px;
}}
QToolBar QToolButton:hover {{
    background: {C_BG_HOVER};
    border-color: {C_BORDER};
}}
QToolBar QToolButton:pressed {{
    background: {C_BG_PRESSED};
}}
QToolBar QToolButton:disabled {{
    color: {C_TEXT_DIM};
}}
QToolBar::separator {{
    width: 1px; background: {C_BORDER};
    margin: 4px 6px;
}}
QToolBar QLabel {{ color: {C_TEXT_DIM}; padding: 0 2px; }}

/* ---------------- 通用按钮 ---------------- */
QPushButton {{
    background: {C_BG_INPUT};
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    padding: 5px 14px;
    color: {C_TEXT};
}}
QPushButton:hover {{ background: {C_BG_HOVER}; border-color: {C_ACCENT}; }}
QPushButton:pressed {{ background: {C_BG_PRESSED}; }}
QPushButton:disabled {{ color: {C_TEXT_DIM}; background: {C_BG_ALT}; }}
QPushButton#primary {{
    background: {C_ACCENT}; border-color: {C_ACCENT}; color: white;
}}
QPushButton#primary:hover {{ background: {C_ACCENT_HOVER}; }}

/* ---------------- 输入控件 ---------------- */
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
    background: {C_BG_INPUT};
    border: 1px solid {C_BORDER};
    border-radius: 3px;
    padding: 3px 6px;
    selection-background-color: {C_ACCENT};
}}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
    border-color: {C_ACCENT};
}}
QComboBox QAbstractItemView {{
    background: {C_BG_ALT};
    border: 1px solid {C_BORDER};
    selection-background-color: {C_ACCENT};
}}
QCheckBox {{ spacing: 6px; color: {C_TEXT}; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {C_BORDER};
    border-radius: 3px;
    background: {C_BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background: {C_ACCENT};
    border-color: {C_ACCENT};
}}

/* ---------------- 停靠面板 ---------------- */
QDockWidget {{
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}
QDockWidget::title {{
    background: {C_BG_ALT};
    border-bottom: 1px solid {C_BORDER};
    padding: 6px 10px;
    font-weight: 600;
    color: {C_TEXT_DIM};
}}
QGroupBox {{
    border: 1px solid {C_BORDER};
    border-radius: 5px;
    margin-top: 12px;
    padding: 8px 8px 6px 8px;
    background: {C_BG_ALT};
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: {C_ACCENT};
}}

/* ---------------- 树 / 列表 / 表格 ---------------- */
QTreeWidget, QListWidget, QTableWidget {{
    background: {C_BG};
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    alternate-background-color: {C_BG_ALT};
    outline: none;
}}
QTreeWidget::item, QListWidget::item {{
    padding: 2px 2px;
    min-height: 20px;
}}
QTreeWidget::item:selected, QListWidget::item:selected {{
    background: {C_ACCENT}; color: white;
}}
QHeaderView::section {{
    background: {C_BG_ALT};
    color: {C_TEXT_DIM};
    border: none;
    border-bottom: 1px solid {C_BORDER};
    border-right: 1px solid {C_BORDER};
    padding: 4px 8px;
    font-weight: 600;
}}

/* ---------------- 标签 / 表单 ---------------- */
QLabel {{ background: transparent; }}
QFormLayout QLabel {{ color: {C_TEXT_DIM}; }}

/* ---------------- 选项卡 ---------------- */
QTabWidget::pane {{
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    background: {C_BG};
}}
QTabBar::tab {{
    background: {C_BG_ALT};
    border: 1px solid {C_BORDER};
    border-bottom: none;
    padding: 6px 14px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    color: {C_TEXT_DIM};
}}
QTabBar::tab:selected {{
    background: {C_BG};
    color: {C_TEXT};
    border-bottom: 2px solid {C_ACCENT};
}}

/* ---------------- 分割条 ---------------- */
QSplitter::handle {{ background: {C_BORDER}; }}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical {{ height: 2px; }}

/* ---------------- 滚动条 ---------------- */
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #4a4d53; min-height: 28px; border-radius: 5px;
}}
QScrollBar::handle:vertical:hover {{ background: #5a5e65; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent; height: 10px; margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: #4a4d53; min-width: 28px; border-radius: 5px;
}}
QScrollBar::handle:horizontal:hover {{ background: #5a5e65; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ---------------- 状态栏 ---------------- */
QStatusBar {{
    background: {C_BG_ALT};
    border-top: 1px solid {C_BORDER};
    color: {C_TEXT_DIM};
}}
QStatusBar QLabel {{ padding: 0 6px; color: {C_TEXT_DIM}; }}
QStatusBar::item {{ border: none; }}

/* ---------------- 工具提示 ---------------- */
QToolTip {{
    background: {C_BG_INPUT};
    color: {C_TEXT};
    border: 1px solid {C_ACCENT};
    padding: 4px 7px;
    border-radius: 3px;
}}

/* ---------------- 控制台 ---------------- */
QPlainTextEdit#console, QTextEdit#console {{
    background: #18191c;
    border: 1px solid {C_BORDER};
    border-radius: 4px;
    font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
    font-size: 11.5px;
    color: {C_TEXT};
}}
"""


def state_color(state: str) -> str:
    return {
        "empty": C_TEXT_DIM,
        "idle": C_TEXT,
        "running": C_RUNNING,
        "finished": C_ACCENT,
    }.get(state, C_TEXT_DIM)


def state_text(state: str) -> str:
    return {
        "empty": "未加载",
        "idle": "就绪",
        "running": "运行中",
        "finished": "已结束",
    }.get(state, state)
