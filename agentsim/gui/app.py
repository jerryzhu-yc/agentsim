"""主窗口：参考 Webots 的专业机器人仿真工作台。

布局：
  菜单栏   文件 / 仿真 / 视图 / 帮助
  工具栏   打开世界、模型库 | Reset、Run、步进 N、步数、实时 |
           机位、接触点、坐标系、线框 | 导出截图
  中央     3D 视口（HUD：世界名/状态/相机/FPS/操作提示）
  左停靠   场景树（世界/机器人/物体层级，选中联动右侧属性）
  右停靠   属性（选中节点详情）+ 任务状态 + 观测树（标签页）
  底停靠   控制台（GUI 事件 + 后端日志，带级别着色）
  状态栏   运行状态指示灯、世界名、tick

信号流（GUI 与 AI Agent 同构，都是 MCP 客户端）：
  Load   -> 本地解析 spec（渲染模型用）+ load_world RPC -> IDLE
  Run    -> run RPC（延迟响应）-> RUNNING（30Hz 拉快照刷新视口/面板）
         -> run 响应到达 -> FINISHED
  Reset  -> reset RPC -> IDLE
  FINISHED 下点 Run/步进 -> 自动 reset 再 run（episode 是一次性的）
"""
from __future__ import annotations

import numpy as np
import yaml
from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                               QDialog, QDockWidget, QFileDialog, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QPushButton,
                               QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from agentsim.gui import theme
from agentsim.gui.mcp_client import McpClient
from agentsim.gui.panels import (AgentConfigDialog, AgentPanel, CodePanel,
                                 ConsolePanel, ObsPanel, PropertiesPanel,
                                 SceneTreePanel)
from agentsim.gui.render_view import RenderView, _CAM_PRESETS
from agentsim.gui.state import UiState
from agentsim.world.schema import load_world as parse_world_file
from agentsim.world.schema import parse_world


class ZooDialog(QDialog):
    """机器人模型库（MuJoCo Menagerie 官方模型），非模态。

    两种用法：
    - 列表条目按住直接拖到 3D 视口 -> 在落点添加该机器人（addRequested）
    - 选中后"生成并打开" -> 完整项目（世界+控制器模板，accept 信号）
    """

    addRequested = Signal(str)          # "添加到当前场景"按钮
    demoRequested = Signal(str)         # "生成演示并运行"按钮

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("机器人模型库（MuJoCo Menagerie）")
        self.resize(620, 560)
        lay = QVBoxLayout(self)
        tip = QLabel("按住条目拖到 3D 视口即可在落点放置机器人（编辑模式下）；"
                     "或选中后\"添加到当前场景\"/\"生成并打开\"完整项目。"
                     "首次使用某模型会联网下载一次（之后永久离线）。")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        self.listw = QListWidget()
        self.listw.setDragEnabled(True)     # 拖到 3D 视口放置
        self.listw.setDragDropMode(
            QAbstractItemView.DragDropMode.DragOnly)
        lay.addWidget(self.listw)
        row = QHBoxLayout()
        btn_add = QPushButton("添加到当前场景")
        btn_add.setToolTip("把选中的机器人加到当前世界（放在相机注视点）")
        btn_add.clicked.connect(self._emit_add)
        row.addWidget(btn_add)
        row.addStretch(1)
        btn_demo = QPushButton("生成演示并运行")
        btn_demo.setToolTip("生成该机器人的运动演示（四足小跑/臂扫掠/摆臂）"
                            "并立即运行——直接看动画")
        btn_demo.clicked.connect(self._emit_demo)
        row.addWidget(btn_demo)
        btn = QPushButton("生成并打开")
        btn.setObjectName("primary")
        btn.setDefault(True)
        btn.clicked.connect(self.accept)
        row.addWidget(btn)
        lay.addLayout(row)
        self.listw.itemDoubleClicked.connect(lambda *_: self.accept())
        self._populate()

    def _emit_demo(self):
        key = self.selected_key
        if key:
            self.demoRequested.emit(key)

    def _emit_add(self):
        key = self.selected_key
        if key:
            self.addRequested.emit(key)

    def _populate(self):
        from agentsim.zoo import MODELS
        for key, m in MODELS.items():
            it = QListWidgetItem(f"{m['title']}  [{m['cat']}] —— {m['desc']}")
            it.setData(Qt.ItemDataRole.UserRole, key)
            self.listw.addItem(it)
        self.listw.setCurrentRow(0)

    @property
    def selected_key(self) -> str | None:
        it = self.listw.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else None


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AgentSim — 机器人仿真平台")
        self.resize(1440, 900)
        self.state = UiState()

        # ---- 编辑态数据：当前世界 spec / YAML 基准目录 / 模型库对话框
        self._spec = None
        self._base_dir = None
        self._zoo_dlg: ZooDialog | None = None
        self._busy = False              # load_world 在途（防重入提交）

        # ---- 后端 MCP 客户端
        self.client = McpClient(self)
        self.client.start()
        self.client.backendDied.connect(self._on_backend_died)
        self.client.backendLog.connect(self._on_backend_log)

        # ---- 中央 3D 视口
        self.view = RenderView()
        self.setCentralWidget(self.view)
        self.view.nodeClicked.connect(self._on_view_node_clicked)
        self.view.bodyMoved.connect(self._on_body_moved)
        self.view.bodyDropped.connect(self._on_body_dropped)
        self.view.dropModelRequested.connect(self._on_drop_model)

        # ---- 面板
        self.scene_tree = SceneTreePanel()
        self.scene_tree.nodeSelected.connect(self._on_node_selected)
        self.props = PropertiesPanel()
        self.props.poseApplied.connect(self._on_pose_applied)
        self.obs = ObsPanel()
        self.console = ConsolePanel()
        self.code = CodePanel()
        self.code.saveRequested.connect(self._on_code_saved)
        self.agent = AgentPanel()
        self.agent.sendRequested.connect(self._on_agent_send)
        self.agent.stopRequested.connect(self._on_agent_stop)
        self.agent.configureRequested.connect(self._on_agent_config)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.props, "属性")
        self.tabs.addTab(self.obs, "观测")
        self.tabs.addTab(self.code, "代码")
        self.tabs.addTab(self.agent, "Agent")

        # Agent worker：HTTP + 工具循环跑在 QThread
        from agentsim.agent import AgentWorker, load_config
        self._agent_cfg = load_config()
        self.agent_worker = AgentWorker(self.client, self)
        self.agent_worker.assistantText.connect(self.agent.append_assistant)
        self.agent_worker.toolCall.connect(self.agent.append_tool)
        self.agent_worker.toolResult.connect(self.agent.append_result)
        self.agent_worker.error.connect(self.agent.append_error)
        self.agent_worker.error.connect(lambda t: self._log(t, "error"))
        self.agent_worker.finishedRound.connect(self._on_agent_round_done)
        self._refresh_agent_state(busy=False)
        self.agent_worker.start()

        self.dock_scene = self._make_dock("场景树", self.scene_tree,
                                          Qt.DockWidgetArea.LeftDockWidgetArea)
        self.dock_right = self._make_dock("属性 / 观测", self.tabs,
                                          Qt.DockWidgetArea.RightDockWidgetArea)
        self.dock_console = self._make_dock("控制台", self.console,
                                            Qt.DockWidgetArea.BottomDockWidgetArea)
        self.dock_right.setMinimumWidth(300)
        self.resizeDocks([self.dock_scene, self.dock_right], [280, 340],
                         Qt.Orientation.Horizontal)
        self.resizeDocks([self.dock_console], [200], Qt.Orientation.Vertical)

        self._build_actions()
        self._build_menubar()
        self._build_toolbar()
        self._build_statusbar()

        # ---- 握手即激活
        self.client.call_tool("ping", {}, lambda e, d: None)

        # ---- 30Hz 观测轮询
        self._obs_pending = False
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._poll)
        self.timer.start()

        self._apply_state("就绪。文件 → 打开世界（worlds/ 下有示例），"
                          "或从模型库新建。")
        self.console.info("GUI 已启动，等待加载世界。")

    # ============================================================ 界面构建

    def _make_dock(self, title: str, widget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.addDockWidget(area, dock)
        return dock

    def _build_actions(self) -> None:
        def mk(text, shortcut=None, tip=None, checkable=False):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip:
                a.setToolTip(tip)
            a.setCheckable(checkable)
            return a

        self.act_open = mk("打开世界…", "Ctrl+O", "加载一个世界 YAML 文件")
        self.act_zoo = mk("机器人模型库…", "Ctrl+M", "从官方模型库生成项目")
        self.act_quit = mk("退出", "Ctrl+Q")
        self.act_quit.triggered.connect(self.close)

        self.act_reset = mk("重置", "F8", "世界回到 YAML 初态（重编译+重启控制器）")
        self.act_run = mk("运行", "F5", "运行到终态（控制器退出/max_ticks）")
        self.act_step = mk("步进", "F7", "推进指定步数后暂停（N 取自工具栏）")
        self.act_screenshot = mk("导出截图…", "Ctrl+S", "把当前视口另存为 PNG")

        self.act_vis_contacts = mk("接触点", checkable=True)
        self.act_vis_contacts.setToolTip("显示接触点")
        self.act_vis_force = mk("接触力", checkable=True)
        self.act_vis_force.setToolTip("显示接触力")
        self.act_vis_com = mk("质心", checkable=True)
        self.act_vis_com.setToolTip("显示各 body 质心")
        self.act_vis_frames = mk("坐标系", checkable=True)
        self.act_vis_frames.setToolTip("显示 body 坐标系")
        self.act_vis_wire = mk("线框", checkable=True)
        self.act_vis_wire.setToolTip("以线框模式渲染")
        self.act_cam_reset = mk("重置机位", "Ctrl+0")

        # 控件（放进工具栏）
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 100000)
        self.spin_n.setValue(100)
        self.spin_n.setToolTip("步进/Run N 的 tick 数（F7）")
        self.spin_n.setFixedWidth(80)
        self.chk_realtime = QCheckBox("实时")
        self.chk_realtime.setChecked(True)
        self.chk_realtime.setToolTip("勾选：按墙钟实时播放（动画连贯）\n"
                                     "取消：全速推进（瞬间出结果）")
        self.cam_combo = QComboBox()
        self.cam_combo.addItems(list(_CAM_PRESETS.keys()))
        self.cam_combo.setToolTip("相机机位")
        self.cam_combo.currentTextChanged.connect(self.view.set_camera_preset)

    def _build_menubar(self) -> None:
        mb = self.menuBar()

        m_file = mb.addMenu("文件(&F)")
        m_file.addAction(self.act_open)
        m_file.addAction(self.act_zoo)
        m_file.addSeparator()
        m_file.addAction(self.act_screenshot)
        m_file.addSeparator()
        m_file.addAction(self.act_quit)

        m_sim = mb.addMenu("仿真(&S)")
        m_sim.addAction(self.act_reset)
        m_sim.addAction(self.act_run)
        m_sim.addAction(self.act_step)
        m_sim.addSeparator()
        rt = m_sim.addAction("实时播放（按墙钟）")
        rt.setCheckable(True)
        rt.setChecked(True)
        rt.triggered.connect(self.chk_realtime.setChecked)
        self.chk_realtime.toggled.connect(rt.setChecked)
        self._rt_menu_action = rt

        m_view = mb.addMenu("视图(&V)")
        m_cam = m_view.addMenu("机位")
        for name in _CAM_PRESETS:
            a = m_cam.addAction(name)
            a.triggered.connect(lambda _=False, n=name: self._select_cam(n))
        m_view.addAction(self.act_cam_reset)
        m_view.addSeparator()
        m_view.addAction(self.act_vis_contacts)
        m_view.addAction(self.act_vis_force)
        m_view.addAction(self.act_vis_com)
        m_view.addAction(self.act_vis_frames)
        m_view.addAction(self.act_vis_wire)
        m_view.addSeparator()
        for dock in (self.dock_scene, self.dock_right, self.dock_console):
            m_view.addAction(dock.toggleViewAction())

        m_help = mb.addMenu("帮助(&H)")
        a_about = m_help.addAction("关于 AgentSim")
        a_about.triggered.connect(self._on_about)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("主工具栏")
        tb.setMovable(False)
        tb.setIconSize(tb.iconSize())
        tb.addAction(self.act_open)
        tb.addAction(self.act_zoo)
        tb.addSeparator()
        tb.addAction(self.act_reset)
        tb.addAction(self.act_run)
        tb.addAction(self.act_step)
        tb.addWidget(QLabel(" 步数 "))
        tb.addWidget(self.spin_n)
        tb.addWidget(self.chk_realtime)
        tb.addSeparator()
        tb.addWidget(QLabel("机位 "))
        tb.addWidget(self.cam_combo)
        tb.addSeparator()
        tb.addAction(self.act_vis_contacts)
        tb.addAction(self.act_vis_force)
        tb.addAction(self.act_vis_com)
        tb.addAction(self.act_vis_frames)
        tb.addAction(self.act_vis_wire)
        tb.addSeparator()
        tb.addAction(self.act_screenshot)

        # 信号
        self.act_open.triggered.connect(self._on_load)
        self.act_zoo.triggered.connect(self._on_zoo)
        self.act_reset.triggered.connect(self._on_reset)
        self.act_run.triggered.connect(self._on_run)
        self.act_step.triggered.connect(self._on_run_n)
        self.act_screenshot.triggered.connect(self._on_screenshot)
        self.act_cam_reset.triggered.connect(
            lambda: self._select_cam("等轴视图"))
        self.act_vis_contacts.toggled.connect(
            lambda v: self.view.set_vis_flag("contacts", v))
        self.act_vis_force.toggled.connect(
            lambda v: self.view.set_vis_flag("contact_force", v))
        self.act_vis_com.toggled.connect(
            lambda v: self.view.set_vis_flag("com", v))
        self.act_vis_frames.toggled.connect(
            lambda v: self.view.set_vis_flag("frames", v))
        self.act_vis_wire.toggled.connect(
            lambda v: self.view.set_vis_flag("wireframe", v))

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self.lbl_dot = QLabel("●")
        self.lbl_state = QLabel("未加载")
        self.lbl_world = QLabel("世界：—")
        self.lbl_tick = QLabel("tick —")
        sb.addWidget(self.lbl_dot)
        sb.addWidget(self.lbl_state)
        sb.addWidget(self._vline())
        sb.addWidget(self.lbl_world, stretch=1)
        sb.addPermanentWidget(self.lbl_tick)

    @staticmethod
    def _vline() -> QLabel:
        lb = QLabel("│")
        lb.setStyleSheet(f"color:{theme.C_BORDER};")
        return lb

    # ============================================================ 状态/日志

    def _log(self, msg: str, level: str = "info") -> None:
        getattr(self.console, level, self.console.info)(msg)

    def _apply_state(self, msg: str = "") -> None:
        s = self.state.value
        # 编辑模式：仅 IDLE（世界已加载且未被 run 改动）允许拖动改位
        self.view.edit_allowed = (s == UiState.IDLE)
        self.act_open.setEnabled(s != UiState.RUNNING)
        self.act_zoo.setEnabled(s != UiState.RUNNING)
        self.act_reset.setEnabled(self.state.can("reset"))
        self.act_run.setEnabled(self.state.can("run"))
        self.act_step.setEnabled(self.state.can("run"))
        self.spin_n.setEnabled(s != UiState.RUNNING)
        # 状态灯 + 文字
        color = theme.state_color(s)
        self.lbl_dot.setStyleSheet(f"color:{color}; font-size:13px;")
        self.lbl_state.setText(theme.state_text(s))
        if msg:
            self.statusBar().showMessage(msg)

    # ============================================================ 动作：文件

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择世界文件", "worlds", "World YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            spec = parse_world_file(path)          # 本地解析（渲染模型用）
        except Exception as e:
            self._apply_state(f"世界文件错误: {e}")
            self._log(f"世界文件错误: {e}", "error")
            return
        self._log(f"加载世界：{path}")
        self.client.call_tool(
            "load_world", {"yaml_path": path},
            lambda e, d: self._on_loaded(e, d, spec))

    def _on_screenshot(self):
        if self.view._frame is None:
            QMessageBox.information(self, "导出截图", "还没有可保存的视口画面。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出截图", "screenshot.png", "PNG 图像 (*.png)")
        if not path:
            return
        if self.view._frame.save(path, "PNG"):
            self._log(f"截图已保存：{path}", "ok")
            self.statusBar().showMessage(f"截图已保存：{path}")
        else:
            self._log("截图保存失败", "error")

    def _on_about(self):
        QMessageBox.about(
            self, "关于 AgentSim",
            "<h3>AgentSim</h3>"
            "<p>轻量多机器人仿真平台（MuJoCo + 多进程控制器）。</p>"
            "<p>快捷键：F5 运行 · F7 步进 · F8 重置 · Ctrl+O 打开世界</p>")

    # ============================================================ 模型库

    def _on_zoo(self):
        """非模态模型库：可拖条目到 3D 视口放置，或选中后生成/添加。"""
        if self._zoo_dlg is None:
            self._zoo_dlg = ZooDialog(self)
            self._zoo_dlg.addRequested.connect(self._add_robot_center)
            self._zoo_dlg.demoRequested.connect(self._on_zoo_demo)
            self._zoo_dlg.accepted.connect(self._on_zoo_generate)
        self._zoo_dlg.show()
        self._zoo_dlg.raise_()
        self._zoo_dlg.activateWindow()

    def _on_zoo_demo(self, key: str):
        """模型库"生成演示并运行"：demo 项目 + 加载后自动 Run 看动画。"""
        if self.state.value == UiState.RUNNING or self._busy:
            self._log("运行中不能生成演示（先等 run 结束或 reset）", "warn")
            return
        self._apply_state(f"模型库：生成 {key} 运动演示中（首次使用需联网下载）…")
        self._log(f"从模型库生成运动演示：{key}")
        self.client.call_tool("zoo_create", {"key": key, "demo": True},
                              self._on_zoo_created)

    def _on_zoo_generate(self):
        """生成完整项目：世界 + 控制器模板（旧 exec 流程改为 accept 信号）。"""
        if self._zoo_dlg is None or not self._zoo_dlg.selected_key:
            return
        key = self._zoo_dlg.selected_key
        self._apply_state(f"模型库：生成 {key} 中（首次使用需联网下载）…")
        self._log(f"从模型库生成：{key}（首次需联网下载）")
        self.client.call_tool("zoo_create", {"key": key}, self._on_zoo_created)

    def _add_robot_center(self, key: str):
        """模型库"添加到当前场景"：放在相机注视点。"""
        self._add_robot_at(key, screen=None, note="相机注视点")

    def _on_drop_model(self, key: str, relx: float, rely: float):
        """模型库条目拖到 3D 视口落点：在落点对应的世界坐标放置机器人。"""
        self._add_robot_at(key, screen=(relx, rely), note="拖放落点")

    def _add_robot_at(self, key: str, screen=None, note: str = ""):
        if self.state.value == UiState.RUNNING or self._busy:
            self._log("运行中不能添加机器人（先等 run 结束或 reset）", "warn")
            return
        if self._spec is None:
            self._log("先加载一个世界，再把机器人加进场景", "warn")
            return
        self._apply_state(f"添加 {key} 到场景（首次使用需联网下载）…")
        self._log(f"从模型库添加机器人：{key}（位置：{note or '默认'}）")

        def ensured(err, data):
            if err:
                self._apply_state(f"添加失败: {data}")
                self._log(f"zoo_create(ensure) 失败: {data}", "error")
                return
            # 落点：拖放射线与地面求交；否则相机注视点
            p = None
            if screen is not None and self.view.rm is not None:
                p = self.view.rm.ray_to_zplane(screen[0], screen[1], 0.0)
            if p is None:
                p = (np.array(self.view.rm.cam.lookat[:2])
                     if self.view.rm is not None else np.zeros(2))
            self._append_robot_yaml(key, data["xml"],
                                    float(p[0]), float(p[1]),
                                    float(data["pos"][2]))

        self.client.call_tool(
            "zoo_create", {"key": key, "ensure_only": True}, ensured)

    def _unique_name(self, base: str) -> str:
        """生成不与现有机器人/物体冲突的名字（go2 / go2_2 / go2_3 …）。"""
        taken = {r.name for r in self._spec.robots}
        taken |= {o.name for o in self._spec.scene.objects}
        if base not in taken:
            return base
        i = 2
        while f"{base}_{i}" in taken:
            i += 1
        return f"{base}_{i}"

    def _append_robot_yaml(self, key: str, xml: str,
                           x: float, y: float, z: float):
        """读当前世界 YAML -> 追加机器人 -> load_world 提交（整链复用）。"""
        def got_yaml(err, text):
            if err:
                self._apply_state(f"read_world 失败: {text}")
                self._log(f"read_world 失败: {text}", "error")
                return
            try:
                d = yaml.safe_load(text)
                robots = d.setdefault("robots", [])
                rd = {"name": self._unique_name(key), "model": xml,
                      "pos": [round(x, 4), round(y, 4), round(z, 4)]}
                robots.append(rd)
                new_text = yaml.safe_dump(d, allow_unicode=True, sort_keys=False)
            except Exception as e:
                self._apply_state(f"YAML 修改失败: {e}")
                self._log(f"YAML 修改失败: {e}", "error")
                return
            name = rd["name"]
            self._reload_yaml(
                new_text,
                ok_msg=f"已添加机器人 {name} 于 "
                       f"({rd['pos'][0]:+.2f}, {rd['pos'][1]:+.2f}, {z:+.2f})")
            self._log(f"添加机器人 {name}（模型 {key}，未绑定控制器；"
                      f"需要控制律请用 模型库 → 生成并打开）", "ok")
        self.client.call_tool("read_world", {}, got_yaml)

    # ============================================================ 编辑提交

    def _on_view_node_clicked(self, kind: str, name: str):
        """3D 点击 -> 场景树反选（树会再发 nodeSelected 联动属性面板）。"""
        self.scene_tree.select_node(kind, name)

    def _on_body_moved(self, kind: str, name: str, dx: float, dy: float):
        self.statusBar().showMessage(
            f"移动 {name}: dx={dx:+.2f}  dy={dy:+.2f}（松手提交到世界 YAML）")

    def _on_body_dropped(self, kind: str, name: str, dx: float, dy: float):
        """拖动松手：read_world -> 改 pos -> load_world(yaml_text) 提交。"""
        if self._spec is None or self.state.value != UiState.IDLE or self._busy:
            return
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return

        def got_yaml(err, text):
            if err:
                self._apply_state(f"read_world 失败: {text}")
                self._log(f"read_world 失败: {text}", "error")
                return
            try:
                d = yaml.safe_load(text)
                moved = False
                if kind == "robot":
                    for r in d.get("robots") or []:
                        if r.get("name") == name:
                            p = list(r.get("pos") or [0, 0, 0])
                            p = [p[0] + dx, p[1] + dy, p[2]]
                            r["pos"] = [round(v, 4) for v in p]
                            moved = True
                            break
                elif kind == "object":
                    for o in (d.get("scene") or {}).get("objects") or []:
                        if o.get("name") == name:
                            p = list(o.get("pos") or [0, 0, 0])
                            p = [p[0] + dx, p[1] + dy, p[2]]
                            o["pos"] = [round(v, 4) for v in p]
                            moved = True
                            break
                if not moved:
                    self._log(f"世界 YAML 里没找到 {kind} '{name}'", "warn")
                    return
                new_text = yaml.safe_dump(d, allow_unicode=True, sort_keys=False)
            except Exception as e:
                self._apply_state(f"YAML 修改失败: {e}")
                self._log(f"YAML 修改失败: {e}", "error")
                return
            self._pending_select = (kind, name)   # 重载后重选（属性面板同步）
            self._reload_yaml(
                new_text,
                ok_msg=f"{name} 移动 ({dx:+.2f}, {dy:+.2f}) 已写入世界 YAML")

        self.client.call_tool("read_world", {}, got_yaml)

    def _on_pose_applied(self, name: str, pos: list, quat):
        """属性面板"应用位姿"：read_world -> 写 pos/quat -> load_world 提交。"""
        if self._spec is None or self.state.value != UiState.IDLE or self._busy:
            return

        def got_yaml(err, text):
            if err:
                self._apply_state(f"read_world 失败: {text}")
                self._log(f"read_world 失败: {text}", "error")
                return
            try:
                d = yaml.safe_load(text)
                rd = next((r for r in d.get("robots") or []
                           if r.get("name") == name), None)
                if rd is None:
                    self._log(f"世界 YAML 里没找到机器人 '{name}'", "warn")
                    return
                rd["pos"] = pos
                if quat is None:              # 单位朝向不落盘（YAML 干净）
                    rd.pop("quat", None)
                else:
                    rd["quat"] = quat
                new_text = yaml.safe_dump(d, allow_unicode=True, sort_keys=False)
            except Exception as e:
                self._apply_state(f"YAML 修改失败: {e}")
                self._log(f"YAML 修改失败: {e}", "error")
                return
            self._pending_select = ("robot", name)
            msg = (f"{name} 位姿已写入：pos=({pos[0]:+.3f}, {pos[1]:+.3f}, "
                   f"{pos[2]:+.3f})" + ("" if quat is None else
                                       f" quat=({', '.join(f'{v:.4f}' for v in quat)})"))
            self._reload_yaml(new_text, ok_msg=msg)

        self.client.call_tool("read_world", {}, got_yaml)

    # ============================================================ Agent

    def _refresh_agent_state(self, busy: bool) -> None:
        self.agent.set_state(configured=self._agent_cfg.usable, busy=busy)

    def _on_agent_send(self, text: str):
        """用户在 Agent 输入框发消息：先回显，再让 worker 处理。"""
        if not self._agent_cfg.usable:
            self._on_agent_config()      # 未配置直接弹配置对话框
            if not self._agent_cfg.usable:
                return
        self.agent.append_user(text)
        self._refresh_agent_state(busy=True)
        self.agent_worker.send(text)

    def _on_agent_stop(self):
        self.agent_worker.stop()

    def _on_agent_round_done(self):
        self._refresh_agent_state(busy=False)

    def _on_agent_config(self):
        dlg = AgentConfigDialog(self)
        if dlg.exec() != AgentConfigDialog.DialogCode.Accepted:
            return
        from agentsim.agent import save_config
        self._agent_cfg = dlg.config()
        save_config(self._agent_cfg)
        # 应用到 worker（下一轮生效）+ 重置历史（system prompt 重注）
        try:
            from agentsim.mcp_server import _TOOLS
            self.agent_worker.configure(self._agent_cfg, _TOOLS)
        except ImportError:
            pass
        if self._spec is not None:
            self.agent_worker.set_history(self._spec.name)
        self._refresh_agent_state(busy=False)
        self._log("Agent 配置已保存", "ok")

    def _on_code_saved(self, robot_name: str, text: str):
        """代码页"保存并运行"：写回控制器文件 -> read_world 重载 -> 自动 Run。

        load_world 会重启控制器进程，新代码即生效（World as Code 同款机制）。
        Ctrl+S 在编辑框聚焦时也走这里（QShortcut 已绑）。
        """
        if self._spec is None or self.state.value != UiState.IDLE or self._busy:
            return
        rd = next((r for r in self._spec.robots if r.name == robot_name), None)
        if rd is None or not rd.controller:
            self._log(f"机器人 {robot_name} 没有控制器路径，无法保存", "warn")
            return
        import pathlib
        path = (pathlib.Path(self._spec.base_dir) / rd.controller).resolve()
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as e:
            self._apply_state(f"写文件失败: {e}")
            self._log(f"写 {path} 失败: {e}", "error")
            return
        self._pending_select = ("robot", robot_name)
        self._pending_run = True
        self._log(f"已保存 {path.name}，重载世界后自动运行…", "ok")

        def got_yaml(err, yaml_text):
            if err:
                self._apply_state(f"read_world 失败: {yaml_text}")
                self._log(f"read_world 失败: {yaml_text}", "error")
                return
            self._reload_yaml(yaml_text, ok_msg=f"{path.name} 已保存并重载")

        self.client.call_tool("read_world", {}, got_yaml)

    def _reload_yaml(self, text: str, ok_msg: str = ""):
        """把修改后的 YAML 全文投喂 load_world，成功后本地重编译 + 全刷新。"""
        self._busy = True

        def done(err, data):
            self._busy = False
            if err:
                self._apply_state(f"load_world 失败: {data}")
                self._log(f"load_world 失败: {data}", "error")
                return
            try:
                spec = parse_world(yaml.safe_load(text))
                spec.base_dir = self._base_dir
            except Exception as e:
                self._apply_state(f"世界解析失败: {e}")
                self._log(f"世界解析失败: {e}", "error")
                return
            self._on_loaded(None, data, spec)
            if ok_msg:
                self._apply_state(ok_msg)
                self._log(ok_msg, "ok")

        self.client.call_tool(
            "load_world", {"yaml_text": text, "base_dir": str(self._base_dir)},
            done)

    def _on_zoo_created(self, err, data):
        if err:
            self._apply_state(f"zoo_create 失败: {data}")
            self._log(f"zoo_create 失败: {data}", "error")
            return
        world = data["world"]
        is_demo = bool(data.get("demo"))     # demo 项目：加载后自动 Run
        try:
            spec = parse_world_file(world)
        except Exception as e:
            self._apply_state(f"生成的世界错误: {e}")
            self._log(f"生成的世界错误: {e}", "error")
            return

        def loaded(e, d):
            self._on_loaded(e, d, spec)
            if is_demo and not e and self.state.can("run"):
                self._do_run(self._run_args())       # 直接看动画

        self.client.call_tool("load_world", {"yaml_path": world}, loaded)
        if is_demo:
            self._apply_state(f"已生成 {data['title']} 运动演示，加载后自动运行…")
            self._log(f"已生成 {data['title']} 演示（demo.py={data['demo']}），"
                      f"加载后自动运行", "ok")
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(data["controller"]))
            self._apply_state(f"已生成并加载 {data['title']}，控制器已打开。")
            self._log(f"已生成 {data['title']}，控制器：{data['controller']}", "ok")

    # ============================================================ 仿真流程

    def _on_loaded(self, err, data, spec):
        if err:
            self._apply_state(f"load_world 失败: {data}")
            self._log(f"load_world 失败: {data}", "error")
            return
        self._spec = spec                        # 编辑基准（拖动提交 YAML 用）
        self._base_dir = spec.base_dir
        self.view.set_world(spec)
        self.view.set_hud_state("就绪")
        self.scene_tree.set_world(spec)
        self.code.set_world(spec, spec.base_dir)
        # Agent worker：注入当前世界名 + 世界目录（本地文件工具根）+
        # MCP 工具清单（内核侧 _TOOLS 的快照——本进程在线 manifest）
        try:
            from agentsim.mcp_server import _TOOLS
            self.agent_worker.configure(self._agent_cfg, _TOOLS)
        except ImportError:
            pass
        self.agent_worker.set_world_dir(spec.base_dir)
        self.agent_worker.set_history(spec.name)
        self.obs.clear()
        self.props.reset_task()
        self.lbl_world.setText(f"世界：{spec.name}")
        self.lbl_tick.setText("tick 0")
        self.state.to(UiState.IDLE)
        self._apply_state(f"已加载 {spec.name}。F5 运行到终态，F7 步进。")
        self._log(f"世界已加载：{spec.name}"
                  f"（机器人 {len(spec.robots)}，物体 {len(spec.scene.objects)}）",
                  "ok")
        self._fetch_obs()
        # 拖动/位姿提交 -> 重载后重选该节点（属性面板数值同步）
        pend = getattr(self, "_pending_select", None)
        if pend:
            self._pending_select = None
            if not self.scene_tree.select_node(*pend):
                self.props.clear_selection()
        # 代码页"保存并运行" -> 重载后自动 Run（控制器进程已吃新代码）
        if getattr(self, "_pending_run", False):
            self._pending_run = False
            if self.state.can("run"):
                self._do_run(self._run_args())

    def _on_reset(self):
        if not self.state.can("reset"):
            return
        self._log("重置世界…")
        self.client.call_tool("reset", {}, self._on_reset_done)

    def _on_reset_done(self, err, data):
        if err:
            self._apply_state(f"reset 失败: {data}")
            self._log(f"reset 失败: {data}", "error")
            return
        self.state.to(UiState.IDLE)
        self.props.reset_task()
        self._apply_state("已重置。")
        self._log("世界已回到初态。")
        self._fetch_obs()

    def _run_args(self, extra: dict | None = None) -> dict:
        args = {"realtime": self.chk_realtime.isChecked()}
        if extra:
            args.update(extra)
        return args

    def _on_run(self):
        self._start_run(self._run_args())

    def _on_run_n(self):
        self._start_run(self._run_args({"ticks": self.spin_n.value()}))

    def _start_run(self, args: dict):
        if self.state.value == UiState.FINISHED:
            # episode 一次性：先 reset 再跑
            self.client.call_tool(
                "reset", {},
                lambda e, d: None if e else self._do_run(args))
            self._apply_state("重置后重跑…")
            return
        if not self.state.can("run"):
            return
        self._do_run(args)

    def _do_run(self, args: dict):
        n = args.get("ticks")
        self.state.to(UiState.RUNNING)
        self.view.set_hud_state("运行中")
        msg = (f"步进 {n} tick（控制器已接管）…" if n
               else "运行中（控制器已接管）…")
        self._apply_state(msg)
        self._log(msg)
        self.client.call_tool("run", args, self._on_run_done)

    def _on_run_done(self, err, stats):
        self.state.to(UiState.FINISHED)
        self.view.set_hud_state("已结束")
        if err:
            self._apply_state(f"run 失败: {stats}")
            self._log(f"run 失败: {stats}", "error")
        else:
            self.props.update_stats(stats)
            self.lbl_tick.setText(f"tick {stats.get('ticks')}")
            succ = stats.get("success")
            crashed = stats.get("crashed") or []
            self._apply_state(
                f"运行结束：ticks={stats.get('ticks')} "
                f"成功={'是' if succ else '否'}  reward={stats.get('reward')}")
            self._log(
                f"结束 ticks={stats.get('ticks')} "
                f"成功={'是' if succ else '否'} reward={stats.get('reward')} "
                f"墙钟 {stats.get('wall_time')}s",
                "ok" if succ else "info")
            if crashed:
                self._log(f"控制器异常退出：{crashed}", "warn")
        self._fetch_obs()

    # ============================================================ 视图/选择

    def _select_cam(self, name: str) -> None:
        idx = self.cam_combo.findText(name)
        if idx >= 0:
            self.cam_combo.setCurrentIndex(idx)

    def _on_node_selected(self, kind: str, name: str, payload) -> None:
        self.props.show_node(kind, name, payload)
        self.view.set_selection(kind, name)   # 场景树 -> 3D 高亮（单向）
        if self.tabs.currentIndex() != 0:
            self.tabs.setCurrentIndex(0)

    # ============================================================ 观测轮询

    def _poll(self):
        if self.state.value == UiState.RUNNING:
            self._fetch_obs()

    def _fetch_obs(self):
        if self._obs_pending:
            return
        self._obs_pending = True

        def done(err, obs):
            self._obs_pending = False
            if err or not isinstance(obs, dict):
                return
            # run 已结束后才到达的旧在途快照（running=True）丢弃：
            # 终态以结束回调里强制拉取的最新观测为准，避免面板被旧 tick 覆盖
            if self.state.value != UiState.RUNNING and obs.get("running"):
                return
            self.view.apply_observation(obs)
            self.view.refresh()
            self.props.update_obs(obs)
            self.obs.update_obs(obs)
            tick = obs.get("tick")
            if tick is not None:
                self.lbl_tick.setText(f"tick {tick}")

        self.client.call_tool("get_observation", {}, done)

    # ============================================================ 兜底

    def _on_backend_log(self, line: str):
        self.console.backend(line)

    def _on_backend_died(self, msg):
        self.state.to(UiState.EMPTY)
        self.view.set_hud_state("后端异常")
        self._apply_state(f"仿真后端异常退出：{msg}")
        self._log(f"仿真后端异常退出：{msg}", "error")

    def closeEvent(self, e):
        self.timer.stop()
        try:
            self.agent_worker.stop()
            self.agent_worker.wait(2000)        # 让 QThread 退出
        except Exception:
            pass
        self.client.shutdown()
        super().closeEvent(e)
