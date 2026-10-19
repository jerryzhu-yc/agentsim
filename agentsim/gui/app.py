"""M1 主窗口：McpClient + RenderView + 面板组装。

信号流（GUI 与 AI Agent 同构，都是 MCP 客户端）：
  Load   -> 本地解析 spec（渲染模型用）+ load_world RPC -> IDLE
  Run    -> run RPC（延迟响应）-> RUNNING（30Hz 拉快照刷新视口/面板）
         -> run 响应到达 -> FINISHED
  Reset  -> reset RPC -> IDLE
  FINISHED 下点 Run -> 自动 reset 再 run（episode 是一次性的）
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel,
                               QMainWindow, QPushButton, QSpinBox, QSplitter,
                               QVBoxLayout, QWidget)

from agentsim.gui.mcp_client import McpClient
from agentsim.gui.panels import make_side_panel
from agentsim.gui.render_view import RenderView
from agentsim.gui.state import UiState
from agentsim.world.schema import load_world as parse_world_file


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AgentSim")
        self.resize(1280, 800)
        self.state = UiState()

        self.client = McpClient(self)
        self.client.start()
        self.client.backendDied.connect(self._on_backend_died)

        # ---- 左：3D 视口 / 右：面板 + 控制
        self.view = RenderView()
        side, self.panels = make_side_panel()
        self.view.setMinimumWidth(500)
        side.setMaximumWidth(420)

        self.btn_load = QPushButton("加载世界")
        self.btn_reset = QPushButton("Reset")
        self.btn_run = QPushButton("Run")
        self.btn_run_n = QPushButton("Run N")
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 100000)
        self.spin_n.setValue(100)
        self.chk_realtime = QCheckBox("实时")
        self.chk_realtime.setChecked(True)   # GUI 默认看动画；全速适合跑批验证
        self.chk_realtime.setToolTip("勾选：按墙钟实时播放（动画连贯）\n"
                                     "取消：全速推进（瞬间出结果）")
        self.status_label = QLabel("启动中…")
        ctrl = QWidget()
        cl = QHBoxLayout(ctrl)
        for w in (self.btn_load, self.btn_reset, self.btn_run,
                  self.btn_run_n, self.spin_n, self.chk_realtime):
            cl.addWidget(w)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(ctrl)
        rl.addWidget(self.status_label)
        rl.addWidget(side)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.view)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        self.setCentralWidget(split)

        # ---- 事件
        self.btn_load.clicked.connect(self._on_load)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_run_n.clicked.connect(self._on_run_n)
        self.client.call_tool("ping", {}, lambda e, d: None)   # 握手即激活

        # ---- 30Hz 观测轮询
        self._obs_pending = False
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._poll)
        self.timer.start()

        self._apply_state("就绪。先加载一个世界（worlds/ 下有示例）。")

    # ------------------------------------------------------------ 状态与按钮

    def _apply_state(self, msg: str = "") -> None:
        s = self.state.value
        self.btn_load.setEnabled(s != UiState.RUNNING)
        self.btn_reset.setEnabled(self.state.can("reset"))
        self.btn_run.setEnabled(self.state.can("run"))
        self.btn_run_n.setEnabled(self.state.can("run"))
        if msg:
            self.status_label.setText(msg)
        self.statusBar().showMessage(f"[{s}] {msg}" if msg else s)

    # ------------------------------------------------------------ 按钮动作

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择世界文件", "worlds", "World YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            spec = parse_world_file(path)          # 本地解析（渲染模型用）
        except Exception as e:
            self._apply_state(f"世界文件错误: {e}")
            return
        self.client.call_tool("load_world", {"yaml_path": path},
                              lambda e, d: self._on_loaded(e, d, spec))

    def _on_loaded(self, err, data, spec):
        if err:
            self._apply_state(f"load_world 失败: {data}")
            return
        self.view.set_world(spec)
        self.panels["world"].update_world(spec)
        self.state.to(UiState.IDLE)
        self._apply_state(f"已加载 {spec.name}。Run 全速跑或 Run N 分段看。")
        self._fetch_obs()

    def _on_reset(self):
        if not self.state.can("reset"):
            return
        self.client.call_tool("reset", {}, self._on_reset_done)

    def _on_reset_done(self, err, data):
        if err:
            self._apply_state(f"reset 失败: {data}")
            return
        self.state.to(UiState.IDLE)
        self._apply_state("已 reset。")
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
            # episode 一次性：先 reset 再跑（便利设计）
            self.client.call_tool(
                "reset", {},
                lambda e, d: None if e else self._do_run(args))
            self._apply_state("reset 后重跑…")
            return
        if not self.state.can("run"):
            return
        self._do_run(args)

    def _do_run(self, args: dict):
        self.state.to(UiState.RUNNING)
        self._apply_state("run 进行中（控制器已接管）…")
        self.client.call_tool("run", args, self._on_run_done)

    def _on_run_done(self, err, stats):
        self.state.to(UiState.FINISHED)
        if err:
            self._apply_state(f"run 失败: {stats}")
        else:
            self.panels["task"].update_stats(stats)
            succ = stats.get("success")
            objs = stats.get("objects", {})
            self._apply_state(
                f"run 结束：ticks={stats.get('ticks')}"
                f" 成功={'是' if succ else '否'}"
                + (f" 终位={ {k: [round(x, 2) for x in v] for k, v in objs.items()} }"
                   if objs else ""))
        self._fetch_obs()

    # ------------------------------------------------------------ 观测轮询

    def _poll(self):
        if self.state.value == UiState.RUNNING:
            self._fetch_obs()

    def _fetch_obs(self):
        if self._obs_pending:
            return
        self._obs_pending = True

        def done(err, obs):
            self._obs_pending = False
            if err:
                return
            if isinstance(obs, dict):
                self.view.apply_observation(obs)
                self.view.refresh()
                self.panels["task"].update_obs(obs)
                self.panels["obs"].update_obs(obs)

        self.client.call_tool("get_observation", {}, done)

    # ------------------------------------------------------------ 兜底

    def _on_backend_died(self, msg):
        self.state.to(UiState.EMPTY)
        self._apply_state(f"仿真后端异常退出：{msg}")

    def closeEvent(self, e):
        self.timer.stop()
        self.client.shutdown()
        super().closeEvent(e)
