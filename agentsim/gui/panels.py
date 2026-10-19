"""右侧面板控件：世界信息 / 任务状态 / 观测树。"""
from __future__ import annotations

from PySide6.QtWidgets import (QFormLayout, QGroupBox, QLabel, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)


class WorldPanel(QGroupBox):
    """世界信息（load_world 返回值 + 本地 spec 概要）。"""

    def __init__(self, parent=None):
        super().__init__("世界", parent)
        self._name = QLabel("（未加载）")
        self._objects = QLabel("-")
        self._robots = QLabel("-")
        self._max_ticks = QLabel("-")
        form = QFormLayout(self)
        form.addRow("名称", self._name)
        form.addRow("物体", self._objects)
        form.addRow("机器人", self._robots)
        form.addRow("max_ticks", self._max_ticks)

    def update_world(self, spec) -> None:
        self._name.setText(spec.name)
        self._objects.setText(", ".join(o.name for o in spec.scene.objects) or "-")
        self._robots.setText(", ".join(r.name for r in spec.robots) or "-")
        self._max_ticks.setText(str(spec.task.max_ticks))


class TaskPanel(QGroupBox):
    """任务状态：tick / 时间 / 成败 / 奖励（大字）。"""

    def __init__(self, parent=None):
        super().__init__("任务", parent)
        self._tick = QLabel("-")
        self._time = QLabel("-")
        self._success = QLabel("-")
        self._reward = QLabel("-")
        for lb in (self._tick, self._time, self._success, self._reward):
            lb.setStyleSheet("font-size:14pt; font-weight:600;")
        form = QFormLayout(self)
        form.addRow("tick", self._tick)
        form.addRow("仿真时间(s)", self._time)
        form.addRow("成功", self._success)
        form.addRow("reward", self._reward)

    def update_obs(self, obs: dict) -> None:
        task = obs.get("task") or {}
        self._tick.setText(str(obs.get("tick", "-")))
        self._time.setText(str(obs.get("time", "-")))
        succ = task.get("success")
        self._success.setText("-" if succ is None else ("是" if succ else "否"))
        self._reward.setText(str(task.get("reward", "-")))

    def update_stats(self, stats: dict) -> None:
        self._tick.setText(f"{stats.get('ticks', '-')} (结束)")
        succ = stats.get("success")
        self._success.setText("-" if succ is None else ("是" if succ else "否"))
        self._reward.setText(str(stats.get("reward", "-")))

    def clear(self) -> None:
        for lb in (self._tick, self._time, self._success, self._reward):
            lb.setText("-")


class ObsPanel(QGroupBox):
    """观测树：逐机器人关节 qpos/ctrl + 传感器读数 + 物体 pos/quat。"""

    def __init__(self, parent=None):
        super().__init__("观测", parent)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["名称", "值"])
        self._tree.setAlternatingRowColors(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self._tree)

    def update_obs(self, obs: dict) -> None:
        self._tree.clear()
        for rname, robs in (obs.get("robots") or {}).items():
            top = QTreeWidgetItem([rname])
            names = robs.get("joint_names") or []
            qpos = robs.get("qpos") or []
            ctrl = robs.get("ctrl") or []
            for i, jn in enumerate(names):
                v = f"q={qpos[i]:+.3f}" if i < len(qpos) else ""
                if i < len(ctrl):
                    v += f" u={ctrl[i]:+.3f}"
                QTreeWidgetItem(top, [jn, v])
            sensors = robs.get("sensors") or {}
            if sensors:                        # M2：有传感器才有该键
                st = QTreeWidgetItem(top, ["sensors"])
                for sn, sv in sensors.items():
                    if isinstance(sv, list):
                        sv = "(" + ", ".join(f"{x:+.3f}" for x in sv) + ")"
                    else:
                        sv = f"{sv:+.3f}"
                    QTreeWidgetItem(st, [sn, sv])
            self._tree.addTopLevelItem(top)
        for oname, oinfo in (obs.get("objects") or {}).items():
            pos = oinfo.get("pos") or []
            quat = oinfo.get("quat") or []
            top = QTreeWidgetItem([
                oname,
                f"pos=({', '.join(f'{x:+.2f}' for x in pos)}) "
                f"quat=({', '.join(f'{x:+.2f}' for x in quat)})"])
            self._tree.addTopLevelItem(top)
        self._tree.expandAll()

    def clear(self) -> None:
        self._tree.clear()


def make_side_panel() -> tuple[QWidget, "dict"]:
    """组装右侧面板栈，返回 (容器, 各面板引用)。"""
    world, task, obs = WorldPanel(), TaskPanel(), ObsPanel()
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.addWidget(world)
    lay.addWidget(task)
    lay.addWidget(obs, stretch=1)
    return box, {"world": world, "task": task, "obs": obs}
