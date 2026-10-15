"""仿真内核：ZMQ ROUTER + 控制器子进程 + 栅栏同步推进。

架构（方案 5.2 节）：
- 每个机器人的控制器是独立 OS 进程，REQ 连内核 ROUTER
- 控制器发 step 请求 -> 内核等所有活跃控制器的请求到齐（栅栏）
  -> 写控制量 -> 推进 n*ctrl_period 个物理步 -> 各自回观测
- 超时策略：last_ctrl=用上次控制量补位继续推进；pause=原地等待
- 控制器崩溃（进程退出）：标记 inactive，内核按策略继续，永不死锁

REQ/REP 严格锁步保证：每个 step 请求有且只有一个回复（obs 或 stop）。
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import mujoco
import numpy as np
import zmq

from agentsim import supervisor
from agentsim.errors import WorldError
from agentsim.world.builder import build_world, resolve_task
from agentsim.world.schema import WorldSpec

# 关节类型 -> qpos 占用宽度
_QPOS_WIDTH = {int(mujoco.mjtJoint.mjJNT_FREE): 7,
               int(mujoco.mjtJoint.mjJNT_BALL): 4}


class Simulation:
    """单实例一次仿真。__init__ 起进程，run 驱动，close 收尾。"""

    def __init__(self, spec: WorldSpec):
        self.spec = spec
        self.model, self.index = build_world(spec)
        self.task = resolve_task(spec.task, self.index)
        self.data = mujoco.MjData(self.model)

        self._ctx = zmq.Context()
        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.setsockopt(zmq.LINGER, 0)
        port = self._router.bind_to_random_port("tcp://127.0.0.1")
        self._addr = f"tcp://127.0.0.1:{port}"
        self._poller = zmq.Poller()
        self._poller.register(self._router, zmq.POLLIN)

        self.tick = 0
        # robot name -> 运行时状态
        self._st: dict[str, dict] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._ident: dict[bytes, str] = {}      # zmq identity -> robot name
        # Supervisor 快照（__init__ 在 reset 用）
        self._snaps: dict[str, tuple] = {}
        supervisor.save_state(self.model, self.data, self._snaps,
                              supervisor.INIT_TAG, tick=0, internal=True)
        self._spawn_controllers()

    # ---------------------------------------------------------------- 子进程

    def _spawn_controllers(self):
        env = dict(os.environ)
        # 控制器子进程要能 import agentsim
        pkg_root = str(pathlib.Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
        for rd in self.spec.robots:
            nu = len(self.index.robots[rd.name].actuators)
            self._st[rd.name] = {
                "active": False,       # 已完成 hello 握手
                "done": False,         # 正常退出
                "crashed": False,      # 进程异常退出
                "ctrl": [0.0] * nu,    # 最近一次真实控制量
                "pending": 0,          # 已收到的待执行 step 步数
                "reply_sock": None,    # 等待回复的 zmq identity
                "lag": 0,              # 超时补位次数
            }
            if not rd.controller:
                continue
            path = pathlib.Path(rd.controller)
            if not path.is_absolute():
                path = self.spec.base_dir / path
            if not path.exists():
                raise WorldError(f"机器人 '{rd.name}' 的控制器不存在: {path}")
            self._procs[rd.name] = subprocess.Popen(
                [sys.executable, str(path),
                 "--sock", self._addr, "--robot", rd.name],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ---------------------------------------------------------------- 收发

    def _send(self, ident: bytes, msg: dict):
        # REQ/ROUTER 协议：[identity, 空分隔帧, payload]
        self._router.send_multipart([ident, b"", json.dumps(msg).encode()])

    def _handle(self, ident: bytes, msg: dict) -> None:
        op = msg.get("op")
        if op == "hello":
            self._hello(ident, msg)
        elif op == "step":
            self._step_request(ident, msg)
        elif op == "bye":
            rname = self._ident.get(ident, msg.get("robot"))
            if rname in self._st:
                self._st[rname]["done"] = True
                self._st[rname]["active"] = False
            # REQ/REP 必须回复，否则控制器 close() 里的 recv 会挂死
            self._send(ident, {"op": "bye_ack"})
        elif op == "supervise":
            self._supervise(ident, msg)
        else:
            self._send(ident, {"op": "error", "msg": f"未知 op: {op!r}"})

    # ---------------------------------------------------------------- supervisor

    def save_state(self, tag: str) -> None:
        supervisor.save_state(self.model, self.data, self._snaps, tag, self.tick)

    def load_state(self, tag: str) -> None:
        self.tick = supervisor.load_state(self.model, self.data,
                                          self._snaps, tag)

    def get_state(self, tag: str) -> tuple:
        snap = self._snaps.get(tag)
        if snap is None:
            raise WorldError(
                f"快照 '{tag}' 不存在；可用: {supervisor.list_tags(self._snaps)}")
        return snap

    def teleport(self, name: str, pos) -> None:
        supervisor.teleport(self.model, self.data, self.index, name, pos)

    def reset(self) -> None:
        """回到编译完成、还没跑过任何 tick 的初始状态。"""
        self.load_state(supervisor.INIT_TAG)

    def _supervise(self, ident: bytes, msg: dict) -> None:
        action = msg.get("action")
        try:
            if action == "save":
                self.save_state(msg.get("tag"))
            elif action == "load":
                self.load_state(msg.get("tag"))
            elif action == "teleport":
                self.teleport(msg.get("body"), msg.get("pos"))
            elif action == "reset":
                self.reset()
            else:
                raise WorldError(
                    f"未知 supervise action: {action!r}，"
                    f"支持: save/load/teleport/reset")
        except WorldError as e:
            self._send(ident, {"op": "error", "msg": str(e)})
            return
        self._send(ident, {"op": "ok", "tick": self.tick,
                           "time": float(self.data.time)})

    def _hello(self, ident: bytes, msg: dict):
        rname = msg.get("robot", "")
        if rname not in self._st:
            self._send(ident, {"op": "error", "msg": f"世界里没有机器人 '{rname}'"})
            return
        rd = next(r for r in self.spec.robots if r.name == rname)
        ridx = self.index.robots[rname]
        ranges = [list(map(float, self.model.actuator_ctrlrange[ridx.actuators[a]]))
                  for a in ridx.actuators]
        self._ident[ident] = rname
        self._send(ident, {
            "op": "init", "robot": rname,
            "actuators": list(ridx.actuators), "ranges": ranges,
            "args": rd.args, "ctrl_period": self.spec.ctrl_period,
            "timestep": self.spec.timestep,
        })
        self._st[rname]["active"] = True

    def _step_request(self, ident: bytes, msg: dict):
        rname = self._ident.get(ident)
        if rname is None:
            self._send(ident, {"op": "error", "msg": "未握手就发 step"})
            return
        st = self._st[rname]
        ctrl = msg.get("ctrl", st["ctrl"])
        ridx = self.index.robots[rname]
        if len(ctrl) != len(ridx.actuators):
            self._send(ident, {"op": "error",
                               "msg": f"控制量长度 {len(ctrl)} != 执行器数 "
                                      f"{len(ridx.actuators)}"})
            st["crashed"] = True
            st["active"] = False
            return
        st["ctrl"] = [float(c) for c in ctrl]
        st["pending"] = int(msg.get("n", 1))
        st["reply_sock"] = ident

    # ---------------------------------------------------------------- 推进

    def _barrier_participants(self) -> list[str]:
        """栅栏参与者 = 进程还在运行且没退出（不管握没握手）的控制器。

        关键点：握手没完成也占栅栏位，内核绝不提前推进——
        否则后连上的控制器相位会错位（M0.3 实测踩过）。
        """
        return [n for n, p in self._procs.items()
                if p.poll() is None
                and not self._st[n]["done"] and not self._st[n]["crashed"]]

    def _barrier_ready(self) -> bool:
        parts = self._barrier_participants()
        if not parts:
            return False
        return all(self._st[n]["pending"] > 0 for n in parts)

    def _advance(self, budget: int | None = None) -> None:
        parts = self._barrier_participants()
        n = min(self._st[n]["pending"] for n in parts)
        if budget is not None:
            n = min(n, budget)   # 不越过 max_ticks
        if n <= 0:
            return
        # 写控制量（名字索引 -> 裸 id，内核唯一裸 id 出现的地方）
        for name in parts:
            st = self._st[name]
            ids = list(self.index.robots[name].actuators.values())
            self.data.ctrl[ids] = st["ctrl"]
            st["pending"] -= n
        for _ in range(n * self.spec.ctrl_period):
            mujoco.mj_step(self.model, self.data)
        self.tick += n
        # 回观测（REQ/REP：每个已归零 pending 的请求恰好一个回复）
        for name in parts:
            st = self._st[name]
            if st["pending"] == 0 and st["reply_sock"] is not None:
                self._send(st["reply_sock"], self._obs_msg(name))
                st["reply_sock"] = None

    def _obs_msg(self, robot_name: str) -> dict:
        ridx = self.index.robots[robot_name]
        qpos, qvel = [], []
        for jid in ridx.joints.values():
            w = _QPOS_WIDTH.get(int(self.model.jnt_type[jid]), 1)
            adr = self.model.jnt_qposadr[jid]
            qpos += list(map(float, self.data.qpos[adr:adr + w]))
            vadr = self.model.jnt_dofadr[jid]
            qvel += list(map(float, self.data.qvel[vadr:vadr + w]))
        bodies = {name: [float(x) for x in self.data.xpos[bid]]
                  for name, bid in ridx.bodies.items()}
        # 场景物体全局可见（控制闭环需要：抓取要看着 cube 的位置）
        objects = {name: [float(x) for x in self.data.xpos[bid]]
                   for name, bid in self.index.objects.items()}
        return {
            "op": "obs", "tick": self.tick, "time": float(self.data.time),
            "ctrl": list(map(float, self.data.ctrl[
                list(ridx.actuators.values())])),
            "joint_names": list(ridx.joints), "qpos": qpos, "qvel": qvel,
            "bodies": bodies, "objects": objects,
        }

    def _check_crashes(self) -> None:
        for name, proc in self._procs.items():
            if proc.poll() is not None and not self._st[name]["done"]:
                # 进程退出且不是 bye 正常退出：看退出码。
                # stop 通知后控制器正常收尾（exit 0、可能没发 bye——用户
                # 脚本未必调 robot.close()）不算崩溃；非零退出码才是。
                if proc.returncode != 0:
                    self._st[name]["crashed"] = True
                self._st[name]["active"] = False

    def _notify_stop(self) -> None:
        for st in self._st.values():
            if st["reply_sock"] is not None:
                self._send(st["reply_sock"], {"op": "stop"})
                st["reply_sock"] = None
                st["pending"] = 0

    # ---------------------------------------------------------------- 主循环

    def run(self, max_ticks: int | None = None) -> dict:
        """推进仿真，返回统计。max_ticks 缺省用世界 task.max_ticks（再缺省 10000）。"""
        if max_ticks is None:
            max_ticks = self.spec.task.max_ticks
        if max_ticks is None:
            max_ticks = 10_000
        timeout_ms = int(self.spec.runtime.controller_timeout)   # schema 单位就是 ms
        policy = self.spec.runtime.timeout_policy
        realtime = self.spec.runtime.realtime
        t_start, sim_start = time.perf_counter(), self.data.time

        # 无控制器的世界：直接空跑（scene_gen 冒烟等用途）
        if not self._procs:
            for _ in range(max_ticks):
                for _ in range(self.spec.ctrl_period):
                    mujoco.mj_step(self.model, self.data)
                self.tick += 1
            return self._stats(t_start, sim_start)

        # 等所有控制器完成握手（进程死光则放弃等待）
        self._check_crashes()   # 先扫一遍：spawn 后即死的控制器要在 stats 里可见
        while True:
            parts = self._barrier_participants()
            if not parts or all(self._st[n]["active"] for n in parts):
                break
            self._poll_once(timeout_ms)
            self._check_crashes()

        while self.tick < max_ticks and self._barrier_participants():
            got_msg = self._poll_once(timeout_ms)
            self._check_crashes()
            parts = self._barrier_participants()
            if not parts:
                break
            # 还有控制器没握手：不推进（防相位错位），继续等
            if not all(self._st[n]["active"] for n in parts):
                continue
            if self._barrier_ready():
                self._advance(budget=max_ticks - self.tick)
            elif policy == "last_ctrl" and not got_msg:
                # 只在真超时（队列空）时补位，队列里还有消息就继续收
                for name in parts:
                    if self._st[name]["pending"] == 0:
                        self._st[name]["pending"] = 1
                        self._st[name]["lag"] += 1
            # pause: 什么都不做，下一轮继续等
            if realtime:
                target = sim_start + (self.tick * self.spec.ctrl_period
                                      * self.spec.timestep)
                behind = target - (time.perf_counter() - t_start)
                if behind > 0:
                    time.sleep(behind)

        self._notify_stop()
        self._drain_controllers()
        return self._stats(t_start, sim_start)

    def _drain_controllers(self, grace_ms: int = 500) -> None:
        """run 收尾的残余消息排空：stop 只能回复"正在等回复"的控制器，
        若控制器恰好在两次 step 之间（obs 刚到手、下一 step 未发出），
        那个 step 会落在 run 返回之后——不排空它就永久阻塞（压测实测
        泄漏）。这里再 poll 一小段，把残余 step 逐个用 stop 回掉。
        """
        deadline = time.perf_counter() + grace_ms / 1000
        while time.perf_counter() < deadline:
            if not self._barrier_participants():
                return
            self._poll_once(50)
            self._check_crashes()
            self._notify_stop()

    def _poll_once(self, timeout_ms: int) -> bool:
        """收一条消息。返回 False 表示 poll 超时（队列空）。

        只有真超时才允许 last_ctrl 补位——否则队列里还有消息没读，
        会把准时的控制器误判成迟到（M0.3 实测踩过：lag=每 tick 一次）。
        """
        events = dict(self._poller.poll(timeout_ms))
        if self._router not in events:
            return False
        parts = self._router.recv_multipart()
        if len(parts) != 3:
            return True
        ident, _, payload = parts
        try:
            msg = json.loads(payload)
        except json.JSONDecodeError:
            self._send(ident, {"op": "error", "msg": "消息不是合法 JSON"})
            return True
        self._handle(ident, msg)
        return True

    def _task_success(self) -> bool | None:
        """终态任务判定：成功 body 的 xpos 落在 zone 内。无任务返回 None。"""
        if self.task.success_body is None:
            return None
        p = self.data.xpos[self.task.success_body]
        lo = np.asarray(self.task.success_min)
        hi = np.asarray(self.task.success_max)
        return bool(np.all(p >= lo) and np.all(p <= hi))

    def _task_reward(self) -> float:
        """终态奖励值（分项求和）。"""
        total = 0.0
        for kind, ia, ib, w in self.task.rewards:
            if kind == "distance":
                total += w * float(np.linalg.norm(
                    self.data.xpos[ia] - self.data.xpos[ib]))
            elif kind == "zone":
                p = self.data.xpos[ia]
                lo, hi = np.asarray(ib[0]), np.asarray(ib[1])
                total += w * float(np.all(p >= lo) and np.all(p <= hi))
            elif kind == "contact":
                # a/b 是 body id，contact 里是 geom id：经 geom_bodyid 对齐
                pair = {ia, ib}
                hit = any(
                    {self.model.geom_bodyid[self.data.contact[i].geom1],
                     self.model.geom_bodyid[self.data.contact[i].geom2]}
                    == pair
                    for i in range(self.data.ncon))
                total += w * float(hit)
        return total

    def _stats(self, t_start: float, sim_start: float) -> dict:
        return {
            "ticks": self.tick,
            "wall_time": round(time.perf_counter() - t_start, 3),
            "sim_time": round(float(self.data.time - sim_start), 3),
            "lag": sum(st["lag"] for st in self._st.values()),
            "done": [n for n, st in self._st.items() if st["done"]],
            "crashed": [n for n, st in self._st.items() if st["crashed"]],
            "nan": bool(not np.all(np.isfinite(self.data.qpos))),
            "success": self._task_success(),
            "reward": round(self._task_reward(), 4),
        }

    # ---------------------------------------------------------------- 收尾

    def close(self):
        for proc in self._procs.values():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self._router.close(0)
        self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
