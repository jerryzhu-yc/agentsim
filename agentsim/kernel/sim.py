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
from agentsim.kernel.task import (robot_obs, sensors_obs, task_reward,
                                  task_success)
from agentsim.world.builder import build_world, resolve_task
from agentsim.world.schema import WorldSpec

# 控制器握手超时（秒）。spawn 后进程活着但迟迟不发 hello——典型如
# Windows 上控制器 import/模型加载卡死——超过此时长放弃等待并报错，
# 否则 run() 永久挂起（GUI 表现为"无仿真、无法停止、没有 tick"）。
# 测试可 monkeypatch 本常量缩短等待。
_HANDSHAKE_TIMEOUT = 30.0


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
            # stdout 必须吞掉：内核若在 MCP 后端里跑，stdout 是 JSON-RPC
            # 协议通道，控制器 print 会污染协议流。stderr 直通继承——
            # 控制器起不来（import 报错等）的真相必须可见，否则
            # Windows 上"无报错但卡死"无从诊断（GUI 控制台/后端日志链路）。
            self._procs[rd.name] = subprocess.Popen(
                [sys.executable, str(path),
                 "--sock", self._addr, "--robot", rd.name],
                env=env, stdout=subprocess.DEVNULL, stderr=None)

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
        sensors_manifest = [{"name": six.name, "type": six.kind, "dim": six.dim}
                            for six in ridx.sensors.values()]
        self._ident[ident] = rname
        self._send(ident, {
            "op": "init", "robot": rname,
            "actuators": list(ridx.actuators), "ranges": ranges,
            "args": rd.args, "ctrl_period": self.spec.ctrl_period,
            "timestep": self.spec.timestep,
            "sensors": sensors_manifest,
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
        qpos, qvel = robot_obs(ridx, self.model, self.data)
        bodies = {name: [float(x) for x in self.data.xpos[bid]]
                  for name, bid in ridx.bodies.items()}
        # 场景物体全局可见（控制闭环需要：抓取要看着 cube 的位置）
        objects = {name: [float(x) for x in self.data.xpos[bid]]
                   for name, bid in self.index.objects.items()}
        msg = {
            "op": "obs", "tick": self.tick, "time": float(self.data.time),
            "ctrl": list(map(float, self.data.ctrl[
                list(ridx.actuators.values())])),
            "joint_names": list(ridx.joints), "qpos": qpos, "qvel": qvel,
            "bodies": bodies, "objects": objects,
        }
        # 传感器读数（声明过的机器人才有这个键；sensordata 在 mj_step 末尾已就绪；
        # 拼接语义见 kernel/task.py，与 RL env 共用）
        if ridx.sensors:
            msg["sensors"] = sensors_obs(ridx, self.data)
        return msg

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

    def run(self, max_ticks: int | None = None,
            on_tick: "callable | None" = None,
            realtime: bool | None = None,
            stop_check: "callable | None" = None) -> dict:
        """推进仿真，返回统计。max_ticks 缺省用世界 task.max_ticks（再缺省 10000）。

        on_tick(tick)：每个控制 tick 推进后的回调（GUI/MCP 用来抽观测快照）。
        回调在同一线程内同步调用，此时 data 处于一致状态，可安全读取。

        realtime：True/False 覆盖世界 runtime.realtime；None 用世界设置。
        开启后按墙钟对齐仿真时间（GUI 看动画用；机器跑批应关闭以全速推进）。

        stop_check()：每个控制 tick 检查一次，返回 True 时提前收尾
        （GUI"停止"按钮 / MCP stop 工具）。收尾语义与跑满完全一致：
        控制器收 stop 正常退出（exit 0 不算崩溃），已跑轨迹不变。
        """
        if max_ticks is None:
            max_ticks = self.spec.task.max_ticks
        if max_ticks is None:
            max_ticks = 10_000
        timeout_ms = int(self.spec.runtime.controller_timeout)   # schema 单位就是 ms
        policy = self.spec.runtime.timeout_policy
        if realtime is None:
            realtime = self.spec.runtime.realtime
        t_start, sim_start = time.perf_counter(), self.data.time

        def _pace():
            """实时节流：把墙钟对齐到当前仿真时间。"""
            if not realtime:
                return
            target = sim_start + (self.tick * self.spec.ctrl_period
                                  * self.spec.timestep)
            behind = target - (time.perf_counter() - t_start)
            if behind > 0:
                time.sleep(behind)

        # 无控制器的世界：直接空跑（scene_gen 冒烟等用途）
        if not self._procs:
            for _ in range(max_ticks):
                if stop_check is not None and stop_check():
                    break
                for _ in range(self.spec.ctrl_period):
                    mujoco.mj_step(self.model, self.data)
                self.tick += 1
                if on_tick is not None:
                    on_tick()
                _pace()
            return self._stats(t_start, sim_start)

        # 等所有控制器完成握手（进程死光则放弃等待）。
        # 必须有 stop 出口和超时：进程活着但握手不完成时（Windows 上
        # 控制器 import/加载卡死的典型症状），原先这里会永久转圈——
        # run() 不返回、stop_check 无人查，GUI 就是"卡住且无法停止"。
        self._check_crashes()   # 先扫一遍：spawn 后即死的控制器要在 stats 里可见
        hs_deadline = time.perf_counter() + _HANDSHAKE_TIMEOUT
        while True:
            parts = self._barrier_participants()
            if not parts or all(self._st[n]["active"] for n in parts):
                break
            if stop_check is not None and stop_check():
                # 停止请求落在握手等待期：此时控制器还没进 step 循环，
                # 无法用 stop 消息优雅收尾，直接终止进程（标记 done，
                # 不计 crashed——这是用户主动停止，不是故障）
                self._abort_controllers()
                return self._stats(t_start, sim_start)
            if time.perf_counter() > hs_deadline:
                diag = "、".join(
                    n for n in parts if not self._st[n]["active"])
                self._abort_controllers()
                raise WorldError(
                    f"控制器握手超时（{_HANDSHAKE_TIMEOUT:.0f}s）: {diag}。"
                    "进程存活但迟迟未发 hello，常见原因：控制器脚本 import "
                    "或模型加载卡死/极慢（Windows 冷启动尤甚）。控制器 "
                    "stderr 已直通内核错误输出，真实报错请到 GUI 控制台/"
                    "后端日志查看。")
            self._poll_once(timeout_ms)
            self._check_crashes()

        while self.tick < max_ticks and self._barrier_participants():
            if stop_check is not None and stop_check():
                break
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
                if on_tick is not None:
                    on_tick()
            elif policy == "last_ctrl" and not got_msg:
                # 只在真超时（队列空）时补位，队列里还有消息就继续收
                for name in parts:
                    if self._st[name]["pending"] == 0:
                        self._st[name]["pending"] = 1
                        self._st[name]["lag"] += 1
            # pause: 什么都不做，下一轮继续等
            _pace()

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

    def _abort_controllers(self, grace_ms: int = 500) -> None:
        """异常中止（握手超时/握手期停止）时的兜底收尾。

        正常收尾靠 stop 消息，但握手等待期的控制器要么还在 Python
        启动中、要么阻塞在等 hello 回复，没有"正在等 step 回复"的
        机会，stop 消息对它们无效——不终止就永久泄漏（进程活着但
        无人应答）。先 drain 一小段给已进 step 循环的控制器一个
        优雅退出的机会，剩下的直接 terminate。
        """
        self._notify_stop()
        self._drain_controllers(grace_ms)
        for name, proc in self._procs.items():
            if proc.poll() is None:
                # 标 done 防 _check_crashes 误报 crashed（用户主动
                # 停止/超时中止不属于进程故障）
                self._st[name]["done"] = True
                self._st[name]["active"] = False
                proc.terminate()

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
        """终态任务判定（语义实现见 kernel/task.py，与 RL env 共用）。"""
        return task_success(self.task, self.data)

    def _task_reward(self) -> float:
        """终态奖励值（语义实现见 kernel/task.py，与 RL env 共用）。"""
        return task_reward(self.task, self.model, self.data)

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
