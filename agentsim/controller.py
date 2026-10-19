"""控制器侧客户端：用户在控制器脚本里唯一要 import 的东西。

用法（Webots 风格）：

    from agentsim.controller import Robot

    robot = Robot()                 # 自动从 --sock/--robot 命令行参数连接内核
    while robot.step():             # 推进一 tick，返回观测 dict；内核叫停时返回 False
        for name in robot.actuators:
            lo, hi = robot.range(name)
            robot.set(name, 目标值)  # 写控制量，下个 tick 生效
        obs = robot.obs             # {"tick","time","qpos","qvel","bodies",...}

约定：
- 控制器是被内核 spawn 的独立进程，通过 ZMQ REQ 与内核栅栏同步
- 一个 Robot 实例对应世界里一个机器人；step() 之外内核不会推消息
"""
from __future__ import annotations

import argparse
import json

import zmq


class Robot:
    """控制器脚本里的机器人句柄。"""

    def __init__(self):
        ns = self._parse_args()
        self.name = ns.robot
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        # 不设 RCVTIMEO：等待无限。真实 Agent 在 load_world 与 run 之间有
        # 长"思考时间"，控制器必须活着等内核开跑（Webots 语义：等世界）。
        # 内核 close() 会 terminate 控制器进程；内核崩溃导致控制器孤儿挂死
        # 属 M0 已知取舍（TODO: M1 心跳）。
        self._sock.connect(ns.sock)

        init = self._rpc({"op": "hello", "robot": self.name})
        if init.get("op") == "error":
            raise RuntimeError(f"内核拒绝连接: {init['msg']}")
        self.actuators: list[str] = init["actuators"]
        self.ranges: list[list[float]] = init["ranges"]
        self.args: dict = init["args"]
        self.ctrl_period: int = init["ctrl_period"]
        self.timestep: float = init["timestep"]
        # 传感器设备清单 [{name,type,dim}]（hello 握手下发；无传感器为 []）
        self.sensors: list[dict] = init.get("sensors", [])
        self._ctrl = [0.0] * len(self.actuators)

        self.obs: dict = {}
        self.tick = -1
        self.time = 0.0

    # -------------------------------------------------- 用户 API

    def set(self, name: str, value: float) -> None:
        """写单个执行器目标值（名字引用，不碰裸 id）"""
        if name not in self.actuators:
            raise KeyError(f"机器人 '{self.name}' 没有执行器 '{name}'，"
                           f"可用: {self.actuators}")
        self._ctrl[self.actuators.index(name)] = float(value)

    def set_all(self, values: list[float]) -> None:
        """按 actuators 顺序一次写全部执行器"""
        if len(values) != len(self.actuators):
            raise ValueError(f"需要 {len(self.actuators)} 个值，实际 {len(values)}")
        self._ctrl = [float(v) for v in values]

    def range(self, name: str) -> tuple[float, float]:
        """执行器控制量范围 (lo, hi)"""
        return tuple(self.ranges[self.actuators.index(name)])

    def step(self, n: int = 1) -> dict | False:
        """提交当前控制量，推进 n 个控制 tick。

        返回观测 dict；内核到达 max_ticks 或关闭时返回 False（调用方应退出循环）。
        """
        reply = self._rpc({"op": "step", "ctrl": self._ctrl, "n": int(n)})
        if reply.get("op") == "stop":
            return False
        if reply.get("op") == "error":
            raise RuntimeError(f"内核报错: {reply['msg']}")
        self.obs = reply
        self.tick = reply["tick"]
        self.time = reply["time"]
        self._ctrl = [float(c) for c in reply["ctrl"]]
        return reply

    def joint_state(self) -> dict:
        """便捷读取：{关节名: 位置}（从最近一次观测取）"""
        return dict(zip(self.joint_names, self.obs.get("qpos", [])))

    def sensor(self, name: str) -> float | list[float]:
        """读传感器最近读数（dim=1 标量；imu 等多维返回列表），从最近一次观测取。

        相机没有读数（只是 viewpoint 声明，截图走 MCP screenshot 工具）。
        """
        vals = (self.obs.get("sensors") or {}).get(name)
        if vals is None:
            declared = [s["name"] for s in self.sensors]
            if name in declared:
                raise RuntimeError(f"传感器 '{name}' 的读数还没到——先 step() 一次")
            raise KeyError(f"机器人 '{self.name}' 没有传感器 '{name}'，"
                           f"可用: {declared}")
        return vals

    def supervise(self, action: str, **params) -> dict:
        """Supervisor 操作：save/load/teleport/reset。

        例: robot.supervise("save", tag="snap")
            robot.supervise("load", tag="snap")
            robot.supervise("teleport", body="cube", pos=[1.0, 1.0, 0.2])
            robot.supervise("reset")
        回包带内核 tick（load/reset 会改变 tick，这里自动同步）。
        """
        reply = self._rpc({"op": "supervise", "action": action, **params})
        if reply.get("op") == "error":
            raise RuntimeError(f"supervisor 错误: {reply['msg']}")
        if "tick" in reply:
            self.tick = reply["tick"]
            self.time = reply.get("time", self.time)
        return reply

    # -------------------------------------------------- 内部

    @property
    def joint_names(self) -> list[str]:
        return self.obs.get("joint_names", [])

    @staticmethod
    def _parse_args():
        # 内核 spawn 时传 --sock/--robot；用户脚本自己的参数用 parse_known 跳过
        ap = argparse.ArgumentParser(add_help=False)
        ap.add_argument("--sock", required=True)
        ap.add_argument("--robot", required=True)
        return ap.parse_known_args()[0]

    def _rpc(self, msg: dict) -> dict:
        self._sock.send(json.dumps(msg).encode())
        return json.loads(self._sock.recv())

    def close(self) -> None:
        try:
            self._sock.send(json.dumps({"op": "bye"}).encode())
            try:
                # 告别收尾：内核会回 bye_ack；短超时兜底（内核可能已停止 poll）
                self._sock.setsockopt(zmq.RCVTIMEO, 200)
                self._sock.recv()
            except zmq.error.Again:
                pass
        except Exception:
            pass
        self._sock.close()
        self._ctx.term()
