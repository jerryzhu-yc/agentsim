"""演示控制器：随机漫步机械臂（pick_cube.yaml 里引用的就是它）。

每个 tick 给每个执行器一个限速随机增量，用于压测内核：
进程间通信、栅栏同步、超时策略全走真实路径。
"""
import random

from agentsim.controller import Robot

robot = Robot()
speed = float(robot.args.get("speed", 0.2))
cur = {n: sum(robot.range(n)) / 2 for n in robot.actuators}

while robot.step():
    for name in robot.actuators:
        lo, hi = robot.range(name)
        cur[name] = min(hi, max(lo, cur[name] + random.uniform(-speed, speed)))
        robot.set(name, cur[name])

robot.close()
