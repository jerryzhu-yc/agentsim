"""M0.5 端到端任务控制器：j1 侧扫推式相位机（reach -> push -> settle）。

几何依据（MuJoCo 真模型 FK + 手算）：
- psi=j2+j3=90 度（j2 贴上限位 1.57，含 ~0.045 rad 重力稳态下沉）时前臂
  水平、指板开合沿世界竖直 —— 指板 +y 面是 10x1.6cm 竖直推面
- 指全开（fl=-0.03/fr=+0.03）时下指板中心 z=wrist_z-0.065 约 0.065，
  落在地面 cube（z 全高 0.10）中部：推点高于重心 1.5cm，滑动先于倾倒
  （滑移 0.147N < 倾覆 0.264N）
- cube 在 (0.635, 0.15)：半径 0.613~0.687 在指板展幅 [0.592, 0.692] 内、
  又在腕本体最大半径 0.598 之外 —— j1 扫描全程只有指板接触 cube
- j1 从 0 扫到 0.55：指板面在方位角 ~0.15 追上 cube 的 -y 面后贴板推行
  （板-指摩擦 1.0 粘住），cube 沿弧线被推到 (0.52, 0.38) 一带，落进
  task.success.zone

控制方式：目标斜坡（独立参考状态限速逼近），避免位置伺服一步到位甩臂。
push 相位刻意用 1/5 慢速（0.08 rad/s）：接触冲击力估算 ~0.13N，
低于倾覆阈值，cube 只滑不翻。
"""
from agentsim.controller import Robot

H = 1.5708
# (相位, j1, j2, j3, fl, fr, 关节斜坡速度 rad/s)
PHASES = [
    ("reach", 0.00, 1.57, 0.0618, -0.03, +0.03, 0.4),
    ("push",  0.55, 1.57, 0.0618, -0.03, +0.03, 0.08),
]
TOL = 0.06         # 前三关节到位容差（含 j2 约 0.045 rad 重力稳态误差）
STABLE = 20       # 连续 N tick 在容差内才算到位
TIMEOUT = 400     # 单相位 tick 上限（防死锁）
SETTLE_MAX = 600   # 推完后等待静止 tick 上限
SETTLE_WIN = 30    # 静止判定窗口
SETTLE_EPS = 1e-3  # 窗口内最大位移

STEP_F = 0.04 * 0.064   # 夹指斜坡每 tick 增量（0.04 m/s）
TICK = 0.064            # 控制周期（ctrl_period 32 x timestep 0.002）

robot = Robot()
phase_i = stable = phase_ticks = 0
settling = False
cube_hist: list[list[float]] = []


def ramp(cur: float, tgt: float, step: float) -> float:
    return cur + max(-step, min(step, tgt - cur))


# 斜坡参考状态：独立于 qpos 维护（若从 qpos 出发，惯性过冲会带着
# 目标一起爬升，位置伺服随即失控——实测 j2 冲过目标 0.19 rad）
ramp_state: list[float] | None = None

while robot.step():
    if settling:
        cube_hist.append(list(robot.obs["objects"]["cube"]))
        if len(cube_hist) > SETTLE_WIN:
            cube_hist.pop(0)
            first = cube_hist[0]
            moved = max(abs(p[k] - first[k])
                        for p in cube_hist for k in range(3))
            if moved < SETTLE_EPS or len(cube_hist) >= SETTLE_MAX:
                break
        continue

    name, j1, j2, j3, fl, fr, speed = PHASES[phase_i]
    q = robot.obs["qpos"]
    if ramp_state is None:
        ramp_state = list(q[:5])          # 首个 tick 从实际 qpos 起步
    step_j = speed * TICK
    for k, tgt in enumerate((j1, j2, j3, fl, fr)):
        step = step_j if k < 3 else STEP_F
        ramp_state[k] = ramp(ramp_state[k], tgt, step)
    robot.set_all(ramp_state)
    err = max(abs(q[k] - g) for k, g in enumerate((j1, j2, j3)))
    phase_ticks += 1
    stable = stable + 1 if err < TOL else 0

    if stable >= STABLE or phase_ticks >= TIMEOUT:
        if name == "push":
            settling = True               # 推完，转入等待 cube 静止
        else:
            phase_i += 1
            stable = phase_ticks = 0

robot.close()
