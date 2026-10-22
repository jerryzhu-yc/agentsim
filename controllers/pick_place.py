"""M7 端到端任务控制器：Franka Panda + OnRobot 2FG7 桌面抓取相位机。

任务（pick_cube.yaml）：桌面 60mm 方块放 A 区 (0.55, 0, 0.71)，panda 基座
站桌后缘 (0, 0, 0.68)——approach 高位 -> 5 段直线航点下探 -> grasp ->
close 闭合 -> 4 段航点抬升 -> lift -> hold，把方块夹起抬到
(0.55, 0, 0.86) 空中成功区并保持。

工具姿态（关键，v5 标定）：夹爪 mount 是 site-attach 的 Rot_x(-90°)
（手指 = gripper body -y -> attachment site +z 法兰法向，实测法兰贴合
-0.0002）。gripper_tcp site 与 body 系对齐（恒等 quat），故正确的
"手指竖直向下"姿态是 tcp 系 x->世界 x（开口）、y->世界 +z、z->世界 -y。
早期标定误把 tcp 的 z 轴当工具轴钉向 -z，夹爪横转 90°（base 碰方块、
link5 穿透、link7_c 蹭桌全是次生症状）。

几何余量（探针实测 body 系 bbox，手指沿 -y、开合沿 ±x）：
- 指板内侧面满开（ctrl=0.038）间距 64.8mm，方块 60mm -> 每侧 2.4mm，
  下探段必须笛卡尔直线；航点链逐段弓高实测 <= 0.65mm
- 指板竖向跨度 38mm（body y[-0.047,-0.009]），tcp 置 0.72 -> 指板
  覆盖方块 0.701~0.739；壳体下缘 = tcp+0.0256 = 0.7456，高于方块顶
  0.74（5.6mm 裕量）；指板最深点 0.701，距桌顶 2.1cm
- 全相位链（含零位->approach 摆臂）臂/夹爪最低 geom z=0.7009，不蹭桌

夹爪语义（实测）：ctrl=0.038 指板张开（内间隙 64.8mm）、ctrl=0 合至
内间隙 36mm——先张开套住 60mm 方块再闭合压紧（position 伺服 kp=100，
方块被挡在 span 60mm 处，每指压力 ~1.35N，摩擦 1.0 下抗 0.98N 重力
余量 2.7 倍）。

控制方式：目标斜坡（独立参考状态限速逼近），避免位置伺服一步到位
甩臂把方块从指间震落；close 相位夹爪伺服被方块挡在行程中段、永远
到不了 ctrl 目标，到位判定不判夹爪、只按 tick 硬超时推进。
"""
from agentsim.controller import Robot

# (相位, q1..q7 伺服 ctrl 目标 rad, 夹爪 ctrl m, 关节斜坡速度 rad/s)
# 关节目标由 6D 位姿 IK 标定（tcp 位置 + 手指朝下 + 开口轴对齐世界 x），
# 航点链 z 每 2.5~3cm 一个 IK 解（warm start 保同族解），逐段弓高 <= 0.65mm。
# 在 IK 解上叠加重力前馈 qfrc_bias/kp（panda 伺服 force=kp(ctrl-q)-kv*qdot，
# 稳态 kp(ctrl-q)=重力矩）：无补偿时稳态下垂 ~0.01 rad -> tcp 偏 5~8mm，
# 会吃掉指板每侧 2.4mm 余量蹭飞方块；补偿后稳态 tcp 误差 0.00mm。
PHASES = [
    ("approach", (-1.3176, 0.8125, 1.4687, -2.1172, -0.9286, 2.0114, -1.7435), 0.038, 0.8),
    ("desc830",  (-1.2793, 0.8376, 1.4001, -2.1300, -0.9798, 2.0574, -1.7195), 0.038, 0.4),
    ("desc800",  (-1.2404, 0.8697, 1.3332, -2.1353, -1.0365, 2.0972, -1.6933), 0.038, 0.4),
    ("desc770",  (-1.1983, 0.9039, 1.2638, -2.1328, -1.0934, 2.1361, -1.6674), 0.038, 0.4),
    ("desc745",  (-1.1635, 0.9333, 1.2070, -2.1259, -1.1395, 2.1672, -1.6469), 0.038, 0.4),
    ("grasp",    (-1.1302, 0.9647, 1.1527, -2.1151, -1.1852, 2.1961, -1.6271), 0.038, 0.3),
    ("close",    (-1.1302, 0.9647, 1.1527, -2.1151, -1.1852, 2.1961, -1.6271), 0.000, 0.3),
    ("rise750",  (-1.1692, 0.9257, 1.2165, -2.1275, -1.1292, 2.1628, -1.6514), 0.000, 0.4),
    ("rise790",  (-1.2256, 0.8776, 1.3083, -2.1356, -1.0526, 2.1129, -1.6860), 0.000, 0.4),
    ("rise830",  (-1.2793, 0.8376, 1.4001, -2.1300, -0.9798, 2.0574, -1.7195), 0.000, 0.4),
    ("lift",     (-1.3176, 0.8125, 1.4687, -2.1172, -0.9286, 2.0114, -1.7435), 0.000, 0.4),
    ("hold",     (-1.3176, 0.8125, 1.4687, -2.1172, -0.9286, 2.0114, -1.7435), 0.000, 0.0),
]
TOL = 0.05        # 7 关节到位容差 rad
STABLE = 20       # 连续 N tick 在容差内才算到位
TIMEOUT = 400     # 单相位 tick 上限（防死锁）
CLOSE_TICKS = 90  # close 相位夹爪行程 ~15 tick + 压紧等待，硬超时推进
HOLD_TICKS = 40   # hold 相位保持时长（供内核 success 判定窗口）
STEP_F = 0.04     # 夹爪斜坡速度 m/s
TICK = 0.064      # 控制周期（ctrl_period 32 x timestep 0.002）

robot = Robot()
phase_i = stable = phase_ticks = 0
ramp_state: list[float] | None = None


def ramp(cur: float, tgt: float, step: float) -> float:
    return cur + max(-step, min(step, tgt - cur))


while robot.step():
    name, q7, grip, speed = PHASES[phase_i]
    q = robot.obs["qpos"]
    if ramp_state is None:
        # 首 tick 从实际状态起步（7 关节 + 夹爪），顺序对应 set_all
        ramp_state = list(q[:7]) + [float(q[7] + q[8])]
    step_j = speed * TICK
    for k, tgt in enumerate(q7):
        ramp_state[k] = ramp(ramp_state[k], tgt, step_j)
    ramp_state[7] = ramp(ramp_state[7], grip, STEP_F * TICK)
    robot.set_all(ramp_state)

    # 到位判定只看 7 个臂关节（夹爪被方块挡住永远到不了目标）
    err = max(abs(q[k] - g) for k, g in enumerate(q7))
    phase_ticks += 1
    stable = stable + 1 if err < TOL else 0

    done = stable >= STABLE or phase_ticks >= TIMEOUT
    if name == "close" and phase_ticks >= CLOSE_TICKS:
        done = True
    # hold：稳定保持满 HOLD_TICKS，或臂已稳定到位（stable 达标）即可收尾
    # ——必须先于下方 done 推进判断，否则 phase_i 会越出 PHASES
    if name == "hold" and (phase_ticks >= HOLD_TICKS or done):
        break
    if done:
        phase_i += 1
        stable = phase_ticks = 0

robot.close()
