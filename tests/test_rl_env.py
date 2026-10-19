"""M3 验收：RL 训练环境（gymnasium.Env 包装）。

核心守护（T4）：同动作序列两个 episode 全程逐位一致——World as Code 的
确定性承诺对 RL 成立（禁 allclose，严格 array_equal）。
"""
import pathlib

import numpy as np
import pytest

from agentsim.rl_env import AgentSimEnv
from agentsim.world.schema import parse_world
import yaml

ROOT = pathlib.Path(__file__).parent.parent
WORLD = ROOT / "worlds" / "pick_cube.yaml"


def _have_screen() -> bool:
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_string("<mujoco/>")
        r = mujoco.Renderer(m, 32, 32)
        r.render()
        r.close()
        return True
    except Exception:
        return False


def _pick_spec(with_sensors: bool = False):
    """pick_cube 的 WorldSpec（controller 字段被 RL env 忽略——不起子进程）。"""
    text = WORLD.read_text(encoding="utf-8")
    if with_sensors:
        anchor = "    controller: ../controllers/pick_place.py   # 侧扫推式相位机"
        text = text.replace(anchor, anchor + "\n    sensors:\n"
            + "      - {name: j1_enc, type: encoder, joint: j1}\n"
            + "      - {name: wrist_imu, type: imu, body: wrist, pos: [0, 0, 0.02]}")
    spec = parse_world(yaml.safe_load(text))
    spec.base_dir = ROOT / "worlds"
    return spec


def _reach_push_action(t: int) -> np.ndarray:
    """T5 探针定值的两相位固定序列（reach 120 tick 摆位 -> a1 慢扫 0.55）。"""
    if t < 120:
        return np.array([0.0, 1.57, 0.0618, -0.03, 0.03], dtype=np.float32)
    a1 = min(0.55, (t - 120 + 1) * 0.005)
    return np.array([a1, 1.57, 0.0618, -0.03, 0.03], dtype=np.float32)


class TestSpaces:
    def test_action_space_matches_actuators(self):
        """动作空间 = 全执行器拼接，界 = ctrlrange（T1）。"""
        env = AgentSimEnv(WORLD)
        assert env.action_space.shape == (5,)
        lo = env.action_space.low
        hi = env.action_space.high
        assert lo[0] == -3.14 and hi[0] == 3.14      # a1
        assert lo[1] == -1.57 and hi[1] == 1.57       # a2
        assert lo[3] == -0.03 and hi[3] == 0.0       # a_l
        env.close()

    def test_obs_dim_qpos_qvel_sensors(self):
        """观测维度 = qpos + qvel + sensors 拼接宽（无传感器 10 / 带 17）。"""
        env = AgentSimEnv(_pick_spec())
        assert env._obs_dim == 5 + 5
        env.close()
        env = AgentSimEnv(_pick_spec(with_sensors=True))
        assert env._obs_dim == 5 + 5 + 1 + 6        # j1_enc + imu(6)
        env.close()

    def test_multi_robot_flat_action(self):
        """多机器人世界：动作空间是各机器人执行器 flat 拼接。"""
        env = AgentSimEnv(ROOT / "worlds" / "twin_arms.yaml")
        assert env.action_space.shape[0] == sum(
            len(r.actuators) for r in env.index.robots.values()) >= 10
        env.close()


class TestResetAndStep:
    def test_reset_deterministic_same_seed(self):
        """同 seed 两次 reset 的 obs 逐位一致（T2）。"""
        env = AgentSimEnv(WORLD)
        o1, i1 = env.reset(seed=42)
        o2, i2 = env.reset(seed=42)
        assert np.array_equal(o1, o2)
        assert i1["tick"] == i2["tick"] == 0
        env.close()

    def test_step_random_rollout_finite(self):
        """随机策略 500 步不崩、reward 有限（T3）。"""
        env = AgentSimEnv(WORLD)
        obs, info = env.reset(seed=7)
        rewards = []
        for _ in range(500):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert np.isfinite(r) and np.all(np.isfinite(obs))
            rewards.append(r)
            if term or trunc:
                obs, info = env.reset()
        assert len(rewards) == 500
        env.close()

    def test_truncated_at_max_ticks(self):
        """tick 到 max_ticks 时 truncated=True。"""
        env = AgentSimEnv(WORLD)
        env.reset(seed=1)
        assert env.task.max_ticks == 5000
        for _ in range(5000):
            obs, r, term, trunc, info = env.step(np.zeros(5, dtype=np.float32))
            if term:
                break
        assert info["tick"] == 5000 and (trunc or term)
        env.close()

    def test_action_dim_error(self):
        env = AgentSimEnv(WORLD)
        env.reset()
        from agentsim.errors import WorldError
        with pytest.raises(WorldError, match="5 维"):
            env.step(np.zeros(3, dtype=np.float32))
        env.close()


class TestBitwiseGuard:
    def test_two_episodes_bitwise_identical(self):
        """T4 合并门禁：同动作序列两个 episode，逐 tick qpos 轨迹 array_equal。"""
        actions = [_reach_push_action(t) for t in range(300)]

        trajectories = []
        for _ in range(2):
            env = AgentSimEnv(WORLD)
            env.reset(seed=0)
            qpos_seq = []
            for a in actions:
                obs, r, term, trunc, info = env.step(a)
                qpos_seq.append(env.data.qpos.copy())
                if term:
                    break
            trajectories.append((np.stack(qpos_seq), term, info["tick"]))
            env.close()
        (q1, term1, tick1), (q2, term2, tick2) = trajectories
        assert term1 == term2 and tick1 == tick2
        assert q1.shape == q2.shape
        assert np.array_equal(q1, q2), "两个 episode 的 qpos 轨迹必须逐位一致"


class TestTaskSuccess:
    def test_reach_push_terminates_success(self):
        """T5 端到端：j1 侧扫推 cube 入 zone，terminated=True。"""
        env = AgentSimEnv(WORLD)
        env.reset(seed=0)
        term = trunc = False
        for t in range(300):
            obs, r, term, trunc, info = env.step(_reach_push_action(t))
            if term:
                break
        assert term and not trunc
        assert info["tick"] < 300
        cube = env.data.xpos[env.index.objects["cube"]]
        assert cube[1] >= 0.25                        # cube 越过 zone 的 y 界
        env.close()


class TestGymCompliance:
    def test_check_env_official(self):
        """T7 官方合规：gymnasium 自带 env_checker 全过（无渲染实例）。"""
        from gymnasium.utils.env_checker import check_env
        env = AgentSimEnv(WORLD)
        check_env(env, skip_render_check=True)
        env.close()

    @pytest.mark.skipif(not _have_screen(), reason="无离屏渲染（EGL）")
    def test_check_env_with_render(self):
        from gymnasium.utils.env_checker import check_env
        env = AgentSimEnv(WORLD, render_mode="rgb_array")
        check_env(env)
        env.reset(seed=0)
        rgb = env.render()
        assert rgb.shape == (600, 800, 3) and rgb.dtype == np.uint8
        env.close()
