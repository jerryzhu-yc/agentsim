"""M5 P5 示例：用 PPO 训练 Unitree Go2 行走（world 是 zoo demo 生成器导出）。

设计取舍：
- 不做 RL 训练本身（M5 边界）；只给一份可运行的 sb3 PPO 模板
- sb3 是可选依赖（不装时此脚本会给出安装提示后退出）
- demo 走 CPG 控制律已经能稳定行走（go2 600 tick 走 10.27m，见 tests/test_zoo.py），
  这里把"奖励 = root 位移 - 关节失配 - 跌倒惩罚"加在 YAML 任务之上做 RL 微调
- 世界：先 zoo.generate(unitree_go2, demo=True) 拿到 go2 demo 世界；随后
  替换 controller（训练由 env.step 控制，demo.py 控制器不再驱动），保留
  demo_engine 的 integrator=implicitfast/impratio=100（四足稳定关键）

跑：
    pip install stable-baselines3
    ~/miniconda3/bin/python examples/train_ppo_go2.py

输出：models/ppo_go2.zip（policy.zip 可被 demo 控制器加载——具体加载位置见
agentsim/controllers/pick_place.py 顶部说明，本示例不实现 policy 加载，
避免 sb3 不在回归路径里）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from agentsim.rl_env import AgentSimEnv  # noqa: E402
from agentsim.zoo import generate  # noqa: E402


def _make_world_yaml(out_dir: Path) -> Path:
    """生成 go2 demo 世界并改造成 RL 训练用：删 controller（env 直驱）。"""
    r = generate("unitree_go2", out_dir=str(out_dir), demo=False)
    # 拷一份 demo 世界的引擎参数（impratio/integrator/ctrl_period），但
    # 用空 controller 模板（不绑 demo.py）——RL 学策略，不用 CPG 摆动
    import yaml
    world_p = Path(r["world"])
    d = yaml.safe_load(world_p.read_text(encoding="utf-8"))
    d["impratio"] = 100
    d["integrator"] = "implicitfast"
    d["ctrl_period"] = 8
    d["robots"][0].pop("controller", None)
    out = out_dir / "world_rl.yaml"
    out.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    return out


class Go2ForwardReward:
    """在 YAML task 之上叠加行走奖励（env 的 task.reward 仍按 YAML 算）。

    - +x 位移（推进速度）*
    - -|z 偏移|（root 高度偏离 0.3 站立位姿）
    - -动作 L2（能耗正则）
    - 跌倒（rootz < 0.15）终止
    """

    def __init__(self, env: AgentSimEnv):
        self.env = env
        self._root_bid = next(iter(env.index.robots["unitree_go2"].bodies.values()))
        self._last_x = 0.0
        self._target_z = 0.30

    def reset(self):
        self._last_x = self.env.data.xpos[self._root_bid][0]

    def reward(self, action: np.ndarray) -> float:
        p = self.env.data.xpos[self._root_bid]
        dx = p[0] - self._last_x
        self._last_x = p[0]
        forward = 5.0 * dx                   # 主要项：每 step x 位移
        z_pen = -3.0 * abs(p[2] - self._target_z)    # 高度偏离罚
        energy = -0.01 * float(np.sum(action ** 2))   # 能耗正则
        return forward + z_pen + energy

    def fell(self) -> bool:
        return float(self.env.data.xpos[self._root_bid][2]) < 0.15


class _RewardWrapper:
    """gymnasium wrapper：叠加 Go2ForwardReward + 跌倒终止。"""

    def __init__(self, env: AgentSimEnv, bonus: Go2ForwardReward):
        self.env = env
        self.bonus = bonus

    # gymnasium.Env API（手写代理，避免继承 gym.Wrapper 拉 gym 依赖图）
    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self, *, seed=None, options=None):
        o, i = self.env.reset(seed=seed, options=options)
        self.bonus.reset()
        return o, i

    def step(self, action):
        o, r, term, trunc, info = self.env.step(action)
        r += self.bonus.reward(np.asarray(action, dtype=float))
        if self.bonus.fell():
            term = True
        return o, r, term, trunc, info

    def close(self):
        self.env.close()


def main():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        print("此示例需要 stable-baselines3：pip install stable-baselines3",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path("/tmp/agentsim_ppo_go2")
    out_dir.mkdir(parents=True, exist_ok=True)
    world = _make_world_yaml(out_dir)
    print(f"世界：{world}")

    env = AgentSimEnv(world, render_mode=None)
    bonus = Go2ForwardReward(env)
    wrapped = _RewardWrapper(env, bonus)
    vec = DummyVecEnv([lambda: wrapped])

    model = PPO(
        "MlpPolicy", vec,
        n_steps=2048, batch_size=64, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, verbose=1,
        tensorboard_log=str(out_dir / "tb"))
    total = 200_000
    print(f"训练 {total} 步…（中途 Ctrl+C 可保存已训练部分）")
    try:
        model.learn(total_timesteps=total)
    except KeyboardInterrupt:
        print("中断训练，保存已训练模型…")
    out = out_dir / "ppo_go2.zip"
    model.save(str(out))
    print(f"已保存：{out}")
    env.close()


if __name__ == "__main__":
    main()
