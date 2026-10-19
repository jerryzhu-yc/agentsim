"""RL 训练环境：把 World as Code 的确定性世界包成 gymnasium Env。

用法：

    from agentsim.rl_env import AgentSimEnv

    env = AgentSimEnv("worlds/pick_cube.yaml")
    obs, info = env.reset(seed=42)
    for _ in range(500):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

设计要点：
- 单进程直驱：复用 world 层（build_world/resolve_task），不走控制器子进程、
  不起 ZMQ——训练吞吐优先；MCP/内核零改动
- reset 重建 MjData：逐位一致的初态（warmstart 零残留），同 YAML = 确定性
  环境，eval 可复现
- 动作空间：所有机器人执行器 flat 拼接（多机器人也训练得了）
- 观测：各机器人 qpos + qvel + 传感器读数按声明顺序拼接（float32）；
  info["raw"] 保留 float64 原始分量
- task 评估/观测组装语义与内核共用（kernel/task.py，一处实现）
"""
from __future__ import annotations

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError(
        "AgentSimEnv 依赖 gymnasium，请安装：pip install gymnasium") from e

from agentsim.errors import WorldError
from agentsim.kernel.task import (joint_width, robot_obs, sensors_obs,
                                  task_reward, task_success)
from agentsim.render import render_offscreen
from agentsim.world.builder import build_world, resolve_task
from agentsim.world.schema import WorldSpec, load_world, parse_world


class AgentSimEnv(gym.Env):
    """一个 YAML 世界 = 一个确定性 gym 环境。"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, world, render_mode: str | None = None):
        if render_mode not in (None, "rgb_array"):
            raise WorldError(f"不支持的 render_mode: {render_mode}，"
                             f"可用: ['rgb_array']")
        self.render_mode = render_mode
        if isinstance(world, WorldSpec):
            self.world_spec = world
        elif isinstance(world, dict):
            self.world_spec = parse_world(world)
        else:
            self.world_spec = load_world(str(world))
        self.model, self.index = build_world(self.world_spec)
        self.task = resolve_task(self.world_spec.task, self.index)
        self.data = mujoco.MjData(self.model)
        self._tick = 0
        self._renderer = None
        self._rend_wh: tuple[int, int] | None = None

        # 动作空间：全机器人执行器 flat 拼接，界 = 各自 ctrlrange
        lows, highs, act_ids = [], [], []
        for rd in self.world_spec.robots:
            for aid in self.index.robots[rd.name].actuators.values():
                lo, hi = self.model.actuator_ctrlrange[aid]
                lows.append(float(lo))
                highs.append(float(hi))
                act_ids.append(int(aid))
        if not act_ids:
            raise WorldError("世界里没有任何执行器，做不成 RL 动作空间")
        self._act_ids = act_ids
        self.action_space = spaces.Box(
            np.asarray(lows, dtype=np.float32),
            np.asarray(highs, dtype=np.float32), dtype=np.float32)

        # 观测空间：各机器人 qpos + qvel + sensors 拼接宽度
        dim = 0
        for rd in self.world_spec.robots:
            ridx = self.index.robots[rd.name]
            qw, vw = joint_width(self.model, ridx)
            dim += qw + vw + sum(six.dim for six in ridx.sensors.values())
        self._obs_dim = dim
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(dim,), dtype=np.float32)

    # -------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # 重建 MjData = 逐位一致初态（零 warmstart 残留；与新进程等价）。
        # mj_forward 只为把 sensordata/xpos 派生量备好给初始 obs——
        # 全新 MjData 无快照可破，不触碰 bit 级重现约束（那约束针对 load_state）
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._tick = 0
        return self._obs(), self._info()

    def step(self, action):
        a = np.asarray(action, dtype=float).reshape(-1)
        if len(a) != len(self._act_ids):
            raise WorldError(
                f"动作需要 {len(self._act_ids)} 维（全执行器拼接），"
                f"实际: {len(a)}")
        self.data.ctrl[self._act_ids] = a
        for _ in range(self.world_spec.ctrl_period):
            mujoco.mj_step(self.model, self.data)
        self._tick += 1
        reward = float(task_reward(self.task, self.model, self.data))
        success = task_success(self.task, self.data)
        terminated = bool(success)
        truncated = (self.task.max_ticks is not None
                     and self._tick >= self.task.max_ticks)
        return self._obs(), reward, terminated, truncated, self._info()

    def render(self):
        if self.render_mode is None:
            raise WorldError("render_mode 未开启（构造时传 render_mode='rgb_array'）")
        if self._renderer is None:            # 渲染器跨 step 缓存复用
            self._renderer = mujoco.Renderer(self.model, 600, 800)
        return render_offscreen(self.model, self.data, 800, 600,
                                renderer=self._renderer)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -------------------------------------------------- 内部
    def _obs(self) -> np.ndarray:
        vec: list[float] = []
        for rd in self.world_spec.robots:
            ridx = self.index.robots[rd.name]
            qpos, qvel = robot_obs(ridx, self.model, self.data)
            vec += qpos + qvel
            sn = sensors_obs(ridx, self.data)
            for six in ridx.sensors.values():       # 按声明顺序拼接
                vals = sn[six.name]
                vec += [vals] if isinstance(vals, float) else list(vals)
        out = np.asarray(vec, dtype=np.float32)
        assert out.shape == (self._obs_dim,), (out.shape, self._obs_dim)
        return out

    def _info(self) -> dict:
        raw = {}
        for rd in self.world_spec.robots:
            ridx = self.index.robots[rd.name]
            qpos, qvel = robot_obs(ridx, self.model, self.data)
            raw[rd.name] = {
                "qpos": qpos, "qvel": qvel,
                "ctrl": list(map(float, self.data.ctrl[
                    list(ridx.actuators.values())])),
                "sensors": sensors_obs(ridx, self.data),
            }
        return {"tick": self._tick, "time": float(self.data.time), "raw": raw}
