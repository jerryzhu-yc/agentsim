"""M0.7 验收：场景模板生成器（方案 5.7 A 档）。

验收标准：每个模板产出的 YAML 全部通过 schema 校验 + 编译 + 冒烟跑
100 tick；同一命令永远产出同一世界。
"""
import glob
import pathlib
import subprocess
import sys

import pytest

from agentsim.kernel.sim import Simulation
from agentsim.scene_gen import TEMPLATES, generate
from agentsim.world.schema import load_world

ROOT = pathlib.Path(__file__).parent.parent

# pallet 依赖 menagerie UR5e 缓存（首次需联网下载一次）
_UR5E = glob.glob(
    "/home/yuancheng-zhu/.cache/mujoco_menagerie/models/"
    "universal_robots_ur5e-*/ur5e.xml")

# (模板, 参数变体)：默认 + 每个参数至少一个变体
CASES = [
    ("tabletop", {}),
    ("tabletop", {"objects": 1}),
    ("tabletop", {"objects": 3}),
    ("push", {}),
    ("maze", {}),
    ("maze", {"rows": 5, "cols": 6}),
    ("race", {}),
    ("race", {"lanes": 3}),
    ("empty", {}),
    ("empty", {"robot": "diff_car"}),
    ("empty", {"robot": "none"}),
]
PALLET_CASES = [("pallet", {}), ("pallet", {"cubes": 1}),
                ("pallet", {"cubes": 6})]


@pytest.mark.parametrize("name,params", CASES)
def test_template_smoke(tmp_path, name, params):
    out = tmp_path / f"{name}.yaml"
    out.write_text(generate(name, tmp_path, params), encoding="utf-8")
    spec = load_world(out)                 # schema 校验
    with Simulation(spec) as sim:          # 编译 + 100 tick 冒烟
        stats = sim.run(max_ticks=100)
    assert stats["ticks"] == 100
    assert not stats["nan"]
    assert stats["crashed"] == []


@pytest.mark.parametrize("name,params", PALLET_CASES)
@pytest.mark.skipif(not _UR5E, reason="menagerie UR5e 未下载，pallet 无法构建")
def test_pallet_smoke(tmp_path, name, params):
    """M6 码垛分区：UR5e+2FG7 + N cube + N 目标区，编译并冒烟 100 tick。"""
    import yaml
    from agentsim.world.builder import build_world

    out = tmp_path / "pallet.yaml"
    text = generate(name, tmp_path, params)
    out.write_text(text, encoding="utf-8")
    spec = load_world(out)
    n = params.get("cubes", 3)
    assert spec.name == f"pallet_{n}"
    assert len(spec.robots) == 1
    assert spec.robots[0].gripper.model == "onrobot_2fg7"
    # A/B 两区物体齐全（N cube + N pad + 1 table）
    names = {o.name for o in spec.scene.objects}
    assert names >= ({f"cube_{i}" for i in range(n)}
                     | {f"pad_{i}" for i in range(n)})
    # 每个 cube 一条 zone reward
    assert len(spec.task.reward) == n

    # 编译 + 夹爪设备并入
    model, index = build_world(spec)
    ridx = index.robots["arm"]
    assert "gripper/gripper_act" in ridx.actuators

    with Simulation(spec) as sim:
        stats = sim.run(max_ticks=100)
    assert stats["ticks"] == 100
    assert not stats["nan"]
    assert stats["crashed"] == []
    # 产出是合法 YAML 且 model 是绝对路径（menagerie 缓存）
    assert str(yaml.safe_load(text)["robots"][0]["model"]).startswith("/")


def test_deterministic(tmp_path):
    for name in TEMPLATES:
        if name == "pallet" and not _UR5E:
            continue               # 无缓存时不触发下载
        assert generate(name, tmp_path) == generate(name, tmp_path)


def test_bad_template_and_params(tmp_path):
    with pytest.raises(Exception, match="未知模板"):
        generate("nope", tmp_path)
    with pytest.raises(Exception, match="超出范围"):
        generate("maze", tmp_path, {"rows": 99})
    with pytest.raises(Exception, match="没有参数"):
        generate("maze", tmp_path, {"bogus": 1})


def test_cli_scenegen(tmp_path):
    out = tmp_path / "maze.yaml"
    r = subprocess.run([sys.executable, "-m", "agentsim", "scenegen", "maze",
                        "--rows", "3", "--cols", "3", "--out", str(out)],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    assert "已生成" in r.stdout
    spec = load_world(out)
    assert spec.name == "maze_3x3"
