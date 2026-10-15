"""M0.7 验收：场景模板生成器（方案 5.7 A 档）。

验收标准：每个模板产出的 YAML 全部通过 schema 校验 + 编译 + 冒烟跑
100 tick；同一命令永远产出同一世界。
"""
import pathlib
import subprocess
import sys

import pytest

from agentsim.kernel.sim import Simulation
from agentsim.scene_gen import TEMPLATES, generate
from agentsim.world.schema import load_world

ROOT = pathlib.Path(__file__).parent.parent

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


def test_deterministic(tmp_path):
    for name in TEMPLATES:
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
