"""M1 心跳验收：内核崩溃时控制器可感知退出。

集成级：测试进程 spawn"内核脚本"（起真 Simulation + dump 控制器 pid 到文件），
内核脚本再模拟死亡（os._exit，不走任何收尾）或正常运行（思考 3s + run + close）。
"""
import os
import pathlib
import subprocess
import sys
import time

import pytest
import zmq

import agentsim.controller as C

ROOT = pathlib.Path(__file__).parent.parent

KERNEL_SRC = '''
import os, sys, time
sys.path.insert(0, {root!r})
from agentsim.world.schema import load_world
from agentsim.kernel.sim import Simulation

spec = load_world({world!r})
sim = Simulation(spec)
with open(sys.argv[1], "w") as f:
    f.write(str(sim._procs["arm"].pid))
mode = sys.argv[2]
if mode == "crash":
    time.sleep(1.5)          # 留时间握手（hello 无人应答也无妨：等待是语义）
    os._exit(9)              # 段错误级死亡：不走任何收尾代码
elif mode == "think":
    time.sleep(3.0)          # Agent 思考时间（远超 100ms 轮询周期）
    stats = sim.run()
    sim.close()
    sys.exit(0 if stats["success"] else 3)
'''


def _pid_alive(pid: int) -> bool:
    # os.kill(pid, 0) 在 Windows 上语义是 CTRL_C_EVENT（发信号），
    # 不是存活探测——跨平台要用 Win32 API
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259    # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _spawn_kernel(tmp_path, mode):
    script = tmp_path / "kernel.py"
    script.write_text(
        KERNEL_SRC.format(root=str(ROOT),
                          world=str(ROOT / "worlds" / "pick_cube.yaml")),
        encoding="utf-8")
    pidfile = tmp_path / "ctrl.pid"
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.Popen(
        [sys.executable, str(script), str(pidfile), mode],
        cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), pidfile


def _wait_pidfile(pidfile, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pidfile.exists() and pidfile.read_text().strip():
            return int(pidfile.read_text().strip())
        time.sleep(0.05)
    raise AssertionError("内核没写出控制器 pid 文件（spawn 失败？）")


def test_kernel_hardkill_orphans_controller(tmp_path):
    """T1 核心验收：内核 os._exit(9) 后，控制器必须在 2s 内退出。"""
    proc, pidfile = _spawn_kernel(tmp_path, "crash")
    cpid = _wait_pidfile(pidfile)
    proc.wait(timeout=15)
    assert proc.returncode == 9
    deadline = time.time() + 2.0
    while time.time() < deadline and _pid_alive(cpid):
        time.sleep(0.1)
    assert not _pid_alive(cpid), f"强杀内核后控制器 {cpid} 应在 2s 内退出"


def test_think_time_no_false_kill(tmp_path):
    """T2：Agent 思考 3s 不误杀；run 正常完成、控制器正常收尾退出。"""
    proc, pidfile = _spawn_kernel(tmp_path, "think")
    cpid = _wait_pidfile(pidfile)
    proc.wait(timeout=60)
    assert proc.returncode == 0          # run success（否则脚本 exit 3）
    assert not _pid_alive(cpid)          # 控制器已走 stop 收尾路径正常退出


def test_orphan_detection_raises(monkeypatch):
    """T3：_recv 孤儿分支——父进程死亡检测触发时 RuntimeError 带明确提示。"""
    robot = C.Robot.__new__(C.Robot)     # 绕过 __init__（无需真内核）
    ctx = zmq.Context()
    robot._sock = ctx.socket(zmq.REQ)
    robot._sock.setsockopt(zmq.LINGER, 0)
    robot._sock.connect("tcp://127.0.0.1:1")   # 无人监听：永远收不到回复
    robot._parent_pid = os.getppid()
    robot._sock.send(b'{"op": "hello"}')       # REQ 入队不阻塞
    # 直接模拟死亡判定（真实机制在 _parent_dead：POSIX 看 getppid 变化，
    # Windows 用 OpenProcess/GetExitCodeProcess，平台差异对 _recv 透明）
    monkeypatch.setattr(C.Robot, "_parent_dead", lambda self: True)
    with pytest.raises(RuntimeError, match="内核已退出"):
        robot._recv()
    robot._sock.close()
    ctx.term()


def test_parent_dead_live_parent():
    """_parent_dead：父进程（pytest 自身）活着时返回 False。"""
    robot = C.Robot.__new__(C.Robot)
    robot._parent_pid = os.getppid()
    assert robot._parent_dead() is False


def test_parent_dead_dead_process():
    """_parent_dead：目标进程已退出时返回 True（跨平台死亡探测）。"""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    robot = C.Robot.__new__(C.Robot)
    robot._parent_pid = p.pid
    assert robot._parent_dead() is True
