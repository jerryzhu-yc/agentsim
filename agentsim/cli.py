"""agentsim 命令行入口（人类用户；AI 用户走 mcp_server）。

用法：
    python -m agentsim run worlds/pick_cube.yaml [--ticks N]
    python -m agentsim mcp                     # MCP stdio 服务器（AI Agent 接入）
    python -m agentsim gui                     # 桌面 GUI（PySide6，需装 PySide6-Essentials）
"""
import argparse
import json


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agentsim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="运行一个世界")
    run_p.add_argument("world", help="世界 YAML 文件路径")
    run_p.add_argument("--ticks", type=int, default=None,
                       help="最大 tick 数（缺省用世界 task.max_ticks）")
    run_p.add_argument("--json", action="store_true",
                       help="以 JSON 输出统计（机器可读）")
    sub.add_parser("mcp", help="启动 MCP stdio 服务器")
    sub.add_parser("gui", help="启动桌面 GUI（可视化 + 观测面板）")
    gen_p = sub.add_parser("scenegen", help="从模板生成世界 YAML")
    gen_p.add_argument("template", help="模板名")
    gen_p.add_argument("--out", default=None,
                       help="输出文件路径（缺省打印到 stdout）")
    # parse_known_args：scenegen 的模板参数是动态的，外层放行、内层消化
    args, rest = ap.parse_known_args(argv)
    if rest and args.cmd != "scenegen":
        ap.error(f"无法识别的参数: {' '.join(rest)}")

    if args.cmd == "run":
        _run(args)
    elif args.cmd == "mcp":
        from agentsim.mcp_server import serve
        serve()
    elif args.cmd == "gui":
        try:
            from agentsim.gui import main as gui_main
        except ImportError as e:
            raise SystemExit(
                f"GUI 需要 PySide6：pip install PySide6-Essentials（{e}）")
        raise SystemExit(gui_main())
    elif args.cmd == "scenegen":
        _scenegen(args, argv)


def _scenegen(args, argv=None):
    import pathlib

    from agentsim import scene_gen

    # 动态模板参数（各模板自己声明；scenegen/--out 等其余参数被跳过）
    ap2 = argparse.ArgumentParser(add_help=False)
    for pname, (ptype, default, help_) in \
            scene_gen.TEMPLATES.get(args.template, (None, {}, None))[1].items():
        ap2.add_argument(f"--{pname}", type=ptype, default=default, help=help_)
    p = vars(ap2.parse_known_args(argv)[0])

    base = pathlib.Path(args.out).resolve().parent if args.out \
        else pathlib.Path.cwd()
    try:
        text = scene_gen.generate(args.template, base, p)
    except Exception as e:
        raise SystemExit(f"生成失败: {e}")
    if args.out:
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
        print(f"已生成 {args.out}")
    else:
        print(text, end="")


def _run(args):
    from agentsim.kernel.sim import Simulation
    from agentsim.world.schema import load_world

    try:
        spec = load_world(args.world)
    except Exception as e:
        raise SystemExit(f"世界加载失败: {e}")

    with Simulation(spec) as sim:
        stats = sim.run(max_ticks=args.ticks)
        objects = {name: [round(float(x), 4) for x in sim.data.xpos[bid]]
                   for name, bid in sim.index.objects.items()}

    if args.json:
        print(json.dumps({**stats, "objects": objects}, ensure_ascii=False))
        return
    print(f"世界 '{spec.name}'：{len(spec.scene.objects)} 物体，"
          f"{len(spec.robots)} 机器人")
    print(f"运行 {stats['ticks']} ticks（墙钟 {stats['wall_time']}s，"
          f"仿真 {stats['sim_time']}s）")
    if stats["lag"]:
        print(f"控制器超时补位 {stats['lag']} 次（策略 {spec.runtime.timeout_policy}）")
    if stats["crashed"]:
        print(f"崩溃的控制器: {stats['crashed']}")
    print(f"物理状态: {'NaN!' if stats['nan'] else '正常'}")
    if stats["success"] is not None:
        print(f"任务: {'成功' if stats['success'] else '未完成'}"
              f"（reward {stats['reward']}）")
    for name, pos in objects.items():
        print(f"  {name} 最终位置: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")


if __name__ == "__main__":
    main()
