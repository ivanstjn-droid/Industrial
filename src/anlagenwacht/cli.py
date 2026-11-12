"""Command line entry point.

    anlagenwacht benchmark [tep|skab|all]
    anlagenwacht investigate tep-4 1
    anlagenwacht serve --port 8000
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="anlagenwacht")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benchmark", help="Run benchmarks on public datasets")
    b.add_argument("which", nargs="?", default="all", choices=["tep", "skab", "maritime", "all"])
    i = sub.add_parser("investigate", help="Investigate an alarm (Claude, otherwise rule-based)")
    i.add_argument("plant", help="e.g. demo, tep-4, skab-valve1-1, vessel-hull_fouling, naval-compressor")
    i.add_argument("alarm", type=int, nargs="?", default=None)
    s = sub.add_parser("serve", help="Start API + web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.cmd == "benchmark":
        from anlagenwacht.evaluate import main as bench

        bench([args.which])
    elif args.cmd == "investigate":
        from anlagenwacht.agent import investigate
        from anlagenwacht.api import get_plant

        plant = get_plant(args.plant)
        if not plant.episodes:
            print("No alarms.")
            return
        alarm = args.alarm or max(plant.episodes, key=lambda e: e["n_samples"] * e["peak_score"])["id"]
        result = investigate(plant, alarm)
        print(result["report"])
        print(f"\n[mode: {result['mode']}, tool calls: {len(result['tool_calls'])}]")
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("anlagenwacht.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
