#!/usr/bin/env python3
"""Run the Liu 2023 reproduction tasks.

The simulations are dense exact state-vector runs.  They are intentionally
kept in one deterministic code path so Task A-F use the same ansatz,
Hamiltonian, transfer and BFGS logic.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from src.experiments import ExperimentSettings, TASKS, run_task
from src.plotter import generate_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tasks",
        nargs="*",
        choices=sorted(TASKS),
        help="Tasks to run. Defaults to all tasks.",
    )
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--base-successes", type=int, default=int(os.getenv("BASE_SUCCESSES", "100")))
    parser.add_argument(
        "--target-successes", type=int, default=int(os.getenv("TARGET_SUCCESSES", "100"))
    )
    parser.add_argument("--max-iter-base", type=int, default=int(os.getenv("MAX_ITER_BASE", "500")))
    parser.add_argument(
        "--max-iter-target", type=int, default=int(os.getenv("MAX_ITER_TARGET", "500"))
    )
    parser.add_argument(
        "--max-trials-base", type=int, default=int(os.getenv("MAX_TRIALS_BASE", "0"))
    )
    parser.add_argument(
        "--max-trials-target", type=int, default=int(os.getenv("MAX_TRIALS_TARGET", "0"))
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", str(max(1, (os.cpu_count() or 2) - 2)))),
    )
    parser.add_argument("--no-history", action="store_true", help="Disable per-iteration JSONL logs")
    parser.add_argument("--no-plots", action="store_true", help="Disable Table 1/Fig. 3/Fig. 4 generation")
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED")) if os.getenv("SEED") else None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = args.tasks or list(TASKS)
    output_root = Path(args.output)
    settings = ExperimentSettings(
        base_successes=args.base_successes,
        target_successes=args.target_successes,
        max_iter_base=args.max_iter_base,
        max_iter_target=args.max_iter_target,
        max_trials_base=args.max_trials_base,
        max_trials_target=args.max_trials_target,
        workers=args.workers,
        record_history=not args.no_history,
        seed=args.seed,
    )

    started = time.time()
    all_results = {}
    for task_name in selected:
        print(f"\n=== {task_name} ===")
        try:
            all_results[task_name] = run_task(TASKS[task_name], output_root, settings)
            for pattern, row in all_results[task_name].items():
                print(
                    f"  {pattern:<6} TTN={row['TTN']:<4} "
                    f"success={row['successful_runs']}/{row['total_trials']} "
                    f"avg_iter={row['avg_iterations']:.1f} "
                    f"G={row['avg_gradient_norm']:.3e}"
                )
        except Exception as exc:
            all_results[task_name] = {"error": str(exc)}
            print(f"  failed: {exc}")

    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    if not args.no_plots:
        figures_dir = output_root / "figures"
        generate_outputs(metrics_dir, figures_dir)
        print(f"Figures written to {figures_dir}")

    print(f"\nComplete in {time.time() - started:.1f}s")
    print(f"Results written to {metrics_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
