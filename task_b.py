#!/usr/bin/env python3
"""Task B: 4-qubit to 8-qubit TFIM, HEA, network transfer."""

from __future__ import annotations

import os
from pathlib import Path

from src.experiments import ExperimentSettings, TASKS, run_task


def settings_from_env() -> ExperimentSettings:
    return ExperimentSettings(
        base_successes=int(os.getenv("BASE_SUCCESSES", "100")),
        target_successes=int(os.getenv("TARGET_SUCCESSES", "100")),
        max_iter_base=int(os.getenv("MAX_ITER_BASE", "500")),
        max_iter_target=int(os.getenv("MAX_ITER_TARGET", "500")),
        max_trials_base=int(os.getenv("MAX_TRIALS_BASE", "0")),
        max_trials_target=int(os.getenv("MAX_TRIALS_TARGET", "0")),
        workers=int(os.getenv("WORKERS", str(max(1, (os.cpu_count() or 2) - 2)))),
        record_history=os.getenv("RECORD_HISTORY", "1") != "0",
        seed=int(os.getenv("SEED")) if os.getenv("SEED") else None,
    )


def main() -> int:
    results = run_task(TASKS["task_b"], Path("outputs"), settings_from_env())
    print("Task B complete")
    for pattern, row in results.items():
        print(f"  {pattern}: TTN={row['TTN']}, avg_iter={row['avg_iterations']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
