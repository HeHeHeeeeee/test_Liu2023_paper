"""Generate paper-style tables and figures from reproduction metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


TASK_LABELS = {
    "task_a": "Task A",
    "task_b": "Task B",
    "task_c": "Task C",
    "task_d": "Task D",
    "task_e": "Task E",
    "task_f": "Task F",
}


def load_results(metrics_dir: Path) -> dict[str, Any]:
    with open(metrics_dir / "all_results.json") as f:
        return json.load(f)


def write_table_1(results: dict[str, Any], out_path: Path) -> None:
    lines = ["# Table 1. Total trial numbers", ""]
    for task_name, rows in results.items():
        if not isinstance(rows, dict) or "error" in rows:
            continue
        lines.append(f"## {TASK_LABELS.get(task_name, task_name)}")
        lines.append("")
        lines.append("| Method | TTN | Successful runs | Success rate | Reached target |")
        lines.append("|---|---:|---:|---:|---|")
        for method, row in rows.items():
            lines.append(
                f"| {method} | {row['TTN']} | "
                f"{row['successful_runs']}/{row['total_trials']} | "
                f"{row['success_rate']:.4f} | {row.get('reached_target_successes', True)} |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines))


def _plot_metric(
    results: dict[str, Any],
    metric: str,
    err_metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    log_y: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task_items = [
        (task_name, rows)
        for task_name, rows in results.items()
        if isinstance(rows, dict) and "error" not in rows
    ]
    if not task_items:
        return

    fig, axes = plt.subplots(
        len(task_items),
        1,
        figsize=(9, max(3.0, 2.1 * len(task_items))),
        constrained_layout=True,
    )
    if len(task_items) == 1:
        axes = [axes]

    colors = ["#31688e", "#35b779", "#fde725", "#cc4778", "#440154"]
    for ax, (task_name, rows) in zip(axes, task_items):
        methods = list(rows.keys())
        values = [rows[m].get(metric, 0.0) for m in methods]
        errors = [rows[m].get(err_metric, 0.0) for m in methods]
        if log_y:
            values = [max(v, 1.0e-16) for v in values]
        x = np.arange(len(methods))
        ax.bar(x, values, yerr=errors, capsize=4, color=colors[: len(methods)])
        ax.set_xticks(x, methods)
        ax.set_ylabel(ylabel)
        ax.set_title(TASK_LABELS.get(task_name, task_name), loc="left", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        if log_y:
            ax.set_yscale("log")

    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def generate_outputs(metrics_dir: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(metrics_dir)
    write_table_1(results, figures_dir / "Table_1.md")
    _plot_metric(
        results,
        "avg_iterations",
        "std_iterations",
        "Iterations",
        "Figure 3. Average iteration times over successful runs",
        figures_dir / "Fig_3_avg_iterations.png",
    )
    _plot_metric(
        results,
        "avg_gradient_norm",
        "std_gradient_norm",
        "Normalized gradient norm",
        "Figure 4. Average normalized gradient norms over all trials",
        figures_dir / "Fig_4_gradient_norms.png",
        log_y=True,
    )
