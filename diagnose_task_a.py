#!/usr/bin/env python3
"""Task A diagnostic runs with isolated logs.

This script is intentionally separate from the paper reproduction outputs.
It writes every diagnostic run under diagnostics/<timestamp>_task_a_base/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from src.ansatz import AnsatzConfig, HEA
from src.engine import apply_cz, apply_rx, apply_rz, zero_state
from src.hamiltonians import TransverseFieldIsing
from src.optimizer import optimize_vqa
from src.transfer import random_parameters


def write_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def make_run_dir(root: Path, name: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_current_bfgs(ansatz: HEA, ham: TransverseFieldIsing, rng: np.random.Generator, max_iter: int):
    init_params = random_parameters(ansatz.n_params, rng)
    result = optimize_vqa(
        ansatz,
        ham.matrix,
        ham.exact_energy,
        init_params,
        max_iter=max_iter,
        record_history=False,
    )
    return {
        "optimizer": "repo_bfgs_finite_diff",
        "success": result.success,
        "cost": result.cost,
        "error": result.error,
        "n_iter": result.n_iter,
    }


def run_scipy_numeric(ansatz: HEA, ham: TransverseFieldIsing, rng: np.random.Generator, max_iter: int):
    init_params = random_parameters(ansatz.n_params, rng)
    result = minimize(
        lambda x: ansatz.cost_function(x, ham.matrix),
        init_params,
        method="BFGS",
        options={"maxiter": max_iter, "gtol": 1.0e-7, "disp": False},
    )
    cost = float(result.fun)
    error = abs(cost - ham.exact_energy)
    return {
        "optimizer": "scipy_bfgs_internal_numeric",
        "success": error < 1.6e-3,
        "cost": cost,
        "error": error,
        "n_iter": int(result.nit),
        "scipy_success": bool(result.success),
        "message": str(result.message),
    }


class HEAEntanglerFirst(HEA):
    def state(self, params: np.ndarray) -> np.ndarray:
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} HEA parameters, got {len(params)}")
        state = zero_state(self.n_qubits)
        for layer in range(self.n_layers):
            for wire in range(self.n_qubits - 1):
                state = apply_cz(state, wire, wire + 1, self.n_qubits)
            state = apply_cz(state, self.n_qubits - 1, 0, self.n_qubits)
            idx = layer * self.n_qubits * 3
            for wire in range(self.n_qubits):
                state = apply_rz(state, params[idx], wire, self.n_qubits)
                state = apply_rx(state, params[idx + 1], wire, self.n_qubits)
                state = apply_rz(state, params[idx + 2], wire, self.n_qubits)
                idx += 3
        return state


class HEAOpenEntangler(HEA):
    def _apply_layer(self, state: np.ndarray, params: np.ndarray, layer: int) -> np.ndarray:
        idx = layer * self.n_qubits * 3
        for wire in range(self.n_qubits):
            state = apply_rz(state, params[idx], wire, self.n_qubits)
            state = apply_rx(state, params[idx + 1], wire, self.n_qubits)
            state = apply_rz(state, params[idx + 2], wire, self.n_qubits)
            idx += 3
        for wire in range(self.n_qubits - 1):
            state = apply_cz(state, wire, wire + 1, self.n_qubits)
        return state


def create_ansatz_variant(name: str) -> HEA:
    config = AnsatzConfig(4, 4, "HEA")
    if name == "paper_ring":
        return HEA(config)
    if name == "entangler_first_ring":
        return HEAEntanglerFirst(config)
    if name == "open_entangler":
        return HEAOpenEntangler(config)
    raise ValueError(f"Unknown ansatz variant: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="diagnostics")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    run_dir = make_run_dir(Path(args.root), "task_a_base")
    events_path = run_dir / "events.jsonl"
    trials_path = run_dir / "trials.jsonl"
    summary_path = run_dir / "summary.json"

    config = {
        "trials": args.trials,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "threshold": 1.6e-3,
    }
    write_jsonl(events_path, {"time": time.time(), "event": "start", "config": config})
    print(f"diagnostic_dir={run_dir}", flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    ansatz_variants = ("paper_ring", "entangler_first_ring", "open_entangler")
    for periodic in (True, False):
        ham = TransverseFieldIsing(4, periodic=periodic)
        for variant in ansatz_variants:
            ansatz = create_ansatz_variant(variant)
            write_jsonl(
                events_path,
                {
                    "time": time.time(),
                    "event": "case_start",
                    "periodic": periodic,
                    "ansatz_variant": variant,
                    "exact_energy": ham.exact_energy,
                    "n_params": ansatz.n_params,
                },
            )
            for optimizer_name, runner in (
                ("repo_bfgs_finite_diff", run_current_bfgs),
                ("scipy_bfgs_internal_numeric", run_scipy_numeric),
            ):
                for trial in range(1, args.trials + 1):
                    started = time.time()
                    row = runner(ansatz, ham, rng, args.max_iter)
                    row.update(
                        {
                            "periodic": periodic,
                            "ansatz_variant": variant,
                            "trial": trial,
                            "exact_energy": ham.exact_energy,
                            "time_seconds": time.time() - started,
                        }
                    )
                    rows.append(row)
                    write_jsonl(trials_path, row)
                    write_jsonl(
                        events_path,
                        {
                            "time": time.time(),
                            "event": "trial_complete",
                            "periodic": periodic,
                            "ansatz_variant": variant,
                            "optimizer": optimizer_name,
                            "trial": trial,
                            "success": row["success"],
                            "error": row["error"],
                            "n_iter": row["n_iter"],
                        },
                    )
                    print(
                        f"periodic={periodic} variant={variant} optimizer={optimizer_name} "
                        f"trial={trial} error={row['error']:.6e} success={row['success']}",
                        flush=True,
                    )

    summary = {
        "config": config,
        "best_error": min(row["error"] for row in rows) if rows else None,
        "successes": sum(1 for row in rows if row["success"]),
        "n_trials": len(rows),
        "by_case": {},
    }
    for periodic in (True, False):
        for variant in ansatz_variants:
            for optimizer in ("repo_bfgs_finite_diff", "scipy_bfgs_internal_numeric"):
                case_rows = [
                    row
                    for row in rows
                    if row["periodic"] == periodic
                    and row["ansatz_variant"] == variant
                    and row["optimizer"] == optimizer
                ]
                summary["by_case"][
                    f"periodic={periodic}|variant={variant}|optimizer={optimizer}"
                ] = {
                    "best_error": min(row["error"] for row in case_rows) if case_rows else None,
                    "successes": sum(1 for row in case_rows if row["success"]),
                    "n_trials": len(case_rows),
                }

    summary_path.write_text(json.dumps(summary, indent=2))
    write_jsonl(events_path, {"time": time.time(), "event": "complete", "summary": summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
