"""Shared experiment runner for Tasks A-F."""

from __future__ import annotations

import json
import hashlib
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

for thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(thread_env, "1")

import numpy as np

from .ansatz import Ansatz, AnsatzConfig, HEA, HVA, NetworkTransferHEA
from .hamiltonians import HeisenbergXXZ, HydrogenMolecule, TransverseFieldIsing
from .optimizer import VQAResult, optimize_vqa
from .transfer import NetworkTransfer, StructureTransfer, generate_ble_params, random_parameters


@dataclass
class TaskConfig:
    name: str
    base_n: int
    target_n: int
    base_layers: int
    target_layers: int
    ansatz: str
    transfer: str
    ham_type: str
    patterns: list[str]
    base_ham_type: str | None = None
    use_mod_hva: bool = False
    include_ble: bool = False

    @property
    def output_prefix(self) -> str:
        return f"{self.name}_{self.base_n}to{self.target_n}"


TASKS: dict[str, TaskConfig] = {
    "task_a": TaskConfig(
        "task_a", 4, 6, 4, 4, "HEA", "network", "ising",
        ["TTT", "RRT", "TTR", "RRR"],
    ),
    "task_b": TaskConfig(
        "task_b", 4, 8, 4, 4, "HEA", "network", "ising",
        ["TTTTT", "TRTRT", "RTRTR", "RRRRR"],
    ),
    "task_c": TaskConfig(
        "task_c", 4, 6, 4, 8, "HEA", "structure", "h2",
        ["TT", "TR", "RT", "RR"],
    ),
    "task_d": TaskConfig(
        "task_d", 4, 6, 4, 8, "HVA", "structure", "xxz1d",
        ["TT", "TR", "RT", "RR"],
    ),
    "task_e": TaskConfig(
        "task_e", 4, 8, 4, 8, "HEA", "structure", "xxz2d",
        ["TTTT", "TRRT", "RTTR", "RRRR"],
        base_ham_type="xxz1d",
    ),
    "task_f": TaskConfig(
        "task_f", 4, 8, 4, 8, "HVA", "structure", "xxz1d",
        ["TT", "TR", "RT", "RR", "BLE"],
        use_mod_hva=True,
        include_ble=True,
    ),
}


@dataclass
class ExperimentSettings:
    success_threshold: float = 1.6e-3
    base_successes: int = 100
    target_successes: int = 100
    max_iter_base: int = 500
    max_iter_target: int = 500
    max_trials_base: int = 0
    max_trials_target: int = 0
    workers: int = max(1, (os.cpu_count() or 2) - 2)
    record_history: bool = True
    seed: int | None = None


def create_output_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "logs": root / "logs",
        "metrics": root / "metrics",
        "pools": root / "pools",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def child_rng(base_seed: int | None, *parts: object) -> np.random.Generator:
    """Create an isolated RNG stream for one reproducible experiment unit."""

    if base_seed is None:
        return np.random.default_rng()
    text = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "little") & 0xFFFFFFFF
    return np.random.default_rng(seed)


def create_hamiltonian(ham_type: str, n_qubits: int):
    if ham_type == "ising":
        return TransverseFieldIsing(n_qubits, j_ising=1.0, h=2.0, periodic=True)
    if ham_type == "h2":
        return HydrogenMolecule(n_atoms=n_qubits // 2, bond_length=0.74)
    if ham_type == "xxz1d":
        return HeisenbergXXZ(
            n_qubits, j_xxz=1.0, delta=2.0, geometry="1d", periodic=True
        )
    if ham_type == "xxz2d":
        return HeisenbergXXZ(
            n_qubits, j_xxz=1.0, delta=2.0, geometry="2d", rows=2, cols=n_qubits // 2
        )
    raise ValueError(f"Unknown Hamiltonian type: {ham_type}")


def create_ansatz_for_task(
    task: TaskConfig,
    n_qubits: int,
    n_layers: int,
    hamiltonian,
    network_base_qubits: int | None = None,
) -> Ansatz:
    if task.transfer == "network":
        return NetworkTransferHEA(
            m_qubits=n_qubits,
            n_qubits=network_base_qubits or n_qubits,
            layers=task.base_layers,
        )

    if task.ansatz == "HEA":
        return HEA(AnsatzConfig(n_qubits=n_qubits, n_layers=n_layers, ansatz_type="HEA"))

    if task.ansatz == "HVA":
        return HVA(
            AnsatzConfig(
                n_qubits=n_qubits,
                n_layers=n_layers,
                ansatz_type="HVA",
                hamiltonian_terms=hamiltonian.hva_terms(),
                use_mod_hva=task.use_mod_hva,
            )
        )

    raise ValueError(f"Unknown ansatz: {task.ansatz}")


def serialize_result(result: VQAResult, include_history: bool = False) -> dict:
    row = {
        "success": result.success,
        "cost": result.cost,
        "error": result.error,
        "n_iter": result.n_iter,
        "init_gradient_norm": result.init_gradient_norm,
    }
    if include_history:
        row["history"] = result.history
    return row


def write_jsonl_row(handle, row: dict) -> None:
    handle.write(json.dumps(row) + "\n")
    handle.flush()


def append_event(output_dirs: dict[str, Path], event: dict) -> None:
    row = {"time": time.time(), **event}
    with open(output_dirs["logs"] / "run_events.jsonl", "a") as handle:
        write_jsonl_row(handle, row)


def load_cached_pool(
    task: TaskConfig,
    settings: ExperimentSettings,
    output_dirs: dict[str, Path],
) -> tuple[list[list[float]], Any, Ansatz] | None:
    pool_path = output_dirs["pools"] / f"{task.output_prefix}_pool.json"
    if not pool_path.exists():
        return None

    try:
        with open(pool_path) as f:
            pool_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    params = pool_data.get("params")
    if not isinstance(params, list) or len(params) < settings.base_successes:
        return None
    if pool_data.get("task") != asdict(task):
        return None
    expected_settings = {
        "success_threshold": settings.success_threshold,
        "base_successes": settings.base_successes,
        "max_iter_base": settings.max_iter_base,
        "seed": settings.seed,
    }
    if pool_data.get("settings") != expected_settings:
        return None

    base_ham = create_hamiltonian(task.base_ham_type or task.ham_type, task.base_n)
    base_ansatz = create_ansatz_for_task(
        task, task.base_n, task.base_layers, base_ham, network_base_qubits=task.base_n
    )
    return params[: settings.base_successes], base_ham, base_ansatz


def collect_base_params(
    task: TaskConfig,
    settings: ExperimentSettings,
    output_dirs: dict[str, Path],
    rng: np.random.Generator,
) -> tuple[list[list[float]], object, Ansatz]:
    cached = load_cached_pool(task, settings, output_dirs)
    if cached is not None:
        append_event(
            output_dirs,
            {
                "event": "base_pool_cache_hit",
                "task": task.name,
                "phase": "base",
                "target_successes": settings.base_successes,
            },
        )
        return cached

    base_ham = create_hamiltonian(task.base_ham_type or task.ham_type, task.base_n)
    base_ansatz = create_ansatz_for_task(
        task, task.base_n, task.base_layers, base_ham, network_base_qubits=task.base_n
    )

    successful_params: list[list[float]] = []
    trial_results: list[dict] = []
    total_trials = 0
    started = time.time()

    log_path = output_dirs["logs"] / f"{task.output_prefix}_base.jsonl"
    iter_path = output_dirs["logs"] / f"{task.output_prefix}_base_iterations.jsonl"
    workers = max(1, settings.workers)
    next_trial = 1
    futures = set()
    append_event(
        output_dirs,
        {
            "event": "base_start",
            "task": task.name,
            "phase": "base",
            "workers": workers,
            "target_successes": settings.base_successes,
            "max_trials": settings.max_trials_base,
        },
    )

    with open(log_path, "w") as log_file, open(iter_path, "w") as iter_file:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            while len(successful_params) < settings.base_successes:
                while (
                    len(futures) < workers
                    and (
                        settings.max_trials_base <= 0
                        or next_trial <= settings.max_trials_base
                    )
                ):
                    futures.add(
                        executor.submit(
                            run_base_trial_worker,
                            task,
                            settings,
                            next_trial,
                            base_ham,
                            base_ansatz,
                        )
                    )
                    next_trial += 1

                if not futures:
                    break

                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        payload = future.result()
                    except Exception as exc:
                        append_event(
                            output_dirs,
                            {
                                "event": "base_trial_error",
                                "task": task.name,
                                "phase": "base",
                                "error": str(exc),
                            },
                        )
                        raise
                    total_trials += 1
                    result_row = payload["result"]
                    result_row["trial"] = payload["trial_index"]
                    trial_results.append(result_row)
                    write_jsonl_row(log_file, result_row)
                    for row in payload["history"]:
                        write_jsonl_row(iter_file, row)
                    if payload["params"] is not None:
                        successful_params.append(payload["params"])
                    print(
                        f"[{task.name} base] trials={total_trials} "
                        f"success={len(successful_params)}/{settings.base_successes}",
                        flush=True,
                    )
                    append_event(
                        output_dirs,
                        {
                            "event": "base_trial_complete",
                            "task": task.name,
                            "phase": "base",
                            "trial": payload["trial_index"],
                            "success": result_row["success"],
                            "cost": result_row["cost"],
                            "error": result_row["error"],
                            "n_iter": result_row["n_iter"],
                            "total_trials": total_trials,
                            "successful_runs": len(successful_params),
                            "target_successes": settings.base_successes,
                        },
                    )

                    if len(successful_params) >= settings.base_successes:
                        for pending in futures:
                            pending.cancel()
                        futures.clear()
                        break

    if len(successful_params) < settings.base_successes:
        append_event(
            output_dirs,
            {
                "event": "base_failed",
                "task": task.name,
                "phase": "base",
                "total_trials": total_trials,
                "successful_runs": len(successful_params),
                "target_successes": settings.base_successes,
            },
        )
        raise RuntimeError(
            f"{task.name} base pool reached max_trials_base={settings.max_trials_base} "
            f"with {len(successful_params)}/{settings.base_successes} successful runs"
        )

    pool_data = {
        "task": asdict(task),
        "settings": {
            "success_threshold": settings.success_threshold,
            "base_successes": settings.base_successes,
            "max_iter_base": settings.max_iter_base,
            "seed": settings.seed,
        },
        "hamiltonian": base_ham.description,
        "exact_energy": base_ham.exact_energy,
        "n_successful": len(successful_params),
        "total_trials": total_trials,
        "time_seconds": time.time() - started,
        "params": successful_params,
    }
    with open(output_dirs["pools"] / f"{task.output_prefix}_pool.json", "w") as f:
        json.dump(pool_data, f, indent=2)
    append_event(
        output_dirs,
        {
            "event": "base_complete",
            "task": task.name,
            "phase": "base",
            "total_trials": total_trials,
            "successful_runs": len(successful_params),
            "time_seconds": pool_data["time_seconds"],
        },
    )

    return successful_params, base_ham, base_ansatz


def generate_initial_params(
    task: TaskConfig,
    pattern: str,
    base_ansatz: Ansatz,
    target_ansatz: Ansatz,
    base_params: list[list[float]],
    rng: np.random.Generator,
) -> np.ndarray:
    if pattern == "BLE":
        if not isinstance(target_ansatz, HVA):
            raise ValueError("BLE is only defined for HVA")
        return generate_ble_params(target_ansatz, rng)

    base_p = np.asarray(base_params[rng.integers(len(base_params))], dtype=float)

    if task.transfer == "network":
        transfer = NetworkTransfer(base_ansatz, task.target_n)
        return transfer.generate_params(base_p, pattern, rng)

    transfer = StructureTransfer(base_ansatz, target_ansatz)
    return transfer.generate_params(base_p, pattern, rng)


def history_rows(
    task: TaskConfig,
    phase: str,
    pattern: str | None,
    trial_index: int,
    result: VQAResult,
) -> list[dict]:
    rows = []
    for item in result.history:
        row = {
            "task": task.name,
            "phase": phase,
            "pattern": pattern,
            "trial": trial_index,
            "success": result.success,
            **item,
        }
        rows.append(row)
    return rows


def run_base_trial_worker(
    task: TaskConfig,
    settings: ExperimentSettings,
    trial_index: int,
    base_ham,
    base_ansatz: Ansatz,
) -> dict:
    rng = child_rng(settings.seed, task.name, "base", trial_index)
    init_params = random_parameters(base_ansatz.n_params, rng)
    result = optimize_vqa(
        base_ansatz,
        base_ham.matrix,
        base_ham.exact_energy,
        init_params,
        max_iter=settings.max_iter_base,
        threshold=settings.success_threshold,
        compute_init_gradient=False,
        record_history=settings.record_history,
    )
    return {
        "trial_index": trial_index,
        "result": serialize_result(result),
        "params": result.params.tolist() if result.success else None,
        "history": history_rows(task, "base", None, trial_index, result),
    }


def run_target_trial_worker(
    task: TaskConfig,
    pattern: str,
    settings: ExperimentSettings,
    base_params: list[list[float]],
    trial_index: int,
    base_ansatz: Ansatz,
    target_ham,
    target_ansatz: Ansatz,
) -> dict:
    rng = child_rng(settings.seed, task.name, pattern, trial_index)
    init_params = generate_initial_params(
        task, pattern, base_ansatz, target_ansatz, base_params, rng
    )
    result = optimize_vqa(
        target_ansatz,
        target_ham.matrix,
        target_ham.exact_energy,
        init_params,
        max_iter=settings.max_iter_target,
        threshold=settings.success_threshold,
        compute_init_gradient=True,
        record_history=settings.record_history,
    )
    return {
        "trial_index": trial_index,
        "result": serialize_result(result),
        "history": history_rows(task, "target", pattern, trial_index, result),
    }


def run_pattern_experiment(
    task: TaskConfig,
    pattern: str,
    settings: ExperimentSettings,
    output_dirs: dict[str, Path],
    base_ansatz: Ansatz,
    base_params: list[list[float]],
    target_ham,
    target_ansatz: Ansatz,
    rng: np.random.Generator,
) -> dict:
    successes = 0
    total_trials = 0
    iterations: list[int] = []
    gradient_norms: list[float] = []
    successful_gradient_norms: list[float] = []
    trial_results: list[dict] = []
    started = time.time()

    log_path = output_dirs["logs"] / f"{task.output_prefix}_{pattern}.jsonl"
    iter_path = output_dirs["logs"] / f"{task.output_prefix}_{pattern}_iterations.jsonl"
    workers = max(1, settings.workers)
    next_trial = 1
    futures = set()
    append_event(
        output_dirs,
        {
            "event": "pattern_start",
            "task": task.name,
            "phase": "target",
            "pattern": pattern,
            "workers": workers,
            "target_successes": settings.target_successes,
            "max_trials": settings.max_trials_target,
        },
    )

    with open(log_path, "w") as log_file, open(iter_path, "w") as iter_file:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            while successes < settings.target_successes:
                while (
                    len(futures) < workers
                    and (
                        settings.max_trials_target <= 0
                        or next_trial <= settings.max_trials_target
                    )
                ):
                    futures.add(
                        executor.submit(
                            run_target_trial_worker,
                            task,
                            pattern,
                            settings,
                            base_params,
                            next_trial,
                            base_ansatz,
                            target_ham,
                            target_ansatz,
                        )
                    )
                    next_trial += 1

                if not futures:
                    break

                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        payload = future.result()
                    except Exception as exc:
                        append_event(
                            output_dirs,
                            {
                                "event": "pattern_trial_error",
                                "task": task.name,
                                "phase": "target",
                                "pattern": pattern,
                                "error": str(exc),
                            },
                        )
                        raise
                    total_trials += 1
                    result_row = payload["result"]
                    result_row["trial"] = payload["trial_index"]
                    trial_results.append(result_row)
                    write_jsonl_row(log_file, result_row)
                    for row in payload["history"]:
                        write_jsonl_row(iter_file, row)

                    gradient_norms.append(result_row["init_gradient_norm"])
                    if result_row["success"]:
                        successes += 1
                        iterations.append(result_row["n_iter"])
                        successful_gradient_norms.append(result_row["init_gradient_norm"])

                    print(
                        f"[{task.name} {pattern}] trials={total_trials} "
                        f"success={successes}/{settings.target_successes}",
                        flush=True,
                    )
                    append_event(
                        output_dirs,
                        {
                            "event": "pattern_trial_complete",
                            "task": task.name,
                            "phase": "target",
                            "pattern": pattern,
                            "trial": payload["trial_index"],
                            "success": result_row["success"],
                            "cost": result_row["cost"],
                            "error": result_row["error"],
                            "n_iter": result_row["n_iter"],
                            "init_gradient_norm": result_row["init_gradient_norm"],
                            "total_trials": total_trials,
                            "successful_runs": successes,
                            "target_successes": settings.target_successes,
                        },
                    )

                    if successes >= settings.target_successes:
                        for pending in futures:
                            pending.cancel()
                        futures.clear()
                        break

    success_rate = successes / total_trials if total_trials else 0.0
    summary = {
        "pattern": pattern,
        "successful_runs": successes,
        "total_trials": total_trials,
        "success_rate": success_rate,
        "TTN": total_trials,
        "reached_target_successes": successes >= settings.target_successes,
        "avg_iterations": float(np.mean(iterations)) if iterations else 0.0,
        "std_iterations": float(np.std(iterations)) if iterations else 0.0,
        "avg_gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
        "std_gradient_norm": float(np.std(gradient_norms)) if gradient_norms else 0.0,
        "avg_successful_gradient_norm": (
            float(np.mean(successful_gradient_norms)) if successful_gradient_norms else 0.0
        ),
        "std_successful_gradient_norm": (
            float(np.std(successful_gradient_norms)) if successful_gradient_norms else 0.0
        ),
        "time_seconds": time.time() - started,
    }
    append_event(
        output_dirs,
        {
            "event": "pattern_complete",
            "task": task.name,
            "phase": "target",
            "pattern": pattern,
            "total_trials": total_trials,
            "successful_runs": successes,
            "target_successes": settings.target_successes,
            "reached_target_successes": summary["reached_target_successes"],
            "time_seconds": summary["time_seconds"],
        },
    )
    return summary


def run_task(
    task: TaskConfig,
    output_root: Path,
    settings: ExperimentSettings | None = None,
) -> dict:
    settings = settings or ExperimentSettings()
    output_dirs = create_output_dirs(output_root)
    append_event(
        output_dirs,
        {
            "event": "task_start",
            "task": task.name,
            "workers": settings.workers,
            "base_successes": settings.base_successes,
            "target_successes": settings.target_successes,
            "max_iter_base": settings.max_iter_base,
            "max_iter_target": settings.max_iter_target,
            "max_trials_base": settings.max_trials_base,
            "max_trials_target": settings.max_trials_target,
            "record_history": settings.record_history,
            "seed": settings.seed,
        },
    )

    base_params, _base_ham, base_ansatz = collect_base_params(
        task, settings, output_dirs, child_rng(settings.seed, task.name, "base")
    )
    target_ham = create_hamiltonian(task.ham_type, task.target_n)
    target_ansatz = create_ansatz_for_task(
        task,
        task.target_n,
        task.target_layers,
        target_ham,
        network_base_qubits=task.base_n,
    )

    results = {}
    for pattern in task.patterns:
        results[pattern] = run_pattern_experiment(
            task,
            pattern,
            settings,
            output_dirs,
            base_ansatz,
            base_params,
            target_ham,
            target_ansatz,
            child_rng(settings.seed, task.name, pattern),
        )

    with open(output_dirs["metrics"] / f"{task.output_prefix}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    append_event(
        output_dirs,
        {
            "event": "task_complete",
            "task": task.name,
            "patterns": list(results.keys()),
        },
    )
    return results
