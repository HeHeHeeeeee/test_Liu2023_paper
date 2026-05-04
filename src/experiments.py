"""Shared experiment runner for Tasks A-F."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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


def serialize_result(result: VQAResult) -> dict:
    return {
        "success": result.success,
        "cost": result.cost,
        "error": result.error,
        "n_iter": result.n_iter,
        "init_gradient_norm": result.init_gradient_norm,
    }


def collect_base_params(
    task: TaskConfig,
    settings: ExperimentSettings,
    output_dirs: dict[str, Path],
    rng: np.random.Generator,
) -> tuple[list[list[float]], object, Ansatz]:
    base_ham = create_hamiltonian(task.base_ham_type or task.ham_type, task.base_n)
    base_ansatz = create_ansatz_for_task(
        task, task.base_n, task.base_layers, base_ham, network_base_qubits=task.base_n
    )

    successful_params: list[list[float]] = []
    trial_results: list[dict] = []
    total_trials = 0
    started = time.time()

    while len(successful_params) < settings.base_successes:
        total_trials += 1
        init_params = random_parameters(base_ansatz.n_params, rng)
        result = optimize_vqa(
            base_ansatz,
            base_ham.matrix,
            base_ham.exact_energy,
            init_params,
            max_iter=settings.max_iter_base,
            threshold=settings.success_threshold,
            compute_init_gradient=False,
        )
        trial_results.append(serialize_result(result))
        if result.success:
            successful_params.append(result.params.tolist())

    pool_data = {
        "task": asdict(task),
        "hamiltonian": base_ham.description,
        "exact_energy": base_ham.exact_energy,
        "n_successful": len(successful_params),
        "total_trials": total_trials,
        "time_seconds": time.time() - started,
        "params": successful_params,
    }
    with open(output_dirs["pools"] / f"{task.output_prefix}_pool.json", "w") as f:
        json.dump(pool_data, f, indent=2)
    with open(output_dirs["logs"] / f"{task.output_prefix}_base.jsonl", "w") as f:
        for row in trial_results:
            f.write(json.dumps(row) + "\n")

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
    trial_results: list[dict] = []
    started = time.time()

    while successes < settings.target_successes:
        total_trials += 1
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
        )
        trial_results.append(serialize_result(result))
        if result.success:
            successes += 1
            iterations.append(result.n_iter)
            gradient_norms.append(result.init_gradient_norm)

    with open(output_dirs["logs"] / f"{task.output_prefix}_{pattern}.jsonl", "w") as f:
        for row in trial_results:
            f.write(json.dumps(row) + "\n")

    success_rate = successes / total_trials
    return {
        "pattern": pattern,
        "successful_runs": successes,
        "total_trials": total_trials,
        "success_rate": success_rate,
        "TTN": total_trials,
        "avg_iterations": float(np.mean(iterations)) if iterations else 0.0,
        "std_iterations": float(np.std(iterations)) if iterations else 0.0,
        "avg_gradient_norm": float(np.mean(gradient_norms)) if gradient_norms else 0.0,
        "std_gradient_norm": float(np.std(gradient_norms)) if gradient_norms else 0.0,
        "time_seconds": time.time() - started,
    }


def run_task(
    task: TaskConfig,
    output_root: Path,
    settings: ExperimentSettings | None = None,
) -> dict:
    settings = settings or ExperimentSettings()
    rng = np.random.default_rng(settings.seed)
    output_dirs = create_output_dirs(output_root)

    base_params, _base_ham, base_ansatz = collect_base_params(
        task, settings, output_dirs, rng
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
            rng,
        )

    with open(output_dirs["metrics"] / f"{task.output_prefix}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results
