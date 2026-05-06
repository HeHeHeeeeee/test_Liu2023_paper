"""Classical optimization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ansatz import Ansatz


@dataclass
class VQAResult:
    success: bool
    cost: float
    error: float
    params: np.ndarray
    n_iter: int
    init_gradient_norm: float
    history: list[dict[str, Any]]


def normalized_gradient_norm(gradient: np.ndarray) -> float:
    return float(np.mean(np.asarray(gradient, dtype=float) ** 2))


def optimize_vqa(
    ansatz: Ansatz,
    hamiltonian: np.ndarray,
    exact_energy: float,
    init_params: np.ndarray,
    max_iter: int = 500,
    threshold: float = 1.6e-3,
    compute_init_gradient: bool = True,
    record_history: bool = True,
) -> VQAResult:
    """Run the paper's BFGS-style VQA optimization from one initialization."""

    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required for BFGS optimization. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    init_params = np.asarray(init_params, dtype=float)
    init_g = 0.0
    if compute_init_gradient:
        init_g = normalized_gradient_norm(ansatz.gradient(init_params, hamiltonian))

    history: list[dict[str, Any]] = []
    if record_history:
        init_cost = ansatz.cost_function(init_params, hamiltonian)
        history.append(
            {
                "iteration": 0,
                "cost": init_cost,
                "error": abs(init_cost - exact_energy),
                "params": init_params.tolist(),
            }
        )

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        return ansatz.cost_and_gradient(x, hamiltonian)

    def callback(xk: np.ndarray) -> None:
        if not record_history:
            return
        cost = ansatz.cost_function(xk, hamiltonian)
        history.append(
            {
                "iteration": len(history),
                "cost": cost,
                "error": abs(cost - exact_energy),
                "params": np.asarray(xk, dtype=float).tolist(),
            }
        )

    result = minimize(
        fun=objective,
        x0=init_params,
        method="BFGS",
        jac=True,
        callback=callback,
        options={"maxiter": max_iter, "gtol": 1.0e-7, "disp": False},
    )

    cost = float(result.fun)
    error = abs(cost - exact_energy)
    return VQAResult(
        success=error < threshold,
        cost=cost,
        error=error,
        params=np.asarray(result.x, dtype=float),
        n_iter=int(result.nit),
        init_gradient_norm=init_g,
        history=history,
    )


def optimize_vqa_autograd(
    ansatz: Ansatz,
    hamiltonian: np.ndarray,
    init_params_np: np.ndarray,
    max_iter: int = 500,
    verbose: bool = False,
    worker_id: str = "W0",
) -> tuple[bool, float, np.ndarray, int]:
    """Backward-compatible wrapper for older task scripts.

    The project no longer uses PyTorch autograd; the name is kept so stale
    imports fail less dramatically while callers are migrated.
    """

    exact_energy = float(np.linalg.eigvalsh(hamiltonian).min())
    result = optimize_vqa(
        ansatz,
        hamiltonian,
        exact_energy,
        init_params_np,
        max_iter=max_iter,
        compute_init_gradient=False,
    )
    if verbose:
        print(
            f"[{worker_id}] BFGS cost={result.cost:.10f} "
            f"error={result.error:.3e} iters={result.n_iter}"
        )
    return result.success, result.cost, result.params, result.n_iter
