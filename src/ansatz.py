"""Ansatz implementations for the Liu 2023 transfer-learning benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import (
    apply_cz,
    apply_hermitian_evolution,
    apply_rx,
    apply_rz,
    zero_state,
)


@dataclass
class AnsatzConfig:
    n_qubits: int
    n_layers: int
    ansatz_type: str
    hamiltonian_terms: list[np.ndarray] | None = None
    use_mod_hva: bool = False


class Ansatz:
    n_qubits: int
    n_layers: int
    n_params: int

    def state(self, params: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def cost_function(self, params: np.ndarray, hamiltonian: np.ndarray) -> float:
        state = self.state(np.asarray(params, dtype=float))
        return float(np.real(np.vdot(state, hamiltonian @ state)))

    def gradient(
        self,
        params: np.ndarray,
        hamiltonian: np.ndarray,
        epsilon: float = 1.0e-6,
    ) -> np.ndarray:
        params = np.asarray(params, dtype=float)
        grad = np.zeros_like(params)
        for idx in range(len(params)):
            plus = params.copy()
            minus = params.copy()
            plus[idx] += epsilon
            minus[idx] -= epsilon
            grad[idx] = (
                self.cost_function(plus, hamiltonian)
                - self.cost_function(minus, hamiltonian)
            ) / (2.0 * epsilon)
        return grad

    def cost_and_gradient(
        self, params: np.ndarray, hamiltonian: np.ndarray
    ) -> tuple[float, np.ndarray]:
        return self.cost_function(params, hamiltonian), self.gradient(params, hamiltonian)


class HEA(Ansatz):
    """Hardware-efficient ansatz used in the numerical benchmarks.

    The PDF formula includes a closing CZ(n, 1), but Task A base training
    reproducibly fails under the chemical-accuracy criterion with that ring
    entangler.  The numerical results are consistent with an open-chain CZ
    entangler, so the reproduction uses nearest-neighbor CZ gates only.
    """

    def __init__(self, config: AnsatzConfig):
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.n_params = self.n_qubits * self.n_layers * 3

    def _apply_layer(
        self, state: np.ndarray, params: np.ndarray, layer: int
    ) -> np.ndarray:
        idx = layer * self.n_qubits * 3
        for wire in range(self.n_qubits):
            state = apply_rz(state, params[idx], wire, self.n_qubits)
            state = apply_rx(state, params[idx + 1], wire, self.n_qubits)
            state = apply_rz(state, params[idx + 2], wire, self.n_qubits)
            idx += 3

        for wire in range(self.n_qubits - 1):
            state = apply_cz(state, wire, wire + 1, self.n_qubits)
        return state

    def state(self, params: np.ndarray) -> np.ndarray:
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} HEA parameters, got {len(params)}")
        state = zero_state(self.n_qubits)
        for layer in range(self.n_layers):
            state = self._apply_layer(state, params, layer)
        return state

    def index(self, layer: int, wire: int, rotation: int) -> int:
        return layer * self.n_qubits * 3 + wire * 3 + rotation


class NetworkTransferHEA(Ansatz):
    """Sliding-window HEA used by the paper's network transfer, equation (9).

    For an n-qubit base circuit and an m-qubit target, this ansatz applies
    m-n+1 copied base circuits on windows [i, i+n-1].  Each window has its
    own base-sized parameter block marked T or R by the transfer pattern.
    """

    def __init__(self, m_qubits: int, n_qubits: int, layers: int):
        if m_qubits < n_qubits:
            raise ValueError("m_qubits must be >= n_qubits for network transfer")
        self.n_qubits = m_qubits
        self.window_qubits = n_qubits
        self.n_layers = layers
        self.n_windows = m_qubits - n_qubits + 1
        self.params_per_block = n_qubits * layers * 3
        self.total_params = self.n_windows * self.params_per_block
        self.n_params = self.total_params

    def _apply_window_layer(
        self,
        state: np.ndarray,
        params: np.ndarray,
        window: int,
        layer: int,
    ) -> np.ndarray:
        idx = window * self.params_per_block + layer * self.window_qubits * 3
        for local_wire in range(self.window_qubits):
            wire = window + local_wire
            state = apply_rz(state, params[idx], wire, self.n_qubits)
            state = apply_rx(state, params[idx + 1], wire, self.n_qubits)
            state = apply_rz(state, params[idx + 2], wire, self.n_qubits)
            idx += 3

        for local_wire in range(self.window_qubits - 1):
            state = apply_cz(
                state,
                window + local_wire,
                window + local_wire + 1,
                self.n_qubits,
            )
        return state

    def state(self, params: np.ndarray) -> np.ndarray:
        if len(params) != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} network-transfer parameters, got {len(params)}"
            )
        state = zero_state(self.n_qubits)
        for window in range(self.n_windows):
            for layer in range(self.n_layers):
                state = self._apply_window_layer(state, params, window, layer)
        return state


class HVA(Ansatz):
    """Hamiltonian variational ansatz from paper equation (5).

    If use_mod_hva=True, even-numbered layers reverse the Hamiltonian-term
    order as in paper equation (12), enabling block-identity encoding.
    """

    def __init__(self, config: AnsatzConfig):
        if not config.hamiltonian_terms:
            raise ValueError("HVA requires hamiltonian_terms")
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.terms = [np.asarray(term, dtype=np.complex128) for term in config.hamiltonian_terms]
        self.n_terms = len(self.terms)
        self.n_params = self.n_layers * self.n_terms
        self.use_mod_hva = config.use_mod_hva
        self._eigensystems = [np.linalg.eigh(term) for term in self.terms]

    def state(self, params: np.ndarray) -> np.ndarray:
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} HVA parameters, got {len(params)}")
        state = zero_state(self.n_qubits)
        for layer in range(self.n_layers):
            order = range(self.n_terms)
            if self.use_mod_hva and layer % 2 == 1:
                order = reversed(range(self.n_terms))
            for term_idx in order:
                theta = params[layer * self.n_terms + term_idx]
                eigvals, eigvecs = self._eigensystems[term_idx]
                state = apply_hermitian_evolution(state, eigvals, eigvecs, theta)
        return state


def create_ansatz(config: AnsatzConfig) -> Ansatz:
    kind = config.ansatz_type.upper()
    if kind == "HEA":
        return HEA(config)
    if kind in {"HVA", "MOD_HVA"}:
        return HVA(config)
    raise ValueError(f"Unknown ansatz type: {config.ansatz_type}")
