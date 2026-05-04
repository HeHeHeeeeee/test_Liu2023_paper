"""Small dense-state simulation utilities for the Liu 2023 experiments."""

from __future__ import annotations

import numpy as np


COMPLEX = np.complex128
I2 = np.eye(2, dtype=COMPLEX)
X = np.array([[0, 1], [1, 0]], dtype=COMPLEX)
Y = np.array([[0, -1j], [1j, 0]], dtype=COMPLEX)
Z = np.array([[1, 0], [0, -1]], dtype=COMPLEX)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def zero_state(n_qubits: int) -> np.ndarray:
    state = np.zeros(2**n_qubits, dtype=COMPLEX)
    state[0] = 1.0
    return state


def kron_all(ops: list[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0]], dtype=COMPLEX)
    for op in ops:
        out = np.kron(out, op)
    return out


def apply_single_qubit_gate(
    state: np.ndarray, gate: np.ndarray, wire: int, n_qubits: int
) -> np.ndarray:
    tensor = state.reshape((2,) * n_qubits)
    tensor = np.moveaxis(tensor, wire, 0)
    tensor = np.tensordot(gate, tensor, axes=([1], [0]))
    tensor = np.moveaxis(tensor, 0, wire)
    return tensor.reshape(2**n_qubits)


def rz(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array(
        [[np.exp(-1j * half), 0.0], [0.0, np.exp(1j * half)]],
        dtype=COMPLEX,
    )


def rx(theta: float) -> np.ndarray:
    half = theta / 2.0
    c = np.cos(half)
    s = np.sin(half)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=COMPLEX)


def apply_rz(state: np.ndarray, theta: float, wire: int, n_qubits: int) -> np.ndarray:
    return apply_single_qubit_gate(state, rz(theta), wire, n_qubits)


def apply_rx(state: np.ndarray, theta: float, wire: int, n_qubits: int) -> np.ndarray:
    return apply_single_qubit_gate(state, rx(theta), wire, n_qubits)


def apply_cz(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    out = state.copy()
    indices = np.arange(2**n_qubits)
    c_mask = (indices >> (n_qubits - 1 - control)) & 1
    t_mask = (indices >> (n_qubits - 1 - target)) & 1
    out[(c_mask == 1) & (t_mask == 1)] *= -1.0
    return out


def build_cz_matrix(n_qubits: int, control: int, target: int) -> np.ndarray:
    matrix = np.eye(2**n_qubits, dtype=COMPLEX)
    indices = np.arange(2**n_qubits)
    c_mask = (indices >> (n_qubits - 1 - control)) & 1
    t_mask = (indices >> (n_qubits - 1 - target)) & 1
    matrix[(c_mask == 1) & (t_mask == 1), (c_mask == 1) & (t_mask == 1)] = -1.0
    return matrix


def pauli_string_matrix(paulis: str) -> np.ndarray:
    return kron_all([PAULI[p] for p in paulis])


def one_qubit_pauli(n_qubits: int, wire: int, pauli: str) -> np.ndarray:
    ops = [I2] * n_qubits
    ops[wire] = PAULI[pauli]
    return kron_all(ops)


def two_qubit_pauli(
    n_qubits: int, wire_a: int, pauli_a: str, wire_b: int, pauli_b: str
) -> np.ndarray:
    ops = [I2] * n_qubits
    ops[wire_a] = PAULI[pauli_a]
    ops[wire_b] = PAULI[pauli_b]
    return kron_all(ops)


def apply_hermitian_evolution(
    state: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray, theta: float
) -> np.ndarray:
    coeffs = eigvecs.conj().T @ state
    coeffs *= np.exp(-1j * theta * eigvals)
    return eigvecs @ coeffs


def exact_ground_energy(matrix: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(matrix).min())
