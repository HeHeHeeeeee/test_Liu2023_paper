"""Transfer-learning-inspired parameter initializations from Liu 2023."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ansatz import Ansatz, HEA, HVA, AnsatzConfig, NetworkTransferHEA


@dataclass
class TransferConfig:
    method: str
    pattern: str
    base_n_qubits: int
    target_n_qubits: int
    base_n_layers: int
    target_n_layers: int


def random_parameters(n_params: int, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.uniform(-np.pi, np.pi, n_params)


class NetworkTransfer:
    """Paper equation (9): sliding-window copies of a trained base HEA."""

    def __init__(self, base_ansatz: HEA | NetworkTransferHEA, target_n_qubits: int):
        self.base_ansatz = base_ansatz
        self.base_n_qubits = getattr(base_ansatz, "window_qubits", base_ansatz.n_qubits)
        self.base_n_layers = base_ansatz.n_layers
        self.target_n_qubits = target_n_qubits
        self.n_windows = target_n_qubits - self.base_n_qubits + 1
        if self.n_windows <= 0:
            raise ValueError("target_n_qubits must be >= base_n_qubits")
        self.params_per_block = self.base_n_qubits * self.base_n_layers * 3

    def generate_params(
        self,
        base_params: np.ndarray,
        pattern: str,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        pattern = pattern.upper()
        if len(pattern) != self.n_windows:
            raise ValueError(
                f"Network-transfer pattern length must be {self.n_windows}, got {len(pattern)}"
            )
        if len(base_params) != self.params_per_block:
            raise ValueError(
                f"Expected base parameter block of length {self.params_per_block}, "
                f"got {len(base_params)}"
            )

        target = np.zeros(self.n_windows * self.params_per_block)
        for window, char in enumerate(pattern):
            start = window * self.params_per_block
            end = start + self.params_per_block
            if char == "T":
                target[start:end] = base_params
            elif char == "R":
                target[start:end] = random_parameters(self.params_per_block, rng)
            else:
                raise ValueError(f"Unknown transfer marker {char!r}")
        return target

    def create_target_ansatz(self) -> NetworkTransferHEA:
        return NetworkTransferHEA(
            m_qubits=self.target_n_qubits,
            n_qubits=self.base_n_qubits,
            layers=self.base_n_layers,
        )


class StructureTransfer:
    """Structure-preserving transfer for HEA/HVA target circuits."""

    def __init__(
        self,
        base_ansatz: Ansatz,
        target_ansatz: Ansatz,
        qubit_offsets: list[int] | None = None,
    ):
        self.base_ansatz = base_ansatz
        self.target_ansatz = target_ansatz
        self.base_n_qubits = base_ansatz.n_qubits
        self.base_n_layers = base_ansatz.n_layers
        self.target_n_qubits = target_ansatz.n_qubits
        self.target_n_layers = target_ansatz.n_layers
        self.qubit_offsets = qubit_offsets

    def generate_params(
        self,
        base_params: np.ndarray,
        pattern: str,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        target = random_parameters(self.target_ansatz.n_params, rng)

        if isinstance(self.target_ansatz, HEA):
            self._copy_hea_blocks(target, base_params, pattern.upper())
        elif isinstance(self.target_ansatz, HVA):
            self._copy_hva_blocks(target, base_params, pattern.upper())
        else:
            raise ValueError(f"Unsupported target ansatz: {type(self.target_ansatz)}")
        return target

    def _default_qubit_offsets(self) -> list[int]:
        if self.qubit_offsets is not None:
            return self.qubit_offsets
        if self.target_n_qubits == self.base_n_qubits:
            return [0]
        if self.target_n_qubits % self.base_n_qubits == 0:
            return list(range(0, self.target_n_qubits, self.base_n_qubits))
        return [0]

    def _copy_hea_blocks(
        self, target: np.ndarray, base_params: np.ndarray, pattern: str
    ) -> None:
        if not isinstance(self.base_ansatz, HEA):
            raise ValueError("HEA structure transfer requires a HEA base ansatz")

        if self.target_n_layers % self.base_n_layers != 0:
            raise ValueError("target layers must be a multiple of base layers for HEA transfer")

        layer_groups = self.target_n_layers // self.base_n_layers
        qubit_offsets = self._default_qubit_offsets()
        expected = layer_groups * len(qubit_offsets)
        if len(pattern) != expected:
            raise ValueError(f"HEA structure pattern length must be {expected}, got {len(pattern)}")

        block_idx = 0
        for layer_group in range(layer_groups):
            target_layer_start = layer_group * self.base_n_layers
            for qubit_offset in qubit_offsets:
                marker = pattern[block_idx]
                block_idx += 1
                if marker == "R":
                    continue
                if marker != "T":
                    raise ValueError(f"Unknown transfer marker {marker!r}")

                for layer in range(self.base_n_layers):
                    for q in range(self.base_n_qubits):
                        if qubit_offset + q >= self.target_n_qubits:
                            continue
                        for rot in range(3):
                            src = self.base_ansatz.index(layer, q, rot)
                            dst = self.target_ansatz.index(
                                target_layer_start + layer,
                                qubit_offset + q,
                                rot,
                            )
                            target[dst] = base_params[src]

    def _copy_hva_blocks(
        self, target: np.ndarray, base_params: np.ndarray, pattern: str
    ) -> None:
        if not isinstance(self.base_ansatz, HVA) or not isinstance(self.target_ansatz, HVA):
            raise ValueError("HVA structure transfer requires HVA base and target ansatzes")
        if self.base_ansatz.n_terms != self.target_ansatz.n_terms:
            raise ValueError("base and target HVA must use the same number of terms")
        if self.target_n_layers % self.base_n_layers != 0:
            raise ValueError("target layers must be a multiple of base layers for HVA transfer")

        layer_groups = self.target_n_layers // self.base_n_layers
        if len(pattern) != layer_groups:
            raise ValueError(f"HVA structure pattern length must be {layer_groups}, got {len(pattern)}")

        block_size = self.base_n_layers * self.base_ansatz.n_terms
        for layer_group, marker in enumerate(pattern):
            if marker == "R":
                continue
            if marker != "T":
                raise ValueError(f"Unknown transfer marker {marker!r}")
            start = layer_group * block_size
            target[start : start + block_size] = base_params[:block_size]


def generate_ble_params(
    target_ansatz: HVA,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Block identity encoding for the modified HVA in paper equation (12)."""

    rng = rng or np.random.default_rng()
    if not target_ansatz.use_mod_hva:
        raise ValueError("BLE requires modified HVA with reversed even layers")
    if target_ansatz.n_layers % 2 != 0:
        raise ValueError("BLE requires an even number of HVA layers")

    params = np.zeros(target_ansatz.n_params)
    for layer in range(0, target_ansatz.n_layers, 2):
        theta = random_parameters(target_ansatz.n_terms, rng)
        start = layer * target_ansatz.n_terms
        next_start = (layer + 1) * target_ansatz.n_terms
        params[start : start + target_ansatz.n_terms] = theta
        params[next_start : next_start + target_ansatz.n_terms] = -theta
    return params


def get_patterns_for_task(task_name: str) -> list[str]:
    patterns = {
        "task_a": ["TTT", "RRT", "TTR", "RRR"],
        "task_b": ["TTTTT", "TRTRT", "RTRTR", "RRRRR"],
        "task_c": ["TT", "TR", "RT", "RR"],
        "task_d": ["TT", "TR", "RT", "RR"],
        "task_e": ["TTTT", "TRRT", "RTTR", "RRRR"],
        "task_f": ["TT", "TR", "RT", "RR", "BLE"],
    }
    return patterns[task_name.lower()]


def create_transfer(
    transfer_type: str,
    base_ansatz: Ansatz,
    target_ansatz: Ansatz | None = None,
    target_n_qubits: int | None = None,
    qubit_offsets: list[int] | None = None,
) -> NetworkTransfer | StructureTransfer:
    if transfer_type == "network":
        if target_n_qubits is None:
            raise ValueError("target_n_qubits is required for network transfer")
        return NetworkTransfer(base_ansatz, target_n_qubits)
    if transfer_type == "structure":
        if target_ansatz is None:
            raise ValueError("target_ansatz is required for structure transfer")
        return StructureTransfer(base_ansatz, target_ansatz, qubit_offsets=qubit_offsets)
    raise ValueError(f"Unknown transfer type: {transfer_type}")
