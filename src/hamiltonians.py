"""Hamiltonians used in Liu et al., New J. Phys. 25, 013039 (2023)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import COMPLEX, exact_ground_energy, one_qubit_pauli, two_qubit_pauli


@dataclass
class DenseHamiltonian:
    n_qubits: int
    matrix: np.ndarray
    exact_energy: float
    description: str


class TransverseFieldIsing(DenseHamiltonian):
    """One-dimensional transverse-field Ising model.

    Paper equation (10):
        H = -J_Ising sum_<i,j> Z_i Z_j - h sum_i X_i

    The Ising benchmarks use the periodic 1D chain, consistent with the
    reported 4-qubit ground energy and with the ring entangler in the HEA.
    """

    def __init__(
        self,
        n_qubits: int,
        j_ising: float = 1.0,
        h: float = 2.0,
        periodic: bool = True,
    ) -> None:
        matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype=COMPLEX)

        n_bonds = n_qubits if periodic else n_qubits - 1
        for i in range(n_bonds):
            j = (i + 1) % n_qubits
            matrix -= j_ising * two_qubit_pauli(n_qubits, i, "Z", j, "Z")

        for i in range(n_qubits):
            matrix -= h * one_qubit_pauli(n_qubits, i, "X")

        super().__init__(
            n_qubits=n_qubits,
            matrix=matrix,
            exact_energy=exact_ground_energy(matrix),
            description=f"TFIM(n={n_qubits}, J={j_ising}, h={h}, periodic={periodic})",
        )


class HeisenbergXXZ(DenseHamiltonian):
    """Heisenberg XXZ model from paper equation (11).

        H = -J_XXZ sum_<i,j> (X_i X_j + Y_i Y_j + Delta Z_i Z_j)
    """

    def __init__(
        self,
        n_qubits: int,
        j_xxz: float = 1.0,
        delta: float = 2.0,
        geometry: str = "1d",
        rows: int | None = None,
        cols: int | None = None,
        periodic: bool | None = None,
    ) -> None:
        self.j_xxz = j_xxz
        self.delta = delta
        self.geometry = geometry.lower()
        self.rows = rows
        self.cols = cols
        self.periodic = periodic

        pairs = self.neighbor_pairs(n_qubits, self.geometry, rows, cols, periodic)
        hx, hy, hz = self.term_matrices(n_qubits, pairs, j_xxz, delta)
        matrix = hx + hy + hz

        super().__init__(
            n_qubits=n_qubits,
            matrix=matrix,
            exact_energy=exact_ground_energy(matrix),
            description=(
                f"XXZ(n={n_qubits}, geometry={self.geometry}, J={j_xxz}, "
                f"Delta={delta}, periodic={periodic})"
            ),
        )
        self.pairs = pairs
        self._hva_terms = [hx, hy, hz]

    @staticmethod
    def neighbor_pairs(
        n_qubits: int,
        geometry: str,
        rows: int | None,
        cols: int | None,
        periodic: bool | None,
    ) -> list[tuple[int, int]]:
        geometry = geometry.lower()
        if geometry == "1d":
            use_periodic = True if periodic is None else periodic
            n_bonds = n_qubits if use_periodic else n_qubits - 1
            return [(i, (i + 1) % n_qubits) for i in range(n_bonds)]

        if geometry != "2d":
            raise ValueError(f"Unknown XXZ geometry: {geometry}")

        if rows is None or cols is None:
            rows, cols = 2, n_qubits // 2
        if rows * cols != n_qubits:
            raise ValueError("rows * cols must equal n_qubits for 2D XXZ")

        use_periodic = False if periodic is None else periodic
        pairs: set[tuple[int, int]] = set()

        def add(a: int, b: int) -> None:
            if a == b:
                return
            pairs.add((a, b) if a < b else (b, a))

        for r in range(rows):
            for c in range(cols):
                q = r * cols + c
                if c + 1 < cols:
                    add(q, r * cols + c + 1)
                elif use_periodic:
                    add(q, r * cols)

                if r + 1 < rows:
                    add(q, (r + 1) * cols + c)
                elif use_periodic:
                    add(q, c)

        return sorted(pairs)

    @staticmethod
    def term_matrices(
        n_qubits: int,
        pairs: list[tuple[int, int]],
        j_xxz: float,
        delta: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dim = 2**n_qubits
        hx = np.zeros((dim, dim), dtype=COMPLEX)
        hy = np.zeros((dim, dim), dtype=COMPLEX)
        hz = np.zeros((dim, dim), dtype=COMPLEX)

        for i, j in pairs:
            hx -= j_xxz * two_qubit_pauli(n_qubits, i, "X", j, "X")
            hy -= j_xxz * two_qubit_pauli(n_qubits, i, "Y", j, "Y")
            hz -= j_xxz * delta * two_qubit_pauli(n_qubits, i, "Z", j, "Z")

        return hx, hy, hz

    def hva_terms(self) -> list[np.ndarray]:
        return self._hva_terms


class HydrogenMolecule(DenseHamiltonian):
    """Linear H_m molecule in STO-3G with Jordan-Wigner mapping.

    This follows the paper's Task C setup.  The dense qubit Hamiltonian is
    generated lazily through OpenFermion + PySCF because the paper does not
    publish all H3 coefficients in the PDF.
    """

    def __init__(
        self,
        n_atoms: int,
        bond_length: float = 0.74,
        basis: str = "sto-3g",
    ) -> None:
        matrix = self._build_matrix(n_atoms, bond_length, basis)
        super().__init__(
            n_qubits=2 * n_atoms,
            matrix=matrix,
            exact_energy=exact_ground_energy(matrix),
            description=f"H{n_atoms}(linear, R={bond_length}A, {basis}, JW)",
        )

    @staticmethod
    def _build_matrix(n_atoms: int, bond_length: float, basis: str) -> np.ndarray:
        try:
            from openfermion import MolecularData, get_sparse_operator, jordan_wigner
            from openfermionpyscf import run_pyscf
        except ImportError as exc:
            raise RuntimeError(
                "Task C requires OpenFermion and PySCF. Install the optional "
                "chemistry dependencies from requirements.txt before running it."
            ) from exc

        geometry = [("H", (0.0, 0.0, i * bond_length)) for i in range(n_atoms)]
        multiplicity = 1 if n_atoms % 2 == 0 else 2
        molecule = MolecularData(
            geometry=geometry,
            basis=basis,
            multiplicity=multiplicity,
            charge=0,
        )
        molecule = run_pyscf(molecule, run_scf=True)
        fermion_hamiltonian = molecule.get_molecular_hamiltonian()
        qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
        sparse = get_sparse_operator(qubit_hamiltonian, n_qubits=2 * n_atoms)
        return sparse.toarray().astype(COMPLEX)


def build_tfim_hamiltonian(
    n_qubits: int, J: float = 1.0, h: float = 2.0
) -> np.ndarray:
    """Backward-compatible helper used by older scripts."""

    return TransverseFieldIsing(n_qubits, j_ising=J, h=h).matrix
