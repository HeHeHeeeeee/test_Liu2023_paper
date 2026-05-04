#!/usr/bin/env python3
"""Fast structural checks for the reproduction code.

This smoke test avoids long BFGS runs.  It verifies the circuit dimensions,
transfer-pattern semantics and modified-HVA BLE identity construction.
"""

from __future__ import annotations

import numpy as np

from src.ansatz import AnsatzConfig, HEA, HVA, NetworkTransferHEA
from src.engine import zero_state
from src.hamiltonians import HeisenbergXXZ, TransverseFieldIsing
from src.transfer import NetworkTransfer, StructureTransfer, generate_ble_params


def main() -> int:
    tfim = TransverseFieldIsing(4)
    assert tfim.matrix.shape == (16, 16)
    assert np.isfinite(tfim.exact_energy)

    base_network = NetworkTransferHEA(m_qubits=4, n_qubits=4, layers=4)
    target_network = NetworkTransferHEA(m_qubits=6, n_qubits=4, layers=4)
    assert base_network.n_params == 48
    assert target_network.n_params == 144

    transfer = NetworkTransfer(base_network, target_n_qubits=6)
    base_params = np.zeros(base_network.n_params)
    target_params = transfer.generate_params(base_params, "TTT", np.random.default_rng(1))
    assert target_params.shape == (target_network.n_params,)

    base_hea = HEA(AnsatzConfig(n_qubits=4, n_layers=4, ansatz_type="HEA"))
    target_hea = HEA(AnsatzConfig(n_qubits=6, n_layers=8, ansatz_type="HEA"))
    structure = StructureTransfer(base_hea, target_hea)
    params = structure.generate_params(np.zeros(base_hea.n_params), "TT", np.random.default_rng(2))
    assert params.shape == (target_hea.n_params,)

    xxz = HeisenbergXXZ(4, geometry="1d")
    mod_hva = HVA(
        AnsatzConfig(
            n_qubits=4,
            n_layers=4,
            ansatz_type="HVA",
            hamiltonian_terms=xxz.hva_terms(),
            use_mod_hva=True,
        )
    )
    ble_params = generate_ble_params(mod_hva, np.random.default_rng(3))
    assert np.allclose(mod_hva.state(ble_params), zero_state(4), atol=1.0e-10)

    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
