# Reproduction Plan

## Paper Targets

The code follows Liu et al. NJP 25, 013039 (2023):

- HEA block: `Rz-Rx-Rz` on every qubit, then ring CZ entanglers.
- HVA block: `prod_m exp(-i theta_{p,m} H_m)`.
- TFIM: `H = -J sum ZZ - h sum X`, with `J=1`, `h=2`.
- XXZ: `H = -J sum (XX + YY + Delta ZZ)`, with `J=1`, `Delta=2`.
- Optimizer: BFGS.
- Success threshold: `1.6e-3`.
- TTN: total trials needed to collect 100 successful target runs.

## Task Configuration

| Task | Base | Target | Ansatz | Transfer |
|---|---|---|---|---|
| A | 4-qubit TFIM, 4-layer HEA | 6-qubit TFIM | Network |
| B | 4-qubit TFIM, 4-layer HEA | 8-qubit TFIM | Network |
| C | H2, 4-layer HEA | H3, 8-layer HEA | Structure |
| D | 4-qubit 1D XXZ, 4-layer HVA | 6-qubit 1D XXZ, 8-layer HVA | Structure |
| E | 4-qubit 1D XXZ, 4-layer HEA | 2x4 XXZ, 8-layer HEA | Structure |
| F | 4-qubit 1D XXZ, 4-layer modified HVA | 8-qubit 1D XXZ, 8-layer modified HVA | Structure + BLE |

## Verification Checklist

- `python smoke_test.py` passes without running long optimizations.
- `python -m py_compile ...` passes for all Python files.
- Base pool JSON files contain optimized successful parameters.
- Each task writes JSONL trial logs and JSON summary metrics.

## Notes

Task C depends on OpenFermion + PySCF because the paper does not include every H3 Pauli coefficient in the PDF. Install the full `requirements.txt` before running it.
