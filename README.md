# Liu 2023 Replication

Replication code for:

**Huan-Yu Liu et al., "Mitigating barren plateaus with transfer-learning-inspired parameter initializations", New J. Phys. 25 (2023) 013039.**

The implementation uses dense NumPy state-vector simulation and SciPy BFGS optimization for the small systems in the paper.

## What Is Implemented

| Task | Model | Base -> Target | Ansatz | Transfer | Patterns |
|---|---|---:|---|---|---|
| A | 1D TFIM | 4 -> 6 qubits | 4-layer HEA | Network | TTT, RRT, TTR, RRR |
| B | 1D TFIM | 4 -> 8 qubits | 4-layer HEA | Network | TTTTT, TRTRT, RTRTR, RRRRR |
| C | H2 -> H3 | 4 -> 6 qubits | 4 -> 8 layer HEA | Structure | TT, TR, RT, RR |
| D | 1D XXZ | 4 -> 6 qubits | 4 -> 8 layer HVA | Structure | TT, TR, RT, RR |
| E | XXZ 1D -> 2x4 | 4 -> 8 qubits | 4 -> 8 layer HEA | Structure | TTTT, TRRT, RTTR, RRRR |
| F | 1D XXZ | 4 -> 8 qubits | 4 -> 8 layer modified HVA | Structure + BLE | TT, TR, RT, RR, BLE |

Task C builds the STO-3G Jordan-Wigner Hamiltonians through OpenFermion + PySCF.

## Run

```bash
python -m pip install -r requirements.txt
python smoke_test.py
python run_replication_parallel.py
```

Run one task:

```bash
python task_a.py
python run_replication_parallel.py task_a task_b
```

Fast debugging run:

```bash
BASE_SUCCESSES=1 TARGET_SUCCESSES=1 MAX_ITER_BASE=2 MAX_ITER_TARGET=2 python task_a.py
```

Outputs are written under `outputs/logs`, `outputs/pools`, and `outputs/metrics`.

## Important Reproduction Details

- Base pools store the optimized successful parameters `theta*`, not the random starting points.
- Network transfer follows paper equation (9): a target circuit is made from sliding-window copies of the trained base HEA.
- Structure transfer copies base-sized layer/qubit blocks into the unchanged target ansatz.
- XXZ HVA uses exactly three Hamiltonian parts: `H_X`, `H_Y`, and `H_Z = -J Delta sum ZZ`.
- Task F BLE uses the modified HVA with reversed even-layer term order and `theta_{p+1,m} = -theta_{p,m}`.
