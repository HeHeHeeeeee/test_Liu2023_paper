# Experiment Report

This file records implementation status, not final numerical claims.

## Fixed Issues

- Replaced the split PyTorch/unfinished-class implementation with one NumPy/SciPy code path.
- Added missing `HEA`, `HVA`, `AnsatzConfig`, `TransverseFieldIsing`, `HeisenbergXXZ`, and `HydrogenMolecule` classes.
- Corrected Task C-F layer counts to match the paper's 4-layer base and 8-layer target structure-transfer setup.
- Corrected base parameter pools to store optimized successful parameters.
- Corrected network transfer to use sliding windows of length `m-n+1`.
- Corrected XXZ HVA terms to `H_X`, `H_Y`, and `H_Z` without an extra transverse field.
- Corrected Task F BLE to use modified-HVA inverse layer pairs.

## Remaining Practical Notes

- Task C requires OpenFermion + PySCF to generate the H2/H3 Hamiltonians.
- Full TTN reproduction is computationally expensive because every reported number requires 100 successful target runs.
- Generated results should be compared against `RESULTS.md`, which lists the paper Table 1 TTNs.
