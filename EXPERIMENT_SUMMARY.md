# Experiment Summary

The previous partial experiment notes were based on an older mixed PyTorch/PennyLane-style implementation and are no longer valid.

Current status:

- Core dense simulator implemented in `src/`.
- Task A-F wrappers use the same shared runner in `src/experiments.py`.
- Full TTN results should be regenerated with `python run_replication_parallel.py`.
- Use `python smoke_test.py` for a fast structural check before long runs.
