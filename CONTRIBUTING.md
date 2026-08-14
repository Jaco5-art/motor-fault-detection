# Contributing

Contributions should preserve the split and evaluation contracts.

1. Create a feature branch.
2. Install the base package with `python -m pip install -e .`.
3. Run `make test` before opening a pull request.
4. Add or update tests for every behavior change.
5. Do not commit raw Kaggle data, generated experiment artifacts, or untrusted checkpoints.
6. Do not use the frozen Test to choose architecture, hyperparameters, thresholds, or wording claims.

New model development requires a new declared protocol and must use Train/Validation only. Comparing a new model against the existing Test after observing its labels would no longer be a valid untouched-Test claim.
