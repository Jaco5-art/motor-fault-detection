# Generated artifacts

This directory is intentionally excluded from Git except for this note. Running the pipeline creates:

- source-data audits and group-split manifests;
- Train-only scalers and feature caches;
- model checkpoints and training histories;
- frozen Test scores, detailed errors, and integrity checks.

Compact aggregate results suitable for public review are versioned under [`results/`](../results/).
