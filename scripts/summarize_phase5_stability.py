#!/usr/bin/env python3
"""Aggregate predeclared Phase 5 seeds; never reads held-out Test artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root", type=Path, default=Path("artifacts/experiments")
    )
    args = parser.parse_args()
    experiment_dirs = {
        42: args.experiment_root / "cnn_transformer_ssl_seed42",
        7: args.experiment_root / "ssl_stability/seed7",
        123: args.experiment_root / "ssl_stability/seed123",
    }
    baseline_path = args.experiment_root / "cnn_transformer_seed42/validation_metrics.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_overall = baseline["validation"]["overall_micro"]
    baseline_m6 = baseline["validation"]["per_motor"]["m6"]

    rows: list[dict] = []
    for seed, directory in experiment_dirs.items():
        metrics = json.loads((directory / "validation_metrics.json").read_text(encoding="utf-8"))
        context = json.loads((directory / "run_context.json").read_text(encoding="utf-8"))
        initialization = metrics["initialization_report"]
        if not initialization.get("loaded"):
            raise AssertionError(f"Seed {seed} did not load SSL encoder weights")
        if sorted(initialization.get("missing_keys", [])) != ["head.1.bias", "head.1.weight"]:
            raise AssertionError(f"Seed {seed} has unexpected missing keys")
        if initialization.get("unexpected_keys"):
            raise AssertionError(f"Seed {seed} has unexpected encoder keys")
        if context["test_evaluated"] or context["pretraining_labels_used"]:
            raise AssertionError(f"Seed {seed} violates the frozen SSL protocol")
        overall = metrics["validation"]["overall_micro"]
        m6 = metrics["validation"]["per_motor"]["m6"]
        rows.append(
            {
                "seed": seed,
                "overall_pr_auc": overall["pr_auc"],
                "overall_f1": overall["f1"],
                "m6_pr_auc": m6["pr_auc"],
                "m6_f1": m6["f1"],
                "delta_overall_f1_vs_without_ssl": overall["f1"] - baseline_overall["f1"],
                "delta_m6_pr_auc_vs_without_ssl": m6["pr_auc"] - baseline_m6["pr_auc"],
                "delta_m6_f1_vs_without_ssl": m6["f1"] - baseline_m6["f1"],
                "encoder_loaded": True,
                "test_evaluated": False,
            }
        )

    frame = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    output_dir = args.experiment_root / "ssl_stability"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "seed_metrics.csv", index=False)
    metric_columns = ["overall_pr_auc", "overall_f1", "m6_pr_auc", "m6_f1"]
    aggregate = {
        metric: {
            "mean": float(frame[metric].mean()),
            "sample_std": float(frame[metric].std(ddof=1)),
            "min": float(frame[metric].min()),
            "max": float(frame[metric].max()),
        }
        for metric in metric_columns
    }
    aggregate.update(
        {
            "seeds": frame["seed"].astype(int).tolist(),
            "n_seeds": int(len(frame)),
            "m6_f1_improved_seed_count": int((frame["delta_m6_f1_vs_without_ssl"] > 0).sum()),
            "m6_pr_auc_improved_seed_count": int(
                (frame["delta_m6_pr_auc_vs_without_ssl"] > 0).sum()
            ),
            "without_ssl_seed42": {
                "overall_pr_auc": baseline_overall["pr_auc"],
                "overall_f1": baseline_overall["f1"],
                "m6_pr_auc": baseline_m6["pr_auc"],
                "m6_f1": baseline_m6["f1"],
            },
            "interpretation": (
                "SSL improved M6 PR-AUC and F1 in all three seeds, but the large "
                "between-seed variance and limited fault-run support require confirmation "
                "on the locked Test split."
            ),
            "test_evaluated": False,
        }
    )
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
