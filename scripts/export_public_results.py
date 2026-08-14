#!/usr/bin/env python3
"""Export compact, non-raw-data result tables for the public GitHub repository."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "artifacts/final_evaluation"
DESTINATION = PROJECT_ROOT / "results"
PUBLIC_TABLES = (
    "model_comparison.csv",
    "per_motor_metrics.csv",
    "event_metrics.csv",
    "nonoverlap_model_comparison.csv",
    "test_support.csv",
    "run_bootstrap_confidence_intervals.csv",
    "ablation_summary.csv",
)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for filename in PUBLIC_TABLES:
        shutil.copy2(SOURCE / filename, DESTINATION / filename)

    manifest = pd.read_csv(
        PROJECT_ROOT / "artifacts/manifests/windows_test.csv", dtype={"run_id": str}
    ).reset_index(drop=True)
    payload = np.load(SOURCE / "test_scores_and_labels.npz")
    metrics = json.loads((SOURCE / "test_metrics.json").read_text(encoding="utf-8"))
    selected_run = "20240503_163963"
    selected = manifest[manifest["run_id"] == selected_run].copy()
    index = selected.index.to_numpy(dtype=np.int64)
    timeline = selected[["run_id", "activity", "start_idx", "end_idx", "m6_label"]].copy()
    timeline["m6_score"] = payload["gmm"][index, 5]
    timeline["m6_threshold"] = metrics["model_metrics"]["GMM"]["thresholds"][5]
    timeline.to_csv(DESTINATION / "representative_m6_timeline.csv", index=False)
    print(f"Exported {len(PUBLIC_TABLES) + 1} public result tables to {DESTINATION}")


if __name__ == "__main__":
    main()
