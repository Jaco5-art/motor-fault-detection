#!/usr/bin/env python3
"""Generate README figures from frozen public result CSVs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/motor-fault-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
OUTPUT = PROJECT_ROOT / "docs/assets"


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 180,
        }
    )


def model_comparison() -> None:
    frame = pd.read_csv(RESULTS / "model_comparison.csv").sort_values("f1")
    labels = frame["model"].str.replace("CNN + Transformer + SSL", "CNN-Transformer + SSL", regex=False)
    labels = labels.str.replace("CNN + Transformer", "CNN-Transformer", regex=False)
    positions = np.arange(len(frame))
    height = 0.36
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.barh(positions - height / 2, frame["pr_auc"], height, label="PR-AUC", color="#2563EB")
    axis.barh(positions + height / 2, frame["f1"], height, label="F1", color="#F59E0B")
    axis.set_yticks(positions, labels)
    axis.set_xlim(0, 0.46)
    axis.set_xlabel("Score on frozen Test set")
    axis.set_title("Model comparison under the same group-held-out Test split", loc="left")
    axis.legend(frameon=False, loc="lower right")
    axis.spines[["top", "right", "left"]].set_visible(False)
    for row, (_, values) in enumerate(frame.iterrows()):
        axis.text(values.pr_auc + 0.008, row - height / 2, f"{values.pr_auc:.3f}", va="center", fontsize=8)
        axis.text(values.f1 + 0.008, row + height / 2, f"{values.f1:.3f}", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT / "model_comparison.png", bbox_inches="tight")
    plt.close(figure)


def per_motor_heatmap() -> None:
    frame = pd.read_csv(RESULTS / "per_motor_metrics.csv")
    pivot = frame.pivot(index="model", columns="motor", values="f1")
    order = [
        "GMM",
        "Isolation Forest",
        "XGBoost",
        "1D-CNN",
        "CNN + Transformer",
        "CNN + Transformer + SSL",
    ]
    pivot = pivot.loc[order, [f"M{motor}" for motor in range(1, 7)]]
    pivot.index = [
        "GMM",
        "Isolation Forest",
        "XGBoost",
        "1D-CNN",
        "CNN-Transformer",
        "CNN-Transformer + SSL",
    ]
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=0.6,
        linewidths=0.7,
        cbar_kws={"label": "F1"},
        ax=axis,
    )
    axis.set_title("Per-motor F1 exposes uneven cross-run generalization", loc="left")
    axis.set_xlabel("Motor")
    axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(OUTPUT / "per_motor_f1_heatmap.png", bbox_inches="tight")
    plt.close(figure)


def representative_timeline() -> None:
    frame = pd.read_csv(RESULTS / "representative_m6_timeline.csv")
    figure, axis = plt.subplots(figsize=(10.5, 4.5))
    axis.plot(frame["start_idx"], frame["m6_score"], color="#2563EB", linewidth=2, label="GMM anomaly score")
    axis.axhline(
        frame["m6_threshold"].iloc[0], color="#DC2626", linestyle="--", linewidth=1.8, label="Frozen threshold"
    )
    active_rows = frame[frame["m6_label"].astype(bool)]
    intervals: list[tuple[int, int]] = []
    for row in active_rows.itertuples(index=False):
        start, end = int(row.start_idx), int(row.end_idx)
        if intervals and start <= intervals[-1][1] + 1:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
    for interval_index, (start, end) in enumerate(intervals):
        axis.axvspan(
            start,
            end,
            color="#FCA5A5",
            alpha=0.28,
            label="Fault-positive window coverage" if interval_index == 0 else None,
        )
    axis.set_ylim(0, 1.04)
    axis.set_xlabel("Window start index")
    axis.set_ylabel("Calibrated anomaly score")
    axis.set_title("Representative M6 Test run: score, threshold, and fault-positive windows", loc="left")
    axis.legend(frameon=False, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(OUTPUT / "representative_m6_timeline.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    model_comparison()
    per_motor_heatmap()
    representative_timeline()
    print(f"Generated 3 README figures in {OUTPUT}")


if __name__ == "__main__":
    main()
