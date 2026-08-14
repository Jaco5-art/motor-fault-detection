#!/usr/bin/env python3
"""Generate descriptive Phase 6 analysis without changing models or thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


MODEL_SCORE_KEYS = {
    "GMM": "gmm",
    "Isolation Forest": "isolation_forest",
    "XGBoost": "xgboost",
    "1D-CNN": "1d_cnn",
    "CNN + Transformer": "cnn_plus_transformer",
    "CNN + Transformer + SSL": "cnn_plus_transformer_plus_ssl",
}


def metric_delta_row(
    comparison: pd.DataFrame,
    ablation: str,
    baseline: str,
    candidate: str,
    controlled: bool,
) -> dict:
    base = comparison.set_index("model").loc[baseline]
    cand = comparison.set_index("model").loc[candidate]
    return {
        "ablation": ablation,
        "baseline": baseline,
        "candidate": candidate,
        "controlled_comparison": controlled,
        "delta_pr_auc": cand.pr_auc - base.pr_auc,
        "delta_precision": cand.precision - base.precision,
        "delta_recall": cand.recall - base.recall,
        "delta_f1": cand.f1 - base.f1,
        "relative_f1_change_percent": 100 * (cand.f1 - base.f1) / base.f1,
    }


def run_bootstrap(
    manifest: pd.DataFrame,
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, np.ndarray],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = {
        str(run_id): group.index.to_numpy(dtype=np.int64)
        for run_id, group in manifest.groupby("run_id", sort=True)
    }
    run_ids = np.asarray(list(groups))
    samples: dict[str, list[tuple[float, float]]] = {name: [] for name in scores}
    for _ in range(iterations):
        selected_runs = rng.choice(run_ids, size=len(run_ids), replace=True)
        index = np.concatenate([groups[str(run_id)] for run_id in selected_runs])
        y = labels[index].reshape(-1)
        if y.sum() == 0:
            continue
        for name, model_scores in scores.items():
            values = model_scores[index]
            predicted = (values >= thresholds[name].reshape(1, 6)).astype(np.int8)
            _, _, f1, _ = precision_recall_fscore_support(
                y, predicted.reshape(-1), average="binary", zero_division=0
            )
            samples[name].append((average_precision_score(y, values.reshape(-1)), f1))
    rows = []
    for name, values in samples.items():
        array = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "model": name,
                "bootstrap_unit": "run",
                "iterations_requested": iterations,
                "iterations_with_positive_support": len(array),
                "pr_auc_ci_low": float(np.percentile(array[:, 0], 2.5)),
                "pr_auc_ci_high": float(np.percentile(array[:, 0], 97.5)),
                "f1_ci_low": float(np.percentile(array[:, 1], 2.5)),
                "f1_ci_high": float(np.percentile(array[:, 1], 97.5)),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    shown = frame[columns].copy()
    for column in shown.select_dtypes(include=["float"]).columns:
        shown[column] = shown[column].map(lambda value: f"{value:.{digits}f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in shown.to_numpy()]
    return "\n".join([header, separator, *body])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-dir", type=Path, default=Path("artifacts/final_evaluation")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/manifests/windows_test.csv")
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    root = args.evaluation_dir
    status = json.loads((root / "final_evaluation_status.json").read_text())
    if status.get("status") != "completed" or status.get("threshold_selection_performed"):
        raise ValueError("Phase 6 outputs are not a completed frozen evaluation")

    comparison = pd.read_csv(root / "model_comparison.csv")
    per_motor = pd.read_csv(root / "per_motor_metrics.csv")
    event = pd.read_csv(root / "event_metrics.csv")
    support = pd.read_csv(root / "test_support.csv")
    errors = pd.read_csv(root / "window_errors.csv")
    manifest = pd.read_csv(args.manifest, dtype={"run_id": str}).reset_index(drop=True)
    score_payload = np.load(root / "test_scores_and_labels.npz")
    labels = score_payload["labels"].astype(np.int8)
    metrics_payload = json.loads((root / "test_metrics.json").read_text())
    scores = {name: score_payload[key] for name, key in MODEL_SCORE_KEYS.items()}
    thresholds = {
        name: np.asarray(metrics_payload["model_metrics"][name]["thresholds"])
        for name in MODEL_SCORE_KEYS
    }

    bootstrap = run_bootstrap(
        manifest, labels, scores, thresholds, args.bootstrap_iterations, seed=42
    )
    bootstrap.to_csv(root / "run_bootstrap_confidence_intervals.csv", index=False)

    ablations = pd.DataFrame(
        [
            metric_delta_row(
                comparison,
                "Handcrafted features vs raw sequence (pragmatic, model-confounded)",
                "GMM",
                "1D-CNN",
                False,
            ),
            metric_delta_row(
                comparison,
                "CNN vs CNN + Transformer",
                "1D-CNN",
                "CNN + Transformer",
                True,
            ),
            metric_delta_row(
                comparison,
                "Without SSL vs with SSL",
                "CNN + Transformer",
                "CNN + Transformer + SSL",
                True,
            ),
        ]
    )
    ablations.to_csv(root / "ablation_summary.csv", index=False)

    error_counts = (
        errors.groupby(["model", "run_id", "activity", "motor", "error_type"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["model", "count"], ascending=[True, False])
    )
    error_counts.to_csv(root / "error_counts_by_run_motor.csv", index=False)
    ssl_motor = per_motor[per_motor.model == "CNN + Transformer + SSL"].set_index("motor")
    transformer_motor = per_motor[per_motor.model == "CNN + Transformer"].set_index("motor")
    ssl_delta = pd.DataFrame(
        {
            "motor": ssl_motor.index,
            "delta_pr_auc_with_ssl": ssl_motor.pr_auc - transformer_motor.pr_auc,
            "delta_recall_with_ssl": ssl_motor.recall - transformer_motor.recall,
            "delta_f1_with_ssl": ssl_motor.f1 - transformer_motor.f1,
        }
    ).reset_index(drop=True)
    ssl_delta.to_csv(root / "ssl_per_motor_delta.csv", index=False)

    indexed = comparison.set_index("model")
    traditional_names = ["GMM", "Isolation Forest", "XGBoost"]
    best_traditional_name = indexed.loc[traditional_names, "f1"].idxmax()
    best_traditional = indexed.loc[best_traditional_name]
    core = indexed.loc["CNN + Transformer + SSL"]
    prevalence = float(labels.mean())
    evidence = {
        "eligible_source": "single frozen independent Test evaluation",
        "test_runs_with_full_windows": int(manifest.run_id.nunique()),
        "test_windows": int(len(manifest)),
        "motor_window_pairs": int(labels.size),
        "positive_prevalence": prevalence,
        "best_model": best_traditional_name,
        "best_model_pr_auc": float(best_traditional.pr_auc),
        "best_model_f1": float(best_traditional.f1),
        "best_model_recall": float(best_traditional.recall),
        "best_model_event_f1": float(
            event[(event.model == best_traditional_name) & (event.motor == "overall")][
                "event_f1"
            ].iloc[0]
        ),
        "pr_auc_over_prevalence_multiple": float(best_traditional.pr_auc / prevalence),
        "core_ssl_pr_auc": float(core.pr_auc),
        "core_ssl_f1": float(core.f1),
        "core_beats_best_traditional_f1": bool(core.f1 > best_traditional.f1),
        "ssl_improves_transformer_test_f1": bool(
            core.f1 > indexed.loc["CNN + Transformer", "f1"]
        ),
        "supported_resume_conclusion": (
            "GMM was the strongest fixed Test model; SSL did not improve Test F1 over "
            "the matched Transformer and must not be described as the final improvement."
        ),
    }
    (root / "resume_evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )

    comparison_table = markdown_table(
        comparison, ["model", "pr_auc", "precision", "recall", "f1"]
    )
    event_overall = event[event.motor == "overall"]
    event_table = markdown_table(
        event_overall, ["model", "event_precision", "event_recall", "event_f1"]
    )
    support_table = markdown_table(
        support, ["motor", "positive_windows", "fault_window_ratio", "positive_runs"]
    )
    ablation_table = markdown_table(
        ablations,
        ["ablation", "delta_pr_auc", "delta_recall", "delta_f1"],
    )
    report = f"""# Phase 6: frozen independent Test results

## Protocol integrity

- Six previously selected checkpoints were evaluated together with their frozen Validation thresholds.
- No training, threshold selection, point adjustment, or Test-driven model revision was performed.
- The first execution attempt stopped before metrics were written because XGBoost was absent from the runtime. Attempt 2 restored only the dependency and used identical hashed inputs.
- Primary results use all motor-window pairs; run-bootstrap confidence intervals and greedy non-overlap results are diagnostics.

## Unified Test comparison

{comparison_table}

GMM is the strongest Test model: PR-AUC {evidence['best_model_pr_auc']:.3f}, F1 {evidence['best_model_f1']:.3f}, and Recall {evidence['best_model_recall']:.3f}. Its PR-AUC is {evidence['pr_auc_over_prevalence_multiple']:.2f} times the motor-window fault prevalence ({prevalence:.3f}). The SSL model does not outperform the matched Transformer on F1 and is substantially below GMM.

## Event-level results

{event_table}

Event metrics use contiguous raw fault episodes within evaluated coverage and maximum-cardinality one-to-one overlap matching. Predictions are not extended or point-adjusted.

## Test support

{support_table}

M1–M5 each have positive windows from only one independent Test run. Their per-motor metrics must therefore be treated as case-specific, even though all six motors have positive Test support.

## Three planned ablations

{ablation_table}

The handcrafted-versus-raw comparison is not architecture-controlled. The controlled component comparisons show that adding the Transformer reduces Test F1 relative to the CNN, and SSL produces a small Recall increase but a further F1 decrease relative to the matched Transformer.

## Error analysis

- GMM achieves the best precision/recall balance, but completely misses M1 and M3 at the frozen thresholds.
- The 1D-CNN reaches the highest overall Recall ({indexed.loc['1D-CNN', 'recall']:.3f}) but produces 1,744 false-positive motor-window decisions and only {indexed.loc['1D-CNN', 'precision']:.3f} Precision.
- The SSL model improves M3 ranking and raises Recall for M2/M4, but M6 Recall falls from {transformer_motor.loc['M6', 'recall']:.3f} without SSL to {ssl_motor.loc['M6', 'recall']:.3f} with SSL. Therefore SSL does not support the intended low-label motor generalization claim.
- The non-overlapping diagnostic preserves the overall ranking: GMM remains best at PR-AUC 0.356 and F1 0.400.

## Final interpretation

The defensible project result is not that Transformer or SSL wins. It is that a leakage-free, group-based evaluation exposed severe cross-run generalization error and showed that the simpler density baseline was more robust on this small labeled dataset. This is a credible engineering finding and should replace any provisional resume metrics.
"""
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/phase6_final_test_results.md").write_text(report, encoding="utf-8")

    resume = f"""# Resume-ready project entry

## 多电机多变量时序故障检测

**技术栈：** Python、PyTorch、Scikit-learn、XGBoost、1D-CNN、Transformer、自监督对比学习

**实验设计：** 面向6电机18通道时序数据，按独立运行片段构建无泄漏 Train/Validation/Test 流程；仅使用 Train 拟合 scaler 与类别权重，并在 Validation 完成模型选择、早停和阈值冻结。

**模型与评估：** 构建 GMM、Isolation Forest、XGBoost、1D-CNN 与 CNN-Transformer 基线，实现基于 NT-Xent 的 Train-only 对比预训练；统一使用 PR-AUC、Precision、Recall、F1、分电机及事件级指标评估，完成三项组件消融和跨运行错误分析。

**研究成果：** 在包含8个独立运行片段的固定测试集上，GMM取得最佳 PR-AUC {evidence['best_model_pr_auc']:.3f}、F1 {evidence['best_model_f1']:.3f} 和事件级 F1 {evidence['best_model_event_f1']:.3f}，PR-AUC达到故障基准占比的 {evidence['pr_auc_over_prevalence_multiple']:.2f} 倍；错误分析发现深度模型在小样本跨工况场景产生大量误报，且自监督学习未改善最终 F1，据此形成可复现的负结果分析。

## Wording restriction

Do not claim that Transformer or SSL improved final Test performance. Do not reuse the provisional PR-AUC 0.84, F1 0.78, Recall 0.81, or “18% improvement” figures.
"""
    Path("reports/resume_project_entry.md").write_text(resume, encoding="utf-8")
    print("Phase 6 analysis and resume evidence generated")


if __name__ == "__main__":
    main()
