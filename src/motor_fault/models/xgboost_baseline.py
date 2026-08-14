from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from motor_fault.evaluation.metrics import (
    compare_thresholds,
    evaluate_multimotor,
    select_support_aware_thresholds,
)
from motor_fault.models.traditional import EmpiricalScoreCalibrator


def stack_motor_targets(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8)
    if x.ndim != 2 or y.shape != (len(x), 6):
        raise ValueError(f"Expected X=[n,d], y=[n,6], got {x.shape}, {y.shape}")
    motor_ids = np.tile(np.arange(6, dtype=np.int8), len(x))
    repeated = np.repeat(x, 6, axis=0)
    one_hot = np.eye(6, dtype=np.float32)[motor_ids]
    return np.concatenate([repeated, one_hot], axis=1), y.reshape(-1), motor_ids


def per_motor_class_weights(labels: np.ndarray, motor_ids: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    weights = np.ones(len(y), dtype=np.float32)
    summary = {}
    for motor in range(6):
        mask = motor_ids == motor
        positives = int(y[mask].sum())
        negatives = int(mask.sum() - positives)
        if positives == 0:
            raise ValueError(f"Training has no positives for Motor {motor + 1}")
        positive_weight = negatives / positives
        weights[mask & (y == 1)] = positive_weight
        summary[f"m{motor + 1}"] = float(positive_weight)
    return weights, summary


def run_xgboost_trials(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    config: dict[str, Any],
    seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("Install requirements-baselines.txt to run XGBoost") from exc

    stacked_train, stacked_y_train, train_motor_ids = stack_motor_targets(x_train, y_train)
    stacked_validation, stacked_y_validation, _ = stack_motor_targets(
        x_validation, y_validation
    )
    train_weights, positive_weights = per_motor_class_weights(
        stacked_y_train, train_motor_ids
    )
    dtrain = xgb.DMatrix(stacked_train, label=stacked_y_train, weight=train_weights)
    dvalidation = xgb.DMatrix(stacked_validation, label=stacked_y_validation)

    trials = []
    fitted = []
    for max_depth, learning_rate in product(
        config["max_depth"], config["learning_rate"]
    ):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "max_depth": int(max_depth),
            "eta": float(learning_rate),
            "min_child_weight": float(config["min_child_weight"]),
            "subsample": float(config["subsample"]),
            "colsample_bytree": float(config["colsample_bytree"]),
            "lambda": float(config["reg_lambda"]),
            "alpha": float(config["reg_alpha"]),
            "seed": seed,
            "nthread": int(config["nthread"]),
        }
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=int(config["num_boost_round"]),
            evals=[(dvalidation, "validation")],
            early_stopping_rounds=int(config["early_stopping_rounds"]),
            maximize=True,
            verbose_eval=False,
        )
        end_iteration = int(booster.best_iteration) + 1
        train_raw_scores = booster.predict(
            dtrain, iteration_range=(0, end_iteration)
        ).reshape(len(x_train), 6)
        validation_raw_scores = booster.predict(
            dvalidation, iteration_range=(0, end_iteration)
        ).reshape(len(x_validation), 6)
        calibrator = EmpiricalScoreCalibrator.fit(train_raw_scores)
        validation_scores = calibrator.transform(validation_raw_scores)
        thresholds, threshold_policies = select_support_aware_thresholds(
            y_validation, validation_scores
        )
        metrics = evaluate_multimotor(y_validation, validation_scores, thresholds)
        trial = {
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "best_iteration": int(booster.best_iteration),
            "best_validation_aucpr": float(booster.best_score),
            "thresholds": thresholds.tolist(),
            "m6_pr_auc": metrics["per_motor"]["m6"]["pr_auc"],
            "m6_f1": metrics["per_motor"]["m6"]["f1"],
            **metrics["overall_micro"],
        }
        trials.append(trial)
        fitted.append(
            (booster, calibrator, validation_scores, metrics, threshold_policies)
        )

    best_index = max(
        range(len(trials)),
        key=lambda index: (
            trials[index]["m6_pr_auc"],
            trials[index]["m6_f1"],
            trials[index]["f1"],
            -trials[index]["best_iteration"],
        ),
    )
    best_trial = trials[best_index]
    booster, calibrator, validation_scores, metrics, threshold_policies = fitted[best_index]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    booster.save_model(destination / "model.ubj")
    import joblib

    joblib.dump(
        {
            "calibrator": calibrator,
            "thresholds": best_trial["thresholds"],
            "threshold_policies": threshold_policies,
        },
        destination / "score_calibrator.joblib",
    )
    pd.DataFrame(trials).to_csv(destination / "validation_trials.csv", index=False)
    return {
        "model": "XGBoost",
        "selection_split": "validation",
        "training_design": "shared window-by-target-motor binary classifier",
        "train_only_positive_weights": positive_weights,
        "model_selection_metric": "m6_validation_pr_auc",
        "threshold_policies": threshold_policies,
        "best_hyperparameters": {
            "max_depth": best_trial["max_depth"],
            "learning_rate": best_trial["learning_rate"],
            "best_iteration": best_trial["best_iteration"],
        },
        "threshold_comparison": compare_thresholds(y_validation, validation_scores),
        "validation": metrics,
        "validation_trials": trials,
    }
