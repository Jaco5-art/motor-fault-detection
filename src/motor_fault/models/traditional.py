from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

from motor_fault.evaluation.metrics import (
    compare_thresholds,
    evaluate_multimotor,
    select_support_aware_thresholds,
)


@dataclass(frozen=True)
class EmpiricalScoreCalibrator:
    """Map higher-is-more-anomalous scores to Train empirical percentiles."""

    sorted_train_scores: tuple[np.ndarray, ...]

    @classmethod
    def fit(cls, scores: np.ndarray) -> "EmpiricalScoreCalibrator":
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError(f"Expected [windows, 6] scores, got {values.shape}")
        return cls(tuple(np.sort(values[:, motor]) for motor in range(6)))

    def transform(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError(f"Expected [windows, 6] scores, got {values.shape}")
        calibrated = np.empty_like(values)
        for motor, reference in enumerate(self.sorted_train_scores):
            calibrated[:, motor] = np.searchsorted(
                reference, values[:, motor], side="right"
            ) / len(reference)
        return calibrated


def _motor_feature_indices(feature_names: tuple[str, ...], motor: int) -> list[int]:
    prefix = f"m{motor}_"
    indices = [index for index, name in enumerate(feature_names) if name.startswith(prefix)]
    if not indices:
        raise ValueError(f"No handcrafted features found for Motor {motor}")
    return indices


def _gmm_scores(
    models: list[GaussianMixture],
    features: np.ndarray,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    output = np.empty((len(features), 6), dtype=np.float64)
    for motor, model in enumerate(models, start=1):
        indices = _motor_feature_indices(feature_names, motor)
        output[:, motor - 1] = -model.score_samples(features[:, indices].astype(np.float64))
    return output


def run_gmm_trials(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    feature_names: tuple[str, ...],
    config: dict[str, Any],
    seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    trials = []
    fitted = []
    for components in config["components"]:
        models = []
        for motor in range(1, 7):
            indices = _motor_feature_indices(feature_names, motor)
            model = GaussianMixture(
                n_components=int(components),
                covariance_type=str(config["covariance_type"]),
                n_init=int(config["n_init"]),
                reg_covar=float(config["reg_covar"]),
                random_state=seed,
            )
            model.fit(x_train[:, indices].astype(np.float64))
            models.append(model)
        train_raw = _gmm_scores(models, x_train, feature_names)
        validation_raw = _gmm_scores(models, x_validation, feature_names)
        calibrator = EmpiricalScoreCalibrator.fit(train_raw)
        validation_scores = calibrator.transform(validation_raw)
        thresholds, threshold_policies = select_support_aware_thresholds(
            y_validation, validation_scores
        )
        metrics = evaluate_multimotor(y_validation, validation_scores, thresholds)
        trial = {
            "components": int(components),
            "thresholds": thresholds.tolist(),
            "m6_pr_auc": metrics["per_motor"]["m6"]["pr_auc"],
            "m6_f1": metrics["per_motor"]["m6"]["f1"],
            **metrics["overall_micro"],
        }
        trials.append(trial)
        fitted.append(
            (models, calibrator, validation_scores, metrics, threshold_policies)
        )

    best_index = max(
        range(len(trials)),
        key=lambda index: (
            trials[index]["m6_pr_auc"],
            trials[index]["m6_f1"],
            trials[index]["f1"],
            -trials[index]["components"],
        ),
    )
    best_trial = trials[best_index]
    models, calibrator, validation_scores, metrics, threshold_policies = fitted[best_index]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "calibrator": calibrator,
            "feature_names": feature_names,
            "thresholds": best_trial["thresholds"],
            "threshold_policies": threshold_policies,
        },
        destination / "model.joblib",
    )
    pd.DataFrame(trials).to_csv(destination / "validation_trials.csv", index=False)
    return {
        "model": "GMM",
        "selection_split": "validation",
        "best_hyperparameters": {"components": best_trial["components"]},
        "model_selection_metric": "m6_validation_pr_auc",
        "threshold_policies": threshold_policies,
        "threshold_comparison": compare_thresholds(y_validation, validation_scores),
        "validation": metrics,
        "validation_trials": trials,
    }


def _iforest_scores(
    models: list[IsolationForest],
    features: np.ndarray,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    output = np.empty((len(features), 6), dtype=np.float64)
    for motor, model in enumerate(models, start=1):
        indices = _motor_feature_indices(feature_names, motor)
        output[:, motor - 1] = -model.score_samples(features[:, indices])
    return output


def run_isolation_forest_trials(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    feature_names: tuple[str, ...],
    config: dict[str, Any],
    seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    trials = []
    fitted = []
    for n_estimators, max_features in product(
        config["n_estimators"], config["max_features"]
    ):
        models = []
        for motor in range(1, 7):
            indices = _motor_feature_indices(feature_names, motor)
            model = IsolationForest(
                n_estimators=int(n_estimators),
                max_samples=config["max_samples"],
                max_features=float(max_features),
                contamination="auto",
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(x_train[:, indices])
            models.append(model)
        train_raw = _iforest_scores(models, x_train, feature_names)
        validation_raw = _iforest_scores(models, x_validation, feature_names)
        calibrator = EmpiricalScoreCalibrator.fit(train_raw)
        validation_scores = calibrator.transform(validation_raw)
        thresholds, threshold_policies = select_support_aware_thresholds(
            y_validation, validation_scores
        )
        metrics = evaluate_multimotor(y_validation, validation_scores, thresholds)
        trial = {
            "n_estimators": int(n_estimators),
            "max_features": float(max_features),
            "thresholds": thresholds.tolist(),
            "m6_pr_auc": metrics["per_motor"]["m6"]["pr_auc"],
            "m6_f1": metrics["per_motor"]["m6"]["f1"],
            **metrics["overall_micro"],
        }
        trials.append(trial)
        fitted.append(
            (models, calibrator, validation_scores, metrics, threshold_policies)
        )

    best_index = max(
        range(len(trials)),
        key=lambda index: (
            trials[index]["m6_pr_auc"],
            trials[index]["m6_f1"],
            trials[index]["f1"],
            -trials[index]["n_estimators"],
        ),
    )
    best_trial = trials[best_index]
    models, calibrator, validation_scores, metrics, threshold_policies = fitted[best_index]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "calibrator": calibrator,
            "feature_names": feature_names,
            "thresholds": best_trial["thresholds"],
            "threshold_policies": threshold_policies,
        },
        destination / "model.joblib",
    )
    pd.DataFrame(trials).to_csv(destination / "validation_trials.csv", index=False)
    return {
        "model": "Isolation Forest",
        "selection_split": "validation",
        "best_hyperparameters": {
            "n_estimators": best_trial["n_estimators"],
            "max_features": best_trial["max_features"],
        },
        "model_selection_metric": "m6_validation_pr_auc",
        "threshold_policies": threshold_policies,
        "threshold_comparison": compare_thresholds(y_validation, validation_scores),
        "validation": metrics,
        "validation_trials": trials,
    }
