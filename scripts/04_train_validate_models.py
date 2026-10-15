"""Train Logistic Regression and LightGBM using train/validation data only."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.url_feature_extractor import (  # noqa: E402
    model_feature_names,
)


PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODELS_DIRECTORY = (
    PROJECT_ROOT
    / "models"
)

REPORTS_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
)

TRAIN_PATH = (
    PROCESSED_DIRECTORY
    / "train_features.csv"
)

VALIDATION_PATH = (
    PROCESSED_DIRECTORY
    / "validation_features.csv"
)

EXTRACTOR_PATH = (
    PROJECT_ROOT
    / "src"
    / "url_feature_extractor.py"
)

RANDOM_SEED = 42


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_sha256(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def choose_f1_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """
    Select the validation threshold that produces the
    highest phishing-class F1 score.
    """

    (
        precision,
        recall,
        thresholds,
    ) = precision_recall_curve(
        labels,
        probabilities,
    )

    if len(thresholds) == 0:
        return 0.5

    f1_values = (
        2.0
        * precision[:-1]
        * recall[:-1]
        / np.maximum(
            precision[:-1]
            + recall[:-1],
            1e-12,
        )
    )

    best_f1 = np.nanmax(
        f1_values
    )

    candidate_indices = np.flatnonzero(
        np.isclose(
            f1_values,
            best_f1,
        )
    )

    if len(candidate_indices) == 0:
        return 0.5

    candidate_thresholds = (
        thresholds[candidate_indices]
    )

    nearest_to_half = int(
        np.argmin(
            np.abs(
                candidate_thresholds
                - 0.5
            )
        )
    )

    selected_index = (
        candidate_indices[
            nearest_to_half
        ]
    )

    return float(
        thresholds[selected_index]
    )


def calculate_metrics(
    model_name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, int | float | str]:
    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    (
        true_negative,
        false_positive,
        false_negative,
        true_positive,
    ) = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    if (
        true_negative
        + false_positive
    ):
        specificity = (
            true_negative
            / (
                true_negative
                + false_positive
            )
        )

        false_positive_rate = (
            false_positive
            / (
                false_positive
                + true_negative
            )
        )

    else:
        specificity = 0.0
        false_positive_rate = 0.0

    if (
        false_negative
        + true_positive
    ):
        false_negative_rate = (
            false_negative
            / (
                false_negative
                + true_positive
            )
        )

    else:
        false_negative_rate = 0.0

    return {
        "model":
            model_name,

        "threshold":
            float(threshold),

        "accuracy":
            accuracy_score(
                labels,
                predictions,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                labels,
                predictions,
            ),

        "precision":
            precision_score(
                labels,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                labels,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                labels,
                predictions,
                zero_division=0,
            ),

        "roc_auc":
            roc_auc_score(
                labels,
                probabilities,
            ),

        "pr_auc":
            average_precision_score(
                labels,
                probabilities,
            ),

        "mcc":
            matthews_corrcoef(
                labels,
                predictions,
            ),

        "specificity":
            specificity,

        "false_positive_rate":
            false_positive_rate,

        "false_negative_rate":
            false_negative_rate,

        "brier_score":
            brier_score_loss(
                labels,
                probabilities,
            ),

        "log_loss":
            log_loss(
                labels,
                probabilities,
                labels=[0, 1],
            ),

        "true_negative":
            int(true_negative),

        "false_positive":
            int(false_positive),

        "false_negative":
            int(false_negative),

        "true_positive":
            int(true_positive),
    }


# ============================================================
# PLOTS
# ============================================================

def save_roc_pr_plot(
    labels: np.ndarray,
    logistic_probabilities: np.ndarray,
    lightgbm_probabilities: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    models = [
        (
            "Logistic Regression",
            logistic_probabilities,
            "#2f6fdb",
        ),
        (
            "LightGBM",
            lightgbm_probabilities,
            "#e45c2a",
        ),
    ]

    for (
        model_name,
        probabilities,
        colour,
    ) in models:
        (
            false_positive_rate,
            true_positive_rate,
            _,
        ) = roc_curve(
            labels,
            probabilities,
        )

        roc_auc = roc_auc_score(
            labels,
            probabilities,
        )

        axes[0].plot(
            false_positive_rate,
            true_positive_rate,
            label=(
                f"{model_name} "
                f"(AUC={roc_auc:.4f})"
            ),
            color=colour,
        )

        (
            precision,
            recall,
            _,
        ) = precision_recall_curve(
            labels,
            probabilities,
        )

        pr_auc = (
            average_precision_score(
                labels,
                probabilities,
            )
        )

        axes[1].plot(
            recall,
            precision,
            label=(
                f"{model_name} "
                f"(AP={pr_auc:.4f})"
            ),
            color=colour,
        )

    axes[0].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1,
    )

    axes[0].set(
        title="Validation ROC Curve",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )

    axes[1].set(
        title=(
            "Validation Precision-Recall Curve"
        ),
        xlabel="Recall",
        ylabel="Precision",
    )

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_confusion_matrix_plot(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
    output_path: Path,
) -> None:
    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=[
            "Legitimate",
            "Phishing",
        ],
        yticklabels=[
            "Legitimate",
            "Phishing",
        ],
        ax=axis,
    )

    axis.set(
        title=(
            f"{model_name} Validation "
            f"Confusion Matrix\n"
            f"Threshold={threshold:.4f}"
        ),
        xlabel="Predicted class",
        ylabel="True class",
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_feature_importance_plot(
    importance: pd.DataFrame,
    output_path: Path,
) -> None:
    top_features = (
        importance
        .head(20)
        .sort_values(
            "gain_importance_percent"
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    axis.barh(
        top_features["feature"],
        top_features[
            "gain_importance_percent"
        ],
        color="#2b7a78",
    )

    axis.set(
        title=(
            "Top 20 LightGBM "
            "URL/Domain Features"
        ),
        xlabel="Gain importance (%)",
        ylabel="Feature",
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# MAIN TRAINING PROCESS
# ============================================================

def main() -> int:
    print("=" * 76)

    print(
        "IT22131010 - Phase 4 "
        "model training and validation"
    )

    print("=" * 76)

    print(
        "The test split will NOT "
        "be read during this phase."
    )

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    required_paths = [
        TRAIN_PATH,
        VALIDATION_PATH,
        EXTRACTOR_PATH,
    ]

    for required_path in required_paths:
        if not required_path.is_file():
            print(
                "ERROR: Required file "
                f"is missing: {required_path}"
            )

            return 1

    MODELS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_names = (
        model_feature_names()
    )

    feature_schema_hash = (
        hashlib.sha256(
            "\n".join(
                feature_names
            ).encode("utf-8")
        ).hexdigest()
    )

    required_columns = {
        "is_phishing",
        *feature_names,
    }

    for file_path in [
        TRAIN_PATH,
        VALIDATION_PATH,
    ]:
        available_columns = set(
            pd.read_csv(
                file_path,
                nrows=0,
            ).columns
        )

        missing_columns = sorted(
            required_columns.difference(
                available_columns
            )
        )

        if missing_columns:
            print(
                f"ERROR: {file_path.name} "
                "is missing features: "
                f"{missing_columns}"
            )

            return 1

    # --------------------------------------------------------
    # Load train and validation sets
    # --------------------------------------------------------

    print(
        f"Reading {len(feature_names)} "
        "numeric features..."
    )

    train_dataframe = pd.read_csv(
        TRAIN_PATH,
        usecols=[
            "is_phishing",
            *feature_names,
        ],
    )

    validation_dataframe = pd.read_csv(
        VALIDATION_PATH,
        usecols=[
            "source_row_id",
            "is_phishing",
            *feature_names,
        ],
    )

    train_features = (
        train_dataframe[
            feature_names
        ].astype(np.float32)
    )

    train_labels = (
        train_dataframe[
            "is_phishing"
        ].to_numpy(dtype=np.int8)
    )

    validation_features = (
        validation_dataframe[
            feature_names
        ].astype(np.float32)
    )

    validation_labels = (
        validation_dataframe[
            "is_phishing"
        ].to_numpy(dtype=np.int8)
    )

    if (
        train_features
        .isna()
        .any()
        .any()
        or validation_features
        .isna()
        .any()
        .any()
    ):
        print(
            "ERROR: NaN model "
            "features were found."
        )

        return 1

    if (
        not np.isfinite(
            train_features.to_numpy()
        ).all()
        or not np.isfinite(
            validation_features.to_numpy()
        ).all()
    ):
        print(
            "ERROR: Infinite model "
            "features were found."
        )

        return 1

    print(
        f"Train rows: "
        f"{len(train_features):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_features):,}"
    )

    print(
        "Train phishing: "
        f"{int(train_labels.sum()):,}"
    )

    print(
        "Train legitimate: "
        f"{int((train_labels == 0).sum()):,}"
    )

    # --------------------------------------------------------
    # Logistic Regression baseline
    # --------------------------------------------------------

    print(
        "\nTraining Logistic "
        "Regression baseline..."
    )

    logistic_model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2_000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    logistic_start_time = (
        time.perf_counter()
    )

    logistic_model.fit(
        train_features,
        train_labels,
    )

    logistic_training_seconds = (
        time.perf_counter()
        - logistic_start_time
    )

    logistic_probabilities = (
        logistic_model.predict_proba(
            validation_features
        )[:, 1]
    )

    logistic_threshold = (
        choose_f1_threshold(
            validation_labels,
            logistic_probabilities,
        )
    )

    print(
        "Logistic Regression completed "
        f"in {logistic_training_seconds:.2f} "
        "seconds"
    )

    print(
        "Logistic validation threshold: "
        f"{logistic_threshold:.6f}"
    )

    # --------------------------------------------------------
    # LightGBM advanced model
    # --------------------------------------------------------

    print(
        "\nTraining LightGBM "
        "advanced model..."
    )

    lightgbm_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=3_000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.10,
        reg_lambda=0.10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )

    lightgbm_start_time = (
        time.perf_counter()
    )

    lightgbm_model.fit(
        train_features,
        train_labels,
        eval_X=validation_features,
        eval_y=validation_labels,
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                first_metric_only=True,
                verbose=True,
            ),
            lgb.log_evaluation(
                period=100
            ),
        ],
    )

    lightgbm_training_seconds = (
        time.perf_counter()
        - lightgbm_start_time
    )

    lightgbm_probabilities = (
        lightgbm_model.predict_proba(
            validation_features
        )[:, 1]
    )

    lightgbm_threshold = (
        choose_f1_threshold(
            validation_labels,
            lightgbm_probabilities,
        )
    )

    print(
        "LightGBM completed in "
        f"{lightgbm_training_seconds:.2f} "
        "seconds"
    )

    print(
        "LightGBM best iteration: "
        f"{lightgbm_model.best_iteration_}"
    )

    print(
        "LightGBM validation threshold: "
        f"{lightgbm_threshold:.6f}"
    )

    # --------------------------------------------------------
    # Calculate validation metrics
    # --------------------------------------------------------

    metric_rows = [
        calculate_metrics(
            (
                "Logistic Regression "
                "- default threshold"
            ),
            validation_labels,
            logistic_probabilities,
            0.5,
        ),
        calculate_metrics(
            (
                "Logistic Regression "
                "- tuned threshold"
            ),
            validation_labels,
            logistic_probabilities,
            logistic_threshold,
        ),
        calculate_metrics(
            "LightGBM - default threshold",
            validation_labels,
            lightgbm_probabilities,
            0.5,
        ),
        calculate_metrics(
            "LightGBM - tuned threshold",
            validation_labels,
            lightgbm_probabilities,
            lightgbm_threshold,
        ),
    ]

    metrics_dataframe = pd.DataFrame(
        metric_rows
    )

    logistic_tuned_metrics = (
        metric_rows[1]
    )

    lightgbm_tuned_metrics = (
        metric_rows[3]
    )

    # Tuned validation F1 is the primary selection metric.
    if (
        lightgbm_tuned_metrics["f1"]
        >= logistic_tuned_metrics["f1"]
    ):
        selected_model_name = "LightGBM"
        selected_model = lightgbm_model
        selected_threshold = (
            lightgbm_threshold
        )
        selected_probabilities = (
            lightgbm_probabilities
        )
        selected_metrics = (
            lightgbm_tuned_metrics
        )

    else:
        selected_model_name = (
            "Logistic Regression"
        )
        selected_model = logistic_model
        selected_threshold = (
            logistic_threshold
        )
        selected_probabilities = (
            logistic_probabilities
        )
        selected_metrics = (
            logistic_tuned_metrics
        )

    # --------------------------------------------------------
    # Save validation metrics
    # --------------------------------------------------------

    metrics_path = (
        RESULTS_DIRECTORY
        / "04_validation_metrics.csv"
    )

    metrics_dataframe.to_csv(
        metrics_path,
        index=False,
    )

    validation_predictions = pd.DataFrame(
        {
            "source_row_id":
                validation_dataframe[
                    "source_row_id"
                ],

            "is_phishing":
                validation_labels,

            "logistic_phishing_probability":
                logistic_probabilities,

            "lightgbm_phishing_probability":
                lightgbm_probabilities,

            "selected_prediction":
                (
                    selected_probabilities
                    >= selected_threshold
                ).astype(np.int8),
        }
    )

    predictions_path = (
        RESULTS_DIRECTORY
        / "04_validation_predictions.csv"
    )

    validation_predictions.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # LightGBM feature importance
    # --------------------------------------------------------

    booster = lightgbm_model.booster_

    gain_importance = (
        booster.feature_importance(
            importance_type="gain"
        )
    )

    split_importance = (
        booster.feature_importance(
            importance_type="split"
        )
    )

    total_gain = float(
        gain_importance.sum()
    )

    if total_gain:
        gain_percentage = (
            gain_importance
            / total_gain
            * 100
        )

    else:
        gain_percentage = np.zeros_like(
            gain_importance
        )

    feature_importance = pd.DataFrame(
        {
            "feature":
                feature_names,

            "gain_importance":
                gain_importance,

            "gain_importance_percent":
                gain_percentage,

            "split_importance":
                split_importance,
        }
    ).sort_values(
        "gain_importance",
        ascending=False,
    )

    importance_path = (
        RESULTS_DIRECTORY
        / "04_lightgbm_feature_importance.csv"
    )

    feature_importance.to_csv(
        importance_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save models
    # --------------------------------------------------------

    logistic_model_path = (
        MODELS_DIRECTORY
        / "04_logistic_regression_baseline.joblib"
    )

    lightgbm_model_path = (
        MODELS_DIRECTORY
        / "04_lightgbm_domain_typosquatting.joblib"
    )

    booster_path = (
        MODELS_DIRECTORY
        / "04_lightgbm_booster.txt"
    )

    selected_bundle_path = (
        MODELS_DIRECTORY
        / "04_selected_model_bundle.joblib"
    )

    joblib.dump(
        logistic_model,
        logistic_model_path,
        compress=3,
    )

    joblib.dump(
        lightgbm_model,
        lightgbm_model_path,
        compress=3,
    )

    booster.save_model(
        str(booster_path)
    )

    selected_bundle = {
        "component": (
            "IT22131010 Domain & "
            "Typosquatting Detection"
        ),

        "model_name":
            selected_model_name,

        "model":
            selected_model,

        "threshold":
            float(selected_threshold),

        "feature_names":
            feature_names,

        "feature_schema_sha256":
            feature_schema_hash,

        "extractor_sha256":
            calculate_sha256(
                EXTRACTOR_PATH
            ),

        "positive_class": {
            1: "phishing",
            0: "legitimate",
        },

        "dataset_mapping": {
            0: "phishing",
            1: "legitimate",
        },

        "random_seed":
            RANDOM_SEED,

        "validation_metrics":
            selected_metrics,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "test_split_used":
            False,
    }

    joblib.dump(
        selected_bundle,
        selected_bundle_path,
        compress=3,
    )

    # --------------------------------------------------------
    # Save decision configuration
    # --------------------------------------------------------

    decision_configuration = {
        "selected_model":
            selected_model_name,

        "phishing_threshold":
            float(selected_threshold),

        "feature_count":
            len(feature_names),

        "feature_schema_sha256":
            feature_schema_hash,

        "lightgbm_best_iteration":
            int(
                lightgbm_model.best_iteration_
            ),

        "selection_metric":
            "tuned validation F1",

        "test_split_used":
            False,
    }

    decision_path = (
        MODELS_DIRECTORY
        / "04_decision_config.json"
    )

    decision_path.write_text(
        json.dumps(
            decision_configuration,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save research plots
    # --------------------------------------------------------

    roc_pr_path = (
        REPORTS_DIRECTORY
        / "04_validation_roc_pr.png"
    )

    confusion_matrix_path = (
        REPORTS_DIRECTORY
        / "04_validation_confusion_matrix.png"
    )

    importance_plot_path = (
        REPORTS_DIRECTORY
        / "04_top_features.png"
    )

    save_roc_pr_plot(
        validation_labels,
        logistic_probabilities,
        lightgbm_probabilities,
        roc_pr_path,
    )

    save_confusion_matrix_plot(
        validation_labels,
        selected_probabilities,
        selected_threshold,
        selected_model_name,
        confusion_matrix_path,
    )

    save_feature_importance_plot(
        feature_importance,
        importance_plot_path,
    )

    # --------------------------------------------------------
    # Save summary report
    # --------------------------------------------------------

    report = {
        "phase":
            4,

        "train_rows":
            len(train_features),

        "validation_rows":
            len(validation_features),

        "numeric_feature_count":
            len(feature_names),

        "logistic_training_seconds":
            logistic_training_seconds,

        "lightgbm_training_seconds":
            lightgbm_training_seconds,

        "lightgbm_best_iteration":
            int(
                lightgbm_model.best_iteration_
            ),

        "logistic_threshold":
            logistic_threshold,

        "lightgbm_threshold":
            lightgbm_threshold,

        "selected_model":
            selected_model_name,

        "selected_threshold":
            selected_threshold,

        "selected_validation_metrics":
            selected_metrics,

        "test_split_used":
            False,
    }

    report_path = (
        REPORTS_DIRECTORY
        / "04_training_validation_summary.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("VALIDATION RESULTS")
    print("=" * 76)

    display_columns = [
        "model",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
        "specificity",
    ]

    print(
        metrics_dataframe[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print(
        "\nSelected model: "
        f"{selected_model_name}"
    )

    print(
        "Frozen validation threshold: "
        f"{selected_threshold:.6f}"
    )

    print(
        "Test split used: NO"
    )

    print(
        "\nTop 10 LightGBM features:"
    )

    print(
        feature_importance[
            [
                "feature",
                "gain_importance_percent",
            ]
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        "\nPhase 4 files created:"
    )

    created_files = [
        logistic_model_path,
        lightgbm_model_path,
        booster_path,
        selected_bundle_path,
        decision_path,
        metrics_path,
        predictions_path,
        importance_path,
        report_path,
        roc_pr_path,
        confusion_matrix_path,
        importance_plot_path,
    ]

    for created_file in created_files:
        print(f"  {created_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())