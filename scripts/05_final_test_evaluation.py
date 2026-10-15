"""Evaluate the frozen Phase 4 model once on the untouched grouped test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
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


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.url_feature_extractor import model_feature_names  # noqa: E402


MODEL_BUNDLE_PATH = PROJECT_ROOT / "models" / "04_selected_model_bundle.joblib"
TEST_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "test_features.csv"
EXTRACTOR_PATH = PROJECT_ROOT / "src" / "url_feature_extractor.py"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
RANDOM_SEED = 42


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final evaluation of the frozen Phase 4 model on the test split."
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1_000,
        help="Number of stratified bootstrap samples for 95%% confidence intervals.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, int | float]:
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
        "mcc": matthews_corrcoef(labels, predictions),
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "brier_score": brier_score_loss(labels, probabilities),
        "log_loss": log_loss(labels, probabilities, labels=[0, 1]),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def bootstrap_confidence_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    iterations: int,
) -> pd.DataFrame:
    if iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations.")

    rng = np.random.default_rng(RANDOM_SEED)
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
        "specificity",
    )
    samples = {name: np.empty(iterations, dtype=np.float64) for name in metric_names}

    for iteration in range(iterations):
        sampled_positive = rng.choice(
            positive_indices, size=len(positive_indices), replace=True
        )
        sampled_negative = rng.choice(
            negative_indices, size=len(negative_indices), replace=True
        )
        sampled_indices = np.concatenate((sampled_negative, sampled_positive))
        sampled_labels = labels[sampled_indices]
        sampled_probabilities = probabilities[sampled_indices]
        sampled_predictions = (
            sampled_probabilities >= threshold
        ).astype(np.int8)

        tn, fp, fn, tp = confusion_matrix(
            sampled_labels, sampled_predictions, labels=[0, 1]
        ).ravel()
        samples["accuracy"][iteration] = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        samples["balanced_accuracy"][iteration] = (sensitivity + specificity) / 2
        samples["precision"][iteration] = precision
        samples["recall"][iteration] = sensitivity
        samples["f1"][iteration] = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity)
            else 0.0
        )
        samples["roc_auc"][iteration] = roc_auc_score(
            sampled_labels, sampled_probabilities
        )
        samples["pr_auc"][iteration] = average_precision_score(
            sampled_labels, sampled_probabilities
        )
        samples["mcc"][iteration] = matthews_corrcoef(
            sampled_labels, sampled_predictions
        )
        samples["specificity"][iteration] = specificity

        if iteration and iteration % 250 == 0:
            print(f"  Bootstrap completed: {iteration:,}/{iterations:,}")

    point_metrics = calculate_metrics(labels, probabilities, threshold)
    rows: list[dict[str, float | int | str]] = []
    for name in metric_names:
        lower, upper = np.percentile(samples[name], [2.5, 97.5])
        rows.append(
            {
                "metric": name,
                "point_estimate": float(point_metrics[name]),
                "ci_lower_95": float(lower),
                "ci_upper_95": float(upper),
                "bootstrap_iterations": iterations,
            }
        )
    return pd.DataFrame(rows)


def save_roc_pr_plot(
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    axes[0].plot(
        fpr,
        tpr,
        color="#e45c2a",
        label=f"ROC-AUC = {roc_auc_score(labels, probabilities):.4f}",
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set(
        title="Frozen Model: Final Test ROC Curve",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    axes[1].plot(
        recall,
        precision,
        color="#2f6fdb",
        label=f"PR-AUC = {average_precision_score(labels, probabilities):.4f}",
    )
    axes[1].set(
        title="Frozen Model: Final Test Precision-Recall Curve",
        xlabel="Recall",
        ylabel="Precision",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_confusion_plot(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
    output_path: Path,
) -> None:
    predictions = (probabilities >= threshold).astype(np.int8)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        ax=axis,
    )
    axis.set(
        title=f"{model_name} Final Test Confusion Matrix\nFrozen threshold={threshold:.4f}",
        xlabel="Predicted class",
        ylabel="True class",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_probability_plot(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(
        probabilities[labels == 0],
        bins=50,
        alpha=0.65,
        color="#2f6fdb",
        label="True legitimate",
    )
    axis.hist(
        probabilities[labels == 1],
        bins=50,
        alpha=0.65,
        color="#e45c2a",
        label="True phishing",
    )
    axis.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Frozen threshold = {threshold:.4f}",
    )
    axis.set(
        title="Final Test Phishing-Probability Distribution",
        xlabel="Predicted phishing probability",
        ylabel="Number of URLs",
        yscale="log",
    )
    axis.grid(alpha=0.20)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    arguments = parse_arguments()
    print("=" * 76)
    print("IT22131010 - Phase 5 frozen-model final test evaluation")
    print("=" * 76)
    print("Protections: no training, no threshold tuning, and no model selection.")

    if arguments.bootstrap_iterations < 100:
        print("ERROR: --bootstrap-iterations must be at least 100.")
        return 1

    for required_path in (MODEL_BUNDLE_PATH, TEST_FEATURES_PATH, EXTRACTOR_PATH):
        if not required_path.is_file():
            print(f"ERROR: Required file is missing: {required_path}")
            return 1

    REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("Loading the frozen Phase 4 model bundle...")
    model_bundle_sha256 = sha256_file(MODEL_BUNDLE_PATH)
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    required_bundle_keys = {
        "model_name",
        "model",
        "threshold",
        "feature_names",
        "feature_schema_sha256",
        "extractor_sha256",
        "test_split_used",
    }
    missing_bundle_keys = sorted(required_bundle_keys.difference(bundle))
    if missing_bundle_keys:
        print(f"ERROR: Model bundle is missing keys: {missing_bundle_keys}")
        return 1
    if bundle["test_split_used"] is not False:
        print("ERROR: The model bundle does not confirm an untouched test split.")
        return 1

    feature_names = list(bundle["feature_names"])
    current_feature_names = model_feature_names()
    current_schema_hash = hashlib.sha256(
        "\n".join(current_feature_names).encode("utf-8")
    ).hexdigest()
    if feature_names != current_feature_names:
        print("ERROR: Current feature order does not match the frozen model bundle.")
        return 1
    if current_schema_hash != bundle["feature_schema_sha256"]:
        print("ERROR: Feature-schema hash does not match the frozen model bundle.")
        return 1
    current_extractor_hash = sha256_file(EXTRACTOR_PATH)
    if current_extractor_hash != bundle["extractor_sha256"]:
        print("ERROR: Feature extractor changed after training. Evaluation stopped.")
        return 1

    threshold = float(bundle["threshold"])
    if not 0.0 < threshold < 1.0:
        print(f"ERROR: Invalid frozen threshold: {threshold}")
        return 1

    metadata_columns = [
        "source_row_id",
        "URL",
        "dataset_label",
        "is_phishing",
        "class_name",
        "hostname",
        "registered_domain",
        "fold_id",
        "nearest_trusted_domain",
    ]
    available_columns = set(pd.read_csv(TEST_FEATURES_PATH, nrows=0).columns)
    required_columns = set(metadata_columns + feature_names)
    missing_columns = sorted(required_columns.difference(available_columns))
    if missing_columns:
        print(f"ERROR: Test feature file is missing columns: {missing_columns}")
        return 1

    print("Reading the untouched grouped test features for the first final evaluation...")
    test_features_sha256 = sha256_file(TEST_FEATURES_PATH)
    test = pd.read_csv(
        TEST_FEATURES_PATH,
        usecols=metadata_columns + feature_names,
    )
    test_x = test[feature_names].astype(np.float32)
    test_y = test["is_phishing"].to_numpy(dtype=np.int8)

    if set(np.unique(test_y)) != {0, 1}:
        print("ERROR: Test labels must contain both 0=legitimate and 1=phishing.")
        return 1
    if test_x.isna().any().any():
        print("ERROR: NaN model features were found in the test split.")
        return 1
    if not np.isfinite(test_x.to_numpy()).all():
        print("ERROR: Infinite model features were found in the test split.")
        return 1

    model = bundle["model"]
    print(f"Model: {bundle['model_name']}")
    print(f"Frozen threshold: {threshold:.6f}")
    print(f"Numeric features: {len(feature_names)}")
    print(f"Test rows: {len(test):,}")
    print(f"Test phishing: {int(test_y.sum()):,}")
    print(f"Test legitimate: {int((test_y == 0).sum()):,}")
    print("Generating predictions without fitting or tuning...")
    probabilities = model.predict_proba(test_x)[:, 1]
    predictions = (probabilities >= threshold).astype(np.int8)
    metrics = calculate_metrics(test_y, probabilities, threshold)

    metrics_row = {
        "model": bundle["model_name"],
        "evaluation_split": "untouched_grouped_test",
        **metrics,
    }
    metrics_frame = pd.DataFrame([metrics_row])
    metrics_path = RESULTS_DIRECTORY / "05_final_test_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)

    predictions_frame = test[metadata_columns].copy()
    predictions_frame["phishing_probability"] = probabilities
    predictions_frame["frozen_threshold"] = threshold
    predictions_frame["predicted_is_phishing"] = predictions
    predictions_frame["predicted_class"] = np.where(
        predictions == 1, "phishing", "legitimate"
    )
    predictions_frame["is_correct"] = (predictions == test_y).astype(np.int8)
    predictions_path = RESULTS_DIRECTORY / "05_test_predictions.csv"
    predictions_frame.to_csv(predictions_path, index=False)

    print(
        f"Calculating stratified 95% confidence intervals "
        f"({arguments.bootstrap_iterations:,} bootstrap samples)..."
    )
    confidence_intervals = bootstrap_confidence_intervals(
        test_y,
        probabilities,
        threshold,
        arguments.bootstrap_iterations,
    )
    confidence_path = (
        RESULTS_DIRECTORY / "05_test_bootstrap_confidence_intervals.csv"
    )
    confidence_intervals.to_csv(confidence_path, index=False)

    error_columns = [
        "source_row_id",
        "URL",
        "is_phishing",
        "class_name",
        "hostname",
        "registered_domain",
        "nearest_trusted_domain",
        "phishing_probability",
        "predicted_is_phishing",
        "predicted_class",
    ]
    errors = predictions_frame.loc[
        predictions_frame["is_correct"] == 0, error_columns
    ].copy()
    errors["error_type"] = np.where(
        errors["is_phishing"] == 0,
        "false_positive",
        "false_negative",
    )
    errors["wrong_prediction_confidence"] = np.where(
        errors["predicted_is_phishing"] == 1,
        errors["phishing_probability"],
        1.0 - errors["phishing_probability"],
    )
    errors = errors.sort_values(
        ["error_type", "wrong_prediction_confidence"],
        ascending=[True, False],
    )
    errors_path = RESULTS_DIRECTORY / "05_misclassified_examples.csv"
    errors.to_csv(errors_path, index=False)

    roc_pr_path = REPORTS_DIRECTORY / "05_test_roc_pr.png"
    confusion_path = REPORTS_DIRECTORY / "05_test_confusion_matrix.png"
    probability_path = REPORTS_DIRECTORY / "05_test_probability_distribution.png"
    save_roc_pr_plot(test_y, probabilities, roc_pr_path)
    save_confusion_plot(
        test_y,
        probabilities,
        threshold,
        str(bundle["model_name"]),
        confusion_path,
    )
    save_probability_plot(test_y, probabilities, threshold, probability_path)

    confidence_records = confidence_intervals.to_dict(orient="records")
    report = {
        "phase": 5,
        "component": "IT22131010 Domain & Typosquatting Detection",
        "evaluation_type": "one-time frozen-model test evaluation",
        "model_name": bundle["model_name"],
        "frozen_threshold": threshold,
        "numeric_feature_count": len(feature_names),
        "test_rows": len(test),
        "test_phishing": int(test_y.sum()),
        "test_legitimate": int((test_y == 0).sum()),
        "metrics": metrics,
        "bootstrap_confidence_intervals_95": confidence_records,
        "bootstrap_iterations": arguments.bootstrap_iterations,
        "integrity": {
            "model_bundle_sha256": model_bundle_sha256,
            "test_features_sha256": test_features_sha256,
            "feature_schema_sha256": current_schema_hash,
            "extractor_sha256": current_extractor_hash,
        },
        "evaluation_controls": {
            "model_fitted_on_test": False,
            "model_selected_on_test": False,
            "threshold_tuned_on_test": False,
            "test_predictions_used_to_change_model": False,
        },
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = REPORTS_DIRECTORY / "05_final_test_summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    interval_lookup = confidence_intervals.set_index("metric")
    summary_lines = [
        "IT22131010 - FINAL FROZEN-MODEL TEST RESULTS",
        "=" * 60,
        f"Model: {bundle['model_name']}",
        f"Frozen threshold: {threshold:.6f}",
        f"Test rows: {len(test):,}",
        f"Phishing rows: {int(test_y.sum()):,}",
        f"Legitimate rows: {int((test_y == 0).sum()):,}",
        "",
        "Final metrics with stratified bootstrap 95% confidence intervals:",
    ]
    for metric_name in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
        "specificity",
    ):
        row = interval_lookup.loc[metric_name]
        summary_lines.append(
            f"  {metric_name}: {row['point_estimate']:.6f} "
            f"[{row['ci_lower_95']:.6f}, {row['ci_upper_95']:.6f}]"
        )
    summary_lines.extend(
        [
            "",
            "Confusion matrix counts:",
            f"  True legitimate: {metrics['true_negative']:,}",
            f"  False phishing alarms: {metrics['false_positive']:,}",
            f"  Missed phishing URLs: {metrics['false_negative']:,}",
            f"  Correct phishing detections: {metrics['true_positive']:,}",
            "",
            "Evaluation controls:",
            "  Model retrained on test: NO",
            "  Model selected on test: NO",
            "  Threshold tuned on test: NO",
            "  Test results used to change the model: NO",
        ]
    )
    summary_text_path = REPORTS_DIRECTORY / "05_final_test_summary.txt"
    summary_text_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 76)
    print("FINAL UNTOUCHED TEST RESULTS")
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
        metrics_frame[display_columns].to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        )
    )
    print("\nConfusion matrix:")
    print(f"  True legitimate: {metrics['true_negative']:,}")
    print(f"  False phishing alarms: {metrics['false_positive']:,}")
    print(f"  Missed phishing URLs: {metrics['false_negative']:,}")
    print(f"  Correct phishing detections: {metrics['true_positive']:,}")
    print(f"\nMisclassified URLs recorded for analysis: {len(errors):,}")
    print("Model retrained on test: NO")
    print("Threshold tuned on test: NO")
    print("\nPhase 5 files created:")
    for path in (
        metrics_path,
        predictions_path,
        confidence_path,
        errors_path,
        report_path,
        summary_text_path,
        roc_pr_path,
        confusion_path,
        probability_path,
    ):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
