#Evaluates the final saved component using the locked test set
"""One-time final evaluation of the frozen corrected domain hybrid.

This script extracts the locked Phase 8 test features, loads the frozen Phase
10 bundle, and evaluates it without fitting, model selection, ensemble-weight
selection, or threshold tuning. URLs are parsed offline and never visited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.final_domain_feature_extractor import (  # noqa: E402
    all_numeric_feature_names,
    extract_final_domain_features,
    model_feature_names,
)


EXPERIMENT_DIRECTORY = PROJECT_ROOT / "data" / "final_domain_experiment"
SPLITS_DIRECTORY = EXPERIMENT_DIRECTORY / "splits"
PROCESSED_DIRECTORY = EXPERIMENT_DIRECTORY / "processed"
TEST_SPLIT_PATH = SPLITS_DIRECTORY / "test_urls.csv"
TEST_FEATURE_PATH = PROCESSED_DIRECTORY / "test_domain_features.csv"
BUNDLE_PATH = PROJECT_ROOT / "models" / "10_corrected_domain_hybrid_bundle.joblib"
VALIDATION_METRICS_PATH = (
    PROJECT_ROOT / "results" / "10_corrected_validation_metrics.csv"
)
MODELS_DIRECTORY = PROJECT_ROOT / "models"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
FINAL_REPORT_PATH = REPORTS_DIRECTORY / "11_final_test_summary.json"

EXPECTED_TEST_ROWS = 35_105
RANDOM_SEED = 2026
DEFAULT_BOOTSTRAP_SAMPLES = 1_000
METADATA_COLUMNS = [
    "source_row_id",
    "URL",
    "dataset_label",
    "is_phishing",
    "class_name",
    "hostname",
    "registered_domain",
    "final_fold_id",
]
EXPLANATION_COLUMNS = [
    "normalized_hostname",
    "normalized_registered_domain",
    "domain_label",
    "candidate_tld",
    "normalized_subdomain",
    "nearest_trusted_domain",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Perform the one-time frozen-model test evaluation. No training "
            "or tuning is performed."
        )
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=f"Stratified bootstrap samples (default: {DEFAULT_BOOTSTRAP_SAMPLES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Bootstrap random seed (default: {RANDOM_SEED})",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_atomically(frame: pd.DataFrame, destination: Path) -> None:
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(destination)


def extract_locked_test_features() -> tuple[pd.DataFrame, float, str]:
    if not TEST_SPLIT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing locked Phase 8 test split: {TEST_SPLIT_PATH}"
        )
    source_hash = sha256_file(TEST_SPLIT_PATH)
    source = pd.read_csv(TEST_SPLIT_PATH, low_memory=False)
    if len(source) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Locked test has {len(source):,} rows; expected {EXPECTED_TEST_ROWS:,}."
        )
    missing = sorted(set(METADATA_COLUMNS).difference(source.columns))
    if missing:
        raise ValueError(f"Locked test split is missing columns: {missing}")
    official_mapping = (source["dataset_label"].astype(int) == 0).astype(int)
    if not np.array_equal(
        official_mapping.to_numpy(), source["is_phishing"].astype(int).to_numpy()
    ):
        raise ValueError("Official label mapping mismatch in locked test split")

    print("Extracting corrected features from the locked test split offline...")
    started = time.perf_counter()
    records: list[dict[str, int | float | str]] = []
    for row_number, url in enumerate(source["URL"].tolist(), start=1):
        records.append(extract_final_domain_features(url))
        if row_number % 25_000 == 0 or row_number == len(source):
            print(f"  Completed {row_number:,}/{len(source):,}")

    extracted = pd.DataFrame.from_records(records)
    expected_columns = [*EXPLANATION_COLUMNS, *all_numeric_feature_names()]
    if extracted.columns.tolist() != expected_columns:
        raise RuntimeError("Final test feature schema/order changed unexpectedly")
    numeric = extracted[all_numeric_feature_names()]
    if numeric.isna().any().any():
        raise RuntimeError("NaN numeric feature detected in locked test")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Infinite numeric feature detected in locked test")

    comparable_mask = extracted["hostname_is_ip"].eq(0)
    if comparable_mask.any():
        match_rate = float(
            (
                source.loc[comparable_mask, "registered_domain"]
                .astype(str)
                .str.lower()
                .to_numpy()
                == extracted.loc[
                    comparable_mask, "normalized_registered_domain"
                ]
                .astype(str)
                .to_numpy()
            ).mean()
        )
        if match_rate < 0.999:
            raise RuntimeError(
                "Unexpected registered-domain normalization mismatch: "
                f"{match_rate:.6f}"
            )

    combined = pd.concat(
        [source[METADATA_COLUMNS].reset_index(drop=True), extracted], axis=1
    )
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_csv_atomically(combined, TEST_FEATURE_PATH)
    elapsed = time.perf_counter() - started
    return combined, elapsed, source_hash


def classification_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    model_name: str,
    threshold: float,
    probabilities: np.ndarray | None = None,
) -> dict[str, int | float | str]:
    tn, fp, fn, tp = confusion_matrix(
        true_labels, predictions, labels=[0, 1]
    ).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    result: dict[str, int | float | str] = {
        "model": model_name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "precision": float(
            precision_score(true_labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(true_labels, predictions, zero_division=0)),
        "f1": float(f1_score(true_labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(true_labels, predictions)),
        "specificity": specificity,
        "false_positive_rate": float(1.0 - specificity),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if probabilities is not None:
        result["roc_auc"] = float(roc_auc_score(true_labels, probabilities))
        result["pr_auc"] = float(
            average_precision_score(true_labels, probabilities)
        )
    return result


def deterministic_component_decision(
    features: dict[str, object],
    probability: float,
    threshold: float,
    strong_rule_features: tuple[str, ...],
) -> tuple[int, str, str]:
    if int(features["exact_trusted_domain"]) == 1:
        return 0, "trusted_allowlist", "exact_trusted_domain"

    triggered = [
        name for name in strong_rule_features if int(features[name]) == 1
    ]
    if int(features["mixed_script"]) == 1 and "mixed_script" not in triggered:
        triggered.append("mixed_script")
    if triggered:
        return (
            1,
            "deterministic_typosquatting_evidence",
            "|".join(triggered),
        )
    return int(probability >= threshold), "frozen_ml_hybrid", ""


def controlled_robustness_test(
    bundle: dict[str, object],
) -> pd.DataFrame:
    cases = [
        ("legitimate_trusted", "https://www.kayak.com/", 0),
        ("wrong_tld_digit", "https://www.kayak.c0m/", 1),
        ("digit_letter", "https://paypa1.com/login", 1),
        ("transposition", "https://microsfot.com/", 1),
        ("added_word", "https://paypal-login.com/", 1),
        ("misleading_subdomain", "https://paypal.com.attacker.net/login", 1),
        ("keyboard_neighbor", "https://goohle.com/", 1),
        ("unicode_homoglyph", "https://раypal.com/", 1),
        ("hyphen_insertion", "https://kay-ak.com/", 1),
        ("missing_dot", "https://kayakcom/", 1),
    ]
    tabular_model = bundle["tabular_model"]
    vectorizer = bundle["character_vectorizer"]
    character_model = bundle["character_model"]
    tabular_weight = float(bundle["tabular_weight"])
    character_weight = float(bundle["character_weight"])
    threshold = float(bundle["threshold"])
    strong_rules = tuple(str(value) for value in bundle["strong_rule_features"])

    rows: list[dict[str, object]] = []
    for case_name, url, expected in cases:
        features = extract_final_domain_features(url)
        tabular_x = pd.DataFrame(
            [[features[name] for name in model_feature_names()]],
            columns=model_feature_names(),
        )
        tabular_probability = float(tabular_model.predict_proba(tabular_x)[0, 1])
        character_x = vectorizer.transform([str(features["domain_label"])])
        character_probability = float(
            character_model.predict_proba(character_x)[0, 1]
        )
        probability = (
            tabular_weight * tabular_probability
            + character_weight * character_probability
        )
        prediction, source, rules = deterministic_component_decision(
            features, probability, threshold, strong_rules
        )
        rows.append(
            {
                "case": case_name,
                "url": url,
                "expected_prediction": expected,
                "prediction": prediction,
                "accepted": prediction == expected,
                "decision_source": source,
                "triggered_rules": rules,
                "hybrid_probability": probability,
            }
        )
    result = pd.DataFrame(rows)
    if not result["accepted"].all():
        failures = result.loc[~result["accepted"], ["case", "url"]]
        raise RuntimeError(
            "Frozen controlled robustness failure:\n"
            + failures.to_string(index=False)
        )
    return result


def bootstrap_confidence_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    number_of_samples: int,
    seed: int,
) -> pd.DataFrame:
    negative_indices = np.flatnonzero(labels == 0)
    positive_indices = np.flatnonzero(labels == 1)
    random_generator = np.random.default_rng(seed)
    metric_names = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
        "specificity",
    )
    bootstrap_values = {name: [] for name in metric_names}

    for sample_number in range(1, number_of_samples + 1):
        sampled_indices = np.concatenate(
            [
                random_generator.choice(
                    negative_indices, size=len(negative_indices), replace=True
                ),
                random_generator.choice(
                    positive_indices, size=len(positive_indices), replace=True
                ),
            ]
        )
        sampled_labels = labels[sampled_indices]
        sampled_probabilities = probabilities[sampled_indices]
        sampled_predictions = (sampled_probabilities >= threshold).astype(np.int8)
        metrics = classification_metrics(
            sampled_labels,
            sampled_predictions,
            "bootstrap",
            threshold,
            sampled_probabilities,
        )
        for metric_name in metric_names:
            bootstrap_values[metric_name].append(float(metrics[metric_name]))
        if sample_number % 250 == 0 or sample_number == number_of_samples:
            print(
                f"  Bootstrap completed: {sample_number:,}/{number_of_samples:,}"
            )

    point_predictions = (probabilities >= threshold).astype(np.int8)
    point_metrics = classification_metrics(
        labels,
        point_predictions,
        "frozen_ml_hybrid",
        threshold,
        probabilities,
    )
    rows = []
    for metric_name in metric_names:
        values = np.asarray(bootstrap_values[metric_name], dtype=np.float64)
        rows.append(
            {
                "metric": metric_name,
                "point_estimate": float(point_metrics[metric_name]),
                "ci_95_lower": float(np.quantile(values, 0.025)),
                "ci_95_upper": float(np.quantile(values, 0.975)),
                "bootstrap_samples": number_of_samples,
                "random_seed": seed,
            }
        )
    return pd.DataFrame(rows)


def save_roc_pr_plot(
    labels: np.ndarray,
    probabilities: np.ndarray,
    destination: Path,
) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    roc_auc = roc_auc_score(labels, probabilities)
    pr_auc = average_precision_score(labels, probabilities)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(
        false_positive_rate,
        true_positive_rate,
        color="#1f77b4",
        label=f"ROC-AUC = {roc_auc:.4f}",
    )
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title("Frozen Hybrid - Final Test ROC")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend(loc="lower right")
    axes[1].plot(
        recall,
        precision,
        color="#d62728",
        label=f"PR-AUC = {pr_auc:.4f}",
    )
    axes[1].set_title("Frozen Hybrid - Final Test Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_confusion_comparison(
    labels: np.ndarray,
    ml_predictions: np.ndarray,
    component_predictions: np.ndarray,
    destination: Path,
) -> None:
    matrices = [
        confusion_matrix(labels, ml_predictions, labels=[0, 1]),
        confusion_matrix(labels, component_predictions, labels=[0, 1]),
    ]
    titles = ["Frozen ML hybrid", "Full component with rules"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, matrix, title in zip(axes, matrices, titles):
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                    fontsize=11,
                )
        axis.set_xticks([0, 1], labels=["Legitimate", "Phishing"])
        axis.set_yticks([0, 1], labels=["Legitimate", "Phishing"])
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_probability_distribution(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0.0, 1.0, 51)
    axis.hist(
        probabilities[labels == 0],
        bins=bins,
        alpha=0.65,
        label="Legitimate",
        color="#2ca02c",
    )
    axis.hist(
        probabilities[labels == 1],
        bins=bins,
        alpha=0.65,
        label="Phishing",
        color="#d62728",
    )
    axis.axvline(
        threshold,
        color="black",
        linestyle="--",
        label=f"Frozen threshold = {threshold:.4f}",
    )
    axis.set_xlabel("Frozen hybrid phishing probability")
    axis.set_ylabel("URL count")
    axis.set_title("Final Test Probability Distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    arguments = parse_arguments()
    if arguments.bootstrap_samples < 100:
        print("ERROR: --bootstrap-samples must be at least 100")
        return 1
    if FINAL_REPORT_PATH.exists():
        print("ERROR: Phase 11 final evaluation has already been completed.")
        print(f"Existing final report: {FINAL_REPORT_PATH}")
        print("The locked test must not be repeatedly used for model development.")
        return 1

    print("=" * 76)
    print("IT22131010 - Phase 11 one-time frozen final test evaluation")
    print("=" * 76)
    print("Protections: no fitting, no weight selection, no threshold tuning")
    print("Safety: offline domain-string analysis only; no URL is visited")
    print("Loading the frozen Phase 10 hybrid bundle...")

    if not BUNDLE_PATH.is_file():
        print(f"ERROR: Frozen Phase 10 bundle was not found: {BUNDLE_PATH}")
        return 1
    bundle_hash_before = sha256_file(BUNDLE_PATH)
    bundle = joblib.load(BUNDLE_PATH)
    required_bundle_keys = {
        "tabular_model",
        "character_vectorizer",
        "character_model",
        "model_feature_names",
        "tabular_weight",
        "character_weight",
        "threshold",
        "strong_rule_features",
    }
    missing_bundle_keys = sorted(required_bundle_keys.difference(bundle))
    if missing_bundle_keys:
        raise ValueError(f"Frozen model bundle is missing: {missing_bundle_keys}")
    if list(bundle["model_feature_names"]) != model_feature_names():
        raise ValueError("Frozen model feature schema does not match the extractor")

    tabular_weight = float(bundle["tabular_weight"])
    character_weight = float(bundle["character_weight"])
    threshold = float(bundle["threshold"])
    if not np.isclose(tabular_weight + character_weight, 1.0):
        raise ValueError("Frozen ensemble weights do not sum to one")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Frozen threshold is outside [0, 1]")

    print(f"Frozen LightGBM weight: {tabular_weight:.2f}")
    print(f"Frozen character weight: {character_weight:.2f}")
    print(f"Frozen threshold: {threshold:.6f}")
    print(f"Model features: {len(model_feature_names())}")
    print("Opening the locked grouped test split for its first final evaluation...")

    REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    test, extraction_seconds, test_hash_before = extract_locked_test_features()
    labels = test["is_phishing"].astype(np.int8).to_numpy()
    print(f"Test rows: {len(test):,}")
    print(f"Test phishing: {int(labels.sum()):,}")
    print(f"Test legitimate: {int((labels == 0).sum()):,}")
    print(f"Feature extraction time: {extraction_seconds:.2f} seconds")

    print("Generating frozen predictions without fitting or tuning...")
    tabular_model = bundle["tabular_model"]
    vectorizer = bundle["character_vectorizer"]
    character_model = bundle["character_model"]
    tabular_probability = tabular_model.predict_proba(
        test[model_feature_names()]
    )[:, 1]
    character_x = vectorizer.transform(test["domain_label"].fillna("").astype(str))
    character_probability = character_model.predict_proba(character_x)[:, 1]
    hybrid_probability = (
        tabular_weight * tabular_probability
        + character_weight * character_probability
    )
    ml_prediction = (hybrid_probability >= threshold).astype(np.int8)

    strong_rules = tuple(str(value) for value in bundle["strong_rule_features"])
    component_predictions: list[int] = []
    decision_sources: list[str] = []
    triggered_rules: list[str] = []
    feature_records = test.to_dict(orient="records")
    for features, probability in zip(feature_records, hybrid_probability):
        prediction, source, rules = deterministic_component_decision(
            features, float(probability), threshold, strong_rules
        )
        component_predictions.append(prediction)
        decision_sources.append(source)
        triggered_rules.append(rules)
    component_prediction = np.asarray(component_predictions, dtype=np.int8)

    ml_metrics = classification_metrics(
        labels,
        ml_prediction,
        "Frozen corrected ML hybrid",
        threshold,
        hybrid_probability,
    )
    component_metrics = classification_metrics(
        labels,
        component_prediction,
        "Full component with deterministic evidence",
        threshold,
        probabilities=None,
    )
    metrics_frame = pd.DataFrame([ml_metrics, component_metrics])

    print(
        f"Calculating stratified 95% confidence intervals "
        f"({arguments.bootstrap_samples:,} bootstrap samples)..."
    )
    confidence_intervals = bootstrap_confidence_intervals(
        labels,
        hybrid_probability,
        threshold,
        arguments.bootstrap_samples,
        arguments.seed,
    )

    print("Rechecking the frozen controlled typosquatting cases...")
    controlled = controlled_robustness_test(bundle)

    predictions = test[
        [
            "source_row_id",
            "URL",
            "registered_domain",
            "domain_label",
            "nearest_trusted_domain",
            "dataset_label",
            "is_phishing",
        ]
    ].copy()
    predictions["tabular_probability"] = tabular_probability
    predictions["character_probability"] = character_probability
    predictions["hybrid_probability"] = hybrid_probability
    predictions["frozen_threshold"] = threshold
    predictions["ml_prediction"] = ml_prediction
    predictions["component_prediction"] = component_prediction
    predictions["decision_source"] = decision_sources
    predictions["triggered_rules"] = triggered_rules
    predictions["ml_correct"] = predictions["ml_prediction"].eq(
        predictions["is_phishing"]
    )
    predictions["component_correct"] = predictions["component_prediction"].eq(
        predictions["is_phishing"]
    )
    predictions["prediction_changed_by_rules"] = predictions[
        "ml_prediction"
    ].ne(predictions["component_prediction"])
    predictions["rule_effect"] = np.select(
        [
            predictions["prediction_changed_by_rules"]
            & ~predictions["ml_correct"]
            & predictions["component_correct"],
            predictions["prediction_changed_by_rules"]
            & predictions["ml_correct"]
            & ~predictions["component_correct"],
            predictions["prediction_changed_by_rules"],
        ],
        ["corrected_ml_error", "introduced_error", "changed_same_correctness"],
        default="unchanged",
    )

    rule_impact = (
        predictions.groupby(["decision_source", "rule_effect"], dropna=False)
        .size()
        .rename("row_count")
        .reset_index()
    )
    misclassified = predictions.loc[
        ~predictions["ml_correct"] | ~predictions["component_correct"]
    ].copy()
    misclassified["misclassification_scope"] = np.select(
        [
            ~misclassified["ml_correct"] & ~misclassified["component_correct"],
            ~misclassified["ml_correct"] & misclassified["component_correct"],
            misclassified["ml_correct"] & ~misclassified["component_correct"],
        ],
        ["both", "ml_only", "component_only"],
        default="none",
    )

    comparison = pd.DataFrame()
    if VALIDATION_METRICS_PATH.is_file():
        validation_metrics = pd.read_csv(VALIDATION_METRICS_PATH)
        selected_validation = validation_metrics.loc[
            validation_metrics["model"].eq(
                "Corrected hybrid - tuned threshold"
            )
        ]
        if len(selected_validation) == 1:
            rows = []
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
                validation_value = float(selected_validation.iloc[0][metric_name])
                test_value = float(ml_metrics[metric_name])
                rows.append(
                    {
                        "metric": metric_name,
                        "validation": validation_value,
                        "test": test_value,
                        "validation_minus_test": validation_value - test_value,
                    }
                )
            comparison = pd.DataFrame(rows)

    metrics_path = RESULTS_DIRECTORY / "11_final_test_metrics.csv"
    predictions_path = RESULTS_DIRECTORY / "11_final_test_predictions.csv"
    confidence_path = RESULTS_DIRECTORY / "11_test_bootstrap_confidence_intervals.csv"
    misclassified_path = RESULTS_DIRECTORY / "11_final_misclassified_examples.csv"
    rule_impact_path = RESULTS_DIRECTORY / "11_rule_impact.csv"
    controlled_path = RESULTS_DIRECTORY / "11_controlled_robustness.csv"
    comparison_path = RESULTS_DIRECTORY / "11_validation_test_comparison.csv"
    write_csv_atomically(metrics_frame, metrics_path)
    write_csv_atomically(predictions, predictions_path)
    write_csv_atomically(confidence_intervals, confidence_path)
    write_csv_atomically(misclassified, misclassified_path)
    write_csv_atomically(rule_impact, rule_impact_path)
    write_csv_atomically(controlled, controlled_path)
    if not comparison.empty:
        write_csv_atomically(comparison, comparison_path)

    roc_pr_path = REPORTS_DIRECTORY / "11_final_test_roc_pr.png"
    confusion_path = REPORTS_DIRECTORY / "11_final_test_confusion_comparison.png"
    distribution_path = REPORTS_DIRECTORY / "11_final_probability_distribution.png"
    save_roc_pr_plot(labels, hybrid_probability, roc_pr_path)
    save_confusion_comparison(
        labels, ml_prediction, component_prediction, confusion_path
    )
    save_probability_distribution(
        labels, hybrid_probability, threshold, distribution_path
    )

    bundle_hash_after = sha256_file(BUNDLE_PATH)
    test_hash_after = sha256_file(TEST_SPLIT_PATH)
    bundle_unchanged = bundle_hash_before == bundle_hash_after
    test_unchanged = test_hash_before == test_hash_after
    if not bundle_unchanged or not test_unchanged:
        raise RuntimeError("Frozen model or locked source test unexpectedly changed")

    report = {
        "phase": 11,
        "evaluation": "one-time frozen final grouped test",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protections": {
            "model_fitted_on_test": False,
            "model_selected_on_test": False,
            "ensemble_weight_selected_on_test": False,
            "threshold_tuned_on_test": False,
            "test_urls_visited": False,
            "frozen_bundle_unchanged": bundle_unchanged,
            "test_source_unchanged": test_unchanged,
        },
        "test_rows": len(test),
        "test_phishing": int(labels.sum()),
        "test_legitimate": int((labels == 0).sum()),
        "test_split_sha256": test_hash_before,
        "frozen_bundle_sha256": bundle_hash_before,
        "feature_extraction_seconds": extraction_seconds,
        "model_feature_count": len(model_feature_names()),
        "tabular_weight": tabular_weight,
        "character_weight": character_weight,
        "frozen_threshold": threshold,
        "frozen_ml_metrics": ml_metrics,
        "full_component_metrics": component_metrics,
        "controlled_robustness_passed": int(controlled["accepted"].sum()),
        "controlled_robustness_total": len(controlled),
        "rule_changed_predictions": int(
            predictions["prediction_changed_by_rules"].sum()
        ),
        "rule_corrected_ml_errors": int(
            predictions["rule_effect"].eq("corrected_ml_error").sum()
        ),
        "rule_introduced_errors": int(
            predictions["rule_effect"].eq("introduced_error").sum()
        ),
        "bootstrap_samples": arguments.bootstrap_samples,
        "bootstrap_seed": arguments.seed,
    }
    FINAL_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary_text_path = REPORTS_DIRECTORY / "11_final_test_summary.txt"
    summary_lines = [
        "IT22131010 - FINAL FROZEN TEST EVALUATION",
        "=" * 64,
        "Model fitted on test: NO",
        "Model/weight selected on test: NO",
        "Threshold tuned on test: NO",
        "URLs visited: NO",
        f"Test rows: {len(test):,}",
        f"Frozen threshold: {threshold:.6f}",
        "",
        "FROZEN ML HYBRID:",
        f"  Accuracy: {float(ml_metrics['accuracy']):.6f}",
        f"  Precision: {float(ml_metrics['precision']):.6f}",
        f"  Recall: {float(ml_metrics['recall']):.6f}",
        f"  F1: {float(ml_metrics['f1']):.6f}",
        f"  ROC-AUC: {float(ml_metrics['roc_auc']):.6f}",
        f"  PR-AUC: {float(ml_metrics['pr_auc']):.6f}",
        f"  MCC: {float(ml_metrics['mcc']):.6f}",
        f"  Specificity: {float(ml_metrics['specificity']):.6f}",
        "",
        "FULL COMPONENT WITH RULES:",
        f"  Accuracy: {float(component_metrics['accuracy']):.6f}",
        f"  Precision: {float(component_metrics['precision']):.6f}",
        f"  Recall: {float(component_metrics['recall']):.6f}",
        f"  F1: {float(component_metrics['f1']):.6f}",
        f"  MCC: {float(component_metrics['mcc']):.6f}",
        f"  Specificity: {float(component_metrics['specificity']):.6f}",
        "",
        f"Controlled robustness: {int(controlled['accepted'].sum())}/{len(controlled)}",
        f"Rules corrected ML errors: {int(predictions['rule_effect'].eq('corrected_ml_error').sum())}",
        f"Rules introduced errors: {int(predictions['rule_effect'].eq('introduced_error').sum())}",
    ]
    summary_text_path.write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

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
        "mcc",
        "specificity",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    print(metrics_frame[display_columns].to_string(index=False))
    print(
        f"\nFrozen ML ROC-AUC: {float(ml_metrics['roc_auc']):.6f}"
    )
    print(f"Frozen ML PR-AUC: {float(ml_metrics['pr_auc']):.6f}")
    print(
        f"Controlled robustness: {int(controlled['accepted'].sum())}/{len(controlled)}"
    )
    print(
        "Rules changed predictions: "
        f"{int(predictions['prediction_changed_by_rules'].sum()):,}"
    )
    print(
        "Rules corrected ML errors: "
        f"{int(predictions['rule_effect'].eq('corrected_ml_error').sum()):,}"
    )
    print(
        "Rules introduced errors: "
        f"{int(predictions['rule_effect'].eq('introduced_error').sum()):,}"
    )
    print("Model retrained on test: NO")
    print("Threshold tuned on test: NO")
    print(f"Frozen model unchanged: {'YES' if bundle_unchanged else 'NO'}")
    print(f"Locked test source unchanged: {'YES' if test_unchanged else 'NO'}")

    print("\nPhase 11 files created:")
    output_paths = [
        TEST_FEATURE_PATH,
        metrics_path,
        predictions_path,
        confidence_path,
        misclassified_path,
        rule_impact_path,
        controlled_path,
    ]
    if not comparison.empty:
        output_paths.append(comparison_path)
    output_paths.extend(
        [
            FINAL_REPORT_PATH,
            summary_text_path,
            roc_pr_path,
            confusion_path,
            distribution_path,
        ]
    )
    for path in output_paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

