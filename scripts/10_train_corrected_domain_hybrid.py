#Trains LightGBM and TF-IDF Logistic Regression, selects weights and threshold, and saves the model

"""Train and validate the corrected domain/typosquatting hybrid detector.

Only the Phase 9 training and validation feature files are read. The locked
test split is deliberately inaccessible in this phase. The previous 99% pilot
model is never loaded or reused.
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
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
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
    EXCLUDED_SHORTCUT_FEATURES,
    extract_final_domain_features,
    model_feature_names,
)


EXPERIMENT_DIRECTORY = PROJECT_ROOT / "data" / "final_domain_experiment"
PROCESSED_DIRECTORY = EXPERIMENT_DIRECTORY / "processed"
TRAIN_PATH = PROCESSED_DIRECTORY / "train_domain_features.csv"
VALIDATION_PATH = PROCESSED_DIRECTORY / "validation_domain_features.csv"
MODELS_DIRECTORY = PROJECT_ROOT / "models"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"

EXPECTED_TRAIN_ROWS = 165_160
EXPECTED_VALIDATION_ROWS = 35_105
RANDOM_SEED = 2026
ENSEMBLE_WEIGHTS = tuple(round(value, 2) for value in np.linspace(0.0, 1.0, 21))

STRONG_RULE_FEATURES: tuple[str, ...] = (
    "invalid_or_wrong_tld",
    "edit_insertion",
    "edit_deletion",
    "edit_substitution",
    "edit_transposition",
    "keyboard_neighbor_substitution",
    "leet_or_digit_letter_match",
    "hyphen_insertion",
    "missing_dot_pattern",
    "added_suspicious_word",
    "misleading_subdomain_pattern",
    "dangerous_unicode_confusable",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the corrected LightGBM + domain-label character hybrid "
            "using train and validation only."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Reproducible random seed (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--max-char-features",
        type=int,
        default=75_000,
        help="Maximum TF-IDF character vocabulary size (default: 75000)",
    )
    parser.add_argument(
        "--char-min-df",
        type=int,
        default=3,
        help="Minimum document frequency for character n-grams (default: 3)",
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


def read_feature_file(path: Path, expected_rows: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Phase 9 file: {path}. Run script 09 without --limit first."
        )
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) != expected_rows:
        raise ValueError(
            f"{path.name} has {len(frame):,} rows; expected {expected_rows:,}. "
            "Run script 09 again without --limit."
        )

    required = {
        "source_row_id",
        "URL",
        "dataset_label",
        "is_phishing",
        "class_name",
        "registered_domain",
        "domain_label",
        *model_feature_names(),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    official_mapping = (frame["dataset_label"].astype(int) == 0).astype(int)
    if not np.array_equal(
        official_mapping.to_numpy(), frame["is_phishing"].astype(int).to_numpy()
    ):
        raise ValueError(f"Official label mapping mismatch in {path.name}")

    numeric = frame[model_feature_names()]
    if numeric.isna().any().any():
        raise ValueError(f"NaN model feature detected in {path.name}")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Infinite model feature detected in {path.name}")
    return frame


def training_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Balance classes and reduce domination by repeated registered domains."""
    labels = frame["is_phishing"].astype(int)
    domain_counts = frame["registered_domain"].astype(str).value_counts()
    group_component = (
        frame["registered_domain"].astype(str).map(domain_counts).pow(-0.5)
    )
    class_counts = labels.value_counts()
    class_component = labels.map(
        {
            class_label: len(labels) / (2.0 * count)
            for class_label, count in class_counts.items()
        }
    )
    weights = (group_component * class_component).to_numpy(dtype=np.float64)
    weights /= weights.mean()
    return weights


def select_f1_threshold(
    true_labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(
        true_labels, probabilities
    )
    if len(thresholds) == 0:
        return 0.5, 0.0
    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1_values = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_f1 = float(np.max(f1_values))
    candidate_indices = np.flatnonzero(np.isclose(f1_values, best_f1))
    best_index = min(
        candidate_indices,
        key=lambda index: abs(float(thresholds[index]) - 0.5),
    )
    return float(thresholds[best_index]), best_f1


def calculate_metrics(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
) -> dict[str, int | float | str]:
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(
        true_labels, predictions, labels=[0, 1]
    ).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "model": model_name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "precision": float(
            precision_score(true_labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(true_labels, predictions, zero_division=0)),
        "f1": float(f1_score(true_labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(true_labels, probabilities)),
        "pr_auc": float(average_precision_score(true_labels, probabilities)),
        "mcc": float(matthews_corrcoef(true_labels, predictions)),
        "specificity": specificity,
        "false_positive_rate": float(1.0 - specificity),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_ensemble(
    true_labels: np.ndarray,
    tabular_probabilities: np.ndarray,
    character_probabilities: np.ndarray,
) -> tuple[float, float, pd.DataFrame]:
    rows: list[dict[str, int | float | str]] = []
    for tabular_weight in ENSEMBLE_WEIGHTS:
        probabilities = (
            tabular_weight * tabular_probabilities
            + (1.0 - tabular_weight) * character_probabilities
        )
        threshold, _ = select_f1_threshold(true_labels, probabilities)
        metrics = calculate_metrics(
            true_labels,
            probabilities,
            threshold,
            f"hybrid_w_{tabular_weight:.2f}",
        )
        metrics["tabular_weight"] = tabular_weight
        metrics["character_weight"] = round(1.0 - tabular_weight, 2)
        rows.append(metrics)

    search = pd.DataFrame(rows)
    search["central_weight_preference"] = -(
        search["tabular_weight"] - 0.5
    ).abs()
    selected_index = search.sort_values(
        ["f1", "mcc", "pr_auc", "central_weight_preference"],
        ascending=[False, False, False, False],
    ).index[0]
    selected = search.loc[selected_index]
    return (
        float(selected["tabular_weight"]),
        float(selected["threshold"]),
        search.drop(columns=["central_weight_preference"]),
    )


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
    axes[0].set_title("Corrected Hybrid - Validation ROC")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend(loc="lower right")

    axes[1].plot(
        recall,
        precision,
        color="#d62728",
        label=f"PR-AUC = {pr_auc:.4f}",
    )
    axes[1].set_title("Corrected Hybrid - Validation Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_confusion_plot(
    labels: np.ndarray,
    predictions: np.ndarray,
    destination: Path,
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )
    axis.set_xticks([0, 1], labels=["Legitimate", "Phishing"])
    axis.set_yticks([0, 1], labels=["Legitimate", "Phishing"])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Corrected Hybrid - Validation Confusion Matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_feature_importance_plot(
    importance: pd.DataFrame,
    destination: Path,
) -> None:
    top = importance.head(20).sort_values("gain_importance_percent")
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(
        top["feature"],
        top["gain_importance_percent"],
        color="#2ca02c",
    )
    axis.set_xlabel("Gain importance (%)")
    axis.set_title("Corrected LightGBM - Top Domain Features")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def deterministic_rule_decision(
    features: dict[str, int | float | str],
    probability: float,
    threshold: float,
) -> tuple[int, list[str], str]:
    if int(features["exact_trusted_domain"]) == 1:
        return 0, ["exact_trusted_domain"], "trusted_allowlist"

    triggered = [
        name for name in STRONG_RULE_FEATURES if int(features[name]) == 1
    ]
    if int(features["mixed_script"]) == 1 and "mixed_script" not in triggered:
        triggered.append("mixed_script")
    if triggered:
        return 1, triggered, "deterministic_typosquatting_evidence"
    return (
        int(probability >= threshold),
        [],
        "corrected_ml_hybrid",
    )


def evaluate_controlled_robustness(
    tabular_model: lgb.LGBMClassifier,
    vectorizer: TfidfVectorizer,
    character_model: LogisticRegression,
    tabular_weight: float,
    threshold: float,
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
    rows: list[dict[str, object]] = []
    for case_name, url, expected_prediction in cases:
        features = extract_final_domain_features(url)
        tabular_frame = pd.DataFrame(
            [[features[name] for name in model_feature_names()]],
            columns=model_feature_names(),
        )
        tabular_probability = float(tabular_model.predict_proba(tabular_frame)[0, 1])
        character_matrix = vectorizer.transform([str(features["domain_label"])])
        character_probability = float(
            character_model.predict_proba(character_matrix)[0, 1]
        )
        hybrid_probability = float(
            tabular_weight * tabular_probability
            + (1.0 - tabular_weight) * character_probability
        )
        prediction, triggered_rules, decision_source = deterministic_rule_decision(
            features,
            hybrid_probability,
            threshold,
        )
        rows.append(
            {
                "case": case_name,
                "url": url,
                "expected_prediction": expected_prediction,
                "prediction": prediction,
                "accepted": prediction == expected_prediction,
                "decision_source": decision_source,
                "triggered_rules": "|".join(triggered_rules),
                "nearest_trusted_domain": features["nearest_trusted_domain"],
                "tabular_probability": tabular_probability,
                "character_probability": character_probability,
                "hybrid_probability": hybrid_probability,
            }
        )
    result = pd.DataFrame(rows)
    if not result["accepted"].all():
        failures = result.loc[~result["accepted"], ["case", "url"]]
        raise RuntimeError(
            "Controlled hybrid robustness failure:\n"
            + failures.to_string(index=False)
        )
    return result


def main() -> int:
    arguments = parse_arguments()
    if arguments.max_char_features <= 0 or arguments.char_min_df <= 0:
        print("ERROR: Character feature settings must be greater than zero.")
        return 1

    print("=" * 76)
    print("IT22131010 - Phase 10 corrected hybrid training and validation")
    print("=" * 76)
    print("Allowed data: Phase 9 train and validation only")
    print("Locked final test read: NO")
    print("Previous pilot model loaded: NO")
    print("Branches: domain LightGBM + domain-label character TF-IDF")
    print("Rules: explicit typosquatting evidence and trusted allowlist")
    print(f"Random seed: {arguments.seed}")

    MODELS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("Reading the complete corrected feature files...")
    train = read_feature_file(TRAIN_PATH, EXPECTED_TRAIN_ROWS)
    validation = read_feature_file(VALIDATION_PATH, EXPECTED_VALIDATION_ROWS)

    domain_overlap = len(
        set(train["registered_domain"].astype(str)).intersection(
            validation["registered_domain"].astype(str)
        )
    )
    if domain_overlap:
        raise RuntimeError(
            f"Registered-domain overlap detected between train/validation: {domain_overlap}"
        )

    feature_names = model_feature_names()
    train_x = train[feature_names]
    validation_x = validation[feature_names]
    train_y = train["is_phishing"].astype(np.int8).to_numpy()
    validation_y = validation["is_phishing"].astype(np.int8).to_numpy()
    sample_weights = training_sample_weights(train)

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Corrected numeric features: {len(feature_names)}")
    print(f"Train phishing: {int(train_y.sum()):,}")
    print(f"Train legitimate: {int((train_y == 0).sum()):,}")
    print(f"Registered-domain overlap: {domain_overlap}")

    print("\nTraining corrected LightGBM domain branch...")
    tabular_started = time.perf_counter()
    tabular_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2_000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.20,
        reg_lambda=0.50,
        random_state=arguments.seed,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    tabular_model.fit(
        train_x,
        train_y,
        sample_weight=sample_weights,
        eval_X=validation_x,
        eval_y=validation_y,
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
    tabular_seconds = time.perf_counter() - tabular_started
    tabular_train_probability = tabular_model.predict_proba(train_x)[:, 1]
    tabular_validation_probability = tabular_model.predict_proba(validation_x)[:, 1]
    print(f"LightGBM completed in {tabular_seconds:.2f} seconds")
    print(f"LightGBM best iteration: {tabular_model.best_iteration_}")

    print("\nTraining domain-label character TF-IDF branch...")
    character_started = time.perf_counter()
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        lowercase=True,
        min_df=arguments.char_min_df,
        max_df=0.999,
        max_features=arguments.max_char_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_character_x = vectorizer.fit_transform(
        train["domain_label"].fillna("").astype(str)
    )
    validation_character_x = vectorizer.transform(
        validation["domain_label"].fillna("").astype(str)
    )
    character_model = LogisticRegression(
        C=2.0,
        solver="liblinear",
        max_iter=300,
        random_state=arguments.seed,
    )
    character_model.fit(
        train_character_x,
        train_y,
        sample_weight=sample_weights,
    )
    character_train_probability = character_model.predict_proba(
        train_character_x
    )[:, 1]
    character_validation_probability = character_model.predict_proba(
        validation_character_x
    )[:, 1]
    character_seconds = time.perf_counter() - character_started
    print(f"Character branch completed in {character_seconds:.2f} seconds")
    print(f"Character vocabulary: {len(vectorizer.vocabulary_):,}")

    print("\nSelecting ensemble weight and threshold on validation only...")
    tabular_weight, frozen_threshold, weight_search = select_ensemble(
        validation_y,
        tabular_validation_probability,
        character_validation_probability,
    )
    character_weight = 1.0 - tabular_weight
    hybrid_train_probability = (
        tabular_weight * tabular_train_probability
        + character_weight * character_train_probability
    )
    hybrid_validation_probability = (
        tabular_weight * tabular_validation_probability
        + character_weight * character_validation_probability
    )

    tabular_threshold, _ = select_f1_threshold(
        validation_y, tabular_validation_probability
    )
    character_threshold, _ = select_f1_threshold(
        validation_y, character_validation_probability
    )
    validation_metrics = pd.DataFrame(
        [
            calculate_metrics(
                validation_y,
                tabular_validation_probability,
                0.5,
                "Corrected LightGBM - default threshold",
            ),
            calculate_metrics(
                validation_y,
                tabular_validation_probability,
                tabular_threshold,
                "Corrected LightGBM - tuned threshold",
            ),
            calculate_metrics(
                validation_y,
                character_validation_probability,
                0.5,
                "Domain-label character model - default threshold",
            ),
            calculate_metrics(
                validation_y,
                character_validation_probability,
                character_threshold,
                "Domain-label character model - tuned threshold",
            ),
            calculate_metrics(
                validation_y,
                hybrid_validation_probability,
                0.5,
                "Corrected hybrid - default threshold",
            ),
            calculate_metrics(
                validation_y,
                hybrid_validation_probability,
                frozen_threshold,
                "Corrected hybrid - tuned threshold",
            ),
        ]
    )
    train_metrics = calculate_metrics(
        train_y,
        hybrid_train_probability,
        frozen_threshold,
        "Corrected hybrid - training",
    )
    final_validation_metrics = calculate_metrics(
        validation_y,
        hybrid_validation_probability,
        frozen_threshold,
        "Corrected hybrid - validation",
    )
    gap_rows = []
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
        train_value = float(train_metrics[metric_name])
        validation_value = float(final_validation_metrics[metric_name])
        gap_rows.append(
            {
                "metric": metric_name,
                "train": train_value,
                "validation": validation_value,
                "train_minus_validation": train_value - validation_value,
            }
        )
    generalization_gap = pd.DataFrame(gap_rows)

    booster = tabular_model.booster_
    gains = booster.feature_importance(importance_type="gain")
    total_gain = float(gains.sum())
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "gain_importance": gains,
            "gain_importance_percent": (
                gains / total_gain * 100.0
                if total_gain > 0
                else np.zeros_like(gains, dtype=float)
            ),
        }
    ).sort_values("gain_importance", ascending=False, ignore_index=True)

    character_feature_names = vectorizer.get_feature_names_out()
    character_coefficients = character_model.coef_[0]
    phishing_indices = np.argsort(character_coefficients)[-30:][::-1]
    legitimate_indices = np.argsort(character_coefficients)[:30]
    character_importance = pd.DataFrame(
        [
            {
                "class_direction": "phishing",
                "character_ngram": character_feature_names[index],
                "coefficient": float(character_coefficients[index]),
            }
            for index in phishing_indices
        ]
        + [
            {
                "class_direction": "legitimate",
                "character_ngram": character_feature_names[index],
                "coefficient": float(character_coefficients[index]),
            }
            for index in legitimate_indices
        ]
    )

    print("Running controlled hybrid robustness evaluation...")
    controlled = evaluate_controlled_robustness(
        tabular_model,
        vectorizer,
        character_model,
        tabular_weight,
        frozen_threshold,
    )

    validation_predictions = validation[
        [
            "source_row_id",
            "URL",
            "registered_domain",
            "domain_label",
            "dataset_label",
            "is_phishing",
        ]
    ].copy()
    validation_predictions["tabular_probability"] = (
        tabular_validation_probability
    )
    validation_predictions["character_probability"] = (
        character_validation_probability
    )
    validation_predictions["hybrid_probability"] = (
        hybrid_validation_probability
    )
    validation_predictions["frozen_threshold"] = frozen_threshold
    validation_predictions["prediction"] = (
        hybrid_validation_probability >= frozen_threshold
    ).astype(np.int8)
    validation_predictions["correct"] = (
        validation_predictions["prediction"]
        == validation_predictions["is_phishing"]
    )

    bundle_path = MODELS_DIRECTORY / "10_corrected_domain_hybrid_bundle.joblib"
    tabular_model_path = MODELS_DIRECTORY / "10_corrected_lightgbm.joblib"
    booster_path = MODELS_DIRECTORY / "10_corrected_lightgbm_booster.txt"
    vectorizer_path = MODELS_DIRECTORY / "10_domain_char_vectorizer.joblib"
    character_model_path = MODELS_DIRECTORY / "10_domain_char_model.joblib"
    decision_path = MODELS_DIRECTORY / "10_corrected_decision_config.json"

    joblib.dump(tabular_model, tabular_model_path, compress=3)
    booster.save_model(str(booster_path))
    joblib.dump(vectorizer, vectorizer_path, compress=3)
    joblib.dump(character_model, character_model_path, compress=3)

    bundle = {
        "phase": 10,
        "model_name": "corrected_domain_hybrid",
        "tabular_model": tabular_model,
        "character_vectorizer": vectorizer,
        "character_model": character_model,
        "model_feature_names": feature_names,
        "character_input": "domain_label_only",
        "tabular_weight": tabular_weight,
        "character_weight": character_weight,
        "threshold": frozen_threshold,
        "positive_class": "phishing",
        "positive_label": 1,
        "official_dataset_mapping": {0: "phishing", 1: "legitimate"},
        "strong_rule_features": list(STRONG_RULE_FEATURES),
        "exact_trusted_domain_override": "legitimate",
        "previous_pilot_model_reused": False,
        "test_split_used": False,
        "random_seed": arguments.seed,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, bundle_path, compress=3)

    decision_configuration = {
        "phase": 10,
        "model_name": "corrected_domain_hybrid",
        "tabular_weight": tabular_weight,
        "character_weight": character_weight,
        "frozen_validation_threshold": frozen_threshold,
        "positive_class": "phishing",
        "strong_rule_features": list(STRONG_RULE_FEATURES),
        "exact_trusted_domain_override": "legitimate",
        "test_split_used": False,
    }
    decision_path.write_text(
        json.dumps(decision_configuration, indent=2), encoding="utf-8"
    )

    metrics_path = RESULTS_DIRECTORY / "10_corrected_validation_metrics.csv"
    predictions_path = RESULTS_DIRECTORY / "10_validation_predictions.csv"
    importance_path = RESULTS_DIRECTORY / "10_lightgbm_feature_importance.csv"
    character_importance_path = (
        RESULTS_DIRECTORY / "10_character_ngram_importance.csv"
    )
    weight_search_path = RESULTS_DIRECTORY / "10_ensemble_weight_search.csv"
    gap_path = RESULTS_DIRECTORY / "10_train_validation_gap.csv"
    controlled_path = RESULTS_DIRECTORY / "10_controlled_hybrid_robustness.csv"
    write_csv_atomically(validation_metrics, metrics_path)
    write_csv_atomically(validation_predictions, predictions_path)
    write_csv_atomically(importance, importance_path)
    write_csv_atomically(character_importance, character_importance_path)
    write_csv_atomically(weight_search, weight_search_path)
    write_csv_atomically(generalization_gap, gap_path)
    write_csv_atomically(controlled, controlled_path)

    roc_pr_path = REPORTS_DIRECTORY / "10_corrected_validation_roc_pr.png"
    confusion_path = REPORTS_DIRECTORY / "10_corrected_validation_confusion.png"
    top_features_path = REPORTS_DIRECTORY / "10_corrected_top_features.png"
    final_predictions = (
        hybrid_validation_probability >= frozen_threshold
    ).astype(np.int8)
    save_roc_pr_plot(validation_y, hybrid_validation_probability, roc_pr_path)
    save_confusion_plot(validation_y, final_predictions, confusion_path)
    save_feature_importance_plot(importance, top_features_path)

    report = {
        "phase": 10,
        "experiment": "corrected domain-focused hybrid training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_file": str(TRAIN_PATH),
        "validation_file": str(VALIDATION_PATH),
        "train_sha256": sha256_file(TRAIN_PATH),
        "validation_sha256": sha256_file(VALIDATION_PATH),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "registered_domain_overlap": domain_overlap,
        "test_split_read": False,
        "previous_pilot_model_loaded": False,
        "one_labeled_dataset_used": True,
        "trusted_allowlist_is_operational_configuration": True,
        "model_feature_count": len(feature_names),
        "model_features": feature_names,
        "excluded_shortcut_features": list(EXCLUDED_SHORTCUT_FEATURES),
        "character_input": "domain_label_only",
        "character_vocabulary_size": len(vectorizer.vocabulary_),
        "sample_weighting": "class balanced and inverse square-root domain frequency",
        "lightgbm_best_iteration": int(tabular_model.best_iteration_),
        "tabular_training_seconds": tabular_seconds,
        "character_training_seconds": character_seconds,
        "selected_tabular_weight": tabular_weight,
        "selected_character_weight": character_weight,
        "frozen_validation_threshold": frozen_threshold,
        "train_metrics_at_frozen_threshold": train_metrics,
        "validation_metrics_at_frozen_threshold": final_validation_metrics,
        "controlled_robustness_passed": int(controlled["accepted"].sum()),
        "controlled_robustness_total": len(controlled),
        "models": {
            "bundle": str(bundle_path),
            "tabular": str(tabular_model_path),
            "booster": str(booster_path),
            "vectorizer": str(vectorizer_path),
            "character": str(character_model_path),
            "decision_config": str(decision_path),
        },
    }
    report_path = REPORTS_DIRECTORY / "10_corrected_training_summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 76)
    print("CORRECTED VALIDATION RESULTS")
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
    print(validation_metrics[display_columns].to_string(index=False))
    print("\nSelected final development model: Corrected domain hybrid")
    print(f"LightGBM weight: {tabular_weight:.2f}")
    print(f"Character-model weight: {character_weight:.2f}")
    print(f"Frozen validation threshold: {frozen_threshold:.6f}")
    print(f"Controlled robustness: {int(controlled['accepted'].sum())}/{len(controlled)}")
    print("Locked final test used: NO")
    print("Previous pilot reused: NO")

    print("\nTRAINING VS VALIDATION - SELECTED HYBRID")
    print(generalization_gap.to_string(index=False))

    print("\nTop 10 corrected LightGBM features:")
    print(
        importance[["feature", "gain_importance_percent"]]
        .head(10)
        .to_string(index=False)
    )

    print("\nPhase 10 files created:")
    for path in (
        bundle_path,
        tabular_model_path,
        booster_path,
        vectorizer_path,
        character_model_path,
        decision_path,
        metrics_path,
        predictions_path,
        importance_path,
        character_importance_path,
        weight_search_path,
        gap_path,
        controlled_path,
        report_path,
        roc_pr_path,
        confusion_path,
        top_features_path,
    ):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
