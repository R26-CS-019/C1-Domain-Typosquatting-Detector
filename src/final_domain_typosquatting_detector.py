"""Canonical offline inference API for the final domain/typosquatting component.

The terminal program and future web API must both call ``predict_url`` from
this module so that they always use the same model, threshold, rule order, and
output mapping. No website is contacted during prediction.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .final_domain_feature_extractor import (
    extract_final_domain_features,
    model_feature_names,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = (
    PROJECT_ROOT / "models" / "10_corrected_domain_hybrid_bundle.joblib"
)

DETAILED_LEGITIMATE = "LEGITIMATE"
DETAILED_PHISHING = "PHISHING"
DETAILED_TYPOSQUAT = "SUSPICIOUS_TYPOSQUAT"
GROUP_LEGITIMATE = "LEGITIMATE"
GROUP_PHISHING = "PHISHING"

RULE_DESCRIPTIONS: dict[str, str] = {
    "invalid_or_wrong_tld": "Invalid or brand-mismatched top-level domain",
    "edit_insertion": "One-character insertion near a trusted domain",
    "edit_deletion": "One-character deletion near a trusted domain",
    "edit_substitution": "One-character substitution near a trusted domain",
    "edit_transposition": "Adjacent-character transposition near a trusted domain",
    "keyboard_neighbor_substitution": "Keyboard-neighbour character substitution",
    "leet_or_digit_letter_match": "Digit-to-letter impersonation such as 0/o or 1/l",
    "hyphen_insertion": "Hyphen inserted into a trusted domain name",
    "missing_dot_pattern": "Missing-dot impersonation of a trusted domain",
    "added_suspicious_word": "Suspicious word added to a trusted brand name",
    "misleading_subdomain_pattern": "Trusted brand placed in a misleading subdomain",
    "dangerous_unicode_confusable": "Unicode homoglyph/confusable domain pattern",
    "mixed_script": "Multiple writing systems mixed in the hostname",
}


@lru_cache(maxsize=4)
def _load_bundle(bundle_path_text: str) -> dict[str, Any]:
    bundle_path = Path(bundle_path_text)
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Frozen Phase 10 model bundle was not found: {bundle_path}"
        )
    bundle = joblib.load(bundle_path)
    required_keys = {
        "tabular_model",
        "character_vectorizer",
        "character_model",
        "model_feature_names",
        "tabular_weight",
        "character_weight",
        "threshold",
        "strong_rule_features",
    }
    missing = sorted(required_keys.difference(bundle))
    if missing:
        raise ValueError(f"Frozen model bundle is missing keys: {missing}")
    if list(bundle["model_feature_names"]) != model_feature_names():
        raise ValueError("Frozen model feature schema does not match the extractor")

    tabular_weight = float(bundle["tabular_weight"])
    character_weight = float(bundle["character_weight"])
    threshold = float(bundle["threshold"])
    if not np.isclose(tabular_weight + character_weight, 1.0):
        raise ValueError("Frozen ensemble weights do not sum to one")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Frozen decision threshold is outside [0, 1]")
    return bundle


def clear_model_cache() -> None:
    """Clear the bundle cache after an explicitly authorized model replacement."""
    _load_bundle.cache_clear()


def _ml_confidence(probability: float, threshold: float) -> str:
    distance = abs(probability - threshold)
    if distance >= 0.30:
        return "HIGH"
    if distance >= 0.12:
        return "MEDIUM"
    return "LOW"


def _model_probabilities(
    features: dict[str, int | float | str],
    bundle: dict[str, Any],
) -> tuple[float, float, float]:
    feature_names = model_feature_names()
    tabular_frame = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names,
    )
    tabular_probability = float(
        bundle["tabular_model"].predict_proba(tabular_frame)[0, 1]
    )
    character_matrix = bundle["character_vectorizer"].transform(
        [str(features["domain_label"])]
    )
    character_probability = float(
        bundle["character_model"].predict_proba(character_matrix)[0, 1]
    )
    hybrid_probability = float(
        float(bundle["tabular_weight"]) * tabular_probability
        + float(bundle["character_weight"]) * character_probability
    )
    return tabular_probability, character_probability, hybrid_probability


def predict_url(
    url: object,
    bundle_path: str | Path | None = None,
) -> dict[str, Any]:
    """Predict one URL using the frozen hybrid and deterministic evidence.

    ``SUSPICIOUS_TYPOSQUAT`` is retained as the detailed component result and is
    mapped to ``PHISHING`` for the group's binary fusion interface.
    """
    raw_url = "" if url is None else str(url).strip()
    if not raw_url:
        raise ValueError("URL cannot be empty")

    selected_bundle_path = Path(bundle_path or DEFAULT_BUNDLE_PATH).resolve()
    bundle = _load_bundle(str(selected_bundle_path))
    features = extract_final_domain_features(raw_url)
    if not str(features["normalized_hostname"]):
        raise ValueError("A valid hostname could not be parsed from the input")

    tabular_probability, character_probability, hybrid_probability = (
        _model_probabilities(features, bundle)
    )
    threshold = float(bundle["threshold"])
    ml_prediction = int(hybrid_probability >= threshold)
    strong_rule_features = tuple(
        str(value) for value in bundle["strong_rule_features"]
    )
    triggered_rules = [
        rule_name
        for rule_name in strong_rule_features
        if int(features[rule_name]) == 1
    ]
    if int(features["mixed_script"]) == 1 and "mixed_script" not in triggered_rules:
        triggered_rules.append("mixed_script")

    exact_trusted = int(features["exact_trusted_domain"]) == 1
    if exact_trusted:
        detailed_prediction = DETAILED_LEGITIMATE
        binary_prediction = GROUP_LEGITIMATE
        predicted_label = 0
        decision_source = "trusted_allowlist"
        confidence = "HIGH"
        applied_rules = ["exact_trusted_domain"]
        evidence = [
            "The registered domain exactly matches the trusted-domain allowlist."
        ]
    elif triggered_rules:
        detailed_prediction = DETAILED_TYPOSQUAT
        binary_prediction = GROUP_PHISHING
        predicted_label = 1
        decision_source = "deterministic_typosquatting_evidence"
        confidence = "HIGH"
        applied_rules = triggered_rules
        evidence = [RULE_DESCRIPTIONS[name] for name in triggered_rules]
    else:
        detailed_prediction = (
            DETAILED_PHISHING if ml_prediction == 1 else DETAILED_LEGITIMATE
        )
        binary_prediction = (
            GROUP_PHISHING if ml_prediction == 1 else GROUP_LEGITIMATE
        )
        predicted_label = ml_prediction
        decision_source = "frozen_ml_hybrid"
        confidence = _ml_confidence(hybrid_probability, threshold)
        applied_rules = []
        comparison = ">=" if ml_prediction == 1 else "<"
        evidence = [
            (
                f"Frozen hybrid probability {hybrid_probability:.6f} "
                f"{comparison} threshold {threshold:.6f}."
            )
        ]

    if not exact_trusted:
        evidence.append(
            "Nearest trusted domain: "
            f"{features['nearest_trusted_domain']} "
            f"(Damerau distance {int(features['nearest_damerau_distance'])}, "
            "Jaro-Winkler "
            f"{float(features['nearest_jaro_winkler_similarity']):.4f})."
        )

    return {
        "component_id": "IT22131010",
        "component": "domain_and_typosquatting",
        "status": "success",
        "offline_analysis_only": True,
        "input_url": raw_url,
        "normalized_hostname": features["normalized_hostname"],
        "registered_domain": features["normalized_registered_domain"],
        "domain_label": features["domain_label"],
        "tld": features["candidate_tld"],
        "valid_tld": bool(features["valid_tld"]),
        "detailed_prediction": detailed_prediction,
        "prediction": binary_prediction,
        "predicted_label": predicted_label,
        "decision_source": decision_source,
        "confidence": confidence,
        "phishing_probability": hybrid_probability,
        "legitimate_probability": float(1.0 - hybrid_probability),
        "tabular_probability": tabular_probability,
        "character_probability": character_probability,
        "frozen_threshold": threshold,
        "tabular_weight": float(bundle["tabular_weight"]),
        "character_weight": float(bundle["character_weight"]),
        "nearest_trusted_domain": features["nearest_trusted_domain"],
        "damerau_distance": int(features["nearest_damerau_distance"]),
        "damerau_similarity": float(features["nearest_damerau_similarity"]),
        "jaro_winkler_similarity": float(
            features["nearest_jaro_winkler_similarity"]
        ),
        "exact_trusted_domain": exact_trusted,
        "applied_rules": applied_rules,
        "evidence": evidence,
        "model_feature_count": len(model_feature_names()),
        "model_name": "corrected_domain_hybrid",
        "group_mapping": {
            "SUSPICIOUS_TYPOSQUAT": "PHISHING",
            "PHISHING": "PHISHING",
            "LEGITIMATE": "LEGITIMATE",
        },
    }


def controlled_inference_cases() -> tuple[tuple[str, str, str], ...]:
    """Return stable inference cases used by terminal/API consistency tests."""
    return (
        ("legitimate_trusted", "https://www.kayak.com/", GROUP_LEGITIMATE),
        ("wrong_tld_digit", "https://www.kayak.c0m/", GROUP_PHISHING),
        ("digit_letter", "https://paypa1.com/login", GROUP_PHISHING),
        ("transposition", "https://microsfot.com/", GROUP_PHISHING),
        ("added_word", "https://paypal-login.com/", GROUP_PHISHING),
        (
            "misleading_subdomain",
            "https://paypal.com.attacker.net/login",
            GROUP_PHISHING,
        ),
        ("keyboard_neighbor", "https://goohle.com/", GROUP_PHISHING),
        ("unicode_homoglyph", "https://раypal.com/", GROUP_PHISHING),
        ("hyphen_insertion", "https://kay-ak.com/", GROUP_PHISHING),
        ("missing_dot", "https://kayakcom/", GROUP_PHISHING),
    )

