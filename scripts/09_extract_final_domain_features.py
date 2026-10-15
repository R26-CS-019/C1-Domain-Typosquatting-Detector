#Extracts final features for every dataset URL
"""Extract corrected domain-only features from train and validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.final_domain_feature_extractor import (  # noqa: E402
    EXCLUDED_SHORTCUT_FEATURES,
    TRUSTED_DOMAINS,
    all_numeric_feature_names,
    evidence_only_feature_names,
    extract_final_domain_features,
    model_feature_names,
)


EXPERIMENT_DIRECTORY = PROJECT_ROOT / "data" / "final_domain_experiment"
SPLITS_DIRECTORY = EXPERIMENT_DIRECTORY / "splits"
PROCESSED_DIRECTORY = EXPERIMENT_DIRECTORY / "processed"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
EXTRACTOR_PATH = PROJECT_ROOT / "src" / "final_domain_feature_extractor.py"
PERMITTED_SPLITS = ("train", "validation")
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
            "Extract corrected domain-only features from the Phase 8 train "
            "and validation splits. The locked test split is unavailable here."
        )
    )
    parser.add_argument(
        "--only",
        choices=("all", *PERMITTED_SPLITS),
        default="all",
        help="Process train+validation or one permitted split (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Development-only row limit per selected split",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25_000,
        help="Progress reporting interval (default: 25000)",
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


def run_controlled_tests() -> pd.DataFrame:
    cases = [
        (
            "legitimate_trusted",
            "https://www.kayak.com/",
            lambda f: f["exact_trusted_domain"] == 1
            and f["wrong_tld"] == 0,
        ),
        (
            "wrong_tld_digit",
            "https://www.kayak.c0m/",
            lambda f: f["nearest_trusted_domain"] == "kayak.com"
            and f["invalid_tld"] == 1
            and f["wrong_tld"] == 1
            and f["leet_or_digit_letter_match"] == 1,
        ),
        (
            "digit_letter",
            "https://paypa1.com/login",
            lambda f: f["leet_or_digit_letter_match"] == 1,
        ),
        (
            "transposition",
            "https://microsfot.com/",
            lambda f: f["edit_transposition"] == 1,
        ),
        (
            "added_word",
            "https://paypal-login.com/",
            lambda f: f["added_suspicious_word"] == 1,
        ),
        (
            "misleading_subdomain",
            "https://paypal.com.attacker.net/login",
            lambda f: f["misleading_subdomain_pattern"] == 1,
        ),
        (
            "keyboard_neighbor",
            "https://goohle.com/",
            lambda f: f["keyboard_neighbor_substitution"] == 1,
        ),
        (
            "unicode_homoglyph",
            "https://раypal.com/",
            lambda f: f["has_punycode"] == 1
            and f["has_unicode"] == 1
            and (
                f["dangerous_unicode_confusable"] == 1
                or f["mixed_script"] == 1
            ),
        ),
        (
            "hyphen_insertion",
            "https://kay-ak.com/",
            lambda f: f["hyphen_insertion"] == 1,
        ),
        (
            "missing_dot",
            "https://kayakcom/",
            lambda f: f["missing_dot_pattern"] == 1,
        ),
    ]

    rows: list[dict[str, object]] = []
    for case_name, url, acceptance_function in cases:
        features = extract_final_domain_features(url)
        rows.append(
            {
                "case": case_name,
                "url": url,
                "nearest_trusted_domain": features[
                    "nearest_trusted_domain"
                ],
                "valid_tld": features["valid_tld"],
                "invalid_tld": features["invalid_tld"],
                "exact_trusted_domain": features["exact_trusted_domain"],
                "damerau_distance": features["nearest_damerau_distance"],
                "jaro_winkler": features[
                    "nearest_jaro_winkler_similarity"
                ],
                "wrong_tld": features["wrong_tld"],
                "digit_letter_match": features[
                    "leet_or_digit_letter_match"
                ],
                "transposition": features["edit_transposition"],
                "keyboard_neighbor": features[
                    "keyboard_neighbor_substitution"
                ],
                "hyphen_insertion": features["hyphen_insertion"],
                "added_suspicious_word": features[
                    "added_suspicious_word"
                ],
                "misleading_subdomain": features[
                    "misleading_subdomain_pattern"
                ],
                "missing_dot": features["missing_dot_pattern"],
                "punycode": features["has_punycode"],
                "unicode_confusable": features[
                    "dangerous_unicode_confusable"
                ],
                "accepted": bool(acceptance_function(features)),
            }
        )

    result = pd.DataFrame(rows)
    if not result["accepted"].all():
        failures = result.loc[~result["accepted"], ["case", "url"]]
        raise RuntimeError(
            "Controlled test failure:\n" + failures.to_string(index=False)
        )
    return result


def extract_split(
    split_name: str,
    limit: int | None,
    progress_every: int,
) -> tuple[pd.DataFrame, float, str, int]:
    if split_name not in PERMITTED_SPLITS:
        raise ValueError(
            "Phase 9 may read only train and validation; test remains locked."
        )

    input_path = SPLITS_DIRECTORY / f"{split_name}_urls.csv"
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Missing Phase 8 split: {input_path}. Run script 08 first."
        )

    input_hash = sha256_file(input_path)
    source = pd.read_csv(input_path, low_memory=False)
    missing_columns = sorted(set(METADATA_COLUMNS).difference(source.columns))
    if missing_columns:
        raise ValueError(f"{input_path.name} is missing columns: {missing_columns}")
    original_rows = len(source)

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than zero")
        source = source.head(limit).copy()

    print(f"\n{split_name.upper()}: extracting {len(source):,} rows...")
    started = time.perf_counter()
    records: list[dict[str, int | float | str]] = []
    for row_number, url in enumerate(source["URL"].tolist(), start=1):
        records.append(extract_final_domain_features(url))
        if row_number % progress_every == 0 or row_number == len(source):
            print(f"  Completed {row_number:,}/{len(source):,}")

    extracted = pd.DataFrame.from_records(records)
    expected_columns = [
        *EXPLANATION_COLUMNS,
        *all_numeric_feature_names(),
    ]
    if extracted.columns.tolist() != expected_columns:
        raise RuntimeError("Corrected feature schema/order changed unexpectedly")

    numeric = extracted[all_numeric_feature_names()]
    if numeric.isna().any().any():
        raise RuntimeError(f"NaN numeric feature detected in {split_name}")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"Infinite numeric feature detected in {split_name}")

    # Phase 8 grouped IP hosts with its legacy last-two-label fallback. The
    # corrected extractor intentionally keeps a complete IP address, so compare
    # normalization only for ordinary DNS hostnames.
    comparable_domain_mask = extracted["hostname_is_ip"].eq(0)
    if comparable_domain_mask.any():
        normalized_match_rate = float(
            (
                source.loc[comparable_domain_mask, "registered_domain"]
                .astype(str)
                .str.lower()
                .to_numpy()
                == extracted.loc[
                    comparable_domain_mask, "normalized_registered_domain"
                ]
                .astype(str)
                .to_numpy()
            ).mean()
        )
    else:
        normalized_match_rate = 1.0
    if normalized_match_rate < 0.999:
        raise RuntimeError(
            "Registered-domain normalization mismatch is unexpectedly high: "
            f"{normalized_match_rate:.6f}"
        )

    combined = pd.concat(
        [source[METADATA_COLUMNS].reset_index(drop=True), extracted], axis=1
    )
    if len(combined) != len(source):
        raise RuntimeError(f"Row-count mismatch in {split_name}")
    elapsed = time.perf_counter() - started
    return combined, elapsed, input_hash, original_rows


def main() -> int:
    arguments = parse_arguments()
    if arguments.progress_every <= 0:
        print("ERROR: --progress-every must be greater than zero")
        return 1

    print("=" * 76)
    print("IT22131010 - Phase 9 corrected domain-only feature extraction")
    print("=" * 76)
    print("Safety: offline domain-string analysis only; no URL is visited.")
    print("Allowed inputs: train and validation only")
    print("Locked final test read: NO")
    print("Previous pilot model loaded: NO")
    print(f"Trusted rule entries: {len(TRUSTED_DOMAINS)}")
    print(f"Corrected model features: {len(model_feature_names())}")
    print(f"Evidence-only features: {len(evidence_only_feature_names())}")
    print(f"Rejected shortcut features: {len(EXCLUDED_SHORTCUT_FEATURES)}")

    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("Running controlled domain/typosquatting tests...")
    controlled = run_controlled_tests()
    controlled_path = RESULTS_DIRECTORY / "09_controlled_domain_tests.csv"
    write_csv_atomically(controlled, controlled_path)
    print(
        "Controlled tests passed: "
        f"{int(controlled['accepted'].sum())}/{len(controlled)}"
    )

    selected_splits = (
        PERMITTED_SPLITS
        if arguments.only == "all"
        else (arguments.only,)
    )
    output_paths: dict[str, Path] = {}
    output_rows: dict[str, int] = {}
    source_rows: dict[str, int] = {}
    source_hashes: dict[str, str] = {}
    elapsed_seconds: dict[str, float] = {}

    for split_name in selected_splits:
        combined, elapsed, source_hash, original_rows = extract_split(
            split_name,
            arguments.limit,
            arguments.progress_every,
        )
        output_path = PROCESSED_DIRECTORY / f"{split_name}_domain_features.csv"
        write_csv_atomically(combined, output_path)
        output_paths[split_name] = output_path
        output_rows[split_name] = len(combined)
        source_rows[split_name] = original_rows
        source_hashes[split_name] = source_hash
        elapsed_seconds[split_name] = round(elapsed, 4)
        print(f"  Saved: {output_path}")
        print(f"  Time: {elapsed:.2f} seconds")

    schema_hash = hashlib.sha256(
        "\n".join(model_feature_names()).encode("utf-8")
    ).hexdigest()
    manifest = {
        "phase": 9,
        "experiment": "corrected domain-focused final experiment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "offline_only": True,
        "previous_pilot_model_loaded": False,
        "test_split_read": False,
        "permitted_input_splits": list(PERMITTED_SPLITS),
        "processed_splits": list(selected_splits),
        "development_limit": arguments.limit,
        "source_rows_before_limit": source_rows,
        "processed_rows": output_rows,
        "source_split_sha256": source_hashes,
        "extractor_sha256": sha256_file(EXTRACTOR_PATH),
        "feature_schema_sha256": schema_hash,
        "trusted_domain_rule_count": len(TRUSTED_DOMAINS),
        "trusted_domains_are_second_labeled_dataset": False,
        "model_feature_count": len(model_feature_names()),
        "model_features": model_feature_names(),
        "evidence_only_feature_count": len(evidence_only_feature_names()),
        "evidence_only_features": evidence_only_feature_names(),
        "excluded_shortcut_feature_count": len(EXCLUDED_SHORTCUT_FEATURES),
        "excluded_shortcut_features": list(EXCLUDED_SHORTCUT_FEATURES),
        "shortcut_policy": (
            "No scheme, HTTPS, path, query, fragment, full-URL formatting, "
            "or subdomain source-marker feature is supplied to the ML model."
        ),
        "controlled_tests_passed": int(controlled["accepted"].sum()),
        "controlled_test_count": len(controlled),
        "output_files": {
            name: str(path) for name, path in output_paths.items()
        },
        "label_mapping": {
            "dataset_label=0": "phishing",
            "dataset_label=1": "legitimate",
            "is_phishing=1": "model positive class",
        },
    }
    manifest_path = REPORTS_DIRECTORY / "09_final_domain_feature_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary_lines = [
        "IT22131010 - PHASE 9 CORRECTED DOMAIN FEATURE SUMMARY",
        "=" * 64,
        "Offline domain analysis only: YES",
        "Previous pilot model loaded: NO",
        "Locked test split read: NO",
        f"Corrected ML features: {len(model_feature_names())}",
        f"Evidence-only features: {len(evidence_only_feature_names())}",
        f"Excluded shortcut features: {len(EXCLUDED_SHORTCUT_FEATURES)}",
        (
            "Controlled tests passed: "
            f"{int(controlled['accepted'].sum())}/{len(controlled)}"
        ),
        f"Development row limit: {arguments.limit}",
        "",
    ]
    for split_name in selected_splits:
        summary_lines.extend(
            [
                f"{split_name.upper()}:",
                f"  Source rows: {source_rows[split_name]:,}",
                f"  Processed rows: {output_rows[split_name]:,}",
                f"  Time: {elapsed_seconds[split_name]:.2f} seconds",
            ]
        )
    summary_lines.extend(
        [
            "",
            "ML shortcut policy:",
            "  Scheme/HTTPS features excluded: YES",
            "  Path/query/fragment features excluded: YES",
            "  Full-URL formatting features excluded: YES",
            "  Subdomain source-marker features excluded from ML: YES",
        ]
    )
    summary_path = REPORTS_DIRECTORY / "09_final_domain_feature_summary.txt"
    summary_path.write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 76)
    print("PHASE 9 RESULT")
    print("=" * 76)
    print("Locked final test read: NO")
    print("Previous pilot model reused: NO")
    print(f"Corrected ML features: {len(model_feature_names())}")
    print(f"Evidence-only features: {len(evidence_only_feature_names())}")
    print(f"Controlled tests passed: {len(controlled)}/{len(controlled)}")
    print("NaN numeric features: 0")
    print("Infinite numeric features: 0")
    print(f"Feature schema SHA-256: {schema_hash}")
    print("\nPhase 9 files created:")
    for path in (
        *output_paths.values(),
        controlled_path,
        manifest_path,
        summary_path,
    ):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
