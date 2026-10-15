"""
IT22131010 - Phase 3
Advanced URL, domain and typosquatting feature extraction.

This script performs offline string analysis only.
It never visits any URL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.url_feature_extractor import (  # noqa: E402
    TRUSTED_DOMAINS,
    extract_url_features,
    model_feature_names,
)


# ============================================================
# DIRECTORIES
# ============================================================

SPLITS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REPORTS_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
)

EXTRACTOR_PATH = (
    PROJECT_ROOT
    / "src"
    / "url_feature_extractor.py"
)


SPLIT_NAMES = (
    "train",
    "validation",
    "test",
)


METADATA_COLUMNS = [
    "source_row_id",
    "URL",
    "dataset_label",
    "is_phishing",
    "class_name",
    "hostname",
    "registered_domain",
    "fold_id",
]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract advanced URL and "
            "typosquatting features."
        )
    )

    parser.add_argument(
        "--only",
        choices=(
            "all",
            "train",
            "validation",
            "test",
        ),
        default="all",
        help=(
            "Process all splits or one "
            "selected split."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional development-only "
            "row limit per split."
        ),
    )

    return parser.parse_args()


# ============================================================
# FILE HELPERS
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


def write_csv_safely(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    temporary_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    dataframe.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(destination)


# ============================================================
# CONTROLLED TYPOSQUATTING TESTS
# ============================================================

def run_controlled_tests() -> pd.DataFrame:
    """
    Test known typosquatting techniques before processing
    the complete dataset.
    """

    test_cases = [
        (
            "legitimate_trusted",
            "https://www.kayak.com/",
        ),
        (
            "wrong_tld_digit",
            "https://www.kayak.c0m/",
        ),
        (
            "digit_letter",
            "https://paypa1.com/login",
        ),
        (
            "transposition",
            "https://microsfot.com/",
        ),
        (
            "added_word",
            "https://paypal-login.com/",
        ),
        (
            "misleading_subdomain",
            "https://paypal.com.attacker.net/login",
        ),
        (
            "keyboard_neighbor",
            "https://goohle.com/",
        ),
        (
            "unicode_homoglyph",
            "https://раypal.com/",
        ),
    ]

    rows = []

    for case_name, url in test_cases:
        features = extract_url_features(url)

        rows.append(
            {
                "case":
                    case_name,

                "url":
                    url,

                "nearest_trusted_domain":
                    features[
                        "nearest_trusted_domain"
                    ],

                "valid_tld":
                    features["valid_tld"],

                "invalid_tld":
                    features["invalid_tld"],

                "exact_trusted_domain":
                    features[
                        "exact_trusted_domain"
                    ],

                "wrong_tld":
                    features["wrong_tld"],

                "damerau_distance":
                    features[
                        "nearest_damerau_distance"
                    ],

                "jaro_winkler":
                    features[
                        "nearest_jaro_winkler_similarity"
                    ],

                "digit_letter_match":
                    features[
                        "leet_or_digit_letter_match"
                    ],

                "transposition":
                    features[
                        "edit_transposition"
                    ],

                "keyboard_neighbor":
                    features[
                        "keyboard_neighbor_substitution"
                    ],

                "added_suspicious_word":
                    features[
                        "added_suspicious_word"
                    ],

                "misleading_subdomain":
                    features[
                        "misleading_subdomain_pattern"
                    ],

                "punycode":
                    features["has_punycode"],

                "unicode_confusable":
                    features[
                        "dangerous_unicode_confusable"
                    ],
            }
        )

    results = pd.DataFrame(rows)

    results = results.set_index(
        "case",
        drop=False,
    )

    acceptance_results = {
        "legitimate_trusted": (
            results.loc[
                "legitimate_trusted",
                "exact_trusted_domain",
            ]
            == 1
        ),

        "wrong_tld_digit": (
            results.loc[
                "wrong_tld_digit",
                "nearest_trusted_domain",
            ]
            == "kayak.com"
            and results.loc[
                "wrong_tld_digit",
                "invalid_tld",
            ]
            == 1
            and results.loc[
                "wrong_tld_digit",
                "wrong_tld",
            ]
            == 1
            and results.loc[
                "wrong_tld_digit",
                "digit_letter_match",
            ]
            == 1
        ),

        "digit_letter": (
            results.loc[
                "digit_letter",
                "digit_letter_match",
            ]
            == 1
        ),

        "transposition": (
            results.loc[
                "transposition",
                "transposition",
            ]
            == 1
        ),

        "added_word": (
            results.loc[
                "added_word",
                "added_suspicious_word",
            ]
            == 1
        ),

        "misleading_subdomain": (
            results.loc[
                "misleading_subdomain",
                "misleading_subdomain",
            ]
            == 1
        ),

        "keyboard_neighbor": (
            results.loc[
                "keyboard_neighbor",
                "keyboard_neighbor",
            ]
            == 1
        ),

        "unicode_homoglyph": (
            results.loc[
                "unicode_homoglyph",
                "punycode",
            ]
            == 1
            and results.loc[
                "unicode_homoglyph",
                "unicode_confusable",
            ]
            == 1
        ),
    }

    results["accepted"] = (
        results["case"]
        .map(acceptance_results)
        .astype(bool)
    )

    if not results["accepted"].all():
        failed_tests = results.loc[
            ~results["accepted"],
            [
                "case",
                "url",
            ],
        ]

        raise RuntimeError(
            "Controlled typosquatting test failed:\n"
            + failed_tests.to_string(
                index=False
            )
        )

    return results.reset_index(
        drop=True
    )


# ============================================================
# EXTRACT FEATURES FROM ONE SPLIT
# ============================================================

def extract_split_features(
    split_name: str,
    row_limit: int | None,
) -> tuple[pd.DataFrame, float]:

    input_path = (
        SPLITS_DIRECTORY
        / f"{split_name}_urls.csv"
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            "Missing Phase 2 split file:\n"
            f"{input_path}\n"
            "Run script 02 first."
        )

    source_dataframe = pd.read_csv(
        input_path,
        low_memory=False,
    )

    missing_columns = sorted(
        set(METADATA_COLUMNS).difference(
            source_dataframe.columns
        )
    )

    if missing_columns:
        raise ValueError(
            f"{input_path.name} is missing "
            f"columns: {missing_columns}"
        )

    if row_limit is not None:
        if row_limit <= 0:
            raise ValueError(
                "--limit must be greater than zero."
            )

        source_dataframe = (
            source_dataframe
            .head(row_limit)
            .copy()
        )

    print(
        f"\n{split_name.upper()}: "
        f"extracting "
        f"{len(source_dataframe):,} rows..."
    )

    starting_time = time.perf_counter()

    feature_records = []

    url_values = (
        source_dataframe["URL"]
        .tolist()
    )

    for row_number, url in enumerate(
        url_values,
        start=1,
    ):
        feature_records.append(
            extract_url_features(url)
        )

        if (
            row_number % 25_000 == 0
            or row_number
            == len(source_dataframe)
        ):
            print(
                f"  Completed "
                f"{row_number:,}/"
                f"{len(source_dataframe):,}"
            )

    feature_dataframe = (
        pd.DataFrame.from_records(
            feature_records
        )
    )

    numeric_feature_columns = (
        model_feature_names()
    )

    expected_columns = [
        "nearest_trusted_domain",
        *numeric_feature_columns,
    ]

    if (
        feature_dataframe.columns.tolist()
        != expected_columns
    ):
        raise RuntimeError(
            "Feature schema or feature "
            "column order changed."
        )

    numeric_dataframe = (
        feature_dataframe[
            numeric_feature_columns
        ]
    )

    if numeric_dataframe.isna().any().any():
        raise RuntimeError(
            f"NaN feature detected "
            f"in the {split_name} split."
        )

    numeric_array = (
        numeric_dataframe.to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        numeric_array
    ).all():
        raise RuntimeError(
            f"Infinite feature detected "
            f"in the {split_name} split."
        )

    combined_dataframe = pd.concat(
        [
            source_dataframe[
                METADATA_COLUMNS
            ].reset_index(drop=True),

            feature_dataframe,
        ],
        axis=1,
    )

    if (
        len(combined_dataframe)
        != len(source_dataframe)
    ):
        raise RuntimeError(
            f"Row-count mismatch "
            f"in the {split_name} split."
        )

    elapsed_seconds = (
        time.perf_counter()
        - starting_time
    )

    return (
        combined_dataframe,
        elapsed_seconds,
    )


# ============================================================
# MAIN PROCESS
# ============================================================

def main() -> int:
    arguments = parse_arguments()

    print("=" * 76)

    print(
        "IT22131010 - Phase 3 "
        "advanced URL feature extraction"
    )

    print("=" * 76)

    print(
        "Safety: offline string analysis only; "
        "no URL is visited."
    )

    print(
        "Trusted rule entries: "
        f"{len(TRUSTED_DOMAINS)}"
    )

    numeric_feature_columns = (
        model_feature_names()
    )

    print(
        "Numeric model features: "
        f"{len(numeric_feature_columns)}"
    )

    # --------------------------------------------------------
    # Create output folders
    # --------------------------------------------------------

    PROCESSED_DIRECTORY.mkdir(
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

    # --------------------------------------------------------
    # Run robustness tests
    # --------------------------------------------------------

    print(
        "Running controlled "
        "typosquatting tests..."
    )

    robustness_results = (
        run_controlled_tests()
    )

    robustness_output_path = (
        RESULTS_DIRECTORY
        / "03_controlled_typosquatting_tests.csv"
    )

    write_csv_safely(
        robustness_results,
        robustness_output_path,
    )

    passed_test_count = int(
        robustness_results[
            "accepted"
        ].sum()
    )

    total_test_count = len(
        robustness_results
    )

    print(
        "Controlled tests passed: "
        f"{passed_test_count}/"
        f"{total_test_count}"
    )

    # --------------------------------------------------------
    # Select splits
    # --------------------------------------------------------

    if arguments.only == "all":
        selected_splits = SPLIT_NAMES

    else:
        selected_splits = (
            arguments.only,
        )

    output_paths = {}
    split_row_counts = {}
    split_elapsed_seconds = {}

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    for split_name in selected_splits:
        (
            combined_dataframe,
            elapsed_seconds,
        ) = extract_split_features(
            split_name,
            arguments.limit,
        )

        output_path = (
            PROCESSED_DIRECTORY
            / f"{split_name}_features.csv"
        )

        write_csv_safely(
            combined_dataframe,
            output_path,
        )

        output_paths[split_name] = (
            output_path
        )

        split_row_counts[split_name] = (
            len(combined_dataframe)
        )

        split_elapsed_seconds[split_name] = (
            round(
                elapsed_seconds,
                4,
            )
        )

        print(
            f"  Saved: {output_path}"
        )

        print(
            f"  Time: "
            f"{elapsed_seconds:.2f} seconds"
        )

    # --------------------------------------------------------
    # Create feature schema checksum
    # --------------------------------------------------------

    feature_schema_text = "\n".join(
        numeric_feature_columns
    )

    feature_schema_hash = hashlib.sha256(
        feature_schema_text.encode(
            "utf-8"
        )
    ).hexdigest()

    # --------------------------------------------------------
    # Create feature manifest
    # --------------------------------------------------------

    feature_manifest = {
        "phase":
            3,

        "offline_only":
            True,

        "source_splits":
            str(SPLITS_DIRECTORY),

        "processed_directory":
            str(PROCESSED_DIRECTORY),

        "trusted_domain_rule_count":
            len(TRUSTED_DOMAINS),

        "trusted_domains_are_labeled_dataset":
            False,

        "numeric_feature_count":
            len(numeric_feature_columns),

        "numeric_features":
            numeric_feature_columns,

        "metadata_columns": [
            *METADATA_COLUMNS,
            "nearest_trusted_domain",
        ],

        "feature_schema_sha256":
            feature_schema_hash,

        "extractor_sha256":
            calculate_sha256(
                EXTRACTOR_PATH
            ),

        "processed_rows":
            split_row_counts,

        "elapsed_seconds":
            split_elapsed_seconds,

        "controlled_tests_passed":
            passed_test_count,

        "controlled_test_count":
            total_test_count,

        "output_files": {
            split_name: str(path)
            for split_name, path
            in output_paths.items()
        },
    }

    manifest_output_path = (
        REPORTS_DIRECTORY
        / "03_feature_manifest.json"
    )

    manifest_output_path.write_text(
        json.dumps(
            feature_manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Create text summary
    # --------------------------------------------------------

    summary_output_path = (
        REPORTS_DIRECTORY
        / "03_feature_extraction_summary.txt"
    )

    summary_lines = [
        (
            "IT22131010 - Phase 3 "
            "Feature Extraction Summary"
        ),

        "=" * 52,

        "Offline URL analysis only: YES",

        (
            "Numeric model features: "
            f"{len(numeric_feature_columns)}"
        ),

        (
            "Trusted rule entries: "
            f"{len(TRUSTED_DOMAINS)}"
        ),

        (
            "Controlled typosquatting "
            "tests passed: "
            f"{passed_test_count}/"
            f"{total_test_count}"
        ),
    ]

    for split_name in selected_splits:
        summary_lines.append(
            f"{split_name}: "
            f"{split_row_counts[split_name]:,} rows, "
            f"{split_elapsed_seconds[split_name]:.2f} seconds"
        )

    summary_lines.extend(
        [
            "NaN numeric features: 0",
            "Infinite numeric features: 0",
            (
                "Feature schema SHA-256: "
                f"{feature_schema_hash}"
            ),
        ]
    )

    summary_output_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print(
        "\n"
        + "\n".join(
            summary_lines[2:]
        )
    )

    print(
        "\nPhase 3 reports created:"
    )

    print(
        f"  {robustness_output_path}"
    )

    print(
        f"  {manifest_output_path}"
    )

    print(
        f"  {summary_output_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())