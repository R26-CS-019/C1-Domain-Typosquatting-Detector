#Cleans the dataset and creates grouped train, validation and test splits
"""Prepare a fresh grouped split for the corrected domain-focused experiment.

The previous 74-feature model is retained as a rejected pilot because it learned
URL-formatting shortcuts. This script never loads that model and never visits a
URL. It creates new split files in a separate directory without changing the
source CSV or the previous experiment files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "PhiUSIIL_Phishing_URL_Dataset.csv"
FINAL_EXPERIMENT_DIRECTORY = PROJECT_ROOT / "data" / "final_domain_experiment"
FINAL_SPLITS_DIRECTORY = FINAL_EXPERIMENT_DIRECTORY / "splits"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
PILOT_MODEL_PATH = PROJECT_ROOT / "models" / "04_selected_model_bundle.joblib"

RANDOM_SEED = 2026
NUMBER_OF_FOLDS = 20
TRAIN_FOLDS = frozenset(range(0, 14))
VALIDATION_FOLDS = frozenset(range(14, 17))
TEST_FOLDS = frozenset(range(17, 20))
SPLIT_ORDER = ("train", "validation", "test")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the new registered-domain-grouped split for the corrected "
            "domain-focused experiment."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to PhiUSIIL_Phishing_URL_Dataset.csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Fixed split seed (default: {RANDOM_SEED})",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hostname_from_url(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return ""
    candidate = text if "://" in text else f"//{text}"
    try:
        hostname = (urlsplit(candidate).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""
    if not hostname:
        return ""
    try:
        return hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return hostname


def registered_domain_from_hostname(
    hostname: str, extractor: tldextract.TLDExtract
) -> str:
    if not hostname:
        return ""
    extracted = extractor(hostname)
    if extracted.suffix:
        return extracted.top_domain_under_public_suffix or hostname
    labels = [label for label in hostname.split(".") if label]
    return ".".join(labels[-2:]) if len(labels) >= 2 else hostname


def split_name_from_fold(fold_id: int) -> str:
    if fold_id in TRAIN_FOLDS:
        return "train"
    if fold_id in VALIDATION_FOLDS:
        return "validation"
    if fold_id in TEST_FOLDS:
        return "test"
    raise ValueError(f"Unexpected fold id: {fold_id}")


def write_csv_atomically(frame: pd.DataFrame, destination: Path) -> None:
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(destination)


def overlap_count(left: pd.Series, right: pd.Series) -> int:
    return len(set(left.tolist()).intersection(right.tolist()))


def main() -> int:
    arguments = parse_arguments()
    dataset_path = arguments.dataset.expanduser().resolve()

    print("=" * 76)
    print("IT22131010 - Phase 8 corrected final experiment split")
    print("=" * 76)
    print("Previous 99% model status: REJECTED PILOT (shortcut learning)")
    print("Safety: offline URL parsing only; no URL will be visited.")
    print(f"Dataset: {dataset_path}")
    print(f"New fixed random seed: {arguments.seed}")
    print("Target: train=70%, validation=15%, locked test=15%")

    if not dataset_path.is_file():
        print(f"ERROR: Dataset CSV was not found: {dataset_path}")
        return 1

    source_hash_before = sha256_file(dataset_path)
    pilot_hash_before = (
        sha256_file(PILOT_MODEL_PATH) if PILOT_MODEL_PATH.is_file() else None
    )
    print(f"Source SHA-256: {source_hash_before}")
    print("Reading only the URL and official label columns...")

    source = pd.read_csv(dataset_path, usecols=["URL", "label"], low_memory=False)
    source.insert(0, "source_row_id", np.arange(len(source), dtype=np.int64))
    source["URL"] = source["URL"].astype("string").str.strip()
    numeric_label = pd.to_numeric(source["label"], errors="coerce")

    if source["URL"].isna().any() or source["URL"].eq("").any():
        print("ERROR: Missing or empty URLs were found.")
        return 1
    if numeric_label.isna().any():
        print("ERROR: Missing or non-numeric labels were found.")
        return 1
    if not set(numeric_label.astype(int).unique()).issubset({0, 1}):
        print("ERROR: Labels outside the official {0, 1} mapping were found.")
        return 1

    source["dataset_label"] = numeric_label.astype(np.int8)
    conflicting_duplicate_counts = source.groupby("URL", sort=False)[
        "dataset_label"
    ].nunique()
    conflicting_duplicate_urls = int((conflicting_duplicate_counts > 1).sum())
    if conflicting_duplicate_urls:
        print(
            "ERROR: Duplicate URLs with conflicting labels were found: "
            f"{conflicting_duplicate_urls}"
        )
        return 1

    original_rows = len(source)
    duplicate_mask = source.duplicated(subset=["URL"], keep="first")
    duplicates_removed = int(duplicate_mask.sum())
    cleaned = source.loc[~duplicate_mask].copy().reset_index(drop=True)
    cleaned["is_phishing"] = (cleaned["dataset_label"] == 0).astype(np.int8)
    cleaned["class_name"] = cleaned["is_phishing"].map(
        {0: "legitimate", 1: "phishing"}
    )

    print(f"Original rows: {original_rows:,}")
    print(f"Exact duplicate rows removed: {duplicates_removed:,}")
    print(f"Clean rows: {len(cleaned):,}")
    print("Parsing hostnames and registered domains offline...")

    cleaned["hostname"] = cleaned["URL"].map(hostname_from_url)
    if cleaned["hostname"].eq("").any():
        count = int(cleaned["hostname"].eq("").sum())
        print(f"ERROR: {count} URLs could not be assigned a hostname.")
        return 1

    extractor = tldextract.TLDExtract(suffix_list_urls=())
    unique_hostnames = cleaned["hostname"].drop_duplicates().tolist()
    domain_lookup = {
        hostname: registered_domain_from_hostname(hostname, extractor)
        for hostname in unique_hostnames
    }
    cleaned["registered_domain"] = cleaned["hostname"].map(domain_lookup)
    if cleaned["registered_domain"].eq("").any():
        print("ERROR: A registered domain could not be assigned.")
        return 1

    mixed_label_domains = int(
        (
            cleaned.groupby("registered_domain")["is_phishing"].nunique()
            > 1
        ).sum()
    )
    print(f"Unique hostnames: {cleaned['hostname'].nunique():,}")
    print(f"Unique registered domains: {cleaned['registered_domain'].nunique():,}")
    print(f"Mixed-label registered domains: {mixed_label_domains:,}")
    print(
        "Creating 20 new stratified group folds; every registered domain "
        "will stay in exactly one split..."
    )

    splitter = StratifiedGroupKFold(
        n_splits=NUMBER_OF_FOLDS,
        shuffle=True,
        random_state=arguments.seed,
    )
    fold_ids = np.full(len(cleaned), -1, dtype=np.int8)
    dummy_features = np.zeros((len(cleaned), 1), dtype=np.int8)
    for fold_id, (_, held_indices) in enumerate(
        splitter.split(
            dummy_features,
            cleaned["is_phishing"].to_numpy(dtype=np.int8),
            groups=cleaned["registered_domain"].to_numpy(),
        )
    ):
        fold_ids[held_indices] = fold_id

    if (fold_ids < 0).any():
        print("ERROR: At least one row was not assigned to a fold.")
        return 1

    cleaned["final_fold_id"] = fold_ids
    cleaned["final_split"] = cleaned["final_fold_id"].map(split_name_from_fold)
    cleaned["final_split"] = pd.Categorical(
        cleaned["final_split"], categories=SPLIT_ORDER, ordered=True
    )
    split_frames = {
        split_name: cleaned.loc[cleaned["final_split"].eq(split_name)].copy()
        for split_name in SPLIT_ORDER
    }

    audit_rows: list[dict[str, object]] = []
    for field in ("URL", "hostname", "registered_domain"):
        for left_name, right_name in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            count = overlap_count(
                split_frames[left_name][field], split_frames[right_name][field]
            )
            audit_rows.append(
                {
                    "check": f"{field}_{left_name}_{right_name}_overlap",
                    "value": count,
                    "passed": count == 0,
                }
            )
    audit_rows.extend(
        [
            {
                "check": "all_clean_rows_assigned_once",
                "value": sum(len(frame) for frame in split_frames.values()),
                "passed": sum(len(frame) for frame in split_frames.values())
                == len(cleaned),
            },
            {
                "check": "duplicate_urls_remaining",
                "value": int(cleaned["URL"].duplicated().sum()),
                "passed": not cleaned["URL"].duplicated().any(),
            },
        ]
    )
    audit = pd.DataFrame(audit_rows)
    if not audit["passed"].all():
        print("ERROR: Leakage or split-integrity failure detected.")
        print(audit.loc[~audit["passed"]].to_string(index=False))
        return 1

    FINAL_SPLITS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "source_row_id",
        "URL",
        "dataset_label",
        "is_phishing",
        "class_name",
        "hostname",
        "registered_domain",
        "final_fold_id",
    ]
    split_paths: dict[str, Path] = {}
    for split_name, split_frame in split_frames.items():
        output_path = FINAL_SPLITS_DIRECTORY / f"{split_name}_urls.csv"
        write_csv_atomically(split_frame[output_columns], output_path)
        split_paths[split_name] = output_path

    class_counts = (
        cleaned.groupby(["final_split", "class_name"], observed=True)
        .size()
        .rename("row_count")
        .reset_index()
    )
    split_totals = (
        cleaned.groupby("final_split", observed=True).size().rename("split_total")
    )
    class_counts["split_total"] = class_counts["final_split"].map(split_totals)
    class_counts["class_percentage"] = (
        class_counts["row_count"] / class_counts["split_total"] * 100
    ).round(4)
    class_counts["dataset_percentage"] = (
        class_counts["row_count"] / len(cleaned) * 100
    ).round(4)

    fold_distribution = (
        cleaned.groupby(["final_fold_id", "class_name"], observed=True)
        .size()
        .rename("row_count")
        .reset_index()
    )

    source_hash_after = sha256_file(dataset_path)
    source_unchanged = source_hash_before == source_hash_after
    pilot_hash_after = (
        sha256_file(PILOT_MODEL_PATH) if PILOT_MODEL_PATH.is_file() else None
    )
    pilot_unchanged = pilot_hash_before == pilot_hash_after
    final_checks = pd.DataFrame(
        [
            {
                "check": "source_csv_sha256_unchanged",
                "value": int(source_unchanged),
                "passed": source_unchanged,
            },
            {
                "check": "pilot_model_unchanged",
                "value": int(pilot_unchanged),
                "passed": pilot_unchanged,
            },
        ]
    )
    audit = pd.concat([audit, final_checks], ignore_index=True)
    if not audit["passed"].all():
        print("ERROR: The source dataset or pilot model unexpectedly changed.")
        return 1

    counts_path = RESULTS_DIRECTORY / "08_final_split_class_counts.csv"
    folds_path = RESULTS_DIRECTORY / "08_final_fold_distribution.csv"
    audit_path = RESULTS_DIRECTORY / "08_final_split_overlap_audit.csv"
    write_csv_atomically(class_counts, counts_path)
    write_csv_atomically(fold_distribution, folds_path)
    write_csv_atomically(audit, audit_path)

    split_summary: dict[str, dict[str, int | float]] = {}
    for split_name, split_frame in split_frames.items():
        split_summary[split_name] = {
            "rows": len(split_frame),
            "percentage": len(split_frame) / len(cleaned) * 100,
            "phishing": int(split_frame["is_phishing"].sum()),
            "legitimate": int((split_frame["is_phishing"] == 0).sum()),
            "unique_hostnames": int(split_frame["hostname"].nunique()),
            "registered_domains": int(
                split_frame["registered_domain"].nunique()
            ),
        }

    report = {
        "phase": 8,
        "experiment": "corrected domain-focused final experiment",
        "previous_model_status": "rejected pilot - URL-format shortcut learning",
        "previous_model_reused": False,
        "source_dataset": str(dataset_path),
        "source_sha256": source_hash_before,
        "source_modified": False,
        "official_label_mapping": {"0": "phishing", "1": "legitimate"},
        "model_positive_class": {"is_phishing=1": "phishing"},
        "random_seed": arguments.seed,
        "split_method": "StratifiedGroupKFold by registered domain",
        "number_of_folds": NUMBER_OF_FOLDS,
        "fold_allocation": {
            "train": sorted(TRAIN_FOLDS),
            "validation": sorted(VALIDATION_FOLDS),
            "test": sorted(TEST_FOLDS),
        },
        "original_rows": original_rows,
        "duplicates_removed": duplicates_removed,
        "clean_rows": len(cleaned),
        "unique_hostnames": int(cleaned["hostname"].nunique()),
        "unique_registered_domains": int(cleaned["registered_domain"].nunique()),
        "mixed_label_registered_domains": mixed_label_domains,
        "splits": split_summary,
        "overlap_checks_passed": bool(audit["passed"].all()),
        "test_split_locked": True,
        "test_must_not_be_read_during_feature_design_or_training": True,
        "one_labeled_dataset_used": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = REPORTS_DIRECTORY / "08_final_split_summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary_lines = [
        "IT22131010 - CORRECTED FINAL DOMAIN EXPERIMENT SPLIT",
        "=" * 64,
        "Previous 99% model: REJECTED PILOT (shortcut learning)",
        f"Source SHA-256: {source_hash_before}",
        f"Random seed: {arguments.seed}",
        f"Clean rows: {len(cleaned):,}",
        f"Unique registered domains: {cleaned['registered_domain'].nunique():,}",
        "",
    ]
    for split_name in SPLIT_ORDER:
        values = split_summary[split_name]
        summary_lines.extend(
            [
                f"{split_name.upper()}:",
                f"  Rows: {values['rows']:,} ({values['percentage']:.4f}%)",
                f"  Phishing: {values['phishing']:,}",
                f"  Legitimate: {values['legitimate']:,}",
                f"  Registered domains: {values['registered_domains']:,}",
            ]
        )
    summary_lines.extend(
        [
            "",
            "Registered-domain overlap across splits: 0",
            "All integrity checks passed: YES",
            "Source CSV modified: NO",
            "Previous pilot model modified: NO",
            "Final test split locked: YES",
        ]
    )
    summary_text_path = REPORTS_DIRECTORY / "08_final_split_summary.txt"
    summary_text_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 76)
    print("CORRECTED SPLIT RESULT")
    print("=" * 76)
    print(f"Source SHA-256 unchanged: {'YES' if source_unchanged else 'NO'}")
    print(f"Pilot model unchanged: {'YES' if pilot_unchanged else 'NO'}")
    print(f"Clean rows: {len(cleaned):,}")
    for split_name in SPLIT_ORDER:
        values = split_summary[split_name]
        print(f"\n{split_name.upper()}:")
        print(f"  Rows: {values['rows']:,} ({values['percentage']:.4f}%)")
        print(f"  Phishing: {values['phishing']:,}")
        print(f"  Legitimate: {values['legitimate']:,}")
        print(f"  Registered domains: {values['registered_domains']:,}")
    print("\nRegistered-domain overlap across splits: 0")
    print("All integrity checks passed: YES")
    print("Final test split locked: YES")
    print("\nPhase 8 files created:")
    for path in (
        split_paths["train"],
        split_paths["validation"],
        split_paths["test"],
        counts_path,
        folds_path,
        audit_path,
        report_path,
        summary_text_path,
    ):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
