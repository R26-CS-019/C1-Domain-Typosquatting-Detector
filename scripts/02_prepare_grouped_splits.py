"""Create deterministic, leakage-safe PhiUSIIL train/validation/test splits.

This script treats URLs as offline strings. It never visits a website and never
modifies the source CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import tldextract
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DATASET_PATH, REPORTS_DIR, RESULTS_DIR  # noqa: E402


ROOT_DATASET_PATH = PROJECT_ROOT / "PhiUSIIL_Phishing_URL_Dataset.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RANDOM_SEED = 42
NUMBER_OF_FOLDS = 20
TRAIN_FOLDS = frozenset(range(0, 14))
VALIDATION_FOLDS = frozenset(range(14, 17))
TEST_FOLDS = frozenset(range(17, 20))
SPLIT_ORDER = ["train", "validation", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove exact duplicate URLs and make grouped 70/15/15 PhiUSIIL splits."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional explicit path to PhiUSIIL_Phishing_URL_Dataset.csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Deterministic split seed (default: {RANDOM_SEED})",
    )
    return parser.parse_args()


def resolve_dataset_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    for candidate in (ROOT_DATASET_PATH, DEFAULT_DATASET_PATH):
        if candidate.is_file():
            return candidate.resolve()
    return ROOT_DATASET_PATH.resolve()


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


def registered_domain_from_host(
    hostname: str, extractor: tldextract.TLDExtract
) -> str:
    if not hostname:
        return ""
    extracted = extractor(hostname)
    return extracted.top_domain_under_public_suffix or hostname


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


def intersection_count(left: pd.Series, right: pd.Series) -> int:
    return len(set(left.tolist()).intersection(right.tolist()))


def main() -> int:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)

    print("=" * 76)
    print("IT22131010 - Phase 2 duplicate handling and grouped split")
    print("=" * 76)
    print(f"Dataset: {dataset_path}")
    print(f"Random seed: {args.seed}")
    print("Target split: train=70%, validation=15%, test=15%")

    if not dataset_path.is_file():
        print("\nERROR: Dataset CSV was not found.")
        print(f"Checked root location: {ROOT_DATASET_PATH}")
        print(f"Checked original location: {DEFAULT_DATASET_PATH}")
        return 1

    source_hash_before = sha256_file(dataset_path)
    print(f"Source SHA-256: {source_hash_before}")
    print("Reading URL and label columns...")

    frame = pd.read_csv(dataset_path, usecols=["URL", "label"], low_memory=False)
    frame.insert(0, "source_row_id", np.arange(len(frame), dtype=np.int64))
    frame["URL"] = frame["URL"].astype("string").str.strip()
    numeric_label = pd.to_numeric(frame["label"], errors="coerce")

    if frame["URL"].isna().any() or frame["URL"].eq("").any():
        print("ERROR: Missing or empty URL values were found.")
        return 1
    if numeric_label.isna().any():
        print("ERROR: Non-numeric or missing labels were found.")
        return 1
    if not set(numeric_label.astype(int).unique()).issubset({0, 1}):
        print("ERROR: Labels outside the official {0, 1} mapping were found.")
        return 1

    frame["dataset_label"] = numeric_label.astype(np.int8)
    duplicate_label_counts = frame.groupby("URL", sort=False)[
        "dataset_label"
    ].nunique()
    conflicting_duplicate_urls = int((duplicate_label_counts > 1).sum())
    if conflicting_duplicate_urls:
        print(
            "ERROR: Duplicate URLs with conflicting labels were found: "
            f"{conflicting_duplicate_urls}"
        )
        return 1

    original_rows = len(frame)
    duplicate_mask = frame.duplicated(subset=["URL"], keep="first")
    duplicates_removed = int(duplicate_mask.sum())
    cleaned = frame.loc[~duplicate_mask].copy().reset_index(drop=True)
    cleaned["is_phishing"] = (cleaned["dataset_label"] == 0).astype(np.int8)
    cleaned["class_name"] = cleaned["is_phishing"].map(
        {0: "legitimate", 1: "phishing"}
    )

    print(f"Original rows: {original_rows:,}")
    print(f"Exact duplicate URL rows removed: {duplicates_removed:,}")
    print(f"Clean rows: {len(cleaned):,}")
    print("Parsing and canonicalizing hostnames offline...")

    cleaned["hostname"] = cleaned["URL"].map(hostname_from_url)
    empty_hostname_rows = int(cleaned["hostname"].eq("").sum())
    if empty_hostname_rows:
        print(f"ERROR: {empty_hostname_rows} URLs could not be assigned a hostname.")
        return 1

    extractor = tldextract.TLDExtract(suffix_list_urls=())
    unique_hosts = cleaned["hostname"].drop_duplicates()
    registered_lookup = {
        hostname: registered_domain_from_host(hostname, extractor)
        for hostname in unique_hosts.tolist()
    }
    cleaned["registered_domain"] = cleaned["hostname"].map(registered_lookup)
    if cleaned["registered_domain"].eq("").any():
        print("ERROR: One or more URLs could not be assigned a registered domain.")
        return 1

    registered_label_counts = cleaned.groupby("registered_domain")[
        "dataset_label"
    ].nunique()
    mixed_label_registered_domains = int((registered_label_counts > 1).sum())

    print(
        "Creating 20 stratified group folds; every registered domain stays in "
        "one fold..."
    )
    splitter = StratifiedGroupKFold(
        n_splits=NUMBER_OF_FOLDS,
        shuffle=True,
        random_state=args.seed,
    )
    fold_ids = np.full(len(cleaned), -1, dtype=np.int8)
    dummy_features = np.zeros((len(cleaned), 1), dtype=np.int8)
    for fold_id, (_, fold_indices) in enumerate(
        splitter.split(
            dummy_features,
            cleaned["is_phishing"].to_numpy(),
            groups=cleaned["registered_domain"].to_numpy(),
        )
    ):
        fold_ids[fold_indices] = fold_id

    if (fold_ids < 0).any():
        print("ERROR: At least one row was not assigned to a fold.")
        return 1

    cleaned["fold_id"] = fold_ids
    cleaned["split"] = cleaned["fold_id"].map(split_name_from_fold)
    cleaned["split"] = pd.Categorical(
        cleaned["split"], categories=SPLIT_ORDER, ordered=True
    )

    split_frames = {
        split_name: cleaned.loc[cleaned["split"].eq(split_name)].copy()
        for split_name in SPLIT_ORDER
    }

    checks: list[dict[str, object]] = []
    for field in ("URL", "hostname", "registered_domain"):
        for left_name, right_name in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlap = intersection_count(
                split_frames[left_name][field], split_frames[right_name][field]
            )
            checks.append(
                {
                    "check": f"{field}_{left_name}_{right_name}_overlap",
                    "value": overlap,
                    "passed": overlap == 0,
                }
            )

    checks.extend(
        [
            {
                "check": "all_clean_rows_assigned_once",
                "value": sum(len(part) for part in split_frames.values()),
                "passed": sum(len(part) for part in split_frames.values())
                == len(cleaned),
            },
            {
                "check": "duplicate_urls_remaining",
                "value": int(cleaned["URL"].duplicated().sum()),
                "passed": not cleaned["URL"].duplicated().any(),
            },
        ]
    )

    overlap_audit = pd.DataFrame(checks)
    if not overlap_audit["passed"].all():
        print("ERROR: Split leakage or an integrity-check failure was detected.")
        print(overlap_audit.loc[~overlap_audit["passed"]].to_string(index=False))
        return 1

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "source_row_id",
        "URL",
        "dataset_label",
        "is_phishing",
        "class_name",
        "hostname",
        "registered_domain",
        "fold_id",
    ]
    output_paths: dict[str, Path] = {}
    for split_name, split_frame in split_frames.items():
        output_path = SPLITS_DIR / f"{split_name}_urls.csv"
        write_csv_atomically(split_frame[output_columns], output_path)
        output_paths[split_name] = output_path

    class_counts = (
        cleaned.groupby(["split", "class_name"], observed=True)
        .size()
        .rename("row_count")
        .reset_index()
    )
    split_totals = cleaned.groupby("split", observed=True).size().rename("split_total")
    class_counts["split_total"] = class_counts["split"].map(split_totals)
    class_counts["class_percentage"] = (
        class_counts["row_count"] / class_counts["split_total"] * 100
    ).round(4)
    class_counts["dataset_percentage"] = (
        class_counts["row_count"] / len(cleaned) * 100
    ).round(4)

    fold_distribution = (
        cleaned.groupby(["fold_id", "class_name"], observed=True)
        .size()
        .rename("row_count")
        .reset_index()
    )

    source_hash_after = sha256_file(dataset_path)
    source_unchanged = source_hash_before == source_hash_after
    overlap_audit.loc[len(overlap_audit)] = {
        "check": "source_csv_sha256_unchanged",
        "value": int(source_unchanged),
        "passed": source_unchanged,
    }
    if not source_unchanged:
        print("ERROR: The source CSV checksum unexpectedly changed.")
        return 1

    counts_path = RESULTS_DIR / "02_split_class_counts.csv"
    folds_path = RESULTS_DIR / "02_fold_distribution.csv"
    overlap_path = RESULTS_DIR / "02_split_overlap_audit.csv"
    write_csv_atomically(class_counts, counts_path)
    write_csv_atomically(fold_distribution, folds_path)
    write_csv_atomically(overlap_audit, overlap_path)

    summary = {
        "phase": 2,
        "source_csv": str(dataset_path),
        "source_sha256": source_hash_before,
        "source_csv_modified": False,
        "official_label_mapping": {"0": "phishing", "1": "legitimate"},
        "model_positive_class": {"is_phishing=1": "phishing"},
        "random_seed": args.seed,
        "split_method": "StratifiedGroupKFold",
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
        "mixed_label_registered_domains": mixed_label_registered_domains,
        "split_rows": {
            name: len(split_frame) for name, split_frame in split_frames.items()
        },
        "split_registered_domains": {
            name: int(split_frame["registered_domain"].nunique())
            for name, split_frame in split_frames.items()
        },
        "all_integrity_checks_passed": bool(overlap_audit["passed"].all()),
        "output_files": {name: str(path) for name, path in output_paths.items()},
    }

    summary_json_path = REPORTS_DIR / "02_grouped_split_summary.json"
    summary_text_path = REPORTS_DIR / "02_grouped_split_summary.txt"
    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report_lines = [
        "IT22131010 - Phase 2 Grouped Split Summary",
        "=" * 48,
        f"Source: {dataset_path}",
        f"Source SHA-256: {source_hash_before}",
        f"Original rows: {original_rows:,}",
        f"Duplicate URL rows removed: {duplicates_removed:,}",
        f"Clean rows: {len(cleaned):,}",
        f"Unique hostnames: {cleaned['hostname'].nunique():,}",
        f"Unique registered domains: {cleaned['registered_domain'].nunique():,}",
        f"Mixed-label registered domains kept in one split: {mixed_label_registered_domains:,}",
        "",
    ]
    for split_name in SPLIT_ORDER:
        part = split_frames[split_name]
        phishing_rows = int(part["is_phishing"].sum())
        legitimate_rows = len(part) - phishing_rows
        report_lines.extend(
            [
                f"{split_name.upper()}:",
                f"  Rows: {len(part):,} ({len(part) / len(cleaned) * 100:.4f}%)",
                f"  Phishing: {phishing_rows:,}",
                f"  Legitimate: {legitimate_rows:,}",
                f"  Registered domains: {part['registered_domain'].nunique():,}",
            ]
        )
    report_lines.extend(
        [
            "",
            "Registered-domain overlap across splits: 0",
            "Hostname overlap across splits: 0",
            "URL overlap across splits: 0",
            "All integrity checks passed: YES",
            "Source CSV modified: NO",
        ]
    )
    summary_text_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(report_lines[3:]))
    print("\nPhase 2 files created:")
    for path in output_paths.values():
        print(f"  {path}")
    for path in (
        summary_json_path,
        summary_text_path,
        counts_path,
        folds_path,
        overlap_path,
    ):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

