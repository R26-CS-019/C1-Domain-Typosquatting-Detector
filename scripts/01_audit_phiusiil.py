"""Read-only audit of the PhiUSIIL dataset before model development."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import tldextract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DATASET_LABEL_NAMES,
    DEFAULT_DATASET_PATH,
    REPORTS_DIR,
    REQUIRED_COLUMNS,
    RESULTS_DIR,
    classify_column,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit PhiUSIIL without changing the source CSV."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Optional path to PhiUSIIL_Phishing_URL_Dataset.csv",
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
        return (urlsplit(candidate).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def registered_domain_from_host(
    host: str, extractor: tldextract.TLDExtract
) -> str:
    if not host:
        return ""
    extracted = extractor(host)
    return extracted.top_domain_under_public_suffix or host


def serializable_number(value: object) -> int | float | None:
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return round(value, 6)
    return int(value)


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()

    print("=" * 72)
    print("IT22131010 - PhiUSIIL pre-training audit")
    print("=" * 72)
    print(f"Dataset: {dataset_path}")

    if not dataset_path.is_file():
        print("\nERROR: Dataset CSV was not found.")
        print(f"Expected default: {DEFAULT_DATASET_PATH}")
        return 1

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    file_size = dataset_path.stat().st_size
    file_hash = sha256_file(dataset_path)
    print(f"File size: {file_size / (1024 * 1024):.2f} MiB")
    print(f"SHA-256:  {file_hash}")
    print("Reading CSV...")

    frame = pd.read_csv(dataset_path, low_memory=False)
    source_columns = frame.columns.tolist()
    missing_required = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_required:
        print(f"ERROR: Required columns are missing: {missing_required}")
        return 1

    label_numeric = pd.to_numeric(frame["label"], errors="coerce")
    invalid_label_rows = int(label_numeric.isna().sum())
    observed_labels = sorted(label_numeric.dropna().astype(int).unique().tolist())
    unexpected_labels = sorted(set(observed_labels).difference(DATASET_LABEL_NAMES))
    if unexpected_labels:
        print(f"ERROR: Unexpected labels found: {unexpected_labels}")
        return 1

    frame = frame.assign(_label=label_numeric.astype("Int64"))
    urls = frame["URL"].astype("string").str.strip()

    exact_duplicate_rows = int(urls.duplicated(keep="first").sum())
    unique_urls = int(urls.nunique(dropna=True))
    url_label_counts = frame.assign(_url=urls).groupby("_url", dropna=False)[
        "_label"
    ].nunique(dropna=True)
    conflicting_url_count = int((url_label_counts > 1).sum())

    print("Parsing URL hostnames offline...")
    hosts = urls.map(hostname_from_url)
    empty_host_rows = int(hosts.eq("").sum())
    unique_hosts = int(hosts[hosts.ne("")].nunique())
    host_label_counts = frame.assign(_host=hosts).loc[lambda x: x["_host"].ne("")].groupby(
        "_host"
    )["_label"].nunique(dropna=True)
    conflicting_host_count = int((host_label_counts > 1).sum())

    print("Resolving registered domains with the packaged public-suffix snapshot...")
    extractor = tldextract.TLDExtract(suffix_list_urls=())
    unique_host_values = hosts[hosts.ne("")].drop_duplicates()
    registered_lookup = {
        host: registered_domain_from_host(host, extractor)
        for host in unique_host_values.tolist()
    }
    registered_domains = hosts.map(registered_lookup).fillna("")
    unique_registered_domains = int(
        registered_domains[registered_domains.ne("")].nunique()
    )
    registered_label_counts = frame.assign(_registered=registered_domains).loc[
        lambda x: x["_registered"].ne("")
    ].groupby("_registered")["_label"].nunique(dropna=True)
    conflicting_registered_domains = int((registered_label_counts > 1).sum())

    label_counts = (
        frame["_label"]
        .value_counts(dropna=False)
        .rename_axis("dataset_label")
        .reset_index(name="row_count")
    )
    label_counts["class_name"] = label_counts["dataset_label"].map(
        DATASET_LABEL_NAMES
    )
    label_counts["percentage"] = (
        label_counts["row_count"] / len(frame) * 100
    ).round(4)
    label_counts = label_counts[
        ["dataset_label", "class_name", "row_count", "percentage"]
    ].sort_values("dataset_label")

    column_scope = pd.DataFrame(
        {
            "column": source_columns,
            "handling": [classify_column(column) for column in source_columns],
            "missing_count": [int(frame[column].isna().sum()) for column in source_columns],
            "dtype": [str(frame[column].dtype) for column in source_columns],
        }
    )
    unknown_columns = column_scope.loc[
        column_scope["handling"].eq("unknown_review_required"), "column"
    ].tolist()

    url_length_stats = urls.str.len().describe()
    summary = {
        "audit_version": 1,
        "source_csv": str(dataset_path),
        "source_size_bytes": file_size,
        "source_sha256": file_hash,
        "row_count": int(len(frame)),
        "column_count": int(len(source_columns)),
        "label_mapping": {"0": "phishing", "1": "legitimate"},
        "observed_labels": observed_labels,
        "invalid_label_rows": invalid_label_rows,
        "label_counts": label_counts.to_dict(orient="records"),
        "missing_url_rows": int(frame["URL"].isna().sum()),
        "missing_domain_rows": int(frame["Domain"].isna().sum()),
        "missing_tld_rows": int(frame["TLD"].isna().sum()),
        "unique_urls": unique_urls,
        "exact_duplicate_url_rows": exact_duplicate_rows,
        "urls_with_conflicting_labels": conflicting_url_count,
        "empty_or_unparseable_hostname_rows": empty_host_rows,
        "unique_hostnames": unique_hosts,
        "hostnames_with_conflicting_labels": conflicting_host_count,
        "unique_registered_domains": unique_registered_domains,
        "registered_domains_with_conflicting_labels": conflicting_registered_domains,
        "url_length": {
            key: serializable_number(value)
            for key, value in url_length_stats.items()
        },
        "unknown_columns_requiring_review": unknown_columns,
        "raw_csv_modified": False,
    }

    json_path = REPORTS_DIR / "01_dataset_audit_summary.json"
    text_path = REPORTS_DIR / "01_dataset_audit_summary.txt"
    label_path = RESULTS_DIR / "01_label_counts.csv"
    scope_path = RESULTS_DIR / "01_column_scope.csv"

    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    label_counts.to_csv(label_path, index=False)
    column_scope.to_csv(scope_path, index=False)

    report_lines = [
        "IT22131010 - PhiUSIIL Dataset Audit",
        "=" * 45,
        f"Source: {dataset_path}",
        f"SHA-256: {file_hash}",
        f"Rows: {len(frame):,}",
        f"Columns: {len(source_columns):,}",
        "Label mapping: 0=phishing, 1=legitimate",
        "",
        "Class distribution:",
    ]
    for row in label_counts.itertuples(index=False):
        report_lines.append(
            f"  {row.dataset_label} ({row.class_name}): "
            f"{row.row_count:,} ({row.percentage:.4f}%)"
        )
    report_lines.extend(
        [
            "",
            f"Unique URLs: {unique_urls:,}",
            f"Exact duplicate URL rows: {exact_duplicate_rows:,}",
            f"URLs with conflicting labels: {conflicting_url_count:,}",
            f"Unique hostnames: {unique_hosts:,}",
            f"Hostnames with conflicting labels: {conflicting_host_count:,}",
            f"Unique registered domains: {unique_registered_domains:,}",
            "Registered domains with conflicting labels: "
            f"{conflicting_registered_domains:,}",
            f"Empty/unparseable hostname rows: {empty_host_rows:,}",
            f"Unknown columns requiring review: {len(unknown_columns):,}",
            "",
            "Decision: do not train until duplicate handling and a grouped",
            "registered-domain split are implemented.",
            "The source CSV was not modified.",
        ]
    )
    text_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(report_lines[4:]))
    print("\nAudit files created:")
    print(f"  {json_path}")
    print(f"  {text_path}")
    print(f"  {label_path}")
    print(f"  {scope_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
