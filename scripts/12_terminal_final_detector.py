#Terminal interface used for the live demonstration
"""Terminal interface for the canonical final domain/typosquatting detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.final_domain_typosquatting_detector import (  # noqa: E402
    controlled_inference_cases,
    predict_url,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the final domain/typosquatting component offline. "
            "No website is visited."
        )
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="Optional URL to evaluate; omit for interactive mode",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete result as JSON",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the ten controlled terminal/API consistency tests",
    )
    return parser.parse_args()


def print_result(result: dict[str, object], as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("\n" + "=" * 72)
    print("IT22131010 - DOMAIN & TYPOSQUATTING RESULT")
    print("=" * 72)
    print(f"Input URL:             {result['input_url']}")
    print(f"Registered domain:     {result['registered_domain']}")
    print(f"TLD:                   {result['tld'] or '(none)'}")
    print(f"Valid TLD:             {result['valid_tld']}")
    print(f"Detailed result:       {result['detailed_prediction']}")
    print(f"Group binary result:   {result['prediction']}")
    print(f"Decision source:       {result['decision_source']}")
    print(f"Confidence:            {result['confidence']}")
    print(
        "Phishing probability: "
        f"{float(result['phishing_probability']):.4%}"
    )
    print(f"Frozen threshold:      {float(result['frozen_threshold']):.6f}")
    print(f"Nearest trusted:       {result['nearest_trusted_domain']}")
    print(f"Damerau distance:      {result['damerau_distance']}")
    print(
        "Jaro-Winkler:         "
        f"{float(result['jaro_winkler_similarity']):.4f}"
    )
    applied_rules = list(result["applied_rules"])
    print(
        "Applied rules:         "
        + (", ".join(applied_rules) if applied_rules else "None")
    )
    print("Evidence:")
    for reason in result["evidence"]:
        print(f"  - {reason}")
    print("=" * 72)


def run_self_test() -> bool:
    print("=" * 72)
    print("IT22131010 - FINAL INFERENCE SELF-TEST")
    print("=" * 72)
    print("Offline analysis only; no URL is visited.\n")
    passed = 0
    cases = controlled_inference_cases()
    for case_name, url, expected in cases:
        result = predict_url(url)
        accepted = result["prediction"] == expected
        passed += int(accepted)
        status = "PASS" if accepted else "FAIL"
        print(
            f"{status:4} | {case_name:23} | "
            f"expected={expected:10} actual={result['prediction']}"
        )
    print(f"\nControlled inference tests: {passed}/{len(cases)}")
    return passed == len(cases)


def evaluate_one(url: str, as_json: bool) -> bool:
    try:
        result = predict_url(url)
    except Exception as error:
        print(f"ERROR: {error}")
        return False
    print_result(result, as_json)
    return True


def interactive_mode(as_json: bool) -> int:
    print("=" * 72)
    print("IT22131010 - FINAL DOMAIN & TYPOSQUATTING DETECTOR")
    print("=" * 72)
    print("Offline string analysis only; websites are not opened.")
    print("Type a URL, or type exit to stop.\n")
    while True:
        try:
            url = input("URL> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            return 0
        if url.lower() in {"exit", "quit", "q"}:
            print("Stopped.")
            return 0
        if not url:
            print("Please enter a URL.")
            continue
        evaluate_one(url, as_json)


def main() -> int:
    arguments = parse_arguments()
    if arguments.self_test:
        return 0 if run_self_test() else 1
    if arguments.url:
        return 0 if evaluate_one(arguments.url, arguments.json) else 1
    return interactive_mode(arguments.json)


if __name__ == "__main__":
    raise SystemExit(main())

