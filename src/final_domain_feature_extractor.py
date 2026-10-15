"""Corrected offline domain and typosquatting feature extraction.

This extractor intentionally avoids URL scheme, HTTPS, path, query, fragment,
and full-URL formatting features. The rejected pilot learned those collection
shortcuts instead of learning domain and typosquatting behaviour.
"""

from __future__ import annotations

import ipaddress
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from urllib.parse import urlsplit

import idna
import tldextract
from confusable_homoglyphs import confusables
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler

from .url_feature_extractor import (
    DIGIT_TO_LETTER,
    KEYBOARD_NEIGHBORS,
    SHORTENER_DOMAINS,
    SUSPICIOUS_WORDS,
    TRUSTED_DOMAINS,
)


EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
VOWELS = frozenset("aeiou")
TRUSTED_DOMAIN_SET = frozenset(TRUSTED_DOMAINS)


def _trusted_parts(domain: str) -> tuple[str, str]:
    extracted = EXTRACTOR(domain)
    return extracted.domain.lower(), extracted.suffix.lower()


TRUSTED_RECORDS: tuple[tuple[str, str, str], ...] = tuple(
    (domain, *_trusted_parts(domain)) for domain in TRUSTED_DOMAINS
)


# These are the only numeric features supplied to the corrected tabular model.
# No scheme, HTTPS, path, query, fragment, slash count, or full-URL feature is
# present. Subdomain features are also excluded from ML to avoid a "www" source
# shortcut; they remain available only as explicit evidence.
MODEL_FEATURES: tuple[str, ...] = (
    "registered_domain_length",
    "domain_label_length",
    "tld_length",
    "domain_letter_count",
    "domain_digit_count",
    "domain_hyphen_count",
    "domain_vowel_count",
    "domain_unique_character_count",
    "domain_digit_ratio",
    "domain_hyphen_ratio",
    "domain_vowel_ratio",
    "domain_unique_character_ratio",
    "domain_entropy",
    "domain_max_repeated_character_run",
    "domain_token_count",
    "domain_starts_with_digit",
    "domain_ends_with_digit",
    "domain_has_double_hyphen",
    "domain_label_starts_or_ends_hyphen",
    "hostname_is_ip",
    "valid_tld",
    "invalid_tld",
    "has_punycode",
    "has_unicode",
    "mixed_script",
    "unicode_script_count",
    "dangerous_unicode_confusable",
    "digit_letter_substitution_count",
    "has_digit_letter_substitution",
    "suspicious_word_count_in_domain",
    "has_suspicious_word_in_domain",
    "is_url_shortener",
    "nearest_damerau_distance",
    "nearest_damerau_similarity",
    "nearest_jaro_winkler_similarity",
    "nearest_label_damerau_distance",
    "nearest_label_damerau_similarity",
    "nearest_label_jaro_winkler_similarity",
    "edit_insertion",
    "edit_deletion",
    "edit_substitution",
    "edit_transposition",
    "keyboard_neighbor_substitution",
    "leet_or_digit_letter_match",
    "hyphen_insertion",
    "wrong_tld",
    "invalid_or_wrong_tld",
    "missing_dot_pattern",
    "added_suspicious_word",
    "high_brand_similarity",
)

EVIDENCE_ONLY_FEATURES: tuple[str, ...] = (
    "normalized_subdomain_count",
    "normalized_subdomain_length",
    "brand_in_subdomain",
    "misleading_subdomain_pattern",
    "exact_trusted_domain",
)

EXCLUDED_SHORTCUT_FEATURES: tuple[str, ...] = (
    "is_https",
    "url_length",
    "url_entropy",
    "url_slash_count",
    "path_length",
    "query_length",
    "fragment_length",
    "url_token_count",
    "url_letter_count",
    "url_digit_count",
    "url_special_count",
    "url_dot_count",
    "url_underscore_count",
    "url_at_count",
    "url_question_count",
    "url_equal_count",
    "url_ampersand_count",
    "url_percent_count",
    "has_suspicious_file_extension",
    "has_userinfo",
    "has_explicit_port",
    "hostname_length",
    "subdomain_length",
    "subdomain_count",
    "max_repeated_character_run_on_hostname",
)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in Counter(text).values()
    )


def _max_run(text: str) -> int:
    if not text:
        return 0
    best = current = 1
    for previous, current_character in zip(text, text[1:]):
        if previous == current_character:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _canonical_hostname(hostname: str) -> str:
    hostname = hostname.lower().rstrip(".")
    if not hostname:
        return ""
    try:
        return idna.encode(hostname).decode("ascii")
    except idna.IDNAError:
        return hostname


def _unicode_hostname(hostname: str) -> str:
    try:
        return idna.decode(hostname)
    except idna.IDNAError:
        return hostname


def _is_ip(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname.strip("[]"))
        return True
    except ValueError:
        return False


def _script_flags(text: str) -> tuple[int, int]:
    scripts: set[str] = set()
    for character in text:
        if not character.isalpha():
            continue
        unicode_name = unicodedata.name(character, "")
        for script in (
            "LATIN",
            "CYRILLIC",
            "GREEK",
            "ARMENIAN",
            "HEBREW",
            "ARABIC",
            "DEVANAGARI",
        ):
            if script in unicode_name:
                scripts.add(script)
                break
    return int(len(scripts) > 1), len(scripts)


def _leet_skeleton(text: str) -> str:
    return "".join(DIGIT_TO_LETTER.get(character, character) for character in text)


def _has_adjacent_transposition(candidate: str, trusted: str) -> bool:
    if len(candidate) != len(trusted) or candidate == trusted:
        return False
    for index in range(len(candidate) - 1):
        swapped = (
            candidate[:index]
            + candidate[index + 1]
            + candidate[index]
            + candidate[index + 2 :]
        )
        if swapped == trusted:
            return True
    return False


def _keyboard_neighbor_substitution(candidate: str, trusted: str) -> bool:
    if len(candidate) != len(trusted):
        return False
    differences = [
        (candidate_character, trusted_character)
        for candidate_character, trusted_character in zip(candidate, trusted)
        if candidate_character != trusted_character
    ]
    if len(differences) != 1:
        return False
    candidate_character, trusted_character = differences[0]
    return (
        candidate_character in KEYBOARD_NEIGHBORS.get(trusted_character, "")
        or trusted_character in KEYBOARD_NEIGHBORS.get(candidate_character, "")
    )


def _normalise_subdomain(subdomain: str) -> str:
    labels = [label for label in subdomain.lower().split(".") if label]
    while labels and labels[0] == "www":
        labels.pop(0)
    return ".".join(labels)


@lru_cache(maxsize=300_000)
def _nearest_trusted(
    registered_domain: str,
) -> tuple[str, str, str, int, float, float]:
    best_domain, best_label, best_suffix = TRUSTED_RECORDS[0]
    best_distance = 10**9
    best_normalized = 0.0
    best_jaro = 0.0
    best_score = -1.0

    for trusted_domain, trusted_label, trusted_suffix in TRUSTED_RECORDS:
        distance = DamerauLevenshtein.distance(registered_domain, trusted_domain)
        normalized = 1.0 - _safe_ratio(
            distance, max(len(registered_domain), len(trusted_domain), 1)
        )
        jaro = JaroWinkler.normalized_similarity(
            registered_domain, trusted_domain
        )
        score = 0.55 * jaro + 0.45 * normalized
        if score > best_score or (
            score == best_score and distance < best_distance
        ):
            best_domain = trusted_domain
            best_label = trusted_label
            best_suffix = trusted_suffix
            best_distance = int(distance)
            best_normalized = float(normalized)
            best_jaro = float(jaro)
            best_score = float(score)

    return (
        best_domain,
        best_label,
        best_suffix,
        best_distance,
        best_normalized,
        best_jaro,
    )


def _parse_domain(url_value: object) -> dict[str, str | int]:
    raw_url = "" if url_value is None else str(url_value).strip()
    parse_value = raw_url if "://" in raw_url else f"//{raw_url}"
    try:
        raw_hostname = urlsplit(parse_value).hostname or ""
    except ValueError:
        raw_hostname = ""

    hostname = _canonical_hostname(raw_hostname)
    unicode_hostname = _unicode_hostname(hostname)
    host_is_ip = _is_ip(hostname)
    extracted = EXTRACTOR(hostname) if hostname else EXTRACTOR("")
    labels = [label for label in hostname.split(".") if label]

    if extracted.suffix or host_is_ip:
        candidate_tld = extracted.suffix.lower()
        domain_label = extracted.domain.lower()
        subdomain = extracted.subdomain.lower()
        registered_domain = (
            extracted.top_domain_under_public_suffix or hostname
        ).lower()
    elif len(labels) >= 2:
        # Preserve invalid final labels such as c0m for wrong-TLD comparison.
        candidate_tld = labels[-1]
        domain_label = labels[-2]
        subdomain = ".".join(labels[:-2])
        registered_domain = ".".join(labels[-2:])
    else:
        candidate_tld = ""
        domain_label = hostname
        subdomain = ""
        registered_domain = hostname

    return {
        "hostname": hostname,
        "unicode_hostname": unicode_hostname,
        "registered_domain": registered_domain,
        "domain_label": domain_label,
        "candidate_tld": candidate_tld,
        "normalized_subdomain": _normalise_subdomain(subdomain),
        "public_suffix_present": int(bool(extracted.suffix)),
        "hostname_is_ip": int(host_is_ip),
    }


def extract_final_domain_features(
    url_value: object,
) -> dict[str, int | float | str]:
    """Return corrected domain features and explanation metadata for one URL."""
    parsed = _parse_domain(url_value)
    hostname = str(parsed["hostname"])
    unicode_hostname = str(parsed["unicode_hostname"])
    registered_domain = str(parsed["registered_domain"])
    domain_label = str(parsed["domain_label"])
    candidate_tld = str(parsed["candidate_tld"])
    normalized_subdomain = str(parsed["normalized_subdomain"])
    host_is_ip = int(parsed["hostname_is_ip"])
    public_suffix_present = int(parsed["public_suffix_present"])

    nearest_domain, nearest_label, nearest_suffix, distance, normalized, jaro = (
        _nearest_trusted(registered_domain)
    )
    label_distance = int(
        DamerauLevenshtein.distance(domain_label, nearest_label)
    )
    label_normalized = 1.0 - _safe_ratio(
        label_distance, max(len(domain_label), len(nearest_label), 1)
    )
    label_jaro = float(
        JaroWinkler.normalized_similarity(domain_label, nearest_label)
    )

    exact_trusted = int(registered_domain in TRUSTED_DOMAIN_SET)
    suffix_matches = candidate_tld == nearest_suffix
    edit_insertion = int(
        suffix_matches
        and label_distance == 1
        and len(domain_label) == len(nearest_label) + 1
    )
    edit_deletion = int(
        suffix_matches
        and label_distance == 1
        and len(domain_label) + 1 == len(nearest_label)
    )
    edit_transposition = int(
        suffix_matches
        and label_distance == 1
        and _has_adjacent_transposition(domain_label, nearest_label)
    )
    edit_substitution = int(
        suffix_matches
        and label_distance == 1
        and len(domain_label) == len(nearest_label)
        and not edit_transposition
    )
    keyboard_substitution = int(
        edit_substitution
        and _keyboard_neighbor_substitution(domain_label, nearest_label)
    )
    leet_match = int(
        registered_domain != nearest_domain
        and (
            _leet_skeleton(registered_domain) == nearest_domain
            or (
                suffix_matches
                and _leet_skeleton(domain_label) == nearest_label
            )
        )
    )
    wrong_tld = int(
        domain_label == nearest_label
        and bool(candidate_tld)
        and candidate_tld != nearest_suffix
    )
    invalid_tld = int(not public_suffix_present and not host_is_ip)
    invalid_or_wrong_tld = int(invalid_tld or wrong_tld)
    hyphen_insertion = int(
        suffix_matches
        and domain_label != nearest_label
        and "-" in domain_label
        and domain_label.replace("-", "") == nearest_label
    )
    missing_dot = int(
        registered_domain != nearest_domain
        and registered_domain.replace(".", "").replace("-", "")
        == nearest_domain.replace(".", "").replace("-", "")
    )
    domain_tokens = TOKEN_PATTERN.findall(domain_label)
    suspicious_word_count = sum(
        token in SUSPICIOUS_WORDS for token in domain_tokens
    )
    added_suspicious_word = int(
        nearest_label in domain_label
        and domain_label != nearest_label
        and any(word in domain_label for word in SUSPICIOUS_WORDS)
    )

    compact_subdomain = re.sub(r"[^a-z0-9]", "", normalized_subdomain)
    brand_in_subdomain = int(
        not exact_trusted
        and any(
            trusted_label in compact_subdomain
            for _, trusted_label, _ in TRUSTED_RECORDS
            if len(trusted_label) >= 4
        )
    )

    mixed_script, script_count = _script_flags(unicode_hostname)
    has_unicode = int(any(ord(character) > 127 for character in unicode_hostname))
    has_punycode = int(
        any(label.startswith("xn--") for label in hostname.split("."))
    )
    try:
        dangerous_unicode = int(
            bool(
                confusables.is_dangerous(
                    unicode_hostname, preferred_aliases=["latin"]
                )
            )
        )
    except Exception:
        dangerous_unicode = mixed_script

    domain_length = len(domain_label)
    domain_letters = sum(character.isalpha() for character in domain_label)
    domain_digits = sum(character.isdigit() for character in domain_label)
    domain_hyphens = domain_label.count("-")
    domain_vowels = sum(character in VOWELS for character in domain_label)
    domain_unique = len(set(domain_label))
    digit_letter_count = sum(
        character in DIGIT_TO_LETTER for character in registered_domain
    )
    normalized_subdomain_labels = [
        label for label in normalized_subdomain.split(".") if label
    ]

    result: dict[str, int | float | str] = {
        "normalized_hostname": hostname,
        "normalized_registered_domain": registered_domain,
        "domain_label": domain_label,
        "candidate_tld": candidate_tld,
        "normalized_subdomain": normalized_subdomain,
        "nearest_trusted_domain": nearest_domain,
        "registered_domain_length": len(registered_domain),
        "domain_label_length": domain_length,
        "tld_length": len(candidate_tld),
        "domain_letter_count": domain_letters,
        "domain_digit_count": domain_digits,
        "domain_hyphen_count": domain_hyphens,
        "domain_vowel_count": domain_vowels,
        "domain_unique_character_count": domain_unique,
        "domain_digit_ratio": _safe_ratio(domain_digits, domain_length),
        "domain_hyphen_ratio": _safe_ratio(domain_hyphens, domain_length),
        "domain_vowel_ratio": _safe_ratio(domain_vowels, domain_length),
        "domain_unique_character_ratio": _safe_ratio(
            domain_unique, domain_length
        ),
        "domain_entropy": _entropy(domain_label),
        "domain_max_repeated_character_run": _max_run(domain_label),
        "domain_token_count": len(domain_tokens),
        "domain_starts_with_digit": int(
            bool(domain_label) and domain_label[0].isdigit()
        ),
        "domain_ends_with_digit": int(
            bool(domain_label) and domain_label[-1].isdigit()
        ),
        "domain_has_double_hyphen": int("--" in domain_label),
        "domain_label_starts_or_ends_hyphen": int(
            domain_label.startswith("-") or domain_label.endswith("-")
        ),
        "hostname_is_ip": host_is_ip,
        "valid_tld": int(bool(public_suffix_present) or host_is_ip),
        "invalid_tld": invalid_tld,
        "has_punycode": has_punycode,
        "has_unicode": has_unicode,
        "mixed_script": mixed_script,
        "unicode_script_count": script_count,
        "dangerous_unicode_confusable": dangerous_unicode,
        "digit_letter_substitution_count": digit_letter_count,
        "has_digit_letter_substitution": int(digit_letter_count > 0),
        "suspicious_word_count_in_domain": suspicious_word_count,
        "has_suspicious_word_in_domain": int(suspicious_word_count > 0),
        "is_url_shortener": int(registered_domain in SHORTENER_DOMAINS),
        "nearest_damerau_distance": distance,
        "nearest_damerau_similarity": normalized,
        "nearest_jaro_winkler_similarity": jaro,
        "nearest_label_damerau_distance": label_distance,
        "nearest_label_damerau_similarity": label_normalized,
        "nearest_label_jaro_winkler_similarity": label_jaro,
        "edit_insertion": edit_insertion,
        "edit_deletion": edit_deletion,
        "edit_substitution": edit_substitution,
        "edit_transposition": edit_transposition,
        "keyboard_neighbor_substitution": keyboard_substitution,
        "leet_or_digit_letter_match": leet_match,
        "hyphen_insertion": hyphen_insertion,
        "wrong_tld": wrong_tld,
        "invalid_or_wrong_tld": invalid_or_wrong_tld,
        "missing_dot_pattern": missing_dot,
        "added_suspicious_word": added_suspicious_word,
        "high_brand_similarity": int(
            not exact_trusted
            and (normalized >= 0.80 or jaro >= 0.88)
        ),
        "normalized_subdomain_count": len(normalized_subdomain_labels),
        "normalized_subdomain_length": len(normalized_subdomain),
        "brand_in_subdomain": brand_in_subdomain,
        "misleading_subdomain_pattern": brand_in_subdomain,
        "exact_trusted_domain": exact_trusted,
    }
    return result


def model_feature_names() -> list[str]:
    """Return the frozen corrected tabular-model feature order."""
    return list(MODEL_FEATURES)


def evidence_only_feature_names() -> list[str]:
    """Return features retained for rules/explanations but excluded from ML."""
    return list(EVIDENCE_ONLY_FEATURES)


def all_numeric_feature_names() -> list[str]:
    """Return every numeric feature produced by the corrected extractor."""
    return [*MODEL_FEATURES, *EVIDENCE_ONLY_FEATURES]

