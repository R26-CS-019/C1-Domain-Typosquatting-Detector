"""Offline URL, domain and typosquatting feature extraction."""

from __future__ import annotations

import ipaddress
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import idna
import tldextract
from confusable_homoglyphs import confusables
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler


# This is a rule configuration, not another training dataset.
TRUSTED_DOMAINS = tuple(
    sorted(
        {
            "adobe.com",
            "airbnb.com",
            "aliexpress.com",
            "amazon.com",
            "americanexpress.com",
            "apple.com",
            "bankofamerica.com",
            "binance.com",
            "boc.lk",
            "booking.com",
            "canva.com",
            "capitalone.com",
            "chase.com",
            "citi.com",
            "coinbase.com",
            "combank.lk",
            "daraz.lk",
            "dhl.com",
            "dialog.lk",
            "discord.com",
            "dropbox.com",
            "ebay.com",
            "expedia.com",
            "facebook.com",
            "fedex.com",
            "github.com",
            "gmail.com",
            "google.com",
            "hnb.net",
            "hsbc.com",
            "icloud.com",
            "instagram.com",
            "kapruka.com",
            "kayak.com",
            "linkedin.com",
            "live.com",
            "mastercard.com",
            "meta.com",
            "microsoft.com",
            "mobitel.lk",
            "netflix.com",
            "notion.so",
            "office.com",
            "outlook.com",
            "paypal.com",
            "peoplesbank.lk",
            "proton.me",
            "reddit.com",
            "roblox.com",
            "sampath.lk",
            "slack.com",
            "slt.lk",
            "spotify.com",
            "steam.com",
            "steamcommunity.com",
            "stripe.com",
            "target.com",
            "tiktok.com",
            "uber.com",
            "ups.com",
            "visa.com",
            "walmart.com",
            "wellsfargo.com",
            "whatsapp.com",
            "wikipedia.org",
            "x.com",
            "yahoo.com",
            "zoom.us",
        }
    )
)


SUSPICIOUS_WORDS = {
    "account",
    "alert",
    "auth",
    "authenticate",
    "bank",
    "billing",
    "confirm",
    "credential",
    "invoice",
    "login",
    "password",
    "payment",
    "recover",
    "reset",
    "secure",
    "security",
    "signin",
    "support",
    "suspend",
    "unlock",
    "update",
    "verification",
    "verify",
    "wallet",
}


SUSPICIOUS_EXTENSIONS = {
    ".apk",
    ".bat",
    ".cmd",
    ".com",
    ".exe",
    ".hta",
    ".jar",
    ".js",
    ".msi",
    ".ps1",
    ".scr",
    ".vbs",
    ".zip",
}


SHORTENER_DOMAINS = {
    "bit.ly",
    "cutt.ly",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "rebrand.ly",
    "shorturl.at",
    "t.co",
    "tiny.cc",
    "tinyurl.com",
}


DIGIT_TO_LETTER = {
    "0": "o",
    "1": "l",
    "2": "z",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
}


KEYBOARD_NEIGHBORS = {
    "q": "wa",
    "w": "qase",
    "e": "wsdr",
    "r": "edft",
    "t": "rfgy",
    "y": "tghu",
    "u": "yhji",
    "i": "ujko",
    "o": "iklp",
    "p": "ol",
    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=()
)


def _trusted_parts(domain: str) -> tuple[str, str]:
    extracted = EXTRACTOR(domain)

    return (
        extracted.domain,
        extracted.suffix,
    )


TRUSTED_RECORDS = tuple(
    (
        domain,
        *_trusted_parts(domain),
    )
    for domain in TRUSTED_DOMAINS
)

TRUSTED_DOMAIN_SET = frozenset(
    TRUSTED_DOMAINS
)


def _safe_ratio(
    numerator: int | float,
    denominator: int | float,
) -> float:
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def _entropy(text: str) -> float:
    if not text:
        return 0.0

    length = len(text)
    counter = Counter(text)

    entropy = 0.0

    for count in counter.values():
        probability = count / length

        entropy -= (
            probability
            * math.log2(probability)
        )

    return entropy


def _maximum_character_run(text: str) -> int:
    if not text:
        return 0

    best = 1
    current = 1

    for previous, character in zip(
        text,
        text[1:],
    ):
        if previous == character:
            current += 1
            best = max(best, current)
        else:
            current = 1

    return best


def _canonical_hostname(
    hostname: str,
) -> str:
    hostname = hostname.lower().rstrip(".")

    if not hostname:
        return ""

    try:
        return idna.encode(
            hostname
        ).decode("ascii")

    except idna.IDNAError:
        return hostname


def _unicode_hostname(
    hostname: str,
) -> str:
    try:
        return idna.decode(hostname)

    except idna.IDNAError:
        return hostname


def _is_ip_address(
    hostname: str,
) -> bool:
    try:
        ipaddress.ip_address(
            hostname.strip("[]")
        )

        return True

    except ValueError:
        return False


def _script_information(
    text: str,
) -> tuple[int, int]:
    scripts = set()

    for character in text:
        if not character.isalpha():
            continue

        unicode_name = unicodedata.name(
            character,
            "",
        )

        for script in (
            "LATIN",
            "CYRILLIC",
            "GREEK",
            "ARMENIAN",
            "HEBREW",
        ):
            if script in unicode_name:
                scripts.add(script)
                break

    mixed_script = int(
        len(scripts) > 1
    )

    return mixed_script, len(scripts)


def _has_adjacent_transposition(
    candidate: str,
    trusted: str,
) -> bool:
    if (
        len(candidate) != len(trusted)
        or candidate == trusted
    ):
        return False

    for index in range(
        len(candidate) - 1
    ):
        swapped = (
            candidate[:index]
            + candidate[index + 1]
            + candidate[index]
            + candidate[index + 2:]
        )

        if swapped == trusted:
            return True

    return False


def _keyboard_neighbor_substitution(
    candidate: str,
    trusted: str,
) -> bool:
    if len(candidate) != len(trusted):
        return False

    differences = []

    for candidate_character, trusted_character in zip(
        candidate,
        trusted,
    ):
        if candidate_character != trusted_character:
            differences.append(
                (
                    candidate_character,
                    trusted_character,
                )
            )

    if len(differences) != 1:
        return False

    candidate_character, trusted_character = (
        differences[0]
    )

    return (
        candidate_character
        in KEYBOARD_NEIGHBORS.get(
            trusted_character,
            "",
        )
        or trusted_character
        in KEYBOARD_NEIGHBORS.get(
            candidate_character,
            "",
        )
    )


def _leet_skeleton(text: str) -> str:
    return "".join(
        DIGIT_TO_LETTER.get(
            character,
            character,
        )
        for character in text
    )


@lru_cache(maxsize=250_000)
def _nearest_trusted_domain(
    registered_domain: str,
) -> tuple[str, str, str, int, float, float]:
    best_domain = TRUSTED_DOMAINS[0]
    best_label = TRUSTED_RECORDS[0][1]
    best_suffix = TRUSTED_RECORDS[0][2]

    best_distance = 10**9
    best_normalized_similarity = 0.0
    best_jaro_similarity = 0.0
    best_score = -1.0

    for (
        trusted_domain,
        trusted_label,
        trusted_suffix,
    ) in TRUSTED_RECORDS:

        distance = (
            DamerauLevenshtein.distance(
                registered_domain,
                trusted_domain,
            )
        )

        normalized_similarity = (
            1.0
            - _safe_ratio(
                distance,
                max(
                    len(registered_domain),
                    len(trusted_domain),
                    1,
                ),
            )
        )

        jaro_similarity = (
            JaroWinkler.normalized_similarity(
                registered_domain,
                trusted_domain,
            )
        )

        combined_score = (
            0.55 * jaro_similarity
            + 0.45 * normalized_similarity
        )

        if (
            combined_score > best_score
            or (
                combined_score == best_score
                and distance < best_distance
            )
        ):
            best_domain = trusted_domain
            best_label = trusted_label
            best_suffix = trusted_suffix
            best_distance = distance

            best_normalized_similarity = (
                normalized_similarity
            )

            best_jaro_similarity = (
                jaro_similarity
            )

            best_score = combined_score

    return (
        best_domain,
        best_label,
        best_suffix,
        int(best_distance),
        float(best_normalized_similarity),
        float(best_jaro_similarity),
    )


def extract_url_features(
    url_value: object,
) -> dict[str, int | float | str]:
    """
    Extract lexical and typosquatting features from one URL.

    The URL is processed only as text.
    No network connection is made.
    """

    if url_value is None:
        raw_url = ""
    else:
        raw_url = str(url_value).strip()

    if "://" in raw_url:
        parsing_value = raw_url
    else:
        parsing_value = f"//{raw_url}"

    try:
        parsed = urlsplit(parsing_value)
        raw_hostname = parsed.hostname or ""

    except ValueError:
        parsed = urlsplit("//invalid.local")
        raw_hostname = ""

    hostname = _canonical_hostname(
        raw_hostname
    )

    unicode_hostname = _unicode_hostname(
        hostname
    )

    hostname_is_ip = _is_ip_address(
        hostname
    )

    if hostname:
        extracted = EXTRACTOR(hostname)
    else:
        extracted = EXTRACTOR("")

    public_suffix = extracted.suffix

    hostname_labels = [
        label
        for label in hostname.split(".")
        if label
    ]

    if public_suffix or hostname_is_ip:
        candidate_tld = public_suffix
        domain_label = extracted.domain
        subdomain = extracted.subdomain

        registered_domain = (
            extracted.top_domain_under_public_suffix
            or hostname
        )

    elif len(hostname_labels) >= 2:
        # This handles invalid TLDs such as kayak.c0m.
        candidate_tld = hostname_labels[-1]
        domain_label = hostname_labels[-2]

        subdomain = ".".join(
            hostname_labels[:-2]
        )

        registered_domain = ".".join(
            hostname_labels[-2:]
        )

    else:
        candidate_tld = ""
        domain_label = hostname
        subdomain = ""
        registered_domain = hostname

    try:
        explicit_port = int(
            parsed.port is not None
        )

    except ValueError:
        explicit_port = 1

    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    lowercase_url = raw_url.lower()

    url_tokens = TOKEN_PATTERN.findall(
        lowercase_url
    )

    hostname_tokens = TOKEN_PATTERN.findall(
        hostname
    )

    url_length = len(raw_url)

    url_letter_count = sum(
        character.isalpha()
        for character in raw_url
    )

    url_digit_count = sum(
        character.isdigit()
        for character in raw_url
    )

    url_special_count = sum(
        not character.isalnum()
        for character in raw_url
    )

    hostname_letter_count = sum(
        character.isalpha()
        for character in hostname
    )

    hostname_digit_count = sum(
        character.isdigit()
        for character in hostname
    )

    suspicious_word_count = sum(
        token in SUSPICIOUS_WORDS
        for token in url_tokens
    )

    digit_letter_count = sum(
        character in DIGIT_TO_LETTER
        for character in hostname
    )

    mixed_script, unicode_script_count = (
        _script_information(
            unicode_hostname
        )
    )

    has_unicode = int(
        any(
            ord(character) > 127
            for character in unicode_hostname
        )
    )

    has_punycode = int(
        any(
            label.startswith("xn--")
            for label in hostname.split(".")
        )
    )

    try:
        dangerous_unicode = int(
            bool(
                confusables.is_dangerous(
                    unicode_hostname,
                    preferred_aliases=[
                        "latin"
                    ],
                )
            )
        )

    except Exception:
        dangerous_unicode = mixed_script

    (
        nearest_domain,
        nearest_label,
        nearest_suffix,
        damerau_distance,
        damerau_similarity,
        jaro_similarity,
    ) = _nearest_trusted_domain(
        registered_domain
    )

    exact_trusted_domain = int(
        registered_domain
        in TRUSTED_DOMAIN_SET
    )

    edit_insertion = int(
        damerau_distance == 1
        and len(registered_domain)
        == len(nearest_domain) + 1
    )

    edit_deletion = int(
        damerau_distance == 1
        and len(registered_domain) + 1
        == len(nearest_domain)
    )

    edit_transposition = int(
        damerau_distance == 1
        and _has_adjacent_transposition(
            registered_domain,
            nearest_domain,
        )
    )

    edit_substitution = int(
        damerau_distance == 1
        and len(registered_domain)
        == len(nearest_domain)
        and not edit_transposition
    )

    keyboard_substitution = int(
        edit_substitution
        and _keyboard_neighbor_substitution(
            registered_domain,
            nearest_domain,
        )
    )

    leet_or_digit_letter_match = int(
        registered_domain
        != nearest_domain
        and _leet_skeleton(
            registered_domain
        ) == nearest_domain
    )

    wrong_tld = int(
        domain_label == nearest_label
        and bool(candidate_tld)
        and candidate_tld
        != nearest_suffix
    )

    invalid_or_wrong_tld = int(
        not hostname_is_ip
        and (
            not bool(public_suffix)
            or (
                domain_label
                == nearest_label
                and candidate_tld
                != nearest_suffix
            )
        )
    )

    hyphen_insertion = int(
        registered_domain
        != nearest_domain
        and "-" in registered_domain
        and registered_domain.replace(
            "-",
            "",
        ) == nearest_domain
    )

    missing_dot_pattern = int(
        registered_domain
        != nearest_domain
        and domain_label.replace(
            "-",
            "",
        ) == nearest_domain.replace(
            ".",
            "",
        )
    )

    added_suspicious_word = int(
        nearest_label in domain_label
        and domain_label != nearest_label
        and any(
            word in domain_label
            for word in SUSPICIOUS_WORDS
        )
    )

    compact_subdomain = re.sub(
        r"[^a-z0-9]",
        "",
        subdomain.lower(),
    )

    brand_in_subdomain = int(
        registered_domain
        not in TRUSTED_DOMAIN_SET
        and any(
            trusted_label
            in compact_subdomain
            for (
                _,
                trusted_label,
                _,
            ) in TRUSTED_RECORDS
            if len(trusted_label) >= 4
        )
    )

    path_extension = PurePosixPath(
        path.lower()
    ).suffix

    maximum_label_length = max(
        (
            len(label)
            for label in hostname_labels
        ),
        default=0,
    )

    average_label_length = _safe_ratio(
        sum(
            len(label)
            for label in hostname_labels
        ),
        len(hostname_labels),
    )

    features = {
        "nearest_trusted_domain":
            nearest_domain,

        "url_length":
            url_length,

        "hostname_length":
            len(hostname),

        "registered_domain_length":
            len(registered_domain),

        "domain_label_length":
            len(domain_label),

        "subdomain_length":
            len(subdomain),

        "path_length":
            len(path),

        "query_length":
            len(query),

        "fragment_length":
            len(fragment),

        "tld_length":
            len(candidate_tld),

        "hostname_label_count":
            len(hostname_labels),

        "subdomain_count":
            len(
                [
                    label
                    for label
                    in subdomain.split(".")
                    if label
                ]
            ),

        "url_token_count":
            len(url_tokens),

        "host_token_count":
            len(hostname_tokens),

        "max_hostname_label_length":
            maximum_label_length,

        "average_hostname_label_length":
            average_label_length,

        "url_letter_count":
            url_letter_count,

        "url_digit_count":
            url_digit_count,

        "url_special_count":
            url_special_count,

        "url_letter_ratio":
            _safe_ratio(
                url_letter_count,
                url_length,
            ),

        "url_digit_ratio":
            _safe_ratio(
                url_digit_count,
                url_length,
            ),

        "url_special_ratio":
            _safe_ratio(
                url_special_count,
                url_length,
            ),

        "host_letter_count":
            hostname_letter_count,

        "host_digit_count":
            hostname_digit_count,

        "host_digit_ratio":
            _safe_ratio(
                hostname_digit_count,
                len(hostname),
            ),

        "url_entropy":
            _entropy(lowercase_url),

        "hostname_entropy":
            _entropy(hostname),

        "max_repeated_character_run":
            _maximum_character_run(
                hostname
            ),

        "url_dot_count":
            raw_url.count("."),

        "hostname_dot_count":
            hostname.count("."),

        "hostname_hyphen_count":
            hostname.count("-"),

        "url_underscore_count":
            raw_url.count("_"),

        "url_slash_count":
            raw_url.count("/"),

        "url_at_count":
            raw_url.count("@"),

        "url_question_count":
            raw_url.count("?"),

        "url_equal_count":
            raw_url.count("="),

        "url_ampersand_count":
            raw_url.count("&"),

        "url_percent_count":
            raw_url.count("%"),

        "is_https":
            int(
                parsed.scheme.lower()
                == "https"
            ),

        "hostname_is_ip":
            int(hostname_is_ip),

        "has_explicit_port":
            explicit_port,

        "has_userinfo":
            int(
                parsed.username is not None
                or parsed.password is not None
            ),

        "valid_tld":
            int(
                bool(public_suffix)
                or hostname_is_ip
            ),

        "invalid_tld":
            int(
                not public_suffix
                and not hostname_is_ip
            ),

        "has_punycode":
            has_punycode,

        "has_unicode":
            has_unicode,

        "mixed_script":
            mixed_script,

        "unicode_script_count":
            unicode_script_count,

        "dangerous_unicode_confusable":
            dangerous_unicode,

        "digit_letter_substitution_count":
            digit_letter_count,

        "has_digit_letter_substitution":
            int(
                digit_letter_count > 0
            ),

        "suspicious_word_count":
            suspicious_word_count,

        "has_suspicious_word":
            int(
                suspicious_word_count > 0
            ),

        "has_suspicious_file_extension":
            int(
                path_extension
                in SUSPICIOUS_EXTENSIONS
            ),

        "is_url_shortener":
            int(
                registered_domain
                in SHORTENER_DOMAINS
            ),

        "has_double_hyphen":
            int(
                "--" in hostname
            ),

        "label_starts_or_ends_hyphen":
            int(
                any(
                    label.startswith("-")
                    or label.endswith("-")
                    for label
                    in hostname_labels
                )
            ),

        "nearest_damerau_distance":
            damerau_distance,

        "nearest_damerau_similarity":
            damerau_similarity,

        "nearest_jaro_winkler_similarity":
            jaro_similarity,

        "exact_trusted_domain":
            exact_trusted_domain,

        "edit_insertion":
            edit_insertion,

        "edit_deletion":
            edit_deletion,

        "edit_substitution":
            edit_substitution,

        "edit_transposition":
            edit_transposition,

        "keyboard_neighbor_substitution":
            keyboard_substitution,

        "leet_or_digit_letter_match":
            leet_or_digit_letter_match,

        "hyphen_insertion":
            hyphen_insertion,

        "wrong_tld":
            wrong_tld,

        "invalid_or_wrong_tld":
            invalid_or_wrong_tld,

        "missing_dot_pattern":
            missing_dot_pattern,

        "added_suspicious_word":
            added_suspicious_word,

        "brand_in_subdomain":
            brand_in_subdomain,

        "misleading_subdomain_pattern":
            brand_in_subdomain,

        "high_brand_similarity":
            int(
                not exact_trusted_domain
                and (
                    damerau_similarity
                    >= 0.80
                    or jaro_similarity
                    >= 0.88
                )
            ),
    }

    return features


def model_feature_names() -> list[str]:
    """
    Return the numeric model-feature order.
    """

    sample_features = extract_url_features(
        "https://example.com/"
    )

    return [
        feature_name
        for feature_name
        in sample_features
        if feature_name
        != "nearest_trusted_domain"
    ]