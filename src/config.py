"""Shared project paths, label semantics, and PhiUSIIL feature scope."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "datset phishuiil"
    / "phiusiil+phishing+url+dataset"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"

# Official PhiUSIIL mapping. The training pipeline will later derive
# is_phishing = 1 - label so that positive-class metrics mean phishing.
DATASET_LABEL_NAMES = {0: "phishing", 1: "legitimate"}

REQUIRED_COLUMNS = {"URL", "Domain", "TLD", "label"}

RAW_URL_DOMAIN_COLUMNS = {"URL", "Domain", "TLD"}

REPRODUCIBLE_URL_COLUMNS = {
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
}

AUDIT_OR_RECREATE_COLUMNS = {
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
}

# These depend on downloaded page content, HTML, scripts, forms, images, or
# page metadata and are outside this component's approved boundary.
OUT_OF_SCOPE_COLUMNS = {
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "Title",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
}


def classify_column(column: str) -> str:
    """Return the approved handling category for a dataset column."""
    if column == "label":
        return "target"
    if column == "FILENAME":
        return "dataset_identifier_exclude"
    if column in RAW_URL_DOMAIN_COLUMNS:
        return "raw_url_domain_input"
    if column in REPRODUCIBLE_URL_COLUMNS:
        return "url_domain_candidate_verify"
    if column in AUDIT_OR_RECREATE_COLUMNS:
        return "precomputed_audit_or_recreate"
    if column in OUT_OF_SCOPE_COLUMNS:
        return "html_content_out_of_scope"
    return "unknown_review_required"

