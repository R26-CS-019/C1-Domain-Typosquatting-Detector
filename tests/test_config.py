from src.config import DATASET_LABEL_NAMES, DEFAULT_DATASET_PATH, classify_column


def test_official_label_mapping_is_not_reversed() -> None:
    assert DATASET_LABEL_NAMES == {0: "phishing", 1: "legitimate"}


def test_default_dataset_filename() -> None:
    assert DEFAULT_DATASET_PATH.name == "PhiUSIIL_Phishing_URL_Dataset.csv"


def test_component_scope_excludes_html_content() -> None:
    assert classify_column("URL") == "raw_url_domain_input"
    assert classify_column("NoOfSubDomain") == "url_domain_candidate_verify"
    assert classify_column("LineOfCode") == "html_content_out_of_scope"
    assert classify_column("HasPasswordField") == "html_content_out_of_scope"

