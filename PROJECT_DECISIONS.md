# Research and implementation decisions

## Dataset

Use only the PhiUSIIL Phishing URL Dataset as the labeled research dataset.
Public-suffix, IANA TLD and Unicode confusable specifications are parsing
standards, not additional labeled phishing datasets.

## Planned system

1. Normalize the supplied URL without visiting it.
2. Parse hostname, registered domain, subdomains, and public suffix.
3. Extract reproducible lexical and structural features.
4. Compare the domain with a trusted-domain reference fitted only on the
   legitimate part of the training split.
5. Combine a LightGBM probability with explicit typosquatting evidence.
6. Return `LEGITIMATE`, `PHISHING`, or `SUSPICIOUS_TYPOSQUAT`, with explanations.

For the group binary output, `SUSPICIOUS_TYPOSQUAT` maps to `PHISHING`.

## Leakage controls

- Split by registered domain, not by randomly mixing URL rows.
- Fit every learned vocabulary, probability table, and trusted-domain reference
  on the training split only.
- Do not use HTML/content columns beginning at `LineOfCode`.
- Audit or recreate PhiUSIIL's precomputed similarity/probability columns instead
  of assuming they are safe at inference time.
- Keep a separate controlled robustness suite for edits such as `kayak.c0m`.

## Evaluation

Report accuracy, precision, recall, F1, ROC-AUC, PR-AUC, MCC, specificity,
false-positive rate, false-negative rate, confusion matrix and inference time.
Ordinary test accuracy alone is not sufficient.

