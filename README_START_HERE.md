# IT22131010 Domain & Typosquatting Detection

This starter builds the domain and URL lexical-analysis component with the single
PhiUSIIL dataset selected for the project.

## Fixed dataset location

The code expects the existing CSV at:

`D:\new pp2 component\datset phishuiil\phiusiil+phishing+url+dataset\PhiUSIIL_Phishing_URL_Dataset.csv`

The spelling `datset phishuiil` is intentional because it matches the current
folder shown in File Explorer. Do not move or rename the dataset yet.

## Phase 1: environment and dataset audit

1. Extract this starter ZIP directly into `D:\new pp2 component`.
2. Open Windows PowerShell.
3. Run:

```powershell
Set-Location "D:\new pp2 component"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
.\ai_research_env\Scripts\python.exe -X utf8 -u .\scripts\01_audit_phiusiil.py
```

The audit is read-only. It does not edit the original CSV. It writes:

- `reports\01_dataset_audit_summary.json`
- `reports\01_dataset_audit_summary.txt`
- `results\01_label_counts.csv`
- `results\01_column_scope.csv`

Send the complete PowerShell output or the generated text report before moving
to model training. The audit confirms the label mapping, duplicate rate, domain
conflicts, and the allowed URL/domain feature scope.

For the official UCI CSV, the expected core values are 235,795 rows, 56 columns,
100,945 phishing rows, 134,850 legitimate rows, and 425 duplicate URL rows. A
different result is a reason to stop and inspect the file before training.

## Component boundary

This component uses the raw URL, hostname, TLD, lexical structure, Unicode and
brand-similarity evidence. HTML, JavaScript, page-content and screenshot features
are excluded so that the module does not overlap other group members.

PhiUSIIL's official label mapping is:

- `0` = phishing
- `1` = legitimate

The training pipeline will convert this internally to `is_phishing`, where
`1` means phishing, to avoid accidental label reversal.

## Safety

Treat all URLs as offline strings. Do not click or open URLs from the phishing
rows. Model development does not require visiting any listed site.
