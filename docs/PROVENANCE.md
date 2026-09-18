# Provenance and release transformations

Source: the authors' `CMV_reputation` working directory, inspected against `Hicss_conference_paper_0918.pdf`. This repository is a clean release with independent history. It does not include the older private `hazelkang/cmv` repository or its dataset/history.

## Included inputs

- H1: model columns from `cmv_experiment_data_19965_regression.csv`.
- Balance: covariate columns from `cmv_experiment_data_18590_balance_check.csv` and `cmv_experiment_data_17336.csv`.
- H2: the analysis variables used by the selected main and optional robustness scripts, from `cmv_experiment_data_17336_with_measures.csv`.
- Content measures: per-submission counts of empirical/non-empirical claims and quote words, extracted from `quantitative_rhetorical_results_new_cleaned.json` using the original measure builder's parsing and URL-replacement rules. No claim quotations are included. Invalid nested annotation JSON is treated as empty, as in the source builder; zero denominators produce zero unsmoothed shares.
- H3: selected per-turn scores from `merged_pipeline/unified_annotations_full.jsonl`; saved numeric package measures from `merged_pipeline/data_package_measures_llmclean.csv`.
- References: aggregate coefficient, rank-sum, table, and validation-score outputs from the same working directory.
- Claim validation: the `label` and `human_check` columns from the 200-row annotated validation workbook. The release recomputes precision/support only; it does not propagate the legacy script's artificial recall calculation.

## Transformations

Only required columns are exported. Usernames and original submission/comment IDs are replaced by release-specific sequential IDs; no lookup is distributed. Challenger membership and dyad/post joins are preserved. Topic labels are replaced by category codes preserving their partition. Raw posts, replies, timestamps, annotation evidence, reasoning text, API drivers, credentials, environments, and exploratory notebooks are not part of the release.

These are pseudonymized research records, not a guarantee that reidentification is impossible. Counts, covariates, and group memberships remain as required for replication. No direct identifiers or original free text are included.

Scripts use repository-relative paths. The H3 analysis requires the full score file and fails if it is absent, replacing an old fallback to the 100-dyad validation subset. Statistical formulas, sample definitions, seeds, and plotting conventions are retained. New scripts check input integrity, rebuild and compare claim measures, reproduce package tests, and compute the explicitly labeled conditional engagement sensitivity.

Package-measure decimal strings are preserved during identifier replacement. Package rank-sum replication uses pandas `float_precision='round_trip'` to reconstruct the original numeric ties. Default float parsing changes some ties in MATTR/receptiveness and yields small rank-test discrepancies despite virtually identical means.

`data/MANIFEST.json` records the released inputs' SHA-256 checksums. Source-to-release identifier lookup tables are not stored in this repository. The `reference/` directory contains archived research outputs; generated outputs are excluded from Git so that verification compares a fresh run with a distinct reference.
