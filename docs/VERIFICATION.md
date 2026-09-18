# Verification record

Verified on 2026-09-18 in a newly created macOS Python 3.9.6 virtual environment using `requirements.txt`. The input package is approximately 15.6 MB uncompressed and needs no API access after dependency installation.

Commands exercised:

```bash
python replicate.py --robustness
python scripts/validate_claim_labels.py
python scripts/verify_inputs.py
python scripts/verify_results.py
```

The core-plus-robustness run took approximately 42 seconds on the test machine. Timing depends on the machine; this is not a performance guarantee.

| Check | Result |
|---|---|
| Eight released input files | SHA-256 checksums pass |
| H1 / H2 samples | 19,965 challengers / 17,336 debates; H2 has 2,784 challenger clusters |
| Rebuilt empirical-content measures and bins | Agree with released H2 analysis columns |
| Claim-annotation validation | Precision/support for 40 labels reproduced from 200 human-check rows |
| H1 saved coefficient output | All 35 coefficients, standard errors, test statistics, and p-values reproduce within numeric tolerance |
| H2 main table | Exact text match with archived no-control empirical-claim table, including the manuscript's displayed Table 4 values |
| H3 indicator tests | All 273 saved tests reproduce: group counts, means, differences, U statistics, and p-values |
| H3 package-measure tests | All 294 saved tests reproduce using round-trip float parsing |
| Figures 2 and 3 source PNGs | Pixel-identical to the corresponding saved working-directory images |
| H2 optional robustness | Topic FE, weighting, CEM, alternative moderators, and poster-control scripts complete; aggregate numeric outputs are checked against saved reference CSVs |
| CEM matched sample | 12,802 debates, agreeing with the saved run |
| Release screening | Code parses; no matched credential-token patterns or local home-directory paths; published data use replacement IDs and restricted numeric/score schemas |

The expensive SIMEX simulation was not rerun at its full default settings. Its script is included as an optional analysis and is not covered by the passed reference comparisons above. Fresh LLM calls, text-feature extraction, and independent human annotation were not performed.

Input checks run before modeling. `verification.json` is written after the output comparisons succeed; optional robustness references are checked when their generated result files exist. This record reports computational agreement with saved outputs. See [Known differences](KNOWN_DIFFERENCES.md) for the manuscript discrepancies that remain.
