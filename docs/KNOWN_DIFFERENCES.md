# Differences from the September 18 manuscript

The release preserves the saved research specifications. The discrepancies below are visible to users so that a successful replication is not mistaken for agreement with every printed statement.

## Table 3

The saved H1 analysis reproduces the printed point estimates, but its HC1 uncertainty estimates differ:

| Effect | Coefficient | Manuscript SE | Saved-analysis SE |
|---|---:|---:|---:|
| Pooled | -0.0006 | 0.0010 | 0.0014 |
| Low | -0.0009 | 0.0010 | 0.0015 |
| Mid | 0.0120 | 0.0050 | 0.0046 |
| High | -0.0232 | 0.0120 | 0.0115 |

The manuscript prints two stars for Mid; the saved analysis assigns three using the same thresholds. The release retains the original model, including redundant subgroup dummies alongside stratum fixed effects, to reproduce its saved output. Its caption retains an older Manzoor reference year; the companion manuscript cites Manzoor et al. (2022). The source audit does not establish why the printed uncertainty values differ. They must not be manually substituted into calculated results.

## H3 sample and measurement

The manuscript describes 8,401 reply dyads. The saved cleaning documentation reports that 115 are excluded, leaving 8,286 dyads with scored turns. The final analysis merges these onto the 17,336-debate experiment frame and zero-fills 9,050 missing indicator rows. Figures 2–3 derive from that full-frame analysis, not an 8,401-row conditional analysis.

`sample_flow.csv` reports these counts. `results_llm_conditional_sensitivity.csv` is a new, separately labeled sensitivity restricted to the 8,286 dyads with annotations. It is not a replacement for the archived results. Zero-filling conflates lack of a scored annotation with zero measured engagement; missing scores must not automatically be interpreted as verified absence of a reply.

The final eight indicators comprise seven LLM-coded binary indicators and deterministic quoting (A1), not eight independently validated LLM judgments. The saved validation protocol describes iterative refinement, including revalidation of B1 and removal of B3. Its nominal 100-dyad validation source is distinct from the 96 sampled indicator judgments. The release includes the saved summary score files without claiming that the entire process constitutes an independent held-out validation study.

## Moderation interpretation

Table 4 rows labeled treatment × Q2/Q3/Q4 are interaction coefficients relative to Q1, not the full treatment effect within that quartile. For example, the low-tier Q4 treatment effect combines the treatment coefficient (-0.004 at displayed precision) and Q4 interaction (+0.043), giving approximately +0.039. The high-tier Q2 effect analogously combines +0.011 and -0.057, approximately -0.046. Statements describing the interaction alone as a quartile-specific treatment effect should be read with that distinction.

## Optional robustness

The optional scripts reproduce their inherited formulas rather than redefining them to fit narrative claims. The weighting script uses `word_count` in its propensity model; its suitability as a pre-treatment variable should be checked against the data provenance. SIMEX perturbs observations on a logit scale, re-bins within the current subset, and labels one global quadratic extrapolation as LOESS. This is not a separate locally fitted extrapolator. These implementation details limit what can be inferred from a successful computational run.
