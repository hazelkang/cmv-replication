# CMV reputation: replication data and code

Replication materials for **How Visible Status Shapes Persuasion and Engagement in Online Deliberation: Evidence from a Field Experiment**, by Hazel Hye Seung Kang, Emaad Manzoor, and Dokyun Lee. This release accompanies the September 18, 2026 manuscript.

The study reanalyzes an r/ChangeMyView field experiment that concealed reputation badges for randomly assigned challengers. The package reproduces the challenger-level persuasion models, empirical-claim-density moderation, and engagement figures from the saved research inputs.

**Read [Known differences](docs/KNOWN_DIFFERENCES.md) when comparing outputs with the manuscript.** Table 3's point estimates reproduce, but its printed standard errors and stars differ from the saved analysis. The engagement figures use a zero-filled 17,336-dyad frame, whereas the manuscript describes 8,401 reply dyads. This release preserves the saved analysis and separately reports a conditional sensitivity analysis; it does not silently change the estimand.

## Download

- [Download all code and data as a ZIP](https://github.com/hazelkang/cmv-replication/archive/refs/heads/main.zip).
- Browse the [data folder](data/) for individual CSV and JSONL files.
- Or clone: `git clone https://github.com/hazelkang/cmv-replication.git`.

All inputs for the default run are included. No API keys, model calls, Reddit login, or separate data download are required. Downloaded data contain analysis variables and saved annotation scores; usernames, Reddit identifiers, original conversation text, and annotation evidence quotations are omitted. Replacement IDs preserve the joins and challenger clusters used by the analysis. They are not original Reddit IDs.

## Run

Use Python **3.9–3.12**. The release pins the scientific libraries used for verification; Python 3.13+ is not supported by these pins.

```bash
cd cmv-replication
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python replicate.py
```

On Windows, activate with `.venv\Scripts\activate` instead. A successful run writes `verification.json` with the comparisons against archived research outputs. Input checksums, sample counts, cluster counts, and rebuilt moderator bins are also checked. The runner stops on a failed check.

```bash
# Add topic fixed effects, weighting, CEM, poster controls and alternative moderators:
python replicate.py --robustness

# Skip figures, for a faster numerical check:
python replicate.py --no-figures

# Expensive optional SIMEX: 500 simulation repetitions and 300 bootstraps by default
python replicate.py --simex
```

SIMEX repetitions are controlled with the `B`, `R`, and `SEED` environment variables. Reduced repetitions are suitable only for a smoke test, not for comparison with a full simulation run. SIMEX uses the inherited implementation, including its quadratic approximation labeled LOESS; see the limitations below.

## Find the results

| Manuscript result | Generated file | Entry point |
|---|---|---|
| Table 3: status concealment and persuasion | `replication_H1/tables/table_h1_treatment_effect.tex` | `replication_H1/analysis/run_h1_treatment_effect.py` |
| Table 4: empirical-claim-density moderation | `replication_H2/tables/table8_factual_density_PctClaims_nocontrols.tex` | `replication_H2/analysis/run_table8.py` |
| Figure 2: engagement overview | `replication_H3_ABC_indicators/merged_pipeline/figures_final/main/fig1_engagement_overview.png` | `replication_H3_ABC_indicators/merged_pipeline/10_final_figures.py` |
| Figure 3: representational transactivity by context | `replication_H3_ABC_indicators/merged_pipeline/figures_final/appendix/appx_A_composite_by_context.png` | Same figure script |
| H3 tests | `replication_H3_ABC_indicators/merged_pipeline/results_wilcoxon_unified_7strata_full.csv` | `07_full_pipeline.py` in that folder |
| H3 reply-only sensitivity | `replication_H3_ABC_indicators/merged_pipeline/results_llm_conditional_sensitivity.csv` | `scripts/engagement_sensitivity.py` |

Figures are written as both PNG and PDF. Figure 3 is the representational-composite figure, **not** the file named `fig2_warmth_by_context`. Balance tables are in `replication_H1/tables/`; optional H2 robustness outputs are in `replication_H2/appendix/` and `replication_H2/robustness/`.

View the verified [Figure 2 preview](docs/figures/figure2_engagement.png) and [Figure 3 preview](docs/figures/figure3_representational.png) without running the code.

## Design and conventions

- `group = 1`: status concealed (treatment); `group = 0`: status visible (control).
- Seven strata: 1–9, 10–19, 20–29, 30–39, 40–49, 50–99, and 100+ pre-experiment deltas.
- Three tiers: Low = stratum 1; Mid = strata 2–4; High = strata 5–7.
- H1: 19,965 challengers, OLS with HC1 standard errors.
- H2: 17,336 root-level debates; standard errors clustered by 2,784 challengers. Quartile Q1 is the reference category. The Table 4 specification excludes the optional position/reputation/length controls.
- H3: 8,286 dyads with scored turns, merged onto 17,336 debates. The remaining 9,050 rows receive zero indicator scores in the inherited full-frame analysis. The conditional sensitivity uses only the 8,286 scored dyads.
- Engagement tests compare visible vs. concealed status within strata using two-sided Mann–Whitney U (Wilcoxon rank-sum) tests, without a multiple-testing adjustment. Figure intervals use 1,000 bootstrap draws with fixed seeds.
- Legacy figure labels “Subjective/Value-driven” and “Factual/Utility-driven” mean empirical-claim-density quartiles Q1–Q2 and Q3–Q4 respectively. These labels should not be read as independent measurements of values or utility.

## Data and measurement scope

See the [data dictionary](docs/DATA_DICTIONARY.md), [provenance](docs/PROVENANCE.md), and [verification record](docs/VERIFICATION.md).

This is an **analysis replication from processed inputs**. The release reconstructs content measures from exported claim counts and engagement measures from per-turn binary scores. It does not reproduce collection of Reddit data, human coding judgments, fresh LLM annotations, or text-based package feature extraction. Saved package measures are included for reproducible downstream tests and figures. The engagement annotation prompt is provided in [prompts/engagement_unified.txt](prompts/engagement_unified.txt); it is documentation and is never executed by the default pipeline.

The original working directory contains exploratory variants beyond this release. This package selects the final merged engagement pipeline and the H1/H2 analysis scripts identified in the source audit. The optional SIMEX implementation is included for transparency; reproducing its computations does not validate its modeling assumptions. The separate historical MC-SIMEX exploration is outside this release.

## Citation and reuse

Please cite the accompanying paper and this repository when using these materials. See [CITATION.cff](CITATION.cff). No additional blanket license or permission to redistribute third-party Reddit content is asserted by this release.
