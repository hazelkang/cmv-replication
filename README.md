# Replication code for the HICSS paper

This repository contains replication code and processed analysis data for **“How Visible Status Shapes Persuasion and Engagement in Online Deliberation: Evidence from a Field Experiment”** by Hazel Hye Seung Kang, Emaad Manzoor, and Dokyun Lee (September 18, 2026 manuscript).

The code reproduces analyses of reputation-badge visibility, persuasion, and decision-maker engagement on Reddit’s r/ChangeMyView. Data are included; usernames and original conversation text are removed. No API keys are required.

## Run

[Download the repository](https://github.com/hazelkang/cmv-replication/archive/refs/heads/main.zip), extract it, and run with Python 3.9–3.12:

```bash
python -m pip install -r requirements.txt
python replicate.py
```

Use `python replicate.py --robustness` to include the additional moderation analyses.

## Paper results

Outputs are saved in `paper_results/`.

| HICSS paper item | Output |
|---|---|
| Table 3: Treatment effects of concealing status | `table3_replication_models.tex` |
| Table 4: Effects moderated by empirical-claim density | `table4_empirical_claim_density.tex` |
| Figure 2: Decision-maker engagement by reputation stratum | `figure2_engagement.png` / `.pdf` |
| Figure 3: Representational transactivity by stratum and context | `figure3_representational_by_context.png` / `.pdf` |

Figures 2–3 match the images embedded in the paper exactly; Table 4 matches numerically. Table 3’s standard errors/significance and the engagement sample description have documented differences. See the [paper comparison](docs/PAPER_RESULTS.md).

Further details: [data dictionary](docs/DATA_DICTIONARY.md) · [verification](docs/VERIFICATION.md) · [participant privacy](docs/PRIVACY.md) · [citation](CITATION.cff).
