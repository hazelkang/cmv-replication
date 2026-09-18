"""Alternative mechanism explorations — poster-level heuristics moderating the status effect.

Replicates the paper's appendix table `tab:poster_heuristics`. Within each status stratum
(Low/Mid/High), regress `success` on `group × C(Mod_Q)` for three OP-level moderators:
  • poster_length_Q       — argument length in characters (quartile of the OP's submission length)
  • reputation_poster_Q   — cumulative Δ points the poster earned before the experiment
  • poster_tenure_Q       — days active on r/ChangeMyView since the poster's first post/comment

NO baseline controls (per the user's instruction). Cluster-robust SE at the challenger level.
Reports T × I(Mod=Q_k) interaction coefficients for k = 2, 3, 4 (Q1 baseline).

Outputs:
  ../../appendix/table_h2_poster_heuristics.tex           (paper-style appendix table)
  ../results/poster_heuristics_interactions.csv           (tidy coefs)
"""
from __future__ import annotations
import os, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

HERE = Path(__file__).parent
REPO = Path(__file__).resolve().parents[3] / "data"
RESULTS_DIR = (HERE / ".." / "results").resolve()
APPENDIX_DIR = (HERE / ".." / ".." / "appendix").resolve()
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(APPENDIX_DIR, exist_ok=True)

DATA = REPO / "cmv_experiment_data_17336_with_measures.csv"
df = pd.read_csv(DATA, low_memory=False)
TREAT, OUTCOME, CLUSTER = "group", "success", "author_comment"
print(f"[load] {DATA.name}  N={len(df):,}  status_group counts: {df.stratum_group.value_counts().sort_index().to_dict()}")

MODERATORS = [
    ("poster_length_Q",     "PosterLen", r"\textit{Poster length quartiles} (argument length in characters)"),
    ("reputation_poster_Q", "PosterRep", r"\textit{Poster status quartiles} (cumulative $\Delta$ points prior to experiment)"),
    ("poster_tenure_Q",     "PosterTen", r"\textit{Poster tenure quartiles} (days active on CMV since first post/comment)"),
]
STATUS = [(1, "Low"), (2, "Mid"), (3, "High")]

def stars(p): return '***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''

rows = []
fits = {}
for mod_col, mod_lab, mod_pretty in MODERATORS:
    for sg, sg_lab in STATUS:
        sub = df[df['stratum_group']==sg].copy()
        sub[mod_col] = sub[mod_col].astype('category')
        fml = f"{OUTCOME} ~ C({TREAT}) * C({mod_col})"
        fit = smf.ols(fml, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub[CLUSTER]})
        fits[(mod_col, sg)] = (fit, len(sub))
        for q in (2, 3, 4):
            term = f"C({TREAT})[T.1]:C({mod_col})[T.{q}]"
            c, se, p = (float(fit.params[term]), float(fit.bse[term]), float(fit.pvalues[term])) \
                       if term in fit.params.index else (np.nan,np.nan,np.nan)
            rows.append(dict(moderator=mod_lab, mod_col=mod_col, mod_pretty=mod_pretty,
                             status=sg_lab, quartile=f"Q{q}",
                             coef=c, se=se, p=p, sig=stars(p), n=len(sub),
                             adj_r2=float(fit.rsquared_adj)))
res = pd.DataFrame(rows)
res.to_csv(RESULTS_DIR / "poster_heuristics_interactions.csv", index=False)
print(f"[saved] {RESULTS_DIR}/poster_heuristics_interactions.csv")
print()
for mod_col, mod_lab, _ in MODERATORS:
    print(f"=== {mod_lab} ===")
    pv = res[res.moderator==mod_lab].pivot(index='quartile', columns='status', values='coef').round(3)
    sg = res[res.moderator==mod_lab].pivot(index='quartile', columns='status', values='sig')
    for q in pv.index:
        print("  " + q + ":  " + "  ".join(f"{s} {pv.loc[q,s]:+.3f}{sg.loc[q,s]}" for s in ['Low','Mid','High']))

# ---- LaTeX table (paper format) ----
def fmt_cell(c, p):
    star = f"^{{{stars(p)}}}" if stars(p) else ""
    return f"${c:.3f}{star}$" if c < 0 else f"${c:.3f}{star}$"  # paper uses bare positive (no \phantom)
def fmt_se(s): return f"$({s:.3f})$"

N_low  = fits[("poster_length_Q", 1)][1]
N_mid  = fits[("poster_length_Q", 2)][1]
N_high = fits[("poster_length_Q", 3)][1]

lines = [
    r"\begin{table}[H]", r"\TABLE",
    r"{Poster\textendash level heuristics moderating status effects\label{tab:poster_heuristics}}",
    r"{\small",
    r"\setlength{\tabcolsep}{5pt}",
    r"\begin{tabular}{lccc}",
    r"\toprule",
    r"& \multicolumn{3}{c}{\textbf{Interaction coefficient} $\;(\delta)$} \\",
    r"\cmidrule(lr){2-4}",
    r"\textbf{Interaction term} & Low & Mid & High \\",
    r"\midrule",
]
for i, (mod_col, mod_lab, mod_pretty) in enumerate(MODERATORS):
    lines.append(rf"\multicolumn{{4}}{{l}}{{{mod_pretty}}} \\[2pt]")
    for q_idx, q in enumerate((2, 3, 4)):
        cells = [res[(res.moderator==mod_lab)&(res.quartile==f"Q{q}")&(res.status==s)].iloc[0]
                 for s in ('Low','Mid','High')]
        lines.append(
            rf"$T_i \!\times I(\text{{{mod_lab}}}=Q_{q})$ & "
            + " & ".join(fmt_cell(c['coef'], c['p']) for c in cells) + r" \\"
        )
        # The SE row gets extra spacing AFTER the last quartile in the block (paper uses
        # \\[6pt] between blocks, \\[8pt] before the Controls/Observations rows).
        is_last_in_block = (q_idx == 2)
        is_last_block    = (i == len(MODERATORS) - 1)
        if is_last_in_block:
            trailer = r"\\[8pt]" if is_last_block else r"\\[6pt]"
        else:
            trailer = r"\\"
        lines.append(r"                                      & "
            + " & ".join(fmt_se(c['se']) for c in cells) + " " + trailer
        )
lines += [
    rf"\textbf{{Controls}}      & No & No & No \\",   # user requested NO controls
    rf"\textbf{{Observations}}  & {N_low:,} & {N_mid:,} & {N_high:,} \\".replace(",","\\,"),
    r"\bottomrule",
    r"\end{tabular}}",
    r"{\textbf{Notes.} Each $\delta$ is the coefficient on $T_i \times I(\cdot)$, where $T_i$ is the treatment indicator (status concealed) and $I(\cdot)$ is the quartile dummy for the moderator.",
    r"Moderator definitions: \emph{Poster length} = argument length in characters; \emph{Poster status} = cumulative $\Delta$ points the poster earned before the experiment; \emph{Poster tenure} = days since the poster's first activity (post or comment) on r/ChangeMyView.",
    r"Standard errors clustered at the challenger level in parentheses. \textbf{No baseline controls.}",
    r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}",
    r"\end{table}",
]
out_tex = APPENDIX_DIR / "table_h2_poster_heuristics.tex"
with open(out_tex, "w") as f: f.write("\n".join(lines))
print(f"\n[saved] {out_tex}")

# ---- verification against paper ----
print("\n" + "="*78)
print("VERIFICATION vs paper")
print("="*78)
paper = {
    ('PosterLen',2,'Low'):  (-0.030, '*', 0.017),
    ('PosterLen',2,'Mid'):  (-0.023, '',  0.030),
    ('PosterLen',2,'High'): ( 0.002, '',  0.026),
    ('PosterLen',3,'Low'):  (-0.027, '',  0.017),
    ('PosterLen',3,'Mid'):  ( 0.000, '',  0.035),
    ('PosterLen',3,'High'): ( 0.025, '',  0.031),
    ('PosterLen',4,'Low'):  (-0.013, '',  0.018),
    ('PosterLen',4,'Mid'):  (-0.028, '',  0.036),
    ('PosterLen',4,'High'): (-0.004, '',  0.026),
    ('PosterRep',2,'Low'):  (-0.012, '',  0.015),
    ('PosterRep',2,'Mid'):  (-0.059, '*', 0.031),
    ('PosterRep',2,'High'): (-0.002, '',  0.025),
    ('PosterRep',3,'Low'):  ( 0.014, '',  0.019),
    ('PosterRep',3,'Mid'):  ( 0.004, '',  0.033),
    ('PosterRep',3,'High'): (-0.008, '',  0.030),
    ('PosterRep',4,'Low'):  (-0.020, '',  0.018),
    ('PosterRep',4,'Mid'):  (-0.015, '',  0.036),
    ('PosterRep',4,'High'): ( 0.021, '',  0.030),
    ('PosterTen',2,'Low'):  (-0.004, '',  0.023),
    ('PosterTen',2,'Mid'):  ( 0.018, '',  0.051),
    ('PosterTen',2,'High'): ( 0.062, '',  0.045),
    ('PosterTen',3,'Low'):  (-0.015, '',  0.021),
    ('PosterTen',3,'Mid'):  ( 0.015, '',  0.043),
    ('PosterTen',3,'High'): ( 0.035, '',  0.032),
    ('PosterTen',4,'Low'):  (-0.010, '',  0.016),
    ('PosterTen',4,'Mid'):  (-0.059, '**',0.030),
    ('PosterTen',4,'High'): (-0.014, '',  0.027),
}
print(f"{'mod':<11}{'Q':<3}{'status':<7}{'paper':<22}{'mine':<22}  match?")
print("-"*78)
match_sign = match_sig = total = 0
for (mod, q, s), (p_c, p_s, p_se) in paper.items():
    r = res[(res.moderator==mod)&(res.quartile==f"Q{q}")&(res.status==s)].iloc[0]
    same_sign = (np.sign(p_c) == np.sign(r['coef'])) or (abs(p_c) < 0.005 and abs(r['coef']) < 0.01)
    same_sig  = (p_s == r['sig']) or (p_s == '*' and r['sig'] == '*') or (p_s == '**' and r['sig'] == '**')
    match_sign += same_sign; match_sig += same_sig; total += 1
    flag = "✓" if (same_sign and same_sig) else ("≈sig" if same_sign else "DIFFER")
    print(f"{mod:<11}{q:<3}{s:<7}{p_c:+.3f}{p_s:<3} ({p_se:.3f})    "
          f"{r['coef']:+.3f}{r['sig']:<3} ({r['se']:.3f})    {flag}")
print("-"*78)
print(f"signs match: {match_sign}/{total}    significance match: {match_sig}/{total}")
