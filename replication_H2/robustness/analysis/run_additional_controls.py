"""H2 appendix — Additional controls (poster-level only).

Replicates the paper's `\\subsection{Additional Controls}` table layout but with the
control set REPLACED by **poster (OP)-level characteristics only** — per the user's
instruction to "throw out all other controls":

  POSTER-level controls (kept):
    Poster Length     (poster_length / 100)
    Poster Status     (reputation_poster / 10)
    Poster Tenure     (poster_tenure / 100)

  DROPPED (challenger-level baseline + engagement/heuristic controls):
    Normalised Position, Pre-experiment Status (challenger), Response Length,
    Challenger Response Count, Challenger Quote Count, Challenger Link Count.

Spec, within each status stratum (Low/Mid/High):
    success ~ C(group)*C(pct_empirical_claims_Q4)
              + poster_length_100 + reputation_poster_10 + poster_tenure_100
  cluster-robust SE at author_comment.

Outputs:
  ../../appendix/table_h2_additional_controls.tex      (paper-style, \\label{tab:empirical_q_interact_posterctrl})
  ../results/additional_controls_posterlevel.csv       (tidy coefs)
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
TREAT, OUTCOME, CLUSTER, MOD = "group", "success", "author_comment", "pct_empirical_claims_Q4"
print(f"[load] {DATA.name}  N={len(df):,}  status_group counts: {df.stratum_group.value_counts().sort_index().to_dict()}")

# Poster-level controls (scaled to match the project's existing convention)
df['poster_length_100']    = df['poster_length']     / 100.0
df['poster_tenure_100']    = df['poster_tenure']     / 100.0
df['reputation_poster_10'] = df['reputation_poster'] / 10.0
df[MOD] = df[MOD].astype('category')

def stars(p): return '***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''
def coef_se(fit, term):
    if term in fit.params.index:
        return float(fit.params[term]), float(fit.bse[term]), float(fit.pvalues[term])
    return np.nan, np.nan, np.nan

STATUS = [(1, 'Low'), (2, 'Mid'), (3, 'High')]
fits = {}
for sg, lab in STATUS:
    sub = df[df['stratum_group']==sg].copy()
    fml = (f"{OUTCOME} ~ C({TREAT})*C({MOD}) "
           f"+ poster_length_100 + reputation_poster_10 + poster_tenure_100")
    fit = smf.ols(fml, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub[CLUSTER]})
    fits[lab] = (fit, len(sub))
    print(f"[fit] {lab:<5}  N={len(sub):,}  adj-R²={fit.rsquared_adj:.3f}")

# Collect coefs of interest
rows = []
for _, lab in STATUS:
    fit, n = fits[lab]
    # Interactions
    for q in (2, 3, 4):
        term = f"C({TREAT})[T.1]:C({MOD})[T.{q}]"
        c, se, p = coef_se(fit, term)
        rows.append(dict(status=lab, n=n, block='Interaction', term=f"T×Q{q}",
                         coef=c, se=se, p=p, sig=stars(p)))
    # Controls
    for term, pretty in [('poster_length_100',    'Poster Length ($\\ell^{op}_{j}/100$)'),
                         ('reputation_poster_10', 'Poster Status ($r^{op}_{j}/10$)'),
                         ('poster_tenure_100',    'Poster Tenure ($\\tau^{op}_{j}/100$)')]:
        c, se, p = coef_se(fit, term)
        rows.append(dict(status=lab, n=n, block='Control', term=pretty,
                         coef=c, se=se, p=p, sig=stars(p)))
    rows.append(dict(status=lab, n=n, block='AdjR2', term='Adj R²',
                     coef=float(fit.rsquared_adj), se=np.nan, p=np.nan, sig=''))
res = pd.DataFrame(rows)
res.to_csv(RESULTS_DIR / "additional_controls_posterlevel.csv", index=False)
print(f"\n[saved] {RESULTS_DIR}/additional_controls_posterlevel.csv")
print()
print(res.pivot_table(index=['block','term'], columns='status', values='coef', aggfunc='first').round(4).to_string())

# ---- LaTeX ----
def fmt_cell(c, se, p):
    star = f"^{{{stars(p)}}}" if stars(p) else ""
    sign = "-" if c < 0 else ""
    val = f"{abs(c):.3f}"
    return f"${sign}{val}{star}\\;({se:.3f})$"

def cells_for(term_label):
    """Return (Low, Mid, High) row records for a given term label."""
    return [res[(res.status==lab)&(res.term==term_label)].iloc[0] for _, lab in STATUS]

lines = [
    r"\begin{table}[H]", r"\TABLE",
    r"{Treatment $\times$ Factual\textendash Content Quartile Interactions \\",
    r"(selected terms and poster-level controls only)\label{tab:empirical_q_interact_posterctrl}}",
    r"{\small", r"\setlength{\tabcolsep}{6pt}",
    r"\begin{tabular}{lccc}",
    r"\toprule",
    r"& Low & Mid & High \\",
    rf"& $(N={fits['Low'][1]:,})$ & $(N={fits['Mid'][1]:,})$ & $(N={fits['High'][1]:,})$ \\",
    r"\midrule",
    r"\textit{Interaction terms} \\[2pt]",
]
for i, q in enumerate((2, 3, 4)):
    cs = cells_for(f"T×Q{q}")
    trailer = r"\\[6pt]" if i == 2 else r"\\"
    lines.append(rf"$T_i \times I(\text{{Factual}}_j = Q{q})$  & "
                 + " & ".join(fmt_cell(c['coef'], c['se'], c['p']) for c in cs) + " " + trailer)
lines.append(r"\textit{Poster-level controls} \\[2pt]")
control_terms = [r'Poster Length ($\ell^{op}_{j}/100$)',
                 r'Poster Status ($r^{op}_{j}/10$)',
                 r'Poster Tenure ($\tau^{op}_{j}/100$)']
for i, pretty in enumerate(control_terms):
    cs = cells_for(pretty)
    trailer = r"\\[4pt]" if i == len(control_terms)-1 else r"\\"
    lines.append(rf"{pretty}  & "
                 + " & ".join(fmt_cell(c['coef'], c['se'], c['p']) for c in cs) + " " + trailer)
lines += [
    rf"Adjusted $R^{{2}}$ & {fits['Low'][0].rsquared_adj:.3f} & {fits['Mid'][0].rsquared_adj:.3f} & {fits['High'][0].rsquared_adj:.3f} \\",
    r"\bottomrule", r"\end{tabular}}",
    r"{\textbf{Notes.} Standard errors clustered at the challenger level are in parentheses. "
    r"Significance: $^{*}p<.10$, $^{**}p<.05$, $^{***}p<.01$. "
    r"\textbf{Only poster-level controls included} (all challenger-level baseline and engagement/heuristic controls dropped per the appendix robustness pass): "
    r"\emph{Poster Length} = OP submission length scaled by 100 characters; "
    r"\emph{Poster Status} = OP's cumulative $\Delta$ points prior to the experiment, scaled by 10; "
    r"\emph{Poster Tenure} = OP's days active on r/ChangeMyView since first post/comment, scaled by 100.}",
    r"\end{table}",
]
out_tex = APPENDIX_DIR / "table_h2_additional_controls.tex"
with open(out_tex, "w") as f: f.write("\n".join(lines))
print(f"[saved] {out_tex}")

# Verification
print("\n" + "="*80)
print("Mine vs paper (interaction coefs; paper had challenger-level controls, mine poster-level)")
print("="*80)
paper = {
    ('T×Q2','Low'):(0.004,'',0.015),  ('T×Q2','Mid'):(-0.029,'',0.031),  ('T×Q2','High'):(-0.057,'**',0.028),
    ('T×Q3','Low'):(0.003,'',0.016),  ('T×Q3','Mid'):(-0.005,'',0.028),  ('T×Q3','High'):(-0.000,'',0.029),
    ('T×Q4','Low'):(0.042,'**',0.018),('T×Q4','Mid'):(-0.042,'',0.038),  ('T×Q4','High'):(-0.026,'',0.028),
}
for (term, col), (p_c, p_s, p_se) in paper.items():
    r = res[(res.status==col)&(res.term==term)].iloc[0]
    print(f"  {term:<6} {col:<5} paper: {p_c:+.3f}{p_s:<3} ({p_se:.3f})    mine: {r['coef']:+.3f}{r['sig']:<3} ({r['se']:.3f})")
