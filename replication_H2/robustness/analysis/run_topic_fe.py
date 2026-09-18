"""Selection-robustness check — Topic Fixed Effects.

Same interaction specification as the CEM appendix table, but on the FULL 17,336 sample with
super-cluster (topic) fixed effects added. Replicates the paper's
\\subsubsection{Topic Fixed Effects} appendix table.

Within each status stratum (Low/Mid/High) fit:

    success_ij = α + β_t T_i
               + Σ_{q=2..4} γ_q I(Factual_j = q)
               + Σ_{q=2..4} δ_q  T_i · I(Factual_j = q)
               + position_z + reputation_10 + word_count_100
               + C(super_cluster)
               + ε_ij

robust SE clustered at author_comment. Report the three δ_q per status stratum.

Outputs:
  results/topic_fe_factual_interaction.csv     — tidy coefs + SE + p, by status × Q
  tables/table_h2_topic_fe_factual_interaction.tex
"""
from __future__ import annotations
import os, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf

HERE = Path(__file__).parent
REPO = Path(__file__).resolve().parents[3] / "data"
RESULTS_DIR = (HERE / ".." / "results").resolve()
TABLES_DIR  = (HERE / ".." / ".." / "appendix").resolve()   # paper-appendix tables
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(TABLES_DIR, exist_ok=True)

DATA = REPO / "cmv_experiment_data_17336_with_measures.csv"
df = pd.read_csv(DATA, low_memory=False)
print(f"[load] {DATA.name}  N={len(df):,}")
print(f"[load] super_cluster levels: {df['super_cluster'].nunique()}  | NaN={df['super_cluster'].isna().sum()}")
print(f"[load] status_group counts: {df.stratum_group.value_counts().sort_index().to_dict()}")

TREAT, OUTCOME, CLUSTER, MOD, TOPIC = "group", "success", "author_comment", "pct_empirical_claims_Q4", "super_cluster"

# Baseline controls (position_z, reputation_10, word_count_100) dropped per user request.
df[MOD]   = df[MOD].astype('category')
df[TOPIC] = df[TOPIC].astype('category')

def stars(p): return '***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''

STATUS = [(1,'Low Status'), (2,'Mid Status'), (3,'High Status')]
fits = {}
for sg, lab in STATUS:
    sub = df[df['stratum_group']==sg].copy()
    fml = f"{OUTCOME} ~ C({TREAT})*C({MOD}) + C({TOPIC})"
    fit = smf.ols(fml, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub[CLUSTER]})
    fits[sg] = (fit, len(sub))
    print(f"[fit] {lab}: N={len(sub):,}  adj-R²={fit.rsquared_adj:.3f}")

# Extract T × Q{2,3,4}
rows=[]
for sg, lab in STATUS:
    fit, n = fits[sg]
    for q in [2,3,4]:
        term = f"C({TREAT})[T.1]:C({MOD})[T.{q}]"
        c, se, p = (float(fit.params[term]), float(fit.bse[term]), float(fit.pvalues[term])) if term in fit.params.index else (np.nan,np.nan,np.nan)
        rows.append(dict(status=lab, quartile=f"Q{q}", coef=c, se=se, p=p, sig=stars(p),
                         n=n, adj_r2=float(fit.rsquared_adj)))
inter = pd.DataFrame(rows)
inter.to_csv(RESULTS_DIR/"topic_fe_factual_interaction.csv", index=False)

print("\n[interaction coefs T × I(Factual=Qq) with topic FE, full sample]")
piv = inter.pivot(index='quartile', columns='status', values='coef').round(3)
sigp = inter.pivot(index='quartile', columns='status', values='sig')
for q in piv.index:
    line=f"  {q}:"
    for s in ['Low Status','Mid Status','High Status']:
        line += f"  {s.split()[0]}={piv.loc[q,s]:+.3f}{sigp.loc[q,s]}"
    print(line)

# LaTeX table — paper format
def fmt_c(c,p):
    sign = "" if c<0 else "\\phantom{-}"
    star = ("\\("+stars(p).replace('*','*')+"\\)") if stars(p) else ""
    # Reuse the paper's \(^{**}\) style:
    if stars(p):
        return f"{sign}{c:.3f}\\(^{{{stars(p)}}}\\)"
    return f"{sign}{c:.3f}"
def fmt_s(s): return f"({s:.3f})"

lines = [
    r"\begin{table}[htbp]", r"\TABLE",
    r"{Treatment (\(T_i\)) Effects by Factual\textendash Evidence Density with Topic FE\label{tab:factual_topicfe}}",
    r"{\small", r"\begin{tabular}{lccc}", r"\toprule",
    r"& \multicolumn{3}{c}{\textbf{Dependent Variable: Persuasion Success} (\(Y_{ij}\))} \\",
    r"\cmidrule(lr){2-4}",
    r"& Low Status & Mid Status & High Status \\",
    r"\midrule",
]
for q in [2,3,4]:
    rs = [inter[(inter.status==lab)&(inter.quartile==f"Q{q}")].iloc[0] for _,lab in STATUS]
    lines.append(
        rf"\(T_i \times I(\text{{Factual}}_j=Q{q})\) & "
        + " & ".join(fmt_c(r['coef'], r['p']) for r in rs) + r" \\"
    )
    lines.append(r"                                   & "
        + " & ".join(fmt_s(r['se']) for r in rs) + r" \\[2pt]"
    )
lines += [r"\midrule",
    r"\textit{Topic FE}                  & Yes & Yes & Yes \\[2pt]",
    r"\textit{Baseline controls}         & No & No & No \\[2pt]",
    r"\midrule",
    rf"Observations                       & {fits[1][1]:,} & {fits[2][1]:,} & {fits[3][1]:,} \\".replace(",","{,}"),
    rf"Adjusted \(R^{{2}}\)                 & {fits[1][0].rsquared_adj:.3f} & {fits[2][0].rsquared_adj:.3f} & {fits[3][0].rsquared_adj:.3f} \\",
    r"\bottomrule", r"\end{tabular}}",
    r"{Baseline category is \(Q1\) (lowest factual density).",
    r"Coefficients are interaction effects \(T_i \times I(\text{Factual}=Qq)\) estimated separately within each status stratum;",
    r"robust standard errors clustered at the challenger level are in parentheses. Topic fixed effects based on super-cluster categories are included.",
    r"\(^{*}p<0.10\), \(^{**}p<0.05\), \(^{***}p<0.01\).}",
    r"\end{table}",
]
out_tex = TABLES_DIR / "table_h2_topic_fe_factual_interaction.tex"
with open(out_tex, "w") as f: f.write("\n".join(lines))
print(f"\n[saved] {out_tex}")
print(f"[saved] {RESULTS_DIR}/topic_fe_factual_interaction.csv")
