"""H1 robustness — 1:1 Coarsened Exact Matching (CEM), faithful replication of the paper's
Appendix table set.

Replicates `FINAL.ipynb` cells 103–116 verbatim: uses the same `cem.coarsen` (Iacus-King-Porro
adaptive H ∈ [1,10] minimising L1 imbalance) on 7 pre-treatment covariates, then 1-to-1 random
matching within each joint coarsened stratum, then runs the treatment×factual-density interaction
within each status group on the matched sample.

Produces two paper-style tables:
  tables/table_h2_cem_balance.tex            ← Original 17,336 vs Matched 12,802 (means + Welch t)
  tables/table_h2_cem_factual_interaction.tex ← T × I(Factual=Q2/Q3/Q4), by status stratum

CSV outputs:
  results/cem_matched_dyads.csv     — matched dataset
  results/cem_balance.csv           — full balance table (pre + post)
  results/cem_factual_interaction.csv — coefs from the 3 stratified interaction fits
"""
from __future__ import annotations
import os, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
from cem.coarsen import coarsen

HERE = Path(__file__).parent
REPO = Path(__file__).resolve().parents[3] / "data"
RESULTS_DIR = (HERE / ".." / "results").resolve()
TABLES_DIR  = (HERE / ".." / ".." / "appendix").resolve()   # paper-appendix tables
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(TABLES_DIR, exist_ok=True)

DATA = REPO / "cmv_experiment_data_17336_with_measures.csv"
df = pd.read_csv(DATA, low_memory=False)
print(f"[load] {DATA.name}  N={len(df):,}  (T={int((df.group==1).sum()):,}, C={int((df.group==0).sum()):,})")

MATCH_VARS = ["comment_karma","link_karma","tenure_i","gold",
              "num_cmv_post","num_cmv_comments","average_comment_length"]
TREAT, OUTCOME, CLUSTER = "group", "success", "author_comment"
MODERATOR = "pct_empirical_claims_Q4"

# ---- ORIGINAL-SAMPLE BALANCE on the full 17,336 (NaN handled per-variable) ----
def balance_row(data, v, label):
    t = data.loc[data[TREAT]==1, v].dropna()
    c = data.loc[data[TREAT]==0, v].dropna()
    tstat, p = stats.ttest_ind(t, c, equal_var=False, nan_policy='omit')
    return dict(sample=label, var=v, N_C=len(c), N_T=len(t),
                mean_C=c.mean(), mean_T=t.mean(), t_stat=float(tstat), p=float(p))

orig_bal = [balance_row(df, v, "Original 17,336") for v in MATCH_VARS]

# ---- 1-to-1 CEM via the cem package (same call as FINAL.ipynb cell 111) ----
# Pass the FULL 17,336 to cem.coarsen so NaN rows form their own strata (and can match each
# other within the NaN cluster). This is what the published table N=12,802 implies.
d = df.copy()
X = coarsen(d[MATCH_VARS + [TREAT]], TREAT, "l1")
strata_id = X[MATCH_VARS].astype(str).agg('_'.join, axis=1)
d["cem_stratum"] = strata_id.values
print(f"[coarsen] cem.coarsen(measure='l1', H∈[1,10]) → {d['cem_stratum'].nunique():,} joint strata")

rng = np.random.default_rng(42); keep=[]
for stratum, g in d.groupby("cem_stratum"):
    t = g.index[g[TREAT]==1].to_numpy(); c = g.index[g[TREAT]==0].to_numpy()
    k = min(len(t), len(c))
    if k == 0: continue
    keep.extend(rng.choice(t, k, replace=False).tolist())
    keep.extend(rng.choice(c, k, replace=False).tolist())
matched = d.loc[keep].copy(); matched["cem_weight"] = 1
print(f"[match] 1-to-1 matched: {len(matched):,}  (T={int((matched[TREAT]==1).sum()):,}, C={int((matched[TREAT]==0).sum()):,})")

# ---- MATCHED-SAMPLE BALANCE ----
match_bal = [balance_row(matched, v, f"Matched {len(matched):,}") for v in MATCH_VARS]

bal_df = pd.DataFrame(orig_bal + match_bal)
bal_df.to_csv(RESULTS_DIR/"cem_balance.csv", index=False)
matched.to_csv(RESULTS_DIR/"cem_matched_dyads.csv", index=False)

# ---- BALANCE TABLE (paper-style) ----
def stars(p): return '***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''
def fmt_mean(x, var):
    """Format means: large karma vars get a thin-space thousands separator; rates ≤ 1 → 3 decimals."""
    if var in ("gold","num_cmv_post"):
        return f"{x:.3f}"
    if abs(x) >= 1000:
        return f"{int(round(x)):,}".replace(",","\\,")
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.2f}"
def fmt_t(t,p): return f"{t:.2f}{stars(p)}"

PRETTY = {"comment_karma":"$\\text{comment\\_karma}_i$",
          "link_karma":"$\\text{link\\_karma}_i$",
          "tenure_i":"$\\text{tenure}_i$",
          "gold":"$\\text{has\\_gold}_i$",
          "num_cmv_post":"$\\text{num\\_cmv\\_post}_i$",
          "num_cmv_comments":"$\\text{num\\_cmv\\_comments}_i$",
          "average_comment_length":"$\\text{average\\_comment\\_length}_i$"}

orig_d = {r['var']:r for r in orig_bal}
matc_d = {r['var']:r for r in match_bal}
n_orig_t = int((df[TREAT]==1).sum()); n_orig_c = int((df[TREAT]==0).sum())
n_matc_t = int((matched[TREAT]==1).sum()); n_matc_c = int((matched[TREAT]==0).sum())

lines = [
    r"\begin{table}[H]", r"\TABLE",
    r"{Covariate Balance Before and After 1-to-1 Matching\label{tab:balance_before_after_compact}}",
    r"{\scriptsize", r"\setlength{\tabcolsep}{5pt}",
    r"\begin{tabular}{lcccc@{\hskip 12pt}cccc}",
    r"\toprule",
    rf"& \multicolumn{{4}}{{c}}{{\textbf{{Original sample}} (N = {n_orig_t+n_orig_c:,})}}".replace(",", "\\,"),
    rf"& \multicolumn{{4}}{{c}}{{\textbf{{Matched sample}}  (N = {len(matched):,})}} \\".replace(",", "\\,"),
    r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
    r"Variable & Mean(Control) & Mean(Treated) & t-statistic & p-value",
    r"         & Mean(Control) & Mean(Treated) & t-statistic & p-value \\",
    r"\midrule",
]
for v in MATCH_VARS:
    o = orig_d[v]; m = matc_d[v]
    lines.append(
        f"{PRETTY[v]:<40} & "
        f"{fmt_mean(o['mean_C'],v)} & {fmt_mean(o['mean_T'],v)} & "
        f"{fmt_t(o['t_stat'],o['p'])} & {o['p']:.3f} & "
        f"{fmt_mean(m['mean_C'],v)} & {fmt_mean(m['mean_T'],v)} & "
        f"{fmt_t(m['t_stat'],m['p'])} & {m['p']:.3f} \\\\[3pt]"
    )
lines += [
    rf"\textbf{{N (Control)}} & {n_orig_c:,} & -- & -- & -- & {n_matc_c:,} & -- & -- & -- \\".replace(",","\\,"),
    rf"\textbf{{N (Treated)}} & -- & {n_orig_t:,} & -- & -- & -- & {n_matc_t:,} & -- & -- \\".replace(",","\\,"),
    r"\bottomrule", r"\end{tabular}}",
    r"{\vspace{2pt}\scriptsize%",
    r"\(^{*}p<0.10\), \(^{**}p<0.05\), \(^{***}p<0.01\).}",
    r"\end{table}",
]
with open(TABLES_DIR/"table_h2_cem_balance.tex","w") as f: f.write("\n".join(lines))
print(f"[saved] {TABLES_DIR}/table_h2_cem_balance.tex")

# ============================================================
# TABLE 2: Treatment × Factual-density interaction, by status stratum
# ============================================================
# Scaled controls (same as run_table8.py)
# Baseline controls (position_z, reputation_10, word_count_100) dropped per user request.
matched[MODERATOR] = matched[MODERATOR].astype('category')

STATUS = [(1,'Low Status'), (2,'Mid Status'), (3,'High Status')]
fits = {}
for sg, lab in STATUS:
    sub = matched[matched['stratum_group']==sg].copy()
    fml = f"{OUTCOME} ~ C({TREAT}) * C({MODERATOR})"
    fit = smf.ols(fml, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub[CLUSTER]})
    fits[sg] = (fit, len(sub))
    print(f"[fit] {lab}: N={len(sub):,}, adj-R^2={fit.rsquared_adj:.3f}")

# Extract interaction coefs T × Q{2,3,4}
def coef_se(fit, term):
    if term in fit.params.index:
        return float(fit.params[term]), float(fit.bse[term]), float(fit.pvalues[term])
    return np.nan, np.nan, np.nan

rows=[]
for sg, lab in STATUS:
    fit, n = fits[sg]
    for q in [2,3,4]:
        term = f"C({TREAT})[T.1]:C({MODERATOR})[T.{q}]"
        c, se, p = coef_se(fit, term)
        rows.append(dict(status=lab, quartile=f"Q{q}", coef=c, se=se, p=p, sig=stars(p),
                         n=n, adj_r2=float(fit.rsquared_adj)))
inter = pd.DataFrame(rows)
inter.to_csv(RESULTS_DIR/"cem_factual_interaction.csv", index=False)
print("\n[interaction coefs T × I(Factual=Qq), CEM matched]")
print(inter.pivot(index='quartile', columns='status', values='coef').round(3).to_string())

def fmt_c(c,p): return f"{c:+.3f}".replace("+0.","\\phantom{-}0.") + ("$^{"+stars(p).replace('*','*')+"}$" if stars(p) else "")
def fmt_s(s): return f"({s:.3f})"

lines2 = [
    r"\begin{table}[H]", r"\TABLE",
    r"{Treatment (\(T_i\)) Effects by Factual\textendash Evidence Density (1-to-1 CEM)\label{tab:factual_cem}}",
    r"{\small", r"\begin{tabular}{lccc}", r"\toprule",
    r"& \multicolumn{3}{c}{\textbf{Dependent Variable: Persuasion Success} (\(Y_{ij}\))} \\",
    r"\cmidrule(lr){2-4}",
    r"& Low Status & Mid Status & High Status \\",
    r"\midrule",
]
for q in [2,3,4]:
    row_c = [inter[(inter.status==lab)&(inter.quartile==f"Q{q}")].iloc[0] for _,lab in STATUS]
    lines2.append(
        rf"\(T_i \times I(\text{{Factual}}_j=Q{q})\) & "
        + " & ".join(fmt_c(r['coef'], r['p']) for r in row_c) + r" \\"
    )
    lines2.append(r"                                   & "
        + " & ".join(fmt_s(r['se']) for r in row_c) + r" \\[2pt]"
    )
lines2 += [r"\midrule",
    r"\textit{Group FE}                  & No & No & No \\[2pt]",
    r"\textit{Baseline controls}         & No & No & No \\[2pt]",
    r"\midrule",
    rf"Observations                       & {fits[1][1]:,} & {fits[2][1]:,} & {fits[3][1]:,} \\".replace(",","{,}"),
    rf"Adjusted \(R^{{2}}\)                 & {fits[1][0].rsquared_adj:.3f} & {fits[2][0].rsquared_adj:.3f} & {fits[3][0].rsquared_adj:.3f} \\",
    r"\bottomrule", r"\end{tabular}}",
    r"{Baseline category is \(Q1\) (lowest factual density).",
    r"Coefficients are interaction effects \(T_i \times I(\text{Factual}=Qq)\) estimated separately within each status stratum;",
    r"robust standard errors clustered at the challenger level are in parentheses.",
    r"\(^{*}p<0.10\), \(^{**}p<0.05\), \(^{***}p<0.01\).}",
    r"\end{table}",
]
with open(TABLES_DIR/"table_h2_cem_factual_interaction.tex","w") as f: f.write("\n".join(lines2))
print(f"\n[saved] {TABLES_DIR}/table_h2_cem_factual_interaction.tex")
print(f"[saved] {RESULTS_DIR}/cem_factual_interaction.csv")
print(f"[saved] {RESULTS_DIR}/cem_balance.csv")
print(f"[saved] {RESULTS_DIR}/cem_matched_dyads.csv  ({len(matched):,} rows)")
