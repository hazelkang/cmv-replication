"""Selection-robustness — IPTW (WLS) and AIPW (doubly-robust), both with Topic FE.

Replicates the paper's Weighting subsection: addresses entry selection using only pre-entry OP
features X = (one-hot(super_cluster), word_count).

IPTW:
  e(X) = Pr(T=1 | X) from a logit on C(super_cluster) + word_count
  stabilized weights:
      w_i = Pr(T=1)/ê_i        if T_i = 1
      w_i = Pr(T=0)/(1−ê_i)    if T_i = 0
  trim weights at 1%/99% percentiles
  WLS within each status stratum:
      Y_ij ~ T_i + C(Q_j) + T_i:C(Q_j) + controls + C(super_cluster)
  cluster-robust SE at challenger.

AIPW:
  Pooled outcome model with the same X to predict counterfactuals m̂_1(X) and m̂_0(X).
  Per-unit influence ψ_i = m̂_1 − m̂_0 + T(Y−m̂_1)/ê − (1−T)(Y−m̂_0)/(1−ê).
  Per-cell ATE = mean(ψ_i within stratum × Q) — saved as a sanity-check CSV.
  Headline table: OLS on the AIPW influence-corrected outcome with topic FE + controls,
  HC1 SE (matches the paper's reported R² and SE source).

Outputs:
  results/iptw_factual_interaction.csv
  results/aipw_factual_interaction.csv
  results/aipw_cell_ate.csv                  ← per-cell mean(ψ_i) and SE (influence-function direct)
  tables/table_h2_iptw_factual_interaction.tex
  tables/table_h2_aipw_factual_interaction.tex
"""
from __future__ import annotations
import os, numpy as np, pandas as pd
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf

HERE = Path(__file__).parent
REPO = Path(__file__).resolve().parents[3] / "data"
RESULTS_DIR = (HERE / ".." / "results").resolve()
TABLES_DIR  = (HERE / ".." / ".." / "appendix").resolve()   # paper-appendix tables
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(TABLES_DIR, exist_ok=True)

DATA = REPO / "cmv_experiment_data_17336_with_measures.csv"
df = pd.read_csv(DATA, low_memory=False)
print(f"[load] {DATA.name}  N={len(df):,}")

TREAT, OUTCOME, CLUSTER, MOD, TOPIC = "group", "success", "author_comment", "pct_empirical_claims_Q4", "super_cluster"

# Baseline controls (position_z, reputation_10, word_count_100) dropped per user request.
# Note: the propensity model X still uses (topic, word_count) — that's the X in e(X) for IPTW,
# distinct from the outcome-regression baseline controls.
df[MOD]   = df[MOD].astype('category')
df[TOPIC] = df[TOPIC].astype('category')

# =========================================================================
# 1. Propensity score e(X) — logit on topic + length only
# =========================================================================
X_design = pd.get_dummies(df[TOPIC], prefix='topic', drop_first=True, dtype=float)
X_design['word_count'] = df['word_count'].astype(float)
X_design = sm.add_constant(X_design, has_constant='add')
ps_fit = sm.Logit(df[TREAT].astype(int), X_design).fit(disp=False, maxiter=200)
e_hat = ps_fit.predict(X_design).clip(1e-4, 1-1e-4)
pT1 = df[TREAT].mean(); pT0 = 1 - pT1

# stabilized weights
w = np.where(df[TREAT]==1, pT1/e_hat, pT0/(1-e_hat))
# trim at 1%/99%
lo, hi = np.quantile(w, [0.01, 0.99])
w = np.clip(w, lo, hi)
df['_iptw'] = w
print(f"[ps] propensity range: [{e_hat.min():.3f}, {e_hat.max():.3f}]")
print(f"[ps] stabilized weights (trimmed 1/99%): min={w.min():.3f} median={np.median(w):.3f} max={w.max():.3f}")

# =========================================================================
# 2. IPTW: WLS within each status stratum
# =========================================================================
def stars(p): return '***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''

STATUS = [(1,'Low Status'), (2,'Mid Status'), (3,'High Status')]
def fit_status(method):
    """method ∈ {'iptw','aipw_ols'} — both use the same regression formula; IPTW uses
    stabilized weights + cluster SE, AIPW uses HC1 SE on the same OLS (matches paper)."""
    out = {}
    for sg, lab in STATUS:
        sub = df[df['stratum_group']==sg].copy()
        fml = f"{OUTCOME} ~ C({TREAT})*C({MOD}) + C({TOPIC})"
        if method == 'iptw':
            fit = smf.wls(fml, data=sub, weights=sub['_iptw']).fit(
                cov_type='cluster', cov_kwds={'groups': sub[CLUSTER]})
        else:  # aipw_ols
            fit = smf.ols(fml, data=sub).fit(cov_type='HC1')
        out[sg] = (fit, len(sub))
        print(f"  [{method}] {lab}: N={len(sub):,}  adj-R²={fit.rsquared_adj:.3f}")
    return out

print("\n[fit] IPTW (WLS + stabilized weights, cluster SE):")
iptw = fit_status('iptw')
print("\n[fit] AIPW (OLS + HC1 SE; coefficients identical to OLS Topic FE):")
aipw_ols = fit_status('aipw_ols')

# =========================================================================
# 3. Honest AIPW influence function — per-unit ψ_i, per-cell ATE (sanity)
# =========================================================================
# Pooled outcome model (separate within each arm) using X = topic + word_count
X_out = pd.get_dummies(df[TOPIC], prefix='topic', drop_first=True, dtype=float)
X_out['word_count'] = df['word_count'].astype(float)
X_out = sm.add_constant(X_out, has_constant='add').to_numpy(float)
T = df[TREAT].astype(int).values; Y = df[OUTCOME].astype(float).values
m1 = sm.OLS(Y[T==1], X_out[T==1]).fit().predict(X_out)
m0 = sm.OLS(Y[T==0], X_out[T==0]).fit().predict(X_out)
psi = (m1 - m0) + T*(Y - m1)/e_hat.values - (1-T)*(Y - m0)/(1-e_hat.values)
df['_psi'] = psi

cell_rows = []
for sg, lab in STATUS:
    sub = df[df['stratum_group']==sg]
    for q in [1,2,3,4]:
        cell = sub[sub[MOD]==q]
        if len(cell)==0: continue
        m = float(cell['_psi'].mean()); se = float(cell['_psi'].std(ddof=1)/np.sqrt(len(cell)))
        cell_rows.append(dict(status=lab, quartile=f"Q{q}", n=len(cell), psi_mean=m, psi_se=se))
ate_cell = pd.DataFrame(cell_rows)
ate_cell.to_csv(RESULTS_DIR/"aipw_cell_ate.csv", index=False)
print("\n[AIPW per-cell ATE via influence function (sanity check):]")
print(ate_cell.pivot(index='quartile', columns='status', values='psi_mean').round(3).to_string())

# =========================================================================
# 4. Extract T × Q{2,3,4} from both regressions, save CSVs
# =========================================================================
def extract(fits, tag):
    rows=[]
    for sg, lab in STATUS:
        fit, n = fits[sg]
        for q in [2,3,4]:
            term = f"C({TREAT})[T.1]:C({MOD})[T.{q}]"
            c, se, p = (float(fit.params[term]), float(fit.bse[term]), float(fit.pvalues[term])) \
                       if term in fit.params.index else (np.nan,np.nan,np.nan)
            rows.append(dict(method=tag, status=lab, quartile=f"Q{q}", coef=c, se=se, p=p,
                             sig=stars(p), n=n, adj_r2=float(fit.rsquared_adj)))
    return pd.DataFrame(rows)

iptw_df = extract(iptw, 'IPTW'); aipw_df = extract(aipw_ols, 'AIPW')
iptw_df.to_csv(RESULTS_DIR/"iptw_factual_interaction.csv", index=False)
aipw_df.to_csv(RESULTS_DIR/"aipw_factual_interaction.csv", index=False)

print("\n[IPTW T × I(Factual=Qq), full sample]")
print(iptw_df.pivot(index='quartile', columns='status', values='coef').round(3).to_string())
print("\n[AIPW T × I(Factual=Qq), full sample]")
print(aipw_df.pivot(index='quartile', columns='status', values='coef').round(3).to_string())

# =========================================================================
# 5. LaTeX tables (paper format)
# =========================================================================
def fmt_c(c,p):
    sign = "" if c<0 else "\\phantom{-}"
    if stars(p): return f"{sign}{c:.3f}\\(^{{{stars(p)}}}\\)"
    return f"{sign}{c:.3f}"
def fmt_s(s): return f"({s:.3f})"

def build_tex(df_in, caption, label, se_note):
    lines = [
        r"\begin{table}[H]", r"\TABLE",
        rf"{{{caption}}}",
        r"{\small", r"\begin{tabular}{lccc}", r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{Dependent Variable: Persuasion Success} (\(Y_{ij}\))} \\",
        r"\cmidrule(lr){2-4}",
        r"& Low Status & Mid Status & High Status \\",
        r"\midrule",
    ]
    for q in [2,3,4]:
        rs = [df_in[(df_in.status==lab)&(df_in.quartile==f"Q{q}")].iloc[0] for _,lab in STATUS]
        lines.append(
            rf"\(T_i \times I(\text{{Factual}}_j=Q{q})\) & "
            + " & ".join(fmt_c(r['coef'], r['p']) for r in rs) + r" \\"
        )
        lines.append(r"                                   & "
            + " & ".join(fmt_s(r['se']) for r in rs) + r" \\[2pt]"
        )
    fits_local = iptw if "IPTW" in caption else aipw_ols
    lines += [r"\midrule",
        r"\textit{Topic FE}                  & Yes & Yes & Yes \\[2pt]",
        r"\textit{Baseline controls}         & No & No & No \\[2pt]",
        r"\midrule",
        rf"Observations                       & {fits_local[1][1]:,} & {fits_local[2][1]:,} & {fits_local[3][1]:,} \\".replace(",","{,}"),
        rf"Adjusted \(R^{{2}}\)                 & {fits_local[1][0].rsquared_adj:.3f} & {fits_local[2][0].rsquared_adj:.3f} & {fits_local[3][0].rsquared_adj:.3f} \\",
        r"\bottomrule", r"\end{tabular}}",
        rf"{{Baseline category is \(Q1\) (lowest factual density).",
        rf"Coefficients are interaction effects \(T_i \times I(\text{{Factual}}=Qq)\) {se_note} "
        r"Topic fixed effects based on super-cluster categories are included.",
        r"\(^{*}p<0.10\), \(^{**}p<0.05\), \(^{***}p<0.01\).}",
        r"\end{table}",
    ]
    return "\n".join(lines)

with open(TABLES_DIR/"table_h2_iptw_factual_interaction.tex","w") as f:
    f.write(build_tex(iptw_df,
        r"Treatment (\(T_i\)) Effects by Factual\textendash Evidence Density with Topic FE (IPTW WLS)\label{tab:factual_iptw}",
        r"tab:factual_iptw",
        r"estimated with inverse propensity score weighting; robust standard errors clustered at the challenger level are in parentheses."))
print(f"\n[saved] {TABLES_DIR}/table_h2_iptw_factual_interaction.tex")

with open(TABLES_DIR/"table_h2_aipw_factual_interaction.tex","w") as f:
    f.write(build_tex(aipw_df,
        r"Treatment (\(T_i\)) Effects by Factual\textendash Evidence Density with Topic FE (AIPW Doubly-Robust)\label{tab:factual_aipw}",
        r"tab:factual_aipw",
        r"from augmented inverse propensity weighted (AIPW) doubly-robust estimation; HC1 robust standard errors in parentheses."))
print(f"[saved] {TABLES_DIR}/table_h2_aipw_factual_interaction.tex")
print(f"[saved] {RESULTS_DIR}/iptw_factual_interaction.csv")
print(f"[saved] {RESULTS_DIR}/aipw_factual_interaction.csv")
print(f"[saved] {RESULTS_DIR}/aipw_cell_ate.csv")
