"""H2 appendix — Alternative measures of factuality (Word-Weighted, Log-Odds).

Same interaction spec as the main H2 moderation analysis, but with two alternative quartile
moderators replacing `pct_empirical_claims_Q4`:
  • pct_empirical_words_Q4  — word-weighted share of empirical content
  • emp_claim_logit_Q4      — log-odds of empirical/non-empirical claim counts (+0.5 continuity)

Per the user's instruction this runs WITHOUT baseline controls
(position_z / reputation_10 / word_count_100). Cluster-robust SE at author_comment.

For each moderator, fits 4 columns:
  (1) Low  status_group=1  — no group FE
  (2) Mid  status_group=2  — no group FE
  (3) High status_group=3  — no group FE
  (4) Full sample          — with C(stratum_group) group FE
Reports T_i baseline + T × I(Q=2,3,4).

Outputs:
  ../../appendix/table_h2_factual_words.tex
  ../../appendix/table_h2_factual_logit.tex
  ../results/alt_measures_factual_words.csv
  ../results/alt_measures_factual_logit.csv
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

# Moderator info for both tables (moderator column, table file name, label key in LaTeX, paper-style x-label, paper-target table label)
MODS = [
    dict(col="pct_empirical_words_Q4",
         tex="table_h2_factual_words.tex",
         label="tab:factual_words",
         x_label=r"Factual\_word\_weighted",
         caption="Treatment Effects by Factual Evidence Density (Word-Weighted)",
         note=r"Word-weighted measure accounts for the length of empirical versus opinion content. Patterns remain consistent with main results: low-status challengers benefit in high-evidence contexts while high-status challengers suffer in mixed-evidence contexts."),
    dict(col="emp_claim_logit_Q4",
         tex="table_h2_factual_logit.tex",
         label="tab:factual_logit",
         x_label=r"Factual\_log",
         caption="Treatment Effects by Factual Evidence Density (Log-Odds)",
         note=r"Log-odds transformation addresses compression at extremes of the proportion scale. Key patterns persist: Q4 benefit for low-status challengers and Q2 penalty for high-status challengers."),
]

def stars(p): return '***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''
def fmt_coef_phantom(c, p, paren_neg=False):
    """tab:factual_logit style: \\phantom{-} before non-negatives in interaction rows."""
    star = f"^{{{stars(p)}}}" if stars(p) else ""
    if c < 0:
        return f"$-{abs(c):.3f}{star}$"
    else:
        return r"$\phantom{-}" + f"{c:.3f}{star}$"
def fmt_coef_plain(c, p):
    """Simple format: leading minus sign / no \\phantom."""
    star = f"^{{{stars(p)}}}" if stars(p) else ""
    return f"${c:.3f}{star}$" if c >= 0 else f"$-{abs(c):.3f}{star}$"
def fmt_se(s): return f"$({s:.3f})$"

def coef_se(fit, term):
    if term in fit.params.index:
        return float(fit.params[term]), float(fit.bse[term]), float(fit.pvalues[term])
    return np.nan, np.nan, np.nan

STATUS = [(1,'Low'), (2,'Mid'), (3,'High')]

for mod in MODS:
    print(f"\n=== {mod['col']} ===")
    df[mod['col']] = df[mod['col']].astype('category')
    fits = {}
    # within-status fits (cols 1-3)
    for sg, lab in STATUS:
        sub = df[df['stratum_group']==sg].copy()
        fml = f"{OUTCOME} ~ C({TREAT}) * C({mod['col']})"
        fit = smf.ols(fml, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub[CLUSTER]})
        fits[lab] = (fit, len(sub))
        print(f"  [fit] {lab:<5} N={len(sub):,}  adj-R²={fit.rsquared_adj:.3f}")
    # full sample with group FE (col 4)
    fml_full = f"{OUTCOME} ~ C({TREAT}) * C({mod['col']}) + C(stratum_group)"
    fit_full = smf.ols(fml_full, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[CLUSTER]})
    fits['Full'] = (fit_full, len(df))
    print(f"  [fit] Full  N={len(df):,}  adj-R²={fit_full.rsquared_adj:.3f}")

    # ---- extract coefs ----
    rows = []
    for col_label in ['Low','Mid','High','Full']:
        fit, n = fits[col_label]
        # T_i baseline
        c, se, p = coef_se(fit, f"C({TREAT})[T.1]")
        rows.append(dict(col=col_label, term='T', q=None, n=n, coef=c, se=se, p=p, sig=stars(p)))
        # T × Q{2,3,4}
        for q in (2,3,4):
            term = f"C({TREAT})[T.1]:C({mod['col']})[T.{q}]"
            c, se, p = coef_se(fit, term)
            rows.append(dict(col=col_label, term=f'TxQ{q}', q=q, n=n, coef=c, se=se, p=p, sig=stars(p)))
    res = pd.DataFrame(rows)
    csv_name = f"alt_measures_{mod['col'].replace('_Q4','')}.csv"
    res.to_csv(RESULTS_DIR / csv_name, index=False)
    print(f"  [saved] {RESULTS_DIR}/{csv_name}")

    # ---- LaTeX ----
    use_phantom = (mod['label'] == 'tab:factual_logit')  # paper formats logit with \phantom{-}
    fmt_c = fmt_coef_phantom if use_phantom else fmt_coef_plain

    def row_coefs(term_name):
        return [res[(res.col==c)&(res.term==term_name)].iloc[0] for c in ('Low','Mid','High','Full')]

    lines = [
        r"\begin{table}[H]", r"\TABLE",
        rf"{{{mod['caption']}\label{{{mod['label']}}}}}",
        r"{\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{Dependent Variable: Persuasion Success ($Y_{ij}$)} \\",
        r"\cmidrule(lr){3-6}",
        r"& & (1) & (2) & (3) & (4) \\",
        r"& & Low & Mid & High & Full Sample \\",
        r"\midrule",
    ]
    # T_i row
    rs = row_coefs('T')
    lines.append(r"$T_i$ & & " + " & ".join(fmt_c(r['coef'], r['p']).replace(r"\phantom{-}","") for r in rs) + r" \\")
    lines.append(r"      & & " + " & ".join(fmt_se(r['se']) for r in rs) + r" \\[4pt]")
    # Interaction rows
    for q_i, q in enumerate([2,3,4]):
        rs = row_coefs(f'TxQ{q}')
        lines.append(
            rf"$\times\; I(\text{{{mod['x_label']}}}_j=Q{q})$ & & "
            + " & ".join(fmt_c(r['coef'], r['p']) for r in rs) + r" \\"
        )
        trailer = r" \\[6pt]" if q == 4 else r" \\[2pt]"
        lines.append(r"      & & " + " & ".join(fmt_se(r['se']) for r in rs) + trailer)
    # FE / controls / obs / R² rows
    lines += [
        r"\textit{Group FE} & & No & No & No & Yes \\[2pt]",
        r"\textit{Controls} & & No & No & No & No \\",
        r"\midrule",
        rf"Observations & & {fits['Low'][1]:,} & {fits['Mid'][1]:,} & {fits['High'][1]:,} & {fits['Full'][1]:,} \\",
        rf"Adjusted $R^2$ & & {fits['Low'][0].rsquared_adj:.3f} & {fits['Mid'][0].rsquared_adj:.3f} & {fits['High'][0].rsquared_adj:.3f} & {fits['Full'][0].rsquared_adj:.3f} \\",
        r"\bottomrule",
        r"\end{tabular}}",
        rf"{{{mod['note']} \textbf{{No baseline controls.}} Cluster-robust standard errors at the challenger level in parentheses. $^{{*}}p<0.10$, $^{{**}}p<0.05$, $^{{***}}p<0.01$.}}",
        r"\end{table}",
    ]
    out_tex = APPENDIX_DIR / mod['tex']
    with open(out_tex, "w") as f: f.write("\n".join(lines))
    print(f"  [saved] {out_tex}")

# ======================================================================
# Verification vs paper
# ======================================================================
print("\n" + "="*88)
print("VERIFICATION vs paper (T × I(Q=q) coefficients)")
print("="*88)
paper = {
    'pct_empirical_words_Q4': {
        ('T',  None,'Low'): (0.004,'',0.012), ('T',None,'Mid'): (0.017,'',0.022), ('T',None,'High'): (0.006,'',0.023), ('T',None,'Full'): (0.007,'',0.010),
        ('TxQ2',2,'Low'):  (-0.018,'',0.017), ('TxQ2',2,'Mid'): (-0.026,'',0.029),  ('TxQ2',2,'High'):(-0.066,'**',0.031),('TxQ2',2,'Full'): (-0.032,'**',0.013),
        ('TxQ3',3,'Low'):  (0.004,'',0.016),  ('TxQ3',3,'Mid'): (-0.009,'',0.030),  ('TxQ3',3,'High'):(-0.008,'',0.031),  ('TxQ3',3,'Full'): (-0.002,'',0.013),
        ('TxQ4',4,'Low'):  (0.029,'*',0.016), ('TxQ4',4,'Mid'): (-0.047,'',0.035),  ('TxQ4',4,'High'):(-0.032,'',0.027),  ('TxQ4',4,'Full'): (0.001,'',0.013),
    },
    'emp_claim_logit_Q4': {
        ('T',  None,'Low'):(-0.001,'',0.012), ('T',None,'Mid'): (0.014,'',0.022),  ('T',None,'High'): (0.002,'',0.021),   ('T',None,'Full'): (0.003,'',0.009),
        ('TxQ2',2,'Low'):  (0.000,'',0.016),  ('TxQ2',2,'Mid'): (-0.026,'',0.028), ('TxQ2',2,'High'):(-0.056,'**',0.028), ('TxQ2',2,'Full'): (-0.018,'',0.013),
        ('TxQ3',3,'Low'):  (0.006,'',0.017),  ('TxQ3',3,'Mid'): (-0.018,'',0.031), ('TxQ3',3,'High'):(0.002,'',0.031),    ('TxQ3',3,'Full'): (0.000,'',0.014),
        ('TxQ4',4,'Low'):  (0.031,'*',0.017), ('TxQ4',4,'Mid'): (-0.023,'',0.036), ('TxQ4',4,'High'):(-0.029,'',0.027),  ('TxQ4',4,'Full'): (0.007,'',0.013),
    },
}
for mod in MODS:
    print(f"\n--- {mod['col']} ---")
    res = pd.read_csv(RESULTS_DIR / f"alt_measures_{mod['col'].replace('_Q4','')}.csv",
                      keep_default_na=False)  # keep empty-string sig as ''
    res['sig'] = res['sig'].fillna('')
    match_sign = match_sig = total = 0
    print(f"{'term':<8}{'col':<7}{'paper':<24}{'mine':<24}match?")
    for (term,q,col),(p_c,p_s,p_se) in paper[mod['col']].items():
        r = res[(res.col==col)&(res.term==term)].iloc[0]
        same_sign = (np.sign(p_c)==np.sign(r['coef'])) or (abs(p_c)<0.005 and abs(r['coef'])<0.01)
        same_sig  = (p_s == r['sig'])
        match_sign += same_sign; match_sig += same_sig; total += 1
        flag = "✓" if (same_sign and same_sig) else ("≈sig" if same_sign else "DIFFER")
        print(f"{term:<8}{col:<7}{p_c:+.3f}{p_s:<3} ({p_se:.3f})    {r['coef']:+.3f}{r['sig']:<3} ({r['se']:.3f})    {flag}")
    print(f"  signs match: {match_sign}/{total}    significance match: {match_sig}/{total}")
