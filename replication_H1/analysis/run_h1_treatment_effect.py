"""H1 treatment-effect replication — user-level `success_rate` regressions.

Replicates FINAL.ipynb §2 (cells 44-46):
  M-main : success_rate ~ C(stratum) + C(group) - 1
  M1     : stratum_1..7 + subgroup_1..3 + group_int_subgroup1..3 - 1          (HC1)
  M3     : stratum_1..7 + group_int_stratum1..7 - 1                            (HC1)

Data: cmv_experiment_data_19965_regression.csv (one row per challenger).
  stratum : 7 levels (Manzoor et al. status bins)
  subgroup: collapses stratum into 3 groups
    subgroup_1 = {1}            Low   (18,892)
    subgroup_2 = {2,3,4}        Mid   (844)
    subgroup_3 = {5,6,7}        High  (229)

Outputs
  results/h1_treatment_effect_coefs.csv        — long tidy table
  tables/table_h1_treatment_effect.tex         — paper-ready (M-main / M1 / M3 side by side)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

from pathlib import Path
HERE = Path(__file__).parent
REPO = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = (HERE / ".." / "results").resolve()
TABLES_DIR  = (HERE / ".." / "tables").resolve()
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR,  exist_ok=True)

DATA = str(REPO / "cmv_experiment_data_19965_regression.csv")

# ---------------------------------------------------------------
# 1. Load and characterize
# ---------------------------------------------------------------
df = pd.read_csv(DATA, low_memory=False)
# Ensure booleans are 0/1 ints for regression
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

# Subgroup labels for reporting
SUBGROUP_LABEL = {1: "Low ({1})", 2: "Mid ({2,3,4})", 3: "High ({5,6,7})"}

# ---------------------------------------------------------------
# 2. Fit models
# ---------------------------------------------------------------
stratum_cols  = [f"stratum_{i}" for i in range(1, 8)]
gxs_cols      = [f"group_int_stratum{i}" for i in range(1, 8)]
subgroup_cols = ["subgroup_1", "subgroup_2", "subgroup_3"]
gxsg_cols     = ["group_int_subgroup1", "group_int_subgroup2", "group_int_subgroup3"]

# M-main: Main effect
M_main = smf.ols("success_rate ~ C(stratum) + C(group) - 1", data=df).fit(cov_type="HC1")

# M1: stratum FE + subgroup dummies + subgroup × group interactions
formula_m1 = "success_rate ~ " + " + ".join(stratum_cols + subgroup_cols + gxsg_cols) + " - 1"
M1 = smf.ols(formula=formula_m1, data=df).fit(cov_type="HC1")

# M3: stratum FE + stratum × group interactions (stratum-specific ATE)
formula_m3 = "success_rate ~ " + " + ".join(stratum_cols + gxs_cols) + " - 1"
M3 = smf.ols(formula=formula_m3, data=df).fit(cov_type="HC1")

# ---------------------------------------------------------------
# 3. Tidy per-coefficient table
# ---------------------------------------------------------------
def tidy(fit, model_tag):
    return pd.DataFrame({
        "model":  model_tag,
        "term":   fit.params.index,
        "coef":   fit.params.values,
        "se":     fit.bse.values,
        "t":      fit.tvalues.values,
        "p":      fit.pvalues.values,
    })

tidy_all = pd.concat([tidy(M_main, "M-main"), tidy(M1, "M1"), tidy(M3, "M3")],
                     ignore_index=True)
_p = RESULTS_DIR / "h1_treatment_effect_coefs.csv"
tidy_all.to_csv(_p, index=False)
print(f"[saved] {_p}  ({len(tidy_all)} rows)")

# ---------------------------------------------------------------
# 4. LaTeX table builder
# ---------------------------------------------------------------
def stars(p):
    if not np.isfinite(p): return ""
    if p < 0.01: return r"^{***}"
    if p < 0.05: return r"^{**}"
    if p < 0.10: return r"^{*}"
    return ""

def fmt(b, se, p):
    """Return two-line LaTeX cell content: coef row + SE row (caller handles layout)."""
    if not np.isfinite(b): return ("---", "")
    return (f"${b:+.4f}{stars(p)}$", f"({se:.4f})")

def coef(fit, term):
    if term in fit.params.index:
        return fit.params[term], fit.bse[term], fit.pvalues[term]
    return (np.nan, np.nan, np.nan)

# Rows shown in the paper-ready table — one for each interpretive coefficient
ROWS = [
    # (pretty_label, (M-main term or None), (M1 term or None), (M3 term or None))
    ("Treatment (pooled)",        "C(group)[T.1]",              None,                    None),
    ("Tx $\\times$ Low subgroup", None,                    "group_int_subgroup1",   None),
    ("Tx $\\times$ Mid subgroup", None,                    "group_int_subgroup2",   None),
    ("Tx $\\times$ High subgroup",None,                    "group_int_subgroup3",   None),
    ("Tx $\\times$ Stratum 1",    None,                    None,                    "group_int_stratum1"),
    ("Tx $\\times$ Stratum 2",    None,                    None,                    "group_int_stratum2"),
    ("Tx $\\times$ Stratum 3",    None,                    None,                    "group_int_stratum3"),
    ("Tx $\\times$ Stratum 4",    None,                    None,                    "group_int_stratum4"),
    ("Tx $\\times$ Stratum 5",    None,                    None,                    "group_int_stratum5"),
    ("Tx $\\times$ Stratum 6",    None,                    None,                    "group_int_stratum6"),
    ("Tx $\\times$ Stratum 7",    None,                    None,                    "group_int_stratum7"),
]

def cell(fit, term):
    if term is None: return ("---", "")
    b, se, p = coef(fit, term)
    return fmt(b, se, p)

def build_latex():
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Treatment Effect of Hiding Reputation Badges on Persuasion (User-level Replication of Manzoor et al.\ 2020)}")
    lines.append(r"\label{tab:h1_treatment_effect}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r" & (1) & (2) & (3) \\")
    lines.append(r"\textit{Dependent Variable: User-level Persuasion Rate} & Main Effect & By Subgroup & By Stratum \\")
    lines.append(r"\midrule")
    for lab, t_main, t1, t3 in ROWS:
        c_main = cell(M_main, t_main); c1 = cell(M1, t1); c3 = cell(M3, t3)
        # Coefficient row
        lines.append(f"{lab} & {c_main[0]} & {c1[0]} & {c3[0]} \\\\")
        # SE row (indented, muted)
        lines.append(f" & {c_main[1]} & {c1[1]} & {c3[1]} \\\\")
    lines.append(r"\midrule")
    lines.append(r"Stratum FE (1--7)   & Yes         & Yes & Yes \\")
    lines.append(r"Subgroup dummies    & ---         & Yes & --- \\")
    lines.append(r"Interaction terms   & ---         & subgroup $\times$ Tx & stratum $\times$ Tx \\")
    # N
    lines.append(f"Observations         & {int(M_main.nobs):,} & {int(M1.nobs):,} & {int(M3.nobs):,} \\\\")
    lines.append(f"Adjusted $R^2$       & {M_main.rsquared_adj:.4f} & {M1.rsquared_adj:.4f} & {M3.rsquared_adj:.4f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize")
    lines.append(r"\item[] OLS with HC1 heteroskedasticity-robust standard errors in "
                 r"parentheses. Unit of analysis is the challenger (one row per "
                 r"user). Dependent variable is the user-level persuasion rate "
                 r"(deltas awarded across all debates the user participated in "
                 r"during the experiment). All three specifications include a "
                 r"full set of stratum fixed effects and are estimated without "
                 r"an intercept. Column (1) reports the pooled average treatment "
                 r"effect. Column (2) allows the treatment effect to vary by "
                 r"status subgroup (Low = stratum 1, Mid = strata 2--4, "
                 r"High = strata 5--7). Column (3) allows the treatment effect "
                 r"to vary across all seven randomization strata.")
    lines.append(r"\item[] $^{*}p < 0.10$, $^{**}p < 0.05$, $^{***}p < 0.01$")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

out = build_latex()
with open(TABLES_DIR / "table_h1_treatment_effect.tex","w") as f:
    f.write(out)
print(f"[saved] {TABLES_DIR}/table_h1_treatment_effect.tex")

# ---------------------------------------------------------------
# 5. Console summary
# ---------------------------------------------------------------
print("\n" + "="*80)
print("M-main  success_rate ~ C(stratum) + C(group) - 1")
print("="*80)
for term in M_main.params.index:
    b, se, p = M_main.params[term], M_main.bse[term], M_main.pvalues[term]
    s = stars(p).replace('^{','').replace('}','')
    print(f"  {term:<30} {b:+.4f}{s:<4}  (se={se:.4f}, p={p:.3f})")
print(f"N = {int(M_main.nobs):,},  Adj-R^2 = {M_main.rsquared_adj:.4f}")

print("\n" + "="*80)
print("M1      stratum_1..7 + subgroup_1..3 + group_int_subgroup1..3 - 1 (HC1)")
print("="*80)
for term in M1.params.index:
    b, se, p = M1.params[term], M1.bse[term], M1.pvalues[term]
    s = stars(p).replace('^{','').replace('}','')
    print(f"  {term:<30} {b:+.4f}{s:<4}  (se={se:.4f}, p={p:.3f})")
print(f"N = {int(M1.nobs):,},  Adj-R^2 = {M1.rsquared_adj:.4f}")

print("\n" + "="*80)
print("M3      stratum_1..7 + group_int_stratum1..7 - 1  (HC1)")
print("="*80)
for term in M3.params.index:
    b, se, p = M3.params[term], M3.bse[term], M3.pvalues[term]
    s = stars(p).replace('^{','').replace('}','')
    print(f"  {term:<30} {b:+.4f}{s:<4}  (se={se:.4f}, p={p:.3f})")
print(f"N = {int(M3.nobs):,},  Adj-R^2 = {M3.rsquared_adj:.4f}")
