"""RCT balance-check replication for CMV reputation experiment.

Replicates three tables from FINAL.ipynb:
  §1-1  pre-experiment cohort, N = 18,590   (cmv_experiment_data_18590_balance_check.csv)
  §1-2  subsampled debates excluding new users, N = 12,983 (filter on 17336 CSV)
  §1-3  all users, N = 17,336               (cmv_experiment_data_17336.csv)

For each, compares treated vs control across 9 pre-experiment covariates:
  overall, by stratum_group (3 levels), and by stratum (7 levels).

Outputs:
  results/balance_check_{version}.csv        — machine-readable
  tables/table_balance_{version}.tex         — paper-ready LaTeX (overall + stratum_group panel)
  tables/table_balance_{version}_bystratum.tex — appendix-style LaTeX (7 strata)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path
HERE = Path(__file__).parent
REPO = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = (HERE / ".." / "results").resolve()
TABLES_DIR  = (HERE / ".." / "tables").resolve()
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR,  exist_ok=True)
def _R(p): return str(REPO / p)
def _RES(p): return str(RESULTS_DIR / p)
def _TAB(p): return str(TABLES_DIR / p)

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
COVARIATES = [
    ("comment_karma",          "Comment karma"),
    ("link_karma",             "Link karma"),
    ("tenure_i",               "Account tenure (days)"),
    ("gold",                   "Reddit gold"),
    ("deltas_received",        "Deltas received (pre-exp)"),
    ("num_cmv_post",           "# CMV posts"),
    ("num_cmv_comments",       "# CMV comments"),
    ("cmv_tenure",             "CMV tenure (days)"),
    ("average_comment_length", "Avg comment length"),
]
STRAT_ORDER = ["Overall", "stratum_group", "stratum"]

# ---------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------
def balance_row(data: pd.DataFrame, treatment_col: str, covar: str,
                level_name: str, level_value):
    valid = data.dropna(subset=[covar, treatment_col])
    t = valid.loc[valid[treatment_col] == 1, covar]
    c = valid.loc[valid[treatment_col] == 0, covar]
    n_t, n_c = len(t), len(c)
    n_total = n_t + n_c
    mean_t = t.mean() if n_t else np.nan
    mean_c = c.mean() if n_c else np.nan
    se_t = t.std() / np.sqrt(n_t) if n_t else np.nan
    se_c = c.std() / np.sqrt(n_c) if n_c else np.nan
    if n_t > 1 and n_c > 1:
        # Welch's t-test to match the notebook (equal_var=False)
        t_stat, p_val = stats.ttest_ind(t, c, equal_var=False)
    else:
        t_stat, p_val = np.nan, np.nan
    return {
        "Level": level_name, "Subgroup": str(level_value), "Variable": covar,
        "N": n_total, "N_Treated": n_t, "N_Control": n_c,
        "Mean_Treated": mean_t, "SE_Treated": se_t,
        "Mean_Control": mean_c, "SE_Control": se_c,
        "Difference": (mean_t - mean_c) if np.isfinite(mean_t) and np.isfinite(mean_c) else np.nan,
        "t_statistic": t_stat, "p_value": p_val,
    }


def balance_table(df: pd.DataFrame, treatment_col: str,
                  covariates=None, strat_vars=None):
    covariates = [c for c, _ in COVARIATES] if covariates is None else covariates
    strat_vars = ["stratum_group", "stratum"] if strat_vars is None else strat_vars
    rows = []
    for covar in covariates:
        rows.append(balance_row(df, treatment_col, covar, "Overall", "All"))
    for strat in strat_vars:
        for val in sorted(df[strat].dropna().unique()):
            sub = df[df[strat] == val]
            for covar in covariates:
                rows.append(balance_row(sub, treatment_col, covar, strat, val))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# LaTeX formatting
# ---------------------------------------------------------------
def star(p):
    if not np.isfinite(p): return ""
    if p < 0.01: return r"^{***}"
    if p < 0.05: return r"^{**}"
    if p < 0.10: return r"^{*}"
    return ""


def fmt_mean_se(m, se):
    if not np.isfinite(m): return "---"
    se_txt = f"{se:.3f}" if np.isfinite(se) else ""
    return f"{m:.3f} ({se_txt})"


def fmt_p(p):
    if not np.isfinite(p): return "---"
    txt = f"{p:.3f}" if p >= 0.001 else "<0.001"
    return f"${txt}{star(p)}$" if star(p) else txt


def label_for(covar):
    for c, lab in COVARIATES:
        if c == covar: return lab
    return covar


def panel_block(sub: pd.DataFrame, header: str, cols_are_covs: list):
    """Assemble LaTeX rows for a panel (Level+Subgroup fixed, one row per covariate)."""
    lines = []
    lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{header}}}}} \\\\")
    for covar in cols_are_covs:
        row = sub[sub["Variable"] == covar].iloc[0]
        lines.append(
            f"\\quad {label_for(covar)} & "
            f"{int(row.N):,} & "
            f"{fmt_mean_se(row.Mean_Treated, row.SE_Treated)} & "
            f"{fmt_mean_se(row.Mean_Control, row.SE_Control)} & "
            f"{row.t_statistic:+.3f} & "
            f"{fmt_p(row.p_value)} \\\\"
        )
    return "\n".join(lines)


def build_main_latex(tbl: pd.DataFrame, n_total: int, tag: str,
                     caption: str, label: str) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r" & & Treated & Control & & \\")
    lines.append(r"Variable & $N$ & mean (SE) & mean (SE) & $t$ & $p$ \\")
    lines.append(r"\midrule")
    covs = [c for c, _ in COVARIATES]
    # Overall panel
    lines.append(panel_block(tbl[tbl["Level"] == "Overall"],
                             f"Overall ($N={n_total:,}$)", covs))
    lines.append(r"\midrule")
    # Stratum-group panels (1=Low, 2=Mid, 3=High)
    strat_name = {1:"Low", 2:"Mid", 3:"High"}
    for v in [1, 2, 3]:
        sub = tbl[(tbl["Level"] == "stratum_group") & (tbl["Subgroup"] == str(v))]
        if sub.empty: continue
        n_sub = int(sub["N"].iloc[0])
        lines.append(panel_block(sub,
            f"Status stratum = {strat_name[v]} ($N={n_sub:,}$)", covs))
        if v != 3: lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize")
    lines.append(r"\item[] Welch's two-sample $t$-test comparing pre-experiment "
                 r"characteristics of challengers assigned to the badge-hidden "
                 r"(treated) and badge-visible (control) conditions. Treatment "
                 r"assignment is stratified by challenger status (Low, Mid, "
                 r"High). Standard errors in parentheses.")
    lines.append(r"\item[] $^{*}p < 0.10$, $^{**}p < 0.05$, $^{***}p < 0.01$")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_bystratum_latex(tbl: pd.DataFrame, n_total: int, tag: str,
                          caption: str, label: str) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r" & & Treated & Control & & \\")
    lines.append(r"Variable & $N$ & mean (SE) & mean (SE) & $t$ & $p$ \\")
    lines.append(r"\midrule")
    covs = [c for c, _ in COVARIATES]
    for s in range(1, 8):
        sub = tbl[(tbl["Level"] == "stratum") & (tbl["Subgroup"] == str(s))]
        if sub.empty: continue
        n_sub = int(sub["N"].iloc[0])
        lines.append(panel_block(sub, f"Stratum {s} ($N={n_sub:,}$)", covs))
        if s != 7: lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize")
    lines.append(r"\item[] Welch's two-sample $t$-test on pre-experiment "
                 r"covariates by randomization stratum (Manzoor et~al.\ 2020, "
                 r"7 levels). Standard errors in parentheses.")
    lines.append(r"\item[] $^{*}p < 0.10$, $^{**}p < 0.05$, $^{***}p < 0.01$")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------
# Driver
# ---------------------------------------------------------------
def summarize(tbl: pd.DataFrame, tag: str):
    overall = tbl[tbl["Level"] == "Overall"]
    n_sig_10 = int(((overall["p_value"] < 0.10)).sum())
    n_sig_05 = int(((overall["p_value"] < 0.05)).sum())
    print(f"[{tag}] Overall — {n_sig_05}/{len(overall)} covariates sig at 5%, {n_sig_10}/{len(overall)} at 10%")
    return {"n_sig_10": n_sig_10, "n_sig_05": n_sig_05, "n_covs": len(overall)}


def run():
    summaries = {}
    # ========= §1-1: pre-experiment 18,590 =========
    pe = pd.read_csv(_R("cmv_experiment_data_18590_balance_check.csv"), low_memory=False)
    tbl_pre = balance_table(pe, treatment_col="treatment")
    tbl_pre.to_csv(_RES("balance_check_1_1_preexp18590.csv"), index=False)
    with open(_TAB("table_balance_1_1_preexp18590.tex"), "w") as f:
        f.write(build_main_latex(tbl_pre, n_total=len(pe), tag="1_1",
            caption="RCT Balance Check: Pre-Experiment Cohort ($N=18{,}590$)",
            label="tab:balance_pre18590"))
    with open(_TAB("table_balance_1_1_preexp18590_bystratum.tex"), "w") as f:
        f.write(build_bystratum_latex(tbl_pre, n_total=len(pe), tag="1_1",
            caption="Balance Check by Randomization Stratum --- Pre-Experiment Cohort ($N=18{,}590$)",
            label="tab:balance_pre18590_bystratum"))
    summaries["1_1"] = summarize(tbl_pre, "1-1 preexp (N=18,590)")

    # ========= §1-2: subsampled debates, N = 12,983 =========
    full = pd.read_csv(_R("cmv_experiment_data_17336.csv"), low_memory=False)
    print(f"[1-2] full N = {len(full):,}")
    exp_filter = full["comment_karma"].notna()
    sub12983 = full.loc[exp_filter].copy()
    print(f"[1-2] after excluding new users (comment_karma NaN): N = {len(sub12983):,}")
    assert len(sub12983) == 12983, f"expected 12,983; got {len(sub12983):,}"
    tbl_sub = balance_table(sub12983, treatment_col="group")
    tbl_sub.to_csv(_RES("balance_check_1_2_subsampled12983.csv"), index=False)
    with open(_TAB("table_balance_1_2_subsampled12983.tex"), "w") as f:
        f.write(build_main_latex(tbl_sub, n_total=len(sub12983), tag="1_2",
            caption="RCT Balance Check: Subsampled Debates, Excluding New Users ($N=12{,}983$)",
            label="tab:balance_sub12983"))
    with open(_TAB("table_balance_1_2_subsampled12983_bystratum.tex"), "w") as f:
        f.write(build_bystratum_latex(tbl_sub, n_total=len(sub12983), tag="1_2",
            caption="Balance Check by Randomization Stratum --- Subsampled Debates ($N=12{,}983$)",
            label="tab:balance_sub12983_bystratum"))
    summaries["1_2"] = summarize(tbl_sub, "1-2 subsampled (N=12,983)")

    # ========= §1-3: all dyads, N = 17,336 =========
    tbl_all = balance_table(full, treatment_col="group")
    tbl_all.to_csv(_RES("balance_check_1_3_all17336.csv"), index=False)
    with open(_TAB("table_balance_1_3_all17336.tex"), "w") as f:
        f.write(build_main_latex(tbl_all, n_total=len(full), tag="1_3",
            caption="RCT Balance Check: All Experimental Dyads ($N=17{,}336$)",
            label="tab:balance_all17336"))
    with open(_TAB("table_balance_1_3_all17336_bystratum.tex"), "w") as f:
        f.write(build_bystratum_latex(tbl_all, n_total=len(full), tag="1_3",
            caption="Balance Check by Randomization Stratum --- All Dyads ($N=17{,}336$)",
            label="tab:balance_all17336_bystratum"))
    summaries["1_3"] = summarize(tbl_all, "1-3 all (N=17,336)")

    return summaries, tbl_pre, tbl_sub, tbl_all


if __name__ == "__main__":
    run()
