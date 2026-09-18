import pandas as pd
import statsmodels.formula.api as smf
import os
from pathlib import Path

HERE = Path(__file__).parent
REPO = Path(__file__).resolve().parents[2] / "data"
TABLES_DIR = (HERE / ".." / "tables").resolve()
os.makedirs(TABLES_DIR, exist_ok=True)

# Prefer the recomputed-from-JSON measures file so the pipeline is fully
# reproducible (run `build_quantiles.py` to regenerate). Falls back to the
# original 17336 CSV if the recomputed file isn't present.
_with = REPO / "cmv_experiment_data_17336_with_measures.csv"
_DATA_PATH = str(_with if _with.exists() else REPO / "cmv_experiment_data_17336.csv")
print(f"[data] loading {_DATA_PATH}")
df = pd.read_csv(_DATA_PATH, low_memory=False)

# Scaled controls (match Manzoor et al. Table 8 format)
df['position_z']     = (df['position'] - df['position'].mean()) / df['position'].std()
df['reputation_10']  = df['reputation'] / 10
df['word_count_100'] = df['word_count'] / 100

# Three content-type moderators, each a 4-level quartile. We treat each as
# categorical (Q1 reference) when building formulae.
MODERATORS = {
    'pct_empirical_claims_Q4':
        dict(pretty="% empirical claims (OP)",   short="PctClaims"),
    'pct_empirical_words_Q4':
        dict(pretty="% empirical words (OP)",    short="PctWords"),
    'emp_claim_logit_Q4':
        dict(pretty="LLM empirical-claim logit", short="EmpLogit"),
}
for m in MODERATORS:
    df[m] = df[m].astype('category')

# Challenger-level cluster: author_comment (2,784 unique challengers).
# responder_id has 17,336 unique values → one per dyad, NOT a challenger cluster.
CLUSTER_VAR = 'author_comment'

def stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''

def fmt_coef(c, p): return f"{c:.3f}{stars(p)}"
def fmt_se(se):     return f"({se:.3f})"

def terms_for(mod):
    """Return (term_order, term_labels) for a given moderator column name."""
    base = {
        'C(group)[T.1]': r'$T_i$',
        f'C(group)[T.1]:C({mod})[T.2]': r'$\times \mathbb{I}(\text{Factual}_j = Q2)$',
        f'C(group)[T.1]:C({mod})[T.3]': r'$\times \mathbb{I}(\text{Factual}_j = Q3)$',
        f'C(group)[T.1]:C({mod})[T.4]': r'$\times \mathbb{I}(\text{Factual}_j = Q4)$',
        'position_z':     r'Normalized Position ($p_{ij}$)',
        'reputation_10':  r'Reputation ($r_i/10$)',
        'word_count_100': r'Response Length ($\ell_{ij}/100$)',
    }
    return list(base.keys()), base


# ==============================================================
# Helper: build the 4-column LaTeX table from a dict of fitted models
# ==============================================================
def build_latex(models, cols, term_order, term_labels,
                caption, label, note_extra):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r" & (1) & (2) & (3) & (4) \\")
    lines.append(r"\textit{Dependent Variable: Persuasion Success ($Y_{ij}$)} & "
                 r"Low Status & Mid Status & High Status & All \\")
    lines.append(r"\midrule")

    # Treatment + interactions (first 4 terms)
    for term in term_order[:4]:
        label_txt = term_labels[term]
        coefs, ses = [], []
        for col in cols:
            fit = models[col]
            if term in fit.params.index:
                c  = fit.params[term]; se = fit.bse[term]; p = fit.pvalues[term]
                coefs.append(fmt_coef(c, p)); ses.append(fmt_se(se))
            else:
                coefs.append(''); ses.append('')
        lines.append(f"{label_txt} & " + " & ".join(coefs) + r" \\")
        lines.append(" & " + " & ".join(ses) + r" \\")

    # Optional controls block
    control_terms = term_order[4:]
    if control_terms:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{5}{l}{\textit{Baseline Controls}} \\")
        for term in control_terms:
            label_txt = term_labels[term]
            coefs, ses = [], []
            for col in cols:
                fit = models[col]
                if term in fit.params.index:
                    c  = fit.params[term]; se = fit.bse[term]; p = fit.pvalues[term]
                    coefs.append(fmt_coef(c, p)); ses.append(fmt_se(se))
                else:
                    coefs.append(''); ses.append('')
            lines.append(f"{label_txt} & " + " & ".join(coefs) + r" \\")
            lines.append(" & " + " & ".join(ses) + r" \\")

    lines.append(r"\midrule")
    lines.append(r"Group FEs & No & No & No & Yes \\")

    n_row = ["Observations"]
    for col in cols:
        n_row.append(f"{int(models[col].nobs):,}")
    lines.append(" & ".join(n_row) + r" \\")

    r2_row = [r"Adjusted $R^2$"]
    for col in cols:
        r2_row.append(f"{models[col].rsquared_adj:.3f}")
    lines.append(" & ".join(r2_row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize")
    lines.append(rf"\item[] {note_extra}")
    lines.append(r"\item[] $^{*}p < 0.10$, $^{**}p < 0.05$, $^{***}p < 0.01$")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

cols = ['Low Status', 'Mid Status', 'High Status', 'All']

def fit_with_controls(mod):
    out = {}
    for label, sub in [
        ('Low Status',  df[df['stratum_group'] == 1]),
        ('Mid Status',  df[df['stratum_group'] == 2]),
        ('High Status', df[df['stratum_group'] == 3]),
    ]:
        fml = (f"success ~ C(group) * C({mod}) "
               f"+ position_z + reputation_10 + word_count_100")
        out[label] = smf.ols(fml, data=sub).fit(
            cov_type='cluster', cov_kwds={'groups': sub[CLUSTER_VAR]})
    fml_p = (f"success ~ C(group) * C({mod}) "
             f"+ position_z + reputation_10 + word_count_100 "
             f"+ C(stratum_group)")
    out['All'] = smf.ols(fml_p, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df[CLUSTER_VAR]})
    return out

def fit_no_controls(mod):
    out = {}
    for label, sub in [
        ('Low Status',  df[df['stratum_group'] == 1]),
        ('Mid Status',  df[df['stratum_group'] == 2]),
        ('High Status', df[df['stratum_group'] == 3]),
    ]:
        fml = f"success ~ C(group) * C({mod})"
        out[label] = smf.ols(fml, data=sub).fit(
            cov_type='cluster', cov_kwds={'groups': sub[CLUSTER_VAR]})
    fml_p = f"success ~ C(group) * C({mod}) + C(stratum_group)"
    out['All'] = smf.ols(fml_p, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df[CLUSTER_VAR]})
    return out

# Back-compat: populate models_with/models_nc for the primary moderator
_primary_mod = 'pct_empirical_claims_Q4'
models_with = fit_with_controls(_primary_mod)
models_nc   = fit_no_controls(_primary_mod)
models = models_with  # legacy alias for downstream print

# Run all 3 moderators, store full set for later table generation
all_fits = {}
for mod in MODERATORS:
    all_fits[mod] = dict(
        with_controls=fit_with_controls(mod),
        no_controls=fit_no_controls(mod),
    )

# ==============================================================
# Build LaTeX — six tables: 3 moderators × {with_controls, no_controls}
# ==============================================================
def note_with(mod_pretty):
    return (r"OLS estimates with standard errors clustered at the "
            r"challenger level (\texttt{author\_comment}) in parentheses. "
            r"Treatment indicates status badge concealed. Factual density "
            rf"is measured here by {mod_pretty}; quartiles range from Q1 "
            r"(lowest, baseline) to Q4 (highest). Controls include "
            r"normalized thread position (0--1), pre-experiment reputation "
            r"(scaled by 10), and response length (scaled by 100). The "
            r"pooled specification (column 4) includes status group fixed "
            r"effects.")

def note_nc(mod_pretty):
    return (r"OLS estimates with standard errors clustered at the "
            r"challenger level (\texttt{author\_comment}) in parentheses. "
            r"Treatment indicates status badge concealed. Factual density "
            rf"is measured here by {mod_pretty}; quartiles range from Q1 "
            r"(lowest, baseline) to Q4 (highest). No controls are "
            r"included; identification relies solely on random assignment "
            r"of \texttt{group}. The pooled specification (column 4) "
            r"includes status group fixed effects.")

print("Tables written:")
for mod, meta in MODERATORS.items():
    short = meta['short']; pretty = meta['pretty']
    term_order_full, term_labels_full = terms_for(mod)
    term_order_nc  = term_order_full[:4]
    term_labels_nc = {k: term_labels_full[k] for k in term_order_nc}

    # With controls
    output_with = build_latex(
        models=all_fits[mod]['with_controls'], cols=cols,
        term_order=term_order_full, term_labels=term_labels_full,
        caption=f"Treatment Effects Moderated by Factual-Evidence Density ({pretty})",
        label=f"tab:factual_density_{short}",
        note_extra=note_with(pretty))
    path_w = f"{TABLES_DIR}/table8_factual_density_{short}.tex"
    with open(path_w, "w") as f: f.write(output_with)
    print(f"  {path_w}")

    # No controls
    output_nc = build_latex(
        models=all_fits[mod]['no_controls'], cols=cols,
        term_order=term_order_nc, term_labels=term_labels_nc,
        caption=f"Treatment Effects Moderated by Factual-Evidence Density ({pretty}, No Controls)",
        label=f"tab:factual_density_{short}_nocontrols",
        note_extra=note_nc(pretty))
    path_nc = f"{TABLES_DIR}/table8_factual_density_{short}_nocontrols.tex"
    with open(path_nc, "w") as f: f.write(output_nc)
    print(f"  {path_nc}")

# Also keep the legacy file name for backwards compatibility with Phase 2 write-up
_compat_order, _compat_labels = terms_for(_primary_mod)
with open(f"{TABLES_DIR}/table8_factual_density_moderation.tex","w") as f:
    f.write(build_latex(models_with, cols, _compat_order, _compat_labels,
                        caption="Treatment Effects Moderated by Factual-Evidence Density",
                        label="tab:factual_density_moderation",
                        note_extra=note_with(MODERATORS[_primary_mod]['pretty'])))
with open(f"{TABLES_DIR}/table8_factual_density_moderation_nocontrols.tex","w") as f:
    f.write(build_latex(models_nc, cols, _compat_order[:4],
                        {k: _compat_labels[k] for k in _compat_order[:4]},
                        caption=("Treatment Effects Moderated by Factual-Evidence "
                                 "Density (No Controls, RCT Identification)"),
                        label="tab:factual_density_moderation_nocontrols",
                        note_extra=note_nc(MODERATORS[_primary_mod]['pretty'])))
print(f"  {TABLES_DIR}/table8_factual_density_moderation.tex  (compat alias)")
print(f"  {TABLES_DIR}/table8_factual_density_moderation_nocontrols.tex  (compat alias)")
def print_summary(title, m_dict, terms):
    print("\n" + "="*72)
    print(title)
    print("="*72)
    for col in cols:
        fit = m_dict[col]
        print(f"\n--- {col} (N={int(fit.nobs):,}, Adj-R^2={fit.rsquared_adj:.3f}) ---")
        for t in terms:
            if t in fit.params.index:
                c  = fit.params[t]; se = fit.bse[t]; p = fit.pvalues[t]
                print(f"  {t:<60} {c:+.3f}{stars(p):<3} (se={se:.3f}, p={p:.3f})")

# Print compact cross-moderator comparison for the headline interaction cells
print("\n" + "="*72)
print("CROSS-MODERATOR COMPARISON — key interaction coefficients")
print("="*72)
for mod in MODERATORS:
    term_order, _ = terms_for(mod)
    for ctrl_tag, fits in [('w/ controls', all_fits[mod]['with_controls']),
                           ('no controls', all_fits[mod]['no_controls'])]:
        print(f"\n{mod}  ({ctrl_tag})")
        for s_label in ['Low Status','Mid Status','High Status','All']:
            fit = fits[s_label]
            vals = {}
            for t in term_order[:4]:
                if t in fit.params.index:
                    vals[t] = (fit.params[t], fit.pvalues[t])
                else:
                    vals[t] = (None, None)
            pieces = []
            for t in term_order[:4]:
                c,p = vals[t]
                if c is None: pieces.append('  n/a  ')
                else:
                    s = stars(p)
                    pieces.append(f"{c:+.3f}{s:<3}")
            print(f"  {s_label:<12} | T: {pieces[0]}  xQ2: {pieces[1]}  xQ3: {pieces[2]}  xQ4: {pieces[3]}")
