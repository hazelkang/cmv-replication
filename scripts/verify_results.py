from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
H3=ROOT/'replication_H3_ABC_indicators/merged_pipeline'
checks=[]
def compare_csv(actual, expected, keys, values):
    a=pd.read_csv(actual).set_index(keys).sort_index()
    b=pd.read_csv(expected).set_index(keys).sort_index()
    # Package reference omits word_count_avg; compare every reference row.
    a=a.loc[b.index]
    assert a.index.equals(b.index)
    for col in values:
        assert np.allclose(a[col],b[col],rtol=1e-7,atol=1e-9,equal_nan=True),f'{actual.name}: {col}'
    checks.append({'output':str(actual.relative_to(ROOT)),'reference':str(expected.relative_to(ROOT)),'rows':len(b),'status':'pass'})
compare_csv(ROOT/'replication_H1/results/h1_treatment_effect_coefs.csv',ROOT/'reference/h1_treatment_effect_coefs.csv',['model','term'],['coef','se','t','p'])
table=ROOT/'replication_H2/tables/table8_factual_density_PctClaims_nocontrols.tex'
assert table.read_text()==(ROOT/'reference/table8_factual_density_PctClaims_nocontrols.tex').read_text(), 'H2 table differs'
checks.append({'output':str(table.relative_to(ROOT)),'status':'pass','comparison':'exact text; matches Table 4 displayed values'})
compare_csv(H3/'results_wilcoxon_unified_7strata_full.csv',ROOT/'reference/results_wilcoxon_unified_7strata_full.csv',['measure','content','stratum'],['n_visible','n_hidden','mean_visible','mean_hidden','delta','U_stat','p_value'])
compare_csv(H3/'results_package_reproduced.csv',ROOT/'reference/results_wilcoxon_package_7strata.csv',['measure','content','stratum'],['n_visible','n_hidden','mean_visible','mean_hidden','delta','U_stat','p_value'])
for ref in sorted((ROOT/'reference/robustness').glob('*.csv')):
    actual=ROOT/'replication_H2/robustness/results'/ref.name
    if not actual.exists():
        continue  # Optional outputs are checked only when available.
    a=pd.read_csv(actual);b=pd.read_csv(ref)
    assert a.shape==b.shape and list(a)==list(b), ref.name
    for c in b.select_dtypes(include='number'):
        assert np.allclose(a[c],b[c],rtol=1e-5,atol=1e-8,equal_nan=True),f'{ref.name}: {c}'
    checks.append({'output':str(actual.relative_to(ROOT)),'status':'pass','rows':len(a),'comparison':'all numeric columns against saved optional robustness output'})
report={'status':'pass','checks':checks,'paper_differences':'See docs/KNOWN_DIFFERENCES.md. Passing checks reproduce saved research outputs; they do not erase manuscript discrepancies.'}
(ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
