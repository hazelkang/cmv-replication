"""Rebuild ECD and moderator quartiles from text-free claim counts."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = pd.read_csv(ROOT/'data/submission_claim_counts.csv')
assert m.submission_id.is_unique
total = m.empirical_cnt + m.non_empirical_cnt
words = m.empirical_words + m.non_empirical_words
m['pct_empirical_claims'] = m.empirical_cnt / total.replace(0, 1)
m['pct_empirical_words'] = m.empirical_words / words.replace(0, 1)
def logit(x):
    x = np.clip(x, 1e-6, 1-1e-6)
    return np.log(x/(1-x))
m['emp_claim_logit'] = logit((m.empirical_cnt+0.5)/(total+1))
m['pct_emp_logit'] = logit(m.pct_empirical_claims)
for c in ['pct_empirical_claims','pct_empirical_words','emp_claim_logit','pct_emp_logit']:
    m[c+'_Q4'] = pd.qcut(m[c], 4, labels=[1,2,3,4]).astype(int)
    m[c+'_Q2'] = (m[c+'_Q4'] == 4).astype(int)
full = pd.read_csv(ROOT/'data/cmv_experiment_data_17336_with_measures.csv')
joined = full.merge(m,on='submission_id',suffixes=('_saved','_rebuilt'),validate='many_to_one')
assert len(joined)==len(full)
for c in ['pct_empirical_claims','pct_empirical_words','emp_claim_logit','pct_emp_logit']:
    for name in [c,c+'_Q4',c+'_Q2']:
        assert np.allclose(joined[name+'_saved'],joined[name+'_rebuilt'],rtol=1e-10,atol=1e-10,equal_nan=True),name
out=ROOT/'replication_H2/results';out.mkdir(parents=True,exist_ok=True)
m.to_csv(out/'measures_with_quantiles.csv',index=False)
print(f'Rebuilt {len(m)} submission-level measures; all released measures and bins agree.')
