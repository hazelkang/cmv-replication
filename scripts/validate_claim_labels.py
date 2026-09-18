"""Reproduce the annotation audit's per-label precision (recall is not identifiable)."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'data/claim_validation_labels.csv').dropna(subset=['label','human_check'])
assert len(df)==200 and set(df.human_check)<={0,1}
result=df.groupby('label').human_check.agg(precision='mean',support='count')
ref=pd.read_csv(ROOT/'reference/annotation_accuracy_per_label.csv',index_col='label')
for c in ['precision','support']:
    assert np.allclose(result.loc[ref.index,c],ref[c],rtol=1e-10,atol=1e-10)
out=ROOT/'replication_H2/results';out.mkdir(parents=True,exist_ok=True)
result.to_csv(out/'claim_annotation_precision.csv')
print(f'Precision/support for {len(result)} labels reproduced from {len(df)} human checks. Recall is not estimated.')
