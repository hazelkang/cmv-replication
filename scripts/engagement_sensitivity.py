"""Reproduce package tests and add a separately labeled scored-dyad sensitivity."""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

ROOT=Path(__file__).resolve().parents[1]
H3=ROOT/'replication_H3_ABC_indicators/merged_pipeline'
llm=pd.read_csv(H3/'data_unified_dyad_measures.csv')
pkg=pd.read_csv(ROOT/'data/data_package_measures_llmclean.csv',float_precision='round_trip')

def calculate(df, columns, sample):
    rows=[]
    for c in columns:
        for content,mask in [('All',np.ones(len(df),dtype=bool)),('Subjective',df.value_driven==1),('Factual',df.value_driven==0)]:
            for s in range(1,8):
                d=df.loc[mask & (df.stratum==s)]
                v=d.loc[d.group==0,c].dropna();h=d.loc[d.group==1,c].dropna()
                stat,p=(np.nan,np.nan)
                if min(len(v),len(h))>=10:stat,p=mannwhitneyu(v,h,alternative='two-sided')
                rows.append(dict(measure=c,sample=sample,content=content,stratum=s,n_visible=len(v),n_hidden=len(h),mean_visible=v.mean(),mean_hidden=h.mean(),delta=v.mean()-h.mean(),U_stat=stat,p_value=p))
    return pd.DataFrame(rows)

pkgcols=[c for c in pkg if c not in ['submission_id','responder_id','group','stratum','pct_empirical_claims_Q4','value_driven']]
calculate(pkg,pkgcols,'saved package frame; missing values excluded per measure').to_csv(H3/'results_package_reproduced.csv',index=False)
conditional=llm[llm.n_op_turns.notna()].copy()
assert len(llm)==17336 and len(conditional)==8286
calculate(conditional,[c for c in llm if c.startswith('mean_')],'conditional on available annotation (8,286 dyads)').to_csv(H3/'results_llm_conditional_sensitivity.csv',index=False)
pd.DataFrame([{'frame':'Full experiment','n':len(llm)},{'frame':'With scored turns','n':len(conditional)},{'frame':'Zero-filled indicator rows','n':llm.n_op_turns.isna().sum()}]).to_csv(H3/'sample_flow.csv',index=False)
print('Package tests and separately labeled 8,286-dyad conditional sensitivity written.')
