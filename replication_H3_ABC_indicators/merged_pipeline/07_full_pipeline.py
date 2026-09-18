"""STEP 7 — FULL PIPELINE: unified LLM annotations + deterministic A1
  → per-dyad means (mean across the OP's turns)
  → ZERO-FILL to the entire 17,336-dyad experiment frame (dyads with no annotation = 0)
  → 7-stratum × {All, Subjective, Factual} Wilcoxon (Mann-Whitney U)
  → main + appendix figures (bootstrap 95% CI, same style as the existing main/appendix figures).

Run AFTER the full LLM run produces unified_annotations_full.jsonl. Falls back to
unified_annotations_100.jsonl (the validation run) if the full file is absent, with a warning,
so the pipeline can be dry-run now.

Measurement set (9 indicators + 4 composites):
  A1 (deterministic {…} count→binary), A2, A3 [representational]
  B1, B2 [operational]   (B3 dropped — failed validation)
  C1, C2, C3 [receptiveness]
  composites: representational = A1_binary+A2+A3 ; operational = B1+B2 ;
              receptiveness = C1+C2+C3 ; transactivity = representational+operational

Per-dyad value = MEAN of the per-turn binary across the dyad's OP turns (A1 uses A1_binary).
Whole-sample convention: every experiment dyad is included; dyads with no LLM annotation get 0
on every measure (zero-fill), matching the established figures.

Outputs (under merged_pipeline/):
  data_unified_dyad_measures.csv
  results_wilcoxon_unified_7strata_full.csv
  figures_final/main/{fig1_composite_overview, fig2_transactivity_by_context}.{pdf,png}
  figures_final/appendix/{appx_A_composite_by_context, appx_B_composite_by_context,
     appx_C_composite_by_context, appx_transactivity_by_context,
     appx_A_subindicators_grid, appx_B_subindicators_grid, appx_C_subindicators_grid}.{pdf,png}
"""
from __future__ import annotations
import os, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
EXPT = Path(__file__).resolve().parents[2] / "data" / "cmv_experiment_data_17336_with_measures.csv"
ANNOT = Path(__file__).resolve().parents[2] / "data" / "unified_annotations_scores.jsonl"
if not ANNOT.exists():
    raise FileNotFoundError("Full annotation scores are required; no validation-subset fallback is permitted.")

OUT_MAIN = HERE/"figures_final"/"main"; OUT_APPX = HERE/"figures_final"/"appendix"
OUT_MAIN.mkdir(parents=True, exist_ok=True); OUT_APPX.mkdir(parents=True, exist_ok=True)

LLM_KS = ["A2","A3","B1","B2","C1","C2","C3"]   # B3 dropped (failed validation: rare + unstable)
ALL_IND = ["A1_binary","A2","A3","B1","B2","C1","C2","C3"]  # A1_binary used in composites

# ---------------- STEP 1: per-dyad means from annotations ----------------
rows=[]
for line in ANNOT.open():
    r=json.loads(line); turns=r.get("turns",[])
    if not turns: continue
    n=len(turns)
    agg={"submission_id":r["submission_id"],"responder_id":r["responder_id"],"n_op_turns":n}
    agg["mean_A1_count"]  = sum(t.get("A1_count",0) for t in turns)/n
    agg["mean_A1_binary"] = sum(t.get("A1_binary",0) for t in turns)/n
    for k in LLM_KS:
        agg[f"mean_{k}"] = sum(int((t.get("scores") or {}).get(k,0)) for t in turns)/n
    rows.append(agg)
dyad=pd.DataFrame(rows)
print(f"[step1] annotated dyads: {len(dyad):,}")

# ---------------- STEP 2: merge to experiment frame + zero-fill ALL 17,336 ----------------
expt=pd.read_csv(EXPT, low_memory=False,
                 usecols=["submission_id","responder_id","group","stratum","pct_empirical_claims_Q4"])
expt["value_driven"]=expt["pct_empirical_claims_Q4"].isin([1,2]).astype(int)
df=expt.merge(dyad, on=["submission_id","responder_id"], how="left")
df=df[df["value_driven"].notna()].copy()
MEAS_RAW=["mean_A1_count","mean_A1_binary"]+[f"mean_{k}" for k in LLM_KS]
for c in MEAS_RAW: df[c]=df[c].fillna(0.0)
print(f"[step2] whole-sample dyads: {len(df):,}  | with annotation: {df['n_op_turns'].notna().sum():,}  | zero-filled: {df['n_op_turns'].isna().sum():,}")

# ---------------- STEP 3: composites ----------------
df["mean_representational"]=df["mean_A1_binary"]+df["mean_A2"]+df["mean_A3"]
df["mean_operational"]     =df["mean_B1"]+df["mean_B2"]
df["mean_receptiveness"]   =df["mean_C1"]+df["mean_C2"]+df["mean_C3"]
df["mean_transactivity"]   =df["mean_representational"]+df["mean_operational"]
df.to_csv(HERE/"data_unified_dyad_measures.csv", index=False)
print(f"[step3] saved data_unified_dyad_measures.csv")

# ---------------- STEP 4: Wilcoxon (7 strata × 3 content cuts) ----------------
TIERS=[1,2,3,4,5,6,7]; LABELS=["1–9","10–19","20–29","30–39","40–49","50–99","100+"]
CUTS=[("All",None),("Subjective",1),("Factual",0)]
N_BOOT=1000
def boot_ci(x, seed=0):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    if len(x)<2: return (np.nan,np.nan)
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),size=(N_BOOT,len(x)))
    bm=x[idx].mean(axis=1); return np.percentile(bm,2.5),np.percentile(bm,97.5)
def stars(p):
    return "" if not np.isfinite(p) else ("***" if p<.01 else "**" if p<.05 else "*" if p<.10 else "")
def cell(data,col):
    out=[]
    for s in TIERS:
        sub=data[data["stratum"]==s]
        v=sub.loc[sub["group"]==0,col]; h=sub.loc[sub["group"]==1,col]
        p=W=np.nan
        if min(len(v),len(h))>=10: W,p=stats.mannwhitneyu(v,h,alternative="two-sided")
        vlo,vhi=boot_ci(v.values,seed=s*10); hlo,hhi=boot_ci(h.values,seed=s*10+1)
        out.append(dict(stratum=s,v_m=v.mean() if len(v) else np.nan,v_lo=vlo,v_hi=vhi,v_n=len(v),
                        h_m=h.mean() if len(h) else np.nan,h_lo=hlo,h_hi=hhi,h_n=len(h),W=W,p=p))
    return pd.DataFrame(out)

MEASURES=[("mean_A1_count","A1 quoting (count)"),("mean_A1_binary","A1 quoting (binary)"),
          ("mean_A2","A2 paraphrase"),("mean_A3","A3 clarification"),
          ("mean_B1","B1 extension"),("mean_B2","B2 critique"),
          ("mean_C1","C1 acknowledgment"),("mean_C2","C2 hedging"),("mean_C3","C3 warmth"),
          ("mean_representational","Representational composite"),("mean_operational","Operational composite"),
          ("mean_receptiveness","Receptiveness composite"),("mean_transactivity","Transactivity composite")]
wrows=[]
for col,lab in MEASURES:
    for cut_lab,cv in CUTS:
        d=df if cv is None else df[df["value_driven"]==cv]
        cs=cell(d,col)
        for _,r in cs.iterrows():
            wrows.append(dict(measure=col,label=lab,content=cut_lab,stratum=int(r["stratum"]),
                              tier=LABELS[int(r["stratum"])-1],n_visible=int(r["v_n"]),n_hidden=int(r["h_n"]),
                              mean_visible=r["v_m"],mean_hidden=r["h_m"],delta=(r["v_m"]-r["h_m"]),
                              U_stat=r["W"],p_value=r["p"],sig=stars(r["p"])))
wf=pd.DataFrame(wrows); wf.to_csv(HERE/"results_wilcoxon_unified_7strata_full.csv",index=False)
print(f"[step4] saved results_wilcoxon_unified_7strata_full.csv ({len(wf)} rows; p<.05={int((wf['p_value']<.05).sum())})")

# ---------------- figures: see 10_final_figures.py ----------------
# Figure generation has moved to 10_final_figures.py (the single source of truth for figures),
# which reads this file's data_unified_dyad_measures.csv together with the package measures and
# emits the reference-matching figures_final/ set. This script now produces MEASURES + WILCOXON only.
print("\n[done] measures + wilcoxon written. Run 10_final_figures.py to (re)build figures_final/.")
print("  measures: data_unified_dyad_measures.csv")
print("  wilcoxon: results_wilcoxon_unified_7strata_full.csv")
