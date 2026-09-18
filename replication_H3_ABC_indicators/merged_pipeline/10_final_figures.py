"""CONSOLIDATED final figures — same STRUCTURE + NAMING as the reference figures/ folder.

Reads the two measure tables produced by 07 (LLM A/B/C) and 08 (package/quantity), both at the
full 17,336-dyad frame (zero-filled), and writes the reference-matching figure set into
figures_final/{main,appendix}/.

MAIN (matches figures/main):
  fig1_engagement_overview   4-panel: Quantity(word count) | Representational | Operational |
                             Receptiveness, with a "Transactivity" bracket over panels 2–3 (overall)
  fig2_warmth_by_context     C3 warmth by {Overall, Subjective, Factual}

APPENDIX (matches figures/appendix; Empath split per request):
  appx_A_composite_by_context        appx_A_subindicators_grid   (A1,A2,A3)
  appx_B_composite_by_context        appx_B_subindicators_grid   (B1,B2)
  appx_C_composite_by_context        appx_C_subindicators_grid   (C1,C2,C3)
  appx_transactivity_AB_by_context
  appx_wordcount_by_context          appx_replies_by_context
  appx_politeness_gratitude_by_context   appx_receptiveness_by_context
  appx_flesch_by_context             appx_mattr_by_context
  appx_empath_<cat>_by_context  for politeness, social, cognitive_processes, certainty,
                                positive_emotion, negative_emotion, disputes
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).parent
OUT_MAIN=HERE/"figures_final"/"main"; OUT_APPX=HERE/"figures_final"/"appendix"
OUT_MAIN.mkdir(parents=True, exist_ok=True); OUT_APPX.mkdir(parents=True, exist_ok=True)

# ---- load both measure tables (both at the 17,336 frame) ----
llm=pd.read_csv(HERE/"data_unified_dyad_measures.csv")
pkg=pd.read_csv(Path(__file__).resolve().parents[2]/"data"/"data_package_measures_llmclean.csv")
PKG_COLS=["word_count_sum","n_replies","word_count_avg","flesch","mattr",
          "empath_politeness","empath_social","empath_cognitive_processes","empath_certainty",
          "empath_positive_emotion","empath_negative_emotion","empath_disputes",
          "polite_gratitude","receptiveness"]
df=llm.merge(pkg[["submission_id","responder_id"]+PKG_COLS], on=["submission_id","responder_id"], how="left")
print(f"[load] dyads={len(df):,}  (LLM {len(llm):,} + package merged)")

# ---- stats helpers (same conventions/style as the reference) ----
TIERS=[1,2,3,4,5,6,7]; LABELS=["1–9","10–19","20–29","30–39","40–49","50–99","100+"]; N_BOOT=1000
def boot_ci(x, seed=0):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    if len(x)<2: return (np.nan,np.nan)
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),size=(N_BOOT,len(x)))
    bm=x[idx].mean(axis=1); return np.percentile(bm,2.5),np.percentile(bm,97.5)
def sig_stars(p): return "" if not np.isfinite(p) else ("***" if p<.01 else "**" if p<.05 else "*" if p<.10 else "")
def cell_stats(data,col):
    rows=[]
    for s in TIERS:
        sub=data[data["stratum"]==s]
        v=sub.loc[sub["group"]==0,col].dropna(); h=sub.loc[sub["group"]==1,col].dropna()
        p=np.nan
        if min(len(v),len(h))>=10: _,p=stats.mannwhitneyu(v,h,alternative="two-sided")
        vlo,vhi=boot_ci(v.values,seed=s*10); hlo,hhi=boot_ci(h.values,seed=s*10+1)
        rows.append({"stratum":s,"v_m":v.mean() if len(v) else np.nan,"v_lo":vlo,"v_hi":vhi,
                     "h_m":h.mean() if len(h) else np.nan,"h_lo":hlo,"h_hi":hhi,"p":p})
    return pd.DataFrame(rows)

COLOR_V="#1f77b4"; COLOR_H="#ff7f0e"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,
    "axes.edgecolor":"#333333","axes.labelcolor":"#222222","xtick.color":"#222222","ytick.color":"#222222","legend.frameon":False})
XLAB="Challenger reputation tier (pre-experiment delta count)"

def _series(ax, s):
    xs=np.array(s["stratum"],float)
    ve=np.vstack([s["v_m"]-s["v_lo"],s["v_hi"]-s["v_m"]]); he=np.vstack([s["h_m"]-s["h_lo"],s["h_hi"]-s["h_m"]])
    ax.errorbar(xs,s["v_m"],yerr=ve,fmt='none',ecolor=COLOR_V,elinewidth=1,capsize=3,capthick=1,alpha=.6,zorder=1)
    ax.errorbar(xs,s["h_m"],yerr=he,fmt='none',ecolor=COLOR_H,elinewidth=1,capsize=3,capthick=1,alpha=.6,zorder=1)
    ax.plot(xs,s["v_m"],marker='o',ms=7,lw=1.8,color=COLOR_V,label='Untreated (status visible)',zorder=2)
    ax.plot(xs,s["h_m"],marker='s',ms=7,lw=1.8,ls='--',color=COLOR_H,label='Treated (status hidden)',zorder=2)
    for _,r in s.iterrows():
        a=sig_stars(r["p"])
        if a:
            top=np.nanmax([r["v_hi"],r["h_hi"]])
            ax.annotate(a,xy=(r["stratum"],top),xytext=(0,6),textcoords='offset points',ha='center',
                        fontsize=11,fontweight='bold' if a!='*' else 'normal',color='#222')
    ax.set_xticks(TIERS); ax.set_xticklabels(LABELS,fontsize=8,rotation=20); ax.grid(axis='y',alpha=.15,lw=.5)

def plot_panel(ax,col,ylabel,subtitle):   # overall-sample single panel (for fig1)
    _series(ax, cell_stats(df,col)); ax.set_ylabel(ylabel,fontsize=10)
    ax.set_title(subtitle,fontsize=12,fontweight='bold',pad=14,color='#333333')
    y0,y1=ax.get_ylim(); ax.set_ylim(y0,y1+0.20*(y1-y0))

def three_panel(col,ylabel,stem):
    fig,ax=plt.subplots(1,3,figsize=(15,5.2),sharey=True)
    for j,(t,cv) in enumerate([("Overall (all dyads)",None),("Subjective (Value-driven)",1),("Factual (Utility-driven)",0)]):
        d=df if cv is None else df[df["value_driven"]==cv]
        _series(ax[j], cell_stats(d,col)); ax[j].set_title(t,fontsize=12,fontweight='bold',pad=14)
        if j==0: ax[j].set_ylabel(ylabel,fontsize=10)
    y0,y1=ax[0].get_ylim(); ax[0].set_ylim(y0,y1+0.15*(y1-y0))
    fig.subplots_adjust(top=0.84,bottom=0.26,left=0.06,right=0.99,wspace=0.10)
    fig.text(0.52,0.10,XLAB,ha='center',va='center',fontsize=10)
    h,l=ax[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',bbox_to_anchor=(0.52,0.005),ncol=2,fontsize=10,frameon=False,columnspacing=2.5)
    for e in ("pdf","png"): fig.savefig(OUT_APPX/f"{stem}.{e}",dpi=300,bbox_inches='tight')
    plt.close(fig); print(f"  [appx] {stem}")

def grid(rows_spec,stem):
    nr=len(rows_spec); fh=2.7*nr+1.6
    fig,ax=plt.subplots(nr,3,figsize=(15,fh),sharey='row',sharex=True)
    if nr==1: ax=ax.reshape(1,3)
    titles=[("Overall (all dyads)",None),("Subjective (Value-driven)",1),("Factual (Utility-driven)",0)]
    for i,(col,rl) in enumerate(rows_spec):
        for j,(t,cv) in enumerate(titles):
            d=df if cv is None else df[df["value_driven"]==cv]
            _series(ax[i,j], cell_stats(d,col))
            if i==0: ax[i,j].set_title(t,fontsize=12,fontweight='bold',pad=14)
            if j==0: ax[i,j].set_ylabel(rl,fontsize=10)
            ax[i,j].set_xticks(TIERS); ax[i,j].set_xticklabels(LABELS if i==nr-1 else [],fontsize=8,rotation=20)
        y0,y1=ax[i,0].get_ylim(); ax[i,0].set_ylim(y0,y1+0.15*(y1-y0))
    fig.subplots_adjust(top=1-0.75/fh,bottom=1.35/fh,left=0.07,right=0.99,wspace=0.10,hspace=0.32)
    fig.text(0.52,0.62/fh,XLAB,ha='center',va='center',fontsize=10)
    h,l=ax[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',bbox_to_anchor=(0.52,0.18/fh),ncol=2,fontsize=10,frameon=False,columnspacing=2.5)
    for e in ("pdf","png"): fig.savefig(OUT_APPX/f"{stem}.{e}",dpi=300,bbox_inches='tight')
    plt.close(fig); print(f"  [appx] {stem}")

# ================= MAIN =================
# fig1_engagement_overview — 4-panel with Transactivity bracket over panels 2–3
fig,axes=plt.subplots(1,4,figsize=(17,5.2),sharey=False)
plot_panel(axes[0],"word_count_sum","Word count per dyad","Quantity")
plot_panel(axes[1],"mean_representational","Representational composite","Representational")
plot_panel(axes[2],"mean_operational","Operational composite","Operational")
plot_panel(axes[3],"mean_receptiveness","Receptiveness composite","Receptiveness")
fig.subplots_adjust(top=0.78,bottom=0.26,left=0.05,right=0.99,wspace=0.30)
pos1=axes[1].get_position(); pos2=axes[2].get_position(); x0,x1=pos1.x0,pos2.x1; y_br=0.90; tick=0.022
fig.add_artist(Line2D([x0,x1],[y_br,y_br],transform=fig.transFigure,color="#555555",linewidth=1.3))
fig.add_artist(Line2D([x0,x0],[y_br,y_br-tick],transform=fig.transFigure,color="#555555",linewidth=1.3))
fig.add_artist(Line2D([x1,x1],[y_br,y_br-tick],transform=fig.transFigure,color="#555555",linewidth=1.3))
fig.text((x0+x1)/2,y_br+0.015,"Transactivity",ha="center",va="bottom",fontsize=14,fontweight="bold",color="#222222")
fig.text(0.52,0.105,XLAB,ha="center",va="center",fontsize=10)
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",bbox_to_anchor=(0.52,0.005),ncol=2,fontsize=10,frameon=False,handletextpad=0.6,columnspacing=2.5)
for e in ("pdf","png"): fig.savefig(OUT_MAIN/f"fig1_engagement_overview.{e}",dpi=300,bbox_inches='tight')
plt.close(fig); print("  [main] fig1_engagement_overview")

# fig2_warmth_by_context
def three_panel_main(col,ylabel,stem):
    fig,ax=plt.subplots(1,3,figsize=(15,5.2),sharey=True)
    for j,(t,cv) in enumerate([("Overall (all dyads)",None),("Subjective (Value-driven)",1),("Factual (Utility-driven)",0)]):
        d=df if cv is None else df[df["value_driven"]==cv]
        _series(ax[j], cell_stats(d,col)); ax[j].set_title(t,fontsize=12,fontweight='bold',pad=14)
        if j==0: ax[j].set_ylabel(ylabel,fontsize=10)
    y0,y1=ax[0].get_ylim(); ax[0].set_ylim(y0,y1+0.15*(y1-y0))
    fig.subplots_adjust(top=0.84,bottom=0.26,left=0.06,right=0.99,wspace=0.10)
    fig.text(0.52,0.10,XLAB,ha='center',va='center',fontsize=10)
    h,l=ax[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',bbox_to_anchor=(0.52,0.005),ncol=2,fontsize=10,frameon=False,columnspacing=2.5)
    for e in ("pdf","png"): fig.savefig(OUT_MAIN/f"{stem}.{e}",dpi=300,bbox_inches='tight')
    plt.close(fig); print(f"  [main] {stem}")
three_panel_main("mean_C3","Warmth (C3) — dyad mean","fig2_warmth_by_context")

# ================= APPENDIX =================
three_panel("mean_representational","Representational composite","appx_A_composite_by_context")
three_panel("mean_operational","Operational composite","appx_B_composite_by_context")
three_panel("mean_receptiveness","Receptiveness composite","appx_C_composite_by_context")
three_panel("mean_transactivity","Transactivity composite (A+B)","appx_transactivity_AB_by_context")
grid([("mean_A1_binary","A1 quoting"),("mean_A2","A2 paraphrase"),("mean_A3","A3 clarification")],"appx_A_subindicators_grid")
grid([("mean_B1","B1 extension"),("mean_B2","B2 critique")],"appx_B_subindicators_grid")
grid([("mean_C1","C1 acknowledgment"),("mean_C2","C2 hedging"),("mean_C3","C3 warmth")],"appx_C_subindicators_grid")
three_panel("word_count_sum","Total word count per dyad","appx_wordcount_by_context")
three_panel("n_replies","Decision-maker replies per dyad","appx_replies_by_context")
three_panel("polite_gratitude","Politeness: gratitude (per 100 words)","appx_politeness_gratitude_by_context")
three_panel("receptiveness","Yeomans receptiveness (dyad mean)","appx_receptiveness_by_context")
three_panel("flesch","Flesch readability","appx_flesch_by_context")
three_panel("mattr","MATTR lexical diversity","appx_mattr_by_context")
for cat in ["politeness","social","cognitive_processes","certainty","positive_emotion","negative_emotion","disputes"]:
    three_panel(f"empath_{cat}",f"Empath: {cat.replace('_',' ')} (per 100 words)",f"appx_empath_{cat}_by_context")

print("\n[done] final figures written to figures_final/ (reference structure + naming).")
