"""Run the offline replication from any working directory."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robustness', action='store_true', help='Also run H2 topic FE, weighting, CEM, controls and alternative moderators')
    p.add_argument('--simex', action='store_true', help='Also run expensive SIMEX; B and R environment variables control repetitions')
    p.add_argument('--no-figures', action='store_true')
    args = p.parse_args()
    scripts = ['scripts/verify_inputs.py', 'scripts/rebuild_claim_measures.py', 'scripts/validate_claim_labels.py',
               'replication_H1/analysis/run_h1_treatment_effect.py',
               'replication_H1/analysis/run_balance_check.py',
               'replication_H2/analysis/run_table8.py',
               'replication_H3_ABC_indicators/merged_pipeline/07_full_pipeline.py',
               'scripts/engagement_sensitivity.py']
    if not args.no_figures:
        scripts += ['replication_H3_ABC_indicators/merged_pipeline/10_final_figures.py']
    if args.robustness:
        scripts += [f'replication_H2/robustness/analysis/{s}.py' for s in
                    ['run_topic_fe', 'run_weighting', 'run_cem', 'run_additional_controls', 'run_alt_measures', 'run_poster_heuristics']]
    if args.simex:
        scripts += ['replication_H2/robustness/analysis/run_simex.py']
    scripts += ['scripts/verify_results.py']
    env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(ROOT / '.cache' / 'matplotlib'), XDG_CACHE_HOME=str(ROOT / '.cache'))
    started = time.monotonic()
    for script in scripts:
        print(f'\n=== {script} ===', flush=True)
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env, check=True)
    print(f'\nReplication completed in {time.monotonic()-started:.1f}s. See verification.json and docs/KNOWN_DIFFERENCES.md.')

if __name__ == '__main__':
    main()
