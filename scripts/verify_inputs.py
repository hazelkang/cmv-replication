from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT/'data/MANIFEST.json').read_text())
for rel, spec in manifest.items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == spec['sha256'], f'Input checksum failed: {rel}'
h1 = pd.read_csv(ROOT/'data/cmv_experiment_data_19965_regression.csv')
h2 = pd.read_csv(ROOT/'data/cmv_experiment_data_17336_with_measures.csv')
assert len(h1) == 19965 and len(h2) == 17336
assert h2.responder_id.is_unique
assert h2.author_comment.nunique() == 2784
assert h1.groupby('subgroup').size().to_dict() == {1:18892, 2:844, 3:229}
assert h2.groupby('stratum_group').size().to_dict() == {1:10244, 2:2866, 3:4226}
assert set(h1.group) == {0,1} and set(h2.group) == {0,1}
print(f'Checksums passed for {len(manifest)} input files; sample sizes and clustering keys verified.')
