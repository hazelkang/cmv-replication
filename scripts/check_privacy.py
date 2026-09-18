"""Check that released participant data contain only replacement IDs and numeric scores."""
from pathlib import Path
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1]
ID_PREFIXES = {'author_comment': 'challenger', 'submission_id': 'post', 'responder_id': 'dyad'}
FORBIDDEN_COLUMNS = {'author', 'author_submission', 'username', 'text', 'submission_text',
                     'conversation_with_op', 'title', 'entry_dt', 'comment_created_utc',
                     'evidence', 'reasoning', 'verification_note'}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    count = 0
    for path in sorted((ROOT / 'data').glob('*.csv')):
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            require(not FORBIDDEN_COLUMNS.intersection(reader.fieldnames or []),
                    f'{path.name}: disallowed identifying/text field')
            for line, row in enumerate(reader, 2):
                for column, value in row.items():
                    if value == '':
                        continue
                    context = f'{path.name}:{line} ({column})'
                    if column in ID_PREFIXES:
                        require(re.fullmatch(ID_PREFIXES[column] + r'_\d{6}', value),
                                context + ': expected a replacement ID')
                    elif column == 'super_cluster':
                        require(re.fullmatch(r'topic_\d{3}', value), context + ': expected a topic code')
                    elif column == 'label':
                        require(path.name == 'claim_validation_labels.csv' and
                                re.fullmatch(r'[A-Z][A-Z0-9_]*', value), context + ': expected an annotation label')
                    elif value not in ('True', 'False'):
                        try:
                            float(value)
                        except (TypeError, ValueError):
                            raise ValueError(context + ': unexpected free text') from None
        count += 1
    path = ROOT / 'data/unified_annotations_scores.jsonl'
    with path.open(encoding='utf-8') as stream:
        for line, text in enumerate(stream, 1):
            record = json.loads(text)
            require(set(record) == {'submission_id', 'responder_id', 'turns'},
                    f'{path.name}:{line}: unexpected fields')
            for column in ('submission_id', 'responder_id'):
                require(re.fullmatch(ID_PREFIXES[column] + r'_\d{6}', record[column]),
                        f'{path.name}:{line}: expected a replacement ID')
            for turn in record['turns']:
                require(set(turn) == {'A1_count', 'A1_binary', 'scores'}, 'Unexpected turn fields')
                require(isinstance(turn['A1_count'], int) and turn['A1_count'] >= 0, 'Invalid quote count')
                require(turn['A1_binary'] in (0, 1), 'Invalid quote indicator')
                require(set(turn['scores']) == {'A2', 'A3', 'B1', 'B2', 'C1', 'C2', 'C3'}, 'Unexpected score fields')
                require(all(v in (0, 1) for v in turn['scores'].values()), 'Unexpected nonbinary score')
    print(f'Privacy schema checks passed: {count} CSV files and the per-turn score file; no original IDs or free-text fields permitted.')

if __name__ == '__main__':
    main()
