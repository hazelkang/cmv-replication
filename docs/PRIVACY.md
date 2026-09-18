# Participant identifier audit

The public release does not distribute participant usernames, original Reddit post/comment IDs, raw conversation text, annotation quotations, or a lookup from replacement IDs to Reddit accounts.

On 2026-09-18, the two published commits preceding this audit were checked, covering 63 distinct Git blobs and all eight released data files. Identifier fields were checked against 21,096 distinct source participant usernames and 19,051 original post/comment IDs. No original identifiers were found in the released identifier fields, no unexpected free-text data fields were found, and no Reddit user-profile references were found in the repository's text files.

- `author_comment` contains replacement challenger codes such as `challenger_000001`, despite its historical column name.
- `submission_id` and `responder_id` contain replacement post and dyad codes.
- The replacement codes preserve joins and clustering. No reverse-lookup file is published.
- Author names in the paper citation and the repository owner's GitHub handle identify the researchers, not study participants, and remain public.

`python scripts/check_privacy.py` checks the released data schemas and replacement-ID formats. It is also run automatically by `python replicate.py`. It rejects original-style identifiers and unexpected text fields; its operation was verified with deliberately invalid username and evidence-field inputs.

The data are **pseudonymized**, not guaranteed impossible to reidentify: analysis covariates remain available for replication. This audit verifies removal of direct participant identifiers and text; it does not claim formal anonymization or protection against every possible linkage attack.
