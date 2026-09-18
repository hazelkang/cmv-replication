# Data dictionary

CSV files use a header row, UTF-8 encoding, and empty fields for missing values. They preserve the source row order. Release-specific IDs carry no public lookup to Reddit.

| File | Rows | Unit |
|---|---:|---|
| `cmv_experiment_data_17336.csv` | 17,336 | Root-level debate (legacy balance frame) |
| `cmv_experiment_data_17336_with_measures.csv` | 17,336 | Root-level debate (recomputed H2 moderators) |
| `cmv_experiment_data_18590_balance_check.csv` | 18,590 | Baseline challenger |
| `cmv_experiment_data_19965_regression.csv` | 19,965 | Challenger |
| `data_package_measures_llmclean.csv` | 17,336 | Root-level debate (saved text-package measures) |
| `submission_claim_counts.csv` | 1,715 | Original post |

## Fields

`claim_validation_labels.csv` additionally contains the 200 original human-check rows with only `label` (the model-assigned annotation category) and `human_check` (1 = confirmed by the human reviewer, 0 = incorrect). No rationale text or post IDs are included. This sample supports per-label precision, not independent recall estimation.

| Field | Meaning |
|---|---|
| `author_comment` | Release-specific challenger ID; preserves clustering across debates. |
| `average_comment_length` | Saved average comment length. |
| `cmv_tenure` | Saved CMV tenure, in days. |
| `comment_karma` | Saved pre-experiment comment karma. |
| `deltas_received` | Saved pre-experiment deltas received. |
| `emp_claim_logit` | Logit of (EV+0.5)/(EV+NEV+1). |
| `emp_claim_logit_Q2` | Legacy binary flag equal to 1 when `emp_claim_logit_Q4` is 4; not a median split. |
| `emp_claim_logit_Q4` | Quartile (1–4) of `emp_claim_logit`, formed at the submission level with pandas.qcut. |
| `empath_certainty` | Saved Empath certainty measure per 100 words, zero-filled by the source pipeline. |
| `empath_cognitive_processes` | Saved Empath cognitive processes measure per 100 words, zero-filled by the source pipeline. |
| `empath_disputes` | Saved Empath disputes measure per 100 words, zero-filled by the source pipeline. |
| `empath_negative_emotion` | Saved Empath negative emotion measure per 100 words, zero-filled by the source pipeline. |
| `empath_politeness` | Saved Empath politeness measure per 100 words, zero-filled by the source pipeline. |
| `empath_positive_emotion` | Saved Empath positive emotion measure per 100 words, zero-filled by the source pipeline. |
| `empath_social` | Saved Empath social measure per 100 words, zero-filled by the source pipeline. |
| `empirical_cnt` | Count of claims with an EV_* annotation label. |
| `empirical_cnt_y` | Rebuilt count of empirical claims, merged into the H2 frame. |
| `empirical_words` | Word count in EV_* evidence quotes, after the source URL normalization. |
| `flesch` | Saved Flesch readability; missing when unavailable. |
| `gold` | Saved Reddit Gold indicator. |
| `group` | Randomized assignment: 1 = status concealed; 0 = visible. |
| `group_int_stratum1` | H1 interaction between concealed-status assignment and the indicated tier/stratum dummy. |
| `group_int_stratum2` | H1 interaction between concealed-status assignment and the indicated tier/stratum dummy. |
| `group_int_stratum3` | H1 interaction between concealed-status assignment and the indicated tier/stratum dummy. |
| `group_int_stratum4` | H1 interaction between concealed-status assignment and the indicated tier/stratum dummy. |
| `group_int_stratum5` | H1 interaction between concealed-status assignment and the indicated tier/stratum dummy. |
| `group_int_stratum6` | H1 interaction between concealed-status assignment and the indicated tier/stratum dummy. |
| `group_int_stratum7` | H1 interaction between concealed-status assignment and the indicated tier/stratum dummy. |
| `group_int_subgroup1` | H1 interaction between concealed-status assignment and the indicated tier/stratum dummy. |
| `group_int_subgroup2` | H1 interaction between concealed-status assignment and the indicated tier/stratum dummy. |
| `group_int_subgroup3` | H1 interaction between concealed-status assignment and the indicated tier/stratum dummy. |
| `link_karma` | Saved pre-experiment link karma. |
| `mattr` | Saved moving-average type-token ratio; missing when unavailable. |
| `n_replies` | Saved count of direct decision-maker replies; zero-filled. |
| `non_empirical_cnt` | Count of claims with an NEV annotation label. |
| `non_empirical_cnt_y` | Rebuilt count of non-empirical claims, merged into the H2 frame. |
| `non_empirical_words` | Word count in NEV evidence quotes, after the source URL normalization. |
| `num_cmv_comments` | Saved CMV comment count. |
| `num_cmv_post` | Saved CMV post count. |
| `pct_emp_logit` | Logit of empirical-claim share clipped to [1e-6,1-1e-6]. |
| `pct_emp_logit_Q2` | Legacy binary flag equal to 1 when `pct_emp_logit_Q4` is 4; not a median split. |
| `pct_emp_logit_Q4` | Quartile (1–4) of `pct_emp_logit`, formed at the submission level with pandas.qcut. |
| `pct_empirical_claims` | EV claim count divided by EV+NEV count; zero for zero denominator. |
| `pct_empirical_claims_Q2` | Legacy binary flag equal to 1 when `pct_empirical_claims_Q4` is 4; not a median split. |
| `pct_empirical_claims_Q4` | Quartile (1–4) of `pct_empirical_claims`, formed at the submission level with pandas.qcut. |
| `pct_empirical_words` | EV quote words divided by EV+NEV quote words; zero for zero denominator. |
| `pct_empirical_words_Q2` | Legacy binary flag equal to 1 when `pct_empirical_words_Q4` is 4; not a median split. |
| `pct_empirical_words_Q4` | Quartile (1–4) of `pct_empirical_words`, formed at the submission level with pandas.qcut. |
| `polite_gratitude` | Saved politeness gratitude measure per 100 words; zero-filled. |
| `position` | Saved comment-position measure. |
| `position_z` | Saved standardized position; main H2 code recomputes it from position. |
| `poster_length` | Saved original-post length in characters. |
| `poster_length_Q` | Saved quartile of original-post length. |
| `poster_tenure` | Saved original-poster CMV tenure. |
| `poster_tenure_Q` | Saved quartile of original-poster tenure. |
| `receptiveness` | Saved Yeomans receptiveness score; missing when unavailable. |
| `reputation` | Saved challenger reputation measure. |
| `reputation_10` | Reputation scaled by 10. |
| `reputation_poster` | Saved original-poster reputation. |
| `reputation_poster_Q` | Saved quartile of original-poster reputation. |
| `responder_id` | Release-specific focal root-comment/dyad ID; not a challenger-level cluster. |
| `stratum` | Seven pre-experiment reputation bins: 1–9, 10–19, 20–29, 30–39, 40–49, 50–99, 100+ deltas. |
| `stratum_1` | H1 dummy for the indicated seven-bin reputation stratum. |
| `stratum_2` | H1 dummy for the indicated seven-bin reputation stratum. |
| `stratum_3` | H1 dummy for the indicated seven-bin reputation stratum. |
| `stratum_4` | H1 dummy for the indicated seven-bin reputation stratum. |
| `stratum_5` | H1 dummy for the indicated seven-bin reputation stratum. |
| `stratum_6` | H1 dummy for the indicated seven-bin reputation stratum. |
| `stratum_7` | H1 dummy for the indicated seven-bin reputation stratum. |
| `stratum_group` | Three tiers: 1 = stratum 1; 2 = strata 2–4; 3 = strata 5–7. |
| `subgroup` | H1 three-tier grouping, using the same partition as stratum_group. |
| `subgroup_1` | H1 dummy for the indicated three-tier reputation group. |
| `subgroup_2` | H1 dummy for the indicated three-tier reputation group. |
| `subgroup_3` | H1 dummy for the indicated three-tier reputation group. |
| `submission_id` | Release-specific original-post ID, shared across H2/H3 files. |
| `success` | Binary persuasion success for the focal debate. |
| `success_rate` | Challenger-level persuasion rate used in H1. |
| `super_cluster` | Topic partition recoded as topic_NNN; substantive topic labels are omitted. |
| `tenure_i` | Saved account tenure, in days. |
| `treatment` | Legacy treatment field retained for balance scripts. |
| `value_driven` | Legacy context flag: 1 for ECD quartiles Q1–Q2, 0 for Q3–Q4. |
| `word_count` | Saved focal response length; retained under its source name. |
| `word_count_avg` | Saved average reply length. |
| `word_count_sum` | Saved total words across cleaned direct replies; zero-filled. |

## Per-turn annotation scores

`unified_annotations_scores.jsonl` contains 8,286 records. Each record has `submission_id`, `responder_id`, and ordered `turns`. Each turn contains `A1_count` (deterministic quote-block count), `A1_binary` (at least one block), and binary `scores` for A2 paraphrase, A3 clarification, B1 extension, B2 critique, C1 acknowledgment, C2 hedging, and C3 warmth. No turn text or evidence quotation is included.

The pipeline averages each indicator over turns in a dyad, then sums A1+A2+A3 for representational transactivity, B1+B2 for operational transactivity, and C1+C2+C3 for receptiveness. Thus composite ranges are 0–3, 0–2, and 0–3. The full-frame analysis fills absent annotation scores with zero; `n_op_turns` stays missing and identifies those rows.

## Sample caveats

H1 and H2 have different units; H2 contains repeated challengers. Do not use `responder_id` for challenger-level clustering. The independent baseline balance file has 18,590 rows and no exported participant IDs. H3 original reply counts, post-cleaning annotation counts, and the full zero-filled frame are distinct samples. See KNOWN_DIFFERENCES.md.
