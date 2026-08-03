# Abstention Policy v1: Frozen Held-Out Validation

## Evaluation identity

- Evaluated commit: `d2279adabe004aef928f6d9ec672ecb076948ec2`
- Branch: `feature/retrieval-abstention`
- Policy: `abstention-calibration-v1`
- Policy schema: `1.2.0`
- Split: `answerability-calibration-validation-v1`
- Validation tuning performed: `false`
- Threshold search performed: `false`
- Calibration queries processed: `0`
- Validation queries processed: `14`

This evaluation records the frozen v1 policy result. The validation split was not used to modify the threshold or rules.

## Headline metrics

| Metric | Result |
|---|---:|
| Total queries | 14 |
| Answerable queries | 6 |
| Out-of-corpus queries | 8 |
| Answerable retained | 4/6 (66.7%) |
| Out-of-corpus rejected | 5/8 (62.5%) |
| False abstentions | 2 |
| False acceptances | 3 |
| Answerability accuracy | 64.3% |
| Coverage | 50.0% |
| Accepted-answer precision | 57.1% |
| Answerable Recall@1 | 33.3% |
| Answerable Recall@5 | 83.3% |
| Answerable Recall@10 | 83.3% |
| Answerable MRR@10 | 0.5139 |
| Answerable nDCG@10 | 0.5014 |
| End-to-end accuracy@10 | 64.3% |

## Per-query outcomes

| Query ID | Label | Action | Reason | Correct action | Relevant@10 | Dense top-1 | Overlap@10 |
|---|---|---|---|---:|---:|---:|---:|
| q_exact_003 | answerable | return_results | baseline_evidence_passed | true | true | 0.875853 | 0 |
| q_partial_004 | answerable | abstain | weak_cross_retriever_evidence | false | true | 0.856908 | 0 |
| q_semantic_001 | answerable | abstain | weak_cross_retriever_evidence | false | false | 0.855867 | 0 |
| q_semantic_003 | answerable | return_results | baseline_evidence_passed | true | true | 0.860370 | 1 |
| q_entity_004 | answerable | return_results | baseline_evidence_passed | true | true | 0.889326 | 1 |
| q_orthographic_001 | answerable | return_results | baseline_evidence_passed | true | true | 0.869215 | 0 |
| q_ooc_004 | out_of_corpus | return_results | baseline_evidence_passed | false | false | 0.836680 | 1 |
| q_ooc_candidate_001 | out_of_corpus | abstain | weak_cross_retriever_evidence | true | false | 0.821464 | 0 |
| q_ooc_candidate_005 | out_of_corpus | abstain | known_out_of_corpus_scope | true | false | 0.860508 | 1 |
| q_ooc_candidate_007 | out_of_corpus | abstain | known_out_of_corpus_scope | true | false | 0.871483 | 1 |
| q_ooc_candidate_009 | out_of_corpus | abstain | top_hit_paratext | true | false | 0.825672 | 2 |
| q_ooc_candidate_016 | out_of_corpus | return_results | baseline_evidence_passed | false | false | 0.793881 | 1 |
| q_ooc_candidate_019 | out_of_corpus | return_results | baseline_evidence_passed | false | false | 0.813511 | 1 |
| q_ooc_candidate_020 | out_of_corpus | abstain | known_out_of_corpus_scope | true | false | 0.855208 | 0 |

## Error analysis

### False abstentions

- `q_partial_004` — صید نزدیک و تو دور انداخته Dense top-1: `0.856908`, overlap@10: `0`. The judged relevant passage was present in the top 10.
- `q_semantic_001` — برای شبیه شدن به مردان باید فرمان را انجام داد و راه را پیمود Dense top-1: `0.855867`, overlap@10: `0`. The judged relevant passage was absent from the top 10.

### False acceptances

- `q_ooc_004` — درمان فشار خون بالا با داروهای جدید Dense top-1: `0.836680`, overlap@10: `1`.
- `q_ooc_candidate_016` — مدل هوش مصنوعی چگونه فرمان کاربر را اجرا می‌کند؟ Dense top-1: `0.793881`, overlap@10: `1`.
- `q_ooc_candidate_019` — چگونه یک نامه الکترونیکی رمزگذاری‌شده ارسال کنیم؟ Dense top-1: `0.813511`, overlap@10: `1`.

## Interpretation

The hybrid retriever is useful as a candidate generator: it retrieved a judged relevant passage in the top 10 for five of six answerable queries.

The frozen abstention policy did not generalize as strongly as it did on calibration. It retained four of six answerable queries and rejected five of eight out-of-corpus queries.

The current policy should therefore be treated as an auditable experimental baseline. It is suitable for a demo that returns source passages and explicit decision reasons, but it is not a production-grade trust gate.

The result suggests that binary overlap@10 is too coarse: a single weak overlap allowed several modern-domain queries to pass, while phrase-like evidence could still be rejected.

## Methodological limitations

- The validation set contains only 14 queries.
- The split is frozen but not fully blind; diagnostics were inspected before the split was finalized.
- No policy threshold or rule was changed after observing this validation result.
- A new development set and a separate blind test set are required before developing and evaluating policy v2.

## Recommended next steps

1. Preserve policy v1 and this validation result unchanged.
2. Build the runtime `TrustedRetriever` around the frozen profile and expose explicit action and reason fields.
3. Return multiple sourced passages rather than presenting the top result as a definitive answer.
4. Develop policy v2 only on new development queries, using richer rank-agreement, phrase-evidence, and domain-shift features.
5. Evaluate policy v2 once on a new blind test set.

## Traceability

- Raw report filename: `najm_abstention_validation_v1.json`
- Raw report SHA-256: `153f60b2001f1148ecc2406a87064ccef34c2368cabb121058d6b3d391f02435`
