# newineGPT 4B v6 (13-14 Nov 2026) — app default since 14 Nov (user decision; v5 kept as baseline)

Qwen3-4B (snapshot 1cfa9a72) + LoRA r16 on `train/v6_build/release/train/data_v6_candidate` = the v5 data + 307 AI-reviewed examples that
teach explaining in own words with citations and bounded partial answers (`train/v6_build/DATA_CARD_V6_CANDIDATE.md`). 1.25 epochs,
lr 1e-4, maxlen 2304, A100 61 min; val loss 0.2808 (v5 0.2840). Files: Q5_K_M-imat (2.9 GB, recommended) and Q4_K_M-imat (2.5 GB), imatrix
from the same calibration text as v5. `SHA256SUMS` verified after download. Try it: `NEWINEGPT_MODEL=<path to the Q5 file> newinegpt ...`.

## Results (all AI-written or AI-graded: exploratory, not the human validation the protocol requires)
Fresh 20 questions from 74 recordings no model trained on (8 explain, 4 synth, 4 partial, 4 unanswerable), same frozen passages for every
model, blind 3-way grading (`eval/explain_fresh/`):

| out of 20 | base 4B | v5 | v6 |
|---|---|---|---|
| explains (/40) | 35 | 12 | 25 |
| faithful (/40) | 20 | 36 | 37 |
| material inventions | 3 | 1 | 0 |
| wrong declines (of 16 answerable) | 0 | 12 | 4 |
| correct declines (of 4 unanswerable) | 3 | 4 | 4 |
| bounded partial answers (of 4) | 1 | 0 | 1 |
| grader preferred | 11 | 1 | 8 |

109-question regression suite through the app pipeline, blind v5/v6 pairs (`eval/blind/<set>/`): correct 177 vs 174, relevant 175 vs 182,
attribution 205 vs 205, refusal 203 vs 204 (each of 218); preferred v5 16, v6 28. Material errors v5 6, v6 7 (4 shared, 3 of them likely a
review-page artefact: summary scripture lists are not shown to the grader). v6 fixed 2 of v5's (Jangfa quote presented as his teaching;
2019 prophecy list) and introduced 2: the laughter moment in "Healing Streams Decmeber 2019" points to 2:20:24 (holding hands) instead of
1:09:46, and a Gerald Young Shemitah answer paraphrases a garbled passage wrongly (future warning told as past; debt release given to the
wrong group).

## Verdict against the rules fixed before training (`train/v6_build/build/EVALUATION_PROTOCOL.md`)
FAILS: explains below the base (25 vs 35); 1 of 4 partial answers bounded (rule: all); 4 wrong declines (rule: at most 1); 2 new material
errors on the regression suite (rule: none). PASSES: faithful 37/40 and 17 above the base; zero invented answers on unanswerable questions;
zero material inventions on the fresh set. By that rule the app keeps v5 as its default. Against v5 itself v6 is better on the fresh set
on every line except limits, and level on the regression suite.

## Decision (14 Nov 2026)
The user chose v6 as the app default after reading these results ("Switch to v6"), knowing it failed the pre-registered rules: an informed
override, not a pass. Same day, a code check was added to the moment route (a "where does the speaker talk about X" answer must cite a
passage that contains X when another passage does), which turns the laughter regression into a correct code answer. The Gerald Young-type
risk (wrong paraphrase of a garbled transcript) remains; human-written fresh questions are still the real test.
