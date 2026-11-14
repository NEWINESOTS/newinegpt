# newineGPT v6 candidate data — explanation add-on (13 November 2026)

**Status: data only. Nothing has been trained. A paid training run needs a separate, explicit approval.**

## What this is

`release/train/data_v6_candidate/` = the unchanged v5 training data + 307 new examples that teach two behaviours the fine-tuned models
lacked in the explanation experiment (they quoted or refused; the base model explained but invented clauses):

| Type | Count | What it teaches |
|---|---|---|
| `explain` | 244 | A 2-3 sentence explanation in own words of a distinction, reason or practical teaching the passage states, each claim cited `[1]` |
| `partial_explain` | 63 | Answer the part the passage supports, then one sentence naming exactly what the passage does not say |

| File | Rows | sha256 (first 16) |
|---|---|---|
| `data_v6_candidate/train.jsonl` | 10,704 (10,397 v5 + 307 new) | `dfec3eff6a532037` |
| `data_v6_candidate/valid.jsonl` | 582 | `53307070e8e051ad` (byte-identical to v5) |
| `data_v6_candidate/test.jsonl` | 582 | `8aa8feed445b4327` (byte-identical to v5) |
| `explanation_v6/add_on.jsonl` | 307 | `4ca633ac24a754d6` |

The v5 training file is an exact byte prefix of the v6 training file. Launch form (when approved): `DATA=v6_build/release/train/data_v6_candidate`.

## How it was built

1. **Sources (Codex).** 600 transcript excerpts with an explicit distinction or reason, mined from the archive; the 90 held-out messages,
   sources used in the explanation and regression tests, duplicate services and disputed/unknown/multi-speaker recordings were excluded.
   Every excerpt is verified against the archive by offset and sha256 at build time.
2. **Drafts (Codex).** The local Qwen3-8B (Q4_K_M, temperature 0, thinking off) drafted a question and a 2-3 sentence answer per excerpt,
   3 in 4 as full explanations and 1 in 4 as partial answers.
3. **First review, sentence by sentence.** Codex AI reviewed 295 drafts; Claude AI subagents reviewed the other 305 under Codex's written
   `REVIEW_INSTRUCTIONS.md`. Every original sentence was labelled S / U / M / L; the reviewer kept, edited or rejected the example.
   Result: 466 accepted (465 after edits), 134 rejected. Of the 8B's 1,201 draft sentences, 713 were supported, 339 mixed, 113 unsupported
   and 36 limits: **38% of the teacher's sentences carried something the excerpt does not say.**
4. **Independent second check of all 466 accepted examples** (Claude AI subagents, blind to the first reviewer's reasoning):
   266 fine, 153 minor, **47 material (10%)**; the rate was the same for Codex's reviews (26/220) and the new ones (21/246).
5. **Drops after the second check: 159.** Material support problem 37; sensitive content (racial, ethnic, political or religious-political
   generalisations, named living people) 35; speaker in doubt (interview answers, testimonies, quoted people, recordings titled after another
   presenter, and every other example from the same recording) 36; minor but a bad habit (contrived "missing detail", padding) 31; minor flaw
   that could not be repaired cleanly or failed its re-check 16; third-round residual check 4.
6. **Repairs: 48.** 64 kept-minor examples with a small addition were sent for minimal repair (delete the added word, link or contrast);
   59 repaired, each re-checked blind by another reviewer; 47 passed the strict line. One more unsupported "rather than" clause was removed
   by a targeted check of all 49 answers using that construction (48 of 49 contrasts are stated in the excerpt).
7. **Assembly (`build/assemble.py`, fail-closed).** Refuses to build unless all 600 drafts have a review, every anchor is an exact excerpt
   substring, every final sentence is S or L, questions are unique and the v5 inputs are unchanged.

All review, second-check, repair and re-check files are in `release/train/explanation_v6/` (`reviews/`, `second_check/`, `rejected.jsonl`,
`sentence_labels.csv`), including the first reviewer's text for every dropped or repaired example.

## What is known to be wrong or unproven

- **AI-only review. No human has checked these examples.** This does not satisfy the human review gate.
- **Residual error.** A third, blind check of a random 40 of the final set found 2 material problems (5%; with 40 examples the true rate
  could plausibly be anywhere from about 1% to 17%) and 14 minor ones. Those 4 worst were dropped; the other 267 examples were not
  re-read in that round, so **roughly 5% of the final set probably still contains an unsupported word, name or link**, and about a third
  has a minor looseness. Each review round found fewer problems but none found zero.
- **Speaker concentration:** 304 of 307 examples are Babs Adewunmi (2 Gloria Norien, 1 Joshua Ortsega), from 214 recordings.
- **Transcript quality:** excerpts are speech-to-text; answers avoid garbled passages but the excerpts shown to the model contain them.
- **Sensitive content was removed from the answers and, where a checker noticed it, from the excerpts; it was not searched for exhaustively.**
- **No held-out examples of the new types.** Validation and test are v5's, so validation loss says nothing about explanation quality.
  The only valid test is the pre-registered protocol (`build/EVALUATION_PROTOCOL.md`): 15 fresh human-written questions, base 4B vs v6,
  blind grading, plus the 109-question regression suite.
- **One inherited v5 training row (`train:7649`) exceeds the trainer's 2,048-token limit by 77 tokens**, so the end of its answer is cut in
  training, exactly as it was for v5. No new example is affected. Remedy for a v6 run: `--maxlen 2304`.
- **Mixture:** the new types are 2.9% of training rows. v5 learned its person behaviour from 433 rows, so the amount is in a workable
  range, but whether 307 examples shift explanatory questions without disturbing v5's strengths is untested.

## What a pass would look like (unchanged from the pre-registered protocol)

On 15 fresh human-written explanation questions, graded blind against the base 4B on the same evidence: explains at least as well as the
base; faithful at least 27/30 and at least 4 points above the base; no invented central definition, cause or attribution; a bounded
partial answer on every deliberately partial question; at most one wrong decline; and no new material error on the 109 regression
questions. If it fails, keep the current app and model.
