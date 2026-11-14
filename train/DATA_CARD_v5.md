# newineGPT training data — v5 (12 Nov 2026, not yet trained on)

Built by `train/make_sft.py` (v5) → `train/filter_person_v5.py` → `train/paraphrase_questions.py`, in `train/data_v5/`.
Total 11,561: train 10,397 · valid 582 · test 582. The hold-out defect is fixed: 0 training answers cite one of the 90 held-out messages (`DATA=data_v5 python3 eval/holdout_audit.py`). That does **not** make the test set independent: 142 of 582 test answers (the multi-message types) also cite messages cited in training, duplicate recordings can have twins in training, and overviews draw on all summaries.

**GPT audit of v5 (12 Nov), applied before any training:**
- Person "what did X prophesy?" answers claimed "X's prophecies" when the passages only mention X near prophecy (e.g. Harvard). Now neutral ("X and prophecy") with a caveat that the passages do not always say who gave each prophecy (`train/fix_person_attribution_v5.py`; same in `make_sft.py` and the app's fallback).
- Some question rewrites changed intent (one added "in the latest sermon on sunday morning"). `train/screen_rewrites_v5.py` reverted every rewrite that adds a content word beyond politeness/wrapper words or drops a real word: 537 reverted, 3,016 rewrites kept.
- The launcher used `train/data` (v3). It now requires `DATA=<folder>` and checks sha256 of each file on the pod before training.
- `stats.json` now gives the final counts (and the as-built counts).

What changed from v3, and why (from the GPT audit and the first real questions):
- **Hold-out fixed** (see Leakage control below).
- **Named people and places** (`person`): "Whats the prophecy Pa Elton made?", "Who was …?", "X and <theme>". Passages come from the app's own search (`app/newinegpt/ask.py`), so training matches serving. 133 answers quote lines that name the person; 300 say "mentioned, but not covered". Lower-quality candidates were dropped (68).
- **Natural phrasing**: bare keywords and casual forms added to topic questions; then 3,553 questions rewritten (3,016 kept after screening, about 26%) by the base Qwen3-4B in casual / short / polite / non-native styles. Only question text changes; passages and answers are untouched (checked). A rewrite was kept only if every quote, name and number survived; originals in `_before_paraphrase/` and `q_template`.
- **Locate in the app's ranked layout** (best passage and near-ties only, strongest last, answer always quotes) alongside the old layout.
- **Lists and counts grounded**: the total is now shown in the prompt; "latest message" examples show the latest messages.

| Type | Count |
|---|---|
| phrase | 2,473 |
| locate | 1,999 |
| summary | 1,764 |
| refusal | 1,029 |
| topic | 889 |
| who_when | 882 |
| followup | 700 |
| register | 444 |
| person | 433 |
| scripture | 350 |
| moment | 341 |
| catalog | 182 |
| overview | 75 |

(v3 notes follow.)

## v3 notes (12 Nov 2026)

v3 adds the remaining everyday use cases, all grounded (answers quote or list only what is in the cited passages):
topic teaching (several messages, each quoted), scripture lookup, the moment within one message, two-turn follow-ups (including ones the passages cannot answer),
date ranges / counts / first and latest by speaker, register by year and by what it points to; quote lookup now cites every passage containing the words
(duplicate recordings) and clips passages so the quote is visible; ~15% of questions lowercased without punctuation and ~6% with a typo. v2 data: `train/data_v2/`.

(v2 notes follow.)

v2 change: **locate** answers now quote the passage's own words, and the passage is chosen by evidence (most key-point content words in 1–2 sentences, with a clear lead over the runner-up). When nothing matches well, the answer says so and offers the nearest passage (300 examples). Reason: in v1 the answers restated the question, and the 4B v1 model learned to attach the question to the wrong passage (47% right message on held-out). v1 data is kept in `train/data_v1/`.

Built by `train/make_sft.py` from `pack/newine_archive.sqlite` (no LLM in the loop). Prompt format: `train/prompt.py` (shared with the app).
Rebuild: `~/newine_local/.venv_pack/bin/python train/make_sft.py` → `train/data/{train,valid,test}.jsonl`, `stats.json`.

## What the model is taught
Answer **only from archive passages supplied in the prompt**, cite them as `[n]` with title, speaker, date and time, and say so when the passages do not cover the question. Knowledge lives in the archive pack, not in the weights.

| Type | Count | Question → answer |
|---|---|---|
| phrase | 2,500 | "Where was '…' said?" → every passage that contains the words, with speaker, date, timestamp (25% misremembered quotes) |
| summary | 1,764 | "What is '<title>' about?" → stored summary + key points with times |
| locate | 1,318 | topic phrased from a key point → the passage that says it, quoted (1,018), or "none clearly says that; nearest is …" (300) |
| topic | 1,288 | "What does the archive teach about X?" (some by year or speaker) → 2–4 messages, each with a verbatim quote and time |
| refusal | 1,029 | speakers/years/titles not in the archive, off-topic questions, shuffled non-quotes, the right message not among the passages |
| who_when | 882 | speaker, date preached, series (unknown stays unknown) |
| register | 722 | prophetic-statement candidates by theme, by year, by what they point to; verbatim, in date order |
| followup | 700 | two turns: summary, then who/when, scriptures, the time of a point, or a question the passages can't answer |
| scripture | 551 | "Which messages preach on Isaiah 60?" → from the summaries' scripture lists; a few "none listed" |
| moment | 346 | "Where in '<title>' does the speaker talk about X?" → time + verbatim quote from that message |
| catalog | 344 | by speaker / year / month / series, date ranges, counts, first and latest |
| overview | 75 | series / year overviews |

Total 11,519: train 10,399 · valid 560 · test 560.

## Leakage control (corrected 12 Nov)
**v3 as trained is not a clean hold-out.** 90 messages (~8%, by hash of msg_id) were held out only as the *anchor* of an example. Answers citing several messages were split by one anchor, so 1,076 of 10,399 v3 training answers cite a held-out message (all 90 are touched), and held-out passages and summaries also appear in training prompts. Measured by `eval/holdout_audit.py`.
**Fixed in the builder (v4, `train/data_v4/`):** a training example never shows or cites a held-out message: uncited held-out passages are removed from its prompt, and the example is dropped if an answer cites or names one. Verified on v4 by `DATA=data_v4 python3 eval/holdout_audit.py`: 0 of 9,366 training answers cite a held-out message and 0 training prompts show one (v4: train 9,366 / valid 536 / test 536; 1,083 training examples dropped, 1,188 had held-out passages removed). Still open: (1) duplicate recordings of one service are separate messages, so a held-out recording can have a twin in training (at least 3 of the 90 share an exact title with a training message) — hold out by service next time; (2) test answers of the multi-message types (topic, register, scripture, catalog) also cite training messages, so only summary, who/when, moment, follow-up and most quote-lookup tests are message-disjoint; (3) series/year overviews are written from all summaries, including held-out ones. Keep a separate, untouched set of human-written questions for the final test.

## Known limitations (read before judging the model)
- Transcripts are machine ASR (Whisper variants); quotes are verbatim ASR text, not audio-checked.
- Summaries, key points and register entries were generated by Qwen3-30B-A3B (Phase 4). 24 summaries were replaced after an AI review; **the human Phase 4 gate (≥95% faithful on 100+100 samples) has not been passed yet.** An AI review judged only ~45–50% of sampled register entries clearly prophetic; answers call them "statements", and AI-rejected entries are excluded.
- Speakers: 898/912 known; many come from a voice classifier (Babs vs others) or AI review of spoken introductions, not human confirmation. 14 unknown.
- Answers are template-built, so phrasing is less varied than natural conversation. Multi-turn is limited to two turns.
- v1 is a deliberate "train now, retrain later" release: rebuild data and retrain when the human reviews in `eval/remaining_queue/` come back.

## Training (4B, Lite)
`train/lora_4b.yaml`: LoRA rank 16 (alpha 32), all attention + MLP projections in all layers, full-precision Qwen3-4B base (bf16), lr 1e-4 cosine, 2 epochs, batch 4, max length 2,048, loss on answers only, seed 2026. MLX on Apple M4 Pro.
Export: `train/export_gguf.sh` fuses the adapter and writes GGUF (bf16, Q4_K_M, Q5_K_M) + SHA256SUMS into `dist/models/` — runs on Windows/Linux/Mac via llama.cpp, Ollama, LM Studio. 8B follows on rented GPU with the same data.
