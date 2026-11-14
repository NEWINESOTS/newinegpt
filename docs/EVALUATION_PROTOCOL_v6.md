# Pre-registered v6 decision protocol — before any v6 training

This file fixes the proposed acceptance rules before a paid run. Training is not part of the dataset build.

## Fresh evaluation

Have an archive-knowledgeable person provide 15 fresh explanation questions, including at least 3 deliberately partial and 3 clearly unanswerable cases, without using this training add-on or the prior 13 development questions. Record gold evidence and expected scope before running models. AI-written or AI-graded questions remain exploratory and cannot be called human validation.

Run base 4B and candidate v6 with identical frozen evidence, prompt, decoding settings, context budget and output budget. Then also run their deployed pipelines for end-to-end behaviour. Freeze model hashes, question/evidence hashes, code and first outputs; hide condition identities and randomise A/B independently for each question. Grade identical answers too.

Fifteen answers have a 30-point maximum on EACH of explains, supported, limits and faithful. Compare the base on these SAME fresh questions; old scores of 24/26 and 19/26 are not this baseline.

## Proposed pass rules (all required)

- Explains score at least the paired fresh base score.
- Faithful score at least 27/30 AND at least 4 points above the paired fresh base; also fewer answers with unsupported substantive claims. If the base is already too close to 30 for this margin, the result is inconclusive rather than weakening the rule after seeing outputs.
- Zero invented central definitions, causal explanations or speaker attributions; every deliberately partial case must provide a supported partial answer with an explicit boundary (category B), not invent the missing part.
- At most one wrong decline on answerable/partly answerable questions; zero invented positive answers on deliberately unsupported cases.
- No new material error on a previously correct case in the frozen 109-question regression suite; preserve essential answers, attribution, citations, exact quote fidelity, timestamp support and correct refusals. Byte-identical output is not the criterion, and old incorrect answers are not successes to preserve.
- Record human preference with ties, severity and evidence for every error; report whole-answer quality and latency/RAM, not just sentence deletion or source counts.

The 15 questions constitute a small experiment. Passing is evidence for a candidate, not proof of near-zero hallucination across the archive. If it fails, preserve the current app/model, report the failure, and do not start another paid run without a new decision. No training or paid resources are authorised here.
