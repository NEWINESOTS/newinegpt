"""Evidence ranking of retrieved passages before they reach the model ("which message says this?" questions).

Why: picking the best of ~5 noisy ASR passages for a loose paraphrase is a ranking job. The 4B model does it unreliably
(v3 eval: an equally good passage in 22 of 36 locate answers; admits "nothing clearly matches" in 2 of 11), while
word-overlap evidence does it reliably. So the app ranks, keeps the best 2-3, and the model only quotes and cites.

Scoring is the same as the one that labelled the training data (train/make_sft.py best_excerpt): the 1-2 consecutive
sentences sharing the most content words (stems = first 6 letters, words > 3 letters, no stopwords) with the question.

  from newinegpt.rank import rank_passages
  kept, weak = rank_passages(question, passages)   # passages: [{'head':..., 'text':...}, ...] (any extra keys kept)
  answer, source = finalize_locate(question, kept, weak, model_answer)   # model_answer=None when weak (no model call needed)
"""
import re
from .retrieve import STOP

# words of the question's wrapper ("which message talks about ...") carry no evidence about the content
WRAPPER = set('''message messages talk talks talking teach taught teaching teachings archive remember speaker speakers says said
saying something sermon preach preached preaching find where which there about like it\'s that\'s please tell know think
heard hear recall recording someone somebody'''.split())
LSTOP = set(STOP) | WRAPPER

CONFIDENT = 0.35   # same thresholds as the training labels: >= 0.35 with a lead is a clear match
WEAK = 0.2         # < 0.2: nothing in the passages really says it


def cwords(t):
    return {w[:6] for w in re.findall(r"[a-z']+", t.lower()) if len(w) > 3 and w not in LSTOP}


def best_excerpt(text, query):
    """(score 0-1, excerpt): the 1-2 consecutive sentences of text sharing the most content words with query."""
    want = cwords(query)
    if not want: return 0.0, ''
    sents = [x.strip() for x in re.split(r'(?<=[.!?])\s+', text) if x.strip()]
    best = (0.0, '')
    for i in range(len(sents)):
        for j in (i + 1, i + 2):
            ex = ' '.join(sents[i:j])
            if len(ex) > 320: continue
            sc = len(want & cwords(ex)) / len(want)
            if sc > best[0]: best = (sc, ex)
    return best


def rank_passages(question, passages, keep=3, margin=0.05):
    """Sort passages by evidence for the question. Returns (kept, weak).
    kept: the best passage and its near-ties (within `margin`), at most `keep`, STRONGEST LAST; each gets 'evidence' (0-1)
      and 'excerpt'. Tested on 81 held-out locate questions (eval/locate_rank_test.py): padding with weaker passages drops
      the right-passage rate from 96% to 65%, so they are left out. With one passage alone the 4B sometimes blends the
      question's words into its "quote" (74% verbatim), so always pass the model's answer through finalize_locate().
    weak=True when even the best passage scores < WEAK: show no_match_answer() instead of asking the model
      (kept is then nearest first)."""
    scored = []
    for i, p in enumerate(passages):
        sc, ex = best_excerpt(p.get('text', ''), question)
        scored.append((sc, -i, dict(p, evidence=round(sc, 3), excerpt=ex)))   # ties keep retrieval order
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    if not scored: return [], True
    top = scored[0][0]
    if top < WEAK:
        return [p for _, _, p in scored[:2]], True
    best = [p for sc, _, p in scored if sc >= top - margin][:keep]
    # the 4B model mostly cites the last passage shown (ranked test: the last one in 28 of 41 multi-passage answers)
    return best[::-1], False


def _where(p, n):
    m = re.match(r'(.*) — (.*?), (.*?)(?:, at ([\d:]+))?(?: \(.*\))?$', p['head'])
    if not m: return f"{p['head']} [{n}]", None
    ti, sp, d, t = m.groups(); return f"“{ti}” ({sp}, {d}) [{n}]", t


def _q(ex):
    ex = ex.rstrip(' …'); return f"“{ex}”" + ('' if ex[-1:] in '.!?' else '.')


def no_match_answer(question, kept):
    """Code-made reply for weak=True (the model admits a miss only ~1 in 4 times). Same wording as the training labels.
    kept: passages from rank_passages (weak), nearest first; 'head' is 'Title — speaker, date, at m:ss'."""
    if not kept: return "None of the passages found clearly says that. Try different words, or add a speaker or year."
    p = kept[0]; w, t = _where(p, 1); ex = p.get('excerpt') if p.get('evidence', 0) > 0 else ''
    return (f"None of the passages found clearly says that. The nearest is {w}" + (f" at {t}" if t else '')
            + (f": {_q(ex)}" if ex else '.') + " It may not be the message you remember; try different words or a speaker or year.")


def locate_answer(kept):
    """Code-made reply for a clear match (used when the model's answer fails answer_problems). kept: as shown to the model."""
    n = len(kept); p = kept[-1]; w, t = _where(p, n)
    return f"That is {w}" + (f", from about {t}" if t else '') + f", where the speaker says: {_q(p['excerpt'])}"


def _norm(x): return ' '.join(re.findall(r"[a-z0-9']+", x.lower()))
def _title(p): m = re.match(r'(.*) — ', p.get('head', '')); return _norm(m.group(1) if m else p.get('head', ''))

QUOTED = re.compile(r'“([^”]+)”|"([^"\n]+)"|‘([^’]{12,})’')   # curly, straight double, curly single (straight single = apostrophes, not checked)


def answer_problems(answer, passages, question='', require_quote=False):
    """List of problems (empty = passes). Checks, per line of the answer:
      - the answer cites at least one passage, and only numbers that exist;
      - every quoted span of 2+ words is either the title of a passage CITED ON THE SAME LINE, words from the question,
        or text found verbatim (case/punctuation ignored; '…' may join parts) in a passage CITED ON THE SAME LINE;
      - require_quote: at least one verbatim transcript quote is present (locate answers).
    Not checked: unquoted claims (paraphrase), text in straight single quotes, and whether the passage truly supports the point."""
    probs = []; n = len(passages); nq = 0
    allc = [int(x) for x in re.findall(r'\[(\d+)\]', answer)]
    if not allc: probs.append('no citation')
    if any(not 1 <= c <= n for c in allc): probs.append('citation number does not exist')
    qn = _norm(question)
    for line in answer.split('\n'):
        cs = [int(x) for x in re.findall(r'\[(\d+)\]', line) if 1 <= int(x) <= n]
        for m in QUOTED.finditer(line):
            q = next(g for g in m.groups() if g); qq = _norm(q.rstrip(' …'))
            if len(qq.split()) < 2 or (qn and qq in qn): continue
            if not cs: probs.append(f'quote without a citation on its line: {q[:60]}'); continue
            if any(qq == _title(passages[c - 1]) for c in cs): continue
            parts = [x for x in (_norm(x) for x in re.split(r'…|\.\.\.', q)) if x]
            if any(all(x in _norm(passages[c - 1].get('text', '')) for x in parts) for c in cs): nq += 1
            else: probs.append(f'quote not in the passage it cites: {q[:60]}')
    if require_quote and not nq and not probs: probs.append('no verbatim quote')
    return probs


def finalize_locate(question, kept, weak, model_answer=None):
    """The whole locate decision in one place -> (answer, source) with source 'code:no-match' | 'model' | 'code:fallback'.
    weak: the code-made no-match reply (do not call the model). Otherwise the model's answer is used only if it passes
    answer_problems with a verbatim quote; a refusal, a missing citation or an unverifiable quote gets the code-made answer."""
    if weak: return no_match_answer(question, kept), 'code:no-match'
    if model_answer and not answer_problems(model_answer, kept, question, require_quote=True): return model_answer, 'model'
    return locate_answer(kept), 'code:fallback'
