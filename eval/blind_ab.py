"""Blind A/B comparison of two models on FRESH user-written questions (eval/golden/questions_final.txt).

  1. Run both models through the same frozen pipeline:
       python3 eval/compare_models.py run v3 <v3.gguf> eval/golden/questions_final.txt
       python3 eval/compare_models.py run v5 <v5.gguf> eval/golden/questions_final.txt
  2. python3 eval/blind_ab.py prepare questions_final v3 v5
       -> eval/blind/questions_final/review.html   (open in a browser: shuffled A/B pairs with the passages each answer was given)
       -> eval/blind/questions_final/grades.csv    (fill in, one row per question)
       -> eval/blind/questions_final/_key.json     (which model is A/B per question: do NOT open until grading is finished)
  3. python3 eval/blind_ab.py score questions_final  -> unblinded totals per model

Grades per answer (A and B), 0 = wrong / 1 = partly / 2 = right:
  correct    - what it says is true to the passages (quotes, dates, speakers)
  relevant   - it answers the question that was asked
  attribution- who said it is right (speaker vs person mentioned); 2 if not applicable
  refusal    - it declines when (and only when) the archive does not cover the question
  preferred  - A, B or same
Identical final answers (e.g. both written by code) are marked IDENTICAL but are graded like any other pair: both can be wrong.
Model names never appear in review.html or grades.csv."""
import csv, html, json, random, sys
from pathlib import Path
R = Path.home() / 'newine_local'; CMP = R / 'eval' / 'compare'; OUT = R / 'eval' / 'blind'
COLS = ['n', 'question', 'identical', 'A_correct', 'A_relevant', 'A_attribution', 'A_refusal', 'B_correct', 'B_relevant', 'B_attribution', 'B_refusal', 'preferred', 'notes']


def load(qset, name):
    rows = [json.loads(l) for l in open(CMP / qset / f'{name}.jsonl')]
    meta = json.load(open(CMP / qset / f'{name}.meta.json'))
    return rows, meta


def prepare(qset, a, b):
    ra, ma = load(qset, a); rb, mb = load(qset, b)
    if ma['code'] != mb['code']: sys.exit('REFUSED: the two runs used different pipeline code; rerun both')
    if [r['question'] for r in ra] != [r['question'] for r in rb]: sys.exit('REFUSED: the two runs answered different questions')
    d = OUT / qset; d.mkdir(parents=True, exist_ok=True)
    if (d / 'grades.csv').exists(): sys.exit(f'REFUSED: {d}/grades.csv exists; move it away first so grades are not overwritten')
    rnd = random.SystemRandom(); key, items = {}, []
    for i, (x, y) in enumerate(zip(ra, rb), 1):
        swap = rnd.random() < 0.5
        A, B = (y, x) if swap else (x, y)
        key[i] = {'A': b if swap else a, 'B': a if swap else b}
        items.append((i, A, B))
    json.dump({'models': [a, b], 'key': key}, open(d / '_key.json', 'w'), indent=1)
    with open(d / 'grades.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(COLS)
        for i, A, B in items: w.writerow([i, A['question'], 'IDENTICAL' if A['answer'] == B['answer'] else ''] + [''] * (len(COLS) - 3))
    e = html.escape
    def passages(r):
        return ''.join(f"<details><summary>[{p['n']}] {e(p['head'])}{' (passage covers ' + e(p['covers']) + ')' if p.get('covers') else ''}</summary><p>{e(p['text'])}</p></details>" for p in r.get('passages', []))
    cards = []
    for i, A, B in items:
        same = A['answer'] == B['answer']
        cards.append(f"""<section><h2>{i}. {e(A['question'])}</h2>{'<p class="same">The two answers are identical. Grade both anyway: both can be wrong.</p>' if same else ''}
<div class="pair"><article><h3>Answer A</h3><pre>{e(A['answer'])}</pre><h4>Passages A was given</h4>{passages(A)}</article>
<article><h3>Answer B</h3><pre>{e(B['answer'])}</pre><h4>Passages B was given</h4>{passages(B)}</article></div></section>""")
    (d / 'review.html').write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind answer review</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--card:#fff;--line:#ddd8cc;--muted:#6b675e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161614;--fg:#ebe8e1;--card:#201f1c;--line:#3a3833;--muted:#a29d92}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px}}
section{{border-top:1px solid var(--line);padding:12px 0}} .pair{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media (max-width:800px){{.pair{{grid-template-columns:1fr}}}} article{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}}
pre{{white-space:pre-wrap;font:inherit}} details{{color:var(--muted);font-size:14px}} .same{{color:var(--muted)}} h2{{font-size:18px}} h3{{margin:0}} h4{{font-size:14px;color:var(--muted)}}
</style></head><body><h1>Blind answer review ({e(qset)})</h1>
<p>Each question shows two answers, A and B, in random order. The model names are hidden. Grade them in <code>grades.csv</code>:
correct, relevant, attribution and refusal each 0 (wrong), 1 (partly) or 2 (right), and preferred = A, B or same. Grade every pair, including identical ones.</p>
{''.join(cards)}</body></html>""")
    print(f'{len(items)} pairs -> {d}/review.html and grades.csv (key in _key.json: do not open before grading)')


def score(qset):
    d = OUT / qset; k = json.load(open(d / '_key.json')); tot = {m: {'correct': 0, 'relevant': 0, 'attribution': 0, 'refusal': 0, 'preferred': 0, 'graded': 0} for m in k['models']}
    for row in csv.DictReader(open(d / 'grades.csv')):
        side = k['key'][row['n']]
        for s in 'AB':
            vals = [row[f'{s}_{c}'] for c in ('correct', 'relevant', 'attribution', 'refusal')]
            if not all(v.strip() for v in vals): continue
            m = side[s]; tot[m]['graded'] += 1
            for c, v in zip(('correct', 'relevant', 'attribution', 'refusal'), vals): tot[m][c] += int(v)
        if row['preferred'].strip().upper() in 'AB' and row['preferred'].strip(): tot[side[row['preferred'].strip().upper()]]['preferred'] += 1
    for m, t in tot.items(): print(m, t)


if __name__ == '__main__':
    {'prepare': lambda: prepare(*sys.argv[2:5]), 'score': lambda: score(sys.argv[2])}[sys.argv[1]]()
