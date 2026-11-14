"""Explanation experiment: can a model EXPLAIN when it is given evidence that is good enough? (separates explanation from retrieval)

13 questions (eval/explain/items.json) whose answer is stated in one archive passage (chosen and read by hand; E11 is partial on purpose).
Every model gets IDENTICAL evidence: that passage plus two related passages from the normal search (distractors), in a fixed order, and the
same prompt: the app's training system prompt + the question + an instruction to explain briefly, cite, and state limits.

  python3 eval/explain_experiment.py build                      -> eval/explain/evidence.jsonl (frozen prompts; refuses to overwrite)
  python3 eval/explain_experiment.py run <name> <gguf>          -> eval/explain/run1/<name>.jsonl (+ .meta.json; refuses to overwrite)
  python3 eval/explain_experiment.py auto                       -> automatic checks per model (citations, gold passage cited, verbatim quotes, limits)
  python3 eval/explain_experiment.py prepare                    -> eval/explain/blind/review.html, grades.csv, _key.json (3 answers per question, shuffled X/Y/Z)
  python3 eval/explain_experiment.py score                      -> unblinded totals (after grading)

Rubric per answer (0 wrong / 1 partly / 2 right): explains (states the requested point directly), supported (each point backed by the passage it
cites; the right passage cited), limits (says what the passages do not cover, or hedges where support is partial), faithful (nothing invented;
quotes verbatim). Plus answerability: A explains the point / B answers part and names what is missing / C related passages only, no claim to
answer / D declines / E claims an answer the passages do not support."""
import csv, hashlib, html, json, os, random, re, subprocess, sys, time, urllib.request
from pathlib import Path
R = Path.home() / 'newine_local'; E = R / 'eval' / 'explain'
sys.path[:0] = [str(R / 'app'), str(R / 'train')]
INSTR = ("Explain the answer in 2-4 sentences in your own words. Cite the passage that supports each point as [n]. "
         "If the passages only partly answer the question, say what they do not cover.")
RUBRIC = ['explains', 'supported', 'limits', 'faithful']
PORT = 8096


def build():
    from newinegpt.ask import Asker, clip
    from prompt import SYSTEM, render_question
    out = E / 'evidence.jsonl'
    if out.exists(): sys.exit(f'REFUSED: {out} exists (frozen evidence)')
    A = Asker(); items = json.load(open(E / 'items.json')); rows = []
    for it in items:
        c = A.A.db.execute('select msg_id, start, end, text from chunks where chunk_id=?', (it['chunk'],)).fetchone()
        t = re.sub(r'\s+', ' ', c['text']); i = t.lower().find(it['anchor'])
        assert i >= 0, it['id']
        j = max(0, t.rfind('. ', 0, max(0, i - 350)) + 2) if i > 350 else 0
        win = t[j:i + 800]; win = win[:win.rfind(' ')] if len(t) > i + 800 else win
        sup = dict(A.P(c['msg_id'], '', A.seg_time(c['msg_id'], c['start'], c['end'], win)), text=win)
        dis = []
        for h in A.A.topic(it['q'], k=10, per_message=1, use_vectors=True):
            if h['msg_id'] == c['msg_id'] or it['anchor'] in h['text'].lower(): continue
            dis.append(dict(A.P(h['msg_id'], '', h['start']), text=clip(h['text'], 650)))
            if len(dis) == 2: break
        pos = random.Random(it['id']).randrange(3)
        ps = dis[:pos] + [sup] + dis[pos:]
        user = render_question(f"{it['q']}\n{INSTR}", [{'head': p['head'], 'text': p['text']} for p in ps])
        rows.append({'id': it['id'], 'question': it['q'], 'key': it['key'], 'support': it['support'], 'gold': pos + 1,
                     'passages': [{'n': k + 1, 'head': p['head'], 'text': p['text']} for k, p in enumerate(ps)],
                     'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]})
    with open(out, 'w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(len(rows), 'items ->', out, '| sha256', hashlib.sha256(out.read_bytes()).hexdigest()[:16])


def run(name, gguf):
    d = E / os.environ.get('EXPLAIN_RUN_DIR', 'run1'); d.mkdir(exist_ok=True); out = d / f'{name}.jsonl'   # v6 (13 Nov): run4_v6, run1 stays untouched
    if out.exists(): sys.exit(f'REFUSED: {out} exists (first-run outputs are preserved)')
    ev = [json.loads(l) for l in open(E / 'evidence.jsonl')]
    try: urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=1); sys.exit(f'REFUSED: something already listens on {PORT}')
    except SystemExit: raise
    except Exception: pass
    srv = subprocess.Popen(['llama-server', '-m', gguf, '--port', str(PORT), '-c', '8192', '--jinja', '-ngl', '99'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(180):
            try: urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2); break
            except Exception: time.sleep(1)
        with open(out, 'w') as f:
            for e in ev:
                body = {'messages': e['messages'], 'temperature': 0, 'max_tokens': 450, 'chat_template_kwargs': {'enable_thinking': False}}
                req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
                ans = json.load(urllib.request.urlopen(req, timeout=600))['choices'][0]['message']['content'].split('</think>')[-1].strip()
                f.write(json.dumps({'id': e['id'], 'answer': ans}, ensure_ascii=False) + '\n'); f.flush()
    finally: srv.terminate()
    json.dump({'model': gguf, 'model_sha256_16': hashlib.sha256(open(gguf, 'rb').read(1 << 26)).hexdigest()[:16],
               'evidence_sha256_16': hashlib.sha256((E / 'evidence.jsonl').read_bytes()).hexdigest()[:16], 'at': time.strftime('%Y-%m-%d %H:%M')},
              open(d / f'{name}.meta.json', 'w'), indent=1)
    print('done', out)


def auto():
    from newinegpt.rank import answer_problems
    ev = {json.loads(l)['id']: json.loads(l) for l in open(E / 'evidence.jsonl')}
    LIM = re.compile(r"do(?:es)? not (?:cover|say|explain|address|define|mention|specify|tell)|don't (?:cover|say|explain|address)|not (?:covered|clear|stated)|"
                     r"only partly|partially|unclear|no(?:t| further) (?:detail|explanation)|beyond (?:this|that)", re.I)
    REF = re.compile(r"isn't covered|not covered by the archive|won't guess|none of the passages", re.I)
    for f in sorted((E / os.environ.get('EXPLAIN_RUN_DIR', 'run1')).glob('*.jsonl')):
        rs = [json.loads(l) for l in open(f)]; c = dict(n=len(rs), refused=0, cites=0, gold=0, quotes_ok=0, limits=0, words=0)
        for r in rs:
            e = ev[r['id']]; a = r['answer']; cited = {int(x) for x in re.findall(r'\[(\d+)\]', a)}
            c['refused'] += bool(REF.search(a)); c['cites'] += bool(cited); c['gold'] += e['gold'] in cited
            c['quotes_ok'] += not answer_problems(a, e['passages'], e['question']); c['limits'] += bool(LIM.search(a)); c['words'] += len(a.split())
        c['words'] //= max(1, c['n'])
        print(f"{f.stem:6} refused {c['refused']}/{c['n']} · cites something {c['cites']}/{c['n']} · cites the supporting passage {c['gold']}/{c['n']} · "
              f"passes citation/quote check {c['quotes_ok']}/{c['n']} · states a limit {c['limits']}/{c['n']} · {c['words']} words on average")


def prepare(runs_dir='run1', out_dir='blind'):
    d = E / out_dir; d.mkdir(exist_ok=True)
    if (d / 'grades.csv').exists(): sys.exit('REFUSED: grades.csv exists')
    ev = [json.loads(l) for l in open(E / 'evidence.jsonl')]
    runs = {f.stem: {json.loads(l)['id']: json.loads(l)['answer'] for l in open(f)} for f in sorted((E / runs_dir).glob('*.jsonl'))}
    names = sorted(runs); rnd = random.SystemRandom(); key = {}; e_ = html.escape; cards = []
    LET = 'XYZ' if len(names) <= 3 else 'WXYZ'[:len(names)]
    cols = ['id', 'question'] + [f'{L}_{k}' for L in LET for k in RUBRIC + ['answerability']] + ['best', 'notes']
    with open(d / 'grades.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(cols)
        for e in ev:
            order = names[:]; rnd.shuffle(order); key[e['id']] = dict(zip(LET, order))
            w.writerow([e['id'], e['question']] + [''] * (len(cols) - 2))
            ans = ''.join(f"<article><h3>Answer {L}</h3><pre>{e_(runs[m][e['id']])}</pre></article>" for L, m in zip(LET, order))
            pas = ''.join(f"<details{' open' if p['n'] == e['gold'] else ''}><summary>[{p['n']}] {e_(p['head'])}</summary><p>{e_(p['text'])}</p></details>" for p in e['passages'])
            cards.append(f"<section><h2>{e['id']}. {e_(e['question'])}</h2><p class=key><b>What a correct answer must say</b> (from passage [{e['gold']}]"
                         f"{', which only partly answers it' if e['support'] == 'partial' else ''}): {e_(e['key'])}</p><h4>Passages given to every model</h4>{pas}"
                         f"<div class=trio>{ans}</div></section>")
    json.dump({'models': names, 'key': key}, open(d / '_key.json', 'w'), indent=1)
    (d / 'review.html').write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Explanation review</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--card:#fff;--line:#ddd8cc;--muted:#6b675e;--key:#f1efe6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161614;--fg:#ebe8e1;--card:#201f1c;--line:#3a3833;--muted:#a29d92;--key:#2a2925}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1300px;padding:16px}}
section{{border-top:1px solid var(--line);padding:12px 0}} .trio{{display:grid;grid-template-columns:repeat({len(LET)},1fr);gap:12px}}
@media (max-width:900px){{.trio{{grid-template-columns:1fr}}}} article{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}}
pre{{white-space:pre-wrap;font:inherit}} details{{color:var(--muted);font-size:14px}} .key{{background:var(--key);padding:8px;border-radius:6px}} h2{{font-size:18px}}
</style></head><body><h1>Explanation review (blind)</h1>
<p>Every model was given the same passages (the supporting one is open). Answers {', '.join(LET)} are shuffled per question; model names are hidden.
In <code>grades.csv</code> grade each answer 0/1/2 for <b>explains</b>, <b>supported</b>, <b>limits</b>, <b>faithful</b>, and give <b>answerability</b>:
A explains the point · B answers part and names what is missing · C related passages only, no claim · D declines · E claims an answer the passages do not support.
<b>best</b> = {', '.join(LET)} or none.</p>{''.join(cards)}</body></html>""")
    print(len(ev), 'questions ->', d / 'review.html')


def score(out_dir='blind'):
    d = E / out_dir; k = json.load(open(d / '_key.json')); tot = {m: dict({c: 0 for c in RUBRIC}, best=0, **{f'cat_{x}': 0 for x in 'ABCDE'}) for m in k['models']}
    for row in csv.DictReader(open(d / 'grades.csv')):
        side = k['key'][row['id']]
        for L, m in side.items():
            for c in RUBRIC:
                if row[f'{L}_{c}'].strip(): tot[m][c] += int(row[f'{L}_{c}'])
            cat = row[f'{L}_answerability'].strip().upper()[:1]
            if cat in 'ABCDE' and cat: tot[m][f'cat_{cat}'] += 1
        b = row['best'].strip().upper()
        if b in side: tot[side[b]]['best'] += 1
    for m, t in tot.items(): print(m, t)


if __name__ == '__main__':
    cmd = sys.argv[1]
    {'build': build, 'auto': auto, 'prepare': lambda: prepare(*sys.argv[2:4]), 'score': lambda: score(*sys.argv[2:3]), 'run': lambda: run(sys.argv[2], sys.argv[3])}[cmd]()
