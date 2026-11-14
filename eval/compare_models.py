"""Compare two GGUF models through the SAME frozen pipeline (GPT audit, 12 Nov).
  python3 eval/compare_models.py run <name> <gguf> <questions.txt>   -> eval/compare/<qset>/<name>.jsonl (+ code hashes)
  python3 eval/compare_models.py report <qset> <nameA> <nameB>      -> side by side
Records, per question: route, the raw model answer, the validator's problems, and whether the final answer came from the model or code,
so model quality is not hidden by code fallbacks. The run refuses to compare if the pipeline code changed between the two runs.
These questions (smoke, Elton, ophanim, locate) shaped development: use them as regression checks only; the final judgement needs fresh
human-written questions (eval/golden/questions_final.txt)."""
import hashlib, json, os, subprocess, sys
from pathlib import Path
R = Path.home() / 'newine_local'; OUT = R / 'eval' / 'compare'
FILES = ['app/newinegpt/ask.py', 'app/newinegpt/rank.py', 'app/newinegpt/retrieve.py', 'app/newinegpt/synonyms.json', 'train/prompt.py']
def code_hash(): return {f: hashlib.sha256((R / f).read_bytes()).hexdigest()[:16] for f in FILES}
if sys.argv[1] == 'run':
    name, gguf, qf = sys.argv[2], str(Path(sys.argv[3]).resolve()), Path(sys.argv[4]).resolve(); d = OUT / qf.stem; d.mkdir(parents=True, exist_ok=True)
    meta = {'model': gguf, 'model_sha256_16': hashlib.sha256(open(gguf, 'rb').read(1 << 26)).hexdigest()[:16], 'code': code_hash(), 'questions': str(qf)}
    json.dump(meta, open(d / f'{name}.meta.json', 'w'), indent=1)
    env = dict(os.environ, NEWINEGPT_MODEL=gguf)
    subprocess.run([str(R / '.venv_pack/bin/python'), '-m', 'newinegpt.ask', '--file', str(qf), '--out', str(d / f'{name}.jsonl')], cwd=R / 'app', env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print('done', d / f'{name}.jsonl')
else:
    qset, a, b = sys.argv[2:5]; d = OUT / qset
    ma, mb = (json.load(open(d / f'{x}.meta.json')) for x in (a, b))
    if ma['code'] != mb['code']: sys.exit(f'pipeline code differs between runs, not comparable:\n{ma["code"]}\n{mb["code"]}')
    ra, rb = ([json.loads(l) for l in open(d / f'{x}.jsonl')] for x in (a, b))
    qs = [l.strip() for l in open(json.load(open(d / f'{a}.meta.json'))['questions']) if l.strip() and not l.startswith('#')]   # the runner skips comments
    for x, rows in ((a, ra), (b, rb)):   # zip() below would silently drop unpaired rows (audit 12 Nov)
        if [r['question'] for r in rows] != qs: sys.exit(f'REFUSED: run {x} is incomplete or out of order ({len(rows)}/{len(qs)} questions)')
    tally = {x: {'model': 0, 'code:fallback': 0, 'model refused': 0} for x in (a, b)}
    for x, rows in ((a, ra), (b, rb)):
        for r in rows:
            if r['source'] == 'model': tally[x]['model'] += 1
            if r['source'] == 'code:fallback': tally[x]['code:fallback'] += 1
            if r.get('model_answer') and any(k in r['model_answer'].lower() for k in ("isn't covered", "can't find", 'not covered', "won't guess")): tally[x]['model refused'] += 1
    print('final answer written by the model / replaced by code / model declined:', tally)
    for x, y in zip(ra, rb):
        print(f"\n## {x['question']}  [{x['kind']}]")
        for n, r in ((a, x), (b, y)):
            print(f"  {n}: {r['source']}" + (f"  problems: {'; '.join(r['problems'])[:150]}" if r['problems'] else ''))
            if r.get('retry_answer'): print('     retry:', r['retry_answer'][:300].replace('\n', ' | '))
            print('     raw:', (r.get('model_answer') or '(no model call)')[:300].replace('\n', ' | '))
