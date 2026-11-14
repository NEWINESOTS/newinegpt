"""Audit of the message hold-out (found by the GPT audit, 12 Nov): the split used only each example's anchor message, so training
answers that cite several messages can cite held-out ones. Measures it, and marks which test examples are clean:
a test example is CLEAN when no message its reference answer cites was cited by any training answer.
Usage: python3 eval/holdout_audit.py [generations.jsonl]  (rescoring of a saved run by clean / leaked)"""
import collections, hashlib, json, re, sqlite3, sys
from pathlib import Path
R = Path.home() / 'newine_local'; sys.path.insert(0, str(R / 'train'))
db = sqlite3.connect(R / 'pack' / 'newine_archive.sqlite'); db.row_factory = sqlite3.Row
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
def fdate(d):
    if not d: return 'date unknown'
    p = d.split('-')
    return f"{int(p[2])} {MONTHS[int(p[1]) - 1]} {p[0]}" if len(p) == 3 else f"{MONTHS[int(p[1]) - 1]} {p[0]}" if len(p) == 2 else p[0]
held = lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 100 < 8
key2mid = collections.defaultdict(set)
for r in db.execute('select msg_id, title, speakers, date_preached from messages'):
    key2mid[f"{r['title']} — {r['speakers'] or 'speaker unknown'}, {fdate(r['date_preached'])}"].add(r['msg_id'])
keys = sorted(key2mid, key=len, reverse=True)
def cited(e):
    """msg_ids of the passages the final answer cites by number (all turns' answers for follow-ups)."""
    user = e['messages'][1]['content']; heads = {int(n): h for n, h in re.findall(r'^\[(\d+)\] (.*)$', user, re.M)}
    out = set()
    for m in e['messages'][2::2]:
        for n in re.findall(r'\[(\d+)\]', m['content']):
            h = heads.get(int(n), '')
            for k in keys:
                if h.startswith(k): out |= key2mid[k]; break
    return out
import os
def load(n): return [json.loads(l) for l in open(R / 'train' / os.environ.get('DATA', 'data') / f'{n}.jsonl')]
if __name__ == '__main__':
    tr, te = load('train'), load('test')
    trc = [cited(e) for e in tr]; seen = set().union(*trc)
    leak = collections.Counter(e['type'] for e, c in zip(tr, trc) if any(held(m) for m in c))
    hm = {m for m in seen if held(m)}; allheld = {m for ks in key2mid.values() for m in ks if held(m)}
    print(f'training examples whose answer cites a held-out message: {sum(leak.values())} of {len(tr)}  {dict(leak)}')
    print(f'held-out messages cited in some training answer: {len(hm)} of {len(allheld)}')
    tec = [cited(e) for e in te]; clean = [not (c & seen) for c in tec]
    by = collections.defaultdict(lambda: [0, 0])
    for e, ok in zip(te, clean): by[e['type']][0] += ok; by[e['type']][1] += 1
    print('test examples clean (no cited message seen in a training answer):', sum(clean), 'of', len(te), {t: f'{a}/{b}' for t, (a, b) in sorted(by.items())})
    json.dump({e['messages'][1]['content']: ok for e, ok in zip(te, clean)}, open(R / 'eval' / f"holdout_clean_test_{os.environ.get('DATA', 'data')}.json", 'w'))
