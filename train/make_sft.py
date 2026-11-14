"""Phase 5 (v1): grounded training examples built straight from the archive pack, no LLM needed.
Each example = system prompt + question + retrieved archive passages (exactly what the app will show the model) -> cited answer.
Types: summary, locate (topic -> message+time), phrase (where was this said), catalog (by speaker/year/series), who_when,
       register (prophetic-statement candidates), overview, refusal (not in archive).
Hold-out: ~8% of messages; any example whose answer rests on a held-out message goes to valid only (no leakage).
Out: train/data/train.jsonl, valid.jsonl, test.jsonl ({"messages": [...], "type": ...}), stats.json
Run: ~/newine_local/.venv_pack/bin/python train/make_sft.py"""
import hashlib, json, os, random, re, sqlite3, sys, collections
from pathlib import Path
R = Path.home() / 'newine_local'; sys.path.insert(0, str(R / 'app'))
from newinegpt.retrieve import Archive, hms
sys.path.insert(0, str(R / 'train')); from prompt import SYSTEM, render_question
from newinegpt.retrieve import STOP as STOP_WORDS
from newinegpt.rank import rank_passages
from newinegpt.ask import Asker
rnd = random.Random(2026)
A = Archive(); db = A.db
OUT = Path(os.environ.get('SFT_OUT', R / 'train' / 'data')); OUT.mkdir(parents=True, exist_ok=True)
msgs = {r['msg_id']: dict(r) for r in db.execute('select * from messages')}
summ = {r['msg_id']: dict(r) for r in db.execute('select * from summaries')}
def held(key): return int(hashlib.md5(key.encode()).hexdigest(), 16) % 100 < 8
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
def fdate(d):
    """2019-06-03 -> 3 June 2019; 2019-06 -> June 2019; 2019 -> 2019; '' -> date unknown"""
    if not d: return 'date unknown'
    p = d.split('-')
    if len(p) == 3: return f"{int(p[2])} {MONTHS[int(p[1]) - 1]} {p[0]}"
    if len(p) == 2: return f"{MONTHS[int(p[1]) - 1]} {p[0]}"
    return p[0]
def meta(mid):
    m = msgs[mid]; return m['title'], m['speakers'] or 'speaker unknown', fdate(m['date_preached'])
def clip(t, n=650):
    t = re.sub(r'\s+', ' ', t or '').strip(); return t if len(t) <= n else t[:n].rsplit(' ', 1)[0] + ' …'
def P(mid, text, t=None, kind=''):
    ti, sp, d = meta(mid); return {'mid': mid, 'head': f"{ti} — {sp}, {d}" + (f", at {hms(t)}" if t is not None else '') + (f" ({kind})" if kind else ''), 'text': clip(text)}
def who(mid):
    ti, sp, d = meta(mid); return f"“{ti}” ({sp}, {d if d != 'date unknown' else 'date unknown'})"
ex = []; DROPPED = collections.Counter(); STRIPPED = collections.Counter()
def is_held(a): return held(a)   # anchors are msg_ids or keys; both are split by the same hash
def noisy(q):
    """Robustness: ~15% of questions lowercased without punctuation, ~6% with one typo (never inside quoted text)."""
    r = rnd.random()
    parts = re.split(r'(“[^”]*”)', q)
    def typo(t):
        ws = t.split(' '); cand = [i for i, w in enumerate(ws) if len(w) > 4 and w.isalpha()]
        if not cand: return t
        i = rnd.choice(cand); w = ws[i]; j = rnd.randrange(1, len(w) - 2); ws[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]; return ' '.join(ws)
    if r < .15: parts = [x if x.startswith('“') else re.sub(r'[?.!,]', '', x.lower()) for x in parts]
    elif r < .21: parts = [x if x.startswith('“') else typo(x) for x in parts]
    return ''.join(parts)
def add(typ, q, passages, answer, anchor, follow=None, keep_order=False):
    """anchor: msg_id (or key) deciding train/valid. follow: [(question2, answer2), ...] -> multi-turn, loss on the last answer only."""
    # Hold-out (fixed 12 Nov after the GPT audit): the split used to look at the anchor only, so answers citing several messages
    # leaked held-out ones into training. Now a TRAINING example (anchor not held) never shows or cites a held-out message:
    # held-out passages the answers do not cite are removed; if an answer cites or names one, the example is dropped.
    if not is_held(anchor):
        base = lambda m_: str(m_).split('@')[0]
        hp = [p_ for p_ in passages if base(p_['mid']) in msgs and held(base(p_['mid']))]
        if hp:
            alltext = answer + ''.join(a2 for _, a2 in follow or [])
            if typ == 'catalog' or any('{' + p_['mid'] + '}' in alltext or msgs[base(p_['mid'])]['title'] in alltext for p_ in hp): DROPPED[typ] += 1; return
            passages = [p_ for p_ in passages if p_ not in hp]; STRIPPED[typ] += 1
            if not passages: DROPPED[typ] += 1; return
    rnd.shuffle(passages) if typ not in ('catalog', 'register', 'scripture') and not keep_order else None
    # answers refer to passages by number: replace {mid} markers with [n]
    num = {p['mid']: i + 1 for i, p in enumerate(passages)}
    fix = lambda a_: re.sub(r'\{([^{}]+)\}', lambda m: f"[{num[m.group(1)]}]" if m.group(1) in num else '', a_).strip()
    msgs_ = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': render_question(noisy(q), passages)}, {'role': 'assistant', 'content': fix(answer)}]
    for q2, a2 in follow or []:
        msgs_ += [{'role': 'user', 'content': noisy(q2)}, {'role': 'assistant', 'content': fix(a2)}]
    ex.append({'type': typ, 'anchor': anchor, 'messages': msgs_})
def distract(mid, k):
    pool = [x for x in summ if x != mid]; return rnd.sample(pool, k)

# 1. summary: "what is <title> about?"
QS = ['What is the message “{t}” about?', 'Summarise “{t}”.', 'Give me the main points of “{t}”.', 'What did {s} teach in “{t}”?',
      'I missed “{t}” — what was it about?', 'Can you outline the message “{t}” with timestamps?']
for mid, s in summ.items():
    if not s['summary'] or mid not in msgs: continue
    ti, sp, d = meta(mid); kps = json.loads(s['key_points'] or '[]')
    body = s['summary'] + ('\nKey points: ' + ' '.join(f"[{hms(k['t'])}] {k['point']}" for k in kps if k.get('t') is not None) if kps else '')
    ctx = [P(mid, body, kind='summary')] + [P(x, summ[x]['summary'], kind='summary') for x in distract(mid, 2)]
    for qt in rnd.sample(QS, 2):
        q = qt.format(t=ti, s=sp if sp != 'speaker unknown' else 'the speaker')
        pts = '\n'.join(f"- {k['point']} (at {hms(k['t'])})" for k in kps if k.get('t') is not None)
        scr = ', '.join(json.loads(s['scriptures'] or '[]')[:8])
        ans = f"{who(mid)} {{{mid}}}: {s['summary']}" + (f"\n\nKey points:\n{pts}" if pts else '') + (f"\n\nScriptures referenced: {scr}." if scr else '')
        add('summary', q, ctx, ans, mid)

# 2. locate: a key point phrased as a topic question -> the passage whose OWN WORDS match it, quoted.
# v1 lesson (4B eval, 47%): answers that restated the question taught the model to assert the question about any passage.
# Now the answer must quote the passage, the choice is made by evidence (any message that says it counts), and weak evidence -> say so.
LSTOP = set(STOP_WORDS)
def cwords(t): return {w[:6] for w in re.findall(r"[a-z']+", t.lower()) if len(w) > 3 and w not in LSTOP}
def best_excerpt(text, kp):
    """1-2 consecutive sentences of the passage that share the most content words with the key point -> (score 0-1, excerpt)."""
    want = cwords(kp)
    if not want: return 0, ''
    sents = [x.strip() for x in re.split(r'(?<=[.!?])\s+', text) if x.strip()]
    best = (0, '')
    for i in range(len(sents)):
        for j in (i + 1, i + 2):
            ex = ' '.join(sents[i:j])
            if len(ex) > 320: continue
            sc = len(want & cwords(ex)) / len(want)
            if sc > best[0]: best = (sc, ex)
    return best
LQ = ['Which message talks about this: {p}', 'Where in the archive is this taught: {p}', 'Find the message where the speaker says something like: {p}',
      'I remember a teaching that {p} Which message was it, and where?']
cand = [(mid, k) for mid, s in summ.items() if mid in msgs for k in json.loads(s['key_points'] or '[]') if k.get('t') is not None and len(k['point']) > 40]
n_conf = n_unsure = n_rank = 0
for mid, k in rnd.sample(cand, min(3400, len(cand))):
    pt = k['point'].rstrip('.')
    hits = A.topic(pt, k=5, per_message=1)
    if len(hits) < 3: continue
    ctx = [P(h['msg_id'], h['text'], h['start']) for h in hits]
    scored = sorted(((best_excerpt(c['text'], pt), h, c) for h, c in zip(hits, ctx)), key=lambda x: -x[0][0])
    (sc, exc), h, c = scored[0]
    q = rnd.choice(LQ).format(p=pt[0].lower() + pt[1:] + ('.' if not pt.endswith('?') else ''))
    if sc >= 0.35 and (len(scored) < 2 or scored[1][0][0] <= sc - 0.05):          # clear winner, quoted
        ans = f"That is {who(h['msg_id'])} {{{h['msg_id']}}}, from about {hms(h['start'])}, where the speaker says: “{exc.rstrip(' …')}”"
        runner = [x for x in scored[1:] if x[0][0] >= 0.3][:1]
        if runner:
            (s2, exc2), h2, _ = runner[0]
            ans += f"\n\nA related passage is in “{msgs[h2['msg_id']]['title']}” at {hms(h2['start'])} {{{h2['msg_id']}}}: “{exc2.rstrip(' …')}”"
        add('locate', q, ctx, ans, h['msg_id']); n_conf += 1
        if rnd.random() < 0.7:   # v5: the layout the app serves (app/newinegpt/rank.py): best passage and near-ties only, strongest last, answer always quotes
            kept, weak = rank_passages(q, [dict(c_) for c_ in ctx])
            if not weak and kept:
                b = kept[-1]
                add('locate', q, [{'mid': x['mid'], 'head': x['head'], 'text': x['text']} for x in kept],
                    f"That is {who(b['mid'])} {{{b['mid']}}}, from about {b['head'].rsplit(', at ', 1)[-1]}, where the speaker says: “{b['excerpt'].rstrip(' …')}”", b['mid'], keep_order=True); n_rank += 1
    elif sc < 0.2 and n_unsure < 300:                                            # nothing really matches: say so, offer the nearest
        ans = (f"None of the passages found clearly says that. The nearest is {who(h['msg_id'])} {{{h['msg_id']}}} at {hms(h['start'])}"
               + (f": “{exc.rstrip(' …')}”" if exc and sc > 0 else '') + ". It may not be the message you remember; try different words or a speaker or year.")
        add('locate', q, ctx, ans, mid); n_unsure += 1
print('locate confident', n_conf, 'uncertain', n_unsure, flush=True)

# 3. phrase: where was this said? Evidence-based: the passage text shown must contain the words; every passage that does is cited.
segs = db.execute("select id, msg_id, start, text from segments where length(text) > 90 order by random() limit 7000").fetchall()
PQ = ['Where did the speaker say “{q}”?', 'Which message has the line “{q}”?', 'Who said “{q}” and when?', 'Find this quote: “{q}”', '“{q}” — where is that from?',
      'Someone quoted “{q}”. Which message is that?', 'In which message do we hear “{q}”?']
NW = lambda x: ' '.join(re.findall(r"[a-z0-9']+", x.lower()))
def clip_around(text, words, n=650):
    """Clip a passage so the quoted words stay visible."""
    t = re.sub(r'\s+', ' ', text or '').strip()
    if len(t) <= n: return t
    i = t.lower().find(words.split(' ')[0]); i = max(0, i - 200) if i >= 0 else 0
    return ('… ' if i else '') + t[i:i + n].rsplit(' ', 1)[0] + ' …'
n = 0
for s in segs:
    if s['msg_id'] not in msgs: continue
    w = re.findall(r"[A-Za-z0-9']+", s['text'])
    if len(w) < 12: continue
    i = rnd.randint(0, len(w) - 10); quote = w[i:i + rnd.randint(9, 14)]
    misq = rnd.random() < .25
    qq = list(quote)
    if misq: del qq[rnd.randrange(1, len(qq) - 1)]
    hits = A.phrase(' '.join(qq), k=4)
    if len({h['msg_id'] for h in hits}) < len(hits): continue          # numbering is per message
    if not any(h['msg_id'] == s['msg_id'] and abs(h['start'] - s['start']) < 31 for h in hits): continue
    full = NW(' '.join(quote))
    ctx = [P(h['msg_id'], '', h['start']) | {'text': clip_around(h['text'], full)} for h in hits]
    has = [h for h, c in zip(hits, ctx) if full in NW(c['text'])]
    tgt = next((h for h in has if h['msg_id'] == s['msg_id']), None)
    if not tgt: continue                                                   # the words must be visible in the target passage
    q = rnd.choice(PQ).format(q=' '.join(qq).lower())
    sp = msgs[s['msg_id']]['speakers'] or 'The speaker (not identified in the archive)'
    ans = (f"The closest wording is in {who(s['msg_id'])} {{{s['msg_id']}}} at {hms(tgt['start'])}: “{' '.join(quote).lower()}”." if misq else
           f"{sp} said it in “{msgs[s['msg_id']]['title']}” ({meta(s['msg_id'])[2]}) {{{s['msg_id']}}}, at {hms(tgt['start'])}.")
    others = [h for h in has if h['msg_id'] != s['msg_id']]
    if others:
        ans += ' The same words also appear in ' + '; '.join(f"{who(h['msg_id'])} {{{h['msg_id']}}} at {hms(h['start'])}" for h in others) + ' — possibly another recording of the same meeting.'
    add('phrase', q, ctx, ans, s['msg_id']); n += 1
    if n >= 2500: break

# 4. catalog: by speaker / year / series / month
def catalog_ex(q, rows, intro, anchor, cap=20):
    rows = sorted(rows, key=lambda r: (r['date_preached'] or '9999', r['title']))
    shown = rows[:cap]
    ctx = [{'mid': 'total', 'head': f"Catalog search: {len(rows)} message{'s' if len(rows) != 1 else ''} found" + (f", the first {cap} by date are listed" if len(rows) > cap else ''), 'text': ''}]   # v5: the total is in the prompt, so the count is grounded
    ctx += [{'mid': r['msg_id'], 'head': f"{r['title']} — {r['speakers'] or 'speaker unknown'}, {fdate(r['date_preached'])}" + (f", series {r['series']}" if r['series'] else ''), 'text': ''} for r in shown]
    lines = '\n'.join(f"- {fdate(r['date_preached'])}: “{r['title']}”" + (f" ({r['speakers']})" if r['speakers'] and 'speaker' not in intro else '') + f" {{{r['msg_id']}}}" for r in shown)
    more = f"\n\nThe archive lists {len(rows)} in total; these are the first {cap} by date." if len(rows) > cap else ''
    add('catalog', q, ctx, f"{intro} ({len(rows)} message{'s' if len(rows) != 1 else ''}):\n{lines}{more}", anchor)
kept = [m for m in msgs.values()]
by = collections.defaultdict(list)
for m in kept:
    for sp in (m['speakers'] or '').split('; '):
        if sp: by[('speaker', sp)].append(m)
    if m['year']: by[('year', m['year'])].append(m)
    if m['series']: by[('series', m['series'])].append(m)
    if m['series'] and m['year']: by[('series_year', m['series'], m['year'])].append(m)
    if len(m['date_preached'] or '') >= 7: by[('month', m['date_preached'][:7])].append(m)
    for sp in (m['speakers'] or '').split('; '):
        if sp and m['year']: by[('speaker_year', sp, m['year'])].append(m)
MN = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
for key, rows in by.items():
    k = key[0]
    if k == 'speaker': q, intro = rnd.choice(['What messages by {0} are in the archive?', 'List what {0} preached.', 'Which sessions did {0} minister in?']).format(key[1]), f"Messages by {key[1]} in the archive"
    elif k == 'year': q, intro = rnd.choice(['What was preached in {0}?', 'Which messages are from {0}?']).format(key[1]), f"Messages preached in {key[1]}"
    elif k == 'series': q, intro = rnd.choice(['What messages are in the {0} series?', 'List the {0} sessions.']).format(key[1]), f"Messages in {key[1]}"
    elif k == 'series_year': q, intro = f"What was preached at {key[1]} in {key[2]}?", f"{key[1]} messages from {key[2]}"
    elif k == 'month':
        y, mo = key[1].split('-'); q, intro = rnd.choice(['What was preached in {0} {1}?', 'Any messages from {0} {1}?']).format(MN[int(mo) - 1], y), f"Messages from {MN[int(mo) - 1]} {y}"
    else: q, intro = f"What did {key[1]} preach in {key[2]}?", f"Messages by {key[1]} in {key[2]}"
    catalog_ex(q, rows, intro, '|'.join(map(str, key)))

# 5. who/when
WQ = ['Who preached “{t}” and when?', 'When was “{t}” preached?', 'Who is the speaker in “{t}”?']
for mid in rnd.sample(list(summ), min(900, len(summ))):
    if mid not in msgs: continue
    m = msgs[mid]; q = rnd.choice(WQ).format(t=m['title'])
    ctx = [{'mid': mid, 'head': f"{m['title']} — {m['speakers'] or 'speaker unknown'}, {fdate(m['date_preached'])}" + (f", series {m['series']}" if m['series'] else ''), 'text': ''}]
    sp = m['speakers'] or ''; d = fdate(m['date_preached']) if m['date_preached'] else ''
    ans = (f"“{m['title']}” was preached by {sp}" if sp else f"The archive does not identify the speaker of “{m['title']}”") + \
          (f" on {d}" if m['date_precision'] == 'day' else f" in {d}" if d else ('; the date is unknown' if sp else ', and the date is unknown')) + \
          (f", in the {m['series']} series" if m['series'] else '') + f". {{{mid}}}"
    if m['speaker_source'] in ('voice_match', 'voice_partial', 'ai_review') and sp and rnd.random() < .5:
        ans += ' (The speaker was identified from the recording, not from the title.)'
    add('who_when', q, ctx, ans, mid)

# 6. register: prophetic-statement candidates by theme / time pointed to
themes = collections.Counter(t.lower() for r in db.execute('select themes from prophetic_register') for t in json.loads(r[0] or '[]'))
RQ = ['What prophetic words were spoken about {0}?', 'Has anything been prophesied about {0}?', 'What declarations about {0} are in the archive?', 'Show me predictions concerning {0}, in date order.']
for th, c in themes.most_common(700):
    if c < 3: continue
    hits = [h for h in A.register(th, k=12) if h.get('ai_review') != 'n'][:6]
    if len(hits) < 2: continue
    one = {}
    for h in hits: one.setdefault(h['msg_id'], h)          # one statement per message, so each [n] holds exactly what is quoted
    hits = sorted(one.values(), key=lambda r: (r['date_preached'] or '9999', r['t']))
    if len(hits) < 2: continue
    ctx = [P(h['msg_id'], h['quote'], h['t'], h['kind']) for h in hits]
    lines = '\n'.join(f"- {fdate(h['date_preached'])}, {h['speaker'] or 'speaker unknown'}, “{msgs[h['msg_id']]['title']}” at {hms(h['t'])} {{{h['msg_id']}}}: “{clip(h['quote'], 260)}”"
                      + (f" (points to: {h['points_to']})" if h['points_to'] else '') for h in hits)
    add('register', rnd.choice(RQ).format(th), ctx, f"These statements in the archive concern {th} (verbatim from the transcripts, in date order):\n{lines}", 'reg|' + th)

# 7. overviews
for r in db.execute('select * from rollups'):
    for qt in ['Give me an overview of {0}.', 'What were the main themes of {0}?', 'Summarise {0}.']:
        name = r['name'] if r['kind'] == 'series' else f"the year {r['name']}"
        ctx = [{'mid': r['key'], 'head': f"Overview of {name} (from the message summaries)", 'text': clip(r['overview'], 1500)}]
        add('overview', qt.format(name), ctx, f"{r['overview']} {{{r['key']}}}", r['key'])

# 8. refusals: the archive does not cover it
NOT_IN = ['T. D. Jakes', 'Joyce Meyer', 'Billy Graham', 'Benny Hinn', 'Joel Osteen', 'David Oyedepo', 'Enoch Adeboye', 'Chris Oyakhilome', 'Creflo Dollar',
          'Bill Johnson', 'Myles Munroe', 'Smith Wigglesworth', 'Derek Prince', 'Kenneth Hagin', 'John Bevere', 'Rick Warren', 'Reinhard Bonnke', 'Paul Enenche']
for sp in NOT_IN:
    if A.catalog(speaker=sp): continue
    for qt in ['What messages by {0} are in the archive?', 'What did {0} preach at NWAS?', 'Did {0} ever minister here?']:
        ctx = [{'mid': 'none', 'head': 'Catalog search: no messages found for this speaker', 'text': ''}]
        add('refusal', qt.format(sp), ctx, f"The archive has no messages by {sp}. It mainly holds messages by Babs Adewunmi and guest ministers at New Wine / SOTS meetings; I can list those instead.", 'ref|' + sp)
for y in list(range(2000, 2013)) + [2014, 2026, 2027]:
    if A.catalog(year=y): continue
    q = rnd.choice(['What was preached in {0}?', 'Show me the {0} messages.']).format(y)
    add('refusal', q, [{'mid': 'none', 'head': f'Catalog search: no messages dated {y}', 'text': ''}],
        f"The archive has no messages dated {y}. Its dated messages run from 2013 to 2025.", f'ref|{y}')
FAKE = ['The Seven Lamps of Zion', 'Dominion Over the Waters of Babylon', 'The Eagle Generation Rising', 'Keys of the Northern Gate', 'The Silver Trumpet Mandate',
        'Prophetic Cartography of Africa', 'The Remnant and the Rainbow', 'Covenant of the Morning Star']
for t in FAKE:
    if A.catalog(title=t): continue
    ctx = [P(h['msg_id'], h['text'], h['start']) for h in A.topic(t, k=3)]
    add('refusal', f'What is the message “{t}” about?', ctx,
        f"I can't find a message titled “{t}” in the archive, and the passages found for those words are from other messages, so I won't guess its content. Check the title, or ask about the topic itself.", 'ref|' + t)
OFF = ['What is the capital of Australia?', 'How do I fix a flat tyre?', 'Who won the 2022 World Cup?', 'Write me Python code to sort a list.',
       'What is the exchange rate of the naira today?', 'What is the best diet for weight loss?', 'Explain quantum entanglement.', 'What time is it in Lagos?',
       'Recommend a good laptop.', 'What did the speaker say about cryptocurrency mining rigs?', 'What is the recipe for jollof rice?', 'Who is the current president of the USA?']
for q in OFF:
    ctx = [P(h['msg_id'], h['text'], h['start']) for h in A.topic(q, k=3)]
    add('refusal', q, ctx, "The archive passages don't address this. I can only answer from the New Wine / SOTS message archive, and nothing there covers this question.", 'ref|' + q)
nonsense = 0
for s in rnd.sample(segs, 400):
    w = re.findall(r"[A-Za-z0-9']+", s['text']);
    if len(w) < 10: continue
    rnd.shuffle(w); q = ' '.join(w[:10]).lower()
    hits = A.phrase(q, k=3)
    if any(h['match'] == 'exact' for h in hits): continue
    ctx = [P(h['msg_id'], h['text'], h['start']) for h in hits] or [{'mid': 'none', 'head': 'Phrase search: no match', 'text': ''}]
    add('refusal', f'Where did the speaker say “{q}”?', ctx, "I can't find that wording in the archive. The nearest passages only share some of the words, so I won't attribute it to any message.", 'ref|n' + q)
    nonsense += 1
    if nonsense >= 150: break

# 8b. more refusal kinds (target 10-15% of all examples)
MORE = ['Kathryn Kuhlman', 'Oral Roberts', 'Charles Spurgeon', 'John Piper', 'Tim Keller', 'Steven Furtick', 'Cindy Jacobs', 'Chuck Pierce', 'Rick Joyner',
        'Bob Jones', 'Kris Vallotton', 'Jentezen Franklin', 'Mike Bickle', 'Lou Engle', 'Matthew Ashimolowo', 'Sam Adeyemi', 'Paul Adefarasin',
        'Femi Emmanuel', 'Mensa Otabil', 'Dag Heward-Mills', 'Joshua Selman', 'Apostle Arome Osayi', 'Jerry Eze', 'Kumuyi', 'Tunde Bakare', 'Mike Murdock',
        'Perry Stone', 'Jonathan Cahn', 'Andrew Wommack', 'Bishop Oyedepo', 'Morris Cerullo', 'Prophet Shepherd Bushiri', 'Uebert Angel']
for sp in MORE:
    if A.catalog(speaker=sp): continue
    for qt in rnd.sample(['What messages by {0} are in the archive?', 'What did {0} preach at NWAS?', 'Did {0} ever minister here?', 'Summarise {0}\'s teaching on prophecy.',
                          'Find {0}\'s message on the glory.'], 3):
        add('refusal', qt.format(sp), [{'mid': 'none', 'head': 'Catalog search: no messages found for this speaker', 'text': ''}],
            f"The archive has no messages by {sp}, so I can't answer from it. I can list messages by the ministers who are in the archive.", 'ref|' + sp)
W1 = ['Seven', 'Silver', 'Hidden', 'Ancient', 'Burning', 'Northern', 'Crimson', 'Twelfth', 'Golden', 'Sealed', 'Eastern', 'Third']
W2 = ['Lamps', 'Gates', 'Trumpets', 'Mountains', 'Rivers', 'Thrones', 'Scrolls', 'Winds', 'Altars', 'Crowns', 'Wells', 'Keys']
W3 = ['of Zion', 'of the Kingdom', 'of the Remnant', 'over Africa', 'of the Last Days', 'of Judah', 'of the Watchmen', 'and the New Dawn']
fakes = set()
while len(fakes) < 160:
    t = f"The {rnd.choice(W1)} {rnd.choice(W2)} {rnd.choice(W3)}"
    if not A.catalog(title=t): fakes.add(t)
for t in sorted(fakes):
    ctx = [P(h['msg_id'], h['text'], h['start']) for h in A.topic(t, k=3, per_message=1)]
    q = rnd.choice(['What is the message “{0}” about?', 'Who preached “{0}”?', 'When was “{0}” preached?', 'Give me the key points of “{0}”.']).format(t)
    add('refusal', q, ctx, f"I can't find a message titled “{t}” in the archive. The passages found for those words come from other messages, so I won't guess. Please check the title, or ask about the topic itself.", 'ref|' + t)
OFF2 = ['How do I bake banana bread?', 'What is the population of Kenya?', 'Translate "good morning" into French.', 'How many planets are in the solar system?',
        'What is the best smartphone in 2025?', 'How do I change my car oil?', 'Who wrote Things Fall Apart?', 'What causes malaria?', 'Explain how a stock market works.',
        'What is the square root of 144?', 'How do I open a bank account?', 'What is the weather forecast for Abuja tomorrow?', 'Give me a workout plan.',
        'Who invented the telephone?', 'What is machine learning?', 'Summarise the latest football transfer news.', 'What are the symptoms of diabetes?',
        'How do I write a CV?', 'What is the tallest building in the world?', 'Recommend a movie for tonight.', 'How do vaccines work?', 'What is inflation?',
        'How far is the moon from the earth?', 'Plan me a holiday to Dubai.', 'What does HTTP stand for?', 'How do I install Windows 11?', 'What is the GDP of Nigeria?',
        'Who painted the Mona Lisa?', 'How do I grow tomatoes?', 'What language is spoken in Brazil?']
for q in OFF2:
    ctx = [P(h['msg_id'], h['text'], h['start']) for h in A.topic(q, k=3, per_message=1)]
    add('refusal', q, ctx, "That isn't covered by the archive. I can only answer from the New Wine / SOTS messages, and none of the passages found addresses this.", 'ref|' + q)
for s in segs[3000:4400]:
    w = re.findall(r"[A-Za-z0-9']+", s['text'])
    if len(w) < 10: continue
    rnd.shuffle(w); q = ' '.join(w[:10]).lower()
    hits = A.phrase(q, k=3)
    if any(h['match'] == 'exact' for h in hits): continue
    ctx = [P(h['msg_id'], h['text'], h['start']) for h in hits] or [{'mid': 'none', 'head': 'Phrase search: no match', 'text': ''}]
    add('refusal', rnd.choice(['Where did the speaker say “{0}”?', 'Which message has “{0}”?']).format(q), ctx,
        "I can't find that wording in the archive. The nearest passages only share some of the words, so I won't attribute it to any message.", 'ref|n' + q)
    nonsense += 1
    if nonsense >= 330: break
# retrieval miss: asked about message X, but X is not among the passages -> say so, don't summarise the wrong one
for mid in rnd.sample([m for m in summ if m in msgs], 320):
    ti = msgs[mid]['title']
    ctx = [P(x, summ[x]['summary'], kind='summary') for x in distract(mid, 3)]
    add('refusal', rnd.choice(['What is the message “{0}” about?', 'Summarise “{0}”.']).format(ti), ctx,
        f"None of the passages found is from “{ti}”, so I can't summarise it from these. The passages are from other messages; try searching for the exact title again.", mid)

# ===================== v3 additions =====================
# 9. topic teaching: several passages, each quoted and cited; only passages that actually contain the topic words are used.
theme_c = collections.Counter(t.lower().strip() for sm in summ.values() for t in json.loads(sm['themes'] or '[]'))
theme_c.update(t.lower().strip() for r in db.execute('select themes from prophetic_register') for t in json.loads(r[0] or '[]'))
TQ = ['What does the archive teach about {t}?', 'What has been preached about {t}?', 'Teach me about {t} from the messages.', 'Give me quotes on {t}.',
      'What do the messages say about {t}?', 'I want to study {t}. Where should I start?',
      '{t}', '{t}', 'teach me about {t}', 'what about {t}', 'anything on {t}?', 'pls what was taught on {t}', 'i need messages on {t}', 'What does the archive say about {t}?']
SPK = [sp for sp, c in collections.Counter(x for m in msgs.values() for x in (m['speakers'] or '').split('; ') if x).items() if c >= 3]
n_topic = 0
for th, c in theme_c.most_common(3000):
    if c < 2 or len(th) < 4: continue
    if n_topic >= 1300: break
    f, extra = {}, ''
    r = rnd.random()
    if r < .2: yr = rnd.choice([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]); f = {'year': yr}; extra = f' in {yr}'
    elif r < .3: sp = rnd.choice(SPK); f = {'speaker': sp}; extra = f' by {sp}'
    hits = A.topic(th, k=6, per_message=1, **f)
    if len(hits) < 3: continue
    ctx = [P(h['msg_id'], h['text'], h['start']) for h in hits]
    good = []
    for h, cx in zip(hits, ctx):
        sc, exc = best_excerpt(cx['text'], th)
        if sc >= 0.6 and exc: good.append((h, exc))
    if len(good) < 2: continue
    good = sorted(good[:4], key=lambda x: (msgs[x[0]['msg_id']]['date_preached'] or '9999'))
    q = rnd.choice(TQ).format(t=th) if not extra else rnd.choice(['What was preached about {t}{e}?', 'What did the messages{e} say about {t}?']).format(t=th, e=extra)
    lines = '\n'.join(f"- {who(h['msg_id'])} {{{h['msg_id']}}}, at {hms(h['start'])}: “{exc.rstrip(' …')}”" for h, exc in good)
    ans = f"Here is what the archive passages say about {th}{extra}:\n{lines}"
    if len(good) < len(hits): ans += f"\n\nThe other passages found mention related ideas but not {th} directly."
    add('topic', q, ctx, ans, 'topic|' + th + extra); n_topic += 1
print('topic', n_topic, flush=True)

# 10. scripture lookup (from the stored summaries' scripture lists)
BOOK = re.compile(r'^\s*((?:[123]\s)?[A-Z][a-z]+(?: of [A-Z][a-z]+)?)\s+(\d+)')
scr = collections.defaultdict(set)
for mid, sm in summ.items():
    if mid not in msgs: continue
    for x in json.loads(sm['scriptures'] or '[]'):
        m = BOOK.match(x or '')
        if m: scr[f'{m.group(1)} {m.group(2)}'].add(mid)
SQ = ['Which messages preach on {r}?', 'Where is {r} used in the messages?', 'Has {r} been taught on?', 'Find messages that reference {r}.']
n_scr = 0
for ref, mids in scr.items():
    if len(mids) < 2: continue
    mids = sorted(mids, key=lambda m: msgs[m]['date_preached'] or '9999')[:8]
    ctx = [{'mid': m, 'head': f"{msgs[m]['title']} — {msgs[m]['speakers'] or 'speaker unknown'}, {fdate(msgs[m]['date_preached'])}",
            'text': 'Scriptures referenced: ' + ', '.join(json.loads(summ[m]['scriptures'] or '[]')[:12])} for m in mids]
    lines = '\n'.join(f"- {who(m)} {{{m}}}" for m in mids)
    add('scripture', rnd.choice(SQ).format(r=ref), ctx, f"These messages reference {ref} (from the message summaries' scripture lists):\n{lines}", 'scr|' + ref); n_scr += 1
for ref in ['Obadiah 3', 'Philemon 2', 'Nahum 3', 'Jude 2', 'Haggai 3', '3 John 2']:
    if ref in scr: continue
    add('scripture', rnd.choice(SQ).format(r=ref), [{'mid': 'none', 'head': 'Scripture search: no message summaries list this reference', 'text': ''}],
        f"No message summary in the archive lists {ref}. It may still be quoted in passing; try a phrase search with words from the verse.", 'scr|' + ref); n_scr += 1
print('scripture', n_scr, flush=True)

# 11. moment within one message: where in "<title>" does the speaker talk about X?
MQ = ['In “{t}”, where does the speaker talk about {p}?', 'At what time in “{t}” is this said: {p}', 'Find the point in “{t}” about {p}']
n_mom = 0
for mid, k in rnd.sample(cand, min(1600, len(cand))):
    pt = k['point'].rstrip('.')
    hits = A.topic(pt, k=4, per_message=4, msg_ids=[mid])
    if len(hits) < 2: continue
    ctx = [P(h['msg_id'], h['text'], h['start']) | {'mid': f"{mid}@{i}"} for i, h in enumerate(hits)]
    sc_ = sorted(((best_excerpt(c['text'], pt), c, h) for c, h in zip(ctx, hits)), key=lambda x: -x[0][0])
    (sc, exc), c, h = sc_[0]
    if sc < 0.4 or (len(sc_) > 1 and sc_[1][0][0] > sc - 0.05): continue
    ti = msgs[mid]['title']
    add('moment', rnd.choice(MQ).format(t=ti, p=pt[0].lower() + pt[1:]), ctx,
        f"In “{ti}” this comes at about {hms(h['start'])} {{{c['mid']}}}, where the speaker says: “{exc.rstrip(' …')}”", mid); n_mom += 1
    if n_mom >= 900: break
print('moment', n_mom, flush=True)

# 12. follow-up questions (two turns; the second answer must come from the same passages)
n_fu = 0
for mid in rnd.sample([m for m in summ if m in msgs and summ[m]['summary']], 700):
    m = msgs[mid]; sm = summ[mid]; ti, sp, d = meta(mid)
    kps = [k for k in json.loads(sm['key_points'] or '[]') if k.get('t') is not None]
    scrs = json.loads(sm['scriptures'] or '[]')[:8]
    body = sm['summary'] + ('\nKey points: ' + ' '.join(f"[{hms(k['t'])}] {k['point']}" for k in kps) if kps else '') + ('\nScriptures: ' + ', '.join(scrs) if scrs else '')
    ctx = [P(mid, body, kind='summary') | {'text': clip(body, 1400)}] + [P(x, summ[x]['summary'], kind='summary') for x in distract(mid, 2)]
    a1 = f"{who(mid)} {{{mid}}}: {sm['summary']}"
    choice = rnd.random()
    if choice < .35:
        q2 = rnd.choice(['Who preached it and when?', 'When was that?', 'Who was the speaker?'])
        a2 = (f"It was preached by {sp}" if m['speakers'] else 'The archive does not identify the speaker') + (f", {d}." if m['date_preached'] else '; the date is unknown.') + f" {{{mid}}}"
    elif choice < .65 and scrs:
        q2 = rnd.choice(['Which scriptures did it use?', 'What Bible passages are referenced?'])
        a2 = f"The summary lists these scriptures: {', '.join(scrs)}. {{{mid}}}"
    elif kps:
        k = rnd.choice(kps)
        q2 = rnd.choice(['At what time does it talk about {p}?', 'Where in the recording is the part about {p}?']).format(p=k['point'][0].lower() + k['point'][1:].rstrip('.'))
        a2 = f"At about {hms(k['t'])} {{{mid}}}: {k['point']}"
    else: continue
    if rnd.random() < .3:   # and a follow-up the passages cannot answer
        q2, a2 = rnd.choice([('What did he say the week after?', "The passages here don't cover what was preached the following week. Ask about that date and I can look it up."),
                             ('Is there a video of it?', "The archive passages only hold transcripts and their details; they don't say whether a video exists.")])
    add('followup', rnd.choice(['What is the message “{t}” about?', 'Summarise “{t}”.']).format(t=ti), ctx, a1, mid, follow=[(q2, a2)]); n_fu += 1
print('followup', n_fu, flush=True)

# 13. more catalog / timeline forms: date ranges, counts, first and latest
n_cat2 = 0
for sp in SPK:
    rows = [m for m in msgs.values() if sp in (m['speakers'] or '').split('; ') and m['date_preached']]
    if len(rows) < 2: continue
    rows.sort(key=lambda r: r['date_preached'])
    for kind in ('first', 'latest', 'count'):
        if kind == 'count':
            catalog_ex(rnd.choice(['How many messages by {0} are in the archive?', 'How many times has {0} preached in the archive?']).format(sp),
                       [m for m in msgs.values() if sp in (m['speakers'] or '').split('; ')], f"The archive has messages by {sp}", f'cnt|{sp}')
        else:
            r = rows[0] if kind == 'first' else rows[-1]
            ctx = [{'mid': x['msg_id'], 'head': f"{x['title']} — {x['speakers']}, {fdate(x['date_preached'])}", 'text': ''} for x in (rows[:20] if kind == 'first' else rows[-20:])]   # v5: the cited message is always shown
            add('catalog', f"What is the {kind} message by {sp} in the archive?", ctx,
                f"The {kind} dated message by {sp} in the archive is {who(r['msg_id'])} {{{r['msg_id']}}}." + (' Some messages have no date, so this covers dated ones only.' if any(not (m['date_preached']) for m in msgs.values() if sp in (m['speakers'] or '')) else ''),
                f'{kind}|{sp}')
        n_cat2 += 1
for y1 in range(2015, 2024):
    y2 = y1 + rnd.choice([1, 2])
    for ser in ('NWAS', 'PSKA', 'Healing Streams', 'Monday SOTS'):
        rows = [m for m in msgs.values() if m['series'] == ser and m['year'] and y1 <= m['year'] <= y2]
        if rows: catalog_ex(f"What {ser} messages were preached between {y1} and {y2}?", rows, f"{ser} messages from {y1} to {y2}", f'rng|{ser}|{y1}'); n_cat2 += 1
print('catalog extra', n_cat2, flush=True)

# 14. register by year / by the time it points to
n_reg2 = 0
for y in range(2016, 2025):
    hits = [h for h in A.register(None, k=40, year=y) if h.get('ai_review') != 'n']
    one = {}
    for h in hits: one.setdefault(h['msg_id'], h)
    hs = rnd.sample(list(one.values()), min(6, len(one)))
    if len(hs) < 2: continue
    hs.sort(key=lambda r: (r['date_preached'] or '9999', r['t']))
    ctx = [P(h['msg_id'], h['quote'], h['t'], h['kind']) for h in hs]
    lines = '\n'.join(f"- {fdate(h['date_preached'])}, “{msgs[h['msg_id']]['title']}” at {hms(h['t'])} {{{h['msg_id']}}}: “{clip(h['quote'], 260)}”" for h in hs)
    add('register', rnd.choice(['What prophetic words were spoken in {0}?', 'Show me declarations from {0}.']).format(y), ctx,
        f"Some statements recorded in {y} (a sample of the archive's candidates, verbatim from the transcripts):\n{lines}", f'regy|{y}'); n_reg2 += 1
for tgt in ['2020', '2021', '2022', '2023', '2024', '2025', 'Nigeria', 'Africa', 'Israel', 'the church', 'the nations', 'Jerusalem', 'America', 'the economy']:
    hits = [h for h in A.register(None, k=200, points_to=tgt) if h.get('ai_review') != 'n']
    one = {}
    for h in hits: one.setdefault(h['msg_id'], h)
    hs = sorted(list(one.values())[:6], key=lambda r: (r['date_preached'] or '9999', r['t']))
    if len(hs) < 2: continue
    ctx = [P(h['msg_id'], h['quote'], h['t'], h['kind']) for h in hs]
    lines = '\n'.join(f"- {fdate(h['date_preached'])}, “{msgs[h['msg_id']]['title']}” at {hms(h['t'])} {{{h['msg_id']}}}: “{clip(h['quote'], 260)}” (points to: {h['points_to']})" for h in hs)
    add('register', rnd.choice(['What was prophesied about {0}?', 'What declarations point to {0}?']).format(tgt), ctx,
        f"Statements in the archive that point to {tgt}, in the order they were spoken:\n{lines}", f'regp|{tgt}'); n_reg2 += 1
print('register extra', n_reg2, flush=True)

# 15. v5: named people and places that are not archive speakers ("Whats the prophecy Pa Elton made?"), built with the APP'S OWN search
# (app/newinegpt/ask.py do_topic: transcript spellings, title-aware, passages trimmed to the sentences about the subject, strongest last),
# so the prompt in training is exactly what the model sees when served. Lesson from the first real questions (12 Nov): the model declined
# natural phrasing and quoted side remarks instead of the line that answers.
ASK = Asker(archive=A)
cap_c, low_c, cap_docs = collections.Counter(), collections.Counter(), collections.defaultdict(set)
for r in db.execute('select msg_id, text from segments'):
    ws_ = re.findall(r"[A-Za-z']+", r['text'])
    for i_, w_ in enumerate(ws_):
        if len(w_) < 5 or not w_.isalpha(): continue
        if w_[0].isupper() and w_[1:].islower() and i_ > 0: cap_c[w_.lower()] += 1; cap_docs[w_.lower()].add(r['msg_id'])
        elif w_.islower(): low_c[w_] += 1
spk_words = {x.lower() for m in msgs.values() for x in re.findall(r"[A-Za-z']+", m['speakers'] or '')}
NAMES = [w_ for w_, c_ in cap_c.items() if c_ >= 15 and len(cap_docs[w_]) >= 6 and low_c[w_] <= c_ * 0.08 and w_ not in spk_words and not w_.endswith('days')]
NAMES += [w_ for w_ in ('elton', 'babalola', 'idahosa', 'branham', 'osborn', 'wagner', 'lindsay', 'wigglesworth', 'bonnke', 'whitefield', 'kuhlman', 'hagin') if w_ not in NAMES and cap_c[w_] >= 4]
T_PRO = ['What did {N} prophesy?', 'Whats the prophecy {N} made?', 'what prophecy did {N} give', 'Did {N} prophesy anything?', 'prophecies of {N}', 'What was prophesied by {N}?']
T_GEN = ['Who was {N}?', 'What does the archive say about {N}?', 'tell me about {N}', '{N}', 'what was said about {N}', 'Who is {N} in the messages?', 'anything about {N}?']
T_TOP = ['What did {N} say about {T}?', '{N} and {T}', 'What does the archive say about {N} and {T}?', 'what was said about {N} and {T}', 'is there a link between {N} and {T}?']
THEMES = [t_ for t_, c_ in theme_c.most_common(400) if 4 <= len(t_) <= 22 and ' ' not in t_.strip()][:200]
def pick_quote(text, hitc, allv, n=300):
    """The 1-2 consecutive pieces (sentences; unpunctuated ASR is cut every ~35 words) that say most about the subject -> verbatim quote."""
    pieces = []
    for sent in [x.strip() for x in re.split(r'(?<=[.!?])\s+', text) if x.strip()]:
        ws_ = sent.split(' ')
        pieces += [sent] if len(sent) <= n else [' '.join(ws_[i_:i_ + 35]) for i_ in range(0, len(ws_), 35)]
    best = (-1, '')
    for i_ in range(len(pieces)):
        for j_ in (i_ + 1, i_ + 2):
            cand_ = ' '.join(pieces[i_:j_])
            if len(cand_) > n + 40: continue
            sc_ = hitc(cand_) * 2 + any(v_ in cand_.lower() for v_ in allv)
            if sc_ > best[0]: best = (sc_, cand_)
    return best[1].strip(' …')
n_person = n_person_no = 0
NAMES = [w_ for w_ in NAMES if not w_.endswith('s')]                            # plurals / peoples ('Indians') are not persons
for name in rnd.sample(NAMES, len(NAMES)):
    N = name.capitalize()
    for fam in ('pro', 'gen', 'top', 'top'):
        if fam == 'top':
            T = rnd.choice(THEMES); q = rnd.choice(T_TOP).format(N=N, T=T); subj = f"{N} and {T}"
        elif fam == 'pro': q = rnd.choice(T_PRO).format(N=N); subj = f"{N} and prophecy"     # not "X's prophecies": naming X near prophecy words does not show X gave them
        else: q = rnd.choice(T_GEN).format(N=N); subj = N
        ASK.last_person = None
        try: ps_, fin_ = ASK.do_topic(q, {})
        except Exception: continue
        lp = ASK.last_person
        if not lp or lp['names'] != [name] or len(ps_) < 2: continue          # only when the app took the named-subject path for exactly this name
        ctx = [{'mid': f"{p_['mid']}@{i_}", 'head': p_['head'], 'text': p_['text']} for i_, p_ in enumerate(ps_)]
        ks = [lp['k'][id(p_)] for p_ in ps_]
        # A passage is used only when the evidence is real: for "who was X" its quote must name X; for prophecy/topic questions the app's
        # 'strong' rule must hold (a message titled after X, or two messages, each using the question's words twice near X) and the quote
        # itself must contain those words. Otherwise the honest label is "mentioned, but not covered" (first build: "What did Dubai prophesy?" got a junk quote).
        strong_ids = {id(p_) for p_ in lp['strong']}
        cand_i = [i_ for i_ in sorted(range(len(ps_)), key=lambda i_: -ks[i_]) if ks[i_] >= 1 and (fam == 'gen' or id(ps_[i_]) in strong_ids)]
        lines, used = [], []
        for i_ in cand_i:
            qt = pick_quote(ps_[i_]['text'], lp['hitc'], lp['allv'])
            titled_ = name in meta(ps_[i_]['mid'])[0].lower()                     # the message is about this person: 'he' is them
            named_q = name in qt.lower()                                          # the exact name (not a fuzzy spelling) is in the quote
            mentions_ = ps_[i_]['text'].lower().count(name)
            if len(qt.split()) < 8 or not named_q: continue                       # fourth pass: the quote itself must name the person (a title match alone let junk through)
            if fam == 'gen' and not (titled_ or mentions_ >= 2): continue          # a passing one-off mention is not "about" them
            if fam != 'gen' and lp['hitc'](qt) < 1: continue
            ti_, sp_, d_ = meta(ps_[i_]['mid']); t_ = ps_[i_]['head'].rsplit(', at ', 1)[-1]
            lines.append(f"- “{ti_}” ({sp_}, {d_}) {{{ctx[i_]['mid']}}}, at {t_}: “{qt}”"); used.append(i_)
            if len(used) >= 3: break
        if used and (fam != 'gen' or len(used) >= 2 or any(name in meta(ps_[i_]['mid'])[0].lower() for i_ in used)):
            ans = f"Here is what the archive passages say about {subj}:\n" + '\n'.join(lines)
            if fam == 'pro': ans += f"\n\nThese passages mention {N} together with prophecy; they do not always say who gave each prophecy, so read each quote before attributing it to {N}."
            elif len(used) < len(ps_) and fam != 'gen': ans += f"\n\nThe other passages mention {N} but not this directly."
            add('person', q, ctx, ans, f'person|{name}', keep_order=True); n_person += 1
        elif fam != 'gen' and n_person_no < 300:
            add('person', q, ctx, f"The passages found mention {N}, but none of them speaks about {subj} directly, so I won't guess. Try different words, or ask what the archive says about {N}.", f'person|{name}', keep_order=True); n_person_no += 1
print('person', n_person, 'person: mentioned but not covered', n_person_no, flush=True)

# split + write
held_ids = {mid for mid in msgs if held(mid)}
tr, va = [], []
for e in ex:
    a = e.pop('anchor'); (va if (a in held_ids or (a not in msgs and held(a))) else tr).append(e)
rnd.shuffle(tr); rnd.shuffle(va)
te, va = va[:len(va) // 2], va[len(va) // 2:]
for name, rows in (('train', tr), ('valid', va), ('test', te)):
    with open(OUT / f'{name}.jsonl', 'w') as f:
        for e in rows: f.write(json.dumps(e, ensure_ascii=False) + '\n')
st = {'train': len(tr), 'valid': len(va), 'test': len(te), 'held_out_messages': len(held_ids), 'by_type': collections.Counter(e['type'] for e in tr + va + te),
      'train_examples_dropped_for_holdout': dict(DROPPED), 'train_examples_with_heldout_passages_removed': dict(STRIPPED)}
json.dump(st, open(OUT / 'stats.json', 'w'), indent=1); print(st)
