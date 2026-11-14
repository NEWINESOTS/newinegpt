"""Thin question -> answer pipeline for newineGPT (no UI). Everything the model card says the app must do, in one place:

  route the question -> retrieve from the archive pack -> (rank) -> prompt in the training format -> model (llama-server)
  -> validate the answer (citations exist, every quote is verbatim in the passage it cites) -> code-made fallback if it fails

  ~/newine_local/.venv_pack/bin/python -m newinegpt.ask "Where did the speaker say 'laughter is a weapon'?"     (from app/)
  ... -m newinegpt.ask            interactive
  ... -m newinegpt.ask --json "question"      full record: route, passages, raw model answer, problems, final answer
  ... -m newinegpt.ask --file questions.txt --out answers.jsonl      batch (one question per line), for human test sets

Works on Windows / Linux / Mac: needs python (numpy, sentence-transformers for topic search), llama.cpp's llama-server on PATH
(or NEWINEGPT_SERVER=http://host:port of one already running) and the GGUF (NEWINEGPT_MODEL=path).
The router is rule-based and deliberately simple; `kind` is printed so a tester can see which path answered.
`source` says who wrote the final text: 'model' or 'code:...' (catalog lists, no-match replies and failed-validation fallbacks are code).
"""
import json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / 'train'))
from prompt import SYSTEM, render_question          # the one prompt format shared with training
from .retrieve import Archive, hms, STOP, LLAMA_SERVER
from .rank import rank_passages, finalize_locate, answer_problems, best_excerpt, no_match_answer, _where, _q, _norm, LSTOP, CONFIDENT, QUOTED

# Default v6 since 14 Nov 2026 (user's decision after the v6 evaluation; v6 failed the pre-registered rules, see dist/models/newinegpt-4b-v6/README.md).
# v5 stays the baseline: NEWINEGPT_MODEL=dist/models/newinegpt-4b-v5/newinegpt-4b-v5-Q5_K_M-imat.gguf
MODEL = os.environ.get('NEWINEGPT_MODEL', str(R / 'dist' / 'models' / 'newinegpt-4b-v6' / 'newinegpt-4b-v6-Q5_K_M-imat.gguf'))
SERVER = os.environ.get('NEWINEGPT_SERVER', '')
PORT = 8094
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
BOOKS = ('Genesis Exodus Leviticus Numbers Deuteronomy Joshua Judges Ruth Samuel Kings Chronicles Ezra Nehemiah Esther Job Psalm Psalms Proverbs '
         'Ecclesiastes Isaiah Jeremiah Lamentations Ezekiel Daniel Hosea Joel Amos Obadiah Jonah Micah Nahum Habakkuk Zephaniah Haggai Zechariah Malachi '
         'Matthew Mark Luke John Acts Romans Corinthians Galatians Ephesians Philippians Colossians Thessalonians Timothy Titus Philemon Hebrews James Peter Jude Revelation').split()
SCRIPTURE = re.compile(r'\b((?:[123] )?(?:' + '|'.join(BOOKS) + r'))\s+(\d{1,3})\b', re.I)
QUOTE = re.compile(r'“([^”]+)”|"([^"]+)"|‘([^’]+)’|(?:^|\s)\'([^\']{12,})\'(?=[\s?.!,]|$)')
REGISTER = re.compile(r'prophes|prophec|prophetic (?:word|statement|declaration)|declar|decree|predict', re.I)
# "who said X?", "was it said by A or B?", "who actually gave it?" (attribution: answered by code from metadata + in-passage wording)
ATTRIB = re.compile(r"\bwho (?:actually |really |first |originally )?(?:said|gave|made|spoke|uttered|declared|prophesied|wrote|released)\b|\b(?:said|spoken|declared) by\b"
                    r"|\bwho (?:is|was) (?:the )?(?:source|author)\b|\bwas it \w+ or \w+"
                    r"|\bdid \w+(?: \w+){0,2} or \w+(?: \w+){0,2} (?:say|said|teach|prophesy|preach|declare)\b", re.I)
MULTI_SPK = re.compile(r'\b(?:two|three|several|different|multiple|various|more than one)\s+(?:different\s+)?(?:speakers|preachers|ministers|people)\b', re.I)
LOCATE = re.compile(r'which message|what message (?:talks|says|teaches)|where in the archive|find the message|i remember|where is (?:this|it) taught|which (?:sermon|teaching)', re.I)
CATALOG = re.compile(r'\b(list|how many|which messages|what messages|what was preached|what did .* preach|messages (?:by|from|in)|sessions|first message|latest message|most recent)\b', re.I)
WHO_WHEN = re.compile(r'\b(who (?:preached|is the speaker|spoke|ministered)|when was|what date|who is the speaker)\b', re.I)
MOMENT = re.compile(r'\b(where (?:in|does)|at what time|find the point|what time)\b', re.I)
MSG_SCRIPT = re.compile(r'\b(scriptures?|bible (?:passages?|references?|verses?)|verses?|texts? (?:did|does) (?:it|he|she) use)\b', re.I)


def fdate(d):
    if not d: return 'date unknown'
    p = d.split('-')
    return f"{int(p[2])} {MONTHS[int(p[1]) - 1]} {p[0]}" if len(p) == 3 else f"{MONTHS[int(p[1]) - 1]} {p[0]}" if len(p) == 2 else p[0]


def clip(t, n=650):
    t = re.sub(r'\s+', ' ', t or '').strip(); return t if len(t) <= n else t[:n].rsplit(' ', 1)[0] + ' …'


def clip_around(text, words, n=650):
    t = re.sub(r'\s+', ' ', text or '').strip()
    if len(t) <= n: return t
    i = t.lower().find(words.split(' ')[0]); i = max(0, i - 200) if i >= 0 else 0
    return ('… ' if i else '') + t[i:i + n].rsplit(' ', 1)[0] + ' …'


class Model:
    """llama-server client; starts a local server on first use unless NEWINEGPT_SERVER points at one."""
    def __init__(self, gguf=MODEL, url=SERVER): self.gguf, self.url, self.proc = gguf, url, None

    def _up(self):
        if self.url: return
        self.url = f'http://127.0.0.1:{PORT}'
        try: urllib.request.urlopen(self.url + '/health', timeout=1); up = True      # one is already running
        except Exception: up = False
        if up:   # reuse it only if it serves THIS model (12 Nov: a leftover v5 server silently answered a 'v3' comparison run)
            ids = [m.get('id', '') for m in json.load(urllib.request.urlopen(self.url + '/v1/models', timeout=5)).get('data', [])]
            if any(Path(i).name == Path(self.gguf).name for i in ids): return
            raise RuntimeError(f'port {PORT} already serves {ids}, not {Path(self.gguf).name}; stop that llama-server first')
        # NEWINEGPT_NGL=0 forces CPU only (the offline package's CPU builds ignore GPU layers anyway)
        self.proc = subprocess.Popen([LLAMA_SERVER, '-m', self.gguf, '--port', str(PORT), '-c', '8192', '--jinja', '-ngl', os.environ.get('NEWINEGPT_NGL', '99')],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import atexit; atexit.register(self.close)
        for _ in range(180):
            try: urllib.request.urlopen(self.url + '/health', timeout=2); return
            except Exception: time.sleep(1)
        raise RuntimeError('llama-server did not start; is it on PATH and does the model file exist?')

    def chat(self, user, max_tokens=700):
        self._up()
        body = {'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}], 'temperature': 0, 'max_tokens': max_tokens,
                'chat_template_kwargs': {'enable_thinking': False}}
        req = urllib.request.Request(self.url + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        out = json.load(urllib.request.urlopen(req, timeout=900))['choices'][0]['message']['content']
        return out.split('</think>')[-1].strip()

    def close(self):
        if self.proc: self.proc.terminate(); self.proc = None


class Asker:
    def __init__(self, archive=None, model=None, use_vectors=True):
        self.A = archive or Archive(); self.M = model or Model(); self.vec = use_vectors
        self.msgs = {r['msg_id']: dict(r) for r in self.A.db.execute('select * from messages')}
        self.titles = {}
        for m in self.msgs.values(): self.titles.setdefault(_norm(m['title']), []).append(m['msg_id'])
        # Shorter forms people type (14 Nov): without the ' - Speaker' / ' - Prophet Babs - Fri,3rd May 2013' / '(PSKA 0718)' endings and/or the
        # leading number ('07 Israel's Finance and the Glory' for '07 Israel's Finance and the Glory - Gerald Young'). Kept only when the
        # short form points to one title and keeps 3+ words.
        short = {}
        for m in self.msgs.values():
            t = m['title'] or ''; forms = set()
            while True:
                t2 = re.sub(r'\s*\([^()]*\)\s*$', '', re.sub(r'\s+[-–]\s+[^-–]*$', '', t)).strip()
                if t2 == t or not t2: break
                forms.add(t2); t = t2
            for f in list(forms) + [m['title'] or '']:
                f2 = re.sub(r'^\d+\s*[.)]?\s*', '', f)
                if f2 != f: forms.add(f2)
            for f in forms:
                nf = _norm(f)
                if len(nf.split()) >= 3 and nf not in self.titles: short.setdefault(nf, set()).add(_norm(m['title']))
        self.titles_short = {nf: self.titles[next(iter(ts))] for nf, ts in short.items() if len(ts) == 1}
        self.speakers = sorted({s for m in self.msgs.values() for s in (m['speakers'] or '').split('; ') if s} | set(self.A.alias), key=len, reverse=True)
        yrs = [m['year'] for m in self.msgs.values() if m['year']]; self.years = (min(yrs), max(yrs))
        try: self.A.db.execute("create virtual table temp.vocab using fts5vocab('main','phrase_fts','row')"); self.vocab = True
        except Exception: self.vocab = False
        self.docs = dict(self.A.db.execute('select term, doc from temp.vocab').fetchall()) if self.vocab else {}
        try: self.syn_groups = [[x.lower() for x in g] for g in json.load(open(Path(__file__).with_name('synonyms.json')))['groups']]
        except Exception: self.syn_groups = []
        self.bykey = {}
        for t, d in self.docs.items():
            if t.isalpha() and len(t) >= 4 and d <= 300: self.bykey.setdefault(self.sound(t)[:2], []).append(t)

    @staticmethod
    def sound(w):
        """Rough sound key so ASR / transliteration spellings meet: ph->f, doubled letters single, plural s dropped."""
        w = re.sub(r'[^a-z]', '', w.lower()).replace('ph', 'f').replace('ck', 'k')
        w = re.sub(r'(.)\1+', r'\1', w)
        return w[:-1] if len(w) > 4 and w.endswith('s') else w

    @staticmethod
    def near(a, b):
        """Sound keys within one edit, or one a stem of the other (ofan / ofanim)."""
        if a == b: return True
        if min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)) and abs(len(a) - len(b)) <= 3: return True
        if abs(len(a) - len(b)) > 1: return False
        if len(a) > len(b): a, b = b, a
        for i in range(len(b)):
            if (a[:i] + a[i + 1:] == b[:i] + b[i + 1:] if len(a) == len(b) else a == b[:i] + b[i + 1:]): return True
        return False

    def rare_words(self, q):
        """Unusual words in the question (5+ letters, in at most 30 transcript segments, e.g. 'ophanims') -> the spellings the transcripts
        use for them ('ofanim', 'ophanin', 'ophans'). Common words are never expanded."""
        out = {}
        for w in dict.fromkeys(re.findall(r'[a-z]{5,}', q.lower())):
            if w in STOP or w in self.COMMON or w in ('whats', 'archive', 'archives', 'teach', 'taught', 'teaching', 'preached', 'about', 'speaker', 'which', 'where', 'there') or self.docs.get(w, 0) > 30: continue
            k = self.sound(w)
            if len(k) < 4: continue
            vs = sorted((t for t in self.bykey.get(k[:2], []) if t != w and self.docs[t] <= 60 and self.near(k, self.sound(t))), key=lambda t: -self.docs[t])[:6]
            if vs or self.docs.get(w, 0): out[w] = [w] + vs
        # synonyms supplied by people who know the archive (synonyms.json): 'ophanim' also means 'wheels'
        ql = ' ' + re.sub(r'[^a-z ]', ' ', q.lower()) + ' '
        for g in self.syn_groups:
            key = next((w for w, vs in out.items() if set(vs) & set(g)), None) or next((m for m in sorted(g, key=len, reverse=True) if f' {m} ' in ql), None)
            if key: out[key] = list(dict.fromkeys(out.get(key, [key]) + g))
        return out

    COMMON = set('what whats which who whom when where why how did does do is are was were the a an in on at of and or for to from by about tell give show find list has have had '
                 'i can could please any there this that it he she they we you his her their our my me lord god jesus christ holy spirit bible archive message messages sermon '
                 'program programme meeting conference prophecy prophecies prophetic'.split()) | set(
                 'prophetess prophet pastor apostle rev reverend dr bishop evangelist minister bro brother sis sister mama papa mrs mr'.split())
    TITLES = r'prophetess|prophet|pastor|apostle|rev|reverend|dr|bishop|evangelist|minister|bro|brother|sis|sister|mama|papa|pa|mrs|mr'

    def is_name(self, w):
        """True when the transcripts mostly write this word capitalised (a name: Elton 77%, Wagner 96%), not a common word that only
        starts a sentence or a question (Evangelism 8%, angels 7%, kingdom 6%). Threshold 35%; cached."""
        if not hasattr(self, '_capshare'): self._capshare = {}
        w = w.lower()
        if w not in self._capshare:
            c = self.A.db.execute('select count(*) from chunks where text glob ?', (f'*{w.capitalize()}*',)).fetchone()[0]
            l = self.A.db.execute('select count(*) from chunks where text glob ?', (f'*{w}*',)).fetchone()[0]
            self._capshare[w] = c / (c + l) if c + l else 1.0           # never seen: trust the capital letter
        return self._capshare[w] >= 0.35

    def speaker_by_part(self, q):
        """A speaker named only in part: a title + one of their names ('Prophetess Fanny' -> Fanny Ekpekurede), or a distinctive
        surname alone ('Ortsega'). Only when exactly one speaker matches. Bare first names are not used: 'David', 'Joshua', 'Mercy'
        are Bible names or common words."""
        if not hasattr(self, '_sparts'):
            self._sparts, self._slast = {}, {}
            for c in {v for v in self.A.alias.values()} | {x for m in self.msgs.values() for x in (m['speakers'] or '').split('; ') if x}:
                ts = [t for t in re.findall(r"[a-z']+", c.lower()) if len(t) > 2]
                for t in ts: self._sparts.setdefault(t, set()).add(c)
                if ts: self._slast.setdefault(ts[-1], set()).add(c)
        for m in re.finditer(r'\b(' + self.TITLES + r')\.?\s+([A-Za-z]+)', q, re.I):
            hit = self._sparts.get(m.group(2).lower(), set())
            if len(hit) == 1: return next(iter(hit)), m.group(0)
        for w in re.findall(r"[A-Za-z']{4,}", q):
            hit = self._slast.get(w.lower(), set())
            if len(hit) == 1 and w.lower() not in self.COMMON and w.lower() not in STOP and self.is_name(w): return next(iter(hit)), w
        for w in re.findall(r"[A-Za-z']{4,}", q):   # a distinctive first name alone ('pls find Daps teaching on worship')
            hit = self._sparts.get(w.lower(), set())
            if (len(hit) == 1 and w.lower() not in self.BIBLE_NAMES and w.lower() not in self.COMMON and w.lower() not in STOP
                    and self.is_name(w)): return next(iter(hit)), w
        return None, None
    # first names of speakers that usually mean someone else (Bible figures) or are common words: never a speaker on their own
    BIBLE_NAMES = set('david joshua moses gideon john philip aaron abel jesse sunday mercy miracle victor purity gloria grace joy faith peter paul '
                      'henry major richman sheba hanatu icon'.split())
    def names(self, q):
        """Proper names in the question that are not archive speakers: capitalised words (any position) that are not common words."""
        out = []
        for w in (re.sub(r"'s$", '', x) for x in re.findall(r"\b[A-Z][a-zA-Z']{2,}\b", q)):
            if w.lower() in self.COMMON or w.lower() in STOP or w.lower() in out or not self.is_name(w): continue
            out.append(w.lower())
        return out

    def spellings(self, name):
        """How the transcripts actually spell a name: ASR glues or mangles it ('Pa Elton' -> 'paelton', 'pahelton'). From the index vocabulary."""
        if not self.vocab or len(name) < 4: return []
        rows = self.A.db.execute("select term from temp.vocab where term like ? and term != ? and length(term) <= ? order by doc desc limit 6", ('%' + name, name, len(name) + 4))
        return [r[0] for r in rows]

    def expand(self, q):
        """-> (query with transcript spellings added, names, {name or rare word: spellings})."""
        nm = self.names(q); variants = {n: list(dict.fromkeys([n] + self.spellings(n) + self.rare_words(n).get(n, [])[1:])) for n in nm}
        extra = [v for n in nm for v in variants[n][1:]] + [v for w, vs in self.rare_words(q).items() if w not in nm for v in vs[1:]]
        return (q + ' ' + ' '.join(extra) if extra else q), nm, variants

    def has_name(self, text, variants):
        t = text.lower(); return all(any(v in t for v in vs) for vs in variants.values())

    # ---------- passages in the training format ----------
    def spk(self, mid, unknown='speaker unknown'):
        """Speaker as shown in answers. Recordings in disputed_speakers.csv (a title names another presenter, or the recording plays someone
        else's video) show the listed speaker as unconfirmed until a reviewer fills confirmed_speaker; then that name is used."""
        if not hasattr(self, '_disputed'):
            import csv
            f = Path(__file__).with_name('disputed_speakers.csv'); self._disputed = {}
            if f.exists():
                for r in csv.DictReader(open(f, encoding='utf-8')):
                    if r.get('confirmed_speaker', '').strip(): self._disputed[r['msg_id']] = ('ok', r['confirmed_speaker'].strip())
                    else:
                        why = f"title names {r['title_names']}" if r.get('title_names') else (r.get('why') or 'recording plays video of other speakers')
                        self._disputed[r['msg_id']] = ('?', why)
        listed = self.msgs[mid]['speakers'] or unknown; d = self._disputed.get(mid)
        if not d: return listed
        if d[0] == 'ok': return d[1]
        return f"{listed} [speaker unconfirmed: {d[1].replace(',', ' ')}]"   # no comma: the 'Title — speaker, date' header is parsed on commas

    def head(self, mid, t=None, kind='', series=False):
        m = self.msgs[mid]
        return (f"{m['title']} — {self.spk(mid)}, {fdate(m['date_preached'])}" + (f", at {hms(t)}" if t is not None else '')
                + (f" ({kind})" if kind else '') + (f", series {m['series']}" if series and m['series'] else ''))

    def P(self, mid, text, t=None, kind=''): return {'mid': mid, 'head': self.head(mid, t, kind), 'text': clip(text)}
    def PH(self, h): return dict(self.P(h['msg_id'], h['text'], h['start']), span=(h['start'], h.get('end') or h['start']))   # passage from a search hit

    def rec_passages(self, passages):
        """Passages as recorded with an answer: 'covers' is the time range of the transcript passage, so a quote's own timestamp
        (the segment where the quoted words start) can be checked against it even when it is later than the header time."""
        out = []
        for i, p in enumerate(passages):
            d = {'n': i + 1, 'head': p['head'], 'text': p.get('text', ''), 'program': self.program(p.get('mid'))}
            if p.get('span') and p['span'][1] > p['span'][0]: d['covers'] = f"{hms(p['span'][0])}–{hms(p['span'][1])}"
            out.append(d)
        return out

    def program(self, mid):
        """Series / programme of a message, from the catalog (code, not the model)."""
        m = self.msgs.get(str(mid).split('@')[0]) if mid else None
        if not m: return ''
        pt = re.sub(r'\s*Audios?\s*$', '', m.get('post_title') or '').strip()
        return ' · '.join(x for x in dict.fromkeys([m.get('series') or '', pt]) if x)

    def who(self, mid): m = self.msgs[mid]; return f"“{m['title']}” ({self.spk(mid)}, {fdate(m['date_preached'])})"

    # ---------- routing ----------
    def filters(self, q):
        f = {}; rest = q
        y = re.search(r'\b((?:19|20)\d{2})\b', q)
        if y: f['year'] = int(y.group(1)); rest = rest.replace(y.group(0), ' ')
        for sp in self.speakers:
            if len(sp) > 3 and re.search(r'\b' + re.escape(sp) + r'\b', q, re.I):
                f['speaker'] = self.A.alias.get(sp.lower(), sp); rest = re.sub(re.escape(sp), ' ', rest, flags=re.I); break
        if 'speaker' not in f:
            sp, said = self.speaker_by_part(q)
            if sp:
                f['speaker'] = sp; rest = re.sub(re.escape(said), ' ', rest, flags=re.I)
                for t in re.findall(r"[A-Za-z']+", sp) + ['gen', 'general', 'pst', 'rabbi', 'rabi']:
                    rest = re.sub(r'\b' + re.escape(t) + r'\b', ' ', rest, flags=re.I)
                rest = re.sub(r'\b(' + self.TITLES + r')\b\.?', ' ', rest, flags=re.I)
        return f, rest

    def find_title(self, q, quoted):
        """A message title named in the question: an exact quoted title, or the whole title appearing in the question (3+ words)."""
        for s in quoted:
            if _norm(s) in self.titles: return self.titles[_norm(s)][0], s
        for s in quoted:
            if _norm(s) in self.titles_short: return self.titles_short[_norm(s)][0], s
        nq = ' ' + _norm(q) + ' '; best = None
        for nt, mids in self.titles.items():
            if len(nt.split()) >= 3 and f' {nt} ' in nq and (best is None or len(nt) > len(best[1])): best = (mids[0], nt)
        if best is None:      # an unquoted short form must be longer (4+ words) to count
            for nt, mids in self.titles_short.items():
                if len(nt.split()) >= 4 and f' {nt} ' in nq and (best is None or len(nt) > len(best[1])): best = (mids[0], nt)
        return best or (None, None)

    def title_hint(self, q):
        """The question's subject is (part of) a message title: all its content words are in one title, and at least one of them is
        distinctive (in 3 titles or fewer). 'the mimshack anointing' -> “The Mimshack Anointing (Monday 18th Aprill 2022)”."""
        if not hasattr(self, 'tw'):
            self.tw = {mid: set(re.findall(r'[a-z]{3,}', m['title'].lower())) for mid, m in self.msgs.items()}
            self.tdf = {}
            for ws in self.tw.values():
                for w in ws: self.tdf[w] = self.tdf.get(w, 0) + 1
        skip = STOP | self.COMMON | {'teach', 'teaching', 'taught', 'preached', 'preach', 'study', 'explain', 'summarise', 'summarize', 'summary', 'about', 'want', 'know', 'more', 'main', 'points',
                                         'bible', 'passages', 'passage', 'referenced', 'references', 'scripture', 'scriptures', 'verses', 'uses', 'used'}
        ws = [w for w in re.findall(r'[a-z]{3,}', q.lower()) if w not in skip]
        if len(ws) < 2 or len(ws) > 6 or not any(self.tdf.get(w, 99) <= 3 for w in ws): return None
        mids = [mid for mid, tws in self.tw.items() if all(w in tws for w in ws)]
        return mids[0] if 1 <= len(mids) <= 2 else None

    def route(self, q):
        quoted = [next(g for g in m.groups() if g).strip() for m in QUOTE.finditer(q)]
        mid, tstr = self.find_title(q, quoted)
        if mid:
            rest = re.sub(re.escape(tstr), ' ', q, flags=re.I)
            if WHO_WHEN.search(q): return 'who_when', {'mid': mid}
            if MOMENT.search(q): return 'moment', {'mid': mid, 'topic': re.sub(r'[“”"‘’]', ' ', MOMENT.sub(' ', rest)).strip(' ?.,:')}
            if MSG_SCRIPT.search(rest): return 'msg_scriptures', {'mid': mid}
            return 'summary', {'mid': mid}
        if quoted and re.search(r'\b(titled|called)\b|\b(message|sermon|teaching)\s*[“"‘\']|\babout\s*\??\s*$', q, re.I): return 'unknown_title', {'title': quoted[0]}
        if not quoted:
            mid = self.title_hint(self.filters(q)[1])   # speaker words are not title words ("Sunday Awugo on the Holy Spirit")
            if mid and MSG_SCRIPT.search(q): return 'msg_scriptures', {'mid': mid}
            if mid: return ('who_when' if WHO_WHEN.search(q) else 'summary'), {'mid': mid}
        long_q = [s for s in quoted if len(s.split()) >= 3]
        if long_q: return 'phrase', {'quote': max(long_q, key=len)}
        s = SCRIPTURE.search(q)
        if s: return 'scripture', {'ref': f'{s.group(1).title()} {s.group(2)}'}
        f, rest = self.filters(q)
        content = [w for w in re.findall(r"[a-z']+", rest.lower()) if len(w) > 3 and w not in STOP and not CATALOG.fullmatch(w)
                   and w not in ('messages', 'message', 'preached', 'preach', 'archive', 'list', 'many', 'sessions', 'minister', 'ministered', 'show', 'give', 'there', 'were', 'have', 'what', 'which')]
        amb = self.ambiguous_speaker(q, f)
        if amb: return 'clarify', amb
        if ATTRIB.search(q): return 'attribution', {'f': f}
        if MULTI_SPK.search(q): return 'speakers', {'f': f}
        if REGISTER.search(q): return 'register', {'f': f, 'theme': ' '.join(w for w in content if not REGISTER.search(w) and w not in ('words', 'word', 'spoken', 'anything', 'about', 'concerning'))}
        if f and (CATALOG.search(q) or not content) and len(content) <= 1: return 'catalog', {'f': f}
        if LOCATE.search(q): return 'locate', {'f': f}
        return 'topic', {'f': f}

    # ---------- handlers: each returns (passages, question_for_model or None, code_answer or None) ----------
    def answer(self, q):
        kind, a = self.route(q); rec = {'question': q, 'kind': kind, 'route': {k: v for k, v in a.items()}, 'model_answer': None, 'problems': []}
        fn = getattr(self, 'do_' + kind); passages, code = fn(q, **a)
        rec['kind_used'] = 'topic' if kind == 'register' and code is not None and len(code) > 3 else kind
        rec['passages'] = self.rec_passages(passages)
        if code is not None and code[1].startswith('code'): rec['answer'], rec['source'] = code; return self.add_programme(rec)
        mq = code[3] if code is not None and len(code) > 3 else q
        if mq != q: rec['question_for_model'] = mq
        out = self.M.chat(render_question(mq, passages)); rec['model_answer'] = out
        if code is not None:                                   # handler supplied its own finalizer
            self.last_retry = None
            r = code[0](out); rec['answer'], rec['source'] = r[0], r[1]; rec['problems'] = code[2](out)
            if self.last_retry is not None: rec['retry_answer'] = self.last_retry; rec['retry_source'] = 'rejected'
            if len(r) > 2:                                     # retried with fewer passages: citations now number those
                rec['first_passages'], rec['retry_answer'] = rec['passages'], r[3]
                rec['passages'] = self.rec_passages(r[2])
            return self.add_programme(rec)
        probs = answer_problems(out, passages, q) if not self.refused(out) else []
        rec['problems'] = probs
        if probs: rec['answer'], rec['source'] = self.listing(passages), 'code:fallback'
        else: rec['answer'], rec['source'] = out, 'model'
        return self.add_programme(rec)

    REFUSAL = re.compile(r"can't find|cannot find|no messages|isn't covered|not covered|don't address|doesn't cover|does not cover|none of the passages|has no messages|won't guess|won't attribute|no message summary", re.I)
    WHICH_PROG = re.compile(r'\b(which|what) (program|programme|conference|meeting|series|event|school)\b|\bwhere was (it|this|that)\b', re.I)
    EXPLAIN = re.compile(r"\bdifference between\b|\bhow (?:do|can|should|would) (?:i|we|you|one)\b|\bhow to\b|\bwhat (?:to do|should)\b|\bwhy\b"
                         r"|\btogether\b|\bexplain|\bwhen (?:you|i|we)\b|\bwithout\b", re.I)
    EXPL_DROP = set("""difference between know whether together same idea both someone something somebody thing things message messages teaching
        give giving make making doing does like way ways step steps find seem seems seemed speaker speakers different short quote quotes each
        show please archive""".split())
    KEEP_COMMON = {'prophecy', 'prophecies', 'prophetic'}          # too common to be names, but real topic words

    def coverage_note(self, rec):
        """'What is the difference between X and Y?', 'how do I ...': a list of quotes that each use one of the words is not an answer.
        Code cannot judge meaning, but it can state a fact: if no cited passage brings the question's key words together, say so
        (blind batch 1: both models quoted accurately without answering the question in about half the pairs)."""
        if rec.get('kind_used', rec['kind']) not in ('topic', 'register') or rec['source'].startswith('code:no') or rec['source'] in ('code:clarify', 'code:attribution', 'code:catalog'): return rec
        q = rec['question']
        if not self.EXPLAIN.search(q) or self.refused(rec['answer']): return rec
        sq = self.filters(q)[1]
        hon = set(re.findall(r'[a-z]+', self.TITLES))
        words = [w for w in dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z']+", sq)) if len(w) >= 4 and w.lower() not in LSTOP and w.lower() not in self.FILLER
                 and w.lower() not in self.ASKWORDS and (w.lower() not in self.COMMON or w.lower() in self.KEEP_COMMON)
                 and w.lower() not in self.EXPL_DROP and w.lower() not in hon]
        stems = {w: self.stem(w) for w in words}
        if len(set(stems.values())) < 2: return rec
        rx = {s_: re.compile(r'\b' + (s_ if '[' in s_ else re.escape(s_))) for s_ in set(stems.values())}
        cited = sorted({int(x) for x in re.findall(r'\[(\d+)\]', rec['answer']) if 1 <= int(x) <= len(rec['passages'])})
        if not cited: return rec
        def cov(n):
            w = self.best_window(rec['passages'][n - 1]['text'], rx, 400).lower(); return {s_ for s_, r in rx.items() if r.search(w)}
        best = max((cov(n) for n in cited), key=len)
        keys = list(dict.fromkeys(w.lower() for w in words))
        if len(best) >= max(2, -(-len(rx) * 2 // 3)):   # words appear together; that still does not show the passage explains the point
            rec['answer'] += ("\n\nNote: these passages use the words of your question, but the app has not checked that they explain it; "
                              "read them as related passages, not as an explanation.")
        else:
            rec['answer'] += ("\n\nNote: no single passage found brings together " + ', '.join(f"“{k}”" for k in keys[:-1]) + f" and “{keys[-1]}”, "
                              "so these are related passages and may not answer the question directly.")
        rec['coverage_note'] = True
        return rec

    NOT_DIRECTLY = re.compile(r"\s*The other passages found mention related ideas but not (.+?) directly\.")
    def drop_false_limit(self, rec):
        """The model (trained on 794 such lines) often ends with 'The other passages found mention related ideas but not X directly.' when an
        uncited passage DOES mention X (graders, 14 Nov: false in most cases). Code checks it: if any passage the answer does not cite contains
        X's words (names and titles aside), or no uncited passage exists, the sentence is removed. 'this' / 'this distinction' cannot be
        checked and stays."""
        if not rec.get('source', '').startswith('model'): return rec
        m = self.NOT_DIRECTLY.search(rec['answer'])
        if not m: return rec
        if not hasattr(self, '_name_tok'):
            self._name_tok = {t for a in self.A.alias for t in re.findall(r'[a-z]+', a.lower())} | set(re.findall(r'[a-z]+', self.TITLES))
        words = [w for w in re.findall(r"[a-z']+", m.group(1).lower()) if len(w) >= 4 and w not in LSTOP and w not in self.FILLER and w not in self._name_tok]
        if re.match(r'(?i)this\b', m.group(1).strip()) or (not words and m.group(1).strip()): return rec
        cited = {int(x) for x in re.findall(r'\[(\d+)\]', rec['answer'])}
        rest = [p for k, p in enumerate(rec['passages'], 1) if k not in cited and p.get('text')]
        if not rest or any(all(re.search(self.stem(w), p['text'].lower()) for w in words) for p in rest):
            rec['answer'] = (rec['answer'][:m.start()] + rec['answer'][m.end():]).strip()
            rec['answer'] = re.sub(r'\n{3,}', '\n\n', rec['answer']); rec['limit_removed'] = m.group(0).strip()
        return rec

    def add_programme(self, rec):
        rec = self.drop_false_limit(rec)
        rec = self.coverage_note(rec)
        """'...and in which program?': the programme of each cited message, from the catalog (code), appended to the answer."""
        if not self.WHICH_PROG.search(rec['question']): return rec
        cited = sorted({int(x) for x in re.findall(r'\[(\d+)\]', rec['answer'])}); seen = {}
        for n in cited:
            if 1 <= n <= len(rec['passages']) and rec['passages'][n - 1].get('program'): seen.setdefault(rec['passages'][n - 1]['program'], []).append(n)
        if seen: rec['answer'] += '\n\nProgramme (from the catalog): ' + '; '.join(f"{pr} " + ''.join(f'[{n}]' for n in ns) for pr, ns in seen.items()) + '.'
        return rec

    def refused(self, out): return bool(self.REFUSAL.search(out)) and not re.search(r'“[^”]{40,}”', out)

    def listing(self, passages):
        """Fallback when the model's answer cannot be verified: show what was found, in code, with nothing asserted."""
        real = [(i + 1, p) for i, p in enumerate(passages) if p.get('mid') != 'none']
        if not real: return "The archive search found nothing for that."
        lines = []
        for n, p in real[:5]:
            w, t = _where(p, n); ex = clip(p.get('text', ''), 220)
            lines.append(f"- {w}" + (f" at {t}" if t else '') + (f": “{ex}”" if ex else ''))
        return "I could not produce a verified answer, so here are the passages the archive search found:\n" + '\n'.join(lines)

    def do_phrase(self, q, quote):
        hits = self.A.phrase(quote, k=4); seen = set(); hits = [h for h in hits if not (h['msg_id'] in seen or seen.add(h['msg_id']))]
        full = _norm(quote)
        ps = [dict(self.P(h['msg_id'], '', h['start']), text=clip_around(h['text'], full)) for h in hits] or [{'mid': 'none', 'head': 'Phrase search: no match', 'text': ''}]
        return ps, None

    def do_unknown_title(self, q, title):
        ps = [self.PH(h) for h in self.A.topic(title, k=3, use_vectors=self.vec)]
        return ps, None

    def do_summary(self, q, mid):
        s = self.A.summary(mid)
        if not s or not s.get('summary'): return [dict(mid=mid, head=self.head(mid, series=True), text='')], None
        kps = [k for k in s['key_points'] if k.get('t') is not None]
        body = s['summary'] + ('\nKey points: ' + ' '.join(f"[{hms(k['t'])}] {k['point']}" for k in kps) if kps else '')
        p = self.P(mid, body, kind='summary'); p['text'] = clip(body, 2500)
        return [p], None

    def do_msg_scriptures(self, q, mid):
        """'Which scriptures does <message> use?': code reads the message summary's scripture list (held-out test: the model invented
        long verse lists such as 'Ephesians 2:75' for this follow-up). Nothing is generated."""
        s = self.A.summary(mid) or {}; sc = [x for x in s.get('scriptures', []) if x]; p = dict(mid=mid, head=self.head(mid, series=True), text='')
        m = self.msgs[mid]; t = f"“{m['title']}” ({self.spk(mid)}, {fdate(m['date_preached'])}) [1]"
        if not sc: return [p], (f"The summary of {t} lists no scriptures. The transcript may still quote some; ask about a verse or topic in it.", 'code:scriptures')
        return [p], (f"The summary of {t} lists these scriptures: " + ', '.join(sc) + '.', 'code:scriptures')

    def do_who_when(self, q, mid): return [dict(mid=mid, head=self.head(mid, series=True), text='')], None

    def do_moment(self, q, mid, topic):
        hits = self.A.topic(topic or q, k=4, per_message=4, msg_ids=[mid], use_vectors=self.vec)
        ps = [self.P(mid, h['text'], h['start']) for h in hits]
        if not ps: return [dict(mid=mid, head=self.head(mid), text='')], None
        kept, weak = rank_passages(topic or q, ps, keep=4, margin=1.0)      # keep all, strongest last
        ti = self.msgs[mid]['title']
        # The moment asked for must be in the passage the answer points to (v6 regression, 14 Nov: 'where does the speaker talk about laughter'
        # was answered with 2:20:24, a passage about holding hands, while [4] at 1:09:46 says 'laughter is a weapon').
        tw = [self.stem(w) for w in dict.fromkeys(re.findall(r"[a-z']+", (topic or '').lower())) if len(w) > 3 and w not in LSTOP and w not in self.FILLER]
        has = lambda p_: any(re.search(w, p_['text'].lower()) for w in tw)
        with_word = [k for k, p_ in enumerate(kept) if has(p_)]
        def code_answer():
            if weak: return f"I can't find a clear match for that in “{ti}”. " + no_match_answer(q, kept).split('. ', 1)[1]
            k = with_word[-1] if with_word else len(kept) - 1          # strongest passage that has the asked word, else the strongest
            p = kept[k]; _, t = _where(p, k + 1); return f"In “{ti}” this comes at about {t} [{k + 1}], where the speaker says: {_q(p['excerpt'])}"
        if weak: return kept, (code_answer(), 'code:no-match')
        def off_moment(out):
            cited = {int(n) for n in re.findall(r'\[(\d+)\]', out) if 1 <= int(n) <= len(kept)}
            return bool(tw and cited and with_word and not any(n - 1 in with_word for n in cited))
        check = lambda out: answer_problems(out, kept, q, require_quote=True) + (
            ['the passage cited does not mention ' + ', '.join(tw) + ' (another passage does)'] if off_moment(out) else [])
        return kept, (lambda out: (out, 'model') if not check(out) else (code_answer(), 'code:fallback'), 'model', check)

    def do_locate(self, q, f):
        hits = self.A.topic(self.expand(q)[0], k=5, per_message=1, use_vectors=self.vec, **f)
        kept, weak = rank_passages(q, [self.PH(h) for h in hits])
        if weak: return kept, (no_match_answer(q, kept), 'code:no-match')
        # "That is <message>" needs real evidence (blind batch 1: a restoration sermon was named for 'recover a forgotten email password').
        # (1) the question's DEFINING words - rare in the archive (<= 400 of 780k segments: email 150, password 24) - must be in a passage;
        # (2) the best passage must be a clear match (>= CONFIDENT), not a loose overlap ('open door ... not walk through immediately' 0.29).
        specific = self.specific_words(q)
        missing = [w for w in specific if not any(self.stem(w) in p['text'].lower() for p in kept)]
        names = ' or '.join(f"“{w}”" for w in missing[:3])
        sim = max(self.A.vec_sim(q, [h['chunk_id'] for h in hits]) or [0.0])
        if missing and len(missing) * 2 >= len(specific) and sim < self.SIM_OFFTOPIC:   # absent words AND nothing close in meaning
            return kept, (f"I can't find a message about that: none of the passages found mentions {names}. The archive may not cover it; try other words.", 'code:no-match')
        best = kept[-1]
        if missing or best.get('evidence', 0) < CONFIDENT or any(self.stem(w) not in best['text'].lower() for w in specific):
            ans = no_match_answer(q, kept[::-1])                    # nearest first, said tentatively
            if missing: ans = ans.replace('None of the passages found clearly says that.', f'None of the passages found clearly says that (none uses the words {names}; it may be said in other words).', 1)
            return kept[::-1], (ans, 'code:no-match')
        check = lambda out: answer_problems(out, kept, q, require_quote=True)
        return kept, (lambda out: finalize_locate(q, kept, False, out), 'model', check)

    SIM_OFFTOPIC = 0.63   # meaning similarity below this: nothing in the archive is close (calibrated on 7 questions only, 13 Nov)
    ASKWORDS = set('explain explains explained explaining mention mentions mentioned describe describes described discuss discusses discussed covers cover'.split())
    def specific_words(self, q):
        """The question's defining words: content words that are rare in the archive (in <= 400 of its ~780,000 transcript segments, or never)."""
        ws = [w for w in dict.fromkeys(re.findall(r"[a-z']+", q.lower())) if len(w) > 3 and w not in LSTOP and w not in self.ASKWORDS and w not in self.FILLER]
        return [w for w in ws if self.docs.get(w, 0) <= 400] if self.docs else []

    def do_topic(self, q, f):
        # A speaker in the question is a filter, not search words: 'what did babs teach on altars' searched passages ABOUT Babs
        # (praise, greetings) instead of Babs's teaching on altars. The model is still shown the user's own question.
        sq = self.filters(q)[1] if f.get('speaker') else q
        qx, nm, variants = self.expand(sq)
        if f.get('speaker'): nm, variants = [], {}     # 'What did Gerald Young say about Shemitah': Shemitah is his topic, found by the word path below
        bare = not re.search(r'\?|\b(what|who|whom|where|when|which|how|why|did|does|do|is|are|was|were|list|find|show|tell|give|can|has|have)\b', q, re.I)
        model_q0 = f"What does the archive teach about {q.strip(' .')}?" if bare and len(q.split()) <= 8 else q    # bare keywords -> a trained question form
        if f.get('speaker'): return self.speaker_topic(q, sq, f, qx)
        if not nm:
            rare = {w_: vs for w_, vs in self.rare_words(sq).items()}; loose = False
            hits = self.A.topic(qx, k=6, per_message=1, use_vectors=self.vec, **f)
            if rare:   # an unusual word the search may drown out: guarantee up to 3 passages that contain it (any spelling), best-matching the rest of the question
                allr = [v for vs in rare.values() for v in vs]; w, a = self.A._where(f)
                sql = f"select c.chunk_id, c.msg_id, c.start, c.end, c.text from chunks c join messages m on m.msg_id = c.msg_id where ({' or '.join('lower(c.text) like ?' for _ in allr)}) and {w} limit 300"
                other = {x[:6] for x in re.findall(r"[a-z']+", sq.lower()) if len(x) > 3 and x not in STOP and x not in rare}
                rx = re.compile(r'\b(' + '|'.join(re.escape(v) for v in sorted(allr, key=len, reverse=True)) + r')' + ('' if loose else r'\b'))
                other -= {v[:6] for v in allr}
                def rsc(h): t = h['text'].lower(); n = len(rx.findall(t)); return (n > 0, sum(x in t for x in other), n)
                seen = set(); picked = []
                per = {}
                for h in sorted((dict(r) for r in self.A.db.execute(sql, [f'%{v}%' for v in allr] + a)), key=rsc, reverse=True):
                    if per.get(h['msg_id'], 0) >= (2 if loose else 1) or not rx.search(h['text'].lower()): continue
                    per[h['msg_id']] = per.get(h['msg_id'], 0) + 1; seen.add(h['msg_id']); picked.append(h)
                    if len(picked) >= 3: break
                def around(h):   # clip to the mention of the word that sits closest to the question's other words (e.g. 'wheels' next to 'courts')
                    t = re.sub(r'\s+', ' ', h['text']); tl = t.lower()
                    i = max((m.start() for m in rx.finditer(tl)), key=lambda i_: sum(x in tl[max(0, i_ - 300):i_ + 350] for x in other), default=0)
                    return clip(('… ' if i > 300 else '') + t[max(0, i - 300):], 650)
                hits = [h for h in hits if h['msg_id'] not in seen][:6 - len(picked)]
                ps = [self.PH(h) for h in hits] + [dict(self.P(h['msg_id'], '', h['start']), text=around(h), span=(h['start'], h['end'])) for h in picked[::-1]]
            else: ps = [self.PH(h) for h in hits]
            ttl = lambda p: p['head'].split(' — ')[0].lower()                 # a message TITLED with the topic is about it ('Presentation on Pa Danladi Kuta')
            rp = [p for p in ps if rare and (sum(p['text'].lower().count(v) for vs in rare.values() for v in vs) >= 2
                                             or any(v in ttl(p) for vs in rare.values() for v in vs if len(v) >= 4) if loose else
                                             any(v in p['text'].lower() for vs in rare.values() for v in vs)
                                             or bool(f.get('speaker')) and any(v in ttl(p) for vs in rare.values() for v in vs if len(v) >= 4))]
            chk0 = lambda out_: ((['model declined although passages contain ' + ', '.join(rare)] if rp else []) if self.refused(out_) else answer_problems(out_, ps, q))
            if model_q0 == q and not rp: return ps, None
            def code0():
                if not rp: return self.listing(ps)
                lines = []
                rrx = {'_rare': rx}
                for p in rp[::-1][:3]:
                    lines.append(self.quote_line(ps, p, self.best_window(p['text'], rrx, 280)))
                if f.get('speaker'):   # 'Here is what the archive passages by Mrs Nuhu say about Pa Danladi Kuta:' (the user's own words for the topic)
                    subj = re.sub(r"^[\W]*((what'?s|what|did|does|do|say|said|teach|taught|preach|preached|about|on|has|have|been)\s+)+", '', sq.strip(), flags=re.I).strip(' ?.') or ', '.join(rare)
                    return f"Here is what the archive passages by {f['speaker']} say about {subj}:\n" + '\n'.join(lines)
                return f"These passages in the archive mention {', '.join(rare)}:\n" + '\n'.join(lines)
            return ps, (lambda out_: (out_, 'model') if not chk0(out_) else (code0(), 'code:fallback'), 'model', chk0, model_q0)
        # A named person: every transcript chunk that names them (any ASR spelling) is a candidate, plus the usual search hits.
        # Rank: names the person, then how often the chunk uses the question's other words (stems), e.g. "prophesied" for "prophecy".
        hits = {h['chunk_id']: h for h in self.A.topic(qx, k=12, per_message=2, use_vectors=self.vec, **f)}
        w, a = self.A._where(f)
        like = ' or '.join('lower(c.text) like ?' for vs in variants.values() for _ in vs)
        sql = f'select c.chunk_id, c.msg_id, c.start, c.end, c.text from chunks c join messages m on m.msg_id = c.msg_id where ({like}) and {w} limit 400'
        for r in self.A.db.execute(sql, [f'%{v}%' for vs in variants.values() for v in vs] + a): hits.setdefault(r['chunk_id'], dict(r))
        titled = [mid for mid, m in self.msgs.items() if self.has_name(m['title'], variants)][:20]
        if titled:
            for r in self.A.db.execute(f"select chunk_id, msg_id, start, end, text from chunks where msg_id in ({','.join('?' * len(titled))})", titled): hits.setdefault(r['chunk_id'], dict(r))
        want = {x[:6] for x in re.findall(r"[a-z']+", sq.lower()) if len(x) > 3 and x not in STOP and x not in nm and x not in ('whats', 'made', 'make', 'said', 'tell', 'program', 'programme')}
        allv = [v for vs in variants.values() for v in vs]
        # 'prophecy' should match prophesy/prophesied/prophecies but not prophet/prophetic (those are everywhere in this archive)
        pat = {x: (r'\bprophe[sc]' if x == 'prophe' else r'\b' + re.escape(x)) for x in want}
        hitc = ((lambda t: sum(len(re.findall(pat[x], t.lower())) for x in want)) if want else
                (lambda t: sum(t.lower().count(v) for v in allv)))             # "who was X?": no other words, so mentions of the person count
        def focus(text, about=False, n=900):
            """The stretch of consecutive sentences (<= n chars) that says most about the subject. A sentence's question-word hits count
            only when it, or a neighbouring sentence, names the person - or when the message title does (about=True: 'he' is them).
            This stops passages that merely repeat a word like 'prophetic' from outranking ones about the person."""
            sents = [x.strip() for x in re.split(r'(?<=[.!?])\s+', re.sub(r'\s+', ' ', text)) if x.strip()] or [text]
            nmd = [any(v in x.lower() for v in allv) for x in sents]
            wt = [hitc(x) if about or any(nmd[max(0, i - 1):i + 2]) else 0 for i, x in enumerate(sents)]
            best = (-1, '')
            for i in range(len(sents)):
                cur, k = '', 0
                for j in range(i, len(sents)):
                    if len(cur) + len(sents[j]) > n and cur: break
                    cur = (cur + ' ' + sents[j]).strip(); k += wt[j]
                    if k > best[0]: best = (k, cur)
            return best[1][:n + 200], best[0]
        cands = []
        for h in hits.values():
            about = self.has_name(self.msgs[h['msg_id']]['title'], variants)
            if not (about or self.has_name(h['text'], variants)): continue
            fx, k = focus(h['text'], about); cands.append((k, h, fx))
        cands.sort(key=lambda x: -x[0]); out, per = [], {}
        for k, h, fx in cands:
            if per.get(h['msg_id'], 0) >= 2: continue
            per[h['msg_id']] = per.get(h['msg_id'], 0) + 1
            out.append((k, dict(self.P(h['msg_id'], '', h['start']), text=fx, span=(h['start'], h.get('end') or h['start'] + 180))))
            if len(out) >= 5: break
        if not out:
            return [self.PH(h) for h in self.A.topic(sq, k=6, per_message=1, use_vectors=self.vec, **f)], None
        # Evidence strong enough to overrule a model refusal: a message titled after the person that uses the question's words twice or more,
        # or two different messages that do. (One stray hit, e.g. 'capital' near 'Australia', must not overrule a correct refusal.)
        k2 = [(k, p) for k, p in out if k >= 2]
        titled_hit = any(self.has_name(p['head'].split(' — ')[0], variants) for k, p in k2)
        strong = [p for k, p in k2] if (titled_hit or len({p['mid'] for k, p in k2}) >= 2) else []
        ps = [p for k, p in out][::-1]                                    # strongest last (the 4B favours the last passage)
        # Also clear evidence: the person AND the question's words in the same 1-2 sentences (best_excerpt >= 0.75), e.g.
        # "Peter Wagner's call for new wineskins". Not 'capital' a few lines from 'Australia' (0.5). Used only to overrule a refusal.
        clear = [p for p in ps if best_excerpt(p['text'], sq)[0] >= 0.75]
        decline_ev = strong or clear
        subject = re.split(r'\?', q)[0].strip()
        subject = re.sub(r"^\s*(what'?s|what is|what was|what are|what were|what did|what does|what do|what has|what have|tell me|which|who was|who is|who were|who are|who)\s+", '', subject, flags=re.I)
        subject = re.sub(r"^\s*(been\s+)?(said|say|preached|preach|taught|teach|spoken|speak|written)?\s*(about\s+|on\s+)?", '', subject, flags=re.I).strip() or q
        subject = re.sub(r"\s+(say|said|says|teach|taught|preach|preached|speak|spoke)\s+(about|on)\s+", ' and ', subject, flags=re.I)
        model_q = f"What does the archive say about {subject}?"           # a question form the model was trained on; the user's wording is kept in the record
        def code_answer():
            lines = []
            nrx = {'_name': re.compile('|'.join(re.escape(v) for v in allv))} if allv else {}
            nrx.update({x: re.compile(pat[x]) for x in want})
            def at_name(t_):   # the sentences that name the person and use the question's words, so the excerpt visibly concerns them
                return self.best_window(t_, nrx, 280) if nrx else clip(t_, 280)
            pool = decline_ev or [p for _, p in out]
            pool = sorted(pool, key=lambda p: not self.has_name(p['head'].split(' — ')[0], variants))   # messages titled after the person first
            for p in pool[:3]:
                lines.append(self.quote_line(ps, p, at_name(p['text'])))
            cav = ("\n\nThese passages mention " + ' '.join(n_.capitalize() for n_ in nm) + " together with prophecy; they do not always say who gave each prophecy."
                   if re.search(r'prophe', q, re.I) else '')
            return f"Here is what the archive passages say about {subject}:\n" + '\n'.join(lines) + cav
        def chk(out_):
            if self.refused(out_): return ['model declined although passages name the subject'] if decline_ev else []
            bad = answer_problems(out_, ps, q)
            if strong and f'[{len(ps)}]' not in out_: bad.append('model left out the strongest passage')
            return bad
        def fin(out_):
            if not chk(out_): return out_, 'model'
            if self.refused(out_) and decline_ev:      # declined despite clear evidence: ask once more, showing only that evidence (strongest last)
                sub = [p for p in ps if p in decline_ev][-3:]
                out2 = self.M.chat(render_question(model_q, sub))
                if not self.refused(out2) and not answer_problems(out2, sub, q): return out2, 'model:retry', sub, out2
                self.last_retry = out2
            return code_answer(), 'code:fallback'
        self.last_person = {'ps': ps, 'k': {id(p): k for k, p in out}, 'strong': strong, 'subject': subject, 'hitc': hitc, 'allv': allv, 'names': nm}   # for train/make_sft.py (train = serve)
        return ps, (fin, 'model', chk, model_q)

    FILLER = set("""had has have was were did does say said says teach taught teaches teaching preach preached preaching speak spoke spoken tell told
        about what whats which when where who whom how pls please find show give talk talks talking saying actually really any anything""".split())

    def stem(self, w):
        """Word stem for matching transcripts: prayer~pray, discernment~discern, leadership~leader, evangelism~evangel(ist|ized), angels~angel."""
        w = re.sub(r"'s$", '', w.lower())
        if re.match(r'prophe[sc]', w): return 'prophe[sc]'          # prophecy ~ prophesy ~ prophesied; not prophet/prophetic
        for suf in ('ments', 'ment', 'ship', 'ness', 'isms', 'ism', 'ings', 'ing', 'ity', 'ies', 'ers', 'er', 'ed', 'es', 's'):
            if w.endswith(suf) and len(w) - len(suf) >= 4: w = w[:-len(suf)]; break
        return w[:7]

    @staticmethod
    def units(text):
        """Sentences; ASR text without punctuation is cut into ~30-word pieces so quotes still start on a word."""
        t = re.sub(r'\s+', ' ', text).strip(); out = []
        for x in re.split(r'(?<=[.!?])\s+', t):
            ws = x.split()
            out += [' '.join(ws[i:i + 15]) for i in range(0, len(ws), 15)] if len(x) > 200 else ([x] if x else [])
        return out

    def best_window(self, text, rx, n):
        """Consecutive sentences (<= n chars) using the most distinct topic stems, then most uses."""
        us = self.units(text); best = (-1, -1, '')
        for i in range(len(us)):
            cur = ''
            for j in range(i, len(us)):
                if cur and len(cur) + len(us[j]) > n: break
                cur = (cur + ' ' + us[j]).strip(); tl = cur.lower()
                sc = (sum(1 for r in rx.values() if r.search(tl)), sum(len(r.findall(tl)) for r in rx.values()))
                if sc > best[:2]: best = (*sc, cur)
        return best[2] or clip(text, n)

    def seg_time(self, mid, lo, hi, quote):
        """Start time of the transcript segment where `quote` begins (segments are ~30 s), searched between lo and hi seconds.
        Falls back to lo. Used so a quote's timestamp points at the quoted words, not at the start of a 2-3 minute passage."""
        want = _norm(quote).split()[:6]
        if not want: return lo
        segs = self.A.db.execute('select start, text from segments where msg_id = ? and end >= ? and start <= ? order by start', (mid, lo - 1, hi + 1)).fetchall()
        joined, marks = '', []
        for st, tx in segs:
            marks.append((len(joined), st)); joined += _norm(tx) + ' '
        i = joined.find(' '.join(want))
        if i < 0: return lo
        return max(st for pos, st in marks if pos <= i)

    def quote_line(self, ps, p, quote):
        """'- “Title” (speaker, date) [n], at m:ss: “quote”' with the time of the quote itself when the passage span is known."""
        n = ps.index(p) + 1; w_, t = _where(p, n)
        if p.get('span'): t = hms(self.seg_time(p['mid'], p['span'][0], p['span'][1], quote))
        return f"- {w_}" + (f", at {t}" if t else '') + f": “{quote.rstrip(' …')}”"

    SAYVERB = r"(?:'s|s'|’s)?\s+(?:call|calls|word|words|prophecy|prophecies|statement|teaching|vision|dream|book|declaration|said|says|saying|declared|prophesied|told|gave|spoke|taught|wrote|released|shared|announced)\b"

    BIBLE_FIGURES = {'david': 'King David', 'joshua': 'Joshua in the Bible', 'moses': 'Moses in the Bible', 'gideon': 'Gideon in the Bible',
                     'john': 'John in the Bible', 'philip': 'Philip in the Bible', 'aaron': 'Aaron in the Bible', 'abel': 'Abel in the Bible',
                     'jesse': 'Jesse in the Bible', 'peter': 'Peter in the Bible', 'paul': 'Paul in the Bible'}
    SPEECH = r'(?:say|said|says|teach|taught|teaches|preach|preached|share|shared|speak|spoke|prophesy|prophesied|pray|prayed|minister|ministered|talk|talked)'

    def ambiguous_speaker(self, q, f):
        """A first name used as a speaker ('what did Mercy teach...', 'Pastor Mercy') that fits more than one archive speaker, or an archive
        speaker and a Bible figure ('what did David say...'). Returns the options to ask about; None when the name is clear."""
        if f.get('speaker'): return None
        self.speaker_by_part('x')
        for m in re.finditer(r"\b(?:(" + self.TITLES + r")\.?\s+)?([A-Za-z][a-z']{2,})\b", q):
            tok = m.group(2); low = tok.lower()
            as_person = bool(m.group(1)) or re.match(r'\s+' + self.SPEECH + r'\b', q[m.end():], re.I) or re.search(r'\bby\s+$', q[:m.start()], re.I)
            if not as_person or not (tok[0].isupper() or m.group(1)): continue
            sp = sorted(self._sparts.get(low, set()))
            if len(sp) > 1 or (len(sp) == 1 and low in self.BIBLE_FIGURES):
                return {'name': tok, 'options': sp, 'bible': self.BIBLE_FIGURES.get(low) if len(sp) == 1 else None}
        return None

    def do_clarify(self, q, name, options, bible=None):
        cnt = lambda sp: sum(1 for m in self.msgs.values() if sp in (m['speakers'] or '').split('; '))   # exact name: 'Mercy A' is not 'Mercy Ali'
        opts = [f"{sp} ({cnt(sp)} message{'s' if cnt(sp) != 1 else ''} in the archive)" for sp in options]
        if bible:
            ans = (f"Do you mean {opts[0]}, or {bible} as the messages speak about him? Ask again with the full name (\"{options[0]}\"), "
                   f"or ask \"what do the messages say about {bible}\".")
        else:
            ans = f"Which {name} do you mean: " + ' or '.join(opts) + "? Ask again with the full name."
        return [dict(mid='none', head='(no passage)', text='')], (ans, 'code:clarify')

    def do_speakers(self, q, f):
        """'Show two different speakers on <topic>, with a quote from each.' Code: the best passage per speaker, from recordings with ONE listed
        speaker only (in a shared recording the archive cannot tell who is speaking). Says so plainly when only one speaker covers it."""
        f = {k_: v_ for k_, v_ in f.items() if k_ != 'speaker'}
        drop = self.FILLER | self.ASKWORDS | self.EXPL_DROP | {'two', 'three', 'several', 'multiple', 'various', 'people', 'preachers', 'ministers', 'explaining', 'with', 'from', 'about', 'more', 'than', 'what', 'compare'}
        words = [w for w in dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z']+", MULTI_SPK.sub(' ', q))) if len(w) >= 4 and w.lower() not in LSTOP and w.lower() not in drop]
        stems = list(dict.fromkeys(self.stem(w) for w in words))
        none = [dict(mid='none', head='(no passage)', text='')]
        if not stems: return none, ("Tell me the topic, and I will look for different speakers on it.", 'code:speakers')
        rx = {s_: re.compile(r'\b' + (s_ if '[' in s_ else re.escape(s_))) for s_ in stems}; lk = lambda s_: s_.split('[')[0]
        cv = lambda t: {s_ for s_, r in rx.items() if r.search(t.lower())}
        w, a = self.A._where(f)
        cov_sql = ' + '.join('(instr(lower(c.text), ?) > 0)' for _ in stems)
        rows = [dict(r) for r in self.A.db.execute(
            f"select c.chunk_id, c.msg_id, c.start, c.end, c.text, m.speakers from chunks c join messages m on m.msg_id = c.msg_id "
            f"where ({' or '.join('lower(c.text) like ?' for _ in stems)}) and {w} and m.speakers is not null and m.speakers != '' and m.speakers not like '%;%' "
            f"order by {cov_sql} desc, length(c.text) limit 3000", [f'%{lk(s_)}%' for s_ in stems] + a + [lk(s_) for s_ in stems])]
        need = max(1, -(-len(stems) * 2 // 3)); best = {}
        for r in rows:
            win = self.best_window(r['text'], rx, 300); sc = (len(cv(win)), min(3, sum(len(x.findall(win.lower())) for x in rx.values())), len(set(win.lower().split())))
            if sc[0] >= need and sc > best.get(r['speakers'], ((0, 0, 0),))[0]: best[r['speakers']] = (sc, r, win)
        top = sorted(best.values(), key=lambda x: x[0], reverse=True)[:3]
        subj = ' '.join(words)
        if not top: return none, (f"I can't find passages about {subj} from a single named speaker in the archive.", 'code:speakers')
        ps, lines = [], []
        for sc, r, win in top:
            p_ = dict(self.P(r['msg_id'], win, self.seg_time(r['msg_id'], r['start'], r['end'], win)), span=(r['start'], r['end'])); ps.append(p_)
            lines.append(self.quote_line(ps, p_, win))
        head = (f"{len(top)} speakers who speak about {subj}. Each passage uses these words; whether they explain the same idea has not been checked:" if len(top) > 1 else
                f"I found only one speaker on {subj} in recordings with a single listed speaker ({top[0][1]['speakers']}), so I cannot show two:")
        return ps, (head + '\n' + '\n'.join(lines) + "\n\nRecordings that list more than one speaker are left out, because the archive cannot tell who is speaking in them.", 'code:speakers')

    def do_attribution(self, q, f):
        """'Who said X?' / 'was X said by A or B?'. Code, not the model: the 4B confuses the person speaking with people they mention.
        Evidence per passage: the recording's speaker (catalog metadata); a candidate the speaker ATTRIBUTES the words to ("Peter Wagner's
        call for new wineskins", "Elton said"); or a candidate merely MENTIONED nearby. The archive holds the speakers' words only."""
        hon = set(re.findall(r'[a-z]+', self.TITLES))
        f = {k_: v_ for k_, v_ in f.items() if k_ != 'speaker'}   # 'did Babs or Gerald Young say...': search every speaker, not only Babs
        # candidate people: runs of capitalised words that are names (Peter Wagner, Pa Elton), not the first word of the question
        cands = []
        zone = (re.search(r'\bby\b(.*)$', q) or re.search(r'\bdid\b(.*?\bor\b.*?)\b(?:say|said|teach|prophesy|preach|declare)\b', q, re.I)
                or re.search(r'\bwas it\b(.*\bor\b.*)$', q))
        zq, z0 = (zone.group(1), zone.start(1)) if zone else ('', 0)
        for m in re.finditer(r"(?:\b(?:Pa|Prophet|Prophetess|Pastor|Apostle|Dr|Rev|Bishop|Mama|Papa)\s+)?((?:[A-Z][a-zA-Z']+\s*){1,3})", zq):
            if m.start() + z0 == 0: continue
            ws = [w for w in m.group(1).split() if w.lower() not in self.COMMON and w.lower() not in STOP and w.lower() not in hon and self.is_name(w)]
            if ws: cands.append(' '.join(ws))
        cands = list(dict.fromkeys(cands))
        self.speaker_by_part('x')                                  # builds the speaker-name index
        def full(c):   # 'Gerald' (Young dropped as a common word) -> 'Gerald Young' when it names exactly one archive speaker
            hits_ = [self._sparts.get(x, set()) for x in c.lower().split()]
            one = set.intersection(*hits_) if hits_ else set()
            return next(iter(one)) if len(one) == 1 else c
        cands = list(dict.fromkeys(full(c) for c in cands))
        claim_src = ATTRIB.sub(' ', q)
        for c in cands: claim_src = re.sub(re.escape(c), ' ', claim_src)
        drop = self.FILLER | hon | {'statement', 'actually', 'really', 'first', 'originally', 'source', 'author', 'quote', 'this', 'that', 'gave', 'made', 'was', 'or', 'it'}
        cwords_ = [w for w in re.findall(r"[A-Za-z][A-Za-z']+", claim_src) if len(w) >= 3 and w.lower() not in LSTOP and w.lower() not in drop]
        stems = list(dict.fromkeys(self.stem(w) for w in cwords_ if len(self.stem(w)) >= 4))
        key = list(dict.fromkeys(self.stem(w) for w in cwords_ if w[0].isupper() and w.lower() not in self.COMMON and self.is_name(w)))
        if not stems: return [dict(mid='none', head='(no passage)', text='')], ("Tell me the words or the topic of the statement, and I will look for who said it.", 'code:attribution')
        rx = {s_: re.compile(r'\b' + (s_ if '[' in s_ else re.escape(s_))) for s_ in stems}
        lk = lambda s_: s_.split('[')[0]
        cv = lambda t: {s_ for s_, r in rx.items() if r.search(t.lower())}
        # candidate spellings as the transcripts have them (Pa Elton -> 'pahelton', 'paelton'), surname and full form
        var = {}
        for c in cands:
            parts = c.lower().split(); v = {c.lower(), parts[-1]} | {x for pt in parts for x in self.spellings(pt)} | {x for pt in parts for x in self.rare_words(pt).get(pt, [])}
            var[c] = sorted({x for x in v if len(x) >= 4}, key=len, reverse=True)
        w, a = self.A._where(f)
        cov_sql = ' + '.join('(instr(lower(c.text), ?) > 0)' for _ in stems)
        rows = [dict(r) for r in self.A.db.execute(
            f"select c.chunk_id, c.msg_id, c.start, c.end, c.text from chunks c join messages m on m.msg_id = c.msg_id where ({' or '.join('lower(c.text) like ?' for _ in stems)}) and {w} "
            f"order by {cov_sql} desc limit 400", [f'%{lk(s_)}%' for s_ in stems] + a + [lk(s_) for s_ in stems])]
        need = max(1, -(-len(stems) * 2 // 3))
        rows = [r for r in rows if all(k in cv(r['text']) for k in key) and len(cv(r['text'])) >= need]
        nm_in = lambda c, t: any(v in t.lower() for v in var[c])
        phrase = re.sub(r"^(that|the|this|it|was|is)\s+", '', _norm(re.sub(r'\b(?:did|was|or|say|said)\b', ' ', claim_src)).strip())   # 'laughter is a weapon', small words kept
        strength = lambda r: (phrase in _norm(r['text']) if len(cwords_) >= 2 else False, len(cv(self.best_window(r['text'], rx, 300))))
        rows.sort(key=lambda r: (*strength(r), len(cv(r['text'])), sum(nm_in(c, r['text']) for c in cands)), reverse=True)
        picked, seen = [], set()
        for r in rows:
            if r['msg_id'] in seen: continue
            seen.add(r['msg_id']); picked.append(r)
            if len(picked) >= 3: break
        if not picked:
            return [dict(mid='none', head='(no passage)', text='')], ("I can't find that statement in the archive, so I can't say who said it. Try other words from it.", 'code:attribution')
        nrx = dict(rx); nrx.update({'_' + c: re.compile('|'.join(re.escape(v) for v in var[c])) for c in cands})
        ps, lines, verdict = [], [], {c: set() for c in cands}
        top = strength(picked[0])
        for r in picked:
            decides = strength(r) == top          # only the clearest matches decide the verdict; weaker ones are shown, not counted
            quote = self.best_window(r['text'], nrx, 300)
            t0 = self.seg_time(r['msg_id'], r['start'], r['end'], quote)
            p = dict(self.P(r['msg_id'], quote, t0), span=(r['start'], r['end'])); ps.append(p)
            spk = self.spk(r['msg_id'])
            notes = []
            for c in cands:
                spoke = any(v in spk.lower() for v in var[c])
                attr = any(re.search(re.escape(v) + self.SAYVERB, quote.lower()) or re.search(r'\b(?:said|says|according to)\s+(?:\w+\s+){0,2}' + re.escape(v), quote.lower()) for v in var[c])
                if spoke: notes.append(f"{c} is the speaker of this recording"); decides and verdict[c].add('speaker')
                elif attr: notes.append(f"the speaker attributes it to {c}"); decides and verdict[c].add('attributed')
                elif nm_in(c, quote): notes.append(f"{c} is mentioned here, not as the source"); decides and verdict[c].add('mentioned')
                else: notes.append(f"{c} is not named here")
            lines.append(self.quote_line(ps, p, quote).replace(': “', f' — spoken by {spk}: “', 1) + ('\n  ' + '; '.join(notes) + '.' if notes else ''))
        att = [c for c in cands if 'attributed' in verdict[c] or 'speaker' in verdict[c]]
        spk0 = self.spk(picked[0]['msg_id'], 'an unknown speaker')
        if cands and len(att) == 1:
            others = [c for c in cands if c not in att]
            how = 'as its speaker' if 'speaker' in verdict[att[0]] else f'as reported by {spk0}'
            summ = f"The archive attributes this to {att[0]} ({how})" + (f", not to {' or '.join(others)}." if others else '.')
        elif cands and len(att) > 1: summ = "The passages connect it with more than one of them: " + ', '.join(att) + ". Read the quotes to see each one's part."
        elif cands: summ = f"The clearest passage is spoken by {spk0}; it does not credit {' or '.join(cands)}."
        else: summ = f"In the recording it is spoken by {spk0}; the passage itself does not name anyone else as its source."
        note = ("\n\nThe archive holds only what its speakers said, so a statement credited to someone else is the speaker's report of it."
                if any('attributed' in v for v in verdict.values()) else '')
        return ps, (summ + '\n' + '\n'.join(lines) + note, 'code:attribution')

    def speaker_topic(self, q, sq, f, qx):
        """'What did <speaker> say about <topic>?' A speaker has few messages, so search all of them for the topic words themselves.
        - none of their passages uses the words -> code says so (no model call), e.g. Gloria Norien on faith, Fanny in 2099;
        - a refusal is overruled only by EVIDENCE: a passage with every key name of the question (Bitcoin, Elijah, Shemitah) and at least
          half the topic words, used twice or in the message title. Then code quotes the sentences that use them."""
        sp = f['speaker']; hon = set(re.findall(r'[a-z]+', self.TITLES))
        toks = re.findall(r"[A-Za-z][A-Za-z']+", sq)
        words = [w_ for w_ in dict.fromkeys(toks) if len(w_) >= 3 and w_.lower() not in LSTOP and w_.lower() not in hon and w_.lower() not in self.FILLER]
        terms = {w_: self.stem(w_) for w_ in words}
        key = list(dict.fromkeys(terms[w_] for w_ in words if w_[0].isupper() and w_.lower() not in self.COMMON and self.is_name(w_)))
        stems = list(dict.fromkeys(terms.values()))
        toks_s = re.findall(r"[A-Za-z][A-Za-z']+", self.WHICH_PROG.sub(' ', sq))
        subj = ' '.join(w_ for w_ in toks_s if w_.lower() not in self.FILLER and w_.lower() not in ('the', 'on', 'did', 'and', 'it', 'programme', 'program') or w_.lower() == 'and').strip() or 'that'
        subj = re.sub(r'^(and\s+)+|(\s+(and|in|of|from))+$', '', subj)     # 'angels in <year>' loses its 'in' with the year
        hits = self.A.topic(qx, k=6, per_message=1, use_vectors=self.vec, **f)
        if not stems: return [self.PH(h) for h in hits], None
        rx = {s_: re.compile(r'\b' + (s_ if '[' in s_ else re.escape(s_))) for s_ in stems}
        lk = lambda s_: s_.split('[')[0]                                  # LIKE pattern for a regex stem
        cover = lambda t: {s_ for s_, r in rx.items() if r.search(t.lower())}
        w, a = self.A._where(f)
        cov_sql = ' + '.join('(instr(lower(c.text), ?) > 0)' for _ in stems)
        rows = [dict(r) for r in self.A.db.execute(
            f"select c.chunk_id, c.msg_id, c.start, c.end, c.text from chunks c join messages m on m.msg_id = c.msg_id "
            f"where ({' or '.join('lower(c.text) like ?' for _ in stems)}) and {w} order by {cov_sql} desc limit 400",
            [f'%{lk(s_)}%' for s_ in stems] + a + [lk(s_) for s_ in stems])]
        rows = [r for r in rows if cover(r['text'])]
        if not rows:
            mids = [mid for mid, m in self.msgs.items() if sp in (m['speakers'] or '')]
            yrs = f" in {f['year']}" if f.get('year') else ''
            ex = '; '.join(f"“{self.msgs[m]['title']}”" for m in mids[:4])
            return ([self.PH(h) for h in hits] or [dict(mid='none', head='(no passage)', text='')],
                    (f"I can't find {sp} speaking about {subj}{yrs} in the archive. " + (f"Their messages here include {ex}. " if ex and not yrs else '')
                     + "Try different words.", 'code:no-match'))
        sc = lambda r: (len(cover(r['text'])), sum(len(x.findall(r['text'].lower())) for x in rx.values()))
        picked, per = [], {}
        for r in sorted(rows, key=sc, reverse=True):
            if per.get(r['msg_id'], 0) >= 2: continue
            per[r['msg_id']] = per.get(r['msg_id'], 0) + 1; picked.append(r)
            if len(picked) >= 3: break
        pm = {r['msg_id'] for r in picked}
        ps = [self.PH(h) for h in hits if h['msg_id'] not in pm][:6 - len(picked)]
        def win(r):
            t_ = self.best_window(r['text'], rx, 650)
            return dict(self.P(r['msg_id'], '', self.seg_time(r['msg_id'], r['start'], r['end'], t_)), text=t_, span=(r['start'], r['end']))
        ps += [win(r) for r in picked[::-1]]   # strongest last
        need = max(1, -(-len(stems) // 2))
        def evidence(p):
            ttl = p['head'].split(' — ')[0]; c = cover(p['text']) | cover(ttl)
            uses = sum(len(x.findall(p['text'].lower())) for x in rx.values())
            return all(k in c for k in key) and len(c) >= need and (uses >= 2 or bool(cover(ttl)))
        ev = [p for p in ps if evidence(p)]
        def code_answer():
            lines = []
            for p in ev[::-1][:3]:
                lines.append(self.quote_line(ps, p, self.best_window(p['text'], rx, 280)))
            yesno = re.match(r'\s*(did|was|were|is|are|does|do|has|have)\b', q, re.I)
            head = f"Here is what the archive passages by {sp} say on this:" if yesno else f"Here is what the archive passages by {sp} say about {subj}:"
            return head + '\n' + '\n'.join(lines)
        # on-topic test for a quote: any topic word in any form (5-letter prefix: creating ~ creativity, angel ~ angels)
        loose = {s_: re.compile(r'\b' + (s_.split('[')[0][:5] if len(s_.split('[')[0]) > 5 else s_)) for s_ in stems}
        on_topic = lambda x: any(r.search(x.lower()) for r in loose.values())
        texts_ = ' '.join(_norm(p['text']) for p in ps)
        def tq(line):   # transcript quotes in a line (5+ words, verbatim in the passages), not message titles
            return [x for x in (next(g for g in m.groups() if g) for m in QUOTED.finditer(line)) if len(x.split()) >= 5 and _norm(x.rstrip(' …')) in texts_]
        def trim(out_):
            """Drop answer lines whose transcript quotes never use the topic words (blind batch: a list of company names under 'family');
            keep the rest. Returns (answer, number dropped); None when no on-topic quote would remain."""
            keep, drop, kept_q = [], 0, 0
            for line in out_.split('\n'):
                qs = tq(line)
                if qs and not any(on_topic(x) for x in qs): drop += 1; continue
                kept_q += bool(qs); keep.append(line)
            if not kept_q: return None
            txt = '\n'.join(keep).rstrip()
            return (txt + (f"\n\n({drop} quote{'s' if drop > 1 else ''} that did not mention {' or '.join(words)} {'were' if drop > 1 else 'was'} left out.)" if drop else ''), drop)
        def chk(out_):
            if self.refused(out_): return ['model declined although passages by the speaker use: ' + ', '.join(sorted(set().union(*(cover(p['text']) for p in ev))))] if ev else []
            bad = answer_problems(out_, ps, q)
            if not bad and tq(out_) and trim(out_) is None: bad.append('no quote uses the topic words: ' + ', '.join(words))
            return bad
        def fin(out_):
            if not chk(out_):
                t_ = trim(out_)
                if t_ is None or not t_[1]: return out_, 'model'
                return t_[0], 'model:trimmed'          # the model's answer minus off-topic quote lines
            return (code_answer(), 'code:fallback') if ev else (self.listing(ps), 'code:fallback')
        return ps, (fin, 'model', chk, q)

    def do_register(self, q, f, theme):
        hits = [h for h in self.A.register(theme or None, k=12 if theme else 200, **f) if h.get('ai_review') != 'n']
        _, nm, variants = self.expand(q)
        if nm:   # a named person/place must actually be in the statement; otherwise the register is the wrong tool -> search the transcripts
            hits = [h for h in hits if self.has_name(h['quote'], variants)]
            if len({h['msg_id'] for h in hits}) < 2: return self.do_topic(q, f)
        one = {}
        for h in hits: one.setdefault(h['msg_id'], h)
        hs = sorted(list(one.values())[:6], key=lambda r: (r['date_preached'] or '9999', r['t']))
        ps = [self.P(h['msg_id'], h['quote'], h['t'], h['kind']) for h in hs] or [{'mid': 'none', 'head': 'Register search: no statements found', 'text': ''}]
        return ps, None

    def do_scripture(self, q, ref):
        mids = []
        for r in self.A.db.execute('select msg_id, scriptures from summaries'):
            if r['msg_id'] in self.msgs and any(re.match(r'\s*' + re.escape(ref) + r'\b', x or '', re.I) for x in json.loads(r['scriptures'] or '[]')): mids.append(r['msg_id'])
        mids = sorted(mids, key=lambda m: self.msgs[m]['date_preached'] or '9999')[:8]
        if not mids: return [{'mid': 'none', 'head': 'Scripture search: no message summaries list this reference', 'text': ''}], None
        ps = [{'mid': m, 'head': self.head(m), 'text': 'Scriptures referenced: ' + ', '.join(json.loads(self.A.db.execute('select scriptures from summaries where msg_id=?', (m,)).fetchone()[0] or '[]')[:12])} for m in mids]
        rest = re.sub(re.escape(ref), ' ', SCRIPTURE.sub(' ', q), flags=re.I)
        drop = self.FILLER | self.ASKWORDS | {'use', 'uses', 'using', 'used', 'reference', 'references', 'referenced', 'preach', 'preached', 'messages', 'message', 'sermons', 'quote', 'quotes', 'cite', 'cites', 'chapter', 'verse', 'scripture'}
        topic = [w for w in dict.fromkeys(re.findall(r"[a-z']+", rest.lower())) if len(w) >= 4 and w not in LSTOP and w not in drop]
        if not topic: return ps, None
        note = (f"\n\nNote: the scripture lists show that these messages reference {ref}; they do not show how each message uses it "
                f"(for example for {', '.join('“' + w + '”' for w in topic[:3])}). Ask about one of the messages to see.")
        chk = lambda out: [] if self.refused(out) else answer_problems(out, ps, q)
        fin = lambda out: ((out if self.refused(out) else out + note), 'model') if not chk(out) else (self.listing(ps) + note, 'code:fallback')
        return ps, (fin, 'model', chk, q)

    def do_catalog(self, q, f):
        """Lists and counts are exact facts, so code answers them (the model cannot know a total it is not shown)."""
        rows = sorted(self.A.catalog(**f), key=lambda r: (r['date_preached'] or '9999', r['title']))
        what = ' '.join(([f"by {f['speaker']}"] if f.get('speaker') else []) + ([f"from {f['year']}"] if f.get('year') else []))
        if not rows:
            return [{'mid': 'none', 'head': 'Catalog search: no messages found', 'text': ''}], (f"The archive has no messages {what}. Its dated messages run from {self.years[0]} to {self.years[1]}.", 'code:catalog')
        shown = rows[:20]; ps = [dict(mid=r['msg_id'], head=self.head(r['msg_id'], series=True), text='') for r in shown]
        lines = '\n'.join(f"- {fdate(r['date_preached'])}: “{r['title']}”" + (f" ({r['speakers']})" if r['speakers'] and not f.get('speaker') else '') + f" [{i + 1}]" for i, r in enumerate(shown))
        more = f"\n\nThe archive lists {len(rows)} in total; these are the first 20 by date." if len(rows) > 20 else ''
        return ps, (f"Messages {what} ({len(rows)} message{'s' if len(rows) != 1 else ''}):\n{lines}{more}", 'code:catalog')


def start_session(name):
    """--session <name>: ordinary use by a reviewer, recorded. Every question and answer is appended to first_answers.jsonl the moment it is
    shown (the FIRST results, kept before anyone fixes anything), with the model and app versions in session.json. After each answer the
    reviewer may rate it with one key; nothing has to be prepared beforehand."""
    import hashlib
    d = R / 'eval' / 'human_use' / f"{time.strftime('%Y%m%d-%H%M')}_{re.sub(r'[^A-Za-z0-9_-]+', '_', name)[:40]}"
    d.mkdir(parents=True, exist_ok=False)
    code = {f: hashlib.sha256((R / f).read_bytes()).hexdigest()[:16] for f in ('app/newinegpt/ask.py', 'app/newinegpt/rank.py', 'app/newinegpt/retrieve.py', 'train/prompt.py')}
    gg = Path(MODEL)
    json.dump({'reviewer': name, 'started': time.strftime('%Y-%m-%d %H:%M'), 'model': str(gg), 'model_sha256_16': hashlib.sha256(open(gg, 'rb').read(1 << 26)).hexdigest()[:16] if gg.exists() else None,
               'code': code}, open(d / 'session.json', 'w'), indent=1)
    return d


RATE_HELP = "Rate it: g = good, p = partly right, w = wrong or misleading, Enter = skip. You can add a note after the letter (e.g. 'p wrong date')."


def main(argv):
    as_json = '--json' in argv; argv = [a for a in argv if a != '--json']
    session = None
    if '--session' in argv:
        i = argv.index('--session'); name = argv[i + 1] if i + 1 < len(argv) else 'reviewer'; argv = argv[:i] + argv[i + 2:]
        session = start_session(name)
        print(f"Recording this session in {session}\nAsk questions the way you normally would. {RATE_HELP}")
    ask = Asker()
    def show(q):
        r = ask.answer(q)
        if as_json: print(json.dumps(r, ensure_ascii=False, indent=1)); return r
        print(f"\n{r['answer']}\n\n[{r['kind']} · written by {r['source']}" + (f" · model answer rejected: {'; '.join(r['problems'])[:160]}" if r['problems'] else '') + ']')
        for p in r['passages']:
            print(f"  [{p['n']}] {p['head']}" + (f"  (programme: {p['program']})" if p.get('program') else ''))
            if os.environ.get('NEWINEGPT_SHOW_PASSAGES') and p.get('text'):     # beta testers check answers against the passage text
                t = re.sub(r'\s+', ' ', p['text']).strip(); print('      ' + (t[:600] + ' …' if len(t) > 600 else t))
        return r
    try:
        if '--file' in argv:
            qs = [l.strip() for l in open(argv[argv.index('--file') + 1]) if l.strip() and not l.startswith('#')]
            out = open(argv[argv.index('--out') + 1], 'w') if '--out' in argv else None
            for q in qs:
                print('\n' + '=' * 100 + '\nQ: ' + q); r = show(q)
                if out: out.write(json.dumps(r, ensure_ascii=False) + '\n'); out.flush()
        elif argv: show(' '.join(argv))
        else:
            n = 0
            while True:
                try: q = input('\nQuestion (empty to quit): ').strip()
                except EOFError: break
                if not q: break
                r = show(q); n += 1
                if session:
                    r = dict(r, n=n, asked_at=time.strftime('%Y-%m-%d %H:%M:%S'))
                    with open(session / 'first_answers.jsonl', 'a', encoding='utf-8') as f: f.write(json.dumps(r, ensure_ascii=False) + '\n')
                    try: g = input(f'{RATE_HELP}\n> ').strip()
                    except EOFError: g = ''
                    if g:
                        with open(session / 'ratings.jsonl', 'a', encoding='utf-8') as f:
                            f.write(json.dumps({'n': n, 'question': q, 'rating': {'g': 'good', 'p': 'partly', 'w': 'wrong'}.get(g[:1].lower(), g[:1]),
                                                'note': g[1:].strip(), 'source': r['source']}, ensure_ascii=False) + '\n')
            if session: print(f"Saved {n} answers in {session}. Thank you.")
    finally: ask.M.close()


if __name__ == '__main__': main(sys.argv[1:])
