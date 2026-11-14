"""Retrieval over pack/newine_archive.sqlite. No network; CPU only is fine.

  Archive(path).phrase(quote)                      -> where a phrase was said (message + timestamp)
  Archive(path).topic(question, filters)           -> passages about a topic (BM25 + embeddings, fused)
  Archive(path).catalog(speaker=, year=, series=, date_from=, date_to=, title=) -> messages
  Archive(path).passage(msg_id, t)                 -> text around a timestamp (for citations)
Filters everywhere: speaker, year, year_from, year_to, series, date_from, date_to (ISO), msg_ids.
"""
import atexit, json, os, re, sqlite3, subprocess, time, urllib.request
from pathlib import Path
import numpy as np

DEFAULT = Path(__file__).resolve().parents[2] / 'pack' / 'newine_archive.sqlite'   # project-relative, so the folder can live anywhere (Windows/Linux)
WORD = re.compile(r"[A-Za-z0-9']+")
STOP = set('a an the and or of to in on at for by with from is are was were be been it this that these those what which who whom when where why how did do does i you he she we they my your his her our their me him us them as about into than then so not no yes'.split())

# llama.cpp server binary: on PATH, or the one shipped in the offline package (its start script sets NEWINEGPT_LLAMA_SERVER).
LLAMA_SERVER = os.environ.get('NEWINEGPT_LLAMA_SERVER', 'llama-server')
EMBED_GGUF = Path(os.environ.get('NEWINEGPT_EMBED_GGUF', Path(__file__).resolve().parents[2] / 'models' / 'bge-small-en-v1.5-f16.gguf'))


class LlamaEmbedder:
    """Query vectors through llama.cpp instead of PyTorch (14 Nov 2026, for the offline Windows / Linux / Intel-Mac packages: PyTorch no
    longer ships for Intel Macs and adds ~1 GB). bge-small-en-v1.5 converted to GGUF f16 (llama.cpp convert_hf_to_gguf.py), CLS pooling,
    normalised here. Checked against sentence-transformers on 124 questions: cosine >= 0.99999, same best chunk 123/124, top-150 overlap 99.8%."""
    def __init__(self, gguf=EMBED_GGUF, port=int(os.environ.get('NEWINEGPT_EMBED_PORT', 8092))):
        self.gguf, self.port, self.url, self.proc = str(gguf), port, f'http://127.0.0.1:{port}', None

    def _up(self):
        try: urllib.request.urlopen(self.url + '/health', timeout=1); up = True
        except Exception: up = False
        if up:
            ids = [m.get('id', '') for m in json.load(urllib.request.urlopen(self.url + '/v1/models', timeout=5)).get('data', [])]
            if any(Path(i).name == Path(self.gguf).name for i in ids): return
            raise RuntimeError(f'port {self.port} already serves {ids}, not {Path(self.gguf).name}')
        self.proc = subprocess.Popen([LLAMA_SERVER, '-m', self.gguf, '--port', str(self.port), '--embedding', '--pooling', 'cls', '-c', '512',
                                      '-b', '512', '-ub', '512', '-ngl', '0', '-t', '2'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.close)
        for _ in range(120):
            try: urllib.request.urlopen(self.url + '/health', timeout=2); return
            except Exception: time.sleep(0.5)
        raise RuntimeError('search-model llama-server did not start')

    def encode(self, text):
        if self.proc is None: self._up()
        req = urllib.request.Request(self.url + '/v1/embeddings', json.dumps({'input': text}).encode(), {'Content-Type': 'application/json'})
        v = np.array(json.load(urllib.request.urlopen(req, timeout=120))['data'][0]['embedding'], dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def close(self):
        if self.proc: self.proc.terminate(); self.proc = None


def hms(t):
    t = int(t); return f'{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}' if t >= 3600 else f'{t // 60}:{t % 60:02d}'

class Archive:
    def __init__(self, path=DEFAULT, embedder=None):
        # a real file: URI (Windows drive letters, spaces, '#' in the folder path); read-only
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.meta = {r['key']: r['value'] for r in self.db.execute('select key, value from meta')}
        self.syn = json.loads(self.meta.get('synonyms', '{}'))
        self.alias = {a.lower(): c for c, al in json.loads(self.meta.get('speaker_aliases', '{}')).items() for a in al + [c]}
        self._emb = embedder; self._vec = None; self._vec_ids = None

    # ---------- filters ----------
    def _where(self, f, alias='m'):
        f = f or {}; w, a = [], []
        if f.get('speaker'):
            sp = self.alias.get(f['speaker'].lower(), f['speaker'])
            w.append(f"{alias}.speakers LIKE ?"); a.append(f'%{sp}%')
        if f.get('year'): w.append(f'{alias}.year = ?'); a.append(int(f['year']))
        if f.get('year_from'): w.append(f'{alias}.year >= ?'); a.append(int(f['year_from']))
        if f.get('year_to'): w.append(f'{alias}.year <= ?'); a.append(int(f['year_to']))
        if f.get('series'): w.append(f"({alias}.series LIKE ? OR {alias}.title LIKE ? OR {alias}.post_title LIKE ?)"); a += [f"%{f['series']}%"] * 3
        if f.get('date_from'): w.append(f'{alias}.date_preached >= ?'); a.append(f['date_from'])
        if f.get('date_to'): w.append(f"{alias}.date_preached <= ? AND {alias}.date_preached != ''"); a.append(f['date_to'] + '~')
        if f.get('msg_ids'): w.append(f"{alias}.msg_id IN ({','.join('?' * len(f['msg_ids']))})"); a += list(f['msg_ids'])
        return (' AND '.join(w) or '1'), a

    def message(self, msg_id):
        r = self.db.execute('select * from messages where msg_id=?', (msg_id,)).fetchone()
        return dict(r) if r else None

    # ---------- catalog ----------
    def catalog(self, title=None, limit=100000, **f):
        w, a = self._where(f)
        if title:
            for tok in [t for t in WORD.findall(title.lower()) if t not in STOP]:
                w += " AND (lower(m.title) LIKE ? OR lower(m.post_title) LIKE ?)"; a += [f'%{tok}%'] * 2
        q = f"select * from messages m where {w} order by m.date_preached, m.title limit ?"
        return [dict(r) for r in self.db.execute(q, a + [limit])]

    # ---------- phrase ----------
    def phrase(self, quote, k=5, **f):
        toks = [t.lower() for t in WORD.findall(quote)]
        if not toks: return []
        w, a = self._where(f)
        tries = [('exact', '"' + ' '.join(toks) + '"')]                                  # exact phrase
        if len(toks) >= 4: tries.append(('near', 'NEAR(' + ' '.join(f'"{t}"' for t in toks) + f', {max(4, len(toks) // 3)})'))  # all words close together
        if len(toks) >= 6:                                                                # tolerate a wrong/missing word: halves as phrases
            h = len(toks) // 2; tries.append(('partial', f'"{" ".join(toks[:h])}" OR "{" ".join(toks[h:])}"'))
        seen, out = set(), []
        for kind, mq in tries:
            sql = f'''select s.id, s.msg_id, s.start, s.end, s.text, bm25(phrase_fts) score from phrase_fts
                      join segments s on s.id = phrase_fts.rowid join messages m on m.msg_id = s.msg_id
                      where phrase_fts match ? and {w} order by score limit ?'''
            try: rows = self.db.execute(sql, [mq] + a + [k * 4]).fetchall()
            except sqlite3.OperationalError: continue
            for r in rows:
                start = r['start']
                # the window is segment + next segment; if the quote's opening words are not in the first one, cite the next one's time
                head = ' '.join(toks[:3]); own = ' '.join(t.lower() for t in WORD.findall(r['text']))
                if head not in own:
                    nxt = self.db.execute('select start from segments where id=? and msg_id=?', (r['id'] + 1, r['msg_id'])).fetchone()
                    if nxt: start = nxt['start']
                key = (r['msg_id'], int(start) // 30)
                if key in seen: continue
                seen.add(key); out.append({'msg_id': r['msg_id'], 'start': start, 'text': self._window(r['id']), 'match': kind})
            if len(out) >= k: break
        return out[:k]

    def _window(self, seg_id, before=1, after=2):
        r = self.db.execute('select msg_id from segments where id=?', (seg_id,)).fetchone()
        rows = self.db.execute('select text from segments where msg_id=? and id between ? and ? order by id', (r['msg_id'], seg_id - before, seg_id + after)).fetchall()
        return ' '.join(x['text'] for x in rows)

    # ---------- topic ----------
    def _fts_query(self, text):
        toks = [t for t in WORD.findall(text.lower()) if t not in STOP and len(t) > 1]
        extra = []
        for key, al in self.syn.items():
            if key.lower() in text.lower() or any(x.lower() in text.lower() for x in al):
                extra += [t for x in [key] + al for t in WORD.findall(x.lower()) if t not in STOP]
        toks = list(dict.fromkeys(toks + extra))
        return ' OR '.join(f'"{t}"' for t in toks)

    def _load_vecs(self):
        if self._vec is None:
            ids, vs = [], []
            for r in self.db.execute('select chunk_id, vec from chunk_vec order by chunk_id'):
                ids.append(r[0]); vs.append(np.frombuffer(r[1], dtype=np.float16))
            self._vec_ids = np.array(ids); self._vec = np.vstack(vs).astype(np.float32) if vs else np.zeros((0, 384), np.float32)
        return self._vec

    def embed_query(self, q):
        # Default: llama.cpp + the GGUF search model (no PyTorch). NEWINEGPT_EMBED_BACKEND=torch keeps the original sentence-transformers path.
        if self._emb is None and os.environ.get('NEWINEGPT_EMBED_BACKEND', 'llama') != 'torch' and EMBED_GGUF.exists():
            self._emb = LlamaEmbedder()
        if isinstance(self._emb, LlamaEmbedder): return self._emb.encode(self.meta['query_prefix'] + q)
        if self._emb is None:
            from sentence_transformers import SentenceTransformer
            # Use the copy shipped with the project (models/<name>) so the app works offline on any machine; the hub name is the fallback.
            local = Path(os.environ.get('NEWINEGPT_EMBED', Path(__file__).resolve().parents[2] / 'models' / self.meta['embed_model'].split('/')[-1]))
            self._emb = SentenceTransformer(str(local) if local.exists() else self.meta['embed_model'], device='cpu')
        return self._emb.encode([self.meta['query_prefix'] + q], normalize_embeddings=True)[0].astype(np.float32)

    def topic(self, question, k=10, per_message=2, use_vectors=True, **f):
        w, a = self._where(f)
        allowed = None
        if w != '1': allowed = {r[0] for r in self.db.execute(f'select msg_id from messages m where {w}', a)}
        ranks = {}
        fq = self._fts_query(question)
        if fq:
            sql = 'select c.chunk_id, c.msg_id from chunks_fts join chunks c on c.chunk_id = chunks_fts.rowid where chunks_fts match ? order by bm25(chunks_fts) limit 400'
            n = 0
            for r in self.db.execute(sql, (fq,)):
                if allowed is not None and r['msg_id'] not in allowed: continue
                ranks.setdefault(r['chunk_id'], {})['bm25'] = n; n += 1
                if n >= 150: break
        if use_vectors and self.meta.get('embed_model'):
            V = self._load_vecs()
            if len(V):
                sims = V @ self.embed_query(question)
                order = np.argsort(-sims)
                n = 0
                cm = None
                if allowed is not None:
                    cm = dict(self.db.execute('select chunk_id, msg_id from chunks').fetchall())
                for j in order[:5000]:
                    cid = int(self._vec_ids[j])
                    if cm is not None and cm[cid] not in allowed: continue
                    ranks.setdefault(cid, {})['vec'] = n; n += 1
                    if n >= 150: break
        fused = sorted(ranks.items(), key=lambda kv: -sum(1.0 / (60 + r) for r in kv[1].values()))
        out, per = [], {}
        for cid, rk in fused:
            c = self.db.execute('select * from chunks where chunk_id=?', (cid,)).fetchone()
            if per.get(c['msg_id'], 0) >= per_message: continue
            per[c['msg_id']] = per.get(c['msg_id'], 0) + 1
            out.append({'chunk_id': cid, 'msg_id': c['msg_id'], 'start': c['start'], 'end': c['end'], 'text': c['text'],
                        'score': round(sum(1.0 / (60 + r) for r in rk.values()), 5), 'via': '+'.join(sorted(rk))})
            if len(out) >= k: break
        return out

    def vec_sim(self, question, chunk_ids):
        """Meaning similarity (cosine, search-model vectors) between the question and each chunk; 0.0 when vectors are unavailable."""
        if not self.meta.get('embed_model'): return [0.0 for _ in chunk_ids]
        V = self._load_vecs()
        if not len(V): return [0.0 for _ in chunk_ids]
        if not hasattr(self, '_vec_pos'): self._vec_pos = {int(c): j for j, c in enumerate(self._vec_ids)}
        qv = self.embed_query(question)
        return [float(V[self._vec_pos[c]] @ qv) if c in self._vec_pos else 0.0 for c in chunk_ids]

    # ---------- Phase 4: summaries, prophetic register, overviews ----------
    def _has(self, table):
        return bool(self.db.execute("select 1 from sqlite_master where name=?", (table,)).fetchone())

    def summary(self, msg_id):
        if not self._has('summaries'): return None
        r = self.db.execute('select * from summaries where msg_id=?', (msg_id,)).fetchone()
        if not r: return None
        d = dict(r)
        for k in ('key_points', 'scriptures', 'themes'): d[k] = json.loads(d[k] or '[]')
        return d

    def register(self, query=None, k=20, points_to=None, **f):
        """Prophetic-statement CANDIDATES, filtered like everything else; ordered by date spoken.
        quote = the transcript's own words (ASR, contiguous; not audio-checked); quote_model = the extracting model's wording.
        Not human-reviewed: an AI review of 100 entries judged only ~45-50% clearly prophetic, so present these as candidates
        (ai_review = y/n/? on the sampled rows, blank elsewhere)."""
        if not self._has('prophetic_register'): return []
        w, a = self._where(f)
        w += " AND coalesce(p.ai_review, '') != 'n'"   # hide entries the 11 Nov AI review rejected (still in the table)
        sql = f'select p.* from prophetic_register p join messages m on m.msg_id = p.msg_id where {w}'
        if query:
            fq = self._fts_query(query)
            if fq:
                sql = f'''select p.* from register_fts join prophetic_register p on p.id = register_fts.rowid join messages m on m.msg_id = p.msg_id
                          where register_fts match ? and {w} order by bm25(register_fts)'''
                a = [fq] + a
        if points_to: sql = f'select * from ({sql}) where points_to like ?'; a.append(f'%{points_to}%')
        rows = [dict(r) for r in self.db.execute(sql + ' limit ?', a + [k])]
        if not query: rows.sort(key=lambda r: (r['date_preached'] or '9999', r['t']))
        for r in rows: r['start'] = r['t']; r['themes'] = json.loads(r['themes'] or '[]')
        return rows

    def overview(self, series=None, year=None):
        if not self._has('rollups'): return None
        key = f'series:{series}' if series else f'year:{year}'
        r = self.db.execute('select * from rollups where key=?', (key,)).fetchone()
        if not r and series:
            r = self.db.execute("select * from rollups where kind='series' and name like ?", (f'%{series}%',)).fetchone()
        if not r: return None
        d = dict(r); d['threads'] = json.loads(d['threads'] or '[]'); return d

    # ---------- citations ----------
    def passage(self, msg_id, t, span=60):
        rows = self.db.execute('select start, text from segments where msg_id=? and end >= ? and start <= ? order by start', (msg_id, t - 5, t + span)).fetchall()
        return ' '.join(r['text'] for r in rows)

    def cite(self, hit):
        m = self.message(hit['msg_id'])
        return f"{m['title']} — {m['speakers'] or 'speaker unknown'}, {m['date_preached'] or 'date unknown'}, at {hms(hit['start'])}"
