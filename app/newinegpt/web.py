"""newineGPT chat window (14 Nov 2026): a local web page in the user's own browser, so non-technical testers never type in a terminal.

  python -m newinegpt.web [--port 8765] [--no-browser]

Python standard library only (the offline package bundles no web framework). Listens on 127.0.0.1 only, and every API call must carry
the random token the browser was opened with (other websites cannot read it), so nothing outside this computer, and no other web page,
can use it. One question is answered at a time (one model).

Memory: every conversation is saved as eval/human_use/conversations/<id>.json and listed in the sidebar after a restart. For review,
answers and ratings are also appended to eval/human_use/<date>_<name>/first_answers.jsonl and ratings.jsonl, as in the terminal session.
"""
import json, os, re, secrets, signal, socket, sys, threading, time, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import ask as A

HERE = Path(__file__).resolve().parent
USE = A.R / 'eval' / 'human_use'; CONV = USE / 'conversations'; SETTINGS = USE / 'settings.json'
TOKEN = secrets.token_urlsafe(24)
LOCK = threading.Lock()
STATE = {'asker': None, 'ready': False, 'error': None, 'session': None, 'name': None, 'n': 0, 'started': time.time()}


def load_settings():
    try: return json.loads(SETTINGS.read_text(encoding='utf-8'))
    except Exception: return {}


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8'); os.replace(tmp, path)


def warm_up():
    """Load the archive and start both llama.cpp servers before the first question (the first answer is otherwise 1-3 minutes on CPU)."""
    try:
        a = A.Asker(); a.A.embed_query('warm up'); a.M._up(); STATE['asker'] = a; STATE['ready'] = True
    except Exception as e:
        STATE['error'] = f'{type(e).__name__}: {e}'


def session_dir():
    if STATE['session'] is None:
        name = STATE['name'] or load_settings().get('name') or 'tester'
        base = USE / f"{time.strftime('%Y%m%d-%H%M')}_{re.sub(r'[^A-Za-z0-9_-]+', '_', name)[:40]}"
        d, k = base, 2
        while d.exists(): d = Path(f'{base}_{k}'); k += 1
        d.mkdir(parents=True)
        info = {'reviewer': name, 'started': time.strftime('%Y-%m-%d %H:%M'), 'model': A.MODEL, 'interface': 'web'}
        save_json(d / 'session.json', info); STATE['session'] = d
    return STATE['session']


def append(path, obj):
    with open(path, 'a', encoding='utf-8') as f: f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def conv_path(cid):
    if not re.fullmatch(r'[A-Za-z0-9_-]{6,40}', cid or ''): raise ValueError('bad conversation id')
    return CONV / f'{cid}.json'


def do_ask(body):
    q = (body.get('question') or '').strip()[:1000]
    if not q: raise ValueError('empty question')
    cid = body.get('conversation') or time.strftime('%Y%m%d%H%M%S') + secrets.token_hex(3)
    if not STATE['ready']:
        raise RuntimeError(STATE['error'] or 'The model is still loading. Please wait a moment and ask again.')
    with LOCK:
        t0 = time.time(); r = STATE['asker'].answer(q); secs = round(time.time() - t0, 1)
        STATE['n'] += 1; n = STATE['n']
    msg = {'question': q, 'answer': r['answer'], 'passages': r.get('passages', []), 'kind': r.get('kind'), 'source': r.get('source'),
           'asked_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'seconds': secs, 'n': n}
    p = conv_path(cid)
    conv = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'id': cid, 'title': q[:80], 'created': msg['asked_at'], 'messages': []}
    conv['messages'].append(msg); conv['updated'] = msg['asked_at']; save_json(p, conv)
    append(session_dir() / 'first_answers.jsonl', dict(r, n=n, asked_at=msg['asked_at'], seconds=secs, conversation=cid))
    return {'conversation': cid, 'message': msg, 'index': len(conv['messages']) - 1}


def do_rate(body):
    cid, i = body.get('conversation'), int(body.get('index', -1))
    rating = {'g': 'good', 'p': 'partly', 'w': 'wrong'}.get((body.get('rating') or '')[:1])
    if not rating: raise ValueError('rating must be good, partly or wrong')
    p = conv_path(cid); conv = json.loads(p.read_text(encoding='utf-8')); m = conv['messages'][i]
    m['rating'] = rating; m['note'] = (body.get('note') or '').strip()[:500]; save_json(p, conv)
    append(session_dir() / 'ratings.jsonl', {'n': m.get('n'), 'question': m['question'], 'rating': rating, 'note': m['note'],
                                              'source': m.get('source'), 'conversation': cid, 'rated_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    return {'ok': True}


def list_convs():
    out = []
    for p in sorted(CONV.glob('*.json'), reverse=True):
        try: c = json.loads(p.read_text(encoding='utf-8')); out.append({'id': c['id'], 'title': c.get('title', ''), 'updated': c.get('updated', ''), 'count': len(c['messages'])})
        except Exception: pass
    return sorted(out, key=lambda c: c['updated'], reverse=True)


class H(BaseHTTPRequestHandler):
    server_version = 'newineGPT'

    def log_message(self, *a): pass

    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store'); self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data:")
        self.end_headers(); self.wfile.write(data)

    def _host_ok(self):        # refuse requests addressed to another host name (DNS rebinding)
        return (self.headers.get('Host') or '').split(':')[0] in ('127.0.0.1', 'localhost')

    def _auth(self):
        return self._host_ok() and secrets.compare_digest(self.headers.get('X-Token', ''), TOKEN)

    def do_GET(self):
        if not self._host_ok(): return self._send(403, {'error': 'forbidden'})
        path = self.path.split('?')[0]
        if path in ('/', '/index.html'):
            return self._send(200, (HERE / 'web_index.html').read_bytes(), 'text/html; charset=utf-8')
        if not self._auth(): return self._send(403, {'error': 'forbidden'})
        try:
            if path == '/api/status':
                s = load_settings()
                return self._send(200, {'ready': STATE['ready'], 'error': STATE['error'], 'name': STATE['name'] or s.get('name'),
                                        'model': Path(A.MODEL).name, 'loading_seconds': int(time.time() - STATE['started'])})
            if path == '/api/conversations': return self._send(200, list_convs())
            if path == '/api/conversation':
                cid = self.path.split('id=', 1)[1] if 'id=' in self.path else ''
                return self._send(200, json.loads(conv_path(cid).read_text(encoding='utf-8')))
        except Exception as e:
            return self._send(400, {'error': str(e)})
        self._send(404, {'error': 'not found'})

    def do_POST(self):
        if not self._auth(): return self._send(403, {'error': 'forbidden'})
        try:
            n = int(self.headers.get('Content-Length') or 0)
            if n > 100_000: raise ValueError('request too large')
            body = json.loads(self.rfile.read(n) or b'{}')
            path = self.path.split('?')[0]
            if path == '/api/ask': return self._send(200, do_ask(body))
            if path == '/api/rate': return self._send(200, do_rate(body))
            if path == '/api/name':
                name = (body.get('name') or '').strip()[:60] or 'tester'; STATE['name'] = name
                s = load_settings(); s['name'] = name; save_json(SETTINGS, s); return self._send(200, {'ok': True})
        except Exception as e:
            return self._send(400 if isinstance(e, ValueError) else 503, {'error': str(e)})
        self._send(404, {'error': 'not found'})


def free_port(start):
    for p in range(start, start + 50):
        with socket.socket() as s:
            try: s.bind(('127.0.0.1', p)); return p
            except OSError: continue
    raise RuntimeError('no free port')


def main(argv):
    port = free_port(int(argv[argv.index('--port') + 1]) if '--port' in argv else 8765)
    # Stop the two llama.cpp helpers on ANY stop (window closed = SIGHUP, kill = SIGTERM), not only a normal exit: otherwise they keep
    # running and holding memory (found 14 Nov: a killed test server left both behind, and the next start silently reused them).
    def _stop(*_): raise KeyboardInterrupt
    for sig in ('SIGTERM', 'SIGHUP', 'SIGBREAK'):
        if hasattr(signal, sig):
            try: signal.signal(getattr(signal, sig), _stop)
            except (ValueError, OSError): pass
    CONV.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=warm_up, daemon=True).start()
    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    url = f'http://127.0.0.1:{port}/?t={TOKEN}'
    (USE / 'last_url.txt').write_text(url + '\n', encoding='utf-8')          # tests and the 'open again' hint read it
    print(f'\nnewineGPT is running in your web browser.\nIf it did not open, copy this address into your browser:\n  {url}\n'
          f'Keep this window open while you use newineGPT. Close it to stop.\n', flush=True)
    if '--no-browser' not in argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        if STATE['asker']:
            STATE['asker'].M.close()
            emb = getattr(STATE['asker'].A, '_emb', None)
            if hasattr(emb, 'close'): emb.close()


if __name__ == '__main__':
    main(sys.argv[1:])
