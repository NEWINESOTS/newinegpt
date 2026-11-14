"""Explainer for explanatory questions ("what is the difference between...", "why...", "how do I know...").

Why: in the explanation experiment (eval/explain/, 13 questions, identical sufficient evidence, AI-graded blind) the fine-tuned models
mostly quoted or refused (explains 0/26 and 4/26), while the base Qwen3-4B explained (24/26) but added unsupported claims in about half
its answers (faithful 19/26). So the base model writes the explanation, and every sentence is then checked against the passage it cites;
sentences the check does not accept are cut. Statements of limits ("the passage does not ...") are kept: they claim nothing.

  ex = Explainer()                                   # NEWINEGPT_EXPLAINER=path to the base GGUF (default train/tools/qwen3-4b-base-Q8_0.gguf)
  res = ex.run(question, passages)                   # passages: [{'head','text'}...] numbered [1].. as shown to the model
  res['answer'] / res['sentences'] (each: text, cites, status kept|limit|cut) / res['raw']
STATUS 13 Nov: EXPERIMENTAL, NOT WIRED INTO THE APP. Calibrated on the 13 graded answers, no checker separated invented claims from fair paraphrase:
  yes/no against the passage kept 4 of 7 flagged inventions; 'list the additions' cut every sentence (also correct ones); yes/no against the
  sentence's own short quote rejected supported claims (E07, E12) and accepted an invention (E11); counting new content words overlapped
  completely (flagged 2-11, accepted 0-10). A 4B model cannot reliably police its own paraphrase. See eval/explain/ and NEWINEGPT_PLAN.md."""
import json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / 'train'))
from prompt import SYSTEM, render_question

EXPLAINER = os.environ.get('NEWINEGPT_EXPLAINER', str(R / 'train' / 'tools' / 'qwen3-4b-base-Q8_0.gguf'))
PORT = 8097
INSTR = ("Explain the answer in 2-4 sentences in your own words. Cite the passage that supports each point as [n]. "
         "If the passages only partly answer the question, say what they do not cover.")
LIMIT = re.compile(r"\b(?:do(?:es)? not|don't|doesn't|isn't|is not|are not|aren't)\s+(?:\w+\s+){0,3}(?:cover|say|explain|address|define|mention|specify|tell|discuss|explore|provide|detail|state|describe|establish|elaborate)"
                   r"|\bnot (?:explicitly|directly|fully|clearly)\b|\bonly partly\b|\bbeyond (?:this|that)\b", re.I)
CHECK_SYS = "You are a strict fact checker. You compare one statement with one source passage and answer with a single word."
CHECK = ('Source passage:\n"""\n{p}\n"""\n\nStatement: {s}\n\nList every word or idea in the statement that the passage does not say: an added detail, '
         'reason, definition, example, or a broader claim than the passage makes. Ignore wording that only restates the passage. '
         'If there is nothing like that, reply NONE. Otherwise reply ADDED: followed by the added items.')


class Explainer:
    def __init__(self, gguf=EXPLAINER, port=PORT):
        self.gguf, self.url, self.proc = gguf, f'http://127.0.0.1:{port}', None; self.port = port

    @property
    def available(self): return Path(self.gguf).exists()

    def _up(self):
        try:
            urllib.request.urlopen(self.url + '/health', timeout=1)
            ids = [m.get('id', '') for m in json.load(urllib.request.urlopen(self.url + '/v1/models', timeout=5)).get('data', [])]
            if any(Path(i).name == Path(self.gguf).name for i in ids): return
            raise RuntimeError(f'port {self.port} already serves {ids}, not {Path(self.gguf).name}')
        except RuntimeError: raise
        except Exception: pass
        if self.proc is None:
            self.proc = subprocess.Popen(['llama-server', '-m', self.gguf, '--port', str(self.port), '-c', '8192', '--jinja', '-ngl', '99'],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(180):
            try: urllib.request.urlopen(self.url + '/health', timeout=2); return
            except Exception: time.sleep(1)
        raise RuntimeError('explainer llama-server did not start')

    def chat(self, system, user, max_tokens):
        self._up()
        body = {'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], 'temperature': 0, 'max_tokens': max_tokens,
                'chat_template_kwargs': {'enable_thinking': False}}
        req = urllib.request.Request(self.url + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        return json.load(urllib.request.urlopen(req, timeout=600))['choices'][0]['message']['content'].split('</think>')[-1].strip()

    def close(self):
        if self.proc: self.proc.terminate(); self.proc = None

    @staticmethod
    def sentences(text):
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z“"(\[])', re.sub(r'\s+', ' ', text).strip())
        out = []
        for s in parts:   # a citation that ended up at the start of the next sentence belongs to the previous one
            m = re.match(r'^((?:\[\d+\][.,;]?\s*)+)(.*)$', s)
            if m and out: out[-1] += ' ' + m.group(1).strip(); s = m.group(2)
            if s.strip(): out.append(s.strip())
        return out

    def supported(self, sentence, text):
        s = re.sub(r'\s*\[\d+\]', '', sentence).strip()
        v = self.chat(CHECK_SYS, CHECK.format(p=text[:2500], s=s), 60).strip()
        self.last_added = '' if v.upper().startswith('NONE') else re.sub(r'^ADDED:\s*', '', v, flags=re.I)[:200]
        return v.upper().startswith('NONE')

    def check(self, raw, passages):
        res = []
        for s in self.sentences(raw):
            cites = sorted({int(x) for x in re.findall(r'\[(\d+)\]', s) if 1 <= int(x) <= len(passages)})
            if LIMIT.search(s) and not cites: res.append({'text': s, 'cites': cites, 'status': 'limit'}); continue
            pool = cites or list(range(1, len(passages) + 1))
            ok, added = [], []
            for n in pool:
                if self.supported(s, passages[n - 1].get('text', '')): ok.append(n)
                else: added.append(self.last_added)
            if ok and not cites: s = s.rstrip('.') + f' [{ok[0]}].'           # an uncited sentence the check accepts gets its source
            res.append({'text': s, 'cites': ok or cites, 'status': 'kept' if ok else ('limit' if LIMIT.search(s) else 'cut'),
                        'added': '' if ok else '; '.join(a_ for a_ in added if a_)[:200]})
        return res

    def run(self, question, passages):
        raw = self.chat(SYSTEM, render_question(f'{question}\n{INSTR}', [{'head': p['head'], 'text': p.get('text', '')} for p in passages]), 450)
        sents = self.check(raw, passages)
        kept = [x for x in sents if x['status'] == 'kept']
        body = ' '.join(x['text'] for x in sents if x['status'] in ('kept', 'limit'))
        return {'raw': raw, 'sentences': sents, 'n_kept': len(kept), 'n_cut': sum(x['status'] == 'cut' for x in sents), 'answer': body}
