"""Resume the v6 GGUF download after the 23:34 network drop (finish.sh fix, 13 Nov): for every byte range of every GGUF, find how far it was
written (first sparse hole after the range start, minus 1 MiB for safety), fetch only the rest over parallel ssh streams (retried), verify
sha256 against the pod's SHA256SUMS, rsync-repair a file that still fails, then terminate the pod.
finish.sh's bug: its final `wait` also waited for its own hard-cap timer subshell, so it could never terminate the pod before the cap."""
import concurrent.futures as cf, hashlib, json, os, subprocess, sys, time, urllib.request
from pathlib import Path
N = 'newinegpt-4b-v6'; T = Path.home() / 'newine_local' / 'train'; D = T / 'pod_out' / N / 'gguf'; LOG = T / f'launch_{N}.log'
env = dict(l.strip().split('=', 1) for l in open(T / f'pod_{N}.env') if '=' in l)
key = [l.split('=', 1)[1].strip().strip('"\'') for l in open(os.environ.get('RUNPOD_ENV', str(Path.home() / '.config/newinegpt/runpod.env'))) if l.strip().startswith(('RUNPOD_API_KEY=', 'export RUNPOD_API_KEY='))][0]
HOST, PORT, POD = env['HOST'], env['PORT'], env['POD']
O = f"-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -i {Path.home()}/.ssh/id_ed25519"
MiB = 1 << 20; PARTS = 4
def log(m): open(LOG, 'a').write(time.strftime('%H:%M:%S ') + m + '\n'); print(time.strftime('%H:%M:%S ') + m, flush=True)
def terminate(why):
    # curl, not urllib: this python.org Python has no CA store (13 Nov: CERTIFICATE_VERIFY_FAILED left the pod running)
    q = json.dumps({'query': f'mutation{{ podTerminate(input:{{podId:"{POD}"}}) }}'})
    subprocess.run(['curl', '-s', '-m', '30', '-H', 'content-type: application/json', '-H', '@-', 'https://api.runpod.io/graphql', '-d', q],
                   input=f'Authorization: Bearer {key}\n', text=True, check=False)
    log(f'terminated {POD} ({why})')
want = {l.split()[1]: l.split()[0] for l in open(D / 'SHA256SUMS')}
sizes = {}
for f in want:
    if f.endswith('.gguf'):
        sizes[f] = int(subprocess.run(f"ssh -n {O} -p {PORT} root@{HOST} 'stat -c %s /root/out/gguf/{f}'", shell=True, capture_output=True, text=True, timeout=60).stdout.strip())
jobs = []
for f, size in sizes.items():
    per = (size // MiB + PARTS) // PARTS; total = -(-size // MiB); fd = os.open(D / f, os.O_RDWR)
    for i in range(PARTS):
        s, e = i * per, min((i + 1) * per, total)
        try: h = os.lseek(fd, s * MiB, os.SEEK_HOLE)
        except OSError: h = s * MiB
        r = max(s, h // MiB - 1)
        if r < e: jobs.append((f, r, e - r))
    os.close(fd)
log(f"resuming: {sum(c for _, _, c in jobs)} MiB left in {len(jobs)} ranges")
def run(j):
    f, skip, cnt = j
    cmd = (f"set -o pipefail; ssh -n {O} -p {PORT} root@{HOST} 'dd if=/root/out/gguf/{f} bs=1M skip={skip} count={cnt} status=none' "
           f"| dd of={D / f} bs=1M seek={skip} conv=notrunc status=none")
    for t in range(1, 9):
        if subprocess.run(['bash', '-c', cmd]).returncode == 0: return True
        log(f'range {f}@{skip}MiB failed (try {t})'); time.sleep(10)
    return False
t0 = time.time()
with cf.ThreadPoolExecutor(len(jobs) or 1) as ex: ok = list(ex.map(run, jobs))
log(f'resume transfer done in {time.time() - t0:.0f} s, all ranges ok: {all(ok)}')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''): h.update(b)
    return h.hexdigest()
for f in sizes:
    p = D / f
    if p.stat().st_size > sizes[f]: os.truncate(p, sizes[f])
    for t in range(3):
        if p.stat().st_size == sizes[f] and sha(p) == want[f]: log(f'{f}: sha256 OK'); break
        log(f'{f}: sha256 mismatch, rsync repair (try {t + 1})')
        subprocess.run(['rsync', '-a', '--partial', '--no-whole-file', '-e', f'ssh {O} -p {PORT}', f'root@{HOST}:/root/out/gguf/{f}', str(D) + '/'])
    else:
        log(f'{f}: STILL FAILING; pod left up for inspection (caps armed)'); sys.exit(2)
assert sha(D / 'imatrix.dat') == want['imatrix.dat']
assert subprocess.run(f"cd {D.parent} && shasum -a 256 -c SHA256SUMS", shell=True).returncode == 0
terminate('finished'); log(f'FINISHED (resume {time.time() - t0:.0f} s)')
