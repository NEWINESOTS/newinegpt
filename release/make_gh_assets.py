"""Cut the GitHub release downloads from the built package (14 Nov 2026).
  python3 release/make_gh_assets.py  ->  release/gh_assets/
    newineGPT-beta-<system>.zip  (one per system: app, launcher, archive, search model, that system's runtime; NO model file)
    newinegpt-4b-v6-Q5_K_M-imat-0000{1,2}-of-00002.gguf, newinegpt-4b-v6-Q4_K_M-imat-0000{1,2}-of-00002.gguf  (made beforehand with
      llama-gguf-split --split-max-size 1900M; GitHub allows 2 GB per file; llama.cpp loads part 2 itself)
    SHA256SUMS.txt  (all downloads)
Each zip has one top folder, newineGPT/, and its own SHA256SUMS.txt for the files inside it."""
import hashlib, subprocess, sys
from pathlib import Path
R = Path.home() / 'newine_local'; PKG = R / 'release' / 'newineGPT_beta_20261114'; OUT = R / 'release' / 'gh_assets'
SYSTEMS = {'windows-x64': ('windows-x64', 'START_Windows.bat'), 'linux-x64': ('linux-x64', 'START_Linux.sh'),
           'mac-intel': ('macos-x64', 'START_Mac.command'), 'mac-apple-silicon': ('macos-arm64', 'START_Mac.command')}
STARTS = {'START_Windows.bat', 'START_Linux.sh', 'START_Mac.command'}


def sha(p, n=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(n), b''): h.update(b)
    return h.hexdigest()


def main():
    listed = [l.rstrip('\n').split('  ', 1) for l in open(PKG / 'SHA256SUMS.txt', encoding='utf-8')]
    for name, (rt, start) in SYSTEMS.items():
        z = OUT / f'newineGPT-beta-{name}.zip'
        if z.exists(): sys.exit(f'REFUSED: {z} exists')
        stage = OUT / '_stage' / name / 'newineGPT'
        keep = [(h, p) for h, p in listed
                if not p.endswith('.gguf') or p.startswith('models/')                       # the v6 model ships separately
                if not (p.startswith('runtime/') and not p.startswith(f'runtime/{rt}/'))
                if not (p in STARTS and p != start)]
        for h, p in keep:
            (stage / p).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['cp', '-c', '-p', str(PKG / p), str(stage / p)], check=True)
        (stage / 'eval' / 'human_use').mkdir(parents=True, exist_ok=True)
        with open(stage / 'SHA256SUMS.txt', 'w', encoding='utf-8', newline='\n') as f:
            for h, p in keep: f.write(f'{h}  {p}\n')
        subprocess.run(['zip', '-r', '-q', '-X', '-n', '.gguf:.whl:.zip:.exe', str(z), 'newineGPT'], cwd=stage.parent, check=True)
        print(f'{z.name}: {len(keep)} files, {z.stat().st_size / 1e6:.0f} MB')
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != 'SHA256SUMS.txt')
    with open(OUT / 'SHA256SUMS.txt', 'w', encoding='utf-8', newline='\n') as f:
        for p in files: f.write(f'{sha(p)}  {p.name}\n')
    print(open(OUT / 'SHA256SUMS.txt').read())


if __name__ == '__main__':
    main()
