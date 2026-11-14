"""Build the offline newineGPT beta folder for Windows x64, Linux x64, Intel Mac and Apple-chip Mac (14 Nov 2026).
  python3 release/build_beta.py   ->  release/newineGPT_beta_20261114/  (+ SHA256SUMS.txt over every file)
Inputs: the app (app/newinegpt, train/prompt.py), the archive (pack/newine_archive.sqlite), the GGUF search model
(models/bge-small-en-v1.5-f16.gguf), v6 Q5 + Q4 (dist/models/newinegpt-4b-v6), and the verified downloads in release/downloads/
(MANIFEST.json: llama.cpp b9580, portable CPython 3.12.16 (python-build-standalone 20261026), numpy 2.5.4 wheels, vc_redist.x64.exe).
Big files are APFS clones (cp -c): no extra disk until the folder is copied elsewhere. Refuses to overwrite an existing build."""
import hashlib, json, shutil, subprocess, sys, tarfile, zipfile
from pathlib import Path
R = Path.home() / 'newine_local'; DL = R / 'release' / 'downloads'; FILES = R / 'release' / 'beta_files'
OUT = R / 'release' / 'newineGPT_beta_20261114'
PLAT = {   # system: (python archive, numpy wheel, llama archive, site-packages path inside python/)
    'windows-x64': ('cpython-3.12.16+20261026-x86_64-pc-windows-msvc-install_only.tar.gz', 'numpy-2.5.4-cp312-cp312-win_amd64.whl',
                    'llama-b9580-bin-win-cpu-x64.zip', 'Lib/site-packages'),
    'linux-x64': ('cpython-3.12.16+20261026-x86_64-unknown-linux-gnu-install_only.tar.gz',
                  'numpy-2.5.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl', 'llama-b9580-bin-ubuntu-x64.tar.gz', 'lib/python3.12/site-packages'),
    'macos-x64': ('cpython-3.12.16+20261026-x86_64-apple-darwin-install_only.tar.gz', 'numpy-2.5.4-cp312-cp312-macosx_10_13_x86_64.whl',
                  'llama-b9580-bin-macos-x64.tar.gz', 'lib/python3.12/site-packages'),
    'macos-arm64': ('cpython-3.12.16+20261026-aarch64-apple-darwin-install_only.tar.gz', 'numpy-2.5.4-cp312-cp312-macosx_11_0_arm64.whl',
                    'llama-b9580-bin-macos-arm64.tar.gz', 'lib/python3.12/site-packages'),
}


def sha(p, n=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(n), b''): h.update(b)
    return h.hexdigest()


def clone(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['cp', '-c', '-p', str(src), str(dst)], check=True)


def unpack_llama(arc, dst):
    tmp = dst.parent / '_tmp'; tmp.mkdir(parents=True)
    if arc.suffix == '.zip':
        with zipfile.ZipFile(arc) as z: z.extractall(tmp)
        root = tmp
    else:
        with tarfile.open(arc) as t: t.extractall(tmp, filter='tar')     # keeps symlinks and exec bits, refuses paths outside tmp
        root = next(tmp.iterdir())
    root.rename(dst)
    if tmp.exists(): tmp.rmdir()


def main():
    if OUT.exists(): sys.exit(f'REFUSED: {OUT} exists')
    man = {n: d for n, _, d in json.load(open(DL / 'MANIFEST.json'))}
    for n, d in man.items():
        assert sha(DL / n) == d, f'download changed: {n}'
    OUT.mkdir(parents=True)
    # app + prompt + archive + search model + v6 model files
    for f in ('__init__.py', 'ask.py', 'rank.py', 'retrieve.py', 'synonyms.json', 'disputed_speakers.csv', 'web.py', 'web_index.html'):
        clone(R / 'app' / 'newinegpt' / f, OUT / 'app' / 'newinegpt' / f)
    clone(R / 'train' / 'prompt.py', OUT / 'train' / 'prompt.py')
    clone(R / 'pack' / 'newine_archive.sqlite', OUT / 'pack' / 'newine_archive.sqlite')
    clone(R / 'models' / 'bge-small-en-v1.5-f16.gguf', OUT / 'models' / 'bge-small-en-v1.5-f16.gguf')
    M = R / 'dist' / 'models' / 'newinegpt-4b-v6'
    for f in ('newinegpt-4b-v6-Q5_K_M-imat.gguf', 'newinegpt-4b-v6-Q4_K_M-imat.gguf', 'README.md'):
        clone(M / f, OUT / 'dist' / 'models' / 'newinegpt-4b-v6' / f)
    want = {l.split()[1]: l.split()[0] for l in open(M / 'SHA256SUMS')}
    for f in ('newinegpt-4b-v6-Q5_K_M-imat.gguf', 'newinegpt-4b-v6-Q4_K_M-imat.gguf'):
        assert sha(OUT / 'dist' / 'models' / 'newinegpt-4b-v6' / f) == want[f], f
    (OUT / 'eval' / 'human_use').mkdir(parents=True)
    # runtimes
    for name, (py, whl, lla, site) in PLAT.items():
        rt = OUT / 'runtime' / name; rt.mkdir(parents=True)
        with tarfile.open(DL / py) as t:   # -> rt/python. Skip share/terminfo (curses terminal database, unused; its names differ
            t.extractall(rt, members=[m for m in t.getmembers() if '/share/terminfo/' not in m.name], filter='tar')  # only by case)
        with zipfile.ZipFile(DL / whl) as z: z.extractall(rt / 'python' / site)
        unpack_llama(DL / lla, rt / 'llama')
        if name == 'windows-x64':
            unpack_llama(DL / 'llama-b9580-bin-win-vulkan-x64.zip', rt / 'llama-vulkan')
            shutil.copy2(DL / 'vc_redist.x64.exe', rt / 'vc_redist.x64.exe')
        if name == 'linux-x64':   # GCC OpenMP runtime: absent from minimal Ubuntu 22.04 (container test, 14 Nov); found via the $ORIGIN rpath
            deb = DL / 'libgomp' / 'libgomp1_12.3.0-1ubuntu1~22.04.3_amd64.deb'
            assert sha(deb) == '870c27299185a5dd4accad3b15bf82a7409fd7073cccaa8025875307da4d0ce2'
            shutil.copy2(DL / 'libgomp' / 'x' / 'usr' / 'lib' / 'x86_64-linux-gnu' / 'libgomp.so.1.0.0', rt / 'llama' / 'libgomp.so.1')
        if not name.startswith('windows'):
            for p in (rt / 'python' / 'lib' / 'python3.12' / 'site-packages').rglob('*.so'): p.chmod(0o755)
    # Symlinks break when the folder is copied through a FAT/exFAT USB stick or a Windows zip tool: replace each with a real copy.
    for p in sorted((OUT / 'runtime').rglob('*'), key=lambda x: -len(x.parts)):
        if p.is_symlink():
            t = p.resolve(); p.unlink()
            if t.is_dir(): shutil.copytree(t, p, symlinks=False)
            else: shutil.copy2(t, p)
    for f in ('start.py', 'START_Windows.bat', 'START_Mac.command', 'START_Linux.sh', 'README_FIRST.txt'):
        shutil.copy2(FILES / f, OUT / f)
    (OUT / 'LICENSES.txt').write_text(
        'Bundled third-party software (unmodified official builds):\n'
        '- llama.cpp b9580 (MIT), github.com/ggml-org/llama.cpp: runtime/*/llama*/ (LICENSE inside each)\n'
        '- CPython 3.12.16 (PSF License), portable builds from github.com/astral-sh/python-build-standalone release 20261026: runtime/*/python/\n'
        '- numpy 2.5.4 (BSD-3-Clause): runtime/*/python/.../site-packages/numpy\n'
        '- libgomp.so.1 from Ubuntu libgomp1 12.3.0 (GCC Runtime Library Exception): runtime/linux-x64/llama/libgomp.so.1\n'
        '- Microsoft Visual C++ Redistributable (Microsoft license terms): runtime/windows-x64/vc_redist.x64.exe\n'
        '- Qwen3-4B (Apache-2.0), base of the newineGPT model; bge-small-en-v1.5 (MIT), search model.\n', encoding='utf-8')
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())
    with open(OUT / 'SHA256SUMS.txt', 'w', encoding='utf-8') as f:
        for p in files: f.write(f'{sha(p)}  {p.relative_to(OUT).as_posix()}\n')
    tot = sum(p.stat().st_size for p in files)
    print(f'{len(files)} files, {tot / 1e9:.2f} GB -> {OUT}')


if __name__ == '__main__':
    main()
