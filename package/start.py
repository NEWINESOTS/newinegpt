"""newineGPT beta launcher (offline). Started by START_Windows.bat / START_Mac.command / START_Linux.sh with the Python bundled for that
computer. Picks the bundled llama.cpp, the model file that fits this computer's memory, checks that everything runs, then starts a
recorded review session (answers and your ratings are saved in eval/human_use/).

  start            ->  opens the newineGPT chat window in your web browser (the default; nothing to type in this window)
  start --terminal ->  the older text-only session in this window
  start "question" ->  answers one question here and exits
  start --q4 / --q5 / --gpu (Windows: use the graphics card through Vulkan) / --check (only run the checks)
"""
import ctypes, os, platform, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'dist' / 'models' / 'newinegpt-4b-v6'


def system():
    m = platform.machine().lower()
    if sys.platform.startswith('win'): return 'windows-x64'
    if sys.platform == 'darwin': return 'macos-arm64' if m in ('arm64', 'aarch64') else 'macos-x64'
    return 'linux-x64'


def ram_gb():
    try:
        if sys.platform.startswith('win'):
            class M(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                            ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            s = M(); s.dwLength = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)); return s.ullTotalPhys / 2**30
        if sys.platform == 'darwin':
            return int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True).stdout) / 2**30
        return int(next(l for l in open('/proc/meminfo') if l.startswith('MemTotal')).split()[1]) / 2**20
    except Exception:
        return 0.0


def find_model(q):
    """The v6 model file for size q ('Q4' or 'Q5'): one file, or the two parts the GitHub release is split into (llama.cpp loads the
    second part itself when given the first). Looked for in the package, next to the unzipped folder, one level up, and in Downloads."""
    names = [f'newinegpt-4b-v6-{q}_K_M-imat.gguf', f'newinegpt-4b-v6-{q}_K_M-imat-00001-of-00002.gguf']
    for d in (MODELS, ROOT, ROOT.parent, ROOT.parent.parent, Path.home() / 'Downloads'):
        for n in names:
            f = d / n
            if f.exists() and ('-of-' not in n or (d / n.replace('00001-of', '00002-of')).exists()):
                return f
    return None


def fail(msg):
    print('\n' + msg + '\n'); input('Press Enter to close.') if sys.stdin.isatty() else None; sys.exit(1)


def main(argv):
    sys.stdout.reconfigure(line_buffering=True)   # our lines before the app's output
    sysname = system(); gpu = '--gpu' in argv
    rt = ROOT / 'runtime' / sysname
    llama = rt / ('llama-vulkan' if gpu and sysname == 'windows-x64' else 'llama')
    server = llama / ('llama-server.exe' if sysname == 'windows-x64' else 'llama-server')
    ram = ram_gb()
    q = 'Q4' if '--q4' in argv else 'Q5' if '--q5' in argv else ('Q5' if ram >= 12 else 'Q4')
    model = find_model(q)
    if model is None and '--q4' not in argv and '--q5' not in argv:       # the other size was downloaded instead: use it
        other = 'Q4' if q == 'Q5' else 'Q5'
        if find_model(other): q, model = other, find_model(other)
    terminal, nobrowser = '--terminal' in argv, '--no-browser' in argv
    argv = [a for a in argv if a not in ('--q4', '--q5', '--gpu', '--check', '--terminal', '--no-browser')] if '--check' not in argv else ['--check']

    print('\n  newineGPT  ·  New Wine · School of the Spirit  (beta)\n  ' + '=' * 54)
    print(f'  {sysname} · {ram:.0f} GB memory · model v6 {q}' + (' · graphics card (Vulkan)' if gpu else '') + '\n')
    if not server.exists(): fail(f'Missing {server}. Download and unzip the newineGPT file for this computer again.')
    if sysname != 'windows-x64':      # a copy through a USB stick or zip tool can drop the 'executable' flag
        for p in [server] + [x for x in llama.iterdir() if x.suffix in ('.so', '.dylib') or '.so.' in x.name]:
            try: p.chmod(p.stat().st_mode | 0o755)
            except OSError: pass
    if model is None:
        fail(f'The model file was not found.\n\nDownload BOTH parts of the model from the newineGPT release page:\n'
             f'  newinegpt-4b-v6-{q}_K_M-imat-00001-of-00002.gguf\n  newinegpt-4b-v6-{q}_K_M-imat-00002-of-00002.gguf\n'
             f'and put them in your Downloads folder, or next to this folder:\n  {ROOT.parent}\nThen start again.')
    if sysname == 'windows-x64':
        try: ctypes.WinDLL('msvcp140.dll')
        except OSError:
            vc = ROOT / 'runtime' / 'windows-x64' / 'vc_redist.x64.exe'
            print('This PC is missing the Microsoft Visual C++ Redistributable (MSVCP140.dll), which llama.cpp needs.')
            if input(f'Install it now from the package ({vc.name}, asks for administrator rights)? [y/n] ').strip().lower().startswith('y'):
                subprocess.run([str(vc), '/passive', '/norestart'])
                try: ctypes.WinDLL('msvcp140.dll')
                except OSError: fail('Still missing. Restart the PC and run START_Windows.bat again.')
            else: fail(f'Run {vc} once, then start again.')
    try:
        v = subprocess.run([str(server), '--version'], capture_output=True, text=True, timeout=60)
        if v.returncode != 0: raise OSError((v.stderr or v.stdout).strip()[-400:])
    except Exception as e:
        hint = ''
        if sys.platform == 'darwin':
            hint = ('\nmacOS may be blocking the downloaded programs. In Terminal run:\n'
                    f'  xattr -dr com.apple.quarantine "{ROOT}"\nthen start again.')
        fail(f'llama.cpp could not start: {e}{hint}')
    if '--check' in argv:
        size = sum(f.stat().st_size for f in model.parent.glob(model.name.replace('-00001-of-00002', '*')))
        print(f'Checks passed: {server.name} runs; model found: {model} ({size / 1e9:.1f} GB).'); return

    os.environ.update(NEWINEGPT_MODEL=str(model), NEWINEGPT_LLAMA_SERVER=str(server), HF_HUB_OFFLINE='1', PYTHONUTF8='1',
                      NEWINEGPT_SHOW_PASSAGES='1', NEWINEGPT_EMBED_GGUF=str(ROOT / 'models' / 'bge-small-en-v1.5-f16.gguf'))
    if sysname != 'macos-arm64' and not gpu: os.environ['NEWINEGPT_NGL'] = '0'    # CPU builds; Apple Macs use the graphics chip (Metal)
    (ROOT / 'eval' / 'human_use').mkdir(parents=True, exist_ok=True)
    if not argv and not terminal:               # the chat window in the browser (default)
        os.chdir(ROOT / 'app'); sys.path.insert(0, str(ROOT / 'app'))
        from newinegpt import web
        try: web.main(['--no-browser'] if nobrowser else [])
        except KeyboardInterrupt: pass
        return
    if argv:
        args = [' '.join(argv)]
    else:
        print('\nAnswers come from the New Wine / SOTS archive and cite their passages [1], [2] ... with title, speaker, date and time.')
        print('Check important answers against the passage shown. ' + ('The first answer takes about a minute while the model loads; '
              'later answers take 20-90 seconds on this computer.' if sysname != 'macos-arm64' else 'The first answer takes about 20 seconds.'))
        name = input('\nYour name (for the review log): ').strip() or 'tester'
        args = ['--session', name]
    # run the app in this process (settings above are read when it is imported); a child process would lose typed-ahead input
    os.chdir(ROOT / 'app'); sys.path.insert(0, str(ROOT / 'app'))
    from newinegpt import ask
    try: ask.main(args)
    except KeyboardInterrupt: pass
    if not argv: print(f'\nYour answers and ratings are saved in {ROOT / "eval" / "human_use"}. Please send that folder back.')


if __name__ == '__main__':
    main(sys.argv[1:])
