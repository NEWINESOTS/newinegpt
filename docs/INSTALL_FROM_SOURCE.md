# Installing newineGPT (local, offline)

newineGPT answers questions about the New Wine / SOTS sermon archive from a computer with no internet connection.
It has three parts: the archive (a SQLite file), a small search model, and a 4B language model run by llama.cpp.

| Platform | Status |
|---|---|
| macOS (Apple silicon) | Tested, 12 Nov 2026 |
| Windows 10/11 | **Unverified** — not yet tested |
| Linux (x86-64) | **Unverified** — not yet tested |

**Which model?** v6 is the **default** (chosen 14 Nov 2026); v5 is kept as the baseline. v6 explains teaching in its own words and declines
far less often than v5 (exploratory test of 20 fresh questions: explains 25 vs 12 of 40, wrong declines 4 vs 12, no invented answers), but it
did not pass every rule fixed before training, and it can paraphrase a garbled transcript wrongly: check important answers against the cited
passage. Details: `dist/models/newinegpt-4b-v6/README.md`. Choose a model with `NEWINEGPT_MODEL` (below).

## 1. What to copy

Copy the whole project folder (any location; paths are found relative to it):

| Path | Size | What it is |
|---|---|---|
| `app/newinegpt/` | small | the app |
| `train/prompt.py` | small | the prompt format the models were trained on (the app imports it) |
| `pack/newine_archive.sqlite` | 306 MB | the archive: transcripts, summaries, catalog |
| `models/bge-small-en-v1.5/` | 128 MB | search model (runs on CPU, offline) |
| `dist/models/newinegpt-4b-v6/newinegpt-4b-v6-Q5_K_M-imat.gguf` | 2.9 GB | v6 (default), recommended quantisation for 8 GB+ RAM |
| `dist/models/newinegpt-4b-v6/newinegpt-4b-v6-Q4_K_M-imat.gguf` | 2.5 GB | v6, for smaller machines |
| `dist/models/newinegpt-4b-v5/newinegpt-4b-v5-Q5_K_M-imat.gguf` | 2.9 GB | v5 (baseline) |
| `dist/models/newinegpt-4b-v5/newinegpt-4b-v5-Q4_K_M-imat.gguf` | 2.5 GB | v5, for smaller machines |
| `dist/models/newinegpt-4b-v3/newinegpt-4b-v3-Q5_K_M-imat.gguf` | 2.9 GB | v3 |

Check the model files after copying: `shasum -a 256 -c SHA256SUMS` (macOS/Linux) or `certutil -hashfile <file> SHA256` (Windows),
compared with the `SHA256SUMS` file in the model's folder.

## 2. Install llama.cpp (`llama-server`)

The app starts `llama-server` itself; it must be on the PATH.

- **macOS:** `brew install llama.cpp`
- **Windows (unverified):** download a release zip from https://github.com/ggml-org/llama.cpp/releases (CPU build, or the CUDA build
  for an NVIDIA GPU), unzip it, and add the folder that contains `llama-server.exe` to PATH.
- **Linux (unverified):** download a release from the same page, or build it: `cmake -B build && cmake --build build --target llama-server -j`.

Check: `llama-server --version`

## 3. Install Python packages

Python 3.11 or newer.

```bash
cd <project folder>
python3 -m venv .venv
.venv/bin/pip install -r app/requirements.txt
```

Windows (unverified): `py -m venv .venv` then `.venv\Scripts\pip install -r app\requirements.txt`.

## 4. Ask a question

```bash
cd <project folder>/app
NEWINEGPT_MODEL=../dist/models/newinegpt-4b-v6/newinegpt-4b-v6-Q5_K_M-imat.gguf HF_HUB_OFFLINE=1 ../.venv/bin/python -m newinegpt.ask "What did Prophetess Fanny say about angels?"
```

Windows PowerShell (unverified):

```powershell
cd <project folder>\app
$env:NEWINEGPT_MODEL = "..\dist\models\newinegpt-4b-v6\newinegpt-4b-v6-Q5_K_M-imat.gguf"; $env:HF_HUB_OFFLINE = "1"
..\.venv\Scripts\python -m newinegpt.ask "What did Prophetess Fanny say about angels?"
```

Many questions from a file: `python -m newinegpt.ask --file questions.txt --out answers.jsonl`.

The first question takes 10-30 seconds while the model loads; later ones are faster. The app starts `llama-server` on port 8094
and refuses to use a server already on that port that serves a different model.

## Settings

| Variable | Meaning |
|---|---|
| `NEWINEGPT_MODEL` | path to the GGUF model file to use |
| `NEWINEGPT_SERVER` | URL of a llama-server that is already running (then no server is started) |
| `NEWINEGPT_EMBED` | path to the search model folder, if not `models/bge-small-en-v1.5` |
| `HF_HUB_OFFLINE=1` | never try to download anything |

## What answers look like

Every answer cites its passages as `[n]` with the message title, speaker, date and timestamp. Some answers are written by code, not
the model (lists, scripture lists, "I can't find ...", who-said-it, and quotes when the model's answer cannot be verified); the
`--json` output shows which (`source`).

## Recording a review session (for someone who knows the archive)

```bash
cd <project folder>/app
HF_HUB_OFFLINE=1 ../.venv/bin/python -m newinegpt.ask --session your_name
```

Ask questions the way you normally would. After each answer you can rate it with one key: `g` good, `p` partly right, `w` wrong or
misleading, or Enter to skip; a note can follow the letter (`p wrong date`). Every question and answer is saved the moment it is shown in
`eval/human_use/<date>_<name>/` (`first_answers.jsonl`, `ratings.jsonl`, and `session.json` with the model and app versions), so the first
results are kept before anything is fixed. Nothing needs to be prepared beforehand.

## Speakers marked "unconfirmed"

Some recordings are listed under one speaker while the title names another presenter, or the recording plays someone else's video. Answers
show these as `Babs Adewunmi [speaker unconfirmed: title names Mrs Nuhu]`. A reviewer fills `confirmed_speaker` in
`app/newinegpt/disputed_speakers.csv` (priority 1 first); the app then uses that name.
