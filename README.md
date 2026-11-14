# newineGPT

**Ask questions about the New Wine / SOTS sermon archive, and get answers that show exactly where in the messages they come from.**

newineGPT runs entirely on your own computer. It needs no internet once it is set up, nothing you type is sent anywhere, and it works on
**Windows, Mac (Apple chip or Intel) and Linux**. Every answer cites the archive passages it used, with the message title, speaker, date
and time, so you can check it.

> **This is a beta.** It can be wrong. If you are testing it, please mark each answer *Good*, *Partly* or *Wrong* (buttons under every
> answer), and send your results back (see [When you are done](#when-you-are-done)).

---

## Install it (no technical knowledge needed)

### Step 1: Download 3 files

Open the **[latest release](https://github.com/NEWINESOTS/newinegpt/releases/latest)** page and, under **Assets**, click to download:

**1. The app for your computer (one file):**

| Your computer | Download |
|---|---|
| Windows 10 or 11 | `newineGPT-beta-windows-x64.zip` |
| Mac with an Apple chip (M1, M2, M3, M4) | `newineGPT-beta-mac-apple-silicon.zip` |
| Mac with an Intel processor | `newineGPT-beta-mac-intel.zip` |
| Linux (Ubuntu 22.04 or newer) | `newineGPT-beta-linux-x64.zip` |

*Not sure which Mac you have?* Apple menu > **About This Mac**: "Chip: Apple M…" means Apple chip; "Processor: Intel…" means Intel.

**2. The model (both parts of one size):**

| Your computer's memory | Download both files |
|---|---|
| 16 GB or more | `newinegpt-4b-v6-Q5_K_M-imat-00001-of-00002.gguf` and `newinegpt-4b-v6-Q5_K_M-imat-00002-of-00002.gguf` |
| 8 GB | `newinegpt-4b-v6-Q4_K_M-imat-00001-of-00002.gguf` and `newinegpt-4b-v6-Q4_K_M-imat-00002-of-00002.gguf` |

*How much memory?* Windows: Settings > System > About > "Installed RAM". Mac: Apple menu > About This Mac > "Memory".
Leave all the downloads in your **Downloads** folder; newineGPT finds the model there by itself.

### Step 2: Unzip and start

<details open>
<summary><b>Windows</b></summary>

1. In your Downloads folder, **right-click** `newineGPT-beta-windows-x64.zip` and choose **Extract All…**, then **Extract**.
2. Open the extracted folder, then the **newineGPT** folder inside it, and **double-click `START_Windows.bat`**.
3. If a blue box says **"Windows protected your PC"**: click **More info**, then **Run anyway**. (The beta is not signed by a company yet.)
4. If it says a **Microsoft Visual C++** part is missing: type `y` and press Enter, and allow it. The installer is included.
5. Your web browser opens newineGPT. A black window also stays open: **leave it open** while you use newineGPT, and close it when you are done.
</details>

<details open>
<summary><b>Mac</b></summary>

1. In Downloads, **double-click** the newineGPT zip file. A **newineGPT** folder appears.
2. Open it and **double-click `START_Mac.command`**.
3. The first time, macOS says it **cannot verify** the file. Click **Done**, then open **System Settings > Privacy & Security**, scroll
   down and click **Open Anyway** next to START_Mac.command, and confirm. Double-click `START_Mac.command` again.
4. Your web browser opens newineGPT. A Terminal window also stays open: **leave it open** while you use newineGPT, and close it when you are done.
</details>

<details>
<summary><b>Linux</b></summary>

```bash
cd ~/Downloads
unzip newineGPT-beta-linux-x64.zip
bash newineGPT/START_Linux.sh
```
Your browser opens newineGPT; keep the terminal open while you use it.
</details>

### Step 3: Ask

Type your name (it only labels your review notes), then ask the way you normally would. For example:

- *What did Prophetess Fanny say about angels?*
- *What is the difference between gifts, ministries and operations?*
- *In "Healing Streams Decmeber 2019", where does the speaker talk about laughter?*
- *Who said "laughter is a weapon"?*

Click a small number like **5** in an answer to see the passage it came from. Your conversations are saved on your computer and listed on
the left, so they are still there next time.

**Next time:** just double-click the START file again.

---

## How fast is it?

| Computer | Time per answer |
|---|---|
| Mac with an Apple chip | about 3–10 seconds |
| Intel Mac (tested: iMac, Core i5-6500, 16 GB) | first answer about 1½ minutes (loading), then about 1 minute |
| Windows / Linux PC, processor only | about 30–90 seconds on a typical 4–8 core PC (about 2½ minutes on a 2-core machine) |
| Windows PC with a graphics card | try `START_Windows.bat --gpu` from a Command Prompt in the folder; often much faster |

While it works, the page shows "Searching the archive and writing an answer…" with a timer. That is normal.

## Something went wrong?

| What you see | What to do |
|---|---|
| "The model file was not found" | Download **both** parts of one model size (Step 1) into Downloads, or next to the newineGPT folder. |
| Windows: "Windows protected your PC" | Click **More info**, then **Run anyway**. |
| Windows: Visual C++ / MSVCP140.dll missing | Answer `y` when asked, or run `runtime\windows-x64\vc_redist.x64.exe` in the folder once. |
| Mac: "cannot be opened" / "damaged" | System Settings > Privacy & Security > **Open Anyway**. Or in Terminal: `xattr -dr com.apple.quarantine ` then drag the newineGPT folder in and press Enter. |
| The browser did not open | The black/Terminal window shows an address starting `http://127.0.0.1:` – copy it into your browser. |
| "newineGPT is not running" in the page | Its window was closed. Double-click the START file again. |
| Very slow | Normal without an Apple chip; close other big programs. With 8 GB of memory, use the Q4 model. |

## When you are done

Your questions, answers and ratings are saved in the **`eval/human_use`** folder inside the newineGPT folder. If you are a tester,
**zip that folder and send it back**. Thank you!

---

## What it can and cannot do (beta)

**Good at:** finding where a topic or phrase comes up (message, speaker, date and time); what a speaker taught on a topic, in their own
words; a message's scriptures and summary; who said a remembered phrase; saying "the archive doesn't cover that" instead of guessing.

**Known issues:**
- It sometimes explains a badly transcribed passage wrongly. Quotes are checked word for word against the transcript; its own wording is not.
- It sometimes says the archive doesn't cover something when a passage does.
- A few recordings list a doubtful speaker; answers mark these `[speaker unconfirmed: …]`.
- Message titles must be close to the archive's own spelling (which has typos, e.g. "Healing Streams Decmeber 2019").
- Transcripts are automatic speech-to-text, so names and Hebrew/Greek words are often misspelled.
- It does not yet use earlier questions in the same conversation ("what about in 2019?" needs the full question).

### Test results so far (all graded by AI; human testing is what this beta is for)

On 20 fresh questions from recordings no model was trained on, graded blind against the same passages (scores out of 40):

| | base Qwen3-4B | newineGPT v5 | **newineGPT v6 (this release)** |
|---|---|---|---|
| Explains the point | 35 | 12 | **25** |
| Faithful to the passages | 20 | 36 | **37** |
| Invented answers (of 20) | 3 | 1 | **0** |
| Wrongly said "not covered" (of 16 answerable) | 0 | 12 | **4** |

v6 did **not** pass every rule set before training (it explains less than the base model and still wrongly declines some questions), and
was chosen anyway for its faithfulness. Details: [docs/MODEL_CARD_v6.md](docs/MODEL_CARD_v6.md); platform tests: [docs/TESTING.md](docs/TESTING.md).

---

## How it works

1. **Search.** Your question is matched against 39,000 transcript passages from 912 recordings (keyword search + meaning search with the
   `bge-small-en-v1.5` model).
2. **Answer.** A 4-billion-parameter model (Qwen3-4B, fine-tuned on about 10,700 examples drawn from the archive) writes an answer from
   the passages it is given, citing them as [1], [2] ….
3. **Check.** Code checks every quote word for word against the transcript, every citation number, the speaker and the timestamp. If the
   model's answer cannot be verified, newineGPT shows the passages it found instead of an unverified answer.

Everything runs through [llama.cpp](https://github.com/ggml-org/llama.cpp) on your computer. The chat page is served only to your own
computer (127.0.0.1) and protected with a random key, so no other device or website can use it.

---

## For developers

| Folder | What is in it |
|---|---|
| `app/newinegpt/` | The app: `ask.py` (routing, prompts, answer checks), `retrieve.py` (search), `rank.py` (passage ranking, quote checks), `web.py` + `web_index.html` (chat window) |
| `package/` | Launcher (`start.py`), START scripts, tester guide (`README_FIRST.txt`) |
| `release/` | `build_beta.py` (offline package for all four systems), `make_gh_assets.py` (these release downloads) |
| `train/` | Prompt format, training-data builder (`make_sft.py`), RunPod LoRA training and pod-side GGUF export scripts, data cards |
| `eval/` | Model comparison, blind A/B grading, explanation experiments, hold-out audit, and the 109-question regression suite (`eval/golden/`) |
| `docs/` | Model card, v6 evaluation protocol, install-from-source guide |

**Run from source:** see [docs/INSTALL_FROM_SOURCE.md](docs/INSTALL_FROM_SOURCE.md). In short: Python 3.12 with `numpy`, `llama-server`
(llama.cpp b9580) on PATH, the archive at `pack/newine_archive.sqlite`, the search model at `models/bge-small-en-v1.5-f16.gguf` and the
model at `dist/models/newinegpt-4b-v6/`; then `cd app && python -m newinegpt.web` (chat window) or `python -m newinegpt.ask "question"`.
The archive, search model and model files are in the release downloads (the archive and search model are inside each app zip).

**Model:** Qwen3-4B + LoRA (r 16, lr 1e-4, 1.25 epochs, max length 2,304) on 10,704 grounded examples, trained on one A100; GGUF
Q5_K_M / Q4_K_M with an importance matrix. Validation loss 0.281.

## Licence: free, not for sale

newineGPT is **free to use, copy and share for non-commercial purposes. It may not be sold, monetized, or used in a paid product or service.**

- **Code** (this repository): [PolyForm Noncommercial License 1.0.0](LICENSE.md).
- **Model and archive data** (the release downloads): [CC BY-NC 4.0](LICENSE-MODEL-AND-DATA.md), with credit to NEWINESOTS / newineGPT.
- The sermon content belongs to its speakers and New Wine / SOTS.

Bundled third-party software keeps its own licence (unmodified official builds): llama.cpp (MIT), CPython 3.12 from python-build-standalone
(PSF), numpy (BSD-3), Microsoft Visual C++ Redistributable (Microsoft licence terms), libgomp from Ubuntu (GCC Runtime Library Exception).
Models: Qwen3-4B (Apache-2.0), bge-small-en-v1.5 (MIT). See `LICENSES.txt` in each download.
