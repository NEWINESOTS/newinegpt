newineGPT  ·  New Wine · School of the Spirit  (BETA) - read this first
=========================================================================

newineGPT answers questions about the New Wine School of the Spirit (SOTS) message archive. It runs entirely on your computer: no
internet is needed once it is set up, and nothing you type is sent anywhere. Every answer shows the archive passages it used, as
[1], [2] ..., with the message title, speaker, date and time.

This is a BETA. It can be wrong. Please ask real questions and mark each answer Good, Partly or Wrong (buttons under every answer).


WHAT YOU NEED
- A computer with Windows 10/11 (64-bit), a Mac (Apple chip or Intel, macOS 13.3 or newer), or 64-bit Linux (Ubuntu 22.04 or newer)
- At least 8 GB of memory (16 GB is better) and about 7 GB of free disk space


SETUP - WINDOWS
1. Make sure you have the model: the two files newinegpt-4b-v6-...-00001-of-00002.gguf and ...-00002-of-00002.gguf in Downloads.
2. Right-click the newineGPT zip file and choose "Extract All...", then click "Extract". (Do not run it from inside the zip.)
3. Open the extracted newineGPT folder and double-click START_Windows.bat
   - A blue box "Windows protected your PC"? Click "More info", then "Run anyway". (This beta is not signed by a company.)
   - "Missing Microsoft Visual C++ ... install it now?" Type y and press Enter, and allow it (it is included in the folder).
4. Your web browser opens newineGPT. A purple window also opens: leave it open while you use newineGPT, and close it when you are done.

SETUP - MAC
1. Make sure the two model files (...-00001-of-00002.gguf and ...-00002-of-00002.gguf) are in Downloads.
2. Double-click the newineGPT zip file to unzip it.
3. Open the newineGPT folder and double-click START_Mac.command
   - macOS says it "cannot be opened" or "cannot verify the developer"? Click Done. Open System Settings > Privacy & Security, scroll
     down, click "Open Anyway" next to START_Mac.command, and confirm. Then double-click START_Mac.command again.
     (Or, in Terminal, type  xattr -dr com.apple.quarantine  followed by a space, drag the newineGPT folder into the window, press Enter.)
4. Your web browser opens newineGPT. A Terminal window also opens: leave it open while you use newineGPT, and close it when you are done.

SETUP - LINUX
1. Put the two model files in ~/Downloads (or next to the newineGPT folder).
2. In a terminal:  cd ~/Downloads  then  unzip newineGPT-beta-linux-x64.zip  then  bash newineGPT/START_Linux.sh
3. Your web browser opens newineGPT. Keep the terminal open while you use it.

If the browser does not open by itself, the window shows an address starting  http://127.0.0.1:  - copy it into your browser.
Next time, just double-click the START file again. Your conversations are listed on the left of the page and are kept on your computer.


WHICH MODEL FILES?
- 16 GB of memory or more: the Q5 files (newinegpt-4b-v6-Q5_K_M-imat-...). Slightly better.
- 8 GB of memory: the Q4 files (newinegpt-4b-v6-Q4_K_M-imat-...). Smaller.
You need BOTH parts (00001 and 00002) of one of them. newineGPT finds them in Downloads or next to its folder by itself.
If you received the complete newineGPT folder (for example on a USB drive), the model is already inside it: skip step 1.


SPEED
- Macs with an Apple chip: a few seconds per answer.
- Windows, Linux and Intel Macs use the processor only: the first answer takes 1-3 minutes (the model loads), later answers about
  30-120 seconds. The page shows a timer while it works. This is normal. Windows PCs with a graphics card can try
  START_Windows.bat --gpu (from a Command Prompt in the folder).


RATING ANSWERS
Under every answer: click Good (correct and useful), Partly (partly right) or Wrong (wrong or misleading). You can add a short note,
e.g. "wrong speaker" or "date is wrong". Click a number like [5] in an answer to see the passage it came from, and check important
answers against it. Judge only what the archive says.


GOOD THINGS TO TRY
- What did <speaker> teach about <topic>?                 - Why ... ? / What is the difference between ... ?
- In "<message title>", where does the speaker talk about <thing>?
- What scriptures are used in "<message title>"?          - Who said "<a phrase you remember>"?
- Something the archive probably does NOT cover (it should say so instead of guessing)


KNOWN ISSUES (please still rate them when you see them)
- It sometimes explains a badly transcribed passage wrongly. Quotes are checked word for word; its own wording is not.
- It sometimes says the archive doesn't cover something when a passage does.
- A few recordings may list the wrong speaker; answers mark these "[speaker unconfirmed: ...]".
- Titles must be close to the archive's spelling (which has typos, e.g. "Healing Streams Decmeber 2019").
- Transcripts are automatic speech-to-text, so names and Hebrew/Greek words are often misspelled.
- It does not yet use earlier questions in the same conversation ("what about in 2019?" needs the full question).


WHEN YOU ARE DONE
Close the purple / Terminal window to stop newineGPT. Your questions, answers and ratings are saved in the folder  eval/human_use
inside the newineGPT folder. Please zip that folder and send it back. Thank you!
