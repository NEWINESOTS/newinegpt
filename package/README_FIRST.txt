newineGPT BETA - read this first
=================================

newineGPT answers questions about the New Wine / SOTS sermon archive. It runs entirely on your computer: no internet is needed once it
is set up, and nothing you type is sent anywhere. Every answer shows the archive passages it used, as [1], [2] ..., with the message
title, speaker, date and time.

This is a BETA. It can be wrong. Please ask real questions and mark each answer right or wrong (see "Rating answers" below).


WHAT YOU NEED
- A computer with Windows 10/11 (64-bit), a Mac (Apple chip or Intel, macOS 13.3 or newer), or 64-bit Linux (Ubuntu 22.04 or newer)
- At least 8 GB of memory (16 GB is better) and about 7 GB of free disk space


SETUP - WINDOWS
1. Make sure you have the model: the two files newinegpt-4b-v6-...-00001-of-00002.gguf and ...-00002-of-00002.gguf in Downloads.
2. Right-click the newineGPT zip file and choose "Extract All...", then click "Extract". (Do not run it from inside the zip.)
3. Open the extracted newineGPT folder and double-click START_Windows.bat
   - A blue box "Windows protected your PC"? Click "More info", then "Run anyway". (This beta is not signed by a company.)
   - "Missing Microsoft Visual C++ ... install it now?" Type y and press Enter, and allow it (it is included in the folder).
4. Type your name and press Enter. You can now ask questions.

SETUP - MAC
1. Make sure the two model files (...-00001-of-00002.gguf and ...-00002-of-00002.gguf) are in Downloads.
2. Double-click the newineGPT zip file to unzip it.
3. Open the newineGPT folder and double-click START_Mac.command
   - macOS says it "cannot be opened" or "cannot verify the developer"? Click Done. Open System Settings > Privacy & Security, scroll
     down, click "Open Anyway" next to START_Mac.command, and confirm. Then double-click START_Mac.command again.
     (Or, in Terminal, type  xattr -dr com.apple.quarantine  followed by a space, drag the newineGPT folder into the window, press Enter.)
4. Type your name and press Enter. You can now ask questions.

SETUP - LINUX
1. Put the two model files in ~/Downloads (or next to the newineGPT folder).
2. In a terminal:  cd ~/Downloads  then  unzip newineGPT-beta-linux-x64.zip  then  bash newineGPT/START_Linux.sh
3. Type your name and press Enter.


WHICH MODEL FILES?
- 16 GB of memory or more: the Q5 files (newinegpt-4b-v6-Q5_K_M-imat-...). Slightly better.
- 8 GB of memory: the Q4 files (newinegpt-4b-v6-Q4_K_M-imat-...). Smaller.
You need BOTH parts (00001 and 00002) of one of them. newineGPT finds them in Downloads or next to its folder by itself.
If you received the complete newineGPT folder (for example on a USB drive), the model is already inside it: skip step 1.


SPEED
- Macs with an Apple chip: a few seconds per answer.
- Windows, Linux and Intel Macs use the processor only: the first answer takes 1-3 minutes (the model loads), later answers about
  30-90 seconds. This is normal. Windows PCs with a graphics card can try START_Windows.bat --gpu (from a Command Prompt in the folder).


RATING ANSWERS (after each answer: one letter, then Enter)
  g = good (correct and useful)      p = partly right      w = wrong or misleading      Enter alone = skip
You can add a short note after the letter, e.g.   w wrong speaker   or   p date is wrong
Check important answers against the passage text shown under the answer. Judge only what the archive says.


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


WHEN YOU ARE DONE
Press Enter on an empty question to quit. Your questions, answers and ratings are saved in the folder  eval/human_use  inside the
newineGPT folder. Please zip that folder and send it back. Thank you!
