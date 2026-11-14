#!/bin/bash
# newineGPT - New Wine School of the Spirit - beta (offline). Double-click to start (first time: right-click > Open). Works on Apple-chip and Intel Macs.
cd "$(dirname "$0")"
printf '\033]0;newineGPT - New Wine School of the Spirit\007'   # window title
# Files downloaded with a browser carry macOS's "downloaded from the internet" flag, and the bundled programs (Python, llama.cpp) are not
# notarised by Apple, so macOS would block them. You approved this start file; it now clears that flag on this folder's own files only.
if xattr -r -l . 2>/dev/null | grep -q com.apple.quarantine; then echo "Allowing this folder's programs to run (clearing macOS's download flag on this folder only)..."; xattr -dr com.apple.quarantine . 2>/dev/null; fi
if [ "$(uname -m)" = "arm64" ]; then RT=runtime/macos-arm64; else RT=runtime/macos-x64; fi
"$RT/python/bin/python3" -X utf8 start.py "$@"
