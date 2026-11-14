#!/bin/bash
# newineGPT - New Wine School of the Spirit - beta (offline). Run: ./START_Linux.sh   (options: --q4 smaller model, --check only test the setup)
cd "$(dirname "$0")"
printf '\033]0;newineGPT - New Wine School of the Spirit\007'   # window title
exec runtime/linux-x64/python/bin/python3 -X utf8 start.py "$@"
