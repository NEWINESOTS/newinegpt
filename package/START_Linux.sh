#!/bin/bash
# newineGPT beta (offline). Run: ./START_Linux.sh   (options: --q4 smaller model, --check only test the setup)
cd "$(dirname "$0")"
exec runtime/linux-x64/python/bin/python3 -X utf8 start.py "$@"
