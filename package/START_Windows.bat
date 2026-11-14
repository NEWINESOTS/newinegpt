@echo off
rem newineGPT - New Wine School of the Spirit - beta (offline). Double-click to start. Options: START_Windows.bat --gpu  (use the graphics card)  --q4  (smaller model)
title newineGPT - New Wine School of the Spirit
color 5F
cd /d "%~dp0"
"%~dp0runtime\windows-x64\python\python.exe" -X utf8 "%~dp0start.py" %*
pause
