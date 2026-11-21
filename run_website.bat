@echo off
REM Double-click or run in the VS Code terminal.  Creates .venv on first run, installs light packages, starts the website.
cd /d "%~dp0"
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate
pip install -q -r requirements_light.txt
python app.py --open
pause
