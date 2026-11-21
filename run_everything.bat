@echo off
REM Runs EVERY script, then opens outputs\index.html (all pictures, graphs and numbers on one page)
cd /d "%~dp0"
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate
pip install -q -r requirements_light.txt
python run_all_steps.py --open
pause
