#!/usr/bin/env bash
# Linux / macOS:  bash run_website.sh
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements_light.txt
python app.py --open
