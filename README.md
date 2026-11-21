# RetinaSense — retinal vessel · artery/vein · biomarker · disease-screening project

## Run the website
```bash
pip install -r requirements_light.txt      # first time only (no PyTorch needed)
python app.py --open                       # http://127.0.0.1:5000
```
(Windows: `run_website.bat` · Linux/macOS: `bash run_website.sh`)

Upload a fundus photo (or pick a sample) and the site reveals, step by step: image cleaning (green · CLAHE · Retinex) →
vessel segmentation → clean-and-connect (threshold → denoise → bridge → skeleton → graph) → artery/vein → six biomarkers vs
reference ranges (animated) → disease screening (Normal / Glaucoma / Diabetic / Hypertensive) → complete output + `results.json`.
The page also shows the segmentation benchmark of the three networks (UNet++, Attention U-Net, UNet++ + SCSE), the artery–vein
results, the normal-vs-disease biomarker chart and the complete UNet++ + SCSE output for a normal and a disease retina.
Benchmark numbers live in `data/results.json`; reference images in `static/reference/`.

## Engines
Copy the trained `.pt` checkpoints into `weights/` and install `requirements.txt` (adds PyTorch) to use the trained networks;
otherwise a classical ridge-filter fallback runs and the badge on the page says so. Disease screening (`screening.py`) is
rule-based on the six biomarkers — a decision-support aid, not a diagnosis.

## Scripts
`python run_all_steps.py` runs every `step_*.py` script (preprocessing, augmentation, segmentation, clean-and-connect,
artery/vein, biomarkers, normal vs disease). Training and evaluation scripts are unchanged.


## What's new (v2)
* **Disease prediction from biomarker values** (`screening.py`): Normal · Glaucoma · Diabetes · High blood pressure, with match %, strength, per-biomarker evidence and a plain-language conclusion.
* **One separate report per image** (multi-upload): Prediction · Output images · Biomarkers · 9 graphs · Trust & quality. 15 animated chart types (donut, gauges, radar, bullet bars, stacked evidence, diverging bars, area, line, polar rose, heat-map, scatter, butterfly, histogram, funnel, ring).
* **Artery Dice vs Vein Dice per image** for Attention U-Net and UNet++ + SCSE (scatter, dumbbell, trend line, difference bars, per-image inspector). Data: `data/per_image_av_dice.csv` — currently digitised from your saved figure (about ±0.01); replace it with exact numbers and the page updates.
* **Trust features**: 9 automatic image/measurement quality checks, SHA-256 + timestamp provenance, honest A/V-reliability warnings, results.json + print-to-PDF report.
* `python validate_screening.py labelled/` measures real accuracy on labelled images (folders normal / glaucoma / diabetes / hbp).
