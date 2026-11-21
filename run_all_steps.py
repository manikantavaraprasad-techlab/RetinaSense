"""
Run every step scripts in order and report which ones produced output.

    python run_all_steps.py                    # uses sample_images/ and weights/
    python run_all_steps.py --input my.png     # same image for every per-image script

Works WITHOUT weights (classical fallback) — with weights/ filled it uses the trained networks.
Scripts that need extra data are only run if you pass --eval-root / --av-root.
"""
import subprocess, sys, time
from pathlib import Path
import argparse

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--open", action="store_true", help="open outputs/index.html when done"); ap.add_argument("--input"); ap.add_argument("--eval-root"); ap.add_argument("--av-root")
a = ap.parse_args()
inp = ["--input", a.input] if a.input else []
jobs = [("step_preprocessing_channels.py", inp), ("step_augmentation_examples.py", inp),
        ("step_segmentation_output.py", inp), ("step_clean_and_connect.py", inp),
        ("step_artery_vein.py", inp), ("step_biomarkers.py", inp),
        ("step_normal_vs_disease.py", []), ("full_pipeline_report.py", inp)]
if a.eval_root: jobs.insert(2, ("step_segmentation_results.py", ["--root", a.eval_root]))
if a.av_root: jobs.append(("step_av_results.py", ["--av-root", a.av_root]))
res = []
for s, args in jobs:
    print("\n" + "=" * 78 + f"\n  {s}\n" + "=" * 78, flush=True)
    t = time.time(); rc = subprocess.call([sys.executable, str(HERE / s), *args]); res.append((s, rc, time.time() - t))
print("\n" + "=" * 78 + "\n  SUMMARY\n" + "=" * 78)
for s, rc, dt in res: print(f"  {'OK    ' if rc == 0 else 'FAILED'}  {s:<40}{dt:6.1f}s")
print(f"\n  pictures are in {HERE/'outputs'}   ·   or run  python app.py  for the website")
