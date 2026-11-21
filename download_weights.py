"""
download_weights.py — Fetch trained checkpoints (Kaggle notebook outputs) and the
datasets with the Kaggle CLI.

One-time setup:  pip install kaggle   and put your API token in ~/.kaggle/kaggle.json
(Kaggle → Settings → API → Create New Token).

    # trained weights: give the slug of each notebook that produced .pt files
    python download_weights.py --kernel YOUR_USER/attention-u-net \
                               --kernel YOUR_USER/v5-unet-plusplus \
                               --kernel YOUR_USER/unetplusplus-with-attention-gates \
                               --kernel YOUR_USER/full-end-to-end

    # datasets used by the code (links are in the .py files / on )
    python download_weights.py --datasets seg odir driveav

Everything lands in ./weights (kernel outputs) or ./data (datasets). The
.pt files are recognised by CONTENT (assets.py), so file names do not matter.
You can instead download the "Output" tab of each Kaggle notebook by hand in the
browser and drop the .pt files into ./weights/ — same result.
"""
import argparse, subprocess, sys
from pathlib import Path

DATASETS = {
    "seg":     ("shakibabsar42/retinal-vessel-fundus-dataset-collection", "data/seg"),
    "odir":    ("andrewmvd/ocular-disease-recognition-odir5k",            "data/odir"),
    "driveav": ("mounishkumar003/drive1-av",                              "data/drive_av"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--kernel", action="append", default=[], help="owner/notebook-slug (repeatable)")
ap.add_argument("--datasets", nargs="*", default=[], choices=list(DATASETS))
a = ap.parse_args()

def run(cmd):
    print(">", " ".join(cmd))
    if subprocess.call(cmd) != 0:
        sys.exit("kaggle CLI failed (is it installed, is ~/.kaggle/kaggle.json present, "
                 "and do you have access to that notebook/dataset?)")

for k in a.kernel:
    Path("weights").mkdir(exist_ok=True)
    run(["kaggle", "kernels", "output", k, "-p", "weights"])
for d in a.datasets:
    slug, dest = DATASETS[d]
    Path(dest).mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "download", "-d", slug, "-p", dest, "--unzip"])
if not a.kernel and not a.datasets:
    ap.print_help()
