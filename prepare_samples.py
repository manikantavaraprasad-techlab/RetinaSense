"""
prepare_samples.py — Build the  samples/normal  and  samples/disease  folders
from REAL labelled fundus photographs, for the normal-vs-disease biomarker
comparison:

    python prepare_samples.py --odir-root /path/to/odir5k --n 20
    python run_analysis.py --input samples

normal   : ODIR-5K eyes whose keywords are exactly "normal fundus"
disease  : ODIR-5K eyes labelled Diabetic retinopathy (default) and, with
           --include-hypertensive, hypertensive retinopathy.

Selection is random with a fixed seed (reproducible) and a manifest.csv with the
ground-truth label of every copied file is written.

You can also copy your own images into samples/normal and samples/disease by hand
(e.g. DRIVE test images as 'normal', STARE images with pathology as 'disease').
"""
import argparse, csv, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--odir-root", required=True)
ap.add_argument("--n", type=int, default=20, help="images per group")
ap.add_argument("--include-hypertensive", action="store_true")
ap.add_argument("--out", default="samples")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

from disease_classifier import parse_odir

df = parse_odir(a.odir_root)
dis = ["Diabetic"] + (["Hypertensive"] if a.include_hypertensive else [])
out = Path(a.out)
for g in ("normal", "disease"):
    (out / g).mkdir(parents=True, exist_ok=True)
rows = []
for grp, labels in (("normal", ["Normal"]), ("disease", dis)):
    sub = df[df.label.isin(labels)]
    # one eye per patient so images are independent
    sub = sub.sample(frac=1, random_state=a.seed).drop_duplicates("patient").head(a.n)
    for _, r in sub.iterrows():
        src = Path(r["path"]); dst = out / grp / f"{r['label'][:3].lower()}_{src.name}"
        shutil.copy2(src, dst)
        rows.append({"file": str(dst), "group": grp, "odir_label": r["label"], "patient": r["patient"]})
    print(f"  {grp:<8}: copied {len(sub)} images")
with open(out / "manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file", "group", "odir_label", "patient"])
    w.writeheader(); w.writerows(rows)
print(f"\nDone → {out}/   now run:  python run_analysis.py --input {out}")
