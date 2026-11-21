"""validate_screening.py — measure how well the biomarker-based prediction agrees with KNOWN labels.

    folder layout:   labelled/normal/*.png   labelled/glaucoma/*   labelled/diabetes/*   labelled/hbp/*
    run:             python validate_screening.py labelled

Prints a confusion matrix + accuracy and writes validation_report.json.  Run this on your own labelled images before presenting
accuracy claims; the website shows only a pattern-match score, never an accuracy number it has not measured.
"""
import json, sys
from pathlib import Path
import engine, screening

MAP = {"normal": "Normal", "glaucoma": "Glaucoma", "diabetes": "Diabetes", "hbp": "High blood pressure"}
root = Path(sys.argv[1] if len(sys.argv) > 1 else "labelled")
rows = []
for folder, label in MAP.items():
    for p in sorted((root / folder).glob("*")) if (root / folder).exists() else []:
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}: continue
        r = engine.analyse(p); pred = screening.screen({k: float(v) for k, v in r["biomarkers"].items()}, av_reliable="AVResNet" in r["engine_label"])["verdict"]
        rows.append((label, pred))
if not rows: sys.exit(f"no images found under {root}/<normal|glaucoma|diabetes|hbp>/")
cls = screening.CLASSES; cm = {a: {b: 0 for b in cls} for a in cls}
for a, b in rows: cm[a][b] += 1
acc = sum(a == b for a, b in rows) / len(rows)
print(f"images: {len(rows)}   accuracy: {100 * acc:.1f}%\n{'true \\ pred':>22}" + "".join(f"{c[:12]:>14}" for c in cls))
for a in cls: print(f"{a:>22}" + "".join(f"{cm[a][b]:>14}" for b in cls))
Path("validation_report.json").write_text(json.dumps({"n": len(rows), "accuracy": acc, "confusion": cm}, indent=2))
