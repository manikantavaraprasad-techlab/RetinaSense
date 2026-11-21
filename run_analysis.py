"""
run_analysis.py — ONE command that runs the whole project on real images.

    python run_analysis.py --input path/to/fundus.jpg
    python run_analysis.py --input path/to/folder_of_images
    python run_analysis.py --input samples            # samples/normal + samples/disease
    python run_analysis.py --input img.png --gt-masks DRIVE/test/1st_manual --show

Every number printed and every picture saved is computed live from the pixels
of the image(s) you give, by the trained networks:

    1 pre-processing (green / CLAHE / Retinex)      2 vessel segmentation (3 U-Nets + ensemble)
    3 post-processing + skeleton + graph            4 artery/vein CNN on every vessel segment
    5 six biomarkers                                6 ODIR-5K disease-category CNN
    7 confidence scores, and Dice/IoU/Sens/Spec if you pass --gt-masks

If a sub-folder layout is used (e.g. samples/normal, samples/disease) the
sub-folder name becomes the group name and a group comparison table is made.

Nothing is simulated: if a trained checkpoint is missing the script stops and
says which one; it never falls back to made-up values.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")           # headless: save figures to disk
import matplotlib.pyplot as plt     # noqa: E402


class Tee:
    """Mirror everything printed to a log file (proof of what was run)."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8"); self.o = sys.stdout
    def write(self, s): self.o.write(s); self.f.write(s)
    def flush(self): self.o.flush(); self.f.flush()


def collect_images(inp: Path):
    from analysis_core import IMG_EXT
    items = []
    if inp.is_file():
        return [("all", inp)]
    direct = sorted(p for p in inp.iterdir() if p.suffix.lower() in IMG_EXT)
    subs = sorted(d for d in inp.iterdir() if d.is_dir())
    for p in direct:
        items.append(("all", p))
    for d in subs:
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                items.append((d.name.lower(), p))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="image file or folder (sub-folders = groups)")
    ap.add_argument("--output", default="outputs/analysis")
    ap.add_argument("--weights-dir", default=None, help="folder with the downloaded .pt files (default: ./weights)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of: attention_unet unetplusplus unetplusplus_scse")
    ap.add_argument("--primary", default="unetplusplus_scse", help="model used for the main figure/tables")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--gt-masks", default=None, help="folder of expert vessel masks (adds Dice/IoU/…)")
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--no-disease", action="store_true")
    ap.add_argument("--all-figures", action="store_true", help="save the 5-panel figure for EVERY model")
    ap.add_argument("--show", action="store_true", help="also pop up the figures")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images")
    a = ap.parse_args()

    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out / "run_log.txt")
    t_start = time.time()

    print("=" * 78)
    print("  RETINAL VESSEL SEGMENTATION · A/V CLASSIFICATION · BIOMARKERS · DISEASE CNN")
    print(f"  started {datetime.now():%Y-%m-%d %H:%M:%S}   python {platform.python_version()}")
    print("=" * 78)

    import cv2, numpy as np, torch                       # noqa: E401
    from config import CFG
    from assets import default_search_dirs, discover_checkpoints, sha256_file, SEG_KINDS
    from inference import load_model
    from av_classification import load_av_cnn
    import analysis_core as core
    import analysis_report as rep

    print(f"  torch {torch.__version__}  opencv {cv2.__version__}  device {CFG['device']}")

    # ── 1. locate + load trained weights ─────────────────────────────────────
    print("\n[1/4] Locating trained checkpoints")
    found = discover_checkpoints(default_search_dirs(a.weights_dir))
    kinds = [k for k in (a.models or SEG_KINDS) if k in found]
    if not kinds:
        sys.exit("\nERROR: no trained segmentation checkpoint found.\n"
                 "  Download the .pt files from your Kaggle notebook outputs into  ./weights/\n"
                 "  (see weights/README.md  or run  python download_weights.py --help).\n"
                 "  Refusing to continue — this script never fabricates results.")
    if "av_cnn" not in found:
        sys.exit("\nERROR: A/V CNN checkpoint (av_cnn.pt) not found in ./weights/ — see weights/README.md")
    if a.primary not in kinds:
        print(f"  primary model '{a.primary}' unavailable -> using '{kinds[-1]}'")
        a.primary = kinds[-1]

    print("\n[2/4] Loading networks")
    seg_models = {k: load_model(str(found[k]), device=CFG["device"]) for k in kinds}
    av_model = load_av_cnn(str(found["av_cnn"]), device=CFG["device"])
    disease_model = None
    if not a.no_disease:
        if "disease" in found:
            from disease_classifier import load_disease_model
            disease_model = load_disease_model(str(found["disease"]), device=CFG["device"])
        else:
            print("  [disease] disease_classifier.pt not found -> disease CNN SKIPPED "
                  "(train it with train_disease_classifier.py)")

    # ── 2. images ────────────────────────────────────────────────────────────
    items_in = collect_images(Path(a.input))
    if a.limit: items_in = items_in[:a.limit]
    if not items_in:
        sys.exit(f"ERROR: no images found in {a.input}")
    groups = sorted({g for g, _ in items_in})
    print(f"\n[3/4] Analysing {len(items_in)} image(s)   groups: {groups}")

    # ── 3. analyse ───────────────────────────────────────────────────────────
    results = []
    for n, (grp, p) in enumerate(items_in, 1):
        print("\n" + "─" * 78 + f"\n  [{n}/{len(items_in)}] {p}   (group: {grp})\n" + "─" * 78)
        gt = core.find_gt_mask(p, Path(a.gt_masks)) if a.gt_masks else None
        if a.gt_masks and gt is None:
            print(f"    (no ground-truth mask matched '{p.stem}')")
        r = core.analyse_image(p, seg_models, av_model, disease_model, threshold=a.threshold,
                               gt_path=gt, include_ensemble=not a.no_ensemble)
        rep.print_image_report(r, a.primary)
        rep.save_image_outputs(r, out, a.primary, all_figures=a.all_figures, show=a.show)
        results.append((grp, r))

    # ── 4. tables across images ──────────────────────────────────────────────
    print("\n[4/4] Summary tables")
    rep.print_summary_table(results, a.primary)
    csvp = rep.write_summary_csv(results, out)
    if len(results) > 1:
        rep.figure_biomarker_overview(results, a.primary, out / "biomarker_overview.png")
    real_groups = [g for g in groups if g != "all"]
    if len(real_groups) >= 2:
        rep.group_comparison([(g, r) for g, r in results if g != "all"], a.primary, out)

    if any("gt_metrics" in r["models"][k] for _, r in results for k in r["models"]):
        print("\n  Segmentation accuracy vs expert masks (mean over the images that had one):")
        print(f"  {'model':<24}{'n':>3}{'Dice':>9}{'IoU':>9}{'Sens':>9}{'Spec':>9}")
        for k in results[0][1]["models"]:
            ms = [r["models"][k]["gt_metrics"] for _, r in results if "gt_metrics" in r["models"][k]]
            if ms:
                f = lambda s: float(np.mean([m[s] for m in ms]))
                print(f"  {rep.disp(k):<24}{len(ms):>3}{f('dice'):9.4f}{f('iou'):9.4f}"
                      f"{f('sensitivity'):9.4f}{f('specificity'):9.4f}")

    # ── provenance manifest ──────────────────────────────────────────────────
    man = {"started": datetime.fromtimestamp(t_start).isoformat(timespec="seconds"),
           "seconds": round(time.time() - t_start, 1), "args": vars(a),
           "torch": torch.__version__, "device": str(CFG["device"]),
           "checkpoints": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in found.items()},
           "images": [{"group": g, "path": r["path"], "sha256": r["sha256"]} for g, r in results]}
    (out / "run_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"\n  CSV      : {csvp}\n  manifest : {out/'run_manifest.json'}  (checkpoint + image SHA-256 hashes)")
    print(f"  log      : {out/'run_log.txt'}\n  figures  : {out}/<image>/figure_{a.primary}.png")
    print(f"\nDone in {time.time()-t_start:.1f}s.")
    if a.show: plt.show()


if __name__ == "__main__":
    main()
