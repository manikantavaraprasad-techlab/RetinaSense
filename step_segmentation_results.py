"""— "Segmentation results"   Dice / IoU / Sensitivity / Specificity per network

These numbers come from comparing each network's mask with EXPERT masks, so the script needs
the trained .pt files AND images with expert masks. Choose one:

  A) the dataset collection used for training (rebuilds the same 20 % validation split, seed 42):
        python_segmentation_results.py --root D:\\data\\retinal-vessel-fundus-dataset-collection
  B) any folder of images + their expert masks (e.g. DRIVE test set):
        python_segmentation_results.py --images DRIVE/test/images --masks DRIVE/test/1st_manual

Saves outputs/_segmentation_results/
    results_table.csv / .md, per_image_metrics.csv, results_bar_chart.png, qualitative_<model>.png
Also prints the best_dice stored inside each checkpoint (the value logged at the end of training).
"""
from _common import *                       # noqa: F401,F403
import subprocess


def main():
    ap = make_parser(__doc__, need_image=False)
    ap.add_argument("--root", default=None); ap.add_argument("--images", default=None); ap.add_argument("--masks", default=None)
    ap.add_argument("--weights-dir", default=None); ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    out = out_dir(a, __file__)


    require_torch()
    from assets import default_search_dirs, discover_checkpoints
    import torch
    found = discover_checkpoints(default_search_dirs(a.weights_dir), verbose=False)
    print("Checkpoint summary (stored inside the .pt files):")
    for k in SEG_ORDER:
        if k in found:
            ck = torch.load(str(found[k]), map_location="cpu", weights_only=False)
            print(f"  {SEG_NAME[k]:<16} best_dice = {float(ck.get('best_dice', 0) or 0):.4f}   epoch {ck.get('epoch', '?')}   ({found[k].name})")
    if not any(k in found for k in SEG_ORDER):
        sys.exit("ERROR: no segmentation checkpoints in weights/ — see weights/README.md")
    import os
    if not (a.root or (a.images and a.masks) or os.environ.get("RETINAL_SEG_ROOT")):
        sys.exit("\nERROR: give the evaluation data:  --root <dataset collection>   or   --images <dir> --masks <dir>\n"
                 "  (DRIVE / STARE / CHASEDB1 are not shipped with this project.)")
    cmd = [sys.executable, str(HERE / "evaluate_segmentation.py"), "--out", str(out), "--threshold", str(a.threshold)]
    for flag, v in (("--root", a.root), ("--images", a.images), ("--masks", a.masks), ("--weights-dir", a.weights_dir)):
        if v: cmd += [flag, v]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
