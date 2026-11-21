"""
_common.py — small helpers shared by the step_*.py scripts.

Every step scripts:
  * can be run from any folder:         python_segmentation_output.py
  * has a default input image:          first image found in  sample_images/
  * accepts your own image:             --input path/to/fundus.png  (file or folder)
  * writes its pictures to:             outputs/<script name>/
  * never invents numbers: scripts that need the trained networks stop with a
    clear message if the .pt files are not in  weights/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")                      # save-only (no pop-up windows)
import matplotlib.pyplot as plt                # noqa: E402
import cv2                                     # noqa: E402
import numpy as np                             # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm", ".gif"}
SEG_ORDER = ("attention_unet", "unetplusplus", "unetplusplus_scse")
SEG_NAME = {"attention_unet": "Attention U-Net", "unetplusplus": "UNet++",
            "unetplusplus_scse": "UNet++ + SCSE"}
IMG_SIZE = 512


def make_parser(description: str, need_image: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    if need_image:
        ap.add_argument("--input", default=None,
                        help="fundus image or folder (default: first image in sample_images/)")
        ap.add_argument("--limit", type=int, default=0, help="only the first N images")
    ap.add_argument("--out", default=None, help="output folder (default outputs/<script name>/)")
    ap.add_argument("--show", action="store_true", help="also open the figures on screen")
    return ap


def add_model_args(ap: argparse.ArgumentParser):
    ap.add_argument("--weights-dir", default=None, help="folder with the trained .pt files (default ./weights)")
    ap.add_argument("--models", nargs="*", default=None, choices=list(SEG_ORDER),
                    help="which segmentation networks to use (default: all that are found)")
    ap.add_argument("--threshold", type=float, default=0.5)


def out_dir(a, script_file: str) -> Path:
    d = Path(a.out) if a.out else HERE / "outputs" / Path(script_file).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def pick_images(a, default_dir: str = "sample_images", n_default: int = 1):
    """Resolve --input (file / folder) or fall back to sample_images/."""
    if a.input:
        p = Path(a.input)
        if not p.exists():
            sys.exit(f"ERROR: input not found: {p}")
        imgs = [p] if p.is_file() else list_images(p)
    else:
        d = HERE / default_dir
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.exists() else []
        imgs = imgs[:n_default]
        if imgs:
            print(f"[input] no --input given -> using {imgs[0].relative_to(HERE)}"
                  + (f" (+{len(imgs)-1} more)" if len(imgs) > 1 else ""))
    if not imgs:
        sys.exit("ERROR: no fundus image found. Pass one with  --input path/to/image.png  "
                 "or put images into  sample_images/")
    if getattr(a, "limit", 0):
        imgs = imgs[:a.limit]
    return imgs


def read_bgr(path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        from PIL import Image
        bgr = cv2.cvtColor(np.array(Image.open(str(path)).convert("RGB")), cv2.COLOR_RGB2BGR)
    return bgr


def to_512(bgr):
    return cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def save_fig(fig, path: Path, show=False, dpi=130):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  saved -> {path}")
    if show:
        plt.show()
    plt.close(fig)


# ── trained networks (only imported when a script really needs them) ─────────
def require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("ERROR: PyTorch is not installed.  Run:  pip install -r requirements.txt")


def load_segmentation_models(a):
    """Returns {kind: model}. Stops with a clear message if no checkpoint exists."""
    require_torch()
    from config import CFG
    from assets import default_search_dirs, discover_checkpoints, SEG_KINDS
    from inference import load_model
    found = discover_checkpoints(default_search_dirs(a.weights_dir))
    kinds = [k for k in (a.models or SEG_KINDS) if k in found]
    if not kinds:
        sys.exit("\nERROR: no trained segmentation checkpoint (.pt) found.\n"
                 "  Copy the files you downloaded from your Kaggle notebooks into  weights/\n"
                 "  (best_model_attention_unet.pt, best_model_unetplusplus.pt, best_model_unetplusplus_scse.pt)\n"
                 "  See weights/README.md.  This script never fabricates results.")
    models = {k: load_model(str(found[k]), device=CFG["device"]) for k in kinds}
    return models, found


def load_av_model(found):
    from config import CFG
    from av_classification import load_av_cnn
    if "av_cnn" not in found:
        sys.exit("\nERROR: artery/vein checkpoint (av_cnn.pt) not found in weights/ — see weights/README.md")
    return load_av_cnn(str(found["av_cnn"]), device=CFG["device"])


def load_disease_model(found):
    if "disease" not in found:
        return None
    from config import CFG
    from disease_classifier import load_disease_model as _l
    return _l(str(found["disease"]), device=CFG["device"])


def have_trained(a, need_av: bool = True) -> bool:
    """True if PyTorch + the trained checkpoints are available (otherwise the step scripts use fallback_runner)."""
    try:
        import engine
        st = engine.status(getattr(a, "weights_dir", None))
    except Exception:
        return False
    ok = ("trained", "trained_seg_only") if not need_av else ("trained",)
    return st["mode"] in ok
