"""
engine.py — ONE entry point used by the website, full_pipeline_report.py and the step scripts.

    from engine import analyse
    res = analyse("my_fundus.jpg")                 # auto: trained networks if weights/ has them, else classical
    res = analyse("my_fundus.jpg", prefer="classical")

res["engine"] is  "trained"  (segmentation from your .pt files)  or  "classical"  (no weights found).
res["engine_label"] says exactly that, and every picture written by result_figures.py is stamped with it.
Nothing is cached: every number is computed from the pixels of the image you pass in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fallback_engine as fe                      # noqa: E402

_CACHE = {}                                       # loaded networks (kept in memory between web requests)
SEG_NAME = {"attention_unet": "Attention U-Net", "unetplusplus": "UNet++", "unetplusplus_scse": "UNet++ + SCSE"}
PREFERRED = ("unetplusplus_scse", "unetplusplus", "attention_unet")


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def find_weights(weights_dir=None) -> dict:
    """{kind: Path} of every trained checkpoint found (empty if PyTorch or the files are missing)."""
    if not torch_available():
        return {}
    from assets import default_search_dirs, discover_checkpoints
    return discover_checkpoints(default_search_dirs(weights_dir), verbose=False)


def status(weights_dir=None) -> dict:
    """What can run right now — shown at the top of the website."""
    found = find_weights(weights_dir)
    seg = [k for k in PREFERRED if k in found]
    if seg and "av_cnn" in found:
        mode, msg = "trained", "Trained networks: " + ", ".join(SEG_NAME[k] for k in seg) + " + AVResNet18"
    elif seg:
        mode, msg = "trained_seg_only", "Trained segmentation (" + ", ".join(SEG_NAME[k] for k in seg) + ") + heuristic artery/vein (av_cnn.pt missing)"
    else:
        why = "PyTorch not installed" if not torch_available() else "no .pt files in weights/"
        mode, msg = "classical", f"Classical fallback ({why}) — copy your trained .pt files into weights/ for the real networks"
    return {"mode": mode, "message": msg, "torch": torch_available(), "found": {k: str(v) for k, v in found.items()}}


def _load(kind, path):
    if path not in _CACHE:
        from config import CFG
        if kind == "av_cnn":
            from av_classification import load_av_cnn
            _CACHE[path] = load_av_cnn(str(path), device=CFG["device"])
        else:
            from inference import load_model
            _CACHE[path] = load_model(str(path), device=CFG["device"])
    return _CACHE[path]


def analyse(image, prefer: str = "auto", threshold: float = 0.5, weights_dir=None) -> dict:
    """image: path or BGR ndarray.  prefer: 'auto' | 'classical'."""
    if isinstance(image, (str, Path)):
        bgr0 = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr0 is None:
            from PIL import Image
            bgr0 = cv2.cvtColor(np.array(Image.open(str(image)).convert("RGB")), cv2.COLOR_RGB2BGR)
        name = Path(image).stem
    else:
        bgr0, name = image, "image"
    if bgr0 is None:
        raise ValueError("could not read the image")
    st = status(weights_dir) if prefer != "classical" else {"mode": "classical", "found": {}}
    if st["mode"] == "classical":
        res = fe.analyse_classical(bgr0, threshold)
    else:
        found = {k: Path(v) for k, v in st["found"].items()}
        seg_kinds = [k for k in PREFERRED if k in found]
        key = seg_kinds[0]
        from inference import predict_single
        bgr512 = cv2.resize(bgr0, (fe.IMG_SIZE, fe.IMG_SIZE), interpolation=cv2.INTER_AREA)
        prob = predict_single(_load(key, found[key]), bgr512)
        av_fn, av_note = None, ""
        if "av_cnn" in found:
            import analysis_core as core
            av_model = _load("av_cnn", found["av_cnn"])
            av_fn = lambda b, v, g: core.classify_edges(b, v, g, av_model)   # noqa: E731
        else:
            av_note = " + heuristic A/V"
        res = fe.analyse_from_prob(bgr0, prob, threshold, av_fn, engine="trained",
                                   engine_label=f"Trained network: {SEG_NAME[key]}" + (" + AVResNet18" if av_fn else av_note),
                                   model_name=SEG_NAME[key])
        # the other trained networks, for the model-comparison panel
        res["other_models"] = {}
        for k in seg_kinds[1:]:
            p2 = predict_single(_load(k, found[k]), bgr512)
            res["other_models"][k] = {"prob": p2, "mask": fe.postprocess_mask(p2, threshold=threshold, min_area=30, gap_px=7)}
    res["name"] = name
    res["orig_hw"] = tuple(bgr0.shape[:2])
    return res
