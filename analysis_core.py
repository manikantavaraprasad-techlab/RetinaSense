"""
analysis_core.py — Real, per-image analysis engine.

Nothing in here is hard-coded: every number returned is computed from the
pixels of the image you pass in, using the trained networks.

    analyse_image()  ->  dict with probability maps, masks, skeleton, graph,
                         A/V masks, 6 biomarkers, confidence scores,
                         (optional) metrics vs ground truth, disease probs.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import cv2
import numpy as np

from config import CFG
from preprocessing import build_multichannel
from postprocessing import postprocess_mask, skeletonize_mask, skeleton_to_graph
from av_visualisation import labels_to_masks
from biomarkers import compute_all_biomarkers

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm", ".gif"}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def read_bgr(path) -> np.ndarray:
    """cv2 first (fast); Pillow fallback for .gif/.ppm etc."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        from PIL import Image
        rgb = np.array(Image.open(str(path)).convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def read_gray(path) -> np.ndarray:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        from PIL import Image
        g = np.array(Image.open(str(path)).convert("L"))
    return g


def find_gt_mask(img_path: Path, gt_dir: Path):
    """Match '01_test.tif' -> '01_manual1.gif' / '01.png' etc. Returns Path|None."""
    if gt_dir is None or not Path(gt_dir).exists():
        return None
    stem = img_path.stem
    m = re.match(r"\d+", stem)
    key = m.group(0) if m else stem
    files = [p for p in Path(gt_dir).iterdir() if p.suffix.lower() in IMG_EXT]
    for p in files:                                   # exact stem
        if p.stem == stem:
            return p
    pref = [p for p in files if re.match(r"0*" + key.lstrip("0") + r"(?!\d)", p.stem)]
    pref = [p for p in pref if "manual2" not in p.stem.lower() and "2ndho" not in p.stem.lower()
            and "_2" not in p.stem.lower()]
    return sorted(pref)[0] if pref else None


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def seg_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p = pred > 0
    g = gt > 0
    tp = float((p & g).sum()); fp = float((p & ~g).sum())
    fn = float((~p & g).sum()); tn = float((~p & ~g).sum())
    eps = 1e-8
    return {
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "iou":  tp / (tp + fp + fn + eps),
        "sensitivity": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "precision":   tp / (tp + fp + eps),
        "accuracy":    (tp + tn) / (tp + tn + fp + fn + eps),
    }


def dice_between(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0; b = b > 0
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# A/V edge classification (with confidence)
# ─────────────────────────────────────────────────────────────────────────────
def classify_edges(bgr, vessel_mask, graph, av_model,
                   patch_size: int = 32, n_samples: int = 3,
                   batch_size: int = 128):
    """
    Same 7-channel patch construction as av_segments.extract_segment_patch()
    (the function used to TRAIN the A/V CNN) but with the multichannel image
    computed once per image instead of once per segment.

    Returns: edge_labels {(u,v):0/1}, edge_conf {(u,v):max softmax},
             edge_p_artery {(u,v): P(artery)}
    """
    import torch
    if graph is None or graph.number_of_edges() == 0:
        return {}, {}, {}

    device = next(av_model.parameters()).device
    H, W = bgr.shape[:2]
    half = patch_size // 2
    mc = build_multichannel(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def patch_at(r, c):
        r1 = max(0, r - half); r2 = r1 + patch_size
        c1 = max(0, c - half); c2 = c1 + patch_size
        if r2 > H: r1 = H - patch_size; r2 = H
        if c2 > W: c1 = W - patch_size; c2 = W
        r1 = max(0, r1); c1 = max(0, c1)
        rgb_c = rgb[r1:r2, c1:c2]; mc_c = mc[r1:r2, c1:c2]
        m_c = vessel_mask[r1:r2, c1:c2]
        if rgb_c.shape[:2] != (patch_size, patch_size):
            rgb_c = cv2.resize(rgb_c, (patch_size, patch_size))
            mc_c = cv2.resize(mc_c, (patch_size, patch_size))
            m_c = cv2.resize(m_c, (patch_size, patch_size),
                             interpolation=cv2.INTER_NEAREST)
        p = np.zeros((7, patch_size, patch_size), np.float32)
        for i in range(3):
            p[i] = rgb_c[:, :, i] / 255.0
            p[4 + i] = mc_c[:, :, i] / 255.0
        p[3] = (m_c > 0).astype(np.float32)
        return p

    keys, patches = [], []
    for u, v, attr in graph.edges(data=True):
        path = attr.get("path", [u, v])
        if len(path) < 2:
            patch = np.zeros((7, patch_size, patch_size), np.float32)
        else:
            arr = np.array(path)
            idx = np.linspace(0, len(arr) - 1, n_samples, dtype=int)
            patch = sum(patch_at(int(arr[i, 0]), int(arr[i, 1])) for i in idx) / len(idx)
        keys.append((u, v)); patches.append(patch)

    probs = []
    av_model.eval()
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            b = torch.from_numpy(np.stack(patches[i:i + batch_size])).float().to(device)
            probs.append(torch.softmax(av_model(b), dim=1).cpu().numpy())
    probs = np.concatenate(probs, 0)                      # (N, 2) [vein, artery]
    labels = probs.argmax(1)
    conf = probs.max(1)
    return ({k: int(l) for k, l in zip(keys, labels)},
            {k: float(c) for k, c in zip(keys, conf)},
            {k: float(p) for k, p in zip(keys, probs[:, 1])})


# ─────────────────────────────────────────────────────────────────────────────
# One model, one image
# ─────────────────────────────────────────────────────────────────────────────
def _downstream(bgr, prob, av_model, threshold, gt512):
    out = {"prob": prob}
    t = time.time()
    vessel = postprocess_mask(prob, threshold=threshold, min_area=30, gap_px=7)
    skel = skeletonize_mask(vessel)
    graph = skeleton_to_graph(skel)
    out.update(vessel_mask=vessel, skeleton=skel, graph=graph)
    out["n_nodes"] = graph.number_of_nodes() if graph is not None else 0
    out["n_edges"] = graph.number_of_edges() if graph is not None else 0

    H, W = vessel.shape
    if graph is not None and graph.number_of_edges() > 0:
        labels, conf, pa = classify_edges(bgr, vessel, graph, av_model)
        artery, vein = labels_to_masks(graph, labels, (H, W), vessel)
    else:
        labels, conf, pa = {}, {}, {}
        artery = np.zeros((H, W), np.uint8); vein = np.zeros((H, W), np.uint8)
    out.update(edge_labels=labels, artery_mask=artery, vein_mask=vein)
    out["biomarkers"] = compute_all_biomarkers(vessel, artery, vein)

    # ── confidence scores (all derived from this image's outputs) ────────────
    v = vessel > 0
    cand = prob > 0.2
    unc = ((prob > 0.3) & (prob < 0.7))[cand].mean() * 100 if cand.any() else 0.0
    cf = np.array(list(conf.values())) if conf else np.array([])
    out["confidence"] = {
        "seg_mean_vessel_prob": float(prob[v].mean()) if v.any() else 0.0,
        "seg_uncertain_pct":    float(unc),
        "av_n_segments":        int(len(cf)),
        "av_mean_conf":         float(cf.mean()) if len(cf) else 0.0,
        "av_pct_conf_ge_0.8":   float((cf >= 0.8).mean() * 100) if len(cf) else 0.0,
        "av_artery_segments":   int(sum(1 for x in labels.values() if x == 1)),
        "av_vein_segments":     int(sum(1 for x in labels.values() if x == 0)),
    }
    if gt512 is not None:
        out["gt_metrics"] = seg_metrics(vessel, gt512)
    out["seconds"] = time.time() - t
    return out


# ─────────────────────────────────────────────────────────────────────────────
# One image, all models
# ─────────────────────────────────────────────────────────────────────────────
def analyse_image(path, seg_models: dict, av_model, disease_model=None,
                  threshold: float = 0.5, gt_path=None,
                  include_ensemble: bool = True, log=print) -> dict:
    from inference import predict_single
    from assets import sha256_file

    path = Path(path)
    bgr0 = read_bgr(path)
    S = CFG["img_size"]
    bgr = cv2.resize(bgr0, (S, S), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    res = {
        "name": path.stem, "path": str(path), "sha256": sha256_file(path),
        "orig_hw": tuple(bgr0.shape[:2]), "rgb": rgb,
        "multichannel": build_multichannel(bgr), "models": {},
    }
    log(f"    image sha256 : {res['sha256'][:16]}…   original size {bgr0.shape[1]}×{bgr0.shape[0]}"
        f"  → resized to {S}×{S}")

    gt512 = None
    if gt_path is not None:
        g = read_gray(gt_path)
        gt512 = cv2.resize((g > 127).astype(np.uint8), (S, S),
                           interpolation=cv2.INTER_NEAREST)
        res["gt_mask"] = gt512; res["gt_path"] = str(gt_path)
        log(f"    ground-truth : {Path(gt_path).name}")

    probs = {}
    for key, m in seg_models.items():
        t = time.time()
        probs[key] = predict_single(m, bgr)
        log(f"    [seg] {key:<18} forward pass {time.time()-t:5.2f}s   "
            f"prob range [{probs[key].min():.3f}, {probs[key].max():.3f}]  "
            f"mean {probs[key].mean():.4f}")
    if include_ensemble and len(probs) > 1:
        probs["ensemble"] = np.mean(list(probs.values()), axis=0).astype(np.float32)

    for key, prob in probs.items():
        res["models"][key] = _downstream(bgr, prob, av_model, threshold, gt512)

    ks = [k for k in probs if k != "ensemble"]
    res["agreement"] = {f"{a} vs {b}": dice_between(res["models"][a]["vessel_mask"],
                                                    res["models"][b]["vessel_mask"])
                        for i, a in enumerate(ks) for b in ks[i + 1:]}

    if disease_model is not None:
        from disease_classifier import predict_disease
        res["disease"] = predict_disease(disease_model, bgr0)
    return res
