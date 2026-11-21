"""
fallback_engine.py — the SAME pipeline as the project, but WITHOUT the trained networks.

Why this exists
---------------
The trained checkpoints (best_model_*.pt, av_cnn.pt) are not inside the project zip, so the
deep-learning stepssegmentation,rtery/vein CNN) cannot run until you copy
those files into weights/.  So that the scripts and the website still give REAL, per-image
results from ANY fundus image you give them, this module replaces only the two learned steps
with classical image processing:

    step                              trained pipeline (needs .pt)       this fallback (no .pt)
    --------------------------------  ---------------------------------  ------------------------------
    pre-processing          green | CLAHE | Retinex            identical (preprocessing.py)
    vessel segmentation Attention U-Net / UNet++ / SCSE    multi-scale ridge filter (Sato)
    clean + connect + skeleton + graph                        identical (postprocessing.py)
    artery / vein          AVResNet18 on 32x32 patches        colour/brightness heuristic
    six biomarkers         identical (biomarkers.py)          identical (biomarkers.py)

Everything computed here comes from the pixels of the image you pass in — nothing is cached or
copied from .  The numbers will NOT match the project (different segmentation
method, different image) and every figure is stamped "classical fallback".
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sato, apply_hysteresis_threshold, threshold_otsu

from preprocessing import build_multichannel
from postprocessing import postprocess_mask, skeletonize_mask, skeleton_to_graph, _bridge_endpoints
from av_visualisation import labels_to_masks
from biomarkers import compute_all_biomarkers

IMG_SIZE = 512
ENGINE_LABEL = "Classical fallback (no trained weights) — NOT the trained networks"


# ───────────────────────────── field of view ────────────────────────────────
def field_of_view(bgr: np.ndarray, erode_frac: float = 0.03) -> np.ndarray:
    """Bool mask of the circular retina area (the black border is excluded)."""
    gray = bgr.max(axis=2).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 3)
    thr = max(10.0, 0.35 * threshold_otsu(gray)) if gray.max() > 10 else 10.0
    fov = gray > thr
    fov = ndi.binary_fill_holes(fov)
    fov = cv2.morphologyEx(fov.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(fov.astype(np.uint8))
    if n > 1:                                                   # keep the biggest blob
        fov = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    r = max(3, int(erode_frac * min(bgr.shape[:2])))
    return cv2.erode(fov.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0


# ───────────────────────── vessel "probability" map ─────────────────────────
def vessel_probability(bgr: np.ndarray, fov: np.ndarray | None = None) -> np.ndarray:
    """
    Multi-scale ridge filter on the CLAHE-enhanced green channel.
    Returns float32 (H, W) in [0, 1].  It is a vesselness score, not a learned probability,
    but it is scaled so that 0.5 is a sensible cut-off for postprocess_mask().
    """
    fov = field_of_view(bgr) if fov is None else fov
    mc = build_multichannel(bgr)
    g = mc[:, :, 1].astype(np.float32) / 255.0                  # CLAHE green (vessels dark)
    bg = cv2.GaussianBlur(g, (0, 0), 20)
    g = g - bg                                                  # flatten the illumination
    g[~fov] = 0
    g = cv2.GaussianBlur(g, (0, 0), 0.8)
    resp = sato(g, sigmas=(1.5, 2.0, 2.5, 3.5), black_ridges=True).astype(np.float32)
    resp[~fov] = 0
    top = np.percentile(resp[fov], 99.5) if fov.any() else 1.0
    resp = np.clip(resp / (top + 1e-8), 0, 1)
    # calibrate so that the 0.5 line separates vessels from background (Otsu on the retina only)
    t = float(threshold_otsu(resp[fov])) if fov.any() else 0.5
    t = float(np.clip(t * 0.8, 0.08, 0.6))
    prob = np.clip(0.5 * resp / t, 0, 1)                     # resp == t  ->  0.5
    prob = np.where(prob > 0.5, 0.5 + 0.5 * np.clip((prob - 0.5) / 0.5, 0, 1), prob).astype(np.float32)
    prob[~fov] = 0
    return prob


# ──────────────────────── artery / vein heuristic ───────────────────────────
def classify_edges_heuristic(bgr: np.ndarray, graph, vessel_mask: np.ndarray):
    """
    Colour/brightness heuristic (arteries: brighter, more orange-red; veins: darker, deeper red),
    the same cue  describe — NOT the AVResNet18.

    Per skeleton segment we measure (a) the mean red-channel brightness and (b) the
    green-channel contrast against the local background along the centre-line, standardise them
    over the whole image and split the segments at the median score.
    The brighter group is called "artery".

    Returns (edge_labels {(u,v): 1 artery / 0 vein}, edge_margin {(u,v): 0..1}, edge_p_artery)
    The "margin" is how far a segment sits from the decision boundary — a heuristic score,
    NOT a softmax confidence.
    """
    if graph is None or graph.number_of_edges() == 0:
        return {}, {}, {}
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    R, G = rgb[:, :, 0], rgb[:, :, 1]
    bgG = cv2.GaussianBlur(G, (0, 0), 12)
    keys, feats = [], []
    for u, v, attr in graph.edges(data=True):
        path = np.array(attr.get("path", [u, v]))
        if len(path) < 4:
            continue
        rr, cc = path[:, 0].clip(0, R.shape[0] - 1), path[:, 1].clip(0, R.shape[1] - 1)
        red = float(np.median(R[rr, cc]))
        contrast = float(np.median(bgG[rr, cc] - G[rr, cc]))
        ratio = float(np.median(R[rr, cc] / (G[rr, cc] + 1.0)))
        keys.append((u, v)); feats.append([red, contrast, ratio])
    if len(keys) < 2:
        return {}, {}, {}
    F = np.array(feats, np.float32)
    Z = (F - F.mean(0)) / (F.std(0) + 1e-6)
    score = Z[:, 0] * 0.5 + Z[:, 2] * 0.5 - Z[:, 1] * 0.25      # bright + orange - dark
    # split at the median: a retina has roughly as many artery as vein segments (deterministic)
    thr = float(np.median(score))
    c0, c1 = score[score <= thr].mean(), score[score > thr].mean()
    labels = {k: int(s > thr) for k, s in zip(keys, score)}
    span = max(1e-6, abs(c1 - c0) / 2)
    margin = {k: float(np.clip(abs(s - thr) / span, 0, 1)) for k, s in zip(keys, score)}
    pa = {k: float(1 / (1 + np.exp(-(s - thr) / span * 2))) for k, s in zip(keys, score)}
    # edges that were too short to measure: copy the label of the nearest measured neighbour at a node
    for u, v, _ in graph.edges(data=True):
        if (u, v) in labels:
            continue
        nb = [labels[e] for n_ in (u, v) for e in list(graph.edges(n_)) if e in labels or (e[1], e[0]) in labels
              for e in [e if e in labels else (e[1], e[0])]]
        labels[(u, v)] = int(round(np.mean(nb))) if nb else 0
        margin[(u, v)] = 0.0; pa[(u, v)] = 0.5
    return labels, margin, pa


# ───────────────────────────── full pipeline ────────────────────────────────
def stages_clean_and_connect(prob, thr=0.5, min_area=30, gap_px=7):
    """The five pictures of(threshold → denoise → bridge → skeleton → graph)."""
    s1 = ((prob >= thr) * 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(s1, connectivity=8)
    s2 = np.zeros_like(s1)
    for l in range(1, n):
        if st[l, cv2.CC_STAT_AREA] >= min_area:
            s2[lab == l] = 255
    s3 = _bridge_endpoints(s2, prob, gap_px=gap_px, prob_threshold=thr)
    n, lab, st, _ = cv2.connectedComponentsWithStats(s3, connectivity=8)
    s3f = np.zeros_like(s3)
    for l in range(1, n):
        if st[l, cv2.CC_STAT_AREA] >= 10:
            s3f[lab == l] = 255
    skel = skeletonize_mask(s3f)
    graph = skeleton_to_graph(skel)
    return s1, s2, s3f, skel, graph


def analyse_from_prob(bgr0: np.ndarray, prob: np.ndarray | None = None, threshold: float = 0.5,
                      av_classifier=None, engine: str = "classical", engine_label: str = ENGINE_LABEL,
                      model_name: str = "Classical ridge filter (Sato)") -> dict:
    """
    Everything AFTER the vessel-probability map→ 21).  Used by BOTH engines:
      * prob=None            -> the classical ridge filter produces the map
      * prob=<trained net>   -> the map comes from a trained network (see engine.py)
      * av_classifier=None   -> colour/brightness heuristic;  or a callable
        f(bgr, vessel_mask, graph) -> (labels, confidence, p_artery)  e.g. the AVResNet18 wrapper.
    """
    bgr = cv2.resize(bgr0, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fov = field_of_view(bgr)
    if prob is None:
        prob = vessel_probability(bgr, fov)
    s1, s2, s3, _, _ = stages_clean_and_connect(prob, threshold)
    vessel = postprocess_mask(prob, threshold=threshold, min_area=30, gap_px=7)
    skel = skeletonize_mask(vessel)
    graph = skeleton_to_graph(skel)
    H, W = vessel.shape
    av_kind = "AVResNet18 softmax confidence" if av_classifier is not None else "heuristic margin (not a probability)"
    if graph is not None and graph.number_of_edges() > 0:
        fn = av_classifier or (lambda b, v, g: classify_edges_heuristic(b, g, v))
        labels, margin, pa = fn(bgr, vessel, graph)
        artery, vein = labels_to_masks(graph, labels, (H, W), vessel)
    else:
        labels, margin, pa = {}, {}, {}
        artery = np.zeros((H, W), np.uint8); vein = np.zeros((H, W), np.uint8)
    bm = compute_all_biomarkers(vessel, artery, vein)
    v = vessel > 0
    mg = np.array(list(margin.values())) if margin else np.array([])
    cand = prob > 0.2
    unc = float(((prob > 0.3) & (prob < 0.7))[cand].mean() * 100) if cand.any() else 0.0
    return {
        "engine": engine, "engine_label": engine_label, "model_name": model_name, "av_score_kind": av_kind,
        "rgb": rgb, "multichannel": build_multichannel(bgr), "fov": fov,
        "prob": prob, "vessel_mask": vessel, "skeleton": skel, "graph": graph,
        "stage_threshold": s1, "stage_denoised": s2, "stage_bridged": s3,
        "artery_mask": artery, "vein_mask": vein, "edge_labels": labels, "biomarkers": bm,
        "n_nodes": graph.number_of_nodes() if graph is not None else 0,
        "n_edges": graph.number_of_edges() if graph is not None else 0,
        "confidence": {
            "seg_mean_vessel_prob": float(prob[v].mean()) if v.any() else 0.0,
            "seg_uncertain_pct": unc,
            "av_n_segments": int(len(labels)),
            "av_mean_score": float(mg.mean()) if len(mg) else 0.0,
            "av_pct_score_ge_0.5": float((mg >= 0.5).mean() * 100) if len(mg) else 0.0,
            "av_pct_score_ge_0.8": float((mg >= 0.8).mean() * 100) if len(mg) else 0.0,
            "av_artery_segments": int(sum(1 for x in labels.values() if x == 1)),
            "av_vein_segments": int(sum(1 for x in labels.values() if x == 0)),
        },
        "extra": {"mask_coverage_%": float(v.mean() * 100), "fov_coverage_%": float(fov.mean() * 100),
                  "artery_pixels": int((artery > 0).sum()), "vein_pixels": int((vein > 0).sum())},
    }


def analyse_classical(bgr0: np.ndarray, threshold: float = 0.5) -> dict:
    """bgr0: any-size BGR fundus image -> result dict (classical segmentation + heuristic A/V)."""
    return analyse_from_prob(bgr0, None, threshold)
