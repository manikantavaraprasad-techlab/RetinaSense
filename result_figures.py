"""
result_figures.py — draws every picture/graph of the project from ONE result dict (engine.analyse()).

    fig_preprocessing    fig_segmentation   fig_clean_connect
    fig_artery_vein fig_biomarkers   fig_completepanels)
    render_all(res, folder)  -> writes all of them as PNG + results.json and returns {name: path}

Every figure carries a footer saying which engine produced it (trained networks or classical fallback).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg") if matplotlib.get_backend().lower() not in ("tkagg", "qtagg", "macosx") else None
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.patches import Patch

from biomarkers import BIOMARKER_LABELS, NORMAL_RANGES, _bm_color, render_av_overlay


def _stamp(fig, res):
    col = "#1b7f3b" if res["engine"] == "trained" else "#b36b00"
    fig.text(.5, .005, f"engine: {res['engine_label']}", ha="center", fontsize=8, color=col)


def fig_preprocessing(res):
    mc, rgb = res["multichannel"], res["rgb"]
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.4))
    panels = [(rgb, "Original fundus", None), (mc[:, :, 0], "1. Green channel\nvessels darkest in green light", "gray"),
              (mc[:, :, 1], "2. CLAHE\nboosts local contrast", "gray"), (mc[:, :, 2], "3. Retinex\nremoves uneven lighting", "gray"),
              (mc, "Stack of the 3 → network input", None)]
    for a, (im, t, cm) in zip(ax, panels):
        a.imshow(im, cmap=cm); a.set_title(t, fontsize=10); a.axis("off")
    fig.suptitle("Step 1 — Clean the image", fontweight="bold"); fig.tight_layout(rect=(0, .03, 1, .90)); _stamp(fig, res)
    return fig


def fig_segmentation(res):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(res["rgb"]); ax[0].set_title("Original fundus")
    h = ax[1].imshow(res["prob"], cmap="hot", vmin=0, vmax=1); ax[1].set_title("Vessel probability map")
    fig.colorbar(h, ax=ax[1], fraction=.046, pad=.02)
    ax[2].imshow(res["vessel_mask"], cmap="gray"); ax[2].set_title("Binary mask (threshold 0.5, cleaned)")
    ov = res["rgb"].copy(); m = res["vessel_mask"] > 0
    ov[m] = (0.5 * ov[m] + 0.5 * np.array([235, 40, 40])).astype(np.uint8)
    ax[3].imshow(ov); ax[3].set_title("Overlay on original")
    ax[3].legend(handles=[Patch(color="#eb2828", label="Detected vessels")], loc="lower right", fontsize=8)
    for a in ax: a.axis("off")
    fig.suptitle(f"What the output looks like — {res['model_name']}", fontweight="bold")
    fig.tight_layout(rect=(0, .03, 1, .90)); _stamp(fig, res)
    return fig


def fig_clean_connect(res):
    s1, s2, s3, skel, g, rgb = res["stage_threshold"], res["stage_denoised"], res["stage_bridged"], res["skeleton"], res["graph"], res["rgb"]
    removed = (s1 > 0) & (s2 == 0); bridged = (s3 > 0) & (s2 == 0)
    r2 = np.stack([s2] * 3, -1).copy(); r2[removed] = [255, 60, 60]
    r3 = np.stack([s3] * 3, -1).copy(); r3[bridged] = [60, 255, 60]
    sk = rgb.copy(); sk[cv2.dilate(skel.astype(np.uint8), np.ones((2, 2))) > 0] = [255, 255, 0]
    fig, ax = plt.subplots(1, 6, figsize=(26, 4.8))
    ax[0].imshow(rgb); ax[0].set_title("Input")
    ax[1].imshow(s1, cmap="gray"); ax[1].set_title(f"1. Threshold 0.5\n{(s1>0).sum():,} px")
    ax[2].imshow(r2); ax[2].set_title(f"2. Remove noise (<30 px)\nred = removed ({removed.sum():,} px)")
    ax[3].imshow(r3); ax[3].set_title(f"3. Bridge gaps (≤7 px)\ngreen = added ({bridged.sum():,} px)")
    ax[4].imshow(sk); ax[4].set_title(f"4. Skeleton (1-px centre-line)\n{int(skel.sum()):,} px")
    ax[5].imshow(sk)
    if g is not None:
        for typ, col in (("junction", "red"), ("endpoint", "cyan")):
            pts = [d["pos"] for _, d in g.nodes(data=True) if d.get("node_type") == typ]
            if pts:
                a_ = np.array(pts); ax[5].scatter(a_[:, 1], a_[:, 0], s=9, c=col, label=typ)
        ax[5].legend(loc="lower left", fontsize=8)
    ax[5].set_title(f"5. Graph: {res['n_nodes']} nodes · {res['n_edges']} segments")
    for a in ax: a.axis("off")
    fig.suptitle("Step 4 — Clean and connect", fontweight="bold"); fig.tight_layout(rect=(0, .03, 1, .90)); _stamp(fig, res)
    return fig


def conf_lines(res):
    c = res["confidence"]; kind = "mean softmax confidence" if res["engine"] == "trained" and "AVResNet" in res["engine_label"] else "mean heuristic score"
    return ["A/V CLASSIFICATION", "", f"artery segments : {c['av_artery_segments']}", f"vein segments   : {c['av_vein_segments']}", "",
            f"{kind} : {c['av_mean_score']:.3f}", f"segments with score ≥ 0.8 : {c['av_pct_score_ge_0.8']:.1f}%", "",
            f"artery pixels : {res['extra']['artery_pixels']:,}", f"vein pixels   : {res['extra']['vein_pixels']:,}", "",
            f"seg. mean P(vessel) on mask : {c['seg_mean_vessel_prob']:.3f}", f"uncertain pixels (0.3–0.7) : {c['seg_uncertain_pct']:.1f}%"]


def fig_artery_vein(res):
    ov = render_av_overlay(res["rgb"], res["artery_mask"], res["vein_mask"])
    fig, ax = plt.subplots(1, 4, figsize=(21, 5.2))
    ax[0].imshow(res["rgb"]); ax[0].set_title("Input fundus")
    ax[1].imshow(res["vessel_mask"], cmap="gray"); ax[1].set_title("Vessel mask")
    ax[2].imshow(ov); ax[2].set_title("Artery (red) / vein (blue)")
    ax[2].legend(handles=[Patch(color="#dc2828", label="Artery"), Patch(color="#2850dc", label="Vein")], loc="lower right", fontsize=9)
    ax[3].text(0, 1, "\n".join(conf_lines(res)), va="top", family="monospace", fontsize=11)
    for a in ax: a.axis("off")
    fig.suptitle("Step 5 — Artery or vein?", fontweight="bold"); fig.tight_layout(rect=(0, .03, 1, .90)); _stamp(fig, res)
    return fig


def bar_chart(ax, bm, title):
    keys = list(BIOMARKER_LABELS)
    for i, k in enumerate(keys):
        lo, hi = NORMAL_RANGES[k]; v = bm[k]
        ax.barh(i, v, color=_bm_color(k, v), height=.55)
        ax.plot([lo, lo], [i - .4, i + .4], "k--", lw=1); ax.plot([hi, hi], [i - .4, i + .4], "k--", lw=1)
        ax.text(max(v, hi) * 1.02, i, f"{v:.3f}   (ref {lo}–{hi})", va="center", fontsize=9)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([f"{BIOMARKER_LABELS[k][0]} ({BIOMARKER_LABELS[k][1]})" for k in keys], fontsize=10)
    ax.invert_yaxis(); ax.set_title(title, fontsize=12); ax.set_xlim(0, max(max(bm.values()), 14) * 1.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("value   (dashed = reference range · green = inside · red = outside)")


def fig_biomarkers(res):
    fig, ax = plt.subplots(figsize=(11, 4.8)); bar_chart(ax, res["biomarkers"], "Six biomarkers")
    fig.tight_layout(rect=(0, .03, 1, .90)); _stamp(fig, res)
    return fig


def fig_complete(res):
    fig, ax = plt.subplots(1, 5, figsize=(26, 5), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.5]})
    ax[0].imshow(res["rgb"]); ax[0].set_title("Original fundus")
    h = ax[1].imshow(res["prob"], cmap="hot", vmin=0, vmax=1); ax[1].set_title("Vessel probability map"); fig.colorbar(h, ax=ax[1], fraction=.046)
    ax[2].imshow(res["vessel_mask"], cmap="gray"); ax[2].set_title(f"Post-processed mask\n({res['extra']['mask_coverage_%']:.1f}% coverage)")
    ax[3].imshow(render_av_overlay(res["rgb"], res["artery_mask"], res["vein_mask"])); ax[3].set_title("A/V classification\nred = artery · blue = vein")
    for a in ax[:4]: a.axis("off")
    bar_chart(ax[4], res["biomarkers"], "Biomarkers (green = normal · red = abnormal)")
    fig.suptitle(f"Complete output — {res['model_name']}", fontweight="bold"); fig.tight_layout(rect=(0, .03, 1, .90)); _stamp(fig, res)
    return fig


def fig_models(res):
    """Only when several trained networks were found: probability maps side by side."""
    om = res.get("other_models") or {}
    if not om:
        return None
    items = [(res["model_name"], res["prob"], res["vessel_mask"])] + [(k, v["prob"], v["mask"]) for k, v in om.items()]
    fig, ax = plt.subplots(2, len(items), figsize=(5 * len(items), 9), squeeze=False)
    for i, (n, p, m) in enumerate(items):
        ax[0, i].imshow(p, cmap="hot", vmin=0, vmax=1); ax[0, i].set_title(f"{n}\nprobability map")
        ax[1, i].imshow(m, cmap="gray"); ax[1, i].set_title("mask")
        ax[0, i].axis("off"); ax[1, i].axis("off")
    fig.suptitle("Model comparison", fontweight="bold"); fig.tight_layout(rect=(0, .02, 1, .93)); _stamp(fig, res)
    return fig


FIGS = [("01_preprocessing", "Step 1 · Clean the image (green · CLAHE · Retinex)", fig_preprocessing),
        ("02_segmentation", "Step 3 · Vessel segmentation — probability map, mask, overlay", fig_segmentation),
        ("03_clean_and_connect", "Step 4 · Clean and connect — threshold → denoise → bridge → skeleton → graph", fig_clean_connect),
        ("04_artery_vein", "Step 5 · Artery (red) or vein (blue)?", fig_artery_vein),
        ("05_biomarkers", "Step 6 · Six biomarkers vs reference ranges", fig_biomarkers),
        ("06_complete_output", "Complete output ", fig_complete),
        ("07_model_comparison", "Model comparison (only when several trained networks are present)", fig_models)]


def values_table(res):
    rows = []
    for k, (label, unit) in BIOMARKER_LABELS.items():
        lo, hi = NORMAL_RANGES[k]; v = float(res["biomarkers"][k])
        rows.append({"key": k, "label": label, "unit": unit, "value": round(v, 4), "ref_lo": lo, "ref_hi": hi,
                     "status": "within" if lo <= v <= hi else "OUTSIDE"})
    return rows


def render_all(res, folder, dpi=110) -> dict:
    """Write every figure + the individual masks + results.json into `folder`. Returns {key: {title, file}}."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, title, fn in FIGS:
        fig = fn(res)
        if fig is None:
            continue
        p = folder / f"{key}.png"; fig.savefig(p, dpi=dpi, bbox_inches="tight"); plt.close(fig)
        out[key] = {"title": title, "file": p.name}
    for nm, im in (("mask_vessels", res["vessel_mask"]), ("mask_artery", res["artery_mask"]), ("mask_vein", res["vein_mask"]),
                   ("prob_map", (res["prob"] * 255).astype(np.uint8)), ("skeleton", (res["skeleton"] > 0).astype(np.uint8) * 255)):
        cv2.imwrite(str(folder / f"{nm}.png"), im)
    doc = {"image": res.get("name"), "original_size_hw": res.get("orig_hw"), "engine": res["engine"], "engine_label": res["engine_label"],
           "segmentation_model": res["model_name"], "biomarkers": values_table(res), "graph": {"nodes": res["n_nodes"], "segments": res["n_edges"]},
           "confidence": res["confidence"], "extra": res["extra"], "av_score_kind": res["av_score_kind"]}
    (folder / "results.json").write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    return out
