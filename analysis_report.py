"""
analysis_report.py — Terminal reports, figures and CSV/Markdown tables built
from the dicts returned by analysis_core.analyse_image().
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from biomarkers import (BIOMARKER_LABELS, NORMAL_RANGES, render_av_overlay,
                        _MODEL_DISPLAY)

_SHORT = {"vessel_density": "VD (%)", "mean_vessel_width": "MVW (px)",
          "vessel_tortuosity": "Tort.", "avr": "AVR",
          "mean_artery_width": "MAW (px)", "mean_vein_width": "MVnW (px)"}


def disp(k):
    return _MODEL_DISPLAY.get(k, "Ensemble (mean of 3)" if k == "ensemble" else k)


# ─────────────────────────────────────────────────────────────────────────────
# Terminal
# ─────────────────────────────────────────────────────────────────────────────
def print_image_report(res: dict, primary: str, log=print):
    log("\n" + "═" * 78)
    log(f"  RESULTS FOR: {res['name']}   (sha256 {res['sha256'][:12]}…)")
    log("═" * 78)
    for key, m in res["models"].items():
        star = "  ◄ primary" if key == primary else ""
        log(f"\n  ── {disp(key)}{star}")
        log(f"     vessel pixels {int((m['vessel_mask']>0).sum()):>7,} "
            f"({(m['vessel_mask']>0).mean()*100:.2f}%)   graph: {m['n_nodes']} nodes / "
            f"{m['n_edges']} segments   (post-proc+AV {m['seconds']:.1f}s)")
        c = m["confidence"]
        log(f"     seg confidence : mean P(vessel) on vessel px = {c['seg_mean_vessel_prob']:.3f}   "
            f"uncertain px (0.3<p<0.7) = {c['seg_uncertain_pct']:.1f}%")
        log(f"     A/V confidence : {c['av_artery_segments']} artery / {c['av_vein_segments']} vein segments   "
            f"mean softmax conf = {c['av_mean_conf']:.3f}   "
            f"segments ≥0.8 conf = {c['av_pct_conf_ge_0.8']:.1f}%")
        if "gt_metrics" in m:
            g = m["gt_metrics"]
            log(f"     vs ground truth: Dice {g['dice']:.4f}  IoU {g['iou']:.4f}  "
                f"Sens {g['sensitivity']:.4f}  Spec {g['specificity']:.4f}")
        log(f"     {'biomarker':<22}{'value':>10}   reference range     status")
        for k, v in m["biomarkers"].items():
            lo, hi = NORMAL_RANGES[k]; nm, un = BIOMARKER_LABELS[k]
            log(f"     {nm:<22}{v:>10.4f}   {lo:>5.2f} – {hi:<5.2f} {un:<5} "
                f"{'within' if lo <= v <= hi else 'OUTSIDE'}")
    if res.get("agreement"):
        log("\n  Model agreement (Dice between final vessel masks):")
        for k, v in res["agreement"].items():
            log(f"     {k:<44}{v:.4f}")
    d = res.get("disease")
    if d:
        log("\n  ODIR-5K disease-category classifier (image-level CNN):")
        for c, p in d["probs"].items():
            log(f"     {c:<14}{p*100:6.2f}%  {'█' * int(p * 30)}")
        log(f"     → predicted: {d['pred']}  (confidence {d['confidence']*100:.1f}%, "
            f"normalised entropy {d['norm_entropy']:.3f})")


def print_summary_table(items: list, primary: str, log=print):
    """items: [(group, res)]"""
    log("\n" + "═" * 100)
    log(f"  BIOMARKER SUMMARY — {disp(primary)}")
    log("═" * 100)
    h = f"{'image':<22}{'group':<10}" + "".join(f"{_SHORT[k]:>11}" for k in BIOMARKER_LABELS)
    h += f"{'  seg-conf':>10}{'  A/V-conf':>10}  disease-CNN"
    log(h); log("─" * 100)
    for grp, r in items:
        m = r["models"][primary]; b = m["biomarkers"]; c = m["confidence"]
        d = r.get("disease")
        row = f"{r['name'][:21]:<22}{grp[:9]:<10}" + "".join(f"{b[k]:>11.3f}" for k in BIOMARKER_LABELS)
        row += f"{c['seg_mean_vessel_prob']:>10.3f}{c['av_mean_conf']:>10.3f}  "
        row += f"{d['pred']} ({d['confidence']*100:.0f}%)" if d else "—"
        log(row)
    log("─" * 100)
    log("  reference: " + "   ".join(f"{_SHORT[k]} {lo:g}–{hi:g}" for k, (lo, hi) in NORMAL_RANGES.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def _skeleton_overlay(rgb, skel):
    o = rgb.copy()
    sk = cv2.dilate(skel.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    o[sk] = (255, 230, 0)
    return o


def _gt_overlay(vessel, gt):
    p, g = vessel > 0, gt > 0
    o = np.zeros(vessel.shape + (3,), np.uint8)
    o[p & g] = (255, 255, 255); o[p & ~g] = (230, 60, 60); o[~p & g] = (60, 120, 255)
    return o


def _biomarker_panel(ax, bms):
    ax.set_xlim(-1.0, 2.7); ax.set_ylim(len(bms) - .4, -.8)
    ax.axvspan(0, 1, color="#2ca02c", alpha=.15, lw=0)
    for i, (k, v) in enumerate(bms.items()):
        lo, hi = NORMAL_RANGES[k]; z = (v - lo) / (hi - lo)
        ok = lo <= v <= hi
        ax.plot([-1, 2.7], [i, i], color="#cccccc", lw=.6, zorder=0)
        ax.scatter([float(np.clip(z, -1, 2))], [i], s=90, zorder=3,
                   color="#2ca02c" if ok else "#d62728", edgecolor="k")
        ax.text(2.04, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(bms)))
    ax.set_yticklabels([f"{BIOMARKER_LABELS[k][0]}\n[{NORMAL_RANGES[k][0]:g}–{NORMAL_RANGES[k][1]:g} {BIOMARKER_LABELS[k][1]}]"
                        for k in bms], fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["ref. low", "ref. high"], fontsize=8)
    ax.set_title("Biomarkers (green band = reference range)", fontsize=10)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def figure_main(res, key, save_path: Path, show=False):
    m = res["models"][key]; rgb = res["rgb"]; mc = res["multichannel"]
    fig = plt.figure(figsize=(21, 15.5))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=.22, wspace=.12)
    fig.suptitle(f"{res['name']}  ·  {disp(key)}", fontsize=16, fontweight="bold", y=.985)

    def im(ax, img, title, cmap=None, **kw):
        h = ax.imshow(img, cmap=cmap, **kw); ax.set_title(title, fontsize=10); ax.axis("off"); return h

    im(fig.add_subplot(gs[0, 0]), rgb, "Input fundus (512×512)")
    im(fig.add_subplot(gs[0, 1]), mc[:, :, 0], "Ch0: green channel", "gray")
    im(fig.add_subplot(gs[0, 2]), mc[:, :, 1], "Ch1: CLAHE green (clip 1.5, 8×8)", "gray")
    im(fig.add_subplot(gs[0, 3]), mc[:, :, 2], "Ch2: Retinex green (σ=50)", "gray")

    ax = fig.add_subplot(gs[1, 0])
    h = im(ax, m["prob"], "Vessel probability map (sigmoid)", "hot", vmin=0, vmax=1)
    plt.colorbar(h, ax=ax, fraction=.046, pad=.02)
    pct = (m["vessel_mask"] > 0).mean() * 100
    im(fig.add_subplot(gs[1, 1]), m["vessel_mask"], f"Post-processed mask ({pct:.1f}% vessel)", "gray")

    ax = fig.add_subplot(gs[1, 2]); ax.imshow(_skeleton_overlay(rgb, m["skeleton"]))
    g = m["graph"]
    if g is not None:
        for typ, col in (("junction", "red"), ("endpoint", "cyan")):
            pts = [d["pos"] for _, d in g.nodes(data=True)
                   if d.get("node_type") == typ and "pos" in d]
            if pts:
                a = np.array(pts); ax.scatter(a[:, 1], a[:, 0], s=7, c=col, label=typ)
        ax.legend(loc="lower left", fontsize=7)
    ax.set_title(f"Skeleton + graph ({m['n_nodes']} nodes, {m['n_edges']} segments)", fontsize=10)
    ax.axis("off")

    av = render_av_overlay(rgb, m["artery_mask"], m["vein_mask"])
    ax = fig.add_subplot(gs[1, 3]); im(ax, av, "A/V classification (red=artery, blue=vein)")
    c = m["confidence"]
    ax.text(.02, .02, f"artery seg: {c['av_artery_segments']}  vein seg: {c['av_vein_segments']}\n"
            f"mean conf {c['av_mean_conf']:.2f}", transform=ax.transAxes, fontsize=8, color="w",
            bbox=dict(boxstyle="round", fc="k", alpha=.55))

    _biomarker_panel(fig.add_subplot(gs[2, 0:2]), m["biomarkers"])

    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    lines = ["QUALITY / CONFIDENCE", "",
             f"mean P(vessel) on mask : {c['seg_mean_vessel_prob']:.3f}",
             f"uncertain pixels       : {c['seg_uncertain_pct']:.1f}%",
             f"A/V mean softmax conf  : {c['av_mean_conf']:.3f}",
             f"A/V segments ≥0.8 conf : {c['av_pct_conf_ge_0.8']:.1f}%"]
    if "gt_metrics" in m:
        gm = m["gt_metrics"]
        lines += ["", "VS EXPERT GROUND TRUTH", f"Dice {gm['dice']:.4f}   IoU {gm['iou']:.4f}",
                  f"Sens {gm['sensitivity']:.4f}  Spec {gm['specificity']:.4f}"]
    lines += ["", "PROVENANCE", f"sha256 {res['sha256'][:20]}…",
              f"orig size {res['orig_hw'][1]}×{res['orig_hw'][0]}"]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)

    ax = fig.add_subplot(gs[2, 3])
    d = res.get("disease")
    if d:
        names = list(d["probs"]); vals = [d["probs"][n] * 100 for n in names]
        cols = ["#1f77b4" if n == d["pred"] else "#b0b0b0" for n in names]
        ax.barh(names, vals, color=cols); ax.set_xlim(0, 100); ax.invert_yaxis()
        for i, v in enumerate(vals): ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)
        ax.set_title(f"ODIR-5K CNN → {d['pred']}", fontsize=10)
        ax.set_xlabel("probability (%)")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    else:
        ax.axis("off"); ax.text(.5, .5, "disease classifier\nnot loaded", ha="center", color="gray")

    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)


def figure_model_comparison(res, save_path: Path, show=False):
    keys = list(res["models"]); has_gt = "gt_mask" in res
    ncol = 4 if has_gt else 3
    fig, axes = plt.subplots(len(keys), ncol, figsize=(4.2 * ncol, 4.2 * len(keys)), squeeze=False)
    for r, k in enumerate(keys):
        m = res["models"][k]
        axes[r, 0].imshow(m["prob"], cmap="hot", vmin=0, vmax=1); axes[r, 0].set_title(f"{disp(k)}\nprobability", fontsize=9)
        axes[r, 1].imshow(m["vessel_mask"], cmap="gray")
        axes[r, 1].set_title(f"mask ({(m['vessel_mask']>0).mean()*100:.1f}%)", fontsize=9)
        axes[r, 2].imshow(render_av_overlay(res["rgb"], m["artery_mask"], m["vein_mask"]))
        axes[r, 2].set_title("A/V", fontsize=9)
        if has_gt:
            axes[r, 3].imshow(_gt_overlay(m["vessel_mask"], res["gt_mask"]))
            gm = m["gt_metrics"]
            axes[r, 3].set_title(f"white=TP red=FP blue=FN\nDice {gm['dice']:.3f}", fontsize=9)
        for a in axes[r]: a.axis("off")
    fig.suptitle(f"{res['name']} — model comparison", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)


def save_image_outputs(res, out_dir: Path, primary: str, all_figures=False, show=False, log=print):
    d = Path(out_dir) / res["name"]; d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "input_512.png"), cv2.cvtColor(res["rgb"], cv2.COLOR_RGB2BGR))
    keys = list(res["models"]) if all_figures else [primary]
    for k in keys:
        m = res["models"][k]
        figure_main(res, k, d / f"figure_{k}.png", show=show and k == primary)
        cv2.imwrite(str(d / f"mask_{k}.png"), m["vessel_mask"])
        cv2.imwrite(str(d / f"prob_{k}.png"), (m["prob"] * 255).astype(np.uint8))
        cv2.imwrite(str(d / f"av_overlay_{k}.png"),
                    cv2.cvtColor(render_av_overlay(res["rgb"], m["artery_mask"], m["vein_mask"]),
                                 cv2.COLOR_RGB2BGR))
    if len(res["models"]) > 1:
        figure_model_comparison(res, d / "model_comparison.png", show=False)
    js = {"image": res["name"], "sha256": res["sha256"], "orig_hw": res["orig_hw"],
          "models": {k: {"biomarkers": m["biomarkers"], "confidence": m["confidence"],
                         "gt_metrics": m.get("gt_metrics"), "graph_nodes": m["n_nodes"],
                         "graph_edges": m["n_edges"]} for k, m in res["models"].items()},
          "agreement": res.get("agreement"), "disease": res.get("disease")}
    (d / "results.json").write_text(json.dumps(js, indent=2))
    log(f"    saved → {d}/  ({', '.join(p.name for p in sorted(d.glob('figure_*.png')))})")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Tables over many images / groups
# ─────────────────────────────────────────────────────────────────────────────
def write_summary_csv(items, out_dir: Path):
    rows = []
    for grp, r in items:
        for k, m in r["models"].items():
            row = {"image": r["name"], "group": grp, "model": k, "sha256": r["sha256"][:16]}
            row.update(m["biomarkers"]); row.update(m["confidence"])
            if "gt_metrics" in m: row.update({f"gt_{a}": b for a, b in m["gt_metrics"].items()})
            if r.get("disease"):
                row["disease_pred"] = r["disease"]["pred"]; row["disease_conf"] = r["disease"]["confidence"]
                row.update({f"p_{c}": p for c, p in r["disease"]["probs"].items()})
            rows.append(row)
    if not rows: return None
    fields = list(dict.fromkeys(f for r in rows for f in r))
    p = Path(out_dir) / "summary_all_images.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return p


def figure_biomarker_overview(items, primary, save_path: Path):
    names = [r["name"] for _, r in items]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, k in zip(axes.ravel(), BIOMARKER_LABELS):
        vals = [r["models"][primary]["biomarkers"][k] for _, r in items]
        lo, hi = NORMAL_RANGES[k]
        ax.axhspan(lo, hi, color="#2ca02c", alpha=.15)
        ax.bar(range(len(vals)), vals, color=["#2ca02c" if lo <= v <= hi else "#d62728" for v in vals])
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{BIOMARKER_LABELS[k][0]} ({BIOMARKER_LABELS[k][1]})", fontsize=10)
    fig.suptitle(f"Biomarkers per image — {disp(primary)}", fontweight="bold")
    fig.tight_layout(); fig.savefig(save_path, dpi=130); plt.close(fig)


def group_comparison(items, primary: str, out_dir: Path, log=print):
    """Mean ± SD per group (e.g. normal vs disease) + Mann–Whitney U test."""
    groups: dict[str, list] = {}
    for g, r in items:
        groups.setdefault(g, []).append(r["models"][primary]["biomarkers"])
    if len(groups) < 2:
        return None
    from scipy.stats import mannwhitneyu
    gnames = list(groups)
    a, b = gnames[0], gnames[1]
    if "normal" in gnames:
        a = "normal"; b = next((g for g in gnames if g != "normal"), b)
    log("\n" + "═" * 92)
    log(f"  GROUP COMPARISON — {disp(primary)} — measured from the images in each folder")
    log("═" * 92)
    hdr = f"{'biomarker':<22}" + "".join(f"{g + ' (n=' + str(len(groups[g])) + ')':>22}" for g in gnames)
    hdr += f"{'ref. range':>14}{'p (MWU)':>10}"
    log(hdr); log("─" * 92)
    rows = []
    for k in BIOMARKER_LABELS:
        cells, row = "", {"biomarker": k}
        for g in gnames:
            v = np.array([x[k] for x in groups[g]])
            cells += f"{v.mean():>12.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:<6.3f}"
            row[f"{g}_mean"] = float(v.mean()); row[f"{g}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            row[f"{g}_n"] = len(v)
        va = [x[k] for x in groups[a]]; vb = [x[k] for x in groups[b]]
        try:
            p = float(mannwhitneyu(va, vb, alternative="two-sided").pvalue) if len(va) > 1 and len(vb) > 1 else float("nan")
        except Exception:
            p = float("nan")
        row["p_mwu"] = p
        lo, hi = NORMAL_RANGES[k]
        log(f"{BIOMARKER_LABELS[k][0]:<22}{cells}{f'{lo:g}–{hi:g}':>14}{p:>10.3f}")
        rows.append(row)
    log("─" * 92)
    nmin = min(len(v) for v in groups.values())
    if nmin < 10:
        log(f"  NOTE: only n={nmin} images in the smallest group — differences are descriptive, "
            f"not statistically conclusive.")
    out_dir = Path(out_dir)
    with open(out_dir / "group_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = ["| Biomarker | " + " | ".join(f"{g} (n={len(groups[g])})" for g in gnames) + " | Reference | p |",
          "|---|" + "---|" * (len(gnames) + 2)]
    for row, k in zip(rows, BIOMARKER_LABELS):
        md.append(f"| {BIOMARKER_LABELS[k][0]} ({BIOMARKER_LABELS[k][1]}) | " +
                  " | ".join(f"{row[g + '_mean']:.3f} ± {row[g + '_sd']:.3f}" for g in gnames) +
                  f" | {NORMAL_RANGES[k][0]:g}–{NORMAL_RANGES[k][1]:g} | {row['p_mwu']:.3f} |")
    (out_dir / "group_comparison.md").write_text("\n".join(md))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, k in zip(axes.ravel(), BIOMARKER_LABELS):
        for i, g in enumerate(gnames):
            v = [x[k] for x in groups[g]]
            ax.scatter(np.random.default_rng(0).normal(i, .05, len(v)), v, alpha=.7, s=30)
            ax.hlines(np.mean(v), i - .25, i + .25, color="k", lw=2)
        lo, hi = NORMAL_RANGES[k]; ax.axhspan(lo, hi, color="#2ca02c", alpha=.15)
        ax.set_xticks(range(len(gnames))); ax.set_xticklabels(gnames)
        ax.set_title(f"{BIOMARKER_LABELS[k][0]} ({BIOMARKER_LABELS[k][1]})")
    fig.suptitle(f"Biomarkers by group — {disp(primary)} (each dot = one real image; bar = mean)", fontweight="bold")
    fig.tight_layout(); fig.savefig(out_dir / "group_comparison.png", dpi=130); plt.close(fig)
    log(f"  saved → {out_dir}/group_comparison.(csv|md|png)")
    return rows
