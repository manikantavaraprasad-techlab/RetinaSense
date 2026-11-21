"""
evaluate_segmentation.py — REGENERATE the results table (Segmentation
results") from the trained checkpoints, on images the networks have not trained on.

Default: rebuilds the SAME validation split that training used
(seed 42, 20 % of each of DRIVE / STARE / CHASEDB1, see dataset.build_loaders)
and evaluates every checkpoint on it.

    python evaluate_segmentation.py
    python evaluate_segmentation.py --root /path/to/retinal-vessel-fundus-dataset-collection
    python evaluate_segmentation.py --images DRIVE/test/images --masks DRIVE/test/1st_manual
    python evaluate_segmentation.py --av          # + pixel-level A/V Dice on DRIVE-AV test

Outputs  outputs/evaluation/
    results_table.csv / .md     (mean ± SD over images, per model)
    per_image_metrics.csv
    results_bar_chart.png, qualitative_<model>.png
    av_results.csv              (with --av)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2


def val_split_samples(root, subsets, val_fraction=0.2):
    """Replicates dataset.build_loaders() split exactly; returns [(name, bgr, gt_mask)]."""
    from config import CFG
    from dataset import RetinalDataset
    samples = []
    for sub in subsets:
        try:
            ds = RetinalDataset(root, subset=sub)
        except Exception as e:
            print(f"  [SKIP] {sub}: {e}"); continue
        n = len(ds); n_val = max(1, int(n * val_fraction))
        idx = np.random.default_rng(CFG["seed"]).permutation(list(range(n))).tolist()
        for i in idx[n - n_val:]:
            p = ds.images[i]
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            gt = (cv2.imread(str(ds._resolve_mask(p.stem)), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
            samples.append((f"{sub}_{p.stem}", bgr, gt))
    return samples


def folder_samples(img_dir, mask_dir):
    from analysis_core import IMG_EXT, read_bgr, read_gray, find_gt_mask
    out = []
    for p in sorted(Path(img_dir).iterdir()):
        if p.suffix.lower() not in IMG_EXT: continue
        g = find_gt_mask(p, Path(mask_dir))
        if g is None: print(f"  (no mask for {p.name})"); continue
        out.append((p.stem, read_bgr(p), (read_gray(g) > 127).astype(np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--images"); ap.add_argument("--masks")
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--out", default="outputs/evaluation")
    ap.add_argument("--av", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    from config import CFG, ROOT_DIR, SEG_SUBSETS
    from assets import default_search_dirs, discover_checkpoints, SEG_KINDS
    from inference import load_model, predict_single
    from analysis_core import seg_metrics
    from postprocessing import postprocess_mask
    from biomarkers import _MODEL_DISPLAY

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    found = discover_checkpoints(default_search_dirs(a.weights_dir))
    kinds = [k for k in SEG_KINDS if k in found]
    if not kinds:
        sys.exit("ERROR: no segmentation checkpoints found (put the .pt files in ./weights/).")
    models = {k: load_model(str(found[k]), device=CFG["device"]) for k in kinds}

    if a.images:
        samples = folder_samples(a.images, a.masks)
    else:
        samples = val_split_samples(a.root or ROOT_DIR, SEG_SUBSETS)
    if not samples:
        sys.exit("ERROR: no evaluation images found — check --root / --images / --masks.")
    print(f"\nEvaluating {len(kinds)} models on {len(samples)} held-out images\n")

    S = CFG["img_size"]; rows = []
    store = {k: [] for k in kinds}
    for name, img, gt in samples:
        bgr = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
        g = cv2.resize((gt > .5).astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
        for k, m in models.items():
            prob = predict_single(m, bgr)
            raw = (prob > a.threshold).astype(np.uint8)
            post = postprocess_mask(prob, threshold=a.threshold)
            for tag, mk in (("raw", raw), ("post", post)):
                met = seg_metrics(mk, g)
                rows.append({"image": name, "model": k, "stage": tag, **met})
            store[k].append((name, bgr, g, prob, post))
        print(f"  {name:<28}" + "  ".join(
            f"{k[:10]}:{[r for r in rows if r['image']==name and r['model']==k and r['stage']=='raw'][0]['dice']:.3f}" for k in kinds))

    with open(out / "per_image_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ── table ────────────────────────────────────────────────────────────────
    mets = ["dice", "iou", "sensitivity", "specificity"]
    table = []
    for k in kinds:
        for stage in ("raw", "post"):
            rr = [r for r in rows if r["model"] == k and r["stage"] == stage]
            table.append({"model": _MODEL_DISPLAY.get(k, k), "stage": stage, "n": len(rr),
                          **{f"{m}_mean": float(np.mean([r[m] for r in rr])) for m in mets},
                          **{f"{m}_sd": float(np.std([r[m] for r in rr], ddof=1)) if len(rr) > 1 else 0.0 for m in mets}})
    with open(out / "results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    md = ["| Model | Output | n | Dice | IoU | Sens. | Spec. |", "|---|---|---|---|---|---|---|"]
    for t in table:
        md.append(f"| {t['model']} | {'threshold only' if t['stage']=='raw' else 'post-processed'} | {t['n']} | " +
                  " | ".join(f"{t[m+'_mean']:.4f} ± {t[m+'_sd']:.4f}" for m in mets) + " |")
    (out / "results_table.md").write_text("\n".join(md))
    print("\n" + "═" * 92 + "\n  SEGMENTATION RESULTS (computed now, held-out images)\n" + "═" * 92)
    print(f"  {'model':<18}{'output':<16}{'n':>3}{'Dice':>16}{'IoU':>16}{'Sens':>16}{'Spec':>16}")
    for t in table:
        print(f"  {t['model']:<18}{('threshold' if t['stage']=='raw' else 'post-proc'):<16}{t['n']:>3}" +
              "".join(f"{t[m+'_mean']:>9.4f}±{t[m+'_sd']:<6.3f}" for m in mets))

    fig, ax = plt.subplots(figsize=(9, 4.5)); w = .8 / len(kinds)
    raw = [t for t in table if t["stage"] == "raw"]
    for i, t in enumerate(raw):
        ax.bar(np.arange(4) + i * w, [t[m + "_mean"] for m in mets], w,
               yerr=[t[m + "_sd"] for m in mets], capsize=3, label=t["model"])
    ax.set_xticks(np.arange(4) + w * (len(kinds) - 1) / 2); ax.set_xticklabels(["Dice", "IoU", "Sens", "Spec"])
    ax.set_ylim(0, 1.05); ax.legend(); ax.set_title(f"Held-out results (n={len(samples)} images, mean ± SD)")
    fig.tight_layout(); fig.savefig(out / "results_bar_chart.png", dpi=140); plt.close(fig)

    for k in kinds:                                           # qualitative panels
        pick = store[k][:4]
        fig, axes = plt.subplots(len(pick), 4, figsize=(14, 3.4 * len(pick)), squeeze=False)
        for r, (name, bgr, g, prob, post) in enumerate(pick):
            ov = np.zeros((S, S, 3), np.uint8); p = post > 0; gg = g > 0
            ov[p & gg] = 255; ov[p & ~gg] = (230, 60, 60); ov[~p & gg] = (60, 120, 255)
            axes[r, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); axes[r, 0].set_title(name, fontsize=8)
            axes[r, 1].imshow(prob, cmap="hot", vmin=0, vmax=1); axes[r, 1].set_title("probability", fontsize=8)
            axes[r, 2].imshow(g, cmap="gray"); axes[r, 2].set_title("expert mask", fontsize=8)
            axes[r, 3].imshow(ov); axes[r, 3].set_title("pred vs expert (white TP, red FP, blue FN)", fontsize=8)
            for x in axes[r]: x.axis("off")
        fig.suptitle(_MODEL_DISPLAY.get(k, k)); fig.tight_layout()
        fig.savefig(out / f"qualitative_{k}.png", dpi=110); plt.close(fig)

    if a.av:
        run_av_eval(out, models, found, a)
    print(f"\nSaved → {out}/")


def run_av_eval(out, models, found, a):
    """Pixel-level A/V Dice of the full pipeline on DRIVE-AV *test* images."""
    from config import CFG
    from av_dataset import load_drive_av_samples, decode_av_label_drive
    from av_classification import load_av_cnn
    from av_visualisation import labels_to_masks
    from postprocessing import postprocess_mask, skeletonize_mask, skeleton_to_graph
    from analysis_core import classify_edges, read_bgr
    from inference import predict_single
    if "av_cnn" not in found:
        print("  [av] av_cnn.pt not found — skipped"); return
    av = load_av_cnn(str(found["av_cnn"]), device=CFG["device"])
    key = "unetplusplus_scse" if "unetplusplus_scse" in models else next(iter(models))
    samples = load_drive_av_samples(splits=("test",))
    if not samples: print("  [av] DRIVE-AV test folder not found (set RETINAL_DRIVE_AV_ROOT)"); return
    S = CFG["img_size"]; rows = []
    for s in samples:
        bgr = cv2.resize(read_bgr(s["image_path"]), (S, S), interpolation=cv2.INTER_AREA)
        lab = cv2.resize(cv2.imread(s["label_path"], cv2.IMREAD_COLOR), (S, S), interpolation=cv2.INTER_NEAREST)
        ga, gv = decode_av_label_drive(lab)
        prob = predict_single(models[key], bgr)
        vm = postprocess_mask(prob, threshold=a.threshold)
        g = skeleton_to_graph(skeletonize_mask(vm))
        L, _, _ = classify_edges(bgr, vm, g, av)
        am, vmk = labels_to_masks(g, L, (S, S), vm)
        d = lambda p, t: float(2 * (p & t).sum() / max(p.sum() + t.sum(), 1))
        pa, pv = am > 0, vmk > 0
        # edge-level accuracy against GT labels on vessel pixels that have a GT label
        ok = ((pa & ga).sum() + (pv & gv).sum()); tot = ((pa | pv) & (ga | gv)).sum()
        rows.append({"image": s["name"], "artery_dice": d(pa, ga), "vein_dice": d(pv, gv),
                     "pixel_AV_accuracy_on_labelled": float(ok / max(tot, 1))})
        print(f"  {s['name']:<22} artery Dice {rows[-1]['artery_dice']:.3f}  vein Dice {rows[-1]['vein_dice']:.3f}"
              f"  A/V px-acc {rows[-1]['pixel_AV_accuracy_on_labelled']:.3f}")
    with open(out / "av_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("  MEAN  artery Dice %.4f  vein Dice %.4f  px-acc %.4f" % tuple(
        np.mean([r[k] for r in rows]) for k in ("artery_dice", "vein_dice", "pixel_AV_accuracy_on_labelled")))


if __name__ == "__main__":
    main()
