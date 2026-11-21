"""
fallback_runner.py — what the step scripts do when the trained .pt files are NOT available.
They compute the same figure from YOUR image with fallback_engine.py (classical, no deep learning)
and stamp it so nobody mistakes it for the trained networks.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from _common import pick_images, save_fig, read_bgr, HERE
import fallback_engine as fe
import result_figures as rf
from biomarkers import BIOMARKER_LABELS, NORMAL_RANGES

NOTICE = ("\n  [fallback] trained weights not found -> using the CLASSICAL pipeline (ridge filter + colour heuristic).\n"
          "             Numbers will differ from the trained results.  Copy your .pt files into weights/ for the trained networks.\n")


def _res(p, a):
    r = fe.analyse_classical(read_bgr(p), getattr(a, "threshold", 0.5)); r["name"] = Path(p).stem
    return r


def print_biomarkers(res):
    print(f"\n  {res['name']}  [{res['engine_label']}]\n  {'biomarker':<22}{'value':>10}   reference      status")
    for r in rf.values_table(res):
        print(f"  {r['label']:<22}{r['value']:>10.4f}   {r['ref_lo']:>5.2f} – {r['ref_hi']:<5.2f}  {r['status']}")


def run(kind: str, a, out: Path, n_default: int = 1):
    """kind: segmentation | clean | av | biomarkers | full"""
    print(NOTICE)
    for p in pick_images(a, n_default=n_default):
        res = _res(p, a)
        c = res["confidence"]
        print(f"  {p.name}: vessel pixels {(res['vessel_mask']>0).sum():,} ({res['extra']['mask_coverage_%']:.2f}%) · graph {res['n_nodes']} nodes / "
              f"{res['n_edges']} segments · artery {c['av_artery_segments']} / vein {c['av_vein_segments']} segments")
        if kind == "full":
            rf.render_all(res, out / p.stem); print(f"  saved -> {out / p.stem}  (all figures + results.json)")
            print_biomarkers(res); continue
        fn = {"segmentation": rf.fig_segmentation, "clean": rf.fig_clean_connect, "av": rf.fig_artery_vein,
              "biomarkers": rf.fig_biomarkers}[kind]
        save_fig(fn(res), out / f"{kind}_{p.stem}_classical.png", getattr(a, "show", False))
        if kind == "biomarkers":
            print_biomarkers(res)
            with open(out / f"biomarkers_{p.stem}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rf.values_table(res)[0])); w.writeheader(); w.writerows(rf.values_table(res))


ARROW = {"vessel_density": "↓", "mean_vessel_width": "↓", "vessel_tortuosity": "↑", "avr": "↓", "mean_artery_width": "↓", "mean_vein_width": "varies"}


def run_normal_vs_disease(a, out: Path, first_image):
    """without trained weights: sample_images/normal vs sample_images/disease (or --normal/--disease)."""
    print(NOTICE)
    res = {}
    for grp in ("normal", "disease"):
        p = first_image(getattr(a, grp), grp)
        r = _res(p, a); res[grp] = r
        print(f"[{grp}] {p}")
        save_fig(rf.fig_complete(r), out / f"complete_output_{grp}_classical.png", getattr(a, "show", False))
        print_biomarkers(r)
    keys = list(BIOMARKER_LABELS)
    fig, ax = plt.subplots(figsize=(13, 4.4)); ax.axis("off")
    cells = [[f"{BIOMARKER_LABELS[k][0]} ({BIOMARKER_LABELS[k][1]})", f"{NORMAL_RANGES[k][0]}–{NORMAL_RANGES[k][1]}",
              f"{res['normal']['biomarkers'][k]:.3f}", f"{res['disease']['biomarkers'][k]:.3f}", ARROW[k],
              "↓" if res['disease']['biomarkers'][k] < res['normal']['biomarkers'][k] else "↑"] for k in keys]
    t = ax.table(cellText=cells, colLabels=["Biomarker", "Reference", "Normal image", "Disease image", "Expected in disease", "Measured change"],
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.8)
    ax.set_title("Normal vs disease — six biomarkers (classical fallback, ONE image per group)", fontsize=12)
    fig.text(.5, .01, res["normal"]["engine_label"], ha="center", fontsize=8, color="#b36b00")
    save_fig(fig, out / "normal_vs_disease_classical.png", getattr(a, "show", False))
    with open(out / "normal_vs_disease_classical.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["biomarker", "reference_lo", "reference_hi", "normal", "disease"])
        for k in keys: w.writerow([k, *NORMAL_RANGES[k], res["normal"]["biomarkers"][k], res["disease"]["biomarkers"][k]])
    print("\n  NOTE: one image per group — shows that the pipeline works, not a clinical statistic.")
