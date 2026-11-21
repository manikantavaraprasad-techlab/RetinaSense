"""— "Normal vs disease retina" and "Complete output: normal vs disease"
    per image:  original | probability map | post-processed mask | A/V classification | biomarker bars
    plus a side-by-side table showing the direction of change (density ↓ widths ↓ tortuosity ↑ AVR ↓)

    python_23_normal_vs_disease.py
    python_23_normal_vs_disease.py --normal path/normal.png --disease path/disease.png

Defaults: first image of sample_images/normal/ and sample_images/disease/.
Needs weights/ (segmentation + av_cnn.pt). Saves outputs/_23_normal_vs_disease/
    complete_output_normal.png, complete_output_disease.png, normal_vs_disease.png, normal_vs_disease.csv
"""
from _common import *                       # noqa: F401,F403
import csv
import analysis_core as core
from biomarkers import (BIOMARKER_LABELS, NORMAL_RANGES, visualise_results, _bm_color)

# direction  expect in disease: density ↓, widths ↓, tortuosity ↑, AV ratio ↓)
ARROW = {"vessel_density": "↓", "mean_vessel_width": "↓", "vessel_tortuosity": "↑",
         "avr": "↓", "mean_artery_width": "↓", "mean_vein_width": "varies"}


def first_image(arg, folder):
    if arg:
        if not Path(arg).exists(): sys.exit(f"ERROR: not found: {arg}")
        return Path(arg)
    imgs = list_images(HERE / "sample_images" / folder)
    if not imgs: sys.exit(f"ERROR: put an image in sample_images/{folder}/ or pass --{folder} path")
    return imgs[0]


def main():
    ap = make_parser(__doc__, need_image=False); add_model_args(ap)
    ap.add_argument("--normal", default=None); ap.add_argument("--disease", default=None)
    ap.add_argument("--primary", default="unetplusplus_scse")
    a = ap.parse_args()
    out = out_dir(a, __file__)
    if not have_trained(a):
        import fallback_runner as fbr; fbr.run_normal_vs_disease(a, out, first_image); return

    models, found = load_segmentation_models(a)
    av = load_av_model(found)
    key = a.primary if a.primary in models else next(iter(models))
    res = {}
    for grp in ("normal", "disease"):
        p = first_image(getattr(a, grp), grp)
        print(f"\n[{grp}] {p}")
        r = core.analyse_image(p, {key: models[key]}, av, None, threshold=a.threshold, include_ensemble=False)
        m = r["models"][key]; res[grp] = (p, m["biomarkers"])
        visualise_results(f"{grp.capitalize()} Retina", key, r["rgb"], m["prob"], m["vessel_mask"],
                          m["artery_mask"], m["vein_mask"], m["biomarkers"], save_path=out / f"complete_output_{grp}.png")
    keys = list(BIOMARKER_LABELS)
    fig, ax = plt.subplots(figsize=(13, 4.4)); ax.axis("off")
    cells = [[f"{BIOMARKER_LABELS[k][0]} ({BIOMARKER_LABELS[k][1]})", f"{NORMAL_RANGES[k][0]}–{NORMAL_RANGES[k][1]}",
              f"{res['normal'][1][k]:.3f}", f"{res['disease'][1][k]:.3f}", ARROW[k],
              "↓" if res['disease'][1][k] < res['normal'][1][k] else "↑"] for k in keys]  # measured on these 2 images
    t = ax.table(cellText=cells, colLabels=["Biomarker", "Reference", "Normal image", "Disease image",
                                            "Expected in disease", "Measured change"], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.8)
    for i, k in enumerate(keys, start=1):
        for j, g in ((2, "normal"), (3, "disease")):
            t[i, j].set_facecolor("#d5f0d5" if _bm_color(k, res[g][1][k]) == "#2ca02c" else "#f6d0d0")
    ax.set_title("Normal vs disease — six biomarkers (green cell = inside reference range, red = outside)", fontsize=12)
    save_fig(fig, out / "normal_vs_disease.png", a.show)
    with open(out / "normal_vs_disease.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["biomarker", "reference_lo", "reference_hi", "normal", "disease"])
        for k in keys: w.writerow([k, *NORMAL_RANGES[k], res["normal"][1][k], res["disease"][1][k]])
    print("\n  NOTE: each number above is measured from ONE image per group; it shows the pipeline works,"
          "\n  it is not a clinical statistic. For group statistics see README §3 (run_analysis.py on samples/).")


if __name__ == "__main__":
    main()
