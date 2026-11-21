"""— "Step 6: Six biomarkers"
    vessel density · mean vessel width · tortuosity · AV ratio · artery width · vein width
    (green bar = inside the reference range, red = outside)

    python_biomarkers.py
    python_biomarkers.py --input my_fundus.png          # or a folder of images

Needs weights/ (segmentation + av_cnn.pt). Saves outputs/_biomarkers/
    biomarkers_<image>.png     bar chart with reference ranges
    biomarkers.csv, biomarkers.json     all six values for every image / model
and prints the table (value, reference range, within/OUTSIDE).
"""
from _common import *                       # noqa: F401,F403
import csv, json
import analysis_core as core
from biomarkers import BIOMARKER_LABELS, NORMAL_RANGES, _bm_color


def bar_chart(ax, bm, title):
    keys = list(BIOMARKER_LABELS)
    for i, k in enumerate(keys):
        lo, hi = NORMAL_RANGES[k]; v = bm[k]
        ax.barh(i, v, color=_bm_color(k, v), height=.55)
        ax.plot([lo, lo], [i - .4, i + .4], "k--", lw=1); ax.plot([hi, hi], [i - .4, i + .4], "k--", lw=1)
        ax.text(max(v, hi) * 1.02, i, f"{v:.3f}   (ref {lo}–{hi})", va="center", fontsize=9)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{BIOMARKER_LABELS[k][0]} ({BIOMARKER_LABELS[k][1]})" for k in keys], fontsize=10)
    ax.invert_yaxis(); ax.set_title(title, fontsize=12); ax.set_xlim(0, max(max(bm.values()), 14) * 1.45)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("value   (dashed lines = reference range, green = normal · red = abnormal)")


def main():
    ap = make_parser(__doc__); add_model_args(ap)
    ap.add_argument("--primary", default="unetplusplus_scse")
    a = ap.parse_args()
    out = out_dir(a, __file__)
    if not have_trained(a, need_av=True):
        import fallback_runner as fbr; fbr.run('biomarkers', a, out, n_default=1); return

    models, found = load_segmentation_models(a)
    av = load_av_model(found)
    key = a.primary if a.primary in models else next(iter(models))
    rows, js = [], {}
    for p in pick_images(a, n_default=1):
        r = core.analyse_image(p, {key: models[key]}, av, None, threshold=a.threshold, include_ensemble=False)
        bm = r["models"][key]["biomarkers"]
        print(f"\n  {p.name}  ({SEG_NAME[key]})\n  {'biomarker':<22}{'value':>10}   reference      status")
        for k, v in bm.items():
            lo, hi = NORMAL_RANGES[k]
            print(f"  {BIOMARKER_LABELS[k][0]:<22}{v:>10.4f}   {lo:>5.2f} – {hi:<5.2f}  {'within' if lo <= v <= hi else 'OUTSIDE'}")
        fig, ax = plt.subplots(figsize=(11, 4.6))
        bar_chart(ax, bm, f"Six biomarkers — {p.stem} · {SEG_NAME[key]}")
        save_fig(fig, out / f"biomarkers_{p.stem}.png", a.show)
        rows.append({"image": p.stem, "model": key, **bm}); js[p.stem] = bm
    with open(out / "biomarkers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "biomarkers.json").write_text(json.dumps(js, indent=2))
    print(f"  saved -> {out/'biomarkers.csv'}")


if __name__ == "__main__":
    main()
