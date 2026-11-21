"""— "Artery–vein results"   Artery Dice vs Vein Dice per image (one panel per network)

Needs: trained weights/ (segmentation + av_cnn.pt) and the DRIVE-AV test images with A/V labels.
    python_av_results.py --av-root D:\\data\\DRIVE_AV           # folder that contains  test/
    (or set the environment variable RETINAL_DRIVE_AV_ROOT)
    python_av_results.py --csv outputs/_av_results/av_results.csv   # re-plot only

Saves outputs/_av_results/av_results.csv and artery_vs_vein_dice.png.

NOTE: the 65.8 % "mean 20-fold accuracy" on  is produced by  run_av_pipeline.py
(leave-one-image-out TRAINING of the A/V CNN, slow). This script measures the finished pipeline.
"""
from _common import *                       # noqa: F401,F403
import csv, os


def plot(rows, out, show):
    keys = list(dict.fromkeys(r["model"] for r in rows))
    fig, ax = plt.subplots(1, len(keys), figsize=(6.6 * len(keys), 6), sharex=True, sharey=True, squeeze=False)
    cols = ["#4a90e2", "#d6608a", "#5aa469"]
    for i, k in enumerate(keys):
        rr = [r for r in rows if r["model"] == k]; x = [float(r["artery_dice"]) for r in rr]; y = [float(r["vein_dice"]) for r in rr]
        ax[0, i].scatter(x, y, s=55, color=cols[i % 3]); lim = (min(x + y) - .03, max(x + y) + .05)
        ax[0, i].plot(lim, lim, "--", color="gray", lw=1, label="artery = vein")
        for xi, yi, r in zip(x, y, rr): ax[0, i].annotate(r["image"], (xi, yi), fontsize=6, xytext=(4, 3), textcoords="offset points")
        ax[0, i].set_title(SEG_NAME.get(k, k)); ax[0, i].set_xlabel("Artery Dice"); ax[0, i].grid(alpha=.3); ax[0, i].legend(fontsize=8)
    ax[0, 0].set_ylabel("Vein Dice")
    fig.suptitle("Artery Dice vs Vein Dice — Per Image", fontsize=14, fontweight="bold")
    fig.tight_layout(); save_fig(fig, out / "artery_vs_vein_dice.png", show)


def main():
    ap = make_parser(__doc__, need_image=False); add_model_args(ap)
    ap.add_argument("--av-root", default=None); ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    out = out_dir(a, __file__)

    if a.csv:
        plot(list(csv.DictReader(open(a.csv))), out, a.show); return
    if a.av_root: os.environ["RETINAL_DRIVE_AV_ROOT"] = a.av_root
    models, found = load_segmentation_models(a)
    av = load_av_model(found)
    from config import CFG
    from av_dataset import load_drive_av_samples, decode_av_label_drive
    from av_visualisation import labels_to_masks
    from postprocessing import postprocess_mask, skeletonize_mask, skeleton_to_graph
    from analysis_core import classify_edges
    from inference import predict_single
    kw = {"root": a.av_root} if a.av_root else {}
    samples = load_drive_av_samples(splits=("test",), **kw)
    if not samples:
        sys.exit("ERROR: DRIVE-AV test images not found. Use --av-root <folder containing test/> or RETINAL_DRIVE_AV_ROOT")
    rows = []
    for key in [k for k in ("attention_unet", "unetplusplus_scse") if k in models] or list(models):
        for s in samples:
            bgr = to_512(read_bgr(s["image_path"]))
            lab = cv2.resize(cv2.imread(s["label_path"], cv2.IMREAD_COLOR), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            ga, gv = decode_av_label_drive(lab)
            vm = postprocess_mask(predict_single(models[key], bgr), threshold=a.threshold)
            g = skeleton_to_graph(skeletonize_mask(vm))
            L, _, _ = classify_edges(bgr, vm, g, av)
            am, vmk = labels_to_masks(g, L, (IMG_SIZE, IMG_SIZE), vm)
            d = lambda p, t: float(2 * (p & t).sum() / max(p.sum() + t.sum(), 1))
            pa, pv = am > 0, vmk > 0
            rows.append({"model": key, "image": s["name"], "artery_dice": d(pa, ga), "vein_dice": d(pv, gv)})
            print(f"  {SEG_NAME[key]:<16}{s['name']:<20} artery Dice {rows[-1]['artery_dice']:.3f}  vein Dice {rows[-1]['vein_dice']:.3f}")
    with open(out / "av_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for key in dict.fromkeys(r["model"] for r in rows):
        rr = [r for r in rows if r["model"] == key]
        print(f"  MEAN {SEG_NAME[key]:<16} artery Dice {np.mean([r['artery_dice'] for r in rr]):.4f}  vein Dice {np.mean([r['vein_dice'] for r in rr]):.4f}")
    plot(rows, out, a.show)


if __name__ == "__main__":
    main()
