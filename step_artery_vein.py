"""(andpicture) — "Step 5: Artery or vein?"   red = arteries, blue = veins

    python_artery_vein.py
    python_artery_vein.py --input my_fundus.png --primary unetplusplus_scse

Needs weights/: at least one segmentation checkpoint + av_cnn.pt.
Saves outputs/_artery_vein/
    av_<image>_<model>.png          figure: input | vessel mask | A/V overlay | confidence box
    av_overlay_<image>_<model>.png  just the red/blue overlay picture
    artery_mask_*.png, vein_mask_*.png
and prints the artery/vein segment counts and A/V confidence scores.
"""
from _common import *                       # noqa: F401,F403
import analysis_core as core
from biomarkers import render_av_overlay
from matplotlib.patches import Patch


def main():
    ap = make_parser(__doc__); add_model_args(ap)
    ap.add_argument("--primary", default="unetplusplus_scse")
    a = ap.parse_args()
    out = out_dir(a, __file__)
    if not have_trained(a, need_av=True):
        import fallback_runner as fbr; fbr.run('av', a, out, n_default=1); return

    models, found = load_segmentation_models(a)
    av = load_av_model(found)
    key = a.primary if a.primary in models else next(iter(models))
    print(f"[model] segmentation = {SEG_NAME[key]}")
    for p in pick_images(a):
        r = core.analyse_image(p, {key: models[key]}, av, None, threshold=a.threshold, include_ensemble=False)
        m = r["models"][key]; c = m["confidence"]
        ov = render_av_overlay(r["rgb"], m["artery_mask"], m["vein_mask"])
        tag = f"{p.stem}_{key}"
        cv2.imwrite(str(out / f"av_overlay_{tag}.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"artery_mask_{tag}.png"), m["artery_mask"])
        cv2.imwrite(str(out / f"vein_mask_{tag}.png"), m["vein_mask"])
        fig, ax = plt.subplots(1, 4, figsize=(21, 5.4))
        ax[0].imshow(r["rgb"]); ax[0].set_title("Input fundus")
        ax[1].imshow(m["vessel_mask"], cmap="gray"); ax[1].set_title("Vessel mask (from segmentation)")
        ax[2].imshow(ov); ax[2].set_title("Artery / vein  (32×32 patch → AVResNet18)")
        ax[2].legend(handles=[Patch(color="#dc2828", label="Artery"), Patch(color="#2850dc", label="Vein")],
                     loc="lower right", fontsize=9)
        ax[3].axis("off")
        ax[3].text(0, 1, "\n".join([
            "A/V CLASSIFICATION", "",
            f"artery segments : {c['av_artery_segments']}", f"vein segments   : {c['av_vein_segments']}", "",
            f"mean softmax confidence : {c['av_mean_conf']:.3f}",
            f"segments with conf ≥ 0.8 : {c['av_pct_conf_ge_0.8']:.1f}%", "",
            f"artery pixels : {int((m['artery_mask']>0).sum()):,}", f"vein pixels   : {int((m['vein_mask']>0).sum()):,}"]),
            va="top", family="monospace", fontsize=12)
        for i in range(3): ax[i].axis("off")
        fig.suptitle(f"Artery–vein classification — {p.stem} · {SEG_NAME[key]}", fontsize=14, fontweight="bold")
        save_fig(fig, out / f"av_{tag}.png", a.show)
        print(f"  artery {c['av_artery_segments']} / vein {c['av_vein_segments']} segments, "
              f"mean confidence {c['av_mean_conf']:.3f}, ≥0.8: {c['av_pct_conf_ge_0.8']:.1f}%")


if __name__ == "__main__":
    main()
