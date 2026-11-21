"""— "What the output looks like"
    Original fundus | vessel probability map | binary mask (threshold 0.5) | overlay on original

    python_segmentation_output.py                       # sample_images/, all networks found
    python_segmentation_output.py --input my_fundus.png --models attention_unet
    python_segmentation_output.py --input folder_with_images --limit 2

Needs the trained segmentation checkpoints in weights/.
Saves outputs/_segmentation_output/
    segmentation_<model>.png                     The figure (one row per image)
    prob_<image>_<model>.png, mask_<image>_<model>.png, overlay_<image>_<model>.png
"""
from _common import *                       # noqa: F401,F403
from matplotlib.patches import Patch


def main():
    ap = make_parser(__doc__); add_model_args(ap)
    a = ap.parse_args()
    out = out_dir(a, __file__)
    if not have_trained(a, need_av=False):
        import fallback_runner as fbr; fbr.run('segmentation', a, out, n_default=2); return

    from inference import predict_single
    models, _ = load_segmentation_models(a)
    imgs = pick_images(a, n_default=2)
    rows = {k: [] for k in models}
    for p in imgs:
        bgr = to_512(read_bgr(p)); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for k, m in models.items():
            prob = predict_single(m, bgr)
            mask = (prob >= a.threshold).astype(np.uint8) * 255
            ov = rgb.copy(); ov[mask > 0] = (0.5 * ov[mask > 0] + 0.5 * np.array([235, 40, 40])).astype(np.uint8)
            rows[k].append((p.stem, rgb, prob, mask, ov))
            cv2.imwrite(str(out / f"prob_{p.stem}_{k}.png"), (prob * 255).astype(np.uint8))
            cv2.imwrite(str(out / f"mask_{p.stem}_{k}.png"), mask)
            cv2.imwrite(str(out / f"overlay_{p.stem}_{k}.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
            print(f"  {p.stem:<24} {SEG_NAME[k]:<16} vessel pixels {(mask>0).sum():>7,} ({(mask>0).mean()*100:.2f}%)  "
                  f"P(vessel) mean on mask {prob[mask>0].mean() if (mask>0).any() else 0:.3f}")
    for k, items in rows.items():
        fig, ax = plt.subplots(len(items), 4, figsize=(18, 4.6 * len(items)), squeeze=False)
        for r, (nm, rgb, prob, mask, ov) in enumerate(items):
            ax[r, 0].imshow(rgb); ax[r, 0].set_title(f"Original fundus\n{nm}", fontsize=10)
            h = ax[r, 1].imshow(prob, cmap="hot", vmin=0, vmax=1); ax[r, 1].set_title("Vessel probability map", fontsize=10)
            fig.colorbar(h, ax=ax[r, 1], fraction=.046, pad=.02)
            ax[r, 2].imshow(mask, cmap="gray"); ax[r, 2].set_title(f"Binary mask (threshold={a.threshold})", fontsize=10)
            ax[r, 3].imshow(ov); ax[r, 3].set_title("Overlay on original", fontsize=10)
            ax[r, 3].legend(handles=[Patch(color="#eb2828", label="Detected vessels")], loc="lower right", fontsize=8)
            for c in range(4): ax[r, c].axis("off")
        fig.suptitle(f"Retinal Vessel Segmentation — {SEG_NAME[k]}", fontsize=14)
        fig.tight_layout()
        save_fig(fig, out / f"segmentation_{k}.png", a.show)


if __name__ == "__main__":
    main()
