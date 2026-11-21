"""— "Step 1: Clean the image"   (green channel · CLAHE · Retinex · stacked input)

    python_preprocessing_channels.py
    python_preprocessing_channels.py --input my_fundus.png

No trained weights needed. Saves into outputs/_preprocessing_channels/:
    <name>_channels.png        figure: original | green | CLAHE | Retinex | stacked
    <name>_ch0_green.png  <name>_ch1_clahe.png  <name>_ch2_retinex.png  <name>_stacked_512.png
"""
from _common import *                       # noqa: F401,F403
from preprocessing import build_multichannel


def main():
    ap = make_parser(__doc__)
    a = ap.parse_args()
    out = out_dir(a, __file__)

    for p in pick_images(a):
        bgr = to_512(read_bgr(p))
        mc = build_multichannel(bgr)                     # (512,512,3): green | CLAHE | Retinex
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        name = p.stem
        for i, tag in enumerate(("ch0_green", "ch1_clahe", "ch2_retinex")):
            cv2.imwrite(str(out / f"{name}_{tag}.png"), mc[:, :, i])
        cv2.imwrite(str(out / f"{name}_stacked_512.png"), mc)

        fig, ax = plt.subplots(1, 5, figsize=(21, 4.6))
        panels = [(rgb, "Original fundus", None),
                  (mc[:, :, 0], "1. Green channel\nvessels are darkest in green light", "gray"),
                  (mc[:, :, 1], "2. CLAHE\nboosts local contrast (clip 1.5, 8×8)", "gray"),
                  (mc[:, :, 2], "3. Retinex\nremoves uneven lighting (σ=50)", "gray"),
                  (mc, "Stack of the 3 → one 512×512 network input", None)]
        for a_, (im, t, cm) in zip(ax, panels):
            a_.imshow(im, cmap=cm); a_.set_title(t, fontsize=10); a_.axis("off")
        fig.suptitle(f"Pre-processing — {name}", fontsize=13, fontweight="bold")
        save_fig(fig, out / f"{name}_channels.png", a.show)
        for i, tag in enumerate(("green", "CLAHE", "Retinex")):
            ch = mc[:, :, i]
            print(f"  channel {i} {tag:<8} range [{ch.min():3d}, {ch.max():3d}]  mean {ch.mean():6.1f}")


if __name__ == "__main__":
    main()
