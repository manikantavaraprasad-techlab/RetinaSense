"""— "Step 4: Clean and connect"
    threshold 0.5 → remove noise (<30 px) → bridge gaps (≤7 px, only true gaps) → skeleton → graph

Two ways to run it:

  A) with the trained networks (weights/ folder filled):
        python_clean_and_connect.py --input fundus.png
  B) WITHOUT weights, from a probability map saved earlier by
     (outputs/_segmentation_output/prob_<name>_<model>.png) or any
     grey-scale vessel-probability / mask image:
        python_clean_and_connect.py --prob outputs/_segmentation_output/prob_x_unetplusplus.png

Saves outputs/_clean_and_connect/<name>_clean_and_connect.png  (+ each stage as its own PNG)
"""
from _common import *                       # noqa: F401,F403
from postprocessing import _bridge_endpoints, skeletonize_mask, skeleton_to_graph


def stages(prob, thr=0.5, min_area=30, gap_px=7):
    s1 = ((prob >= thr) * 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(s1, connectivity=8)
    s2 = np.zeros_like(s1)
    for l in range(1, n):
        if st[l, cv2.CC_STAT_AREA] >= min_area:
            s2[lab == l] = 255
    s3 = _bridge_endpoints(s2, prob, gap_px=gap_px, prob_threshold=thr)
    n, lab, st, _ = cv2.connectedComponentsWithStats(s3, connectivity=8)
    s3f = np.zeros_like(s3)
    for l in range(1, n):
        if st[l, cv2.CC_STAT_AREA] >= 10:
            s3f[lab == l] = 255
    skel = skeletonize_mask(s3f)
    graph = skeleton_to_graph(skel)
    return s1, s2, s3f, skel, graph


def draw(name, rgb, prob, thr, out, show):
    s1, s2, s3, skel, g = stages(prob, thr)
    removed = (s1 > 0) & (s2 == 0); bridged = (s3 > 0) & (s2 == 0)
    r2 = np.stack([s2] * 3, -1).copy(); r2[removed] = [255, 60, 60]       # removed noise = red
    r3 = np.stack([s3] * 3, -1).copy(); r3[bridged] = [60, 255, 60]       # bridged gaps  = green
    sk = rgb.copy(); sk[cv2.dilate(skel.astype(np.uint8), np.ones((2, 2))) > 0] = [255, 255, 0]
    fig, ax = plt.subplots(1, 6, figsize=(30, 5.4))
    ax[0].imshow(rgb); ax[0].set_title("Input", fontsize=11)
    ax[1].imshow(s1, cmap="gray"); ax[1].set_title(f"1. Threshold {thr}\n{(s1>0).sum():,} px", fontsize=11)
    ax[2].imshow(r2); ax[2].set_title(f"2. Remove noise (<30 px)\nred = removed ({removed.sum():,} px)", fontsize=11)
    ax[3].imshow(r3); ax[3].set_title(f"3. Bridge gaps (≤7 px, true gaps only)\ngreen = added ({bridged.sum():,} px)", fontsize=11)
    ax[4].imshow(sk); ax[4].set_title(f"4. Skeleton (1-px centre-line)\n{int(skel.sum()):,} px", fontsize=11)
    ax[5].imshow(sk)
    if g is not None:
        for typ, col in (("junction", "red"), ("endpoint", "cyan")):
            pts = [d["pos"] for _, d in g.nodes(data=True) if d.get("node_type") == typ]
            if pts:
                a_ = np.array(pts); ax[5].scatter(a_[:, 1], a_[:, 0], s=9, c=col, label=typ)
        ax[5].legend(loc="lower left", fontsize=8)
        ax[5].set_title(f"5. Graph: {g.number_of_nodes()} nodes · {g.number_of_edges()} segments", fontsize=11)
    for a_ in ax: a_.axis("off")
    fig.suptitle(f"Clean and connect — {name}", fontsize=14, fontweight="bold")
    save_fig(fig, out / f"{name}_clean_and_connect.png", show)
    for tag, im in (("1_threshold", s1), ("2_noise_removed", s2), ("3_gaps_bridged", s3),
                    ("4_skeleton", (skel * 255).astype(np.uint8))):
        cv2.imwrite(str(out / f"{name}_{tag}.png"), im)
    print(f"  {name}: threshold {(s1>0).sum():,}px → denoise {(s2>0).sum():,}px → bridged {(s3>0).sum():,}px → "
          f"skeleton {int(skel.sum()):,}px → graph {g.number_of_nodes() if g else 0} nodes / {g.number_of_edges() if g else 0} segments")


def main():
    ap = make_parser(__doc__)
    add_model_args(ap)
    ap.add_argument("--prob", default=None, help="saved probability map / mask (grey-scale image) — runs without weights")
    ap.add_argument("--rgb", default=None, help="fundus image to draw on (only with --prob; optional)")
    a = ap.parse_args()
    out = out_dir(a, __file__)


    if a.prob:
        g = cv2.imread(a.prob, cv2.IMREAD_GRAYSCALE)
        if g is None: sys.exit(f"ERROR: cannot read {a.prob}")
        prob = cv2.resize(g, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        rgb = cv2.cvtColor(to_512(read_bgr(a.rgb)), cv2.COLOR_BGR2RGB) if a.rgb else np.stack([(prob * 255).astype(np.uint8)] * 3, -1)
        draw(Path(a.prob).stem, rgb, prob, a.threshold, out, a.show); return

    if not have_trained(a, need_av=False):
        import fallback_runner as fbr; fbr.run('clean', a, out); return
    from inference import predict_single
    models, _ = load_segmentation_models(a)
    key = "unetplusplus_scse" if "unetplusplus_scse" in models else next(iter(models))
    print(f"[model] using {SEG_NAME[key]}")
    for p in pick_images(a):
        bgr = to_512(read_bgr(p))
        draw(f"{p.stem}_{key}", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), predict_single(models[key], bgr), a.threshold, out, a.show)


if __name__ == "__main__":
    main()
