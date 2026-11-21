"""— "Step 2: More data from little data"   (flip · rotate · brightness/contrast · elastic warp)

    python_augmentation_examples.py
    python_augmentation_examples.py --input img.png --mask img_vessel_mask.png

No trained weights needed (re-implemented with OpenCV using the same ranges as
augmentation.py so it runs without PyTorch/albumentations). If --mask is given,
the SAME geometric change is applied to the mask (second row) — "same change is
applied to photo and answer mask".

Saves outputs/_augmentation_examples/<name>_augmentations.png
"""
from _common import *                       # noqa: F401,F403
from preprocessing import build_multichannel


def elastic(img, rng, alpha=40, sigma=6, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha * 3
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha * 3
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(img, (x + dx).astype(np.float32), (y + dy).astype(np.float32),
                     interp, borderMode=cv2.BORDER_REFLECT_101)


def affine(img, ang, scale, tx, ty, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale); M[:, 2] += (tx * w, ty * h)
    return cv2.warpAffine(img, M, (w, h), flags=interp, borderMode=cv2.BORDER_REFLECT_101)


def build_variants(rng):
    """(title, image_fn, mask_fn_or_None, kind)"""
    ang = float(rng.uniform(8, 15)); sc = 1.04
    seed_el = int(rng.integers(0, 10**6))
    return [
        ("Horizontal flip", lambda i: cv2.flip(i, 1), lambda m: cv2.flip(m, 1)),
        ("Vertical flip", lambda i: cv2.flip(i, 0), lambda m: cv2.flip(m, 0)),
        ("Rotate 90°", lambda i: cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE), lambda m: cv2.rotate(m, cv2.ROTATE_90_CLOCKWISE)),
        (f"Rotate {ang:.0f}° + scale + shift", lambda i: affine(i, ang, sc, .03, -.03),
         lambda m: affine(m, ang, sc, .03, -.03, cv2.INTER_NEAREST)),
        ("Brightness +20% / contrast", lambda i: cv2.convertScaleAbs(i, alpha=1.2, beta=255 * 0.15), None),
        ("Gamma 0.8", lambda i: (255 * (i / 255.0) ** 0.8).astype(np.uint8), None),
        ("CLAHE (random clip)", lambda i: cv2.createCLAHE(3.0, (8, 8)).apply(cv2.cvtColor(i, cv2.COLOR_RGB2GRAY))[..., None].repeat(3, 2), None),
        ("Elastic warp (α=40, σ=6)", lambda i: elastic(i, np.random.default_rng(seed_el)),
         lambda m: elastic(m, np.random.default_rng(seed_el), interp=cv2.INTER_NEAREST)),
        ("Occlusion (coarse dropout)", lambda i: _dropout(i, np.random.default_rng(seed_el)), None),
        ("Sensor noise", lambda i: np.clip(i + np.random.default_rng(seed_el).normal(0, 9, i.shape), 0, 255).astype(np.uint8), None),
    ]


def _dropout(img, rng):
    o = img.copy(); h, w = o.shape[:2]
    for _ in range(int(rng.integers(3, 8))):
        s = int(rng.integers(16, 33)); r = int(rng.integers(0, h - s)); c = int(rng.integers(0, w - s))
        o[r:r + s, c:c + s] = 0
    return o


def main():
    ap = make_parser(__doc__)
    ap.add_argument("--mask", default=None, help="optional vessel mask of the same image")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = out_dir(a, __file__)

    for p in pick_images(a):
        rgb = cv2.cvtColor(to_512(read_bgr(p)), cv2.COLOR_BGR2RGB)
        mask = None
        if a.mask:
            m = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize((m > 127).astype(np.uint8) * 255, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        rng = np.random.default_rng(a.seed)
        variants = build_variants(rng)
        n = len(variants) + 1; ncol = 4; nrow = -(-n // ncol) * (2 if mask is not None else 1)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow), squeeze=False)
        for ax in axes.ravel(): ax.axis("off")
        step = 2 if mask is not None else 1

        def put(idx, im, title, is_mask=False):
            r, c = (idx // ncol) * step, idx % ncol
            axes[r, c].imshow(im, cmap="gray" if is_mask else None); axes[r, c].set_title(title, fontsize=10)
            return r, c

        put(0, rgb, "Original photo")
        if mask is not None:
            axes[0 + 1, 0].imshow(mask, cmap="gray"); axes[1, 0].set_title("Original mask", fontsize=10)
        for i, (title, fi, fm) in enumerate(variants, start=1):
            r, c = put(i, fi(rgb.copy()), title)
            if mask is not None:
                axes[r + 1, c].imshow(fm(mask.copy()) if fm else mask, cmap="gray")
                axes[r + 1, c].set_title("mask: same change" if fm else "mask: unchanged (photo-only change)", fontsize=9)
        fig.suptitle(f"Augmentation examples — {p.stem}   (≈20× more training samples)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out / f"{p.stem}_augmentations.png", a.show)


if __name__ == "__main__":
    main()
