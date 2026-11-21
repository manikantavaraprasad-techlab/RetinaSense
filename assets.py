"""
assets.py — Locate trained checkpoints automatically.

Instead of relying on exact file names, every *.pt / *.pth file found in the
search folders is opened and classified by its CONTENT:

    ckpt["arch"] == "AttentionUNet"      -> segmentation 'attention_unet'
    ckpt["arch"] == "UnetPlusPlusDS"     -> segmentation 'unetplusplus'
    ckpt["arch"] == "unetplusplus_scse"  -> segmentation 'unetplusplus_scse'
    ckpt has 'in_channels' + 'model_state' (no 'arch') -> 'av_cnn'
    ckpt["task"] == "disease_classifier" -> 'disease'

So you can simply drop the files downloaded from the Kaggle notebook output
into the  weights/  folder, whatever they are called.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

SEG_KINDS = ("attention_unet", "unetplusplus", "unetplusplus_scse")
_ARCH_TO_KIND = {
    "AttentionUNet":     "attention_unet",
    "UnetPlusPlusDS":    "unetplusplus",
    "unetplusplus_scse": "unetplusplus_scse",
}


def default_search_dirs(extra: str | None = None) -> list[Path]:
    here = Path(__file__).resolve().parent
    dirs = []
    if extra:
        dirs.append(Path(extra))
    env = os.environ.get("RETINAL_WEIGHTS")
    if env:
        dirs.append(Path(env))
    dirs += [here / "weights", here, here / "outputs" / "checkpoints",
             Path.cwd() / "weights", Path.cwd(),
             Path("/kaggle/working"), Path("/kaggle/working/outputs/checkpoints")]
    seen, out = set(), []
    for d in dirs:
        d = d.resolve() if d.exists() else d
        if d not in seen and d.exists():
            seen.add(d); out.append(d)
    return out


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _classify(path: Path):
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return None, None
    if not isinstance(ck, dict):
        return None, None
    if ck.get("task") == "disease_classifier":
        return "disease", ck
    arch = ck.get("arch")
    if arch in _ARCH_TO_KIND:
        return _ARCH_TO_KIND[arch], ck
    if "model_state" in ck and "in_channels" in ck and "patch_size" in ck:
        return "av_cnn", ck
    return None, None


def discover_checkpoints(search_dirs: list[Path] | None = None,
                         verbose: bool = True) -> dict:
    """Returns {kind: Path}. If several files of one kind exist the one with the
    best stored dice (segmentation) or the newest one is used."""
    search_dirs = search_dirs or default_search_dirs()
    cands: dict[str, list] = {}
    scanned = set()
    for d in search_dirs:
        # top level of every dir; recursive only inside a folder called weights
        it = d.rglob("*") if d.name.lower() == "weights" else d.glob("*")
        for p in it:
            if p.suffix.lower() not in (".pt", ".pth") or p in scanned:
                continue
            scanned.add(p)
            kind, ck = _classify(p)
            if kind:
                cands.setdefault(kind, []).append(
                    (float(ck.get("best_dice", 0) or 0), p.stat().st_mtime, p))
    found = {}
    for kind, lst in cands.items():
        lst.sort(key=lambda t: (t[0], t[1]), reverse=True)
        found[kind] = lst[0][2]
    if verbose:
        print("[assets] searched:", ", ".join(str(d) for d in search_dirs))
        for k in (*SEG_KINDS, "av_cnn", "disease"):
            print(f"[assets]   {k:<18}: {found.get(k, '— not found —')}")
    return found
