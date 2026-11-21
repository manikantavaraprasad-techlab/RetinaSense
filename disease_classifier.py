"""
disease_classifier.py — ODIR-5K disease-category classifier
(Normal / Diabetic / Hypertensive), the item listed as "future work" in the
project notes.

NOTE ON "STAGE": ODIR-5K provides *category* labels per eye (plus free-text
keywords), not a numeric severity grade for every disease, so the classifier
predicts the category the project calls "stage": Normal, Diabetic, Hypertensive.

Contains:
    preprocess_fundus()   — crop black border, square-pad, CLAHE, resize
    DiseaseNet            — EfficientNet-B0 (ImageNet pre-trained) with new head
    load_disease_model()  — load checkpoint produced by train_disease_classifier.py
    predict_disease()     — real softmax probabilities for one image
    parse_odir()          — build a per-eye table (path, label, patient) from ODIR-5K
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

CLASSES = ["Normal", "Diabetic", "Hypertensive"]
IMG_SIZE = 320
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ─────────────────────────────────────────────────────────────────────────────
def preprocess_fundus(bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """BGR uint8 -> RGB uint8 (size×size): FOV crop, square pad, local contrast."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 12)
    if len(ys) > 100:
        bgr = bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = bgr.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = bgr
    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)


def to_tensor(rgb_u8: np.ndarray) -> torch.Tensor:
    x = (rgb_u8.astype(np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class DiseaseNet(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), pretrained: bool = True):
        super().__init__()
        import torchvision
        weights = None
        if pretrained:
            try:
                weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            self.net = torchvision.models.efficientnet_b0(weights=weights)
        except Exception as e:           # no internet -> cannot fetch ImageNet weights
            print(f"[disease] could not load ImageNet weights ({e}); "
                  f"training from random initialisation.")
            self.net = torchvision.models.efficientnet_b0(weights=None)
        in_f = self.net.classifier[1].in_features
        self.net.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, n_classes))

    def forward(self, x):
        return self.net(x)


def load_disease_model(path: str, device=None):
    from config import CFG
    device = device or CFG["device"]
    ck = torch.load(path, map_location=device, weights_only=False)
    model = DiseaseNet(len(ck["classes"]), pretrained=False).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    model.classes = list(ck["classes"])
    model.img_size = int(ck.get("img_size", IMG_SIZE))
    model._ckpt_meta = {"path": str(path), "epoch": ck.get("epoch"),
                        "val_macro_f1": ck.get("val_macro_f1"),
                        "test_macro_f1": ck.get("test_macro_f1"),
                        "test_accuracy": ck.get("test_accuracy")}
    print(f"[disease] loaded '{path}'  classes={model.classes}  "
          f"val macro-F1={ck.get('val_macro_f1')}  test macro-F1={ck.get('test_macro_f1')}")
    return model


@torch.no_grad()
def predict_disease(model, bgr: np.ndarray) -> dict:
    """Softmax probabilities (average of image and its horizontal flip)."""
    device = next(model.parameters()).device
    rgb = preprocess_fundus(bgr, model.img_size)
    x = to_tensor(rgb).unsqueeze(0).to(device)
    p = (torch.softmax(model(x), 1) + torch.softmax(model(torch.flip(x, [3])), 1)) / 2
    p = p[0].cpu().numpy()
    order = int(p.argmax())
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    return {"probs": {c: float(v) for c, v in zip(model.classes, p)},
            "pred": model.classes[order], "confidence": float(p[order]),
            "norm_entropy": ent}


# ─────────────────────────────────────────────────────────────────────────────
# ODIR-5K parsing
# ─────────────────────────────────────────────────────────────────────────────
_IMG_EXT = {".jpg", ".jpeg", ".png"}


def _label_from_keywords(kw) -> str | None:
    kw = str(kw).lower().strip()
    if kw in ("normal fundus", "normal fundus,", ""):
        return "Normal" if kw.startswith("normal") else None
    ht = "hypertensi" in kw
    rest = re.sub(r"hypertensi\w*(\s+retinopathy)?", "", kw)     # remove HT phrase first
    dr = "retinopathy" in rest                                     # DR still present?
    others = re.sub(r"(normal fundus|retinopathy|non\s*proliferative|nonproliferative|"
                    r"proliferative|mild|moderate|severe|diabetic|[,\s])", "", rest)
    if dr and not ht and others == "":
        return "Diabetic"
    if ht and not dr and others == "":
        return "Hypertensive"
    return None                      # other disease / mixed -> excluded


def parse_odir(root: str, verbose: bool = True):
    """Returns a pandas DataFrame: path, label, patient, eye."""
    import pandas as pd
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"ODIR-5K folder not found: {root}")

    # locate the table
    table = None
    for pat in ("full_df.csv", "*.csv", "*.xlsx"):
        for p in sorted(root.rglob(pat)):
            try:
                df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_excel(p)
            except Exception:
                continue
            if {"Left-Fundus", "Left-Diagnostic Keywords"} <= set(df.columns):
                table = (p, df); break
        if table: break
    if table is None:
        raise RuntimeError(f"No ODIR table (with 'Left-Fundus' / 'Left-Diagnostic Keywords' "
                           f"columns) found under {root}")
    tpath, df = table

    index: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in _IMG_EXT:
            old = index.get(p.name)
            # prefer original photos over 'preprocessed' copies
            if old is None or ("preprocessed" in old.as_posix().lower()
                               and "preprocessed" not in p.as_posix().lower()):
                index[p.name] = p

    rows, miss, excl = [], 0, 0
    id_col = "ID" if "ID" in df.columns else None
    for _, r in df.iterrows():
        pid = str(r[id_col]) if id_col else str(r["Left-Fundus"]).split("_")[0]
        for eye, fcol, kcol in (("left", "Left-Fundus", "Left-Diagnostic Keywords"),
                                ("right", "Right-Fundus", "Right-Diagnostic Keywords")):
            lab = _label_from_keywords(r[kcol])
            if lab is None:
                excl += 1; continue
            p = index.get(str(r[fcol]))
            if p is None:
                miss += 1; continue
            rows.append({"path": str(p), "label": lab, "patient": pid, "eye": eye})
    out = pd.DataFrame(rows)
    if verbose:
        print(f"[ODIR] table: {tpath}")
        print(f"[ODIR] usable eyes: {len(out)}   excluded (other/mixed disease): {excl}   "
              f"image file missing: {miss}")
        if len(out):
            print(out["label"].value_counts().to_string())
    return out
