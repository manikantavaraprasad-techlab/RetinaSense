"""
train_disease_classifier.py — Train the ODIR-5K Normal / Diabetic / Hypertensive
classifier and report REAL held-out test metrics.

    python train_disease_classifier.py --odir-root /path/to/odir5k

Outputs
    weights/disease_classifier.pt          checkpoint used by run_analysis.py
    outputs/disease/metrics.json           accuracy, macro-F1, per-class P/R/F1, AUC
    outputs/disease/confusion_matrix.png
    outputs/disease/roc_curves.png
    outputs/disease/training_curves.png
    outputs/disease/split_patients.csv     which patients are in train/val/test

Patients (not images) are split, so the left and right eye of one patient never
appear in both training and test.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config import CFG, seed_everything
from disease_classifier import (CLASSES, IMG_SIZE, DiseaseNet, parse_odir,
                                preprocess_fundus, to_tensor)
from analysis_core import read_bgr


class EyeDS(Dataset):
    def __init__(self, df, train: bool, size: int):
        self.df, self.train, self.size = df.reset_index(drop=True), train, size
        self.cls = {c: i for i, c in enumerate(CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        rgb = preprocess_fundus(read_bgr(r["path"]), self.size)
        if self.train:
            if np.random.rand() < .5: rgb = rgb[:, ::-1]
            if np.random.rand() < .5: rgb = rgb[::-1, :]
            k = np.random.randint(4); rgb = np.rot90(rgb, k)
            rgb = np.clip(rgb.astype(np.float32) * np.random.uniform(.85, 1.15)
                          + np.random.uniform(-12, 12), 0, 255).astype(np.uint8)
        return to_tensor(np.ascontiguousarray(rgb)), self.cls[r["label"]]


def patient_split(df, seed):
    """Stratified patient-level 70/15/15 split."""
    rank = {c: i for i, c in enumerate(["Hypertensive", "Diabetic", "Normal"])}  # rarest first
    pl = (df.assign(r=df["label"].map(rank)).groupby("patient")["r"].min()
          .map({v: k for k, v in rank.items()}))
    pids, labs = pl.index.values, pl.values
    tr, rest, _, lrest = train_test_split(pids, labs, test_size=.30, stratify=labs,
                                          random_state=seed)
    va, te = train_test_split(rest, test_size=.5, stratify=lrest, random_state=seed)
    return (df[df.patient.isin(tr)], df[df.patient.isin(va)], df[df.patient.isin(te)])


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval(); P, Y = [], []
    for x, y in loader:
        x = x.to(device)
        p = (torch.softmax(model(x), 1) + torch.softmax(model(torch.flip(x, [3])), 1)) / 2
        P.append(p.cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odir-root", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--size", type=int, default=IMG_SIZE)
    ap.add_argument("--max-normal", type=int, default=3000,
                    help="cap on Normal eyes in the TRAIN split (class balance / speed)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save", default="weights/disease_classifier.pt")
    ap.add_argument("--out", default="outputs/disease")
    a = ap.parse_args()

    seed_everything(CFG["seed"])
    dev = CFG["device"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    Path(a.save).parent.mkdir(parents=True, exist_ok=True)

    df = parse_odir(a.odir_root)
    tr, va, te = patient_split(df, CFG["seed"])
    if (tr.label == "Normal").sum() > a.max_normal:
        keep = tr[tr.label == "Normal"].sample(a.max_normal, random_state=CFG["seed"])
        tr = (__import__("pandas").concat([tr[tr.label != "Normal"], keep]))
    print(f"\n[split] train={len(tr)}  val={len(va)}  test={len(te)} eyes (patient-disjoint)")
    for n, d in (("train", tr), ("val", va), ("test", te)):
        print(f"   {n:<5}", d.label.value_counts().to_dict())
    (__import__("pandas").concat([tr.assign(split="train"), va.assign(split="val"),
                                  te.assign(split="test")])
     [["patient", "eye", "label", "split", "path"]]
     .to_csv(out / "split_patients.csv", index=False))

    mk = lambda d, t, sh: DataLoader(EyeDS(d, t, a.size), batch_size=a.batch, shuffle=sh,
                                     num_workers=a.workers, pin_memory=dev.type == "cuda")
    ltr, lva, lte = mk(tr, True, True), mk(va, False, False), mk(te, False, False)

    model = DiseaseNet(len(CLASSES), pretrained=True).to(dev)
    counts = np.array([(tr.label == c).sum() for c in CLASSES], np.float32)
    w = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w.to(dev), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    hist = {"train_loss": [], "val_macro_f1": [], "val_acc": []}
    best, best_state, bad = -1.0, None, 0
    print(f"\n{'ep':>3} {'train_loss':>11} {'val_acc':>8} {'val_macroF1':>12} {'sec':>6}")
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); model.train(); tot = 0.0; n = 0
        for x, y in ltr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        P, Y = predict_all(model, lva, dev)
        f1 = f1_score(Y, P.argmax(1), average="macro"); acc = accuracy_score(Y, P.argmax(1))
        hist["train_loss"].append(tot / n); hist["val_macro_f1"].append(f1); hist["val_acc"].append(acc)
        print(f"{ep:3d} {tot/n:11.4f} {acc:8.4f} {f1:12.4f} {time.time()-t0:6.1f}")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop (no val macro-F1 gain for {a.patience} epochs)"); break

    model.load_state_dict(best_state)
    P, Y = predict_all(model, lte, dev)
    pred = P.argmax(1)
    pr, rc, f1c, sup = precision_recall_fscore_support(Y, pred, labels=range(len(CLASSES)),
                                                      zero_division=0)
    aucs = {}
    for i, c in enumerate(CLASSES):
        if len(np.unique(Y == i)) == 2:
            aucs[c] = float(roc_auc_score(Y == i, P[:, i]))
    cm = confusion_matrix(Y, pred, labels=range(len(CLASSES)))
    metrics = {
        "n_test_eyes": int(len(Y)), "best_epoch": best_ep,
        "accuracy": float(accuracy_score(Y, pred)),
        "macro_f1": float(f1_score(Y, pred, average="macro")),
        "balanced_accuracy": float(np.mean([cm[i, i] / max(cm[i].sum(), 1) for i in range(len(CLASSES))])),
        "per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]),
                          "f1": float(f1c[i]), "support": int(sup[i]),
                          "auc_ovr": aucs.get(c)} for i, c in enumerate(CLASSES)},
        "confusion_matrix": cm.tolist(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n" + "=" * 64 + "\n  HELD-OUT TEST RESULTS (patient-disjoint)\n" + "=" * 64)
    print(f"  accuracy {metrics['accuracy']:.4f}   macro-F1 {metrics['macro_f1']:.4f}   "
          f"balanced-acc {metrics['balanced_accuracy']:.4f}")
    print(f"  {'class':<14}{'prec':>7}{'recall':>8}{'F1':>7}{'AUC':>7}{'n':>6}")
    for c, m in metrics["per_class"].items():
        auc = f"{m['auc_ovr']:.3f}" if m["auc_ovr"] is not None else "  n/a"
        print(f"  {c:<14}{m['precision']:7.3f}{m['recall']:8.3f}{m['f1']:7.3f}{auc:>7}{m['support']:6d}")
    print("  confusion matrix (rows=true, cols=pred):\n", cm)

    # ── figures ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES, rotation=20)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("ODIR-5K test confusion matrix")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, c in enumerate(CLASSES):
        if c in aucs:
            fpr, tpr, _ = roc_curve(Y == i, P[:, i]); ax.plot(fpr, tpr, label=f"{c} (AUC {aucs[c]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.legend(); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("One-vs-rest ROC (test)"); fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(hist["train_loss"]); ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(hist["val_macro_f1"], label="val macro-F1"); ax[1].plot(hist["val_acc"], label="val acc")
    ax[1].legend(); ax[1].set_xlabel("epoch"); fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150); plt.close(fig)

    torch.save({"task": "disease_classifier", "model_state": model.state_dict(),
                "classes": CLASSES, "img_size": a.size, "epoch": best_ep,
                "val_macro_f1": round(best, 4), "test_macro_f1": round(metrics["macro_f1"], 4),
                "test_accuracy": round(metrics["accuracy"], 4)}, a.save)
    print(f"\nSaved checkpoint -> {a.save}\nSaved metrics/figures -> {out}/")


if __name__ == "__main__":
    main()
