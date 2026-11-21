"""screening.py — biomarker-based disease screening + per-step images for the website."""
from __future__ import annotations
import cv2, numpy as np
from biomarkers import NORMAL_RANGES, BIOMARKER_LABELS, render_av_overlay

CLASSES = ["Normal", "Glaucoma", "Diabetes", "High blood pressure"]
CLASS_LONG = {"Normal": "Normal retinal vasculature", "Glaucoma": "Glaucoma-type pattern",
              "Diabetes": "Diabetic-retinopathy pattern (diabetes)", "High blood pressure": "Hypertensive-retinopathy pattern (high BP)"}
KEYS = ("vessel_density", "mean_vessel_width", "vessel_tortuosity", "avr", "mean_artery_width", "mean_vein_width")
MEANING = {
    ("vessel_density", -1): "Sparse vessel network (rarefaction) \u2013 typical of glaucoma and long-standing hypertension.",
    ("vessel_density", +1): "Dense / proliferating vessels \u2013 seen in diabetic retinopathy.",
    ("mean_vessel_width", -1): "Vessels narrower than normal \u2013 attenuation, linked to glaucoma and hypertension.",
    ("mean_vessel_width", +1): "Vessels wider than normal \u2013 dilation.",
    ("vessel_tortuosity", +1): "Vessels more twisted than normal \u2013 a hallmark of diabetic retinopathy.",
    ("avr", -1): "Arteries narrow relative to veins \u2013 the classic hypertensive-retinopathy sign.",
    ("avr", +1): "AVR above range \u2013 usually an artery/vein labelling issue, not a disease sign.",
    ("mean_artery_width", -1): "Narrow arterioles \u2013 hypertension (and glaucoma).",
    ("mean_artery_width", +1): "Wide arterioles.",
    ("mean_vein_width", +1): "Dilated veins \u2013 diabetic retinopathy.",
    ("mean_vein_width", -1): "Thin veins.",
}
# weights: how strongly a deviation (in units of the reference-range width) points to each condition
W = {"Glaucoma": {("vessel_density", -1): 2.2, ("mean_vessel_width", -1): 1.6, ("mean_artery_width", -1): .8, ("avr", -1): .3},
     "Diabetes": {("vessel_tortuosity", +1): 1.0, ("mean_vein_width", +1): 1.6, ("vessel_density", +1): 1.2, ("mean_vessel_width", +1): .5},
     "High blood pressure": {("avr", -1): 2.5, ("mean_artery_width", -1): 1.8, ("vessel_tortuosity", +1): .35, ("vessel_density", -1): .4}}
CAP = {"vessel_tortuosity": 1.5}           # tortuosity range is narrow (0.15) so cap its pull
NORMAL_LOGIT, GAIN = 3.4, 4.0


def _dev(x, lo, hi):
    """signed deviation from the reference range, in units of range width (0 = inside)."""
    w = hi - lo
    return 0.0 if lo <= x <= hi else (x - hi) / w if x > hi else (x - lo) / w


def screen(bm: dict, av_reliable: bool = True) -> dict:
    """Predict Normal / Glaucoma / Diabetes / High blood pressure from the six biomarker VALUES.
    Transparent rule-based scoring (every number in the result can be traced to a biomarker). Screening aid, NOT a diagnosis."""
    v = {k: float(bm[k]) for k in KEYS}
    dev = {k: float(np.clip(_dev(v[k], *NORMAL_RANGES[k]), -3, 3)) for k in KEYS}
    soft = {"avr": .4, "mean_artery_width": .6} if not av_reliable else {}          # heuristic A/V -> trust AVR-derived evidence less
    avr_suspect = v["avr"] > 0.9
    if avr_suspect: soft["avr"] = 0.0
    evidence = {c: [] for c in W}
    for c, ws in W.items():
        for (k, sgn), w in ws.items():
            d = dev[k] * sgn
            if d <= 0: continue
            val = min(d, CAP.get(k, 3.0)) * w * soft.get(k, 1.0)
            evidence[c].append({"key": k, "label": BIOMARKER_LABELS[k][0], "contribution": round(val, 3), "direction": "low" if sgn < 0 else "high"})
    score = {c: sum(e["contribution"] for e in evidence[c]) for c in W}
    logits = {"Normal": NORMAL_LOGIT, **{c: GAIN * s for c, s in score.items()}}
    m = max(logits.values()); ex = {c: np.exp(l - m) for c, l in logits.items()}; z = sum(ex.values())
    prob = {c: round(100 * float(ex[c] / z), 1) for c in CLASSES}
    top = max(prob, key=prob.get); ranked = sorted(prob, key=prob.get, reverse=True)
    sec = [c for c in ranked[1:] if c != "Normal" and prob[c] >= 15]
    ts = score.get(top, 0.0)
    sev = "None" if top == "Normal" else "Mild" if ts < .5 else "Moderate" if ts < 1.5 else "Marked"
    findings = []
    for k in KEYS:
        lo, hi = NORMAL_RANGES[k]; x = v[k]; st = "low" if x < lo else "high" if x > hi else "normal"
        findings.append({"key": k, "label": BIOMARKER_LABELS[k][0], "unit": BIOMARKER_LABELS[k][1], "value": round(x, 3), "lo": lo, "hi": hi,
                         "status": st, "deviation": round(dev[k], 2),
                         "meaning": "Inside the reference range." if st == "normal" else MEANING.get((k, -1 if st == "low" else 1), "")})
    out = [f["label"] for f in findings if f["status"] != "normal"]
    if top == "Normal":
        reasons = (["All six biomarkers sit inside their reference ranges."] if not out else
                   [f"No condition gathered meaningful evidence. {len(out)} biomarker(s) are outside range ({', '.join(out)}) but not in a direction that points to glaucoma, diabetes or high blood pressure" + (" (the A/V-based ones are down-weighted because the artery/vein split is unreliable)." if not av_reliable else ".")])
    reasons = (reasons if top == "Normal" else
               [f"{e['label']} is {e['direction']} ({v[e['key']]:.2f}) \u2192 supports {top.lower()}." for e in sorted(evidence[top], key=lambda e: -e["contribution"])])
    rel_notes = []
    if not av_reliable: rel_notes.append("Artery/vein split came from a colour heuristic (no trained A/V weights): AVR and artery-width evidence was down-weighted.")
    if avr_suspect: rel_notes.append(f"AVR {v['avr']:.2f} is far above the 0.67\u20130.75 range \u2013 almost certainly an A/V labelling artefact, so it was ignored for the blood-pressure score.")
    rel = "High" if av_reliable and not avr_suspect else "Reduced"
    risk = {c: round(float(100 * (1 - np.exp(-score[c] * 1.3))), 1) for c in W}                 # 0-100 pattern strength per condition
    return {"verdict": top, "verdict_long": CLASS_LONG[top], "probabilities": prob, "confidence": prob[top], "severity": sev,
            "secondary": sec, "risk": risk, "scores": {c: round(s, 3) for c, s in score.items()}, "evidence": evidence, "findings": findings,
            "outside_range": out, "reasons": reasons, "reliability": rel, "reliability_notes": rel_notes,
            "method": f"logit(Normal)={NORMAL_LOGIT}; logit(condition)={GAIN} \u00d7 \u03a3 weight \u00d7 deviation-from-range (in range-widths); softmax \u2192 match %.",
            "note": "Rule-based screening from biomarker values \u2013 the match % is a pattern-match score, not a calibrated clinical probability. Decision-support aid, not a medical diagnosis."}


def step_images(res: dict, folder) -> list:
    """Write one PNG per pipeline step; returns [{group,key,title,file}] in display order."""
    from pathlib import Path
    f = Path(folder); rgb = res["rgb"]; mc = res["multichannel"]; o = []
    def put(group, key, title, im):
        im = np.asarray(im)
        if im.ndim == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        else: im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(f / f"{key}.png"), im); o.append({"group": group, "key": key, "title": title, "file": f"{key}.png"})
    u8 = lambda x: (np.clip(x, 0, 1) * 255).astype(np.uint8) if x.dtype != np.uint8 else x
    m = res["vessel_mask"] > 0; s1, s2, s3 = (res[k] > 0 for k in ("stage_threshold", "stage_denoised", "stage_bridged"))
    put("pre", "p0", "Original fundus", rgb)
    for i, (k, t) in enumerate((("p1", "Green channel"), ("p2", "CLAHE contrast"), ("p3", "Retinex lighting fix"))): put("pre", k, t, u8(mc[:, :, i]))
    put("pre", "p4", "3-channel network input", u8(mc))
    pm = cv2.applyColorMap(u8(res["prob"]), cv2.COLORMAP_HOT); put("seg", "g1", "Vessel probability map", cv2.cvtColor(pm, cv2.COLOR_BGR2RGB))
    put("seg", "g2", "Binary mask", m.astype(np.uint8) * 255)
    ov = rgb.copy(); ov[m] = (.45 * ov[m] + .55 * np.array([255, 50, 60])).astype(np.uint8); put("seg", "g3", "Overlay on fundus", ov)
    put("cc", "c1", f"1 · Threshold 0.5 ({s1.sum():,} px)", s1.astype(np.uint8) * 255)
    r2 = np.stack([s2 * 255] * 3, -1).astype(np.uint8); r2[s1 & ~s2] = (255, 60, 60); put("cc", "c2", f"2 · Denoise (red = {int((s1 & ~s2).sum()):,} px removed)", r2)
    r3 = np.stack([s3 * 255] * 3, -1).astype(np.uint8); r3[s3 & ~s2] = (60, 255, 90); put("cc", "c3", f"3 · Bridge gaps (green = {int((s3 & ~s2).sum()):,} px added)", r3)
    sk = rgb.copy(); sk[cv2.dilate((res["skeleton"] > 0).astype(np.uint8), np.ones((2, 2))) > 0] = (255, 235, 0); put("cc", "c4", f"4 · Skeleton ({int((res['skeleton'] > 0).sum()):,} px)", sk)
    gr = sk.copy()
    if res["graph"] is not None:
        for _, d in res["graph"].nodes(data=True):
            y, x = d["pos"]; cv2.circle(gr, (int(x), int(y)), 3, (255, 40, 40) if d.get("node_type") == "junction" else (0, 230, 255), -1)
    put("cc", "c5", f"5 · Graph ({res['n_nodes']} nodes · {res['n_edges']} segments)", gr)
    put("av", "a1", "Artery (red) / Vein (blue)", render_av_overlay(rgb, res["artery_mask"], res["vein_mask"]))
    return o
