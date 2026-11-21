"""analytics.py — extra, image-specific numbers for the website graphs (all computed from the pixels / graph of THIS image)."""
from __future__ import annotations
import hashlib, time
import cv2, numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d


def _f(x, n=2): return round(float(x), n)


def width_hist(res):
    m = res["vessel_mask"] > 0
    dist = distance_transform_edt(m) * 2.0
    sk = res["skeleton"] > 0
    bins = np.arange(1, 14, 1.0)                                      # 1..13 px
    out = {"bins": [f"{int(b)}" for b in bins[:-1]]}
    for k, mk in (("all", sk), ("artery", sk & (res["artery_mask"] > 0)), ("vein", sk & (res["vein_mask"] > 0))):
        h, _ = np.histogram(np.clip(dist[mk], 1, 12.99), bins=bins)
        h = gaussian_filter1d(h.astype(float), 1.0, mode='nearest')            # smooth the integer-pixel quantisation
        out[k] = [_f(100 * x / max(h.sum(), 1e-9), 1) for x in h]
        out[k + "_median"] = _f(np.median(dist[mk])) if mk.any() else 0
    return out


def ring_profile(res, n=8):
    m = res["vessel_mask"] > 0; fov = res["fov"] > 0
    ys, xs = np.nonzero(fov)
    if len(ys) == 0: return {"rings": [], "density": []}
    cy, cx = ys.mean(), xs.mean(); R = np.sqrt(fov.sum() / np.pi)
    yy, xx = np.indices(m.shape); r = np.hypot(yy - cy, xx - cx) / R
    dens = []
    for i in range(n):
        sel = fov & (r >= i / n) & (r < (i + 1) / n)
        dens.append(_f(100 * m[sel].mean(), 2) if sel.sum() > 50 else 0)
    return {"rings": [f"{int(100 * (i + 1) / n)}%" for i in range(n)], "density": dens}


def orientation_rose(res, nb=12):
    g = res["graph"]; acc = np.zeros(nb)
    if g is not None:
        for u, v, d in g.edges(data=True):
            dy, dx = v[0] - u[0], v[1] - u[1]
            if dy == 0 and dx == 0: continue
            ang = (np.degrees(np.arctan2(-dy, dx)) % 180); acc[int(ang // (180 / nb)) % nb] += d.get("length", 1)
    acc = 100 * acc / max(acc.sum(), 1)
    return {"labels": [f"{int(i * 180 / nb)}°" for i in range(nb)], "values": [_f(a, 1) for a in acc]}


def density_grid(res, n=6):
    m = res["vessel_mask"] > 0; fov = res["fov"] > 0; H, W = m.shape; out = []
    for i in range(n):
        row = []
        for j in range(n):
            ys, xs = slice(i * H // n, (i + 1) * H // n), slice(j * W // n, (j + 1) * W // n)
            f = fov[ys, xs]
            row.append(_f(100 * m[ys, xs][f].mean(), 1) if f.mean() > .35 else None)
        out.append(row)
    return out


def prob_hist(res, nb=10):
    p = res["prob"][res["fov"] > 0]; p = p[p > .05]
    h, _ = np.histogram(p, bins=nb, range=(0, 1))
    return {"labels": [f"{i / nb:.1f}" for i in range(nb)], "values": [_f(100 * x / max(h.sum(), 1), 1) for x in h],
            "confident_pct": _f(100 * ((p < .3) | (p > .7)).mean(), 1) if p.size else 0}


def funnel(res):
    s1, s2, s3, sk = (int((res[k] > 0).sum()) for k in ("stage_threshold", "stage_denoised", "stage_bridged", "skeleton"))
    return [{"label": "Thresholded", "value": s1}, {"label": "Noise removed", "value": s2}, {"label": "Gaps bridged", "value": s3}, {"label": "Centre-line", "value": sk}]


def segment_scatter(res, cap=160):
    g = res["graph"]; pts = []
    if g is not None:
        lab = res.get("edge_labels") or {}
        for u, v, d in g.edges(data=True):
            L = d.get("length", 0); ch = float(np.hypot(v[0] - u[0], v[1] - u[1]))
            if L < 12 or ch < 4: continue
            t = L / ch
            if t > 3: continue
            a = lab.get((u, v), lab.get((v, u), lab.get(str(sorted([u, v])), None)))
            pts.append({"len": _f(L, 1), "tort": _f(t, 3), "t": None if a is None else int(a)})
    rng = np.random.default_rng(0)
    if len(pts) > cap: pts = [pts[i] for i in sorted(rng.choice(len(pts), cap, replace=False))]
    return pts


def quadrants(res):
    a, v = res["artery_mask"] > 0, res["vein_mask"] > 0; H, W = a.shape; out = []
    for nm, ys, xs in (("Upper-left", slice(0, H // 2), slice(0, W // 2)), ("Upper-right", slice(0, H // 2), slice(W // 2, W)),
                       ("Lower-left", slice(H // 2, H), slice(0, W // 2)), ("Lower-right", slice(H // 2, H), slice(W // 2, W))):
        out.append({"label": nm, "artery": int(a[ys, xs].sum()), "vein": int(v[ys, xs].sum())})
    return out


def quality(res, src_path=None):
    """Honest data-quality checks the page shows next to every verdict."""
    g = cv2.cvtColor(res["rgb"], cv2.COLOR_RGB2GRAY); fov = res["fov"] > 0
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var()); bright = float(g[fov].mean()) if fov.any() else 0; contrast = float(g[fov].std()) if fov.any() else 0
    cov = res["extra"]["mask_coverage_%"]; unc = res["confidence"]["seg_uncertain_pct"]; h, w = res.get("orig_hw", g.shape)
    ncomp = 0
    if res["graph"] is not None and res["graph"].number_of_nodes():
        import networkx as nx; ncomp = nx.number_connected_components(res["graph"])
    av_ok = "AVResNet" in res["engine_label"]
    chk = lambda name, val, ok, warn_txt, ok_txt: {"name": name, "value": val, "status": "pass" if ok else "warn", "text": ok_txt if ok else warn_txt}
    items = [
        chk("Image resolution", f"{w}×{h} px", min(h, w) >= 400, "Small image – thin vessels may be missed (the pipeline upsamples to 512).", "Resolution adequate."),
        chk("Retina (FOV) coverage", f"{res['extra']['fov_coverage_%']:.0f} %", res["extra"]["fov_coverage_%"] >= 55, "Retina fills little of the frame.", "Retina fills the frame."),
        chk("Focus / sharpness", f"{sharp:.0f}", sharp >= 40, "Image looks soft or blurred.", "Image is sharp enough."),
        chk("Illumination", f"{bright:.0f}/255", 45 <= bright <= 200, "Under- or over-exposed.", "Exposure acceptable."),
        chk("Contrast", f"{contrast:.0f}", contrast >= 18, "Low contrast – vessels hard to separate.", "Contrast acceptable."),
        chk("Vessel coverage", f"{cov:.1f} %", 4 <= cov <= 16, "Outside the usual 4–16 % – check the mask overlay.", "Within the usual 4–16 %."),
        chk("Segmentation certainty", f"{100 - unc:.0f} % confident", unc <= 35, "Many uncertain pixels (0.3–0.7).", "Most pixels confidently classified."),
        chk("Vessel-tree connectivity", f"{ncomp} component(s)", ncomp <= 40, "Fragmented tree.", "Tree reasonably connected."),
        chk("Artery/vein classifier", "trained AVResNet18" if av_ok else "colour heuristic", av_ok, "Heuristic A/V – AVR is less reliable.", "Trained A/V network used."),
    ]
    score = int(round(100 * sum(i["status"] == "pass" for i in items) / len(items)))
    return {"items": items, "score": score, "grade": "Good" if score >= 80 else "Fair" if score >= 55 else "Poor"}


def provenance(src_path, res, seconds):
    sha = hashlib.sha256(open(src_path, "rb").read()).hexdigest() if src_path else ""
    return {"sha256": sha[:16], "analysed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "engine": res["engine_label"], "model": res["model_name"],
            "threshold": 0.5, "deterministic": True, "seconds": seconds}


def build(res, src_path=None):
    return {"width_hist": width_hist(res), "rings": ring_profile(res), "rose": orientation_rose(res), "grid": density_grid(res),
            "prob_hist": prob_hist(res), "funnel": funnel(res), "segments": segment_scatter(res), "quadrants": quadrants(res),
            "quality": quality(res, src_path)}
