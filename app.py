"""
app.py — the website.   python app.py   ->   open  http://127.0.0.1:5000

 * upload a fundus image  -> every figure, graph and value of the pipeline is computed from THAT image
 * 2nd tab                -> every result printed in the project (saved copies, clearly labelled)

Engine: trained networks if PyTorch + the .pt files in weights/ are present, otherwise the classical
fallback (fallback_engine.py).  The badge at the top of the page always says which one is running.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from PIL import UnidentifiedImageError  # noqa: F401
from flask import Flask, jsonify, render_template, request, send_from_directory, abort

import csv
import engine
import analytics
import result_figures as rf
import screening

HERE = Path(__file__).resolve().parent
WEB_OUT = HERE / "outputs" / "web"
SAMPLES = HERE / "sample_images"
ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".ppm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
WEB_OUT.mkdir(parents=True, exist_ok=True)


def _samples():
    return sorted(str(p.relative_to(SAMPLES)).replace("\\", "/") for p in SAMPLES.rglob("*") if p.suffix.lower() in ALLOWED)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    st = engine.status()
    st["samples"] = _samples()
    return jsonify(st)


@app.post("/api/analyze")
def api_analyze():
    rid = uuid.uuid4().hex[:10]
    folder = WEB_OUT / rid; folder.mkdir(parents=True)
    f = request.files.get("image"); sample = request.form.get("sample")
    if f and f.filename:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED:
            return jsonify(error=f"unsupported file type '{ext}' (use png / jpg / tif / bmp)"), 400
        src = folder / f"input{ext}"; f.save(src); name = Path(f.filename).stem
    elif sample:
        p = (SAMPLES / sample).resolve()
        if SAMPLES.resolve() not in p.parents or not p.exists():
            return jsonify(error="unknown sample"), 400
        src = folder / f"input{p.suffix}"; src.write_bytes(p.read_bytes()); name = p.stem
    else:
        return jsonify(error="no image received"), 400
    prefer = "classical" if request.form.get("force_classical") == "1" else "auto"
    try:
        t = time.time()
        res = engine.analyse(src, prefer=prefer); res["name"] = name
        figs = rf.render_all(res, folder)
        steps = screening.step_images(res, folder)
        secs = time.time() - t
    except (ValueError, OSError, UnidentifiedImageError):
        return jsonify(error="could not read that file as an image — please upload a fundus photo (png / jpg / tif / bmp)"), 400
    except Exception as e:                                           # show the real reason in the page
        app.logger.exception("analysis failed")
        return jsonify(error=f"analysis failed: {type(e).__name__}: {e}"), 500
    base = f"/outputs/web/{rid}/"
    doc = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    doc.update({
        "id": rid, "seconds": round(secs, 2), "input": base + src.name,
        "figures": [{"key": k, "title": v["title"], "url": base + v["file"]} for k, v in figs.items()],
        "masks": [{"title": t, "url": base + fn} for t, fn in (("Vessel mask", "mask_vessels.png"), ("Artery mask", "mask_artery.png"),
                  ("Vein mask", "mask_vein.png"), ("Probability map", "prob_map.png"), ("Skeleton", "skeleton.png"))],
        "results_json": base + "results.json", "steps": [dict(s, url=base + s["file"]) for s in steps],
        "screening": screening.screen({b["key"]: b["value"] for b in doc["biomarkers"]}, av_reliable="AVResNet" in res["engine_label"]),
        "analytics": analytics.build(res, str(src)), "provenance": analytics.provenance(str(src), res, round(secs, 2)),
    })
    return jsonify(doc)


@app.get("/api/av_dice")
def api_av_dice():
    """Per-image artery-Dice vs vein-Dice for Attention U-Net and UNet++ + SCSE (data/per_image_av_dice.csv)."""
    rows = []
    with open(HERE / "data" / "per_image_av_dice.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({"image": r["image"], **{k: float(r[k]) for k in ("attention_artery", "attention_vein", "scse_artery", "scse_vein")}})
    return jsonify({"rows": rows, "source": "digitised",
                    "note": "Values were read off the saved scatter figure (accuracy about ±0.01). Replace data/per_image_av_dice.csv with your exact numbers and this page updates automatically."})


@app.get("/api/reference")
def api_reference():
    return jsonify(json.loads((HERE / "data" / "results.json").read_text(encoding="utf-8")))


@app.get("/outputs/web/<rid>/<path:fn>")
def web_file(rid, fn):
    return send_from_directory(WEB_OUT / rid, fn)


@app.get("/samples/<path:fn>")
def sample_file(fn):
    return send_from_directory(SAMPLES, fn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=5000); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="open the browser")
    a = ap.parse_args()
    st = engine.status()
    print("\n" + "=" * 70 + f"\n  RetinaSense website\n  engine : {st['message']}\n  open   : http://{a.host}:{a.port}\n" + "=" * 70 + "\n")
    if a.open:
        import threading, webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{a.host}:{a.port}")).start()
    app.run(host=a.host, port=a.port, debug=False, threaded=True)
