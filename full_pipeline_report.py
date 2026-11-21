"""
Everything for ONE image in a single figure + terminal report:
pre-processing channels, probability map, mask, skeleton+graph, A/V overlay, six biomarkers,
CONFIDENCE SCORES and (if weights/disease_classifier.pt exists) the DISEASE probabilities.

    python full_pipeline_report.py
    python full_pipeline_report.py --input my_fundus.png --gt-mask my_expert_mask.png

Saves outputs/full_pipeline_report/<image>/figure_<model>.png, results.json, mask_*.png, ...
(This is a thin wrapper around run_analysis.py's engine.)
"""
from _common import *                       # noqa: F401,F403
import analysis_core as core
import analysis_report as rep


def main():
    ap = make_parser(__doc__); add_model_args(ap)
    ap.add_argument("--primary", default="unetplusplus_scse")
    ap.add_argument("--gt-mask", default=None, help="expert vessel mask → adds Dice/IoU/Sens/Spec")
    a = ap.parse_args()
    out = out_dir(a, __file__)
    if not have_trained(a, need_av=True):
        import fallback_runner as fbr; fbr.run('full', a, out, n_default=1); return

    models, found = load_segmentation_models(a)
    av = load_av_model(found); dis = load_disease_model(found)
    if dis is None: print("[disease] weights/disease_classifier.pt not found -> disease panel skipped")
    key = a.primary if a.primary in models else list(models)[-1]
    for p in pick_images(a):
        r = core.analyse_image(p, models, av, dis, threshold=a.threshold, gt_path=a.gt_mask)
        rep.print_image_report(r, key)
        rep.save_image_outputs(r, out, key, show=a.show)


if __name__ == "__main__":
    main()
