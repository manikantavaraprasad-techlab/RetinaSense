# weights/

Put the trained checkpoints here (any file names — they are recognised by content):

| what | produced by | looks like |
|---|---|---|
| Attention U-Net | `train_attention_unet.py` | `best_model_attention_unet.pt` |
| UNet++ (deep supervision) | `train_unetplusplus.py` | `best_model_unetplusplus.pt` |
| UNet++ + SCSE | `train_unetplusplus_scse.py` | `best_model_unetplusplus_scse.pt` |
| A/V CNN (AVResNet18) | `run_av_pipeline.py` | `av_cnn.pt` (saved in `outputs/checkpoints/`) |
| Disease CNN (ODIR-5K) | `train_disease_classifier.py` | `disease_classifier.pt` |

Getting them from Kaggle: open each training notebook → **Output** tab → download the
`.pt` files (or `python download_weights.py --kernel OWNER/NOTEBOOK-SLUG`).

`python run_analysis.py` prints which file it found for each model. If a checkpoint is
missing it stops with an error — it never invents numbers.
