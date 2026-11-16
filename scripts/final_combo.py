# Final recipe (fixed on validation, 1.78): mean of
#   A) fusion transformer ensemble (flow features + dash spectrum, 3 CNNs x 5 TFs)
#   B) the original flow ensemble rescaled by the local (+-10 s) dash scale
# then the usual post-processing. mode "val" scores validation; mode "test" builds the
# test prediction from frozen models and only then reads the test labels to score it.
import sys, numpy as np
sys.path.insert(0, "/cache/v2")
from dash_fuse import *

mode = sys.argv[1]
if mode == "val":
    split, A = "Validation", np.load("/cache/v2/preds/val_dash_ms_a02_ens.npy")
    old = np.load("/cache/ckpt/final_val_pred_3cnn.npy")
else:
    split, A = "Testing", np.load("/cache/v2/preds/test_dash_ms_a02_ens.npy")
    old = np.load("/cache/results/test_pred_submission_raw.npy")
B, info = calibrate(split, old, 6, local=200)
final = post((A + B) / 2)
np.save(f"/cache/v2/{mode}_pred_final.npy", final)
print("dash scale per clip:", " ".join(f"{n.split('_')[-1]}:{k:.3f}" for n, _, k in info))
y = labels(split)                                      # read only after the prediction is saved
print(f"{split}: old {rmse(post(old), y):.2f} | A fusion TF {rmse(post(A), y):.2f} | "
      f"B old+dash scale {rmse(post(B), y):.2f} | FINAL mean(A,B) {rmse(final, y):.2f} "
      f"(normalized {rmse(final, y)/35.75:.3f})")
for j, n in enumerate(names(split)):
    print(f"  {n.split('_')[-1]:5s} v={y[j].mean():6.1f}  RMSE {rmse(final[j], y[j]):.2f}  bias {np.mean(final[j]-y[j]):+.2f}")