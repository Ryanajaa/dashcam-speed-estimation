# Validation: where is the error? Mid-clip vs the first/last 5 s of each clip.
import numpy as np, sys
sys.path.insert(0, "/cache/v2")
from dash_fuse import *

y = labels("Validation")
def brk(name, p):
    e = p - y
    edge = np.concatenate([e[:, :100], e[:, 1100:]], 1)
    print(f"{name:34s} all {rmse(p, y):.2f} | mid-clip {np.sqrt((e[:, 100:1100]**2).mean()):.2f} | "
          f"first/last 5 s {np.sqrt((edge**2).mean()):.2f} | share of squared error at edges {(edge**2).sum()/(e**2).sum():.2f}")
old = np.load("/cache/ckpt/final_val_pred_3cnn.npy")
nod = np.load("/cache/v2/preds/val_nodash_single_ens.npy")
fus = np.load("/cache/v2/preds/val_dash_ms_a02_ens.npy")
loc, _ = calibrate("Validation", old, 6, local=200)
brk("Stage 10c flow ensemble", post(old))
brk("control: 1 TF, no dash", post(nod))
brk("A: fusion TF ensemble", post(fus))
brk("B: Stage 10c + local dash scale", post(loc))