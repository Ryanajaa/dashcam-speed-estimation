import numpy as np, sys
sys.path.insert(0, "/cache/v2")
from dash_fuse import *

CONF = 6
yva = labels("Validation")
pv = np.load(CACHE / "ckpt" / "final_val_pred_3cnn.npy")      # 3 CNNs x 5 TF, training-only models
print(f"validation, existing ensemble: raw {rmse(pv, yva):.2f} | post {rmse(post(pv), yva):.2f}")
res = {"existing": post(pv)}
for name, local in [("per-clip scale", None), ("local scale +-10 s", 200), ("local scale +-20 s", 400)]:
    pc, info = calibrate("Validation", pv, CONF, local=local)
    res[name] = post(pc)
    print(f"{name:20s}: post {rmse(res[name], yva):.2f}   scales " +
          " ".join(f"{n.split('_')[-1]}:{k:.3f}({c})" for n, c, k in info))
print("\nper clip RMSE / bias")
print(f"{'clip':6s} {'speed':>6s} " + " ".join(f"{k[:18]:>20s}" for k in res))
for j, n in enumerate(names("Validation")):
    print(f"{n.split('_')[-1]:6s} {yva[j].mean():6.1f} " +
          " ".join(f"{rmse(r[j], yva[j]):9.2f} {np.mean(r[j]-yva[j]):+9.2f} " for r in res.values()))