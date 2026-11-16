import numpy as np, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/cache/v2")
from dash_fuse import *

y = labels("Testing"); nm = names("Testing")
old = post(np.load("/cache/results/test_pred_submission_raw.npy"))
new = np.load("/cache/v2/test_pred_final.npy")
fig, ax = plt.subplots(2, 4, figsize=(22, 8)); ax = ax.ravel()
for j in range(8):
    a = ax[j]
    a.plot(y[j], "k", lw=2, label="truth")
    a.plot(old[j], "tab:orange", lw=1, label=f"Stage 12 (flow only) {rmse(old[j], y[j]):.2f}")
    a.plot(new[j], "tab:blue", lw=1, label=f"Stage 14 (flow + dashes) {rmse(new[j], y[j]):.2f}")
    a.set_title(nm[j]); a.set_xlabel("frame"); a.set_ylabel("km/h"); a.legend(fontsize=8)
plt.suptitle(f"TEST: Stage 12 {rmse(old, y):.2f} km/h  ->  Stage 14 {rmse(new, y):.2f} km/h")
plt.tight_layout()
(CACHE / "results" / "figs").mkdir(parents=True, exist_ok=True)
plt.savefig(CACHE / "results" / "figs" / "test_stage14_vs_stage12.png", dpi=90)
print("saved results/figs/test_stage14_vs_stage12.png")