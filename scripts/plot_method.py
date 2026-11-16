# One training clip: the paint signal, one window's spectrum, and the dash speed against the label.
import numpy as np, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
sys.path.insert(0, "/cache/v2")
from dash_fuse import *

n, split = "Comma_2k19_69", "Training"
fr = np.load(CACHE / "frames" / split / f"{n}.npy")
y = np.load(CACHE / "labels" / split / f"{n}.npy")
g = fr[:400, 48:112].astype(np.float32)
paint = ((g - median_filter(g, size=(1, 1, 31))) > 18).sum(2)
S, c = load_dash(split, n)
vd, cf = dash_read(S, c, y)
fig, ax = plt.subplots(1, 4, figsize=(24, 4.5))
ax[0].imshow(fr[200], cmap="gray"); ax[0].axhline(48, c="r"); ax[0].axhline(112, c="r")
ax[0].set_title(f"{n} frame 200, road rows 48-112 (v = {y[200]:.0f} km/h)")
ax[1].imshow(paint.T, aspect="auto", cmap="magma"); ax[1].set_xlabel("frame"); ax[1].set_ylabel("road row")
ax[1].set_title("paint pixels per row over time: each stripe is one dash")
k = int(np.argmin(np.abs(c - 200)))
ax[2].plot(FREQS, S[k]); ax[2].set_xlim(0, 5)
f_true = window_mean(y, c)[k] / 3.6 / P
ax[2].axvline(f_true, c="k", ls="--", label=f"label speed / 14.63 m = {f_true:.2f} Hz"); ax[2].legend()
ax[2].set_title("spectrum of the window around frame 200"); ax[2].set_xlabel("Hz")
ok = cf > 6
ax[3].plot(y, "k", lw=2, label="label"); ax[3].scatter(c[ok], vd[ok], s=8, c="g", label="dash speed (14.63 m x f)")
ax[3].set_title("dash speedometer vs label"); ax[3].legend(); ax[3].set_xlabel("frame")
plt.tight_layout()
(CACHE / "results" / "figs").mkdir(parents=True, exist_ok=True)
plt.savefig(CACHE / "results" / "figs" / "dash_method.png", dpi=80)
print("saved results/figs/dash_method.png")