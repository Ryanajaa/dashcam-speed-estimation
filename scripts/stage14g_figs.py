import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
DATA, CACHE = Path("/data"), Path("/cache")
FIGS = CACHE / "results" / "figs"; FIGS.mkdir(parents=True, exist_ok=True)
key = lambda p: int(p.stem.split("_")[-1])
N = 1200
# ---- Stage 14 test predictions (saved in 14f, smoothed + clipped) and true speeds ----
fs = sorted((CACHE / "frames" / "Testing").glob("*.npy"), key=key)
names = [f.stem for f in fs]; ids = [n.split("_")[-1] for n in names]
yte = np.stack([np.load(CACHE / "labels" / "Testing" / f.name) for f in fs]).astype(np.float32)
pred = np.load(CACHE / "v2" / "test_pred_final.npy")
assert pred.shape == yte.shape, (pred.shape, yte.shape)
ytr = np.concatenate([np.load(f) for f in sorted((CACHE / "labels" / "Training").glob("*.npy"))])
R = float(np.sqrt(((pred - yte) ** 2).mean()))
sig_tr = float(ytr.std())
r_seg = np.sqrt(((pred - yte) ** 2).mean(1))
print(f"overall test RMSE {R:.3f} km/h (Stage 14f: 1.45) | normalized {R/sig_tr:.4f}")
ts = [pd.read_csv(DATA / "Testing" / n / "labels.csv", usecols=["frame", "timestamp"])
        .sort_values("frame")["timestamp"].values for n in names]
cont = [False] + [0 < ts[k][0] - ts[k - 1][-1] < 0.2 for k in range(1, len(ts))]
# ---- (a) segments side by side: same code as Stage 13c ----
fig, ax = plt.subplots(figsize=(20, 6.5))
for k in range(len(names)):
    x = k * N + np.arange(N)
    ax.plot(x, yte[k], c="k", lw=2.0, label="actual speed" if k == 0 else None)
    ax.plot(x, pred[k], c="r", lw=1.2, label="predicted" if k == 0 else None)
    if k > 0:
        ax.axvline(k * N - 0.5, c="0.6", ls="--", lw=1)
    lab = f"{ids[k]}" + ("  (cont. prev)" if cont[k] else "") + f"\nRMSE {r_seg[k]:.2f}"
    ax.text(k * N + N / 2, 0.97, lab, transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=10, color="0.3")
ymax = max(yte.max(), pred.max())
ax.set_ylim(0, ymax * 1.12); ax.set_xlim(-N * 0.05, len(names) * N + N * 0.05)
ax.set_xticks([k * N + N / 2 for k in range(len(names))]); ax.set_xticklabels(ids)
ax.set_xlabel("held-out test segments: independent 1-minute drives (dashed lines mark segment cuts)")
ax.set_ylabel("speed (km/h)")
ax.grid(axis="y", alpha=0.3)
ax.set_title("Day Five model (flow + lane-dash speedometer, Stage 14): test set, segments pieced side by side (NOT a continuous timeline)\n"
             f"overall RMSE {R:.3f} km/h   ·   normalized RMSE (÷ σ ={sig_tr:.2f}) = {R/sig_tr:.4f}", fontsize=13)
ax.legend(loc="lower center", ncol=2, fontsize=11)
plt.tight_layout()
plt.savefig(FIGS / "test_segments_side_by_side_stage14.png", dpi=150); plt.show()
# ---- (b) error across all 9,600 frames: same code as Stage 12 (b) ----
W = 100
e = pred - yte
roll = np.sqrt(np.stack([np.convolve(r ** 2, np.ones(W), "same") / np.convolve(np.ones(N), np.ones(W), "same")
                         for r in e]))
x = np.arange(yte.size)
fig, ax = plt.subplots(figsize=(18, 4.5))
ax.plot(x, np.abs(e).ravel(), lw=0.4, c="0.65", label="absolute error per frame")
ax.plot(x, roll.ravel(), lw=1.6, c="tab:blue", label=f"rolling RMSE ({W} frames = 5 s)")
ax.axhline(R, c="r", ls="--", lw=1.2, label=f"overall RMSE {R:.2f} km/h")
for k in range(1, len(names)):
    ax.axvline(k * N, c="k", lw=0.6, alpha=0.5)
for k, s in enumerate(ids):
    ax.text(k * N + 600, 0.97, s, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
ax.set_xlim(0, yte.size); ax.set_ylim(0, None)
ax.set_xlabel("test frame (8 subsets × 1,200 frames, in order)"); ax.set_ylabel("km/h")
ax.set_title("Test set (Stage 14): error across all 9,600 frames")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.savefig(FIGS / "test_error_all_frames_stage14.png", dpi=150); plt.show()
print("saved", FIGS / "test_segments_side_by_side_stage14.png", "and", FIGS / "test_error_all_frames_stage14.png")
