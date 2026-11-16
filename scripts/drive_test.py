# Training + validation only: is the per-clip dash/label ratio shared within a continuous drive
# (tyres / label calibration) or does it change between road sections of one drive (stripe spacing)?
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/cache/v2")
from dash_fuse import *

rows = []
for split in ["Training", "Validation"]:
    y = labels(split)
    for j, n in enumerate(names(split)):
        ts = pd.read_csv(f"/data/{split}/{n}/labels.csv", usecols=["frame", "timestamp"]).sort_values("frame")["timestamp"].values
        S, c = load_dash(split, n)
        vd, cf = dash_read(S, c, y[j])
        vt = window_mean(y[j], c)
        ok = np.isfinite(vd) & (cf > 6) & (vt > 25)
        r = np.median(vd[ok] / vt[ok]) if ok.sum() >= 20 else np.nan
        rows.append(dict(split=split, clip=n.split("_")[-1], t0=ts[0], t1=ts[-1], ratio=r, n=int(ok.sum())))
d = pd.DataFrame(rows)
# drive id: a clip continues the previous one if its first timestamp follows the last within 0.2 s
drive, k = [], 0
for i in range(len(d)):
    if i > 0 and d.split[i] == d.split[i - 1] and 0 < d.t0[i] - d.t1[i - 1] < 0.2:
        pass
    else:
        k += 1
    drive.append(k)
d["drive"] = drive
g = d.dropna(subset=["ratio"]).groupby("drive")
multi = [x for _, x in g if len(x) >= 2]
print(f"clips with a dash ratio: {d.ratio.notna().sum()} | drives: {d.drive.nunique()} | drives with 2+ such clips: {len(multi)}")
within = np.concatenate([x.ratio.values - x.ratio.mean() for x in multi]) if multi else np.array([])
dof = sum(len(x) - 1 for x in multi)
sd_within = np.sqrt((within ** 2).sum() / dof) if dof else np.nan
sd_all = d.ratio.std()
means = np.array([x.ratio.mean() for x in multi])
print(f"spread of dash/label across all clips      : {sd_all*100:.2f} %")
print(f"spread within one continuous drive (pooled): {sd_within*100:.2f} %  (dof {dof})")
print(f"spread of drive means (drives with 2+ clips): {means.std(ddof=1)*100:.2f} %")
print("\nper drive (2+ clips): clip ids and ratio")
for x in multi:
    print("  " + "  ".join(f"{c}:{r:.3f}" for c, r in zip(x["clip"], x["ratio"])))
