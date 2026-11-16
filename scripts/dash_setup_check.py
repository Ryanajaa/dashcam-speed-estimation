# Training + validation labels only: how far is the dash reading from the label, per clip,
# and does it depend on the camera setup (hood row from the Stage 13b label-free check)?
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/cache/v2")
from dash_fuse import *

dc = pd.read_csv("/cache/results/domain_check.csv").set_index("subset")
rows = []
for split in ["Training", "Validation"]:
    y = labels(split)
    for j, n in enumerate(names(split)):
        S, c = load_dash(split, n)
        vd, cf = dash_read(S, c, y[j])
        vt = window_mean(y[j], c)
        ok = np.isfinite(vd) & (cf > 6) & (vt > 25)
        if ok.sum() >= 20:
            rows.append(dict(split=split, hood=dc.loc[n, "hood_row"], v=float(vt[ok].mean()),
                             ratio=float(np.median(vd[ok] / vt[ok]))))
d = pd.DataFrame(rows)
d["setup"] = np.where(d.hood < 820, "hood row ~782", "hood row ~856")
print(f"clips with enough dash windows: {len(d)} of 68")
for s, g in d.groupby("setup"):
    print(f"  {s}: {len(g):2d} clips | dash / label  mean {g.ratio.mean():.4f}  std {g.ratio.std():.4f}")
print(f"  all : std {d.ratio.std():.4f} | corr with speed {np.corrcoef(d.ratio, d.v)[0,1]:+.2f}")
print(f"  at 100 km/h a {d.ratio.std()*100:.1f}% per-clip spread is about {d.ratio.std()*100:.1f} km/h of bias per clip")