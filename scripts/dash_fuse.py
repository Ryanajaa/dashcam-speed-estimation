# Read dash speed from the stored spectra, guided by a prior speed, and
# rescale the flow-model predictions per clip. P is the 48 ft California
# broken-line period; nothing here is fitted to validation or test labels.
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

CACHE = Path("/cache")
DASH = CACHE / "v2" / "dash_ld"
key = lambda p: int(p.stem.split("_")[-1])
P = 48 * 0.3048                    # 14.63 m
WIN, FPS, NFFT = 128, 20.0, 1024
FREQS = np.fft.rfftfreq(NFFT, 1 / FPS)
BAND = 0.25                        # search +-25% around the prior frequency

def names(split):
    return [f.stem for f in sorted((CACHE / "frames" / split).glob("*.npy"), key=key)]

def labels(split):
    return np.stack([np.load(CACHE / "labels" / split / f"{n}.npy") for n in names(split)])

rmse = lambda p, y: float(np.sqrt(((p - y) ** 2).mean()))
smooth = lambda p: savgol_filter(median_filter(p, size=(1, 15), mode="nearest"), 41, 2, axis=1)
post = lambda p: np.clip(smooth(p), 0, None)

def dash_read(S, centers, prior_kmh):
    """Per window: dash speed (km/h) and confidence, searching near the prior."""
    v, conf = np.full(len(S), np.nan), np.zeros(len(S))
    for i, (s, c) in enumerate(zip(S, centers)):
        pv = prior_kmh[max(0, c - WIN // 2):c + WIN // 2].mean()
        f0 = pv / 3.6 / P
        if f0 < 0.35:                                   # too slow: < ~1 cycle per window
            continue
        m = (FREQS > f0 * (1 - BAND)) & (FREQS < f0 * (1 + BAND))
        if m.sum() < 2:
            continue
        idx = np.where(m)[0]
        k = idx[np.argmax(s[idx])]
        if k == idx[0] or k == idx[-1]:                 # peak on the edge: not a real peak
            continue
        a, b, cc = np.log(s[k - 1] + 1e-12), np.log(s[k] + 1e-12), np.log(s[k + 1] + 1e-12)
        d = 0.5 * (a - cc) / (a - 2 * b + cc)
        v[i] = (k + d) * FPS / NFFT * P * 3.6
        wide = (FREQS > 0.3) & (FREQS < 5.0)
        conf[i] = s[k] / (np.median(s[wide]) + 1e-12)
    return v, conf

def load_dash(split, n):
    z = np.load(DASH / split / f"{n}.npz")
    return z["S"], z["centers"]

def window_mean(p, centers):
    return np.array([p[max(0, c - WIN // 2):c + WIN // 2].mean() for c in centers])

def calibrate(split, pred, conf_min, min_n=8, local=None):
    """pred: (clips, 1200) raw flow-model predictions. Returns rescaled predictions."""
    out, info = pred.copy(), []
    for j, n in enumerate(names(split)):
        S, c = load_dash(split, n)
        vd, cf = dash_read(S, c, pred[j])
        vf = window_mean(pred[j], c)
        ok = np.isfinite(vd) & (cf > conf_min) & (vf > 25)
        r = vd[ok] / vf[ok]
        if ok.sum() < min_n:
            info.append((n, ok.sum(), 1.0)); continue
        if local is None:
            k = float(np.median(r)); out[j] = pred[j] * k
        else:                                           # time-varying scale, rolling median
            cc = c[ok]; kk = np.empty(1200)
            for t in range(1200):
                w = np.abs(cc - t) <= local
                kk[t] = np.median(r[w]) if w.sum() >= min_n else np.median(r)
            out[j] = pred[j] * kk; k = float(np.median(r))
        info.append((n, ok.sum(), k))
    return out, info

if __name__ == "__main__":
    # 1) choose the confidence threshold on TRAINING labels only
    #    (prior = true speed x 1.05, standing in for an imperfect model)
    ytr = labels("Training")
    print("training: dash accuracy by confidence (prior = truth x 1.05)")
    allv, allc, ally = [], [], []
    for j, n in enumerate(names("Training")):
        S, c = load_dash("Training", n)
        vd, cf = dash_read(S, c, ytr[j] * 1.05)
        allv += list(vd); allc += list(cf); ally += list(window_mean(ytr[j], c))
    allv, allc, ally = map(np.array, (allv, allc, ally))
    for th in [2, 3, 4, 6, 8, 12]:
        m = np.isfinite(allv) & (allc > th) & (ally > 25)
        e = allv[m] / ally[m] - 1
        print(f"  conf>{th:>2}: windows {m.sum():5d}  median|err| {np.median(np.abs(e))*100:.2f}%  "
              f"|err|<3%: {np.mean(np.abs(e)<0.03):.2f}  bias {np.median(e)*100:+.2f}%")