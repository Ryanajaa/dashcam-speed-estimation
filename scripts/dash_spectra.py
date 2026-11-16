# Lane-dash spectra for every clip (images only, no labels).
# California broken lane lines repeat every 48 ft = 14.63 m, so the paint in each
# road row flickers at f = v / 14.63 m. Per window we store the row-averaged
# power spectrum of "paint pixels per row"; the speed is read off later.
import numpy as np, sys
from pathlib import Path
from scipy.ndimage import median_filter
from scipy.signal import detrend
from concurrent.futures import ProcessPoolExecutor

CACHE = Path("/cache")
WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 128
OUT = CACHE / "v2" / ("dash_ld" if WIN == 128 else f"dash_ld{WIN}")
key = lambda p: int(p.stem.split("_")[-1])
HOP, FPS, NFFT = 8, 20.0, 1024
R0, R1 = 48, 112          # road rows in the 128 x 384 crop (above the hood)

def spectra(fr):
    g = fr[:, R0:R1].astype(np.float32)                         # (T, h, W)
    bg = median_filter(g, size=(1, 1, 31))                      # local road brightness
    paint = ((g - bg) > 18).sum(2).astype(np.float32)           # (T, h) paint pixels per row
    x = paint                                                   # detrended per window below
    han = np.hanning(WIN).astype(np.float32)[:, None]
    centers, S = [], []
    for a in range(0, len(x) - WIN + 1, HOP):
        w = x[a:a + WIN]
        w = detrend(w, axis=0)                                  # linear: no ripple near the peak
        p = np.abs(np.fft.rfft(w * han, n=NFFT, axis=0)) ** 2           # (F, h)
        p = p / (p.sum(0, keepdims=True) + 1e-6)                # each row votes equally
        S.append(p.mean(1)); centers.append(a + WIN // 2)
    return np.array(centers), np.array(S, np.float32)

def job(args):
    split, f = args
    out = OUT / split / (f.stem + ".npz")
    if out.exists():
        return
    c, S = spectra(np.load(f))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / ("tmp_" + out.name)
    np.savez(tmp, centers=c, S=S)
    tmp.replace(out)

if __name__ == "__main__":
    jobs = [(s, f) for s in ["Validation", "Testing", "Training"]
            for f in sorted((CACHE / "frames" / s).glob("*.npy"), key=key)]
    with ProcessPoolExecutor(4) as ex:
        for i, _ in enumerate(ex.map(job, jobs), 1):
            if i % 8 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    print("done")