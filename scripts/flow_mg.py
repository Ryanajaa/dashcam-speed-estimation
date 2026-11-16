import cv2, os, numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

CACHE = Path("/cache")

def flow_subset(args):
    split, name, gap = args
    cv2.setNumThreads(1)
    out = CACHE / f"flow_g{gap}" / split / name
    if out.exists():
        return
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fr = np.load(CACHE / "frames" / split / name)
    n = len(fr)
    fl = np.zeros((n, fr.shape[1], fr.shape[2], 2), np.float16)
    for t in range(n - gap):
        fl[t] = dis.calc(fr[t], fr[t + gap], None)
    fl[n - gap:] = fl[n - gap - 1]                 # pad tail
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / ("tmp_" + name)
    np.save(tmp, fl)
    os.replace(tmp, out)                            # atomic

if __name__ == "__main__":
    for p in CACHE.glob("flow_g*/*/tmp_*"):         # leftovers from killed runs
        p.unlink()
    jobs = [(s, f.name, g) for g in (16,) for s in ["Training", "Validation", "Testing"]
            for f in sorted((CACHE / "frames" / s).glob("*.npy"))]
    print(len(jobs), "jobs", flush=True)
    with ProcessPoolExecutor(6) as ex:
        futs = [ex.submit(flow_subset, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()                              # raise errors here, not silently
            if i % 4 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}", flush=True)
    print("done")
