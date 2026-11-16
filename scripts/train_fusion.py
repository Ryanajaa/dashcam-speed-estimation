# Temporal transformer on [frozen flow-CNN features + lane-dash spectrum].
# Same transformer recipe as Stage 6c/11b; the new parts are the dash input and a
# physically exact scale augmentation: speed x s  <=>  dash frequency x s, with the
# flow-CNN features left unchanged. The model has to take the absolute speed level
# from the dashes (fixed 48 ft period) and the shape from the flow.
#
#   python3 train_fusion.py val  [--nodash]   train on Training, score on Validation
#   python3 train_fusion.py test [--nodash]   refit on Training+Validation, predict Testing (labels not read)
import sys, math, time, argparse, numpy as np, torch, torch.nn as nn
import torch.nn.functional as Fn
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["val", "test"])
ap.add_argument("--nodash", action="store_true")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--cnns", default="0,1,2")
ap.add_argument("--aug", type=float, default=0.08)       # scale augmentation, +- fraction
ap.add_argument("--edge", type=float, default=0.0)     # share of windows placed at a clip end
ap.add_argument("--tag", default="")
args = ap.parse_args()

CACHE = Path("/cache"); dev = "cuda"
OUT = CACHE / "v2" / "preds"; OUT.mkdir(parents=True, exist_ok=True)
key = lambda p: int(p.stem.split("_")[-1])
WIN, B, STEPS = 64, 64, 3000
FPS, NFFT, DWIN = 20.0, 1024, 128
K = 128                                                   # log-frequency bins for the dash input
FGRID = np.exp(np.linspace(np.log(0.3), np.log(4.5), K))  # Hz
FREQS = np.fft.rfftfreq(NFFT, 1 / FPS)

rmse = lambda p, y: float(np.sqrt(((p - y) ** 2).mean()))
smooth = lambda p: savgol_filter(median_filter(p, size=(1, 15), mode="nearest"), 41, 2, axis=1)
post = lambda p: np.clip(smooth(p), 0, None)

def names(split):
    return [f.stem for f in sorted((CACHE / "frames" / split).glob("*.npy"), key=key)]

def labels(split):
    return np.stack([np.load(CACHE / "labels" / split / f"{n}.npy") for n in names(split)]).astype(np.float32)

DASH_DIRS = [("dash_ld", 128), ("dash_ld64", 64)]           # long (precise) and short (fast) windows

def dash_frames(split):
    """Per frame: fine dash spectra of the nearest long and short window, (clips, 1200, 2*Fband),
    and how stale each window is, (clips, 1200, 2): (frame - window centre) / window length."""
    band = (FREQS >= 0.25) & (FREQS <= 5.0)
    spec, stale = [], []
    for n in names(split):
        ss, st = [], []
        for d, w in DASH_DIRS:
            z = np.load(CACHE / "v2" / d / split / f"{n}.npz")
            c, S = z["centers"], z["S"][:, band]
            idx = np.abs(np.arange(1200)[:, None] - c[None]).argmin(1)
            ss.append(S[idx]); st.append((np.arange(1200) - c[idx]) / w)
        spec.append(np.concatenate(ss, -1)); stale.append(np.stack(st, -1))
    return (np.stack(spec).astype(np.float32), np.stack(stale).astype(np.float32),
            FREQS[band].astype(np.float32))

class DashSampler:
    """Samples the fine spectrum at FGRID / s (s = speed scale) and normalises it."""
    def __init__(self, fband):
        self.f0, self.df, self.nf = float(fband[0]), float(fband[1] - fband[0]), len(fband)
        self.grid = torch.tensor(FGRID, device=dev, dtype=torch.float32)
    def __call__(self, S, s=None):                        # S: (..., nf), s: broadcastable to S[..., :1]
        g = self.grid if s is None else self.grid / s
        pos = ((g - self.f0) / self.df).clamp(0, self.nf - 1.001)
        i0 = pos.floor().long(); w = pos - i0
        shape = S.shape[:-1] + (K,)
        i0 = i0.expand(shape); w = w.expand(shape)
        v = torch.gather(S, -1, i0) * (1 - w) + torch.gather(S, -1, i0 + 1) * w
        v = torch.log(v + 1e-6)
        return (v - v.mean(-1, keepdim=True)) / (v.std(-1, keepdim=True) + 1e-3)

def dash_input(D, St, s=None):
    nf = D.shape[-1] // 2
    return torch.cat([samp(D[..., :nf], s), samp(D[..., nf:], s), St], -1)

class FusionTF(nn.Module):
    def __init__(s, din=512, ddash=2 * K + 2, d=256, L=4, heads=4, win=WIN, p=0.2, dash=True):
        super().__init__()
        s.dash = dash
        s.inp = nn.Sequential(nn.Dropout(p), nn.Linear(din, d))
        if dash:
            s.inp_d = nn.Sequential(nn.Linear(ddash, d), nn.GELU(), nn.Linear(d, d))
        s.norm = nn.LayerNorm(d)
        s.pos = nn.Parameter(torch.zeros(1, win, d)); nn.init.trunc_normal_(s.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        s.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
    def forward(s, x, xd=None):
        h = s.inp(x)
        if s.dash:
            h = h + s.inp_d(xd)
        return s.out(s.enc(s.norm(h) + s.pos)).squeeze(-1)

if args.mode == "val":
    TRAIN, EVAL = ["Training"], "Validation"
    feat_dir = lambda c: CACHE / ("feats_mg4" if c == 0 else f"feats_mg4_s{c}")
else:
    TRAIN, EVAL = ["Training", "Validation"], "Testing"
    feat_dir = lambda c: CACHE / f"feats_mg4_trval_s{c}"

ytr = np.concatenate([labels(s) for s in TRAIN])
YM, YS = float(ytr.mean()), float(ytr.std())
Dtr_np, Str_np, fband = zip(*[dash_frames(s) for s in TRAIN]); fband = fband[0]
Dtr = torch.from_numpy(np.concatenate(Dtr_np)).to(dev)
Str = torch.from_numpy(np.concatenate(Str_np)).to(dev)
Dev_, Sev_, _ = dash_frames(EVAL)
Dev_, Sev_ = torch.from_numpy(Dev_).to(dev), torch.from_numpy(Sev_).to(dev)
samp = DashSampler(fband)
Ytr = torch.from_numpy(ytr).to(dev)
ar_ = torch.arange(WIN, device=dev)[None]
use_dash = not args.nodash
name = ("nodash" if args.nodash else "dash") + args.tag

@torch.no_grad()
def predict(model, X, D, St, stride=8):
    model.eval()
    N, T, _ = X.shape
    Dn = dash_input(D, St) if use_dash else None
    starts = list(range(0, T - WIN + 1, stride))
    if starts[-1] != T - WIN: starts.append(T - WIN)
    acc = torch.zeros(N, T, device=dev); cnt = torch.zeros(N, T, device=dev)
    for a in starts:
        acc[:, a:a + WIN] += model(X[:, a:a + WIN], Dn[:, a:a + WIN] if use_dash else None)
        cnt[:, a:a + WIN] += 1
    return (acc / cnt).cpu().numpy() * YS + YM

all_preds = []
for c in [int(x) for x in args.cnns.split(",")]:
    Ftr = np.concatenate([np.load(feat_dir(c) / f"{s}.npy") for s in TRAIN])
    Xtr = torch.from_numpy(Ftr).float().to(dev)
    FM, FS = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-3
    Xtr = (Xtr - FM) / FS
    Xev = (torch.from_numpy(np.load(feat_dir(c) / f"{EVAL}.npy")).float().to(dev) - FM) / FS
    NTR = Xtr.shape[0]
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        model = FusionTF(dash=use_dash).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1, s / STEPS))))
        t0 = time.time()
        for step in range(1, STEPS + 1):
            model.train()
            i = torch.randint(0, NTR, (B,), device=dev)
            a = torch.randint(0, 1200 - WIN + 1, (B,), device=dev)
            if args.edge > 0:                             # oversample windows at the clip ends
                e = torch.rand(B, device=dev) < args.edge
                a = torch.where(e, torch.randint(0, 2, (B,), device=dev) * (1200 - WIN), a)
            ar = a[:, None] + ar_
            x = Xtr[i[:, None], ar]; y = Ytr[i[:, None], ar]
            x = x + 0.1 * torch.randn_like(x)
            xd = None
            if use_dash:
                s = torch.exp((torch.rand(B, 1, 1, device=dev) * 2 - 1) * math.log1p(args.aug))
                xd = dash_input(Dtr[i[:, None], ar], Str[i[:, None], ar], s)
                y = y * s[:, :, 0]
            loss = Fn.smooth_l1_loss(model(x, xd), (y - YM) / YS)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        p = predict(model, Xev, Dev_, Sev_)
        all_preds.append(p)
        msg = f"[{name}] cnn {c} seed {seed}: {time.time()-t0:.0f}s"
        if args.mode == "val":
            msg += f" | val raw {rmse(p, labels(EVAL)):.2f} post {rmse(post(p), labels(EVAL)):.2f}"
        print(msg, flush=True)
        del model, opt, sched
    del Xtr, Xev; torch.cuda.empty_cache()

ens = np.mean(all_preds, 0)
np.save(OUT / f"{args.mode}_{name}_ens.npy", ens)
np.save(OUT / f"{args.mode}_{name}_all.npy", np.stack(all_preds))
if args.mode == "val":
    yv = labels(EVAL)
    print(f"[{name}] ENSEMBLE ({len(all_preds)} models): raw {rmse(ens, yv):.2f} | post {rmse(post(ens), yv):.2f}")
    print("per clip RMSE:", np.sqrt(((post(ens) - yv) ** 2).mean(1)).round(2))
    print("per clip bias:", (post(ens) - yv).mean(1).round(2))
else:
    print(f"[{name}] saved test predictions ({len(all_preds)} models); test labels not read")