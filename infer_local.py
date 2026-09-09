"""
=============================================================================
NATIVE-PREVALENCE INFERENCE FROM SAVED WEIGHTS  (local GPU)
=============================================================================
    python -u infer_local.py

Nothing to edit if the shards and weights are where the earlier steps left
them. Otherwise:

    python -u infer_local.py "D:\\xcohort_full" "C:\\...\\results\\weights"

WHY THIS EXISTS

The learned models were trained and evaluated on a subsample of the windows
(7.36% positive) while the handcrafted baseline was evaluated on everything
(0.34%). That difference alone could account for the gap between them, and
the subsampled test sets held too few negatives to resample down far enough
to check. This script closes the gap: it loads each trained model and scores
it on every window of its held-out set, at the prevalence of continuous
recording. Nothing is retrained.

Running locally rather than on Kaggle avoids uploading 32 GB and allows both
cross-cohort directions to be scored rather than one.

WHAT IT REPORTS
    AUROC, AUPRC and the lift over chance at native prevalence
    the same across a sweep of resampled prevalences
    event-level sensitivity against false alarms per hour, with the
        denominator now real recording time, so these are comparable both
        with the handcrafted results and with the clinical literature
    median detection delay

WHICH RUNS ARE SCORED
    S3, both directions, and S1 where the subject's shard is present. S2
    folds were assigned by GroupKFold inside the sweep and that assignment
    is not recoverable from the run identifier, so they are listed as
    skipped. S3 carries the paper's central comparison.

OUTPUT  D:\\native_inference\\
    native_metrics.csv, native_prevalence.csv, native_event.csv
    scores\\<run_id>.npz   per-window scores, for anything computed later
=============================================================================
"""
from pathlib import Path
import sys, time, warnings, traceback
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

try:
    import braindecode.models as M
except ImportError:
    sys.exit("pip install braindecode")

# ----------------------------------------------------------------- config
SHARD_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"D:\xcohort_full")
WGT_DIR   = Path(sys.argv[2]) if len(sys.argv) > 2 else None
OUT       = Path(r"D:\native_inference")
BATCH     = 1024          # RTX-class card; drop to 256 if memory is tight
WIN_S     = 4.0
PREVS     = [0.50, 0.25, 0.10, 0.0736, 0.05, 0.01, None]   # None = native
DEV       = "cuda" if torch.cuda.is_available() else "cpu"

RES = OUT / "results"; RES.mkdir(parents=True, exist_ok=True)
SC  = OUT / "scores";  SC.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
T0 = time.time()

def log(m): print(m, flush=True)
def el(): return (time.time() - T0) / 60


def find_weights():
    if WGT_DIR and WGT_DIR.exists():
        return sorted(WGT_DIR.glob("*.pt"))
    for c in (Path(r"C:\Users\Anurag Tiwari\Downloads\results\weights"),
              Path(r"D:\results\weights"), Path(r"D:\weights"),
              Path(__file__).resolve().parent / "weights"):
        if c.exists() and any(c.glob("*.pt")):
            return sorted(c.glob("*.pt"))
    sys.exit("weights folder not found -- pass it as the second argument")


SHARDS = {p.stem: p for p in SHARD_DIR.glob("*.npz") if p.stem != "manifest"}
WEIGHTS = find_weights()
log("shards  : %d  (%s)" % (len(SHARDS), SHARD_DIR))
log("weights : %d  (%s)" % (len(WEIGHTS), WEIGHTS[0].parent))
log("device  : %s%s\n" % (DEV, " / " + torch.cuda.get_device_name(0)
                          if DEV == "cuda" else ""))
if not SHARDS:
    sys.exit("no shards -- run extract_full.py first")


# ----------------------------------------------------------------- helpers
def parse_run(rid):
    p = rid.split("_")
    si = next(i for i, x in enumerate(p) if x.startswith("s") and x[1:].isdigit())
    fold = "_".join(p[2:si]).replace("2siena", "->siena").replace("2chb", "->chb")
    return p[0], p[1], fold, int(p[si][1:])


def test_subjects(setting, fold):
    chb   = sorted(s for s in SHARDS if s.lower().startswith("chb"))
    siena = sorted(s for s in SHARDS if s.upper().startswith("PN"))
    if setting == "S1":
        return [fold] if fold in SHARDS else None
    if setting == "S3":
        return siena if fold.endswith("siena") else chb
    return None          # S2 fold membership is not recoverable; see header


def build(name, n_ch, n_t, sfreq=256.0):
    cls = getattr(M, name, None)
    if cls is None:
        return None
    for kw in (dict(n_chans=n_ch, n_outputs=2, n_times=n_t, sfreq=sfreq),
               dict(n_chans=n_ch, n_outputs=2, n_times=n_t),
               dict(n_chans=n_ch, n_outputs=2,
                    input_window_seconds=n_t / sfreq, sfreq=sfreq)):
        try:
            m = cls(**kw)
            with torch.no_grad():
                o = m(torch.zeros(2, n_ch, n_t))
            if o.ndim == 3:
                m = nn.Sequential(m, nn.AdaptiveAvgPool1d(1), nn.Flatten())
            return m
        except Exception:
            continue
    return None


def score_shard(model, path):
    z = np.load(path, allow_pickle=True)
    X = z["X"]
    if X.size == 0:
        return None
    y = z["y"].astype(int)
    key = "rec" if "rec" in z.files else ("file" if "file" in z.files else None)
    rec = z[key].astype(str) if key else np.array(["?"] * len(y))
    od = z["order"].astype(int) if "order" in z.files else np.arange(len(y))
    S = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb = torch.from_numpy(X[i:i+BATCH].astype(np.float32)).to(DEV)
            with torch.autocast(DEV, enabled=(DEV == "cuda")):
                o = model(xb)
            S.append(torch.softmax(o.float(), 1)[:, 1].cpu().numpy())
    return np.concatenate(S), y, rec, od


def merge_events(mask, order):
    o = np.argsort(order); m = mask[o]; oo = order[o]
    ev, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j+1]:
                j += 1
            ev.append((oo[i], oo[j])); i = j + 1
        else:
            i += 1
    return ev


def event_curve(y, s, rec, od, n_thr=40):
    rows, hours = [], len(y) * WIN_S / 3600
    for thr in np.unique(np.quantile(s, np.linspace(0.90, 0.99995, n_thr))):
        pred = s >= thr
        tp = fp = ne = 0
        delays = []
        for r in np.unique(rec):
            m = rec == r
            tev = merge_events(y[m] == 1, od[m])
            pev = merge_events(pred[m], od[m])
            ne += len(tev); hit = {}
            for a, b in pev:
                ov = [k for k, (c, e) in enumerate(tev) if not (b < c or a > e)]
                if ov:
                    for k in ov:
                        d = max(0, a - tev[k][0]) * WIN_S
                        if k not in hit or d < hit[k]:
                            hit[k] = d
                else:
                    fp += 1
            tp += len(hit); delays += list(hit.values())
        if ne:
            rows.append(dict(threshold=float(thr), sensitivity=tp/ne,
                             fa_per_hour=fp/hours, n_events=ne, hours=hours,
                             median_delay_s=float(np.median(delays)) if delays else np.nan))
    return rows


# ----------------------------------------------------------------- main
met, prev_rows, ev_rows, skipped = [], [], [], []
rng = np.random.default_rng(1234)
runs = [(p.stem, p) for p in WEIGHTS]
log("scoring %d runs\n" % len(runs))

for k, (rid, wpath) in enumerate(runs, 1):
    try:
        name, setting, fold, seed = parse_run(rid)
    except Exception:
        skipped.append((rid, "unparseable name")); continue
    subs = test_subjects(setting, fold)
    if not subs:
        skipped.append((rid, "S2 fold membership not recoverable"
                        if setting == "S2" else "test shard absent"))
        continue
    if (SC / (rid + ".npz")).exists():
        continue

    ck = torch.load(wpath, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    model = build(name, ck.get("n_chans", 17), ck.get("n_times", 1024))
    if model is None:
        skipped.append((rid, "cannot rebuild %s" % name)); continue
    try:
        model.load_state_dict(sd)
    except Exception as e:
        skipped.append((rid, "state dict mismatch: %s" % str(e)[:60])); continue
    model = model.to(DEV)

    S = Y = R = O = None
    parts = [score_shard(model, SHARDS[s]) for s in subs if s in SHARDS]
    parts = [p for p in parts if p is not None]
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    if not parts:
        skipped.append((rid, "no windows")); continue

    s = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    rec = np.concatenate([p[2] for p in parts]); od = np.concatenate([p[3] for p in parts])
    np.savez_compressed(SC / (rid + ".npz"), score=s.astype(np.float32),
                        label=y.astype(np.int8), rec=rec, order=od.astype(np.int32))

    prev = float(y.mean()); ap = average_precision_score(y, s)
    row = dict(run_id=rid, model=name, setting=setting, fold=fold, seed=seed,
               n=len(y), pos=int(y.sum()), prevalence=prev,
               auroc=float(roc_auc_score(y, s)), auprc=float(ap), lift=ap/prev,
               brier=float(brier_score_loss(y, s)), hours=len(y)*WIN_S/3600)
    met.append(row)
    pd.DataFrame([row]).to_csv(RES / "native_metrics.csv", mode="a",
        header=not (RES / "native_metrics.csv").exists(), index=False)
    log("[%d/%d] %-40s AUROC=%.3f AUPRC=%.4f lift=%.1fx (%.0f h) | %.1f min"
        % (k, len(runs), rid, row["auroc"], row["auprc"], row["lift"],
           row["hours"], el()))

    pos_i = np.where(y == 1)[0]; neg_i = np.where(y == 0)[0]
    for p in PREVS:
        if p is None:
            idx = np.arange(len(y))
        else:
            nn_ = int(round(len(pos_i) * (1 - p) / p))
            if nn_ > len(neg_i):
                continue
            idx = np.concatenate([pos_i, rng.choice(neg_i, nn_, replace=False)])
        prev_rows.append(dict(run_id=rid, model=name, setting=setting, fold=fold,
                              seed=seed, target_prev=p if p else prev,
                              actual_prev=float(y[idx].mean()), n=len(idx),
                              auroc=float(roc_auc_score(y[idx], s[idx])),
                              auprc=float(average_precision_score(y[idx], s[idx]))))
    for r in event_curve(y, s, rec, od):
        r.update(run_id=rid, model=name, setting=setting, fold=fold, seed=seed)
        ev_rows.append(r)

pd.DataFrame(prev_rows).to_csv(RES / "native_prevalence.csv", index=False)
pd.DataFrame(ev_rows).to_csv(RES / "native_event.csv", index=False)

# ----------------------------------------------------------------- report
log("\n" + "#" * 74)
log("SUMMARY -- %d runs scored, %d skipped, %.1f min" % (len(met), len(skipped), el()))
log("#" * 74)
if skipped:
    from collections import Counter
    log("\nskipped, by reason:")
    for why, n in Counter(w for _, w in skipped).most_common():
        log("   %-45s %d" % (why, n))

if met:
    m = pd.DataFrame(met)
    log("\n=== at native prevalence (%.3f%% positive) ===" % (100*m.prevalence.mean()))
    log(m.groupby(["setting", "model"])[["auroc", "auprc", "lift"]]
          .mean().round(4).to_string())
    log("\nHandcrafted baseline on the same continuous data:")
    log("   S3 chb->siena 0.692   S3 siena->chb 0.680   (AUROC, both features)")

    e = pd.DataFrame(ev_rows)
    if not e.empty:
        log("\n=== event level, real recording hours ===")
        for st in sorted(e.setting.unique()):
            sub = e[e.setting == st]
            log("--- %s (%.0f h) ---" % (st, sub.hours.mean()))
            for tgt in (0.1, 0.5, 1.0, 2.0, 5.0):
                v = [g[g.fa_per_hour <= tgt].sensitivity.max()
                     if (g.fa_per_hour <= tgt).any() else 0.0
                     for _, g in sub.groupby("run_id")]
                log("   at %.1f FA/h: sensitivity %.3f" % (tgt, np.mean(v)))
            near = sub[(sub.fa_per_hour > 0.5) & (sub.fa_per_hour < 2.0)]
            if not near.empty and near.median_delay_s.notna().any():
                log("   median detection delay near 1 FA/h: %.0f s"
                    % near.median_delay_s.median())
        log("\nHandcrafted on the same data, for comparison:")
        log("   S3 at 1.0 FA/h: 0.011   at 2.0: 0.170   at 5.0: 0.290")
        log("   These denominators match, so the two are directly comparable.")

    p = pd.DataFrame(prev_rows)
    if not p.empty:
        log("\n=== prevalence sweep ===")
        log(p.groupby(["setting", "target_prev"])[["auroc", "auprc"]]
              .mean().round(4).to_string())

log("\nwritten to %s" % RES)
log("per-window scores in %s" % SC)
