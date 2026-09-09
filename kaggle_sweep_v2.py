"""
=============================================================================
CROSS-COHORT SEIZURE DETECTION -- SWEEP v2
=============================================================================
Paste this whole file into ONE Kaggle cell.
Save Version -> Save & Run All (Commit), with the GPU accelerator enabled.
Then close the browser: a committed run is unaffected by that, and by the
inactivity timeout that ends interactive sessions.

WHAT THIS ADDS OVER THE FIRST SWEEP
-----------------------------------
1. The score of every test window is saved, together with its subject,
   cohort and position in the recording. The first sweep saved only AUROC
   and AUPRC, which left four analyses impossible after the fact:
        matched-prevalence comparison against the handcrafted baseline
        event-level sensitivity and false alarms per hour
        calibration curves and reliability diagrams
        subject-level bootstrap confidence intervals
   All four become possible with no further GPU time.

2. S1 runs on every subject with enough positives on both sides of the
   split rather than six chosen by hand. In the first sweep only four were
   usable, which is why two architectures returned S1 below S2 -- an
   ordering that cannot be real and that reflects the estimate rather than
   the models. The thresholds below are deliberately low: the subsampled
   set holds 3405 positives spread over 38 subjects, so a high threshold
   admits almost nobody. A coverage table is printed before the sweep
   starts, so the effect of the thresholds is visible immediately rather
   than after an hour of training.

3. Three seeds are run for a small subset of architectures, so it can be
   said which differences exceed stochastic variation.

4. Brier score and expected calibration error are recorded per run.

RESUMING ACROSS SESSIONS
------------------------
/kaggle/working/ is wiped between sessions, so resuming needs the previous
outputs attached as a dataset:

    download metrics.csv, manifest.csv and preds/ from the Output tab
    create a Kaggle Dataset from them
    attach it to the notebook alongside the raw data
    commit again

The RESTORE block finds it automatically. On a first run it says so and
starts fresh.

Everything is also echoed into the log after every model, and the log is
saved with the notebook, so the numbers survive even if the working
directory is lost.

OUTPUTS  (under /kaggle/working/)
    metrics.csv        one row per (model, setting, fold, seed)
    preds/<run>.npz    score, label, subject, cohort, order per window
    manifest.csv       completed runs, used for resuming
=============================================================================
"""

import os, sys, glob, time, shutil, hashlib, traceback, warnings
warnings.filterwarnings("ignore")

try:
    import braindecode  # noqa
except ImportError:
    os.system("pip install -q braindecode")

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold
import braindecode.models as M

# ----------------------------------------------------------------- config
WORK     = "/kaggle/working"
PRED_DIR = os.path.join(WORK, "preds"); os.makedirs(PRED_DIR, exist_ok=True)
METRICS  = os.path.join(WORK, "metrics.csv")
MANIFEST = os.path.join(WORK, "manifest.csv")

EPOCHS, PATIENCE, BATCH, LR = 14, 4, 256, 1e-3
SEEDS         = [1234, 5678, 9012]

# S1 admission thresholds. 3405 positives over 38 subjects means the median
# subject has fewer than a hundred; at a 70/30 split a threshold of 80 on
# the training side admitted only four subjects, which is what the first
# sweep suffered from. These values are set to admit a useful number while
# still requiring enough positives for the estimate to mean anything.
S1_MIN_POS    = 30        # training-side positives
S1_MIN_TEST   = 8         # test-side positives

N_FOLDS       = 5
TIME_BUDGET_H = 11.2      # stop cleanly before the platform's 12 h limit

# Three seeds on all fourteen models does not fit in one session. Seeds
# beyond the first run only for this subset, chosen to span the parameter
# range from the smallest model to the largest.
MULTISEED_SUBSET = ["EEGNet", "ShallowFBCSPNet", "SPARCNet"]

MODELS = [
    "EEGNet",            "EEGITNet",        "EEGTCNet",
    "SincShallowNet",    "SCCNet",          "EEGInceptionERP",
    "ShallowFBCSPNet",   "EEGNeX",          "ATCNet",
    "TIDNet",            "Deep4Net",        "EEGSimpleConv",
    "EEGConformer",      "SPARCNet",
]

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()

def elapsed_h(): return (time.time() - T0) / 3600
def log(m): print(m, flush=True)


# ----------------------------------------------------------------- RESTORE
def restore():
    for man in glob.glob("/kaggle/input/**/manifest.csv", recursive=True):
        src = os.path.dirname(man)
        for f in ("metrics.csv", "manifest.csv"):
            p = os.path.join(src, f)
            if os.path.exists(p):
                shutil.copy(p, WORK)
        n = 0
        sp = os.path.join(src, "preds")
        srcs = glob.glob(os.path.join(sp, "*.npz")) if os.path.isdir(sp) \
               else [p for p in glob.glob(os.path.join(src, "*.npz"))
                     if os.path.getsize(p) < 5e8]
        for p in srcs:
            shutil.copy(p, PRED_DIR); n += 1
        done = len(pd.read_csv(MANIFEST)) if os.path.exists(MANIFEST) else 0
        log("restored from %s: %d completed runs, %d prediction files"
            % (src, done, n))
        return
    log("no checkpoint attached; starting fresh")

restore()


# ----------------------------------------------------------------- data
def load():
    c = [p for p in glob.glob("/kaggle/input/**/*.npz", recursive=True)
         if os.path.getsize(p) > 5e8]
    if not c:
        sys.exit("raw_windows.npz not found under /kaggle/input")
    p = max(c, key=os.path.getsize)
    log("data: %s (%.0f MB)" % (p, os.path.getsize(p) / 1e6))
    d = np.load(p, allow_pickle=True)
    X = d["X"].astype(np.float32); y = d["y"].astype(np.int64)
    coh = np.asarray(d["cohort"]).astype(str)
    sub = np.asarray(d["subject"]).astype(str)
    order = np.zeros(len(y), np.int64)
    for s in np.unique(sub):
        m = np.where(sub == s)[0]
        order[m] = np.arange(len(m))
    log("X %s | positives %d (%.2f%%) | subjects %d"
        % (X.shape, y.sum(), 100 * y.mean(), len(set(sub))))
    log("cohorts: %s" % pd.Series(coh).value_counts().to_dict())
    log("device: %s%s" % (DEV, " / " + torch.cuda.get_device_name(0)
                          if DEV == "cuda" else ""))
    return X, y, coh, sub, order

X, y, coh, sub, order = load()
N_CH, N_T, SFREQ = X.shape[1], X.shape[2], 256.0
CHB = coh == "chb"; CHB_IDX = np.where(CHB)[0]


# ----------------------------------------------------------------- models
def build(name):
    if not hasattr(M, name):
        return None, "absent from this braindecode version"
    cls, last = getattr(M, name), None
    for kw in (dict(n_chans=N_CH, n_outputs=2, n_times=N_T, sfreq=SFREQ),
               dict(n_chans=N_CH, n_outputs=2, n_times=N_T),
               dict(n_chans=N_CH, n_outputs=2,
                    input_window_seconds=N_T / SFREQ, sfreq=SFREQ)):
        try:
            m = cls(**kw)
            with torch.no_grad():
                o = m(torch.zeros(2, N_CH, N_T))
            if o.ndim == 3:
                m = nn.Sequential(m, nn.AdaptiveAvgPool1d(1), nn.Flatten())
                with torch.no_grad():
                    o = m(torch.zeros(2, N_CH, N_T))
            if o.shape[-1] != 2:
                last = "output shape %s" % (tuple(o.shape),); continue
            return m, None
        except Exception as e:
            last = "%s: %s" % (type(e).__name__, str(e)[:90])
    return None, last


# ----------------------------------------------------------------- training
def train_eval(name, tr, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m, err = build(name)
    if m is None:
        return None, None, err
    m = m.to(DEV)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr)); cut = int(0.9 * len(perm))
    tr_i, va_i = tr[perm[:cut]], tr[perm[cut:]]

    def dl(ii, sh, bs=BATCH):
        return DataLoader(TensorDataset(torch.from_numpy(X[ii]).float(),
                                        torch.from_numpy(y[ii]).long()),
                          batch_size=bs, shuffle=sh, num_workers=2,
                          pin_memory=(DEV == "cuda"))

    d_tr, d_va, d_te = dl(tr_i, True), dl(va_i, False, 512), dl(te, False, 512)
    pos = max(1, int(y[tr_i].sum()))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(
        [1.0, (len(tr_i) - pos) / pos], dtype=torch.float32, device=DEV))
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV == "cuda"))

    def scores(loader):
        m.eval(); S, Y = [], []
        with torch.no_grad():
            for xb, yb in loader:
                with torch.cuda.amp.autocast(enabled=(DEV == "cuda")):
                    o = m(xb.to(DEV, non_blocking=True))
                S.append(torch.softmax(o.float(), 1)[:, 1].cpu().numpy())
                Y.append(yb.numpy())
        return np.concatenate(S), np.concatenate(Y)

    best, state, bad = -1.0, None, 0
    for _ in range(EPOCHS):
        m.train()
        for xb, yb in d_tr:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEV == "cuda")):
                loss = crit(m(xb.to(DEV, non_blocking=True)),
                            yb.to(DEV, non_blocking=True))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        s, yv = scores(d_va)
        v = roc_auc_score(yv, s) if len(np.unique(yv)) > 1 else 0.5
        if v > best:
            best, bad = v, 0
            state = {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if state:
        m.load_state_dict(state)
    s_te, y_te = scores(d_te)
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return s_te, y_te, None


# ----------------------------------------------------------------- splits
def s1_coverage():
    """Per-subject positive counts either side of the 70/30 split, so the
    effect of the thresholds is visible before any training happens."""
    rows = []
    for s in np.unique(sub[CHB]):
        ii = np.where(CHB & (sub == s))[0]
        c = int(0.7 * len(ii))
        rows.append(dict(subject=str(s), n=len(ii),
                         train_pos=int(y[ii[:c]].sum()),
                         test_pos=int(y[ii[c:]].sum())))
    d = pd.DataFrame(rows).sort_values("train_pos", ascending=False)
    d["admitted"] = (d.train_pos >= S1_MIN_POS) & (d.test_pos >= S1_MIN_TEST)
    return d

COV = s1_coverage()
S1_SUBS = COV[COV.admitted].subject.tolist()

log("\n=== S1 coverage (thresholds: >=%d train, >=%d test positives) ==="
    % (S1_MIN_POS, S1_MIN_TEST))
log(COV.to_string(index=False))
log("admitted: %d of %d CHB-MIT subjects" % (len(S1_SUBS), len(COV)))
if len(S1_SUBS) < 6:
    log("WARNING: few subjects admitted. The S1 estimate will remain weak,")
    log("and S1 < S2 can appear for individual architectures. Lower")
    log("S1_MIN_POS if the counts above allow it.")

S2_SPLITS = list(GroupKFold(n_splits=N_FOLDS)
                 .split(CHB_IDX, y[CHB_IDX], sub[CHB_IDX]))

def split_of(setting, fold):
    if setting == "S1":
        ii = np.where(CHB & (sub == fold))[0]; c = int(0.7 * len(ii))
        return ii[:c], ii[c:]
    if setting == "S2":
        tri, tei = S2_SPLITS[int(fold[4:])]
        return CHB_IDX[tri], CHB_IDX[tei]
    if fold == "chb->siena":
        return np.where(coh == "chb")[0], np.where(coh == "siena")[0]
    return np.where(coh == "siena")[0], np.where(coh == "chb")[0]

JOBS = ([("S1", s) for s in S1_SUBS]
        + [("S2", "fold%d" % k) for k in range(N_FOLDS)]
        + [("S3", "chb->siena"), ("S3", "siena->chb")])
log("\njobs per seed: %d  (S1 %d, S2 %d, S3 2)"
    % (len(JOBS), len(S1_SUBS), N_FOLDS))


# ----------------------------------------------------------------- persist
def run_id(name, setting, fold, seed):
    h = hashlib.md5(("%s|%s|%s|%d" % (name, setting, fold, seed)).encode()
                    ).hexdigest()[:8]
    return "%s_%s_%s_s%d_%s" % (name, setting,
                                str(fold).replace("->", "2").replace("-", ""),
                                seed, h)

def done_set():
    if not os.path.exists(MANIFEST):
        return set()
    m = pd.read_csv(MANIFEST)
    return set(zip(m.model, m.setting, m.fold.astype(str), m.seed))

def record(row, s_te=None, y_te=None, te_idx=None):
    pd.DataFrame([row]).to_csv(METRICS, mode="a",
                               header=not os.path.exists(METRICS), index=False)
    pd.DataFrame([{k: row[k] for k in ("model", "setting", "fold", "seed")}]
                 ).to_csv(MANIFEST, mode="a",
                          header=not os.path.exists(MANIFEST), index=False)
    if s_te is not None:
        np.savez_compressed(os.path.join(PRED_DIR, row["run_id"] + ".npz"),
                            score=s_te.astype(np.float32),
                            label=y_te.astype(np.int8),
                            subject=sub[te_idx], cohort=coh[te_idx],
                            order=order[te_idx].astype(np.int32))

def ece(s, yy, bins=15):
    e, n, edges = 0.0, len(yy), np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (s >= edges[i]) & (s <= 1.0 if i == bins - 1 else s < edges[i + 1])
        if m.sum():
            e += m.sum() / n * abs(yy[m].mean() - s[m].mean())
    return float(e)

def dump(tag=""):
    if not os.path.exists(METRICS):
        return
    r = pd.read_csv(METRICS)
    log("\n" + "=" * 74)
    log("METRICS DUMP %s  (%d runs, %.2f h)" % (tag, len(r), elapsed_h()))
    log("=" * 74)
    log(r.to_csv(index=False))
    log("=" * 74 + "\n")


# ----------------------------------------------------------------- sweep
def sweep():
    done = done_set()
    if done:
        log("resuming: %d runs already recorded" % len(done))

    for name in MODELS:
        if elapsed_h() > TIME_BUDGET_H:
            log("\n!! budget reached, stopping before %s" % name); break
        probe, err = build(name)
        if probe is None:
            log("\n### %-16s SKIPPED (%s)" % (name, err))
            record(dict(run_id="-", model=name, setting="build", fold="-",
                        seed=-1, auroc=np.nan, auprc=np.nan, brier=np.nan,
                        ece=np.nan, n_test=0, pos_test=0, prevalence=np.nan,
                        secs=0, note=err))
            continue
        npar = sum(p.numel() for p in probe.parameters()); del probe
        seeds = SEEDS if name in MULTISEED_SUBSET else SEEDS[:1]
        log("\n### %-16s %9d params   seeds=%s" % (name, npar, seeds))

        for seed in seeds:
            for setting, fold in JOBS:
                if (name, setting, str(fold), seed) in done:
                    continue
                if elapsed_h() > TIME_BUDGET_H:
                    log("   budget reached mid-model"); dump("(budget)"); return
                tr, te = split_of(setting, fold)
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    continue
                t = time.time()
                try:
                    s_te, y_te, note = train_eval(name, tr, te, seed)
                except Exception as e:
                    s_te, y_te, note = None, None, \
                        "%s: %s" % (type(e).__name__, str(e)[:110])
                    log("   ERROR %s %s s%d: %s" % (setting, fold, seed, note))
                    traceback.print_exc()

                if s_te is None:
                    record(dict(run_id="-", model=name, setting=setting,
                                fold=str(fold), seed=seed, auroc=np.nan,
                                auprc=np.nan, brier=np.nan, ece=np.nan,
                                n_test=len(te), pos_test=int(y[te].sum()),
                                prevalence=float(y[te].mean()),
                                secs=round(time.time() - t, 1), note=note))
                    continue

                rid = run_id(name, setting, fold, seed)
                row = dict(run_id=rid, model=name, setting=setting,
                           fold=str(fold), seed=seed,
                           auroc=float(roc_auc_score(y_te, s_te)),
                           auprc=float(average_precision_score(y_te, s_te)),
                           brier=float(brier_score_loss(y_te, s_te)),
                           ece=ece(s_te, y_te), n_test=len(te),
                           pos_test=int(y_te.sum()),
                           prevalence=float(y_te.mean()),
                           secs=round(time.time() - t, 1), note="")
                record(row, s_te, y_te, te)
                log("   %-3s %-12s s%d  AUROC=%.3f AUPRC=%.4f ECE=%.3f  (%4.0fs | %.2f h)"
                    % (setting, fold, seed, row["auroc"], row["auprc"],
                       row["ece"], row["secs"], elapsed_h()))
        dump("after " + name)


# ----------------------------------------------------------------- report
def report():
    if not os.path.exists(METRICS):
        log("no metrics"); return
    r = pd.read_csv(METRICS); r = r[r.auroc.notna()]
    if r.empty:
        log("no successful runs"); return

    log("\n" + "#" * 74)
    log("SUMMARY -- %d runs, %d architectures, %.2f h"
        % (len(r), r.model.nunique(), elapsed_h()))
    log("#" * 74)

    piv = r.pivot_table(index="model", columns="setting",
                        values="auroc", aggfunc="mean")
    piv = piv[[c for c in ("S1", "S2", "S3") if c in piv.columns]]
    log("\n=== mean AUROC ==="); log(piv.round(3).to_string())
    log("\n=== mean AUPRC ===")
    log(r.pivot_table(index="model", columns="setting",
                      values="auprc", aggfunc="mean").round(4).to_string())

    log("\n=== calibration by setting ===")
    log(r.groupby("setting")[["ece", "brier"]].mean().round(4).to_string())
    log("If ranking holds under transfer while ECE rises, the failure is one")
    log("of operating point rather than of ranking.")

    if {"S1", "S2", "S3"} <= set(piv.columns):
        piv["S1-S2"] = (piv.S1 - piv.S2).round(3)
        piv["S2-S3"] = (piv.S2 - piv.S3).round(3)
        log("\n=== generalisation gaps ===")
        log(piv[["S1", "S2", "S3", "S1-S2", "S2-S3"]].to_string())
        b = piv[["S1-S2", "S2-S3"]].dropna()
        log("\nmean S1->S2 %.3f | mean S2->S3 %.3f"
            % (b["S1-S2"].mean(), b["S2-S3"].mean()))
        if len(b) >= 5:
            from scipy.stats import wilcoxon, ttest_rel
            log("Wilcoxon p=%.4f | paired t p=%.4f | direction %d/%d"
                % (wilcoxon(b["S1-S2"], b["S2-S3"]).pvalue,
                   ttest_rel(b["S1-S2"], b["S2-S3"]).pvalue,
                   (b["S1-S2"] > b["S2-S3"]).sum(), len(b)))
        n_bad = int((piv["S1-S2"] < 0).sum())
        if n_bad:
            log("%d architectures still show S1 < S2; with %d S1 subjects the"
                % (n_bad, len(S1_SUBS)))
            log("S1 estimate remains the weakest part of the design.")

    ms = r[r.model.isin(MULTISEED_SUBSET)]
    if ms.seed.nunique() > 1:
        log("\n=== seed variance ===")
        log(ms.groupby(["model", "setting"]).auroc
              .agg(["mean", "std", "count"]).round(4).to_string())
        log("Differences below about twice these SDs are not architectural.")

    log("\nS1 subjects covered: %d" % r[r.setting == "S1"].fold.nunique())
    log("\n=== runtime by model (s) ===")
    log(r.groupby("model").secs.sum().sort_values().round(0).to_string())

    n = len(glob.glob(os.path.join(PRED_DIR, "*.npz")))
    log("\nper-window predictions: %d runs saved to %s" % (n, PRED_DIR))
    log("These support, with no further GPU time:")
    log("  matched-prevalence comparison with the handcrafted baseline")
    log("  event-level sensitivity and false alarms per hour")
    log("  reliability diagrams")
    log("  subject-level bootstrap confidence intervals")
    log("\nTo continue in a new session: download metrics.csv, manifest.csv")
    log("and preds/, upload them as a Kaggle Dataset, attach it, and commit")
    log("again. RESTORE will pick up from here.")
    dump("(final)")


if __name__ == "__main__":
    try:
        sweep()
    except KeyboardInterrupt:
        log("\ninterrupted")
    except Exception:
        log("\nfatal:"); traceback.print_exc()
    finally:
        report()
        log("\nwall time %.2f h" % elapsed_h())
