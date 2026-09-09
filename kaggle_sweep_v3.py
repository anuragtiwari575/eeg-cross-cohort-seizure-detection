"""
=============================================================================
CROSS-COHORT SEIZURE DETECTION -- SWEEP v3
=============================================================================
Paste the whole file into ONE Kaggle cell.
Set Accelerator to GPU T4 x2, then Save Version -> Save & Run All (Commit).
Close the browser; a committed run is unaffected by that.

CHECK BEFORE COMMITTING
    the log must say  device: cuda / Tesla T4
    if it says  device: cpu  the accelerator was not selected and the run
    will take roughly twenty times as long

WHAT THIS ADDS OVER v2
    v2 established seed variance for one architecture. The claim the paper
    rests on is that representation matters more than architecture, so the
    seed analysis has to cover more than one architecture or the claim is
    supported by a single data point.

    1. Four architectures spanning 2.4e3 to 1.1e6 parameters, five seeds
       each, in all three settings.

    2. Model weights are saved. v2 saved only per-window scores, which was
       enough for matched prevalence, event-level scoring, calibration and
       bootstrap intervals, but not for anything requiring a forward pass
       on data the model never saw. With weights on disk, a later run can
       score the full continuous set at native prevalence without
       retraining anything.

    3. Per-window scores are saved as before.

RESUMING
    /kaggle/working/ is wiped between sessions. To continue:
        download metrics.csv, manifest.csv, preds/ and weights/
        upload them as a Kaggle Dataset
        attach it alongside the raw data and commit again
    The RESTORE block finds it and skips completed runs.

    Weights are the bulky part, roughly 5-30 MB per run. If the upload is
    awkward, SAVE_WEIGHTS can be set to False and only the scores kept; the
    resume logic works either way.

OUTPUTS  (/kaggle/working/)
    metrics.csv          one row per (model, setting, fold, seed)
    preds/<run>.npz      score, label, subject, cohort, order per window
    weights/<run>.pt     state dict of the trained model
    manifest.csv         completed runs
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
WORK      = "/kaggle/working"
PRED_DIR  = os.path.join(WORK, "preds");   os.makedirs(PRED_DIR, exist_ok=True)
WGT_DIR   = os.path.join(WORK, "weights"); os.makedirs(WGT_DIR, exist_ok=True)
METRICS   = os.path.join(WORK, "metrics.csv")
MANIFEST  = os.path.join(WORK, "manifest.csv")

EPOCHS, PATIENCE, BATCH, LR = 14, 4, 256, 1e-3
SEEDS         = [1234, 5678, 9012, 3456, 7890]
S1_MIN_POS    = 30
S1_MIN_TEST   = 8
N_FOLDS       = 5
TIME_BUDGET_H = 11.2
SAVE_WEIGHTS  = True

# Four architectures across three orders of magnitude. EEGNet is included
# again so that v2's numbers can be checked against this run under two
# further seeds.
MODELS = [
    "EEGNet",           #     2 402 params
    "ShallowFBCSPNet",  #    33 282
    "ATCNet",           #   113 242
    "SPARCNet",         # 1 141 889
]

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()

def elapsed_h(): return (time.time() - T0) / 3600
def log(m): print(m, flush=True)


# ----------------------------------------------------------------- restore
def restore():
    for man in glob.glob("/kaggle/input/**/manifest.csv", recursive=True):
        src = os.path.dirname(man)
        for f in ("metrics.csv", "manifest.csv"):
            if os.path.exists(os.path.join(src, f)):
                shutil.copy(os.path.join(src, f), WORK)
        n_p = n_w = 0
        for sub, dst, cnt in (("preds", PRED_DIR, "p"), ("weights", WGT_DIR, "w")):
            d = os.path.join(src, sub)
            files = glob.glob(os.path.join(d, "*")) if os.path.isdir(d) else []
            for p in files:
                shutil.copy(p, dst)
                if cnt == "p": n_p += 1
                else: n_w += 1
        if not n_p:                                   # flat upload fallback
            for p in glob.glob(os.path.join(src, "*.npz")):
                if os.path.getsize(p) < 5e8:
                    shutil.copy(p, PRED_DIR); n_p += 1
            for p in glob.glob(os.path.join(src, "*.pt")):
                shutil.copy(p, WGT_DIR); n_w += 1
        done = len(pd.read_csv(MANIFEST)) if os.path.exists(MANIFEST) else 0
        log("restored from %s: %d runs, %d prediction files, %d weight files"
            % (src, done, n_p, n_w))
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
        m = np.where(sub == s)[0]; order[m] = np.arange(len(m))
    log("X %s | positives %d (%.2f%%) | subjects %d"
        % (X.shape, y.sum(), 100 * y.mean(), len(set(sub))))
    log("device: %s%s" % (DEV, " / " + torch.cuda.get_device_name(0)
                          if DEV == "cuda" else ""))
    if DEV == "cpu":
        log("!! GPU NOT SELECTED -- this run will take about twenty times")
        log("!! as long. Cancel, set Accelerator to GPU, and commit again.")
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


def train_eval(name, tr, te, seed, wpath=None):
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
        if SAVE_WEIGHTS and wpath:
            torch.save({"state_dict": state, "model": name,
                        "n_chans": N_CH, "n_times": N_T, "sfreq": SFREQ,
                        "val_auroc": float(best)}, wpath)
    s_te, y_te = scores(d_te)
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return s_te, y_te, None


# ----------------------------------------------------------------- splits
def s1_coverage():
    rows = []
    for s in np.unique(sub[CHB]):
        ii = np.where(CHB & (sub == s))[0]; c = int(0.7 * len(ii))
        rows.append(dict(subject=str(s), n=len(ii),
                         train_pos=int(y[ii[:c]].sum()),
                         test_pos=int(y[ii[c:]].sum())))
    d = pd.DataFrame(rows).sort_values("train_pos", ascending=False)
    d["admitted"] = (d.train_pos >= S1_MIN_POS) & (d.test_pos >= S1_MIN_TEST)
    return d

COV = s1_coverage()
S1_SUBS = COV[COV.admitted].subject.tolist()
log("\nS1 subjects (>=%d train, >=%d test positives): %d of %d"
    % (S1_MIN_POS, S1_MIN_TEST, len(S1_SUBS), len(COV)))
log("  %s" % S1_SUBS)

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
log("jobs per model per seed: %d   total planned: %d"
    % (len(JOBS), len(JOBS) * len(SEEDS) * len(MODELS)))


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
    # seed-major order, so an interrupted run still covers every
    # architecture at one seed rather than one architecture at every seed
    for seed in SEEDS:
        for name in MODELS:
            if elapsed_h() > TIME_BUDGET_H:
                log("\n!! budget reached"); dump("(budget)"); return
            probe, err = build(name)
            if probe is None:
                log("\n### %-16s SKIPPED (%s)" % (name, err))
                record(dict(run_id="-", model=name, setting="build", fold="-",
                            seed=seed, auroc=np.nan, auprc=np.nan, brier=np.nan,
                            ece=np.nan, n_test=0, pos_test=0, prevalence=np.nan,
                            secs=0, note=err))
                continue
            npar = sum(p.numel() for p in probe.parameters()); del probe
            log("\n### seed %d  %-16s %9d params" % (seed, name, npar))

            for setting, fold in JOBS:
                if (name, setting, str(fold), seed) in done:
                    continue
                if elapsed_h() > TIME_BUDGET_H:
                    log("   budget reached mid-model"); dump("(budget)"); return
                tr, te = split_of(setting, fold)
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    continue
                rid = run_id(name, setting, fold, seed)
                wpath = os.path.join(WGT_DIR, rid + ".pt") if SAVE_WEIGHTS else None
                t = time.time()
                try:
                    s_te, y_te, note = train_eval(name, tr, te, seed, wpath)
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
                log("   %-3s %-12s AUROC=%.3f AUPRC=%.4f ECE=%.3f  (%4.0fs | %.2f h)"
                    % (setting, fold, row["auroc"], row["auprc"], row["ece"],
                       row["secs"], elapsed_h()))
            dump("after seed %d / %s" % (seed, name))


# ----------------------------------------------------------------- report
def report():
    if not os.path.exists(METRICS):
        log("no metrics"); return
    r = pd.read_csv(METRICS); r = r[r.auroc.notna()]
    if r.empty:
        log("no successful runs"); return

    log("\n" + "#" * 74)
    log("SUMMARY -- %d runs, %d architectures, %d seeds, %.2f h"
        % (len(r), r.model.nunique(), r.seed.nunique(), elapsed_h()))
    log("#" * 74)

    piv = r.pivot_table(index="model", columns="setting",
                        values="auroc", aggfunc="mean")
    piv = piv[[c for c in ("S1", "S2", "S3") if c in piv.columns]]
    log("\n=== mean AUROC ==="); log(piv.round(3).to_string())

    log("\n=== seed variance: SD of AUROC over seeds, per configuration ===")
    g = (r.groupby(["model", "setting", "fold"]).auroc
           .agg(["mean", "std", "count"]).reset_index())
    tab = g.groupby(["model", "setting"]).agg(
        mean_auroc=("mean", "mean"), mean_sd=("std", "mean"),
        max_sd=("std", "max"), n_cfg=("std", "size")).round(4)
    log(tab.to_string())
    log("\nDifferences between architectures smaller than about twice these")
    log("SDs cannot be attributed to the architecture.")

    log("\n=== between-architecture spread vs within-architecture seed SD ===")
    for st in ("S1", "S2", "S3"):
        sub_r = r[r.setting == st]
        if sub_r.empty:
            continue
        arch = sub_r.groupby("model").auroc.mean()
        seed_sd = g[g.setting == st]["std"].mean()
        log("  %-3s architecture spread %.3f  |  mean seed SD %.3f  |  ratio %.1f"
            % (st, arch.max() - arch.min(), seed_sd,
               (arch.max() - arch.min()) / seed_sd if seed_sd else np.nan))
    log("A ratio near or below 1 means the architectures are not")
    log("distinguishable given the noise from initialisation alone.")

    if {"S1", "S2", "S3"} <= set(piv.columns):
        piv["S1-S2"] = (piv.S1 - piv.S2).round(3)
        piv["S2-S3"] = (piv.S2 - piv.S3).round(3)
        log("\n=== generalisation gaps ===")
        log(piv[["S1", "S2", "S3", "S1-S2", "S2-S3"]].to_string())
        b = piv[["S1-S2", "S2-S3"]].dropna()
        log("mean S1->S2 %.3f | mean S2->S3 %.3f"
            % (b["S1-S2"].mean(), b["S2-S3"].mean()))

    log("\n=== calibration by setting ===")
    log(r.groupby("setting")[["ece", "brier"]].mean().round(4).to_string())
    log("\n=== runtime by model (s) ===")
    log(r.groupby("model").secs.sum().sort_values().round(0).to_string())

    np_, nw = (len(glob.glob(os.path.join(PRED_DIR, "*.npz"))),
               len(glob.glob(os.path.join(WGT_DIR, "*.pt"))))
    log("\nsaved: %d prediction files, %d weight files" % (np_, nw))
    log("Weights allow the full continuous set to be scored later without")
    log("retraining, which is what the native-prevalence comparison needs.")
    log("\nTo continue: download metrics.csv, manifest.csv, preds/ and")
    log("weights/, upload as a Dataset, attach it, and commit again.")
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
