"""
=============================================================================
CROSS-COHORT SEIZURE DETECTION - MASTER BENCHMARK
=============================================================================
Self-contained. Paste this whole thing into ONE Kaggle notebook cell,
then use  Save Version -> Save & Run All (Commit)  with GPU enabled.

Why commit mode: interactive Kaggle sessions die after ~1 h of inactivity
and their working directory is wiped. A committed (batch) run is immune to
that, keeps going with the browser closed, and permanently saves both the
log and /kaggle/working/ as notebook output.

Limits you cannot change: 12 h per GPU session, 30 GPU h per week.
This benchmark is expected to need 6-9 h, so it should fit in one commit.
If it does not, just commit again - every completed run is skipped on
re-run, so it resumes where it stopped.

WHAT IT DOES
  Trains N architectures x 3 generalisation settings:
    S1  within-patient   time split inside one CHB-MIT subject
    S2  cross-patient    5-fold GroupKFold, folds split by subject
    S3  cross-cohort     CHB-MIT <-> Siena, both directions
  The headline question is whether S2 -> S3 costs anything, i.e. whether
  crossing a dataset boundary is harder than meeting a new patient.

SAFETY
  Results are written three ways after every single run:
    1. /kaggle/working/results_dl.csv          (notebook output)
    2. /kaggle/working/backup/results_dl.csv   (second copy)
    3. printed to the log as CSV text          (survives everything)
=============================================================================
"""

# ----------------------------------------------------------------- imports
import os, sys, time, glob, json, traceback, warnings
warnings.filterwarnings("ignore")

try:
    import braindecode  # noqa
except ImportError:
    os.system("pip install -q braindecode")

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
import braindecode.models as M

# ----------------------------------------------------------------- config
RESULTS  = "/kaggle/working/results_dl.csv"
BACKUP   = "/kaggle/working/backup/results_dl.csv"
os.makedirs("/kaggle/working/backup", exist_ok=True)

SEED         = 1234
EPOCHS       = 20
PATIENCE     = 4
BATCH        = 256
LR           = 1e-3
S1_SUBJECTS  = 6      # how many within-patient subjects (most seizures first)
N_FOLDS      = 5      # for S2
TIME_BUDGET_H = 11.3  # stop cleanly before Kaggle's 12 h cut-off

MODELS = [
    # small first, so a truncated run still covers many architectures
    "EEGNet",            # Lawhern 2018   - most widely used compact CNN
    "EEGITNet",          # Salami 2022
    "EEGTCNet",          # Ingolfsson 2020
    "SincShallowNet",    # Borra 2020
    "SCCNet",            # Wei 2019
    "EEGInceptionERP",   # Santamaria-Vazquez 2020
    "ShallowFBCSPNet",   # Schirrmeister 2017 - field default baseline
    "EEGNeX",            # Chen 2024
    "ATCNet",            # Altaheri 2022
    "TIDNet",            # Kostas 2020
    "Deep4Net",          # Schirrmeister 2017
    "EEGSimpleConv",     # Ouahidi 2023
    "EEGConformer",      # Song 2022 - CNN + transformer
    "SPARCNet",          # Jing 2023 - built for seizure / IIIC patterns
]

torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T_START = time.time()


def elapsed_h():
    return (time.time() - T_START) / 3600


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------- data
def load_data():
    cands = glob.glob("/kaggle/input/**/*.npz", recursive=True)
    if not cands:
        sys.exit("No .npz found under /kaggle/input - attach the dataset.")
    path = max(cands, key=lambda p: os.path.getsize(p))
    log("data: %s (%.0f MB)" % (path, os.path.getsize(path) / 1e6))
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    coh = d["cohort"].astype(str)
    sub = d["subject"].astype(str)
    log("X %s | positives %d (%.2f%%) | subjects %d"
        % (X.shape, y.sum(), 100 * y.mean(), len(set(sub))))
    log("cohorts: %s" % pd.Series(coh).value_counts().to_dict())
    log("device: %s%s" % (DEV, " / " + torch.cuda.get_device_name(0) if DEV == "cuda" else ""))
    return X, y, coh, sub


X, y, coh, sub = load_data()
N_CHANS, N_TIMES = X.shape[1], X.shape[2]
SFREQ = 256.0


# ----------------------------------------------------------------- models
def build(name):
    """Try the signatures braindecode has used across versions."""
    if not hasattr(M, name):
        return None, "not in this braindecode version"
    cls = getattr(M, name)
    last = None
    for kw in (dict(n_chans=N_CHANS, n_outputs=2, n_times=N_TIMES, sfreq=SFREQ),
               dict(n_chans=N_CHANS, n_outputs=2, n_times=N_TIMES),
               dict(n_chans=N_CHANS, n_outputs=2,
                    input_window_seconds=N_TIMES / SFREQ, sfreq=SFREQ)):
        try:
            m = cls(**kw)
            with torch.no_grad():
                o = m(torch.zeros(2, N_CHANS, N_TIMES))
            if o.ndim == 3:                    # some models return (B, C, T)
                m = nn.Sequential(m, nn.AdaptiveAvgPool1d(1), nn.Flatten())
                with torch.no_grad():
                    o = m(torch.zeros(2, N_CHANS, N_TIMES))
            if o.shape[-1] != 2:
                last = "unexpected output shape %s" % (tuple(o.shape),)
                continue
            return m, None
        except Exception as e:
            last = "%s: %s" % (type(e).__name__, str(e)[:100])
    return None, last


# ----------------------------------------------------------------- training
def run_once(name, tr, te):
    """Train on index set tr, evaluate on te. Returns (auroc, auprc, note)."""
    m, err = build(name)
    if m is None:
        return None, None, err
    m = m.to(DEV)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(tr))
    cut = int(0.9 * len(idx))
    tr_i, va_i = tr[idx[:cut]], tr[idx[cut:]]

    def mk(ii, shuffle, bs=BATCH):
        return DataLoader(TensorDataset(torch.from_numpy(X[ii]).float(),
                                        torch.from_numpy(y[ii]).long()),
                          batch_size=bs, shuffle=shuffle,
                          num_workers=2, pin_memory=(DEV == "cuda"))

    dl_tr, dl_va, dl_te = mk(tr_i, True), mk(va_i, False, 512), mk(te, False, 512)

    pos = max(1, int(y[tr_i].sum()))
    w = torch.tensor([1.0, (len(tr_i) - pos) / pos], dtype=torch.float32, device=DEV)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV == "cuda"))

    def scores(dl):
        m.eval(); S, Y = [], []
        with torch.no_grad():
            for xb, yb in dl:
                with torch.cuda.amp.autocast(enabled=(DEV == "cuda")):
                    o = m(xb.to(DEV, non_blocking=True))
                S.append(torch.softmax(o.float(), 1)[:, 1].cpu().numpy())
                Y.append(yb.numpy())
        return np.concatenate(S), np.concatenate(Y)

    best, bstate, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        m.train()
        for xb, yb in dl_tr:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEV == "cuda")):
                loss = crit(m(xb.to(DEV, non_blocking=True)),
                            yb.to(DEV, non_blocking=True))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        s, yv = scores(dl_va)
        v = roc_auc_score(yv, s) if len(np.unique(yv)) > 1 else 0.5
        if v > best:
            best, bad = v, 0
            bstate = {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if bstate:
        m.load_state_dict(bstate)

    s, yt = scores(dl_te)
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()
    if len(np.unique(yt)) < 2:
        return None, None, "single-class test set"
    return roc_auc_score(yt, s), average_precision_score(yt, s), None


# ----------------------------------------------------------------- splits
chb = coh == "chb"
chb_idx = np.where(chb)[0]
s1_subs = (pd.Series(y[chb]).groupby(sub[chb]).sum()
           .sort_values(ascending=False).head(S1_SUBJECTS).index.tolist())
s2_splits = list(GroupKFold(n_splits=N_FOLDS)
                 .split(chb_idx, y[chb_idx], sub[chb_idx]))
log("S1 subjects: %s" % s1_subs)

JOBS = ([("S1", s) for s in s1_subs]
        + [("S2", "fold%d" % k) for k in range(N_FOLDS)]
        + [("S3", "chb->siena"), ("S3", "siena->chb")])


def split_idx(setting, fold):
    if setting == "S1":
        ii = np.where(chb & (sub == fold))[0]
        c = int(0.7 * len(ii))
        return ii[:c], ii[c:]
    if setting == "S2":
        tri, tei = s2_splits[int(fold[4:])]
        return chb_idx[tri], chb_idx[tei]
    if fold == "chb->siena":
        return np.where(coh == "chb")[0], np.where(coh == "siena")[0]
    return np.where(coh == "siena")[0], np.where(coh == "chb")[0]


# ----------------------------------------------------------------- persist
def save_row(row):
    df = pd.DataFrame([row])
    hdr = not os.path.exists(RESULTS)
    df.to_csv(RESULTS, mode="a", header=hdr, index=False)
    df.to_csv(BACKUP, mode="a", header=not os.path.exists(BACKUP), index=False)


def dump_all(tag=""):
    """Print the full results table into the log. The log is committed with the
    notebook, so these numbers survive even if the working dir is lost."""
    if not os.path.exists(RESULTS):
        return
    r = pd.read_csv(RESULTS)
    log("\n" + "=" * 70)
    log("FULL RESULTS DUMP %s  (%d runs, %.2f h elapsed)" % (tag, len(r), elapsed_h()))
    log("=" * 70)
    log(r.to_csv(index=False))
    log("=" * 70 + "\n")


# ----------------------------------------------------------------- benchmark
def benchmark():
    done = set()
    if os.path.exists(RESULTS):
        prev = pd.read_csv(RESULTS)
        done = set(zip(prev.model, prev.setting, prev.fold.astype(str)))
        log("resuming: %d runs already recorded" % len(done))

    for name in MODELS:
        if elapsed_h() > TIME_BUDGET_H:
            log("\n!! time budget reached, stopping before %s" % name)
            break

        probe, err = build(name)
        if probe is None:
            log("\n### %-18s SKIPPED (%s)" % (name, err))
            save_row(dict(model=name, setting="build", fold="-", auroc=np.nan,
                          auprc=np.nan, n_test=0, pos_test=0, note=err, secs=0))
            continue
        nparam = sum(p.numel() for p in probe.parameters())
        del probe
        log("\n### %-18s (%d params)" % (name, nparam))

        for setting, fold in JOBS:
            if (name, setting, str(fold)) in done:
                continue
            if elapsed_h() > TIME_BUDGET_H:
                log("   time budget reached mid-model, stopping")
                dump_all("(budget stop)")
                return
            tr, te = split_idx(setting, fold)
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            t0 = time.time()
            try:
                a, p, note = run_once(name, tr, te)
            except Exception as e:
                a, p, note = None, None, "%s: %s" % (type(e).__name__, str(e)[:120])
                log("   ERROR in %s %s: %s" % (setting, fold, note))
                traceback.print_exc()
            save_row(dict(model=name, setting=setting, fold=fold, auroc=a, auprc=p,
                          n_test=len(te), pos_test=int(y[te].sum()),
                          note=note or "", secs=round(time.time() - t0, 1)))
            log("   %-3s %-12s AUROC=%s AUPRC=%s  (%4.0fs | %.2f h total)"
                % (setting, fold,
                   "%.3f" % a if a is not None else "  -  ",
                   "%.4f" % p if p is not None else "   -   ",
                   time.time() - t0, elapsed_h()))
        dump_all("after " + name)


# ----------------------------------------------------------------- summary
def summarise():
    if not os.path.exists(RESULTS):
        log("no results")
        return
    r = pd.read_csv(RESULTS)
    r = r[r.auroc.notna()]
    if r.empty:
        log("no successful runs")
        return

    log("\n" + "#" * 70)
    log("SUMMARY - %d runs across %d architectures" % (len(r), r.model.nunique()))
    log("#" * 70)

    piv = r.pivot_table(index="model", columns="setting",
                        values="auroc", aggfunc="mean").round(3)
    log("\n=== mean AUROC ===")
    log(piv.to_string())

    log("\n=== mean AUPRC (base rate %.4f) ===" % y.mean())
    log(r.pivot_table(index="model", columns="setting",
                      values="auprc", aggfunc="mean").round(4).to_string())

    if {"S1", "S2", "S3"} <= set(piv.columns):
        piv["S1-S2"] = (piv.S1 - piv.S2).round(3)
        piv["S2-S3"] = (piv.S2 - piv.S3).round(3)
        log("\n=== generalisation gaps ===")
        log(piv[["S1", "S2", "S3", "S1-S2", "S2-S3"]].to_string())
        log("\nmean drop S1->S2: %.3f" % piv["S1-S2"].mean())
        log("mean drop S2->S3: %.3f" % piv["S2-S3"].mean())
        log("\nIf S2->S3 is near zero across architectures, crossing a dataset")
        log("boundary costs no more than meeting a new patient.")

    log("\n=== S3 by direction ===")
    log(r[r.setting == "S3"].pivot_table(index="model", columns="fold",
        values="auroc", aggfunc="mean").round(3).to_string())

    log("\n=== per-fold spread (S2) ===")
    s2 = r[r.setting == "S2"].groupby("model").auroc.agg(["mean", "std", "min", "max"])
    log(s2.round(3).to_string())

    log("\n=== runtime ===")
    log(r.groupby("model").secs.sum().sort_values().round(0).to_string())

    dump_all("(final)")


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    try:
        benchmark()
    except KeyboardInterrupt:
        log("\ninterrupted")
    except Exception:
        log("\nfatal error:")
        traceback.print_exc()
    finally:
        summarise()
        log("\ntotal wall time: %.2f h" % elapsed_h())
        log("results at: %s  (backup: %s)" % (RESULTS, BACKUP))
