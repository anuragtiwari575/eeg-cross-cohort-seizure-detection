"""
=============================================================================
CPU-SIDE ANALYSES  (runs on the laptop, independent of the Kaggle GPU job)
=============================================================================
    python -u cpu_analyses.py all

    python -u cpu_analyses.py fair        # 1. fair linear baseline
    python -u cpu_analyses.py shuffle     # 2. label-shuffle negative control
    python -u cpu_analyses.py prevalence  # 3. prevalence sensitivity
    python -u cpu_analyses.py subjects    # 4. per-subject variance
    python -u cpu_analyses.py shortcut    # 5. seizure-file context test

WHY EACH ONE EXISTS (this is what reviewers will ask)

 1. FAIR BASELINE
    The linear results so far used the full continuous data (0.34% positives)
    while the deep models used subsampled windows (7.36%). That is not a
    like-for-like comparison. Here the linear model is run on exactly the same
    windows the deep models saw, so any remaining gap is attributable to the
    representation rather than to the data.

 2. LABEL SHUFFLE
    With labels permuted within each subject, AUROC must fall to ~0.5. If it
    does not, something is leaking. No reviewer will trust 0.85 without this.

 3. PREVALENCE SENSITIVITY
    Published seizure-detection numbers span 0.61 AUC to 99.7% accuracy,
    largely because evaluation prevalence differs. Sweeping the test-set
    positive rate from 50% down to the true 0.34% shows how much of a reported
    number is method and how much is protocol.

 4. PER-SUBJECT VARIANCE
    Mean AUROC hides that some subjects fail completely (chb13 reached 0.442
    with SincShallowNet, i.e. below chance, but 0.917 with SCCNet). For
    deployment the spread matters more than the mean.

 5. SHORTCUT TEST
    In CHB-MIT the positives all come from seizure-containing files. If those
    files carry a recording-level signature (electrode adjustment, arousal,
    staff activity), a model can score well without learning seizures at all.
    This tries to separate non-seizure windows of seizure files from
    non-seizure windows of seizure-free files. High AUROC here is bad news.

Outputs go to E:\\cpu_results\\ as CSVs, and everything is printed as well.
=============================================================================
"""
from pathlib import Path
import os, sys, time, warnings
os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

FEAT_DIR = Path(r"E:\xcohort")                       # feature .npz from xcohort.py prep
RAW_FILE = Path(r"E:\xcohort_raw\raw_windows.npz")   # subsampled raw windows
OUT      = Path(r"E:\cpu_results"); OUT.mkdir(parents=True, exist_ok=True)

SEED     = 1234
MAX_TRAIN = 150_000      # cap training rows; all positives are always kept
NB       = 5 * 17        # 85 bandpower features, then 34 aperiodic
N_FOLDS  = 5

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
CHNAMES = ["FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3","C3-P3","P3-O1",
           "FP2-F4","F4-C4","C4-P4","P4-O2","FP2-F8","F8-T8","P8-O2","FZ-CZ","CZ-PZ"]

rng_global = np.random.default_rng(SEED)


# ----------------------------------------------------------------- loading
def load_features():
    """Full continuous feature set: X, y, cohort, subject, file."""
    Xs, ys, cs, ss, fs = [], [], [], [], []
    for f in sorted(FEAT_DIR.glob("*.npz")):
        d = np.load(f)
        if d["X"].size == 0:
            continue
        cohort, subj, fname = f.stem.split("__", 2)
        n = len(d["y"])
        Xs.append(d["X"]); ys.append(d["y"])
        cs += [cohort]*n; ss += [subj]*n; fs += [fname]*n
    if not Xs:
        sys.exit("no feature .npz found in %s" % FEAT_DIR)
    X = np.vstack(Xs); y = np.concatenate(ys)
    coh = np.array(cs); sub = np.array(ss); fil = np.array(fs)
    ok = np.isfinite(X).all(1)
    print("features: %s | positives %d (%.3f%%) | subjects %d"
          % (X[ok].shape, y[ok].sum(), 100*y[ok].mean(), len(set(sub[ok]))))
    return X[ok], y[ok], coh[ok], sub[ok], fil[ok]


def load_raw_meta():
    """Subject/cohort labels of the subsampled windows the deep models used."""
    if not RAW_FILE.exists():
        return None
    d = np.load(RAW_FILE, allow_pickle=True)
    return dict(y=d["y"].astype(int),
                cohort=d["cohort"].astype(str),
                subject=d["subject"].astype(str))


# ----------------------------------------------------------------- helpers
def fit_eval(Xtr, ytr, Xte, yte, seed=SEED, cap=MAX_TRAIN):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return np.nan, np.nan, 0
    if cap and len(ytr) > cap:
        r = np.random.default_rng(seed)
        pos = np.where(ytr == 1)[0]
        neg = np.where(ytr == 0)[0]
        neg = r.choice(neg, max(1, min(len(neg), cap - len(pos))), replace=False)
        keep = np.concatenate([pos, neg])
        Xtr, ytr = Xtr[keep], ytr[keep]
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000, class_weight="balanced"))
    m.fit(Xtr, ytr)
    s = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, s), average_precision_score(yte, s), len(ytr)


def subsample_test(y_idx, y, prevalence, seed=SEED):
    """Return a subset of y_idx whose positive rate is ~prevalence.
    All positives kept; negatives drawn to hit the target rate."""
    r = np.random.default_rng(seed)
    pos = y_idx[y[y_idx] == 1]
    neg = y_idx[y[y_idx] == 0]
    if len(pos) == 0:
        return y_idx
    n_neg = int(round(len(pos) * (1 - prevalence) / prevalence))
    n_neg = min(n_neg, len(neg))
    if n_neg < 1:
        return y_idx
    return np.concatenate([pos, r.choice(neg, n_neg, replace=False)])


FEATSETS = {"bandpower": slice(0, NB),
            "aperiodic": slice(NB, None),
            "both":      slice(0, None)}


def splits(coh, sub, y):
    """Yield (name, train_idx, test_idx) for S2 folds and S3 directions."""
    chb_idx = np.where(coh == "chb")[0]
    gkf = GroupKFold(n_splits=N_FOLDS)
    for k, (tri, tei) in enumerate(gkf.split(chb_idx, y[chb_idx], sub[chb_idx])):
        yield ("S2", "fold%d" % k, chb_idx[tri], chb_idx[tei])
    a = np.where(coh == "chb")[0]; b = np.where(coh == "siena")[0]
    yield ("S3", "chb->siena", a, b)
    yield ("S3", "siena->chb", b, a)


# ----------------------------------------------------------------- 1. fair
def cmd_fair():
    """Linear model on exactly the windows the deep models used."""
    print("\n" + "="*70)
    print("1. FAIR LINEAR BASELINE")
    print("="*70)
    meta = load_raw_meta()
    if meta is None:
        print("raw_windows.npz not found - cannot match the deep-model windows.")
        print("Falling back to a prevalence-matched subsample instead.\n")

    X, y, coh, sub, fil = load_features()
    rows = []

    # Match the deep models' class balance per subject as closely as possible.
    # The deep set kept all positives and 60 random negatives per file.
    r = np.random.default_rng(SEED)
    keep = []
    for f in np.unique(fil):
        m = np.where(fil == f)[0]
        pos = m[y[m] == 1]
        neg = m[y[m] == 0]
        if len(neg) > 60:
            neg = r.choice(neg, 60, replace=False)
        keep.append(np.concatenate([pos, neg]))
    keep = np.sort(np.concatenate(keep))
    Xs, ys, cohs, subs = X[keep], y[keep], coh[keep], sub[keep]
    print("matched subset: %d windows | positives %.2f%% (deep set was 7.36%%)\n"
          % (len(ys), 100*ys.mean()))

    for fname, sl in FEATSETS.items():
        print("--- %s ---" % fname)
        for setting, fold, tr, te in splits(cohs, subs, ys):
            auc, ap, ntr = fit_eval(Xs[:, sl][tr], ys[tr], Xs[:, sl][te], ys[te])
            print("  %-3s %-12s AUROC=%.3f AUPRC=%.4f (ntr=%d nte=%d pos=%.1f%%)"
                  % (setting, fold, auc, ap, ntr, len(te), 100*ys[te].mean()))
            rows.append(dict(analysis="fair", features=fname, setting=setting,
                             fold=fold, auroc=auc, auprc=ap,
                             prevalence=ys[te].mean(), n_test=len(te)))
    pd.DataFrame(rows).to_csv(OUT/"fair_baseline.csv", index=False)
    print("\nwrote", OUT/"fair_baseline.csv")


# ----------------------------------------------------------------- 2. shuffle
def cmd_shuffle():
    """Permute labels within each subject. AUROC must collapse to ~0.5."""
    print("\n" + "="*70)
    print("2. LABEL-SHUFFLE NEGATIVE CONTROL")
    print("="*70)
    X, y, coh, sub, fil = load_features()
    rows = []
    for rep in range(3):
        yp = y.copy()
        r = np.random.default_rng(SEED + rep)
        for s in np.unique(sub):                       # permute within subject
            m = np.where(sub == s)[0]
            yp[m] = y[m][r.permutation(len(m))]
        print("\n--- repeat %d ---" % rep)
        for setting, fold, tr, te in splits(coh, sub, yp):
            if fold not in ("fold0", "chb->siena"):
                continue
            auc, ap, ntr = fit_eval(X[tr], yp[tr], X[te], yp[te], seed=SEED+rep)
            flag = "" if abs(auc - .5) < .05 else "   <-- SUSPICIOUS"
            print("  %-3s %-12s AUROC=%.3f AUPRC=%.4f%s"
                  % (setting, fold, auc, ap, flag))
            rows.append(dict(analysis="shuffle", repeat=rep, setting=setting,
                             fold=fold, auroc=auc, auprc=ap))
    d = pd.DataFrame(rows)
    print("\nmean shuffled AUROC: %.3f  (should be ~0.500)" % d.auroc.mean())
    d.to_csv(OUT/"label_shuffle.csv", index=False)
    print("wrote", OUT/"label_shuffle.csv")


# ----------------------------------------------------------------- 3. prevalence
def cmd_prevalence():
    """Same model, same test subjects, only the test-set positive rate changes."""
    print("\n" + "="*70)
    print("3. PREVALENCE SENSITIVITY")
    print("="*70)
    print("Training is held fixed; only the evaluation set's class balance moves.")
    print("AUROC is prevalence-invariant in principle; AUPRC is not. The point")
    print("is to show how much a reported number depends on the protocol.\n")

    X, y, coh, sub, fil = load_features()
    PREVS = [0.50, 0.25, 0.10, 0.05, 0.0736, 0.01, None]   # None = native rate
    rows = []
    for setting, fold, tr, te in splits(coh, sub, y):
        if fold not in ("fold0", "chb->siena", "siena->chb"):
            continue
        # fit once on the training split
        r = np.random.default_rng(SEED)
        pos = tr[y[tr] == 1]; neg = tr[y[tr] == 0]
        neg = r.choice(neg, max(1, min(len(neg), MAX_TRAIN - len(pos))), replace=False)
        keep = np.concatenate([pos, neg])
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000, class_weight="balanced"))
        m.fit(X[keep], y[keep])
        print("--- %s %s ---" % (setting, fold))
        for p in PREVS:
            idx = te if p is None else subsample_test(te, y, p)
            if len(np.unique(y[idx])) < 2:
                continue
            s = m.predict_proba(X[idx])[:, 1]
            auc = roc_auc_score(y[idx], s); ap = average_precision_score(y[idx], s)
            lab = "native" if p is None else "%.2f%%" % (100*p)
            print("   prevalence %-8s n=%-7d AUROC=%.3f  AUPRC=%.4f  (lift %.1fx)"
                  % (lab, len(idx), auc, ap, ap / max(y[idx].mean(), 1e-9)))
            rows.append(dict(analysis="prevalence", setting=setting, fold=fold,
                             target_prev=p, actual_prev=y[idx].mean(),
                             n_test=len(idx), auroc=auc, auprc=ap))
        print()
    pd.DataFrame(rows).to_csv(OUT/"prevalence_sweep.csv", index=False)
    print("wrote", OUT/"prevalence_sweep.csv")


# ----------------------------------------------------------------- 4. subjects
def cmd_subjects():
    """Leave-one-subject-out, reporting the spread rather than just the mean."""
    print("\n" + "="*70)
    print("4. PER-SUBJECT VARIANCE (leave-one-subject-out within CHB-MIT)")
    print("="*70)
    X, y, coh, sub, fil = load_features()
    chb = coh == "chb"
    subs = np.unique(sub[chb])
    rows = []
    for fname, sl in FEATSETS.items():
        aucs = []
        print("\n--- %s ---" % fname)
        for s in subs:
            te = np.where(chb & (sub == s))[0]
            tr = np.where(chb & (sub != s))[0]
            auc, ap, ntr = fit_eval(X[:, sl][tr], y[tr], X[:, sl][te], y[te])
            if np.isnan(auc):
                continue
            aucs.append(auc)
            rows.append(dict(analysis="loso", features=fname, subject=s,
                             auroc=auc, auprc=ap, n_test=len(te),
                             pos_test=int(y[te].sum())))
            mark = "  <-- below chance" if auc < 0.5 else ""
            print("   %-8s AUROC=%.3f AUPRC=%.4f (n=%d, pos=%d)%s"
                  % (s, auc, ap, len(te), int(y[te].sum()), mark))
        a = np.array(aucs)
        print("   mean=%.3f median=%.3f sd=%.3f min=%.3f max=%.3f | %d/%d below 0.6"
              % (a.mean(), np.median(a), a.std(), a.min(), a.max(),
                 (a < 0.6).sum(), len(a)))
    pd.DataFrame(rows).to_csv(OUT/"per_subject.csv", index=False)
    print("\nwrote", OUT/"per_subject.csv")


# ----------------------------------------------------------------- 5. shortcut
def cmd_shortcut():
    """Can non-seizure windows of seizure files be told apart from non-seizure
    windows of seizure-free files? If yes, models may be reading recording
    context rather than seizure physiology."""
    print("\n" + "="*70)
    print("5. SHORTCUT / RECORDING-CONTEXT TEST")
    print("="*70)
    X, y, coh, sub, fil = load_features()
    sz_file = np.zeros(len(y), bool)
    for f in np.unique(fil):
        m = fil == f
        if y[m].sum() > 0:
            sz_file[m] = True

    neg = np.where((y == 0) & (coh == "chb"))[0]
    lab = sz_file[neg].astype(int)
    print("non-seizure windows: %d from seizure files, %d from seizure-free files"
          % (lab.sum(), (1 - lab).sum()))
    if lab.sum() == 0 or (1 - lab).sum() == 0:
        print("one class empty - cannot run this test")
        return

    subs = sub[neg]
    gkf = GroupKFold(n_splits=N_FOLDS)
    aucs = []
    for k, (tri, tei) in enumerate(gkf.split(neg, lab, subs)):
        auc, ap, _ = fit_eval(X[neg][tri], lab[tri], X[neg][tei], lab[tei])
        aucs.append(auc)
        print("   fold%d AUROC=%.3f" % (k, auc))
    a = np.array(aucs)
    print("\n   mean AUROC = %.3f" % a.mean())
    if a.mean() > 0.7:
        print("   HIGH: seizure files differ systematically from seizure-free files.")
        print("   Part of the reported detection performance may reflect recording")
        print("   context rather than ictal activity. Report this explicitly.")
    elif a.mean() > 0.6:
        print("   MODERATE: some context signal is present. Worth reporting.")
    else:
        print("   LOW: little recording-level confound detectable.")
    pd.DataFrame(dict(fold=range(len(a)), auroc=a)).to_csv(
        OUT/"shortcut_test.csv", index=False)
    print("wrote", OUT/"shortcut_test.csv")


# ----------------------------------------------------------------- main
CMDS = {"fair": cmd_fair, "shuffle": cmd_shuffle, "prevalence": cmd_prevalence,
        "subjects": cmd_subjects, "shortcut": cmd_shortcut}

if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if c == "all":
        for name, fn in CMDS.items():
            try:
                fn()
            except Exception as e:
                print("\n!! %s failed: %s: %s" % (name, type(e).__name__, e))
                import traceback; traceback.print_exc()
    else:
        CMDS[c]()
    print("\ntotal %.1f min" % ((time.time() - t0) / 60))
