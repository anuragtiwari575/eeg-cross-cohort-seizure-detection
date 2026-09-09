"""
Follow-up analyses for the cross-cohort seizure detection study.

    python -u xcohort2.py loso     # leave-one-subject-out cross-patient (the key number)
    python -u xcohort2.py shift    # decode which features shift, and where
    python -u xcohort2.py norm     # does per-subject normalisation close the gap?
    python -u xcohort2.py all      # all three, back to back

Reads the .npz files already produced by xcohort.py prep.
"""
from pathlib import Path
import os, sys, warnings
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

OUT = Path(r"E:\xcohort")
RES = Path(r"E:\xcohort_results"); RES.mkdir(parents=True, exist_ok=True)

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
CHNAMES = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
           "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "P8-O2",
           "FZ-CZ", "CZ-PZ"]
NB = 5 * 17                     # 85 bandpower features, then 34 aperiodic


def feature_name(i):
    """Map a column index back to a human-readable feature name."""
    if i < NB:
        ch, b = divmod(i, len(BAND_NAMES))
        return "%s %s" % (CHNAMES[ch], BAND_NAMES[b])
    j = i - NB
    ch, k = divmod(j, 2)
    return "%s %s" % (CHNAMES[ch], "exponent" if k == 0 else "fit_r2")


def load_all():
    rows = []
    for f in sorted(OUT.glob("*.npz")):
        d = np.load(f)
        if d["X"].size == 0:
            continue
        cohort, subj, _ = f.stem.split("__", 2)
        rows.append((cohort, subj, d["X"], d["y"]))
    if not rows:
        sys.exit("no data - run 'xcohort.py prep' first")
    X = np.vstack([r[2] for r in rows])
    y = np.concatenate([r[3] for r in rows])
    coh = np.concatenate([[r[0]] * len(r[3]) for r in rows])
    sub = np.concatenate([[r[1]] * len(r[3]) for r in rows])
    ok = np.isfinite(X).all(1)
    return X[ok], y[ok], coh[ok], sub[ok]


def fit_eval(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, average_precision_score
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None, None
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
    m.fit(Xtr, ytr)
    s = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, s), average_precision_score(yte, s)


SETS = {"bandpower": slice(0, NB),
        "aperiodic": slice(NB, None),
        "both": slice(0, None)}


# ---------------------------------------------------------------- 1. LOSO
def cmd_loso():
    """Leave-one-subject-out within each cohort, and cross-cohort for comparison.

    This replaces the single arbitrary A/B split, which was the weakest part of
    the earlier analysis. LOSO gives a distribution rather than one number.
    """
    X, y, coh, sub = load_all()
    all_rows = []

    for cohort in ["chb", "siena"]:
        subs = np.unique(sub[coh == cohort])
        print("\n===== LOSO within %s (%d subjects) =====" % (cohort, len(subs)))
        for fname, sl in SETS.items():
            Xf = X[:, sl]
            aucs, aps, used = [], [], []
            for s in subs:
                te = (coh == cohort) & (sub == s)
                tr = (coh == cohort) & (sub != s)
                a, p = fit_eval(Xf[tr], y[tr], Xf[te], y[te])
                if a is None:
                    continue
                aucs.append(a); aps.append(p); used.append(s)
                all_rows.append(dict(cohort=cohort, features=fname, held_out=s,
                                     auroc=a, auprc=p, n_test=int(te.sum()),
                                     pos_test=int(y[te].sum())))
            if aucs:
                aucs = np.array(aucs)
                print("  %-10s n=%2d  AUROC mean=%.3f median=%.3f sd=%.3f "
                      "min=%.3f max=%.3f | AUPRC mean=%.4f"
                      % (fname, len(aucs), aucs.mean(), np.median(aucs), aucs.std(),
                         aucs.min(), aucs.max(), np.mean(aps)))

    print("\n===== cross-cohort (for comparison) =====")
    for fname, sl in SETS.items():
        Xf = X[:, sl]
        a1, p1 = fit_eval(Xf[coh == "chb"], y[coh == "chb"],
                          Xf[coh == "siena"], y[coh == "siena"])
        a2, p2 = fit_eval(Xf[coh == "siena"], y[coh == "siena"],
                          Xf[coh == "chb"], y[coh == "chb"])
        print("  %-10s chb->siena AUROC=%.3f AUPRC=%.4f | siena->chb AUROC=%.3f AUPRC=%.4f"
              % (fname, a1, p1, a2, p2))
        all_rows.append(dict(cohort="cross", features=fname, held_out="chb->siena",
                             auroc=a1, auprc=p1, n_test=int((coh == "siena").sum()),
                             pos_test=int(y[coh == "siena"].sum())))
        all_rows.append(dict(cohort="cross", features=fname, held_out="siena->chb",
                             auroc=a2, auprc=p2, n_test=int((coh == "chb").sum()),
                             pos_test=int(y[coh == "chb"].sum())))

    df = pd.DataFrame(all_rows)
    df.to_csv(RES / "loso_results.csv", index=False)
    print("\nwrote", RES / "loso_results.csv")

    # Is cross-cohort worse than cross-patient? One-sample test per feature set.
    from scipy.stats import wilcoxon
    print("\n===== is cross-cohort worse than LOSO-within-chb? =====")
    for fname in SETS:
        w = df[(df.cohort == "chb") & (df.features == fname)].auroc.values
        x = df[(df.cohort == "cross") & (df.features == fname)].auroc.values
        if len(w) < 5:
            continue
        # how many LOSO folds fall below the cross-cohort value?
        for tag, val in zip(["chb->siena", "siena->chb"], x):
            below = (w < val).sum()
            print("  %-10s %-12s cross=%.3f | LOSO mean=%.3f | %d/%d folds worse than cross"
                  % (fname, tag, val, w.mean(), below, len(w)))


# ---------------------------------------------------------------- 2. shift
def cmd_shift():
    """Which features shift between cohorts, and are they the discriminative ones?"""
    from scipy.stats import ks_2samp
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X, y, coh, sub = load_all()
    a, b = X[coh == "chb"], X[coh == "siena"]

    ks = np.array([ks_2samp(a[:, i], b[:, i]).statistic for i in range(X.shape[1])])

    # discriminativeness: |standardised coefficient| from a seizure model on CHB
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
    m.fit(a, y[coh == "chb"])
    imp = np.abs(m[-1].coef_.ravel())

    df = pd.DataFrame({
        "idx": np.arange(X.shape[1]),
        "feature": [feature_name(i) for i in range(X.shape[1])],
        "family": ["bandpower" if i < NB else "aperiodic" for i in range(X.shape[1])],
        "ks": ks,
        "seizure_importance": imp,
        "chb_mean": a.mean(0), "siena_mean": b.mean(0),
    }).sort_values("ks", ascending=False)
    df.to_csv(RES / "feature_shift.csv", index=False)

    print("===== 15 largest cohort shifts =====")
    print(df.head(15).to_string(index=False,
          columns=["feature", "family", "ks", "seizure_importance", "chb_mean", "siena_mean"],
          float_format=lambda v: "%.3f" % v))

    print("\n===== mean KS by family =====")
    print(df.groupby("family").ks.agg(["mean", "median", "max"]).round(3).to_string())

    print("\n===== mean KS by band =====")
    bp = df[df.family == "bandpower"].copy()
    bp["band"] = [f.split()[-1] for f in bp.feature]
    print(bp.groupby("band").ks.mean().round(3).sort_values(ascending=False).to_string())

    r = np.corrcoef(df.ks, df.seizure_importance)[0, 1]
    print("\ncorrelation between shift magnitude and seizure importance: r=%+.3f" % r)
    print("  (positive = the informative features are also the ones that shift)")
    print("\nwrote", RES / "feature_shift.csv")


# ---------------------------------------------------------------- 3. norm
def cmd_norm():
    """Does per-subject standardisation close the cross-cohort gap?

    If yes, that is an actionable recommendation: normalise per recording before
    transfer. If no, the gap is in the discriminative structure, not the scaling.
    """
    X, y, coh, sub = load_all()

    def per_subject_z(Xin):
        Z = np.empty_like(Xin)
        for s in np.unique(sub):
            m = sub == s
            mu, sd = Xin[m].mean(0), Xin[m].std(0) + 1e-8
            Z[m] = (Xin[m] - mu) / sd
        return Z

    def per_subject_rank(Xin):
        from scipy.stats import rankdata
        Z = np.empty_like(Xin)
        for s in np.unique(sub):
            m = sub == s
            Z[m] = np.apply_along_axis(lambda c: rankdata(c) / len(c), 0, Xin[m])
        return Z

    variants = {"raw": lambda v: v,
                "per-subject z": per_subject_z,
                "per-subject rank": per_subject_rank}

    for vname, fn in variants.items():
        print("\n===== %s =====" % vname)
        Xv = fn(X)
        for fname, sl in SETS.items():
            Xf = Xv[:, sl]
            a1, p1 = fit_eval(Xf[coh == "chb"], y[coh == "chb"],
                              Xf[coh == "siena"], y[coh == "siena"])
            a2, p2 = fit_eval(Xf[coh == "siena"], y[coh == "siena"],
                              Xf[coh == "chb"], y[coh == "chb"])
            print("  %-10s chb->siena AUROC=%.3f AUPRC=%.4f | siena->chb AUROC=%.3f AUPRC=%.4f"
                  % (fname, a1, p1, a2, p2))

    print("\nbase rate (positive fraction): %.4f" % y.mean())
    print("  AUPRC should be read against this, not against 0.5")


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "all"
    if c == "all":
        cmd_loso(); cmd_shift(); cmd_norm()
    else:
        {"loso": cmd_loso, "shift": cmd_shift, "norm": cmd_norm}[c]()
