"""
=============================================================================
HANDCRAFTED BASELINE AT NATIVE PREVALENCE, SCORED AT THE EVENT LEVEL
=============================================================================
    python -u handcrafted_event.py

Nothing to edit if the feature files are where the earlier scripts left
them; otherwise pass the folder:

    python -u handcrafted_event.py "D:\\xcohort"

WHY THIS EXISTS

The learned models were trained and evaluated on a subsample of the
windows (7.36% positive), because holding every window as raw time series
would have needed about 70 GB. Two consequences followed. Their evaluation
prevalence did not match the handcrafted baseline's, and their test sets
did not contain enough negatives to resample down to the native rate --- the
matched-prevalence sweep bottomed out at 5%.

The handcrafted features have neither problem. They are stored for every
window, so this script evaluates them where a deployed system actually
operates: on continuous recording at 0.34% positive, scored the way a
clinician scores a detector.

WHAT IT PRODUCES

 1. Native-prevalence performance, per setting and per feature family:
    AUROC, AUPRC and the lift over chance, with subject-level bootstrap
    intervals.

 2. Event-level sensitivity against false alarms per hour. Contiguous
    positive predictions are merged into events; an annotated seizure
    counts as detected if any predicted event overlaps it. Unlike the
    learned-model version of this analysis, the denominator here is real
    recording time, because no windows were dropped. These false-alarm
    rates are therefore directly comparable with the clinical literature.

 3. Detection delay: how far into a seizure the first positive window
    falls.

OUTPUT goes to a folder beside the feature directory.
=============================================================================
"""
from pathlib import Path
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

WIN_S    = 4.0
N_FOLDS  = 5
N_BOOT   = 2000
SEED     = 1234
MAX_TRAIN = 150_000        # cap on training rows; all positives always kept
NB       = 5 * 17          # 85 band-power features, then 34 aperiodic

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.8, "lines.linewidth": 1.2,
})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "grey": "#999999"}


# ----------------------------------------------------------------- locate
def find_features():
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        return Path(sys.argv[1])
    here = Path(__file__).resolve().parent
    for c in [Path(r"D:\xcohort"), here / "xcohort", here,
              Path.home() / "Downloads" / "xcohort", Path.cwd()]:
        if c.exists() and any(c.glob("chb*__*.npz")) or \
           (c.exists() and any(c.glob("siena*__*.npz"))):
            return c
    print("Could not find the feature .npz files (named like")
    print("  chb__chb01__chb01_01.npz).  Pass the folder as an argument:")
    print('  python handcrafted_event.py "D:\\xcohort"')
    sys.exit(1)


FEAT = find_features()
OUT  = FEAT.parent / "handcrafted_event"
RES  = OUT / "results"; RES.mkdir(parents=True, exist_ok=True)
FIG  = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
print("features : %s" % FEAT)
print("output   : %s\n" % OUT)


# ----------------------------------------------------------------- load
def load():
    Xs, ys, cs, ss, fs, os_ = [], [], [], [], [], []
    for f in sorted(FEAT.glob("*.npz")):
        d = np.load(f)
        if "X" not in d.files or d["X"].size == 0:
            continue
        try:
            cohort, subj, fname = f.stem.split("__", 2)
        except ValueError:
            continue
        n = len(d["y"])
        Xs.append(d["X"]); ys.append(d["y"])
        cs += [cohort] * n; ss += [subj] * n; fs += [fname] * n
        os_.append(np.arange(n))               # window index within recording
    if not Xs:
        sys.exit("no feature files found in %s" % FEAT)
    X = np.vstack(Xs); y = np.concatenate(ys).astype(int)
    coh = np.array(cs); sub = np.array(ss); fil = np.array(fs)
    order = np.concatenate(os_)
    ok = np.isfinite(X).all(1)
    X, y, coh, sub, fil, order = X[ok], y[ok], coh[ok], sub[ok], fil[ok], order[ok]
    print("windows %s | positives %d (%.3f%%) | subjects %d | recordings %d"
          % (X.shape, y.sum(), 100 * y.mean(), len(set(sub)), len(set(fil))))
    print("total recording time: %.1f h\n" % (len(y) * WIN_S / 3600))
    return X, y, coh, sub, fil, order


X, y, coh, sub, fil, order = load()
FEATSETS = {"bandpower": slice(0, NB), "aperiodic": slice(NB, None),
            "both": slice(0, None)}
CHB_IDX = np.where(coh == "chb")[0]
S2_SPLITS = list(GroupKFold(n_splits=N_FOLDS)
                 .split(CHB_IDX, y[CHB_IDX], sub[CHB_IDX]))


def splits():
    for k, (tri, tei) in enumerate(S2_SPLITS):
        yield "S2", "fold%d" % k, CHB_IDX[tri], CHB_IDX[tei]
    a = np.where(coh == "chb")[0]; b = np.where(coh == "siena")[0]
    yield "S3", "chb->siena", a, b
    yield "S3", "siena->chb", b, a


def fit_predict(Xtr, ytr, Xte, seed=SEED):
    if len(ytr) > MAX_TRAIN:
        r = np.random.default_rng(seed)
        pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
        neg = r.choice(neg, max(1, min(len(neg), MAX_TRAIN - len(pos))),
                       replace=False)
        keep = np.concatenate([pos, neg]); Xtr, ytr = Xtr[keep], ytr[keep]
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000, class_weight="balanced"))
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


# ----------------------------------------------------------------- events
def merge_events(mask, ordr):
    """Contiguous runs of True, in window order -> [(first, last), ...]"""
    o = np.argsort(ordr); m = mask[o]; oo = ordr[o]
    ev, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            ev.append((oo[i], oo[j])); i = j + 1
        else:
            i += 1
    return ev


def event_metrics(y_true, score, recs, ordr, thr):
    """Per recording, so that events never span a file boundary."""
    pred = score >= thr
    tp = fp = n_ev = 0
    delays = []
    hours = len(y_true) * WIN_S / 3600
    for rec in np.unique(recs):
        m = recs == rec
        tev = merge_events(y_true[m] == 1, ordr[m])
        pev = merge_events(pred[m], ordr[m])
        n_ev += len(tev)
        hit = {}
        for a, b in pev:
            ov = [k for k, (c, e) in enumerate(tev) if not (b < c or a > e)]
            if ov:
                for k in ov:
                    c = tev[k][0]
                    d = max(0, a - c) * WIN_S
                    if k not in hit or d < hit[k]:
                        hit[k] = d
            else:
                fp += 1
        tp += len(hit)
        delays += list(hit.values())
    return tp, fp, n_ev, hours, delays


# ----------------------------------------------------------------- 1
def native_prevalence():
    print("=" * 70)
    print("1. NATIVE-PREVALENCE PERFORMANCE  (%.3f%% positive)" % (100 * y.mean()))
    print("=" * 70)
    print("No resampling: every window of the held-out set is scored.\n")
    rng = np.random.default_rng(SEED)
    rows = []
    for setting, fold, tr, te in splits():
        for fname, sl in FEATSETS.items():
            s = fit_predict(X[:, sl][tr], y[tr], X[:, sl][te])
            auc = roc_auc_score(y[te], s); ap = average_precision_score(y[te], s)
            # subject-level bootstrap
            subs = np.unique(sub[te]); by = {u: np.where(sub[te] == u)[0] for u in subs}
            b = []
            for _ in range(N_BOOT):
                pick = rng.choice(subs, len(subs), replace=True)
                idx = np.concatenate([by[u] for u in pick])
                if len(np.unique(y[te][idx])) > 1:
                    b.append(roc_auc_score(y[te][idx], s[idx]))
            lo, hi = np.percentile(b, [2.5, 97.5]) if b else (np.nan, np.nan)
            prev = float(y[te].mean())
            rows.append(dict(setting=setting, fold=fold, features=fname,
                             auroc=auc, lo=lo, hi=hi, auprc=ap, prevalence=prev,
                             lift=ap / prev, n_test=len(te),
                             pos_test=int(y[te].sum()), n_subjects=len(subs)))
            print("  %-3s %-12s %-10s AUROC=%.3f [%.3f, %.3f]  AUPRC=%.4f  lift=%.1fx"
                  % (setting, fold, fname, auc, lo, hi, ap, ap / prev))
    d = pd.DataFrame(rows)
    d.to_csv(RES / "native_prevalence.csv", index=False)
    print("\nmeans by setting and feature set:")
    print(d.groupby(["setting", "features"])[["auroc", "auprc", "lift"]]
            .mean().round(4).to_string())
    print("\nwrote native_prevalence.csv")
    return d


# ----------------------------------------------------------------- 2
def event_level():
    print("\n" + "=" * 70)
    print("2. EVENT-LEVEL SENSITIVITY AND FALSE ALARMS PER HOUR")
    print("=" * 70)
    print("Every window is retained, so the denominator is real recording time")
    print("and these rates are directly comparable with the clinical")
    print("literature -- unlike the learned-model version, which was computed")
    print("on a subsample.\n")
    rows, delay_rows = [], []
    for setting, fold, tr, te in splits():
        for fname, sl in FEATSETS.items():
            if fname != "both":            # keep the run tractable
                continue
            s = fit_predict(X[:, sl][tr], y[tr], X[:, sl][te])
            qs = np.unique(np.quantile(s, np.linspace(0.90, 0.99995, 45)))
            for thr in qs:
                tp, fp, ne, hrs, dl = event_metrics(y[te], s, fil[te], order[te], thr)
                if ne == 0:
                    continue
                rows.append(dict(setting=setting, fold=fold, features=fname,
                                 threshold=float(thr), sensitivity=tp / ne,
                                 fa_per_hour=fp / hrs, n_events=ne, hours=hrs))
                if dl:
                    delay_rows.append(dict(setting=setting, fold=fold,
                                           fa_per_hour=fp / hrs,
                                           median_delay_s=float(np.median(dl)),
                                           n_detected=len(dl)))
    d = pd.DataFrame(rows); d.to_csv(RES / "event_level_handcrafted.csv", index=False)
    dd = pd.DataFrame(delay_rows)
    if not dd.empty:
        dd.to_csv(RES / "detection_delay.csv", index=False)

    print("sensitivity at fixed false-alarm rates (mean over folds):")
    for setting in ("S2", "S3"):
        sub_d = d[d.setting == setting]
        if sub_d.empty:
            continue
        print("--- %s ---" % setting)
        for tgt in (0.1, 0.5, 1.0, 2.0, 5.0):
            v = []
            for _, g in sub_d.groupby("fold"):
                ok = g[g.fa_per_hour <= tgt]
                v.append(ok.sensitivity.max() if len(ok) else 0.0)
            print("   at %.1f FA/h: sensitivity %.3f" % (tgt, np.mean(v)))
        print()

    if not dd.empty:
        near = dd[(dd.fa_per_hour > 0.5) & (dd.fa_per_hour < 2.0)]
        if not near.empty:
            print("median detection delay near 1 FA/h: %.0f s (over %d folds)"
                  % (near.median_delay_s.median(), near.fold.nunique()))

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for setting, col in (("S2", C["blue"]), ("S3", C["green"])):
        sd = d[d.setting == setting]
        if sd.empty:
            continue
        g = sd.groupby("threshold")[["fa_per_hour", "sensitivity"]].mean()
        g = g.sort_values("fa_per_hour")
        ax.plot(g.fa_per_hour, g.sensitivity, marker="o", ms=3, color=col,
                label=setting)
    for x in (0.1, 1.0):
        ax.axvline(x, color=C["grey"], ls=":", lw=.8)
    ax.set_xscale("log"); ax.set_xlabel("false alarms per hour")
    ax.set_ylabel("event-level sensitivity"); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.set_title("handcrafted features, continuous data", loc="left")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIG / ("F13_event_handcrafted." + e))
    plt.close(fig)
    print("wrote event_level_handcrafted.csv, F13_event_handcrafted")
    return d


if __name__ == "__main__":
    try:
        native_prevalence()
    except Exception as e:
        print("FAILED native_prevalence: %s: %s" % (type(e).__name__, e))
        import traceback; traceback.print_exc()
    try:
        event_level()
    except Exception as e:
        print("FAILED event_level: %s: %s" % (type(e).__name__, e))
        import traceback; traceback.print_exc()
    print("\nresults : %s" % RES)
    print("figures : %s" % FIG)
