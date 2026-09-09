"""
=============================================================================
ANALYSES FROM SAVED PREDICTIONS
=============================================================================
Just run it. Nothing to edit.

    python -u preds_analyses.py

It searches the usual places for the prediction files written by the sweep,
puts its output beside them, and runs four analyses on CPU. No GPU, no
retraining -- everything uses the per-window scores that were already saved.

WHAT IT COMPUTES AND WHY

 1. MATCHED PREVALENCE
    The learned models were evaluated at 7.36% positive while the
    handcrafted features were evaluated at the native 0.34%, so the two
    were not directly comparable. Here each model's own test windows are
    resampled across a range of prevalences with the trained model held
    fixed, so any remaining difference is attributable to the
    representation rather than to the class balance.

 2. SUBJECT-LEVEL BOOTSTRAP
    Forty thousand windows are not forty thousand independent
    observations; they come from a few dozen people. Resampling windows
    would give intervals several times too narrow. Subjects are resampled
    instead, with all of a subject's windows moving together.

 3. EVENT-LEVEL SENSITIVITY AND FALSE ALARMS PER HOUR
    Window-level AUROC is not what a clinician acts on. Contiguous positive
    predictions are merged into events; an annotated seizure counts as
    detected if any predicted event overlaps it, and every other predicted
    event is a false alarm.

 4. RELIABILITY DIAGRAMS
    ECE is a single number; the curve behind it shows whether the model is
    over- or under-confident.

CAVEAT, stated in the output as well
    The windows are a subsample of continuous recording, so the hours in
    the false-alarm rate are effective rather than wall-clock. The rate is
    therefore optimistic. It is comparable between models here, not against
    figures in the literature.
=============================================================================
"""
from pathlib import Path
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

WIN_S, N_BOOT, SEED = 4.0, 2000, 1234
PREVS = [0.50, 0.25, 0.10, 0.0736, 0.05, 0.01, 0.0034]


# ----------------------------------------------------------------- locate
def find_preds():
    """Look wherever the prediction files plausibly are."""
    here = Path(__file__).resolve().parent
    cands = [here, here / "preds_archive", here / "preds", here / "preds_work",
             Path.home() / "Downloads", Path.home() / "Downloads" / "preds_archive",
             Path.home() / "Downloads" / "preds", Path(r"D:\preds"),
             Path(r"D:\xcohort_preds"), Path.cwd()]
    seen, best = set(), None
    for c in cands:
        try:
            if not c.exists() or str(c) in seen:
                continue
            seen.add(str(c))
            f = [p for p in c.rglob("*.npz") if p.stat().st_size < 5e8]
            if f and (best is None or len(f) > len(best[1])):
                best = (c, f)
        except Exception:
            continue
    if best is None:
        print("Could not find any .npz prediction files.")
        print("Searched:")
        for c in seen:
            print("   ", c)
        print("\nPut the unzipped prediction files next to this script and")
        print("run it again, or pass the folder as an argument:")
        print("   python preds_analyses.py \"C:\\path\\to\\preds\"")
        sys.exit(1)
    return best


if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
    PRED_DIR = Path(sys.argv[1])
    FILES = [p for p in PRED_DIR.rglob("*.npz") if p.stat().st_size < 5e8]
else:
    PRED_DIR, FILES = find_preds()

OUT = PRED_DIR.parent / "analysis_output"
RES = OUT / "results"; RES.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
print("predictions : %s  (%d files)" % (PRED_DIR, len(FILES)))
print("output       : %s\n" % OUT)

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
from sklearn.metrics import roc_auc_score, average_precision_score


# ----------------------------------------------------------------- load
def parse_name(stem):
    p = stem.split("_")
    try:
        si = next(i for i, x in enumerate(p) if x.startswith("s") and x[1:].isdigit())
    except StopIteration:
        return p[0], p[1] if len(p) > 1 else "?", "?", 0
    fold = "_".join(p[2:si]).replace("2siena", "->siena").replace("2chb", "->chb")
    return p[0], p[1], fold, int(p[si][1:])


def load_runs():
    runs = []
    for f in sorted(FILES):
        try:
            d = np.load(f, allow_pickle=True)
            if not {"score", "label"} <= set(d.files):
                continue
            m, st, fold, seed = parse_name(f.stem)
            runs.append(dict(model=m, setting=st, fold=fold, seed=seed,
                             score=d["score"].astype(np.float64),
                             label=d["label"].astype(int),
                             subject=d["subject"].astype(str) if "subject" in d.files
                                     else np.array(["?"] * len(d["label"])),
                             order=d["order"].astype(int) if "order" in d.files
                                   else np.arange(len(d["label"]))))
        except Exception as e:
            print("  skipped %s (%s)" % (f.name, type(e).__name__))
    if not runs:
        sys.exit("no usable prediction files")
    print("loaded %d runs" % len(runs))
    print(pd.DataFrame([{k: r[k] for k in ("model", "setting", "seed")}
                        for r in runs]).groupby(["model", "setting"]).size().to_string())
    return runs


# ----------------------------------------------------------------- 1
def matched(runs):
    print("\n" + "=" * 70)
    print("1. MATCHED-PREVALENCE COMPARISON")
    print("=" * 70)
    print("Trained model fixed; only the evaluation class balance moves.")
    print("Handcrafted reference at native prevalence: 0.709 (S2), 0.684 (S3).\n")
    rng = np.random.default_rng(SEED)
    rows = []
    for r in runs:
        y, s = r["label"], r["score"]
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        if len(pos) < 10 or len(neg) < 10:
            continue
        for p in PREVS:
            n_neg = int(round(len(pos) * (1 - p) / p))
            if n_neg > len(neg):
                continue
            idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
            rows.append(dict(model=r["model"], setting=r["setting"], fold=r["fold"],
                             seed=r["seed"], target_prev=p, n=len(idx),
                             auroc=roc_auc_score(y[idx], s[idx]),
                             auprc=average_precision_score(y[idx], s[idx])))
    d = pd.DataFrame(rows)
    if d.empty:
        print("nothing to do"); return
    d.to_csv(RES / "matched_prevalence.csv", index=False)
    for st in ("S1", "S2", "S3"):
        sub = d[d.setting == st]
        if sub.empty:
            continue
        print("--- %s ---" % st)
        for p, row in sub.groupby("target_prev")[["auroc", "auprc"]].mean().iterrows():
            print("   prevalence %-7s AUROC=%.3f  AUPRC=%.4f  (lift %.1fx)"
                  % ("%.2f%%" % (100 * p), row.auroc, row.auprc, row.auprc / p))
        print()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for k, st in enumerate(("S2", "S3")):
        sub = d[d.setting == st]
        if sub.empty:
            continue
        g = sub.groupby("target_prev")[["auroc", "auprc"]].mean()
        col = [C["blue"], C["green"]][k]
        ax[0].plot(g.index * 100, g.auroc, marker="o", ms=4, color=col, label=st)
        ax[1].plot(g.index * 100, g.auprc, marker="o", ms=4, color=col, label=st)
    xs = np.array(PREVS) * 100
    ax[1].plot(xs, xs / 100, ls=":", color=C["grey"], label="chance")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("evaluation prevalence (%)")
        a.axvspan(20, 60, color=C["grey"], alpha=.12, lw=0)
        a.axvline(0.34, color="#666", ls="--", lw=.9)
    ax[0].set_ylabel("AUROC"); ax[0].set_ylim(.5, 1.0)
    ax[0].legend(loc="lower right"); ax[0].set_title("(a) AUROC", loc="left")
    ax[1].set_yscale("log"); ax[1].set_ylabel("AUPRC")
    ax[1].legend(loc="upper left"); ax[1].set_title("(b) AUPRC", loc="left")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIG / ("F10_matched_prevalence." + e))
    plt.close(fig)
    print("wrote matched_prevalence.csv, F10_matched_prevalence")


# ----------------------------------------------------------------- 2
def bootstrap(runs):
    print("\n" + "=" * 70)
    print("2. SUBJECT-LEVEL BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)
    print("Subjects resampled with replacement, all of a subject's windows")
    print("moving together. Resampling windows would give intervals several")
    print("times too narrow.\n")
    rng = np.random.default_rng(SEED)
    rows = []
    for r in runs:
        y, s, sb = r["label"], r["score"], r["subject"]
        subs = np.unique(sb)
        auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan
        lo = hi = np.nan
        if len(subs) >= 3:
            by = {u: np.where(sb == u)[0] for u in subs}
            b = []
            for _ in range(N_BOOT):
                pick = rng.choice(subs, len(subs), replace=True)
                idx = np.concatenate([by[u] for u in pick])
                if len(np.unique(y[idx])) > 1:
                    b.append(roc_auc_score(y[idx], s[idx]))
            if b:
                lo, hi = np.percentile(b, [2.5, 97.5])
        rows.append(dict(model=r["model"], setting=r["setting"], fold=r["fold"],
                         seed=r["seed"], auroc=auc, lo=lo, hi=hi,
                         n_subjects=len(subs)))
    d = pd.DataFrame(rows)
    d.to_csv(RES / "bootstrap_ci.csv", index=False)
    for st, g in d.groupby("setting"):
        w = (g.hi - g.lo).mean()
        print("  %-3s AUROC %.3f  mean 95%% CI width %s  (%d runs, %d subjects)"
              % (st, g.auroc.mean(),
                 "%.3f" % w if np.isfinite(w) else "n/a (single subject)",
                 len(g), g.n_subjects.iloc[0]))
    dd = d.dropna(subset=["lo"])
    if not dd.empty:
        print("\n  widest intervals:")
        for _, r in dd.assign(w=dd.hi - dd.lo).nlargest(3, "w").iterrows():
            print("   %-3s %-12s s%d  %.3f [%.3f, %.3f]"
                  % (r.setting, r.fold, r.seed, r.auroc, r.lo, r.hi))
    print("wrote bootstrap_ci.csv")


# ----------------------------------------------------------------- 3
def merge_events(mask, order):
    o = np.argsort(order); m = mask[o]; oo = order[o]
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


def event(runs):
    print("\n" + "=" * 70)
    print("3. EVENT-LEVEL SENSITIVITY AND FALSE ALARMS PER HOUR")
    print("=" * 70)
    print("Contiguous positive predictions merged into events; a seizure counts")
    print("as detected if any predicted event overlaps it.")
    print("CAVEAT: these windows are a subsample of continuous recording, so the")
    print("hours are effective rather than wall-clock and the false-alarm rate")
    print("is optimistic. Comparable between models, not against the literature.\n")
    rows = []
    for r in runs:
        y, s, sb, od = r["label"], r["score"], r["subject"], r["order"]
        if y.sum() == 0:
            continue
        for thr in np.unique(np.quantile(s, np.linspace(0.50, 0.9995, 40))):
            pred = s >= thr
            tp = fp = ne = 0; hrs = 0.0
            for u in np.unique(sb):
                m = sb == u
                hrs += m.sum() * WIN_S / 3600
                tev = merge_events(y[m] == 1, od[m])
                pev = merge_events(pred[m], od[m])
                ne += len(tev); hit = set()
                for a, b in pev:
                    ov = [k for k, (c, e) in enumerate(tev) if not (b < c or a > e)]
                    if ov:
                        hit.update(ov)
                    else:
                        fp += 1
                tp += len(hit)
            if ne and hrs:
                rows.append(dict(model=r["model"], setting=r["setting"],
                                 fold=r["fold"], seed=r["seed"],
                                 threshold=float(thr), sensitivity=tp / ne,
                                 fa_per_hour=fp / hrs, n_events=ne, hours=hrs))
    d = pd.DataFrame(rows)
    if d.empty:
        print("nothing to do"); return
    d.to_csv(RES / "event_level.csv", index=False)
    for st in ("S1", "S2", "S3"):
        sub = d[d.setting == st]
        if sub.empty:
            continue
        print("--- %s ---" % st)
        for tgt in (0.1, 0.5, 1.0, 2.0):
            v = []
            for _, g in sub.groupby(["model", "fold", "seed"]):
                ok = g[g.fa_per_hour <= tgt]
                v.append(ok.sensitivity.max() if len(ok) else 0.0)
            print("   at %.1f FA/h: sensitivity %.3f" % (tgt, np.mean(v)))
        print()
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for st, col in (("S1", C["grey"]), ("S2", C["blue"]), ("S3", C["green"])):
        sub = d[d.setting == st]
        if sub.empty:
            continue
        g = sub.groupby("threshold")[["fa_per_hour", "sensitivity"]].mean()
        g = g.sort_values("fa_per_hour")
        ax.plot(g.fa_per_hour, g.sensitivity, marker="o", ms=3, color=col, label=st)
    ax.set_xscale("log"); ax.set_xlabel("false alarms per hour")
    ax.set_ylabel("event-level sensitivity"); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right"); ax.set_title("event-level operating curve", loc="left")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIG / ("F11_event_level." + e))
    plt.close(fig)
    print("wrote event_level.csv, F11_event_level")


# ----------------------------------------------------------------- 4
def calib(runs):
    print("\n" + "=" * 70)
    print("4. RELIABILITY DIAGRAMS")
    print("=" * 70)
    bins = np.linspace(0, 1, 16)
    setts = [s for s in ("S1", "S2", "S3") if any(r["setting"] == s for r in runs)]
    fig, axes = plt.subplots(1, max(len(setts), 1),
                             figsize=(2.5 * max(len(setts), 1), 2.7), sharey=True)
    axes = np.atleast_1d(axes)
    out = []
    for ax, st in zip(axes, setts):
        sel = [r for r in runs if r["setting"] == st]
        s = np.concatenate([r["score"] for r in sel])
        y = np.concatenate([r["label"] for r in sel])
        xs, ys = [], []
        for i in range(15):
            m = (s >= bins[i]) & (s <= 1.0 if i == 14 else s < bins[i + 1])
            if m.sum() > 30:
                xs.append(s[m].mean()); ys.append(y[m].mean())
                out.append(dict(setting=st, bin_mid=s[m].mean(),
                                observed=y[m].mean(), n=int(m.sum())))
        ax.plot([0, 1], [0, 1], ls=":", color=C["grey"], lw=.9)
        ax.plot(xs, ys, marker="o", ms=3.5, color=C["blue"])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("predicted probability")
        ax.set_title("%s (%d runs)" % (st, len(sel)), loc="left")
        print("  %-3s mean predicted %.3f vs observed %.3f  ->  %s"
              % (st, s.mean(), y.mean(),
                 "over-confident" if s.mean() > y.mean() else "under-confident"))
    axes[0].set_ylabel("observed frequency")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIG / ("F12_reliability." + e))
    plt.close(fig)
    pd.DataFrame(out).to_csv(RES / "reliability.csv", index=False)
    print("wrote reliability.csv, F12_reliability")


if __name__ == "__main__":
    runs = load_runs()
    for fn in (matched, bootstrap, calib, event):
        try:
            fn(runs)
        except Exception as e:
            print("FAILED %s: %s: %s" % (fn.__name__, type(e).__name__, e))
            import traceback; traceback.print_exc()
    print("\nresults : %s" % RES)
    print("figures : %s" % FIG)
