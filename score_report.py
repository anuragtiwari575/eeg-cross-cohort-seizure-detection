"""
=============================================================================
METRICS FROM THE SAVED SCORES
=============================================================================
    python -u score_report.py

Reads every .npz in D:\\native_inference\\scores and recomputes the full set
of metrics from it. Nothing is re-run on the GPU.

WHY THIS IS SEPARATE FROM infer_local.py

infer_local.py writes the prevalence sweep and the event curves only for the
runs it scores in that session. When it resumes after an interruption it
skips the runs already on disk, so those tables end up holding only the
runs from the final session. The per-window scores, however, are complete:
every run that finished wrote its scores before anything else. Recomputing
from them is therefore both simpler and safer than trying to merge partial
tables, and it can be repeated at no cost whenever the analysis changes.

WHAT IT PRODUCES  (D:\\native_inference\\results)
    native_metrics_all.csv      AUROC, AUPRC, lift, Brier per run
    native_prevalence_all.csv   the same runs across resampled prevalences
    native_event_all.csv        event-level operating points
    F14_native_comparison.pdf   learned against handcrafted, both metrics
    F15_event_native.pdf        event-level curves for both representations

The handcrafted figures it compares against were produced by
handcrafted_event.py on exactly the same windows, so the two are directly
comparable: same recordings, same prevalence, same definition of an event,
and a denominator of real recording time in both cases.
=============================================================================
"""
from pathlib import Path
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

SC   = Path(r"D:\native_inference\scores")
RES  = Path(r"D:\native_inference\results"); RES.mkdir(parents=True, exist_ok=True)
FIG  = Path(r"D:\native_inference\figures"); FIG.mkdir(parents=True, exist_ok=True)
HAND = Path(r"D:\handcrafted_event\results")     # for the comparison
WIN_S, N_BOOT, SEED = 4.0, 2000, 1234
PREVS = [0.50, 0.25, 0.10, 0.0736, 0.05, 0.01, None]

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
     "red": "#D55E00", "grey": "#999999", "purple": "#CC79A7"}

files = sorted(SC.glob("*.npz"))
if not files:
    sys.exit("no scores in %s" % SC)
print("%d scored runs\n" % len(files))


def parse(rid):
    p = rid.split("_")
    si = next(i for i, x in enumerate(p) if x.startswith("s") and x[1:].isdigit())
    fold = "_".join(p[2:si]).replace("2siena", "->siena").replace("2chb", "->chb")
    return p[0], p[1], fold, int(p[si][1:])


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


rng = np.random.default_rng(SEED)
met, prv, evt = [], [], []

for k, f in enumerate(files, 1):
    z = np.load(f, allow_pickle=True)
    s = z["score"].astype(np.float64); y = z["label"].astype(int)
    key = "rec" if "rec" in z.files else ("file" if "file" in z.files else None)
    rec = z[key].astype(str) if key else np.array(["?"] * len(y))
    od = z["order"].astype(int) if "order" in z.files else np.arange(len(y))
    if len(np.unique(y)) < 2:
        continue
    name, setting, fold, seed = parse(f.stem)
    prev = float(y.mean()); ap = average_precision_score(y, s)
    met.append(dict(run_id=f.stem, model=name, setting=setting, fold=fold,
                    seed=seed, n=len(y), pos=int(y.sum()), prevalence=prev,
                    auroc=float(roc_auc_score(y, s)), auprc=float(ap),
                    lift=ap / prev, brier=float(brier_score_loss(y, s)),
                    hours=len(y) * WIN_S / 3600))

    pos_i = np.where(y == 1)[0]; neg_i = np.where(y == 0)[0]
    for p in PREVS:
        if p is None:
            idx = np.arange(len(y))
        else:
            nn = int(round(len(pos_i) * (1 - p) / p))
            if nn > len(neg_i):
                continue
            idx = np.concatenate([pos_i, rng.choice(neg_i, nn, replace=False)])
        prv.append(dict(run_id=f.stem, model=name, setting=setting, fold=fold,
                        seed=seed, target_prev=p if p else prev,
                        actual_prev=float(y[idx].mean()),
                        auroc=float(roc_auc_score(y[idx], s[idx])),
                        auprc=float(average_precision_score(y[idx], s[idx]))))

    hours = len(y) * WIN_S / 3600
    for thr in np.unique(np.quantile(s, np.linspace(0.90, 0.99995, 40))):
        pred = s >= thr
        tp = fp = ne = 0; delays = []
        for r in np.unique(rec):
            m = rec == r
            tev = merge_events(y[m] == 1, od[m])
            pev = merge_events(pred[m], od[m])
            ne += len(tev); hit = {}
            for a, b in pev:
                ov = [i for i, (c, e) in enumerate(tev) if not (b < c or a > e)]
                if ov:
                    for i in ov:
                        d = max(0, a - tev[i][0]) * WIN_S
                        if i not in hit or d < hit[i]:
                            hit[i] = d
                else:
                    fp += 1
            tp += len(hit); delays += list(hit.values())
        if ne:
            evt.append(dict(run_id=f.stem, model=name, setting=setting,
                            fold=fold, seed=seed, threshold=float(thr),
                            sensitivity=tp / ne, fa_per_hour=fp / hours,
                            n_events=ne, hours=hours,
                            median_delay_s=float(np.median(delays)) if delays else np.nan))
    if k % 20 == 0:
        print("  %d/%d" % (k, len(files)), flush=True)

M = pd.DataFrame(met); P = pd.DataFrame(prv); E = pd.DataFrame(evt)
M.to_csv(RES / "native_metrics_all.csv", index=False)
P.to_csv(RES / "native_prevalence_all.csv", index=False)
E.to_csv(RES / "native_event_all.csv", index=False)

# ----------------------------------------------------------------- report
print("\n" + "#" * 70)
print("NATIVE PREVALENCE (%.3f%% positive), CONTINUOUS DATA" % (100 * M.prevalence.mean()))
print("#" * 70)
print("\n=== learned representations ===")
print(M.groupby(["setting", "model"])[["auroc", "auprc", "lift"]]
       .mean().round(4).to_string())
print("\n=== learned, by setting ===")
print(M.groupby("setting")[["auroc", "auprc", "lift"]].mean().round(4).to_string())

hand = HAND / "native_prevalence.csv"
if hand.exists():
    H = pd.read_csv(hand)
    Hb = H[H.features == "both"]
    print("\n=== handcrafted, same windows ===")
    print(Hb.groupby("setting")[["auroc", "auprc", "lift"]].mean().round(4).to_string())
    for st in ("S3",):
        a = M[M.setting == st]; b = Hb[Hb.setting == st]
        if len(a) and len(b):
            print("\n%s: learned AUROC %.3f vs handcrafted %.3f  (gap %.3f)"
                  % (st, a.auroc.mean(), b.auroc.mean(), a.auroc.mean()-b.auroc.mean()))
            print("%s: learned lift %.1fx vs handcrafted %.1fx  (ratio %.1fx)"
                  % (st, a.lift.mean(), b.lift.mean(), a.lift.mean()/b.lift.mean()))

print("\n=== architecture spread vs seed SD, at native prevalence ===")
g = M.groupby(["model", "setting", "fold"]).auroc.agg(["mean", "std"]).reset_index()
for st in sorted(M.setting.unique()):
    a = M[M.setting == st].groupby("model").auroc.mean()
    sd = g[g.setting == st]["std"].mean()
    if len(a) > 1 and np.isfinite(sd) and sd > 0:
        print("  %s: spread %.3f | seed SD %.3f | ratio %.2f"
              % (st, a.max() - a.min(), sd, (a.max() - a.min()) / sd))

print("\n=== event level, real recording hours ===")
for st in sorted(E.setting.unique()):
    sub = E[E.setting == st]
    print("--- %s (%.0f h) ---" % (st, sub.hours.mean()))
    for tgt in (0.1, 0.5, 1.0, 2.0, 5.0):
        v = [g2[g2.fa_per_hour <= tgt].sensitivity.max()
             if (g2.fa_per_hour <= tgt).any() else 0.0
             for _, g2 in sub.groupby("run_id")]
        print("   at %.1f FA/h: sensitivity %.3f" % (tgt, np.mean(v)))
    near = sub[(sub.fa_per_hour > 0.5) & (sub.fa_per_hour < 2.0)]
    if not near.empty and near.median_delay_s.notna().any():
        print("   median detection delay near 1 FA/h: %.0f s"
              % near.median_delay_s.median())

hev = HAND / "event_level_handcrafted.csv"
if hev.exists():
    HE = pd.read_csv(hev)
    print("\n--- handcrafted, same data, for comparison ---")
    for st in sorted(HE.setting.unique()):
        sub = HE[HE.setting == st]
        line = []
        for tgt in (0.1, 1.0, 2.0, 5.0):
            v = [g2[g2.fa_per_hour <= tgt].sensitivity.max()
                 if (g2.fa_per_hour <= tgt).any() else 0.0
                 for _, g2 in sub.groupby("fold")]
            line.append("%.1f: %.3f" % (tgt, np.mean(v)))
        print("   %s  " % st + "  ".join(line))

# ----------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
for k, st in enumerate(sorted(M.setting.unique())):
    sub = P[P.setting == st]
    if sub.empty:
        continue
    g2 = sub.groupby("target_prev")[["auroc", "auprc"]].mean().sort_index()
    col = [C["blue"], C["green"], C["purple"]][k % 3]
    ax[0].plot(g2.index*100, g2.auroc, marker="o", ms=4, color=col, label="learned "+st)
    ax[1].plot(g2.index*100, g2.auprc, marker="o", ms=4, color=col, label="learned "+st)
if hand.exists():
    Hb = pd.read_csv(hand); Hb = Hb[Hb.features == "both"]
    for st, mk in (("S3", "s"), ("S2", "^")):
        h = Hb[Hb.setting == st]
        if len(h):
            ax[0].scatter(h.prevalence*100, h.auroc, marker=mk, s=22,
                          color=C["orange"], zorder=5,
                          label="handcrafted " + st)
            ax[1].scatter(h.prevalence*100, h.auprc, marker=mk, s=22,
                          color=C["orange"], zorder=5, label="handcrafted " + st)
xs = np.array([p for p in PREVS if p]) * 100
ax[1].plot(np.sort(xs), np.sort(xs)/100, ls=":", color=C["grey"], label="chance")
for a in ax:
    a.set_xscale("log"); a.set_xlabel("evaluation prevalence (%)")
    a.axvline(0.34, color="#666", ls="--", lw=.9)
ax[0].set_ylabel("AUROC"); ax[0].set_ylim(.5, 1.02)
ax[0].legend(loc="lower right", fontsize=6); ax[0].set_title("(a) AUROC", loc="left")
ax[1].set_yscale("log"); ax[1].set_ylabel("AUPRC")
ax[1].legend(loc="upper left", fontsize=6); ax[1].set_title("(b) AUPRC", loc="left")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(FIG / ("F14_native_comparison." + e))
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
for st, col in (("S3", C["blue"]), ("S1", C["grey"])):
    sub = E[E.setting == st]
    if sub.empty:
        continue
    g2 = sub.groupby("threshold")[["fa_per_hour", "sensitivity"]].mean().sort_values("fa_per_hour")
    ax.plot(g2.fa_per_hour, g2.sensitivity, marker="o", ms=3, color=col,
            label="learned " + st)
if hev.exists():
    HE = pd.read_csv(hev)
    h = HE[HE.setting == "S3"]
    if len(h):
        g2 = h.groupby("threshold")[["fa_per_hour", "sensitivity"]].mean().sort_values("fa_per_hour")
        ax.plot(g2.fa_per_hour, g2.sensitivity, marker="s", ms=3,
                color=C["orange"], label="handcrafted S3")
for x in (0.1, 1.0):
    ax.axvline(x, color=C["grey"], ls=":", lw=.8)
ax.set_xscale("log"); ax.set_xlabel("false alarms per hour")
ax.set_ylabel("event-level sensitivity"); ax.set_ylim(0, 1.02)
ax.legend(loc="lower right", fontsize=6)
ax.set_title("continuous data, real recording hours", loc="left")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(FIG / ("F15_event_native." + e))
plt.close(fig)

print("\nwritten to %s" % RES)
print("figures  F14_native_comparison, F15_event_native  in %s" % FIG)
