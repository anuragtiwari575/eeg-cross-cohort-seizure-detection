"""
=============================================================================
RESULTS FIGURES  (F2-F5)
=============================================================================
    python -u make_figures.py all

    python -u make_figures.py f2   # architectures across S1/S2/S3
    python -u make_figures.py f3   # prevalence sweep (AUROC vs AUPRC)
    python -u make_figures.py f4   # per-subject spread
    python -u make_figures.py f5   # negative controls

Reads from E:\\cpu_results\\ :
    prevalence_sweep.csv, per_subject.csv, label_shuffle.csv,
    shortcut_test.csv, fair_baseline.csv, results_dl.csv

F2 needs results_dl.csv - download it from the Kaggle notebook output and
drop it into E:\\cpu_results\\ first.

Writes 300-dpi PNG and vector PDF to E:\\figures\\.

Changes in this revision
    F3  legend moved to lower right, clear of the curves
    F4  the below-chance counts moved above the axis, clear of the whiskers
    F5  unchanged (correct as rendered)
=============================================================================
"""
from pathlib import Path
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(r"E:\cpu_results")
FIG = Path(r"E:\figures"); FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2,
})

C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999",
     "sky": "#56B4E9"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("%s.%s" % (name, ext)))
    plt.close(fig)
    print("  wrote %s.png / .pdf" % name)


def need(fname):
    p = RES / fname
    if not p.exists():
        print("  SKIP: %s not found" % p)
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------- F2
def fig2():
    print("F2 architectures")
    dl = need("results_dl.csv")
    if dl is None:
        print("  (download results_dl.csv from the Kaggle notebook first)")
        return
    dl = dl[dl.auroc.notna()]
    piv = dl.pivot_table(index="model", columns="setting",
                         values="auroc", aggfunc="mean")
    for c in ("S1", "S2", "S3"):
        if c not in piv:
            piv[c] = np.nan
    piv = piv[["S1", "S2", "S3"]].dropna(how="all").sort_values("S2")

    lin = need("fair_baseline.csv")
    lin_s2 = lin_s3 = None
    if lin is not None:
        b = lin[lin.features == "both"]
        lin_s2 = b[b.setting == "S2"].auroc.mean()
        lin_s3 = b[b.setting == "S3"].auroc.mean()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3),
                             gridspec_kw=dict(width_ratios=[1.45, 1]))

    ax = axes[0]
    x = [0, 1, 2]
    for m, row in piv.iterrows():
        ax.plot(x, [row.S1, row.S2, row.S3], marker="o", ms=3.5,
                color=C["blue"], alpha=.5, lw=1.0)
        ax.annotate(m, (2.05, row.S3), va="center", fontsize=5.8,
                    color="#333333")
    ax.plot(x, [piv.S1.mean(), piv.S2.mean(), piv.S3.mean()],
            marker="s", ms=5, color=C["red"], lw=2.0, zorder=5,
            label="mean, learned representations")
    if lin_s2 is not None:
        ax.hlines([lin_s2], .82, 1.18, color=C["orange"], lw=2.2, zorder=5)
        ax.hlines([lin_s3], 1.82, 2.18, color=C["orange"], lw=2.2, zorder=5)
        ax.plot([], [], color=C["orange"], lw=2.2,
                label="handcrafted features")
    ax.axhline(.5, color=C["grey"], ls=":", lw=.8)
    ax.text(-.12, .505, "chance", fontsize=6, color=C["grey"], va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(["S1\nwithin-patient", "S2\ncross-patient",
                        "S3\ncross-cohort"])
    ax.set_ylabel("AUROC")
    ax.set_xlim(-.20, 2.95)
    ax.set_ylim(.45, 1.0)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02))
    ax.set_title("(a) generalisation by setting", loc="left")

    ax = axes[1]
    d1 = piv.S1 - piv.S2
    d2 = piv.S2 - piv.S3
    yy = np.arange(len(piv))
    ax.barh(yy - .2, d1, height=.38, color=C["blue"],
            label="S1 $\\rightarrow$ S2  (new patient)")
    ax.barh(yy + .2, d2, height=.38, color=C["sky"],
            label="S2 $\\rightarrow$ S3  (new cohort)")
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(yy); ax.set_yticklabels(piv.index, fontsize=6)
    ax.set_xlabel("AUROC drop")
    ax.legend(loc="lower right")
    ax.set_title("(b) cost of each boundary", loc="left")

    txt = "mean drop\nS1$\\rightarrow$S2  %.3f\nS2$\\rightarrow$S3  %.3f" % (
        d1.mean(), d2.mean())
    ok = d1.notna() & d2.notna()
    if ok.sum() >= 5:
        from scipy.stats import wilcoxon
        try:
            p = wilcoxon(d1[ok], d2[ok]).pvalue
            txt += "\nWilcoxon $p$ = %.3f\n($n$ = %d)" % (p, ok.sum())
        except Exception:
            pass
    ax.text(.98, .98, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=6, color="#333333", linespacing=1.4)

    fig.tight_layout()
    save(fig, "F2_architectures")
    print("  mean S1->S2 %.3f | mean S2->S3 %.3f" % (d1.mean(), d2.mean()))


# ---------------------------------------------------------------- F3
def fig3():
    print("F3 prevalence")
    d = need("prevalence_sweep.csv")
    if d is None:
        return
    d = d.dropna(subset=["actual_prev", "auroc", "auprc"])

    folds = [f for f in ["fold0", "chb->siena", "siena->chb"] if f in set(d.fold)]
    labels = {"fold0": "S2  cross-patient",
              "chb->siena": "S3  CHB-MIT $\\rightarrow$ Siena",
              "siena->chb": "S3  Siena $\\rightarrow$ CHB-MIT"}
    cols = [C["blue"], C["orange"], C["green"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))

    for k, f in enumerate(folds):
        s = d[d.fold == f].sort_values("actual_prev")
        axes[0].plot(s.actual_prev * 100, s.auroc, marker="o", ms=4,
                     color=cols[k], label=labels.get(f, f))
        axes[1].plot(s.actual_prev * 100, s.auprc, marker="o", ms=4,
                     color=cols[k], label=labels.get(f, f))
    xs = np.array(sorted(d.actual_prev.unique())) * 100
    axes[1].plot(xs, xs / 100, ls=":", lw=1.0, color=C["grey"],
                 label="chance (= prevalence)")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("evaluation prevalence (%)")
        ax.axvspan(20, 60, color=C["grey"], alpha=.12, lw=0, zorder=0)
        ax.axvline(0.34, color="#666666", ls="--", lw=.9, zorder=1)

    ax = axes[0]
    ax.axhline(0.5, color=C["grey"], ls=":", lw=.8)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("(a) AUROC is prevalence-invariant", loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.06))
    ax.annotate("native\n0.34%", xy=(0.34, 1.0), xytext=(0.34, 0.905),
                ha="center", fontsize=6, color="#444444",
                arrowprops=dict(arrowstyle="->", lw=.7, color="#666666"))
    ax.text(34, 0.99, "balanced\nevaluation", ha="center", va="top",
            fontsize=6, color="#444444")
    ax.text(0.115, 0.505, "chance", fontsize=6, color=C["grey"], va="bottom")

    ax = axes[1]
    ax.set_yscale("log")
    ax.set_ylabel("AUPRC")
    ax.set_title("(b) AUPRC is not", loc="left")
    ax.legend(loc="upper left")

    fig.tight_layout()
    save(fig, "F3_prevalence")

    for f in folds:
        s = d[d.fold == f]
        print("  %-14s AUROC range %.3f | AUPRC ratio %.0fx"
              % (f, s.auroc.max() - s.auroc.min(),
                 s.auprc.max() / max(s.auprc.min(), 1e-12)))


# ---------------------------------------------------------------- F4
def fig4():
    print("F4 per-subject")
    d = need("per_subject.csv")
    if d is None:
        return

    order = [o for o in ["bandpower", "aperiodic", "both"] if o in set(d.features)]
    nice = {"bandpower": "band power", "aperiodic": "aperiodic", "both": "both"}
    cols = {"bandpower": C["blue"], "aperiodic": C["orange"], "both": C["green"]}

    lo = max(0.0, d.auroc.min() - 0.06)
    hi = min(1.0, d.auroc.max() + 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3),
                             gridspec_kw=dict(width_ratios=[2.3, 1]))

    ax = axes[0]
    ref = d[d.features == order[-1]].set_index("subject").auroc
    subs = ref.sort_values().index.tolist()
    xx = np.arange(len(subs))
    for f in order:
        s = d[d.features == f].set_index("subject").auroc.reindex(subs)
        ax.plot(xx, s.values, marker="o", ms=3, lw=.9,
                color=cols[f], label=nice[f], alpha=.85)
    ax.axhline(0.5, color=C["red"], ls="--", lw=1.0)
    ax.text(len(subs) - .2, 0.5, " chance", fontsize=6, color=C["red"],
            va="center", ha="left")
    ax.axhspan(lo, 0.5, color=C["red"], alpha=.06, lw=0, zorder=0)
    ax.set_xticks(xx); ax.set_xticklabels(subs, rotation=90, fontsize=6)
    ax.set_ylabel("AUROC (held-out subject)")
    ax.set_ylim(lo, hi)
    ax.set_xlim(-.7, len(subs) + 1.6)
    ax.legend(loc="upper left", ncol=3)
    ax.set_title("(a) leave-one-subject-out, CHB-MIT", loc="left")

    ax = axes[1]
    data = [d[d.features == f].auroc.values for f in order]
    bp = ax.boxplot(data, widths=.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=1.2))
    for patch, f in zip(bp["boxes"], order):
        patch.set_facecolor(cols[f]); patch.set_alpha(.3)
        patch.set_edgecolor(cols[f])
    rng = np.random.default_rng(0)
    for i, f in enumerate(order):
        v = d[d.features == f].auroc.values
        ax.scatter(np.full(len(v), i + 1) + rng.uniform(-.09, .09, len(v)),
                   v, s=9, color=cols[f], alpha=.8, zorder=3, lw=0)
        # counts placed above the axis so they never touch a whisker
        ax.text(i + 1, 1.015, "%d/%d < 0.5" % ((v < .5).sum(), len(v)),
                ha="center", va="bottom", fontsize=6, color="#444444",
                transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color=C["red"], ls="--", lw=1.0)
    ax.axhspan(lo, 0.5, color=C["red"], alpha=.06, lw=0, zorder=0)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([nice[f] for f in order], rotation=15)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("AUROC")
    ax.set_title("(b) distribution", loc="left", pad=14)

    fig.tight_layout()
    save(fig, "F4_per_subject")

    for f in order:
        v = d[d.features == f].auroc
        print("  %-10s mean %.3f  sd %.3f  min %.3f  below 0.5: %d/%d"
              % (f, v.mean(), v.std(), v.min(), (v < .5).sum(), len(v)))


# ---------------------------------------------------------------- F5
def fig5():
    print("F5 controls")
    sh = need("label_shuffle.csv")
    sc = need("shortcut_test.csv")
    fair = need("fair_baseline.csv")
    if sh is None or sc is None:
        return

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    bars, vals, errs, colours = [], [], [], []

    if fair is not None:
        b = fair[fair.features == "both"]
        for setting, lab in [("S2", "actual labels\ncross-patient"),
                             ("S3", "actual labels\ncross-cohort")]:
            v = b[b.setting == setting].auroc
            bars.append(lab); vals.append(v.mean()); errs.append(v.std())
            colours.append(C["blue"])

    bars.append("shuffled\nlabels")
    vals.append(sh.auroc.mean()); errs.append(sh.auroc.std())
    colours.append(C["grey"])

    bars.append("recording context\nnon-seizure only")
    vals.append(sc.auroc.mean()); errs.append(sc.auroc.std())
    colours.append(C["grey"])

    xx = np.arange(len(bars))
    ax.bar(xx, vals, yerr=errs, capsize=3, color=colours, edgecolor="none",
           width=.62, error_kw=dict(lw=.9, ecolor="#444444"))
    for x, v, e in zip(xx, vals, errs):
        ax.text(x, v + e + .012, "%.3f" % v, ha="center", fontsize=6.5,
                color="#333333")
    ax.axhline(0.5, color=C["red"], ls="--", lw=1.0)
    ax.text(len(bars) - .45, 0.505, "chance", fontsize=6.5, color=C["red"],
            ha="right", va="bottom")
    ax.set_xticks(xx); ax.set_xticklabels(bars, fontsize=6.5)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.40, 0.92)
    ax.set_title("negative controls", loc="left")

    fig.tight_layout()
    save(fig, "F5_controls")
    print("  shuffled  %.3f (sd %.3f)" % (sh.auroc.mean(), sh.auroc.std()))
    print("  context   %.3f (sd %.3f)" % (sc.auroc.mean(), sc.auroc.std()))


CMDS = {"f2": fig2, "f3": fig3, "f4": fig4, "f5": fig5}

if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "all"
    order = [fig3, fig4, fig5, fig2] if c == "all" else [CMDS[c]]
    for fn in order:
        try:
            fn()
        except Exception as e:
            print("  FAILED: %s: %s" % (type(e).__name__, e))
            import traceback; traceback.print_exc()
    print("\nfigures in %s" % FIG)
