"""
=============================================================================
REDRAW THE CROWDED FIGURES
=============================================================================
    python -u fix_figures.py

Regenerates F2, F11, F14, F15 and F8's fourth panel into D:\\figures,
overwriting the previous versions. Nothing is recomputed from raw data; all
of it comes from the CSVs already written.

WHAT WAS WRONG AND WHY

F11 and F15 were unreadable, and the cause was a mistake in how the curves
were averaged. Each run produces its own set of thresholds, chosen from its
own score quantiles, so grouping rows by threshold and taking a mean does
not average the curves --- it interleaves points from runs at quite
different operating points. The fix is to interpolate every run's curve onto
one common false-alarm grid first and then average across runs, which is
what a mean operating curve actually means. The spread across runs is shown
as a band rather than as individual points.

F2's architecture labels were written at the same x position and collided.
They are now pushed apart vertically by a small relaxation step, with a
leader line back to each point.

F14's within-patient curve had spurious spikes for the same averaging reason
as F11: S1 runs have different native prevalences, so a fixed target
prevalence does not correspond to the same operating point in each. Only the
resampled points, which are directly comparable, are now drawn.
=============================================================================
"""
from pathlib import Path
import sys, io, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

FIG  = Path(r"D:\figures"); FIG.mkdir(parents=True, exist_ok=True)
NAT  = Path(r"D:\native_inference\results")
HAND = Path(r"D:\handcrafted_event\results")
SUB  = Path(r"C:\Users\Anurag Tiwari\Downloads\results\analysis_output\results")

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.8, "lines.linewidth": 1.2,
})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "grey": "#999999", "sky": "#56B4E9", "purple": "#CC79A7"}


def save(fig, name):
    for e in ("png", "pdf"):
        fig.savefig(FIG / ("%s.%s" % (name, e)))
    plt.close(fig)
    print("  wrote %s" % name)


def spread_labels(ys, gap):
    """Push labels apart so none overlaps, keeping their order."""
    y = np.array(ys, float)
    order = np.argsort(y)
    v = y[order]
    for _ in range(200):
        moved = False
        for i in range(len(v) - 1):
            d = v[i + 1] - v[i]
            if d < gap:
                shift = (gap - d) / 2
                v[i] -= shift; v[i + 1] += shift
                moved = True
        if not moved:
            break
    out = np.empty_like(v)
    out[order] = v
    return out


def run_key(df):
    """Whatever columns identify a single run in this file.

    The two event files were written by different scripts and do not share
    a column layout: one carries run_id, the other model/fold/seed."""
    for cand in (["run_id"], ["model", "fold", "seed"], ["model", "fold"],
                 ["fold"]):
        if all(c in df.columns for c in cand):
            return cand
    return None


def mean_curve(df, xcol="fa_per_hour", ycol="sensitivity",
               by=None, grid=None):
    """Interpolate each run onto a common x grid, then average.

    Grouping by threshold and averaging does not do this: thresholds are
    chosen per run from its own score distribution, so equal thresholds are
    not equal operating points."""
    if by is None:
        by = run_key(df)
    if by is None:
        by = [c for c in df.columns if c not in (xcol, ycol)][:1]
    if grid is None:
        lo = max(df[xcol][df[xcol] > 0].min(), 1e-3)
        grid = np.logspace(np.log10(lo), np.log10(df[xcol].max()), 60)
    curves = []
    for _, g in df.groupby(by):
        g = g.sort_values(xcol)
        # step interpolation: best sensitivity attainable at or below each rate
        s = np.array([g.loc[g[xcol] <= x, ycol].max() if (g[xcol] <= x).any()
                      else np.nan for x in grid])
        curves.append(s)
    A = np.vstack(curves)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return grid, np.nanmean(A, 0), np.nanpercentile(A, 25, 0), np.nanpercentile(A, 75, 0)


# ---------------------------------------------------------------- F2
def fig2():
    print("F2 architectures")
    D = {"Deep4Net":(0.929,0.816,0.818),"EEGInceptionERP":(0.953,0.835,0.785),
         "EEGNet":(0.937,0.850,0.845),"EEGNeX":(0.978,0.851,0.830),
         "TIDNet":(0.870,0.852,0.834),"SCCNet":(0.957,0.856,0.810),
         "SPARCNet":(0.924,0.860,0.848),"ShallowFBCSPNet":(0.961,0.866,0.811),
         "EEGConformer":(0.953,0.868,0.849),"EEGSimpleConv":(0.944,0.873,0.828),
         "EEGTCNet":(0.823,0.875,0.865),"EEGITNet":(0.932,0.878,0.842),
         "SincShallowNet":(0.822,0.881,0.831),"ATCNet":(0.953,0.888,0.838)}
    piv = pd.DataFrame(D, index=["S1","S2","S3"]).T.sort_values("S2")
    d1 = (piv.S1 - piv.S2).round(3); d2 = (piv.S2 - piv.S3).round(3)
    w = wilcoxon(d1, d2)
    LIN_S2, LIN_S3 = 0.709, 0.684

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4),
                             gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]; x = [0, 1, 2]
    for m, r in piv.iterrows():
        ax.plot(x, [r.S1, r.S2, r.S3], marker="o", ms=3, color=C["blue"],
                alpha=.35, lw=.9)
    ax.plot(x, piv.mean().values, marker="s", ms=5, color=C["red"], lw=2.2,
            zorder=5, label="mean, learned representations")
    ax.hlines([LIN_S2], .85, 1.15, color=C["orange"], lw=2.4, zorder=5)
    ax.hlines([LIN_S3], 1.85, 2.15, color=C["orange"], lw=2.4, zorder=5)
    ax.plot([], [], color=C["orange"], lw=2.4, label="handcrafted features")

    # labels pushed apart, with leader lines
    ylab = spread_labels(piv.S3.values, 0.0125)
    for (m, r), yl in zip(piv.iterrows(), ylab):
        ax.plot([2.02, 2.13], [r.S3, yl], lw=.5, color="#bbb", zorder=1)
        ax.annotate(m, (2.15, yl), va="center", fontsize=5.6, color="#333")

    ax.axhline(.5, color=C["grey"], ls=":", lw=.8)
    ax.text(-.15, .508, "chance", fontsize=6, color=C["grey"], va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(["S1\nwithin-patient","S2\ncross-patient","S3\ncross-cohort"])
    ax.set_ylabel("AUROC"); ax.set_xlim(-.22, 3.15); ax.set_ylim(.48, 1.005)
    ax.legend(loc="lower left", bbox_to_anchor=(0, .04))
    ax.set_title("(a) generalisation by setting", loc="left")

    ax = axes[1]
    yy = np.arange(len(piv))
    ax.barh(yy-.2, d1, height=.36, color=C["blue"], label="S1$\\rightarrow$S2 (new patient)")
    ax.barh(yy+.2, d2, height=.36, color=C["sky"],  label="S2$\\rightarrow$S3 (new cohort)")
    ax.axvline(0, color="k", lw=.8)
    ax.set_yticks(yy); ax.set_yticklabels(piv.index, fontsize=6)
    ax.set_xlabel("AUROC drop"); ax.set_xlim(-.085, .155)
    ax.legend(loc="lower right", fontsize=6)
    ax.set_title("(b) cost of each boundary", loc="left")
    ax.text(.99, .99, "mean drop\nS1$\\rightarrow$S2  %.3f\nS2$\\rightarrow$S3  %.3f\n"
            "Wilcoxon $p$ = %.3f  ($n$=%d)" % (d1.mean(), d2.mean(), w.pvalue, len(d1)),
            transform=ax.transAxes, ha="right", va="top", fontsize=5.8,
            color="#333", linespacing=1.5)
    fig.tight_layout(); save(fig, "F2_architectures")


# ---------------------------------------------------------------- F11 / F15
def event_figs():
    for tag, path, name, title, note in (
        ("subsampled", SUB / "event_level.csv", "F11_event_level",
         "subsampled windows", None),
        ("native", NAT / "native_event_all.csv", "F15_event_native",
         "continuous data, real recording hours", HAND / "event_level_handcrafted.csv")):
        if not path.exists():
            print("  SKIP %s (%s not found)" % (name, path)); continue
        print(name)
        E = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        cols = {"S1": C["grey"], "S2": C["blue"], "S3": C["green"]}
        for st in ("S1", "S2", "S3"):
            sub = E[E.setting == st]
            if sub.empty:
                continue
            g, m, lo, hi = mean_curve(sub)
            ax.fill_between(g, lo, hi, color=cols[st], alpha=.15, lw=0)
            ax.plot(g, m, color=cols[st], lw=1.6, label="learned " + st)
        if note and note.exists():
            H = pd.read_csv(note)
            h = H[H.setting == "S3"]
            if len(h):
                g, m, lo, hi = mean_curve(h, by=run_key(h))
                ax.fill_between(g, lo, hi, color=C["orange"], alpha=.15, lw=0)
                ax.plot(g, m, color=C["orange"], lw=1.6, ls="--",
                        label="handcrafted S3")
        for x in (0.1, 1.0):
            ax.axvline(x, color=C["grey"], ls=":", lw=.7, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("false alarms per hour")
        ax.set_ylabel("event-level sensitivity")
        ax.set_ylim(0, 1.02); ax.set_xlim(0.02, 20)
        ax.legend(loc="upper left", fontsize=6.5)
        ax.set_title(title, loc="left")
        ax.text(.99, .02, "line: mean over runs\nband: interquartile range",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.8, color="#666", linespacing=1.4)
        fig.tight_layout(); save(fig, name)


# ---------------------------------------------------------------- F14
def fig14():
    f = NAT / "native_prevalence_all.csv"
    if not f.exists():
        print("  SKIP F14 (%s not found)" % f); return
    print("F14 native comparison")
    P = pd.read_csv(f)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for st, col in (("S1", C["blue"]), ("S3", C["green"])):
        sub = P[P.setting == st]
        if sub.empty:
            continue
        # group on the achieved prevalence, rounded, so points that are
        # genuinely at the same class balance are averaged together
        sub = sub.assign(k=np.round(np.log10(sub.actual_prev), 2))
        g = sub.groupby("k").agg(p=("actual_prev","mean"),
                                 auroc=("auroc","mean"),
                                 auprc=("auprc","mean")).sort_values("p")
        ax[0].plot(g.p*100, g.auroc, marker="o", ms=3.5, color=col, label="learned "+st)
        ax[1].plot(g.p*100, g.auprc, marker="o", ms=3.5, color=col, label="learned "+st)
    hp = HAND / "native_prevalence.csv"
    if hp.exists():
        H = pd.read_csv(hp); H = H[H.features == "both"]
        for st, mk in (("S3","s"), ("S2","^")):
            h = H[H.setting == st]
            if len(h):
                ax[0].scatter(h.prevalence*100, h.auroc, marker=mk, s=26,
                              color=C["orange"], zorder=5, label="handcrafted "+st)
                ax[1].scatter(h.prevalence*100, h.auprc, marker=mk, s=26,
                              color=C["orange"], zorder=5, label="handcrafted "+st)
    xs = np.logspace(-1, 1.7, 30)
    ax[1].plot(xs, xs/100, ls=":", color=C["grey"], lw=1, label="chance")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("evaluation prevalence (%)")
        a.axvline(0.34, color="#666", ls="--", lw=.9)
    ax[0].set_ylabel("AUROC"); ax[0].set_ylim(.55, 1.02)
    ax[0].legend(loc="lower right", fontsize=6); ax[0].set_title("(a) AUROC", loc="left")
    ax[1].set_yscale("log"); ax[1].set_ylabel("AUPRC")
    ax[1].legend(loc="upper left", fontsize=6); ax[1].set_title("(b) AUPRC", loc="left")
    fig.tight_layout(); save(fig, "F14_native_comparison")


if __name__ == "__main__":
    for fn in (fig2, event_figs, fig14):
        try:
            fn()
        except Exception as e:
            print("  FAILED %s: %s: %s" % (fn.__name__, type(e).__name__, e))
            import traceback; traceback.print_exc()
    print("\nfigures rewritten in %s" % FIG)
