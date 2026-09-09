"""
=============================================================================
CONCEPT AND METHOD FIGURES  (F1, F6-F9)
=============================================================================
    python -u make_figures_concept.py all

    python -u make_figures_concept.py f1   # pipeline schematic (drawn)
    python -u make_figures_concept.py f6   # spectral evidence (data)
    python -u make_figures_concept.py f7   # the two boundaries (drawn)
    python -u make_figures_concept.py f8   # embedding structure (data)
    python -u make_figures_concept.py f9   # window labelling rule (drawn)

Changes in this revision

  F8 (a) The subject-identity probe was wrong and returned nothing. It used
         GroupKFold with the subject as the grouping variable while the
         subject was also the label, so every held-out fold contained a
         single class and the AUROC was undefined. Grouping by subject is
         incoherent when subject identity IS the target: the correct probe
         is a stratified split of windows. Fixed.

     (b) A third comparison is added: seizure/non-seizure separability
         measured the same way. Cohort and subject separability mean little
         in isolation; what matters is how they compare with the signal the
         model is meant to use.

     (c) The silhouette annotation is placed in axis coordinates, so it no
         longer collides with the chance line.

  The interpretation is left open on purpose. Cohort identity being
  recoverable does not by itself mean cohort shift costs performance --
  a model can ignore structure that exists. The numbers printed at the end
  are what the discussion should be written from.

Writes 300-dpi PNG and vector PDF to E:\\figures\\.
=============================================================================
"""
from pathlib import Path
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle

CHB   = Path(r"E:\CPU-Intel 7\C_Drive\Compressed\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0")
SIENA = Path(r"E:\CPU-Intel 7\C_Drive\archive\siena-scalp-eeg-database-1.0.0")
FEAT  = Path(r"E:\xcohort")
RES   = Path(r"E:\cpu_results"); RES.mkdir(parents=True, exist_ok=True)
FIG   = Path(r"E:\figures"); FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.8, "lines.linewidth": 1.2,
})

C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
     "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999",
     "sky": "#56B4E9", "light": "#EFEFEF"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("%s.%s" % (name, ext)))
    plt.close(fig)
    print("  wrote %s.png / .pdf" % name)


def box(ax, x, y, w, h, head, body, fc="white", ec="#444444",
        fs_head=7.5, fs_body=7, lw=.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.010,rounding_size=0.018",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if head:
        ax.text(x + w / 2, y + h - 0.035, head, ha="center", va="top",
                fontsize=fs_head, fontweight="bold", zorder=3, color="#222222")
        ax.text(x + w / 2, y + (h - 0.055) / 2, body, ha="center", va="center",
                fontsize=fs_body, zorder=3, linespacing=1.4)
    else:
        ax.text(x + w / 2, y + h / 2, body, ha="center", va="center",
                fontsize=fs_body, zorder=3, linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, style="-|>", color="#444444", lw=.9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=8, color=color, lw=lw,
                                 zorder=1, shrinkA=1, shrinkB=1))


# ---------------------------------------------------------------- F1
def fig1():
    print("F1 pipeline")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, .01, .66, .185, .26, "CHB-MIT",
        "23 paediatric subjects\n256 Hz  ·  bipolar\n60 Hz notch applied",
        fc="#E8F1F8", ec=C["blue"], fs_body=6.6)
    box(ax, .01, .30, .185, .26, "Siena",
        "14 adult subjects\n512 Hz  ·  referential\n50 Hz line noise",
        fc="#FDF2E3", ec=C["orange"], fs_body=6.6)

    box(ax, .235, .40, .165, .48, "harmonisation",
        "17 bipolar\nderivations\n\nband-pass 1–40 Hz\n\nresample 256 Hz\n\n"
        "per-recording\nz-score",
        fc=C["light"], ec="#666666", fs_body=6.6)
    arrow(ax, .196, .79, .235, .74)
    arrow(ax, .196, .43, .235, .54)

    box(ax, .445, .52, .155, .36, "windowing",
        "4 s, no overlap\n\n≥50% overlap\n→ positive\n0% → negative\n"
        "else discarded",
        fc=C["light"], ec="#666666", fs_body=6.6)
    arrow(ax, .401, .70, .445, .70)

    box(ax, .445, .27, .155, .18, None,
        "1 000 311 windows\n3 405 positive (0.34%)\n38 subjects",
        fc="white", ec="#999999", fs_body=6.2)
    arrow(ax, .5225, .52, .5225, .45, style="-")

    ys = [.775, .545, .315]
    heads = ["S1   within-patient", "S2   cross-patient", "S3   cross-cohort"]
    bodies = ["time split inside one subject",
              "grouped 5-fold over subjects",
              "CHB-MIT ↔ Siena, both directions"]
    cols = [C["grey"], C["blue"], C["green"]]
    for y, h, b, c in zip(ys, heads, bodies, cols):
        box(ax, .655, y - .085, .335, .17, h, b, ec=c,
            fs_head=7.2, fs_body=6.6)
        arrow(ax, .601, .70, .655, y, color=c)

    ax.text(.822, .10, "identical windows, splits and preprocessing "
                       "for every model",
            ha="center", fontsize=6.4, color="#555555", style="italic")

    save(fig, "F1_pipeline")


# ---------------------------------------------------------------- F6
def fig6():
    print("F6 spectra")
    try:
        import mne
        from neurodsp.spectral import compute_spectrum
    except ImportError:
        print("  SKIP: needs mne + neurodsp")
        return

    def psd_of(path, a, b=None, secs=300):
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        raw.rename_channels({c: c.replace("EEG ", "").strip() for c in raw.ch_names})
        up = {c.upper(): c for c in raw.ch_names}
        fs = raw.info["sfreq"]; n = int(secs * fs)
        if b is None:
            x = raw.get_data(picks=[up[a]])[0][:n]
        else:
            x = (raw.get_data(picks=[up[a]])[0][:n] -
                 raw.get_data(picks=[up[b]])[0][:n])
        return compute_spectrum(x, fs, method="welch",
                                nperseg=int(fs), noverlap=int(fs // 2))

    chb_f = sorted(CHB.glob("chb01/chb01_*.edf"))
    sie_f = sorted(SIENA.rglob("PN*/*.edf"))
    if not chb_f or not sie_f:
        print("  SKIP: source EDFs not found")
        return

    fr1, ps1 = psd_of(chb_f[0], "CZ-PZ")
    fr2, ps2 = psd_of(sie_f[0], "CZ", "PZ")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, fr, ps, name, col, mains in [
            (axes[0], fr1, ps1, "(a) CHB-MIT  (chb01, CZ-PZ)", C["blue"], 60),
            (axes[1], fr2, ps2, "(b) Siena  (PN00, CZ-PZ derived)", C["orange"], 50)]:
        m = (fr > 0.5) & (fr <= 120)
        ax.loglog(fr[m], ps[m], color=col, lw=1.0, zorder=3)
        ax.axvspan(1, 40, color=C["green"], alpha=.10, lw=0, zorder=0)
        ax.axvline(mains, color=C["red"], ls="--", lw=.9, zorder=1)
        ax.set_xlabel("frequency (Hz)")
        ax.set_title(name, loc="left")
        ax.set_xticks([1, 10, 40, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(0.9, 130)
        ax.text(0.03, 0.90, "analysed band\n1–40 Hz", transform=ax.transAxes,
                fontsize=6.2, color="#1B7A5A", ha="left", va="top")
        ax.text(mains, ax.get_ylim()[1] * 0.9, "%d Hz" % mains, fontsize=6,
                color=C["red"], ha="center", va="top")

    axes[0].set_ylabel("PSD (V$^2$/Hz)")

    i60 = np.argmin(abs(fr1 - 60)); i70 = np.argmin(abs(fr1 - 70))
    axes[0].annotate("notch already applied:\npower dips at 60 Hz and\n"
                     "rises again by 70 Hz",
                     xy=(60, ps1[i60]), xytext=(1.6, ps1[i60] * 0.06),
                     fontsize=6, color="#333333",
                     arrowprops=dict(arrowstyle="->", lw=.7, color="#555555"))
    i50 = np.argmin(abs(fr2 - 50))
    axes[1].annotate("un-notched 50 Hz\nline noise, with\nharmonic at 100 Hz",
                     xy=(50, ps2[i50]), xytext=(2.0, ps2[i50] * 0.05),
                     fontsize=6, color="#333333",
                     arrowprops=dict(arrowstyle="->", lw=.7, color="#555555"))

    fig.tight_layout()
    save(fig, "F6_spectra")
    print("  CHB   60 Hz %.2e vs 70 Hz %.2e   (dip = notch already applied)"
          % (ps1[i60], ps1[i70]))
    print("  Siena 50 Hz %.2e vs 48 Hz %.2e   (peak = line noise)"
          % (ps2[i50], ps2[np.argmin(abs(fr2 - 48))]))


# ---------------------------------------------------------------- F7
def fig7():
    print("F7 boundaries")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))

    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((.03, .16), .43, .70, fc="#E8F1F8",
                           ec=C["blue"], lw=1.0, zorder=1))
    ax.add_patch(Rectangle((.54, .16), .43, .70, fc="#FDF2E3",
                           ec=C["orange"], lw=1.0, zorder=1))
    ax.text(.245, .90, "cohort A", ha="center", fontsize=7.5, color=C["blue"])
    ax.text(.755, .90, "cohort B", ha="center", fontsize=7.5, color=C["orange"])

    rng = np.random.default_rng(3)
    for cx, col in [(.245, C["blue"]), (.755, C["orange"])]:
        for k in range(4):
            ex = cx + (k % 2 - .5) * .205
            ey = .37 + (k // 2) * .28
            ax.add_patch(Ellipse((ex, ey), .165, .185, fc="white",
                                 ec=col, lw=.8, alpha=.95, zorder=2))
            p = rng.normal([ex, ey], [.031, .034], size=(14, 2))
            ax.scatter(p[:, 0], p[:, 1], s=2.5, color=col, alpha=.75, zorder=3)
        ax.text(cx, .215, "each ellipse = one patient", ha="center",
                fontsize=6, color="#555555")

    arrow(ax, .335, .65, .155, .65, style="<|-|>", color=C["blue"], lw=1.1)
    ax.text(.245, .695, "patient shift", ha="center", fontsize=6.5,
            color=C["blue"])
    arrow(ax, .47, .51, .54, .51, style="<|-|>", color="#444444", lw=1.1)
    ax.text(.505, .085, "cohort shift", ha="center", fontsize=6.5,
            color="#444444")
    ax.set_title("(a) two boundaries a model can cross", loc="left")

    ax = axes[1]
    x = [0, 1, 2]
    ax.plot(x, [.94, .70, .45], marker="o", ms=4, color=C["grey"],
            ls="--", label="assumed")
    ax.plot(x, [.94, .70, .68], marker="o", ms=4, color=C["red"],
            label="measured here")
    ax.axhline(.5, color=C["grey"], ls=":", lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["within-\npatient", "cross-\npatient", "cross-\ncohort"])
    ax.set_ylabel("performance")
    ax.set_ylim(.35, 1.02)
    ax.set_yticks([])
    ax.set_xlim(-.25, 2.35)
    ax.legend(loc="lower left")
    ax.annotate("", xy=(2.16, .68), xytext=(2.16, .45),
                arrowprops=dict(arrowstyle="<->", lw=.9, color=C["red"]))
    ax.text(2.10, .565, "the gap\nin question", ha="right", fontsize=6.5,
            color=C["red"], va="center")
    ax.set_title("(b) the hypothesis under test", loc="left")

    fig.tight_layout()
    save(fig, "F7_boundaries")


# ---------------------------------------------------------------- F8
def fig8():
    """How much structure does each grouping carry, and how does that compare
    with the seizure signal itself?"""
    print("F8 embedding")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import silhouette_score

    Xs, ys, cs, ss = [], [], [], []
    for f in sorted(FEAT.glob("*.npz")):
        d = np.load(f)
        if d["X"].size == 0:
            continue
        cohort, subj, _ = f.stem.split("__", 2)
        n = len(d["y"])
        Xs.append(d["X"]); ys.append(d["y"])
        cs += [cohort] * n; ss += [subj] * n
    if not Xs:
        print("  SKIP: no feature .npz in %s" % FEAT)
        return
    X = np.vstack(Xs); y = np.concatenate(ys)
    coh = np.array(cs); sub = np.array(ss)
    ok = np.isfinite(X).all(1)
    X, y, coh, sub = X[ok], y[ok], coh[ok], sub[ok]

    # balanced subsample per subject, plus every positive we can keep
    rng = np.random.default_rng(0)
    idx = []
    for s in np.unique(sub):
        m = np.where(sub == s)[0]
        idx.append(rng.choice(m, min(400, len(m)), replace=False))
    idx = np.concatenate(idx)
    pos = np.where(y == 1)[0]
    idx = np.unique(np.concatenate([idx, rng.choice(pos, min(600, len(pos)),
                                                    replace=False)]))
    Xp, cohp, subp, yp = X[idx], coh[idx], sub[idx], y[idx]

    Xz = StandardScaler().fit_transform(Xp)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xz)
    ev = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.6),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, .95]))

    ax = axes[0]
    for c, col, lab in [("chb", C["blue"], "CHB-MIT"),
                        ("siena", C["orange"], "Siena")]:
        m = cohp == c
        ax.scatter(Z[m, 0], Z[m, 1], s=2, color=col, alpha=.35, lw=0, label=lab)
    ax.legend(markerscale=4, loc="upper right")
    ax.set_title("(a) by cohort", loc="left")

    ax = axes[1]
    subs = np.unique(subp)
    cmap = plt.get_cmap("tab20")
    for i, s in enumerate(subs):
        m = subp == s
        ax.scatter(Z[m, 0], Z[m, 1], s=2, color=cmap(i % 20), alpha=.45, lw=0)
    ax.set_title("(b) by subject (%d)" % len(subs), loc="left")

    ax = axes[2]
    m0, m1 = yp == 0, yp == 1
    ax.scatter(Z[m0, 0], Z[m0, 1], s=2, color=C["grey"], alpha=.25, lw=0,
               label="non-seizure")
    ax.scatter(Z[m1, 0], Z[m1, 1], s=5, color=C["red"], alpha=.75, lw=0,
               label="seizure")
    ax.legend(markerscale=3, loc="upper right")
    ax.set_title("(c) by label", loc="left")

    for ax in axes[:3]:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("PC1 (%.0f%%)" % ev[0])
    axes[0].set_ylabel("PC2 (%.0f%%)" % ev[1])

    # ---------------- panel (d): separability of each grouping
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    skf = StratifiedKFold(5, shuffle=True, random_state=0)

    coh_i = pd.factorize(cohp)[0]
    auc_c = cross_val_score(lr, Xp, coh_i, cv=skf, scoring="roc_auc").mean()

    # subject identity: one-vs-rest over windows. Grouping by subject here
    # would be incoherent, since the subject IS the label.
    aucs_s = []
    for s in subs:
        lab = (subp == s).astype(int)
        if lab.sum() < 50:
            continue
        try:
            aucs_s.append(cross_val_score(lr, Xp, lab, cv=skf,
                                          scoring="roc_auc").mean())
        except Exception:
            pass
    auc_s = float(np.mean(aucs_s)) if aucs_s else np.nan

    # the signal the model is actually meant to use, measured the same way
    auc_y = np.nan
    if len(np.unique(yp)) > 1 and yp.sum() >= 50:
        auc_y = cross_val_score(lr, Xp, yp, cv=skf, scoring="roc_auc").mean()

    sub_i = pd.factorize(subp)[0]
    sil_c = silhouette_score(Z, coh_i, sample_size=6000, random_state=0)
    sil_s = silhouette_score(Z, sub_i, sample_size=6000, random_state=0)

    ax = axes[3]
    labs = ["cohort", "subject", "seizure"]
    vals = [auc_c, auc_s, auc_y]
    colsb = [C["purple"], C["green"], C["red"]]
    ax.bar(range(3), vals, color=colsb, width=.6, edgecolor="none")
    ax.axhline(0.5, color=C["grey"], ls=":", lw=.9)
    for x, v in zip(range(3), vals):
        if np.isfinite(v):
            ax.text(x, v + .015, "%.2f" % v, ha="center", fontsize=6.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("recoverable from features\n(AUROC)")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("(d) what is encoded", loc="left")
    ax.text(0.02, 0.02, "silhouette (PC1–2)\ncohort %+.3f\nsubject %+.3f"
            % (sil_c, sil_s), transform=ax.transAxes, ha="left", va="bottom",
            fontsize=5.8, color="#444444", linespacing=1.4)

    fig.tight_layout()
    save(fig, "F8_embedding")

    pd.DataFrame([dict(auc_cohort=auc_c, auc_subject=auc_s, auc_seizure=auc_y,
                       n_subject_probes=len(aucs_s),
                       sil_cohort=sil_c, sil_subject=sil_s,
                       pc1=ev[0], pc2=ev[1], n_windows=len(idx))]
                 ).to_csv(RES / "embedding_separability.csv", index=False)

    print("  PC1 %.1f%%  PC2 %.1f%%" % (ev[0], ev[1]))
    print("  recoverable  cohort %.3f | subject %.3f (mean of %d one-vs-rest)"
          " | seizure %.3f" % (auc_c, auc_s, len(aucs_s), auc_y))
    print("  silhouette   cohort %+.3f | subject %+.3f" % (sil_c, sil_s))
    print()
    print("  Read these together, not separately:")
    print("   - if cohort >> subject, the cohort boundary is a real and")
    print("     distinct source of structure, and S2 ~ S3 means the models")
    print("     are ignoring structure that is present")
    print("   - if cohort ~ subject, the two boundaries are alike and S2 ~ S3")
    print("     follows directly")
    print("   - compare both against the seizure column: structure that is")
    print("     more recoverable than the label itself is worth reporting")
    print("  wrote", RES / "embedding_separability.csv")


# ---------------------------------------------------------------- F9
def fig9():
    print("F9 labelling")
    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    ax.set_xlim(-1, 41); ax.set_ylim(-.55, 1.35); ax.axis("off")

    sz0, sz1 = 14.5, 27.0
    ax.add_patch(Rectangle((sz0, .28), sz1 - sz0, .46, fc="#F6D6C6",
                           ec=C["red"], lw=1.0, zorder=1))
    ax.text((sz0 + sz1) / 2, .82, "annotated seizure", ha="center",
            fontsize=7, color=C["red"])

    for i in range(10):
        a, b = i * 4, (i + 1) * 4
        ov = max(0, min(b, sz1) - max(a, sz0)) / 4.0
        if ov >= .5:
            fc, ec, lab = "#F2A07B", C["red"], "1"
        elif ov == 0:
            fc, ec, lab = "#DCE7F0", C["blue"], "0"
        else:
            fc, ec, lab = "white", "#999999", "–"
        ax.add_patch(Rectangle((a + .12, .32), 3.76, .38, fc=fc, ec=ec,
                               lw=.9, zorder=3,
                               hatch="///" if lab == "–" else None))
        ax.text(a + 2, .51, lab, ha="center", va="center", fontsize=7.5,
                zorder=4, color="#222222")
        ax.text(a + 2, .19, "%.0f%%" % (100 * ov), ha="center", fontsize=5.5,
                color="#666666")

    ax.annotate("", xy=(0, .05), xytext=(40, .05),
                arrowprops=dict(arrowstyle="-", lw=.8, color="#666666"))
    for t in range(0, 41, 8):
        ax.text(t, -.10, "%d s" % t, ha="center", fontsize=6, color="#666666")

    leg = [plt.Line2D([0], [0], marker="s", ls="", ms=6, mfc="#F2A07B",
                      mec=C["red"], label="positive  (≥50% overlap)"),
           plt.Line2D([0], [0], marker="s", ls="", ms=6, mfc="#DCE7F0",
                      mec=C["blue"], label="negative  (0% overlap)"),
           plt.Line2D([0], [0], marker="s", ls="", ms=6, mfc="white",
                      mec="#999999", label="discarded  (partial overlap)")]
    ax.legend(handles=leg, loc="upper center", ncol=3, fontsize=6.5,
              bbox_to_anchor=(0.5, 1.22))
    ax.text(0, -.38, "4 s windows, no overlap between windows",
            fontsize=6.5, color="#555555", style="italic")

    save(fig, "F9_labelling")


CMDS = {"f1": fig1, "f6": fig6, "f7": fig7, "f8": fig8, "f9": fig9}

if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "all"
    order = [fig1, fig9, fig7, fig6, fig8] if c == "all" else [CMDS[c]]
    for fn in order:
        try:
            fn()
        except Exception as e:
            print("  FAILED: %s: %s" % (type(e).__name__, e))
            import traceback; traceback.print_exc()
    print("\nfigures in %s" % FIG)
