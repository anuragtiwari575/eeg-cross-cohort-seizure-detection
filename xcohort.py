"""
Cross-cohort seizure detection: CHB-MIT vs Siena.

    python -u xcohort.py probe    # Siena spectrum + montage check  (~5 min)
    python -u xcohort.py prep     # windows + features, both cohorts (~1-3 h)
    python -u xcohort.py exp      # S1/S2/S3 experiments            (~10 min)
    python -u xcohort.py probe2   # identity probing / mechanism    (~5 min)

Requires: mne, neurodsp, specparam, scikit-learn, pandas, matplotlib
"""
from pathlib import Path
import os, re, sys, time, warnings
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

# ---------------------------------------------------------------- paths
CHB   = Path(r"E:\CPU-Intel 7\C_Drive\Compressed\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0")
SIENA = Path(r"E:\CPU-Intel 7\C_Drive\archive\siena-scalp-eeg-database-1.0.0")
OUT   = Path(r"E:\xcohort"); OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- settings
FS, WIN_S, BAND = 256, 4.0, (1., 40.)
N_WORKERS = 12
MAX_SZ_DUR = 3600        # seizure entries longer than this are treated as errors

# CHB-MIT bipolar names -> (anode, cathode) in Siena's older nomenclature
BIPOLAR = [("FP1-F7", ("Fp1", "F7")), ("F7-T7", ("F7", "T3")), ("T7-P7", ("T3", "T5")),
           ("P7-O1", ("T5", "O1")),   ("FP1-F3", ("Fp1", "F3")), ("F3-C3", ("F3", "C3")),
           ("C3-P3", ("C3", "P3")),   ("P3-O1", ("P3", "O1")),   ("FP2-F4", ("Fp2", "F4")),
           ("F4-C4", ("F4", "C4")),   ("C4-P4", ("C4", "P4")),   ("P4-O2", ("P4", "O2")),
           ("FP2-F8", ("Fp2", "F8")), ("F8-T8", ("F8", "T4")),   ("P8-O2", ("T6", "O2")),
           ("FZ-CZ", ("Fz", "Cz")),   ("CZ-PZ", ("Cz", "Pz"))]
CHNAMES = [b[0] for b in BIPOLAR]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 40)}


# ---------------------------------------------------------------- annotations
def _secs(t):
    """First HH.MM.SS (or HH:MM:SS) in the string -> seconds. None if absent.

    Siena annotations sometimes carry extra text, e.g.
        '11.41.04 opure 11.40.43'
        '15.43.53 (CLINICAL ONSET); 15.43.59 (ELECTRIC ONSET)'
    Taking the first match keeps the clinical onset.
    """
    m = re.search(r"(\d{1,2})[.:](\d{2})[.:](\d{2})", t)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def chb_seizures():
    """{(subject, file): [(start_s, end_s), ...]}"""
    out = {}
    for s in sorted(CHB.glob("chb*/chb*-summary.txt")):
        for b in re.split(r"\n(?=File Name:)", s.read_text(errors="ignore")):
            n = re.search(r"File Name:\s*(\S+)", b)
            if not n:
                continue
            st = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)", b)]
            en = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)", b)]
            out[(s.parent.name, n.group(1))] = list(zip(st, en))
    return out


def siena_seizures(verbose=False):
    """{(subject, file): [(start_s, end_s), ...]} relative to registration start."""
    out, n_ok, n_bad = {}, 0, 0
    for f in sorted(SIENA.rglob("Seizures-list-*.txt")):
        subj = f.stem.split("-")[-1]
        txt = f.read_text(errors="ignore")
        for blk in re.split(r"\n(?=Seizure n)", txt):
            fn = re.search(r"File name:\s*(\S+)", blk)
            rs = re.search(r"Registration start time:\s*(.*)", blk)
            ss = re.search(r"Seizure start time:\s*(.*)", blk)
            se = re.search(r"Seizure end time:\s*(.*)", blk)
            if not (fn and rs and ss and se):
                continue
            r0, t1, t2 = _secs(rs.group(1)), _secs(ss.group(1)), _secs(se.group(1))
            if None in (r0, t1, t2):
                n_bad += 1
                if verbose:
                    print("   unparseable time in %s / %s" % (f.name, fn.group(1)))
                continue
            a = (t1 - r0) % 86400
            b = (t2 - r0) % 86400
            if b <= a or (b - a) > MAX_SZ_DUR:
                n_bad += 1
                if verbose:
                    print("   implausible duration %ds in %s / %s"
                          % (b - a, f.name, fn.group(1)))
                continue
            out.setdefault((subj, fn.group(1)), []).append((a, b))
            n_ok += 1
    if verbose:
        print("   parsed %d seizure entries, skipped %d" % (n_ok, n_bad))
    return out


# ---------------------------------------------------------------- loading
def load_bipolar(path, cohort):
    """-> (data [17 x n], fs) resampled to FS, bandpassed. (None, None) if channels missing."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.rename_channels({c: c.replace("EEG ", "").strip() for c in raw.ch_names})
    up = {c.upper(): c for c in raw.ch_names}

    if cohort == "chb":
        if any(n not in up for n in CHNAMES):
            return None, None
        raw.pick([up[n] for n in CHNAMES])
        raw.reorder_channels([up[n] for n in CHNAMES])
        data = raw.get_data()
    else:
        rows = []
        for name, (a, b) in BIPOLAR:
            if a.upper() not in up or b.upper() not in up:
                return None, None
            rows.append(raw.get_data(picks=[up[a.upper()]])[0] -
                        raw.get_data(picks=[up[b.upper()]])[0])
        data = np.array(rows)

    info = mne.create_info(CHNAMES, raw.info["sfreq"], "eeg", verbose="ERROR")
    r2 = mne.io.RawArray(data, info, verbose="ERROR")
    r2.filter(*BAND, verbose="ERROR")
    if abs(r2.info["sfreq"] - FS) > 1:
        r2.resample(FS, verbose="ERROR")
    return r2.get_data(), FS


def feats(win, fs):
    """bandpower (5 x 17) + aperiodic exponent & r2 (2 x 17) -> 1D vector."""
    from neurodsp.spectral import compute_spectrum
    from specparam import SpectralModel
    bp, ap = [], []
    for ch in win:
        fr, ps = compute_spectrum(ch, fs, method="welch",
                                  nperseg=int(fs), noverlap=int(fs // 2))
        tot = ps[(fr >= 1) & (fr <= 40)].sum() + 1e-20
        for lo, hi in BANDS.values():
            bp.append(ps[(fr >= lo) & (fr < hi)].sum() / tot)
        sm = SpectralModel(aperiodic_mode="fixed", peak_width_limits=(0.5, 12),
                           max_n_peaks=6, min_peak_height=0.15, verbose=False)
        try:
            sm.fit(fr, ps, [1, 40])
            r2 = float(sm.results.metrics.results["gof_rsquared"])
            a = np.atleast_1d(np.asarray(sm.get_params("aperiodic"), float)).ravel()
            ap += [float(a[-1]), r2]
        except Exception:
            ap += [np.nan, np.nan]
    return np.array(bp + ap, dtype=np.float32)


def process_file(job):
    cohort, subj, fname, path, sz = job
    tag = OUT / ("%s__%s__%s.npz" % (cohort, subj, Path(fname).stem))
    if tag.exists():
        return "skip " + fname
    try:
        data, fs = load_bipolar(path, cohort)
        if data is None:
            np.savez_compressed(tag, X=np.zeros((0,)), y=np.zeros((0,)))
            return "nochan " + fname
        w = int(WIN_S * fs)
        nw = data.shape[1] // w
        X, y = [], []
        for i in range(nw):
            t0, t1 = i * WIN_S, (i + 1) * WIN_S
            ov = sum(max(0, min(t1, b) - max(t0, a)) for a, b in sz) / WIN_S
            if 0 < ov < 0.5:
                continue                      # drop ambiguous boundary windows
            X.append(feats(data[:, i * w:(i + 1) * w], fs))
            y.append(1 if ov >= 0.5 else 0)
        X = np.array(X, np.float32)
        y = np.array(y, np.int8)
        np.savez_compressed(tag, X=X, y=y)
        return "ok %s n=%d pos=%d" % (fname, len(y), int(y.sum()))
    except Exception as e:
        return "FAIL %s: %s: %s" % (fname, type(e).__name__, e)


def build_jobs():
    jobs = []
    for (subj, fn), sz in chb_seizures().items():
        p = CHB / subj / fn
        if p.exists():
            jobs.append(("chb", subj, fn, p, sz))
    ss = siena_seizures()
    for f in sorted(SIENA.rglob("PN*/*.edf")):
        jobs.append(("siena", f.parent.name, f.name, f,
                     ss.get((f.parent.name, f.name), [])))
    return jobs


def load_all():
    rows = []
    for f in sorted(OUT.glob("*.npz")):
        d = np.load(f)
        if d["X"].size == 0:
            continue
        cohort, subj, _ = f.stem.split("__", 2)
        rows.append((cohort, subj, d["X"], d["y"]))
    if not rows:
        sys.exit("no data - run 'prep' first")
    X = np.vstack([r[2] for r in rows])
    y = np.concatenate([r[3] for r in rows])
    coh = np.concatenate([[r[0]] * len(r[3]) for r in rows])
    sub = np.concatenate([[r[1]] * len(r[3]) for r in rows])
    ok = np.isfinite(X).all(1)
    return X[ok], y[ok], coh[ok], sub[ok]


# ---------------------------------------------------------------- commands
def cmd_probe():
    import mne, matplotlib.pyplot as plt
    from neurodsp.spectral import compute_spectrum

    files = sorted(SIENA.rglob("PN*/*.edf"))
    f = files[0]
    raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
    raw.rename_channels({c: c.replace("EEG ", "").strip() for c in raw.ch_names})
    fs = raw.info["sfreq"]
    print("file:", f.name, "| sfreq", fs)
    print("channels:", raw.ch_names)

    up = {c.upper(): c for c in raw.ch_names}
    missing = [(n, a, b) for n, (a, b) in BIPOLAR
               if a.upper() not in up or b.upper() not in up]
    print("\nmissing bipolar pairs in this file:",
          missing if missing else "none - all 17 derivable")

    print("\nchecking all %d Siena files for the 17 pairs:" % len(files))
    bad = []
    for g in files:
        r = mne.io.read_raw_edf(g, preload=False, verbose="ERROR")
        u = {c.replace("EEG ", "").strip().upper() for c in r.ch_names}
        miss = [n for n, (a, b) in BIPOLAR if a.upper() not in u or b.upper() not in u]
        if miss:
            bad.append((g.name, miss))
    print("  %d/%d files complete" % (len(files) - len(bad), len(files)))
    for n, m in bad[:10]:
        print("   ", n, "missing", m)

    x = raw.get_data(picks=[up["CZ"]])[0] - raw.get_data(picks=[up["PZ"]])[0]
    fr, ps = compute_spectrum(x[:int(300 * fs)], fs, method="welch",
                              nperseg=int(fs), noverlap=int(fs // 2))
    print("\nPSD (looking for notch / line noise):")
    for f0 in [10, 20, 30, 40, 45, 48, 50, 52, 55, 60, 70, 80, 100]:
        i = np.argmin(abs(fr - f0))
        print("  %4.0f Hz  %.3e" % (fr[i], ps[i]))

    print("\nparsing Siena seizure annotations:")
    ss = siena_seizures(verbose=True)
    print("   annotations for %d files" % len(ss))
    for k, v in list(ss.items())[:5]:
        print("   ", k, v)

    plt.loglog(fr[fr > 0], ps[fr > 0])
    plt.axvline(40, ls=":", c="r")
    plt.xlabel("Hz"); plt.title("Siena CZ-PZ"); plt.show()


def cmd_prep():
    from multiprocessing import Pool
    jobs = build_jobs()
    print("%d files (%d chb, %d siena)" % (
        len(jobs), sum(j[0] == "chb" for j in jobs),
        sum(j[0] == "siena" for j in jobs)), flush=True)
    t0 = time.time()
    with Pool(N_WORKERS, maxtasksperchild=10) as p:
        for i, m in enumerate(p.imap_unordered(process_file, jobs, chunksize=1), 1):
            if m.startswith("FAIL") or i % 10 == 0:
                el = time.time() - t0
                print("[%d/%d] %.1fm eta %.1fm | %s"
                      % (i, len(jobs), el / 60, el / i * (len(jobs) - i) / 60, m), flush=True)
    X, y, coh, sub = load_all()
    print("\nX", X.shape, "| positive rate %.4f" % y.mean())
    print(pd.crosstab(coh, y))


def _eval(Xtr, ytr, Xte, yte, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, average_precision_score
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
    m.fit(Xtr, ytr)
    s = m.predict_proba(Xte)[:, 1]
    auc, ap = roc_auc_score(yte, s), average_precision_score(yte, s)
    print("  %-28s AUROC=%.3f  AUPRC=%.3f  (ntr=%d nte=%d pos_te=%d)"
          % (tag, auc, ap, len(ytr), len(yte), int(yte.sum())))
    return auc, ap


def cmd_exp():
    X, y, coh, sub = load_all()
    nb = 5 * 17
    SETS = {"bandpower": slice(0, nb),
            "aperiodic": slice(nb, X.shape[1]),
            "both": slice(0, X.shape[1])}

    for fname, sl in SETS.items():
        print("\n===== %s =====" % fname)
        Xf = X[:, sl]

        print(" S1 within-patient (chb, per subject 70/30 by time):")
        aucs = []
        for s in np.unique(sub[coh == "chb"]):
            m = (coh == "chb") & (sub == s)
            n = m.sum(); cut = int(.7 * n)
            r = _eval(Xf[m][:cut], y[m][:cut], Xf[m][cut:], y[m][cut:], s)
            if r:
                aucs.append(r[0])
        if aucs:
            print("  mean AUROC %.3f (n=%d subjects)" % (np.mean(aucs), len(aucs)))

        print(" S2 cross-patient (chb split by subject):")
        subs = np.unique(sub[coh == "chb"]); half = len(subs) // 2
        tr = np.isin(sub, subs[:half]) & (coh == "chb")
        te = np.isin(sub, subs[half:]) & (coh == "chb")
        _eval(Xf[tr], y[tr], Xf[te], y[te], "chb A->B")

        print(" S3 cross-cohort:")
        _eval(Xf[coh == "chb"], y[coh == "chb"],
              Xf[coh == "siena"], y[coh == "siena"], "chb -> siena")
        _eval(Xf[coh == "siena"], y[coh == "siena"],
              Xf[coh == "chb"], y[coh == "chb"], "siena -> chb")


def cmd_probe2():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    from scipy.stats import ks_2samp

    X, y, coh, sub = load_all()
    nb = 5 * 17

    print("cohort identity recoverable from features? (5-fold AUROC)")
    for name, sl in [("bandpower", slice(0, nb)), ("aperiodic", slice(nb, X.shape[1]))]:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        s = cross_val_score(m, X[:, sl], (coh == "siena").astype(int),
                            cv=5, scoring="roc_auc")
        print("  %-12s AUROC=%.3f" % (name, s.mean()))

    print("\nlargest distribution shifts (KS statistic):")
    ks = [(i, ks_2samp(X[coh == "chb", i], X[coh == "siena", i]).statistic)
          for i in range(X.shape[1])]
    for i, d in sorted(ks, key=lambda t: -t[1])[:10]:
        kind = "bandpower" if i < nb else "aperiodic"
        print("  feat %3d (%-9s)  KS=%.3f" % (i, kind, d))


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "probe"
    {"probe": cmd_probe, "prep": cmd_prep,
     "exp": cmd_exp, "probe2": cmd_probe2}[c]()
