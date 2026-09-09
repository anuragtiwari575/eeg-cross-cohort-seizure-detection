"""
=============================================================================
EXTRACT FULL CONTINUOUS RAW WINDOWS  (for native-prevalence inference)
=============================================================================
    python -u extract_full.py
    python -u extract_full.py "D:\\path\\to\\chb-mit"  "D:\\path\\to\\siena"

WHY THIS EXISTS

The learned models were trained and evaluated on a subsample: every positive
window, but only 60 negative windows per recording, giving 7.36% positive.
The handcrafted baseline was evaluated on everything, at the native 0.34%.
The two are therefore not directly comparable, and the subsampled test sets
do not contain enough negatives to resample down -- the matched-prevalence
sweep bottomed out at 5%.

This script writes every window of both cohorts as raw time series, so that
the already-trained models can be scored where a deployed system operates.
Nothing is retrained; the weights saved by the sweep are reused.

WHY IT WRITES SHARDS

One million windows of 17 channels by 1024 samples is about 70 GB as
float32, which neither fits in memory nor uploads to Kaggle. Two things
reduce it:

    float16 halves it, and the models run in mixed precision anyway
    sharding by subject keeps each file small enough to upload and lets
        inference stream one shard at a time

The result is roughly 35 GB across ~38 shards. That is still too much to
upload whole, so INFERENCE_SUBSET below selects the folds that actually
need scoring: the two cross-cohort directions need Siena and CHB-MIT
respectively, which is all of it, but a single S2 fold needs only its five
held-out subjects. Start with the S3 directions, which are the comparison
the paper turns on, and add S2 folds if the upload budget allows.

PREPROCESSING is identical to xcohort.py and extract_raw.py: the same 17
bipolar derivations, the same 1-40 Hz band, resampling to 256 Hz, and
per-recording z-scoring. Any difference here would invalidate the
comparison, so the loaders are imported from xcohort.py rather than
reimplemented.

OUTPUT  D:\\xcohort_full\\<subject>.npz
    X       float16, (n_windows, 17, 1024)
    y       int8
    rec     source recording of each window
    order   index of the window within its recording
=============================================================================
"""
from pathlib import Path
import os, sys, time, importlib.util, warnings
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

# ----------------------------------------------------------------- config
OUTDIR      = Path(r"D:\xcohort_full")
WORKERS     = 4          # each holds one recording in memory; keep modest
BIG_GB      = 2.0        # above this, read channel by channel
DTYPE       = np.float16

# Which subjects to write. "all" is ~35 GB; the S3 directions need all of
# both cohorts, so start there only if the upload budget allows. Otherwise
# name the subjects of one held-out fold.
SUBSET      = "all"      # or e.g. ["chb01", "chb02", "PN00"]


def load_xcohort():
    """Reuse the validated loaders rather than reimplementing them."""
    here = Path(__file__).resolve().parent
    for c in (here / "xcohort.py", Path(r"D:\eeg_project\xcohort.py"),
              Path.cwd() / "xcohort.py"):
        if c.exists():
            spec = importlib.util.spec_from_file_location("xcohort", str(c))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            print("using loaders from %s" % c)
            return m
    sys.exit("xcohort.py not found -- put it beside this script")


xc = load_xcohort()

if len(sys.argv) > 2:
    xc.CHB, xc.SIENA = Path(sys.argv[1]), Path(sys.argv[2])
print("CHB-MIT : %s  (%s)" % (xc.CHB, "found" if xc.CHB.exists() else "MISSING"))
print("Siena   : %s  (%s)" % (xc.SIENA, "found" if xc.SIENA.exists() else "MISSING"))
if not (xc.CHB.exists() and xc.SIENA.exists()):
    sys.exit("edit the paths at the top of xcohort.py, or pass them as arguments")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- helpers
def est_gb(path):
    import mne
    r = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    g = r.n_times * len(r.ch_names) * 8 / 1e9
    del r
    return g


def load_lowmem(path, cohort):
    """Same output as xc.load_bipolar without holding all channels at once."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    raw.rename_channels({c: c.replace("EEG ", "").strip() for c in raw.ch_names})
    up = {c.upper(): c for c in raw.ch_names}
    rows = []
    for name, (a, b) in xc.BIPOLAR:
        if cohort == "chb":
            if name not in up:
                return None, None
            rows.append(raw.get_data(picks=[up[name]])[0].astype(np.float32))
        else:
            if a.upper() not in up or b.upper() not in up:
                return None, None
            sa = raw.get_data(picks=[up[a.upper()]])[0].astype(np.float32)
            sb = raw.get_data(picks=[up[b.upper()]])[0].astype(np.float32)
            rows.append(sa - sb); del sa, sb
    fs_in = raw.info["sfreq"]; del raw
    data = np.asarray(rows, np.float32); del rows
    info = mne.create_info(xc.CHNAMES, fs_in, "eeg", verbose="ERROR")
    r2 = mne.io.RawArray(data, info, verbose="ERROR"); del data
    r2.filter(*xc.BAND, verbose="ERROR")
    if abs(r2.info["sfreq"] - xc.FS) > 1:
        r2.resample(xc.FS, verbose="ERROR")
    return r2.get_data().astype(np.float32), xc.FS


def windows_of(job):
    """One recording -> (X, y, file, order) over every window."""
    cohort, subj, fname, path, sz = job
    big = est_gb(path) > BIG_GB
    data, fs = (load_lowmem(path, cohort) if big else xc.load_bipolar(path, cohort))
    if data is None:
        return None
    w = int(xc.WIN_S * fs)
    nw = data.shape[1] // w
    mu = data.mean(1, keepdims=True); sd = data.std(1, keepdims=True) + 1e-12
    keep, lab = [], []
    for i in range(nw):
        t0, t1 = i * xc.WIN_S, (i + 1) * xc.WIN_S
        ov = sum(max(0, min(t1, b) - max(t0, a)) for a, b in sz) / xc.WIN_S
        if 0 < ov < 0.5:            # ambiguous boundary window, as before
            continue
        keep.append(i); lab.append(1 if ov >= 0.5 else 0)
    if not keep:
        return None
    X = np.stack([(data[:, i * w:(i + 1) * w] - mu) / sd for i in keep])
    del data
    return (X.astype(DTYPE), np.array(lab, np.int8),
            np.array([fname] * len(keep)), np.array(keep, np.int32))


def process_subject(args):
    subj, jobs = args
    out = OUTDIR / ("%s.npz" % subj)
    if out.exists():
        return "skip %s" % subj
    Xs, ys, fs_, os_ = [], [], [], []
    for j in jobs:
        try:
            r = windows_of(j)
        except Exception as e:
            return "FAIL %s/%s: %s: %s" % (subj, j[2], type(e).__name__, e)
        if r is None:
            continue
        Xs.append(r[0]); ys.append(r[1]); fs_.append(r[2]); os_.append(r[3])
    if not Xs:
        np.savez_compressed(out, X=np.zeros((0,), DTYPE), y=np.zeros((0,), np.int8))
        return "empty %s" % subj
    X = np.concatenate(Xs); y = np.concatenate(ys)
    # 'rec' rather than 'file': savez_compressed's own first parameter is
    # named file, so a file= keyword collides with it.
    np.savez_compressed(out, X=X, y=y, rec=np.concatenate(fs_),
                        order=np.concatenate(os_))
    return "ok %s  n=%d pos=%d  %.0f MB" % (subj, len(y), int(y.sum()),
                                            out.stat().st_size / 1e6)


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = xc.build_jobs()
    by_subj = {}
    for j in jobs:
        by_subj.setdefault(j[1], []).append(j)
    if SUBSET != "all":
        by_subj = {k: v for k, v in by_subj.items() if k in SUBSET}
    todo = [(s, v) for s, v in sorted(by_subj.items())
            if not (OUTDIR / ("%s.npz" % s)).exists()]
    print("\n%d subjects, %d still to write -> %s"
          % (len(by_subj), len(todo), OUTDIR), flush=True)

    t0 = time.time()
    with Pool(WORKERS, maxtasksperchild=2) as p:
        for i, m in enumerate(p.imap_unordered(process_subject, todo), 1):
            el = time.time() - t0
            print("[%d/%d] %.1fm eta %.1fm | %s"
                  % (i, len(todo), el / 60, el / i * (len(todo) - i) / 60, m),
                  flush=True)

    files = sorted(OUTDIR.glob("*.npz"))
    tot = sum(f.stat().st_size for f in files) / 1e9
    n = pos = 0
    rows = []
    for f in files:
        d = np.load(f)
        if d["X"].size == 0:
            continue
        n += len(d["y"]); pos += int(d["y"].sum())
        rows.append(dict(subject=f.stem, windows=len(d["y"]),
                         positives=int(d["y"].sum()),
                         mb=round(f.stat().st_size / 1e6, 1)))
    man = pd.DataFrame(rows)
    man.to_csv(OUTDIR / "manifest.csv", index=False)
    print("\n%d shards, %.1f GB total" % (len(files), tot))
    print("%d windows, %d positive (%.3f%%)" % (n, pos, 100 * pos / max(n, 1)))
    print(man.to_string(index=False))
    print("\nwrote manifest.csv")
    print("\nUpload the shards you need as a Kaggle Dataset, together with the")
    print("weights/ folder from the sweep, then run infer_full.py there.")
    print("The two cross-cohort directions need all of Siena and all of")
    print("CHB-MIT respectively; a single S2 fold needs only its five")
    print("held-out subjects, which is a much smaller upload.")
