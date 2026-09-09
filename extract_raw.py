"""
Extract subsampled RAW windows (17 x 1024) for deep-learning baselines.

    python -u extract_raw.py

Keeps every seizure window and a random NEG_PER_FILE sample of non-seizure
windows, so the output stays small enough to upload to Kaggle.

Resumable: already-processed files are skipped, so it is safe to re-run.

Memory: the largest Siena recordings are 5+ hours at 512 Hz across 45 channels,
which is several GB as float64. Big files are therefore routed to a low-memory
path (channels read one at a time) and processed by fewer workers.

Output: E:\\xcohort_raw\\raw_windows.npz  (X float16, y, cohort, subject)
"""
from pathlib import Path
import os, sys, time, warnings, importlib.util
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
import numpy as np

# reuse the loaders / annotation parsers already validated in xcohort.py
_spec = importlib.util.spec_from_file_location(
    "xcohort", str(Path(__file__).with_name("xcohort.py")))
xc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xc)

OUTDIR = Path(r"E:\xcohort_raw"); OUTDIR.mkdir(parents=True, exist_ok=True)
NEG_PER_FILE = 60          # random non-seizure windows kept per file
BIG_GB       = 2.0         # files above this (as float64) use the low-memory path
WORKERS_SMALL = 10
WORKERS_BIG   = 2          # big files are memory-bound, not CPU-bound


def _est_gb(path):
    import mne
    r = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    gb = r.n_times * len(r.ch_names) * 8 / 1e9
    del r
    return gb


def _load_lowmem(path, cohort):
    """Same output as xc.load_bipolar but never holds all channels at once."""
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
            rows.append(sa - sb)
            del sa, sb
    fs_in = raw.info["sfreq"]
    del raw

    data = np.asarray(rows, dtype=np.float32)
    del rows
    info = mne.create_info(xc.CHNAMES, fs_in, "eeg", verbose="ERROR")
    r2 = mne.io.RawArray(data, info, verbose="ERROR")
    del data
    r2.filter(*xc.BAND, verbose="ERROR")
    if abs(r2.info["sfreq"] - xc.FS) > 1:
        r2.resample(xc.FS, verbose="ERROR")
    return r2.get_data().astype(np.float32), xc.FS


def process(job):
    cohort, subj, fname, path, sz = job
    tag = OUTDIR / ("%s__%s__%s.npz" % (cohort, subj, Path(fname).stem))
    if tag.exists():
        return "skip " + fname
    try:
        big = _est_gb(path) > BIG_GB
        data, fs = (_load_lowmem(path, cohort) if big
                    else xc.load_bipolar(path, cohort))
        if data is None:
            np.savez_compressed(tag, X=np.zeros((0,), np.float16),
                                y=np.zeros((0,), np.int8))
            return "nochan " + fname

        w = int(xc.WIN_S * fs)
        nw = data.shape[1] // w
        pos_idx, neg_idx = [], []
        for i in range(nw):
            t0, t1 = i * xc.WIN_S, (i + 1) * xc.WIN_S
            ov = sum(max(0, min(t1, b) - max(t0, a)) for a, b in sz) / xc.WIN_S
            if ov >= 0.5:
                pos_idx.append(i)
            elif ov == 0:
                neg_idx.append(i)

        rng = np.random.default_rng(abs(hash((subj, fname))) % (2 ** 32))
        if len(neg_idx) > NEG_PER_FILE:
            neg_idx = sorted(rng.choice(neg_idx, NEG_PER_FILE, replace=False).tolist())
        keep = sorted(pos_idx + neg_idx)
        if not keep:
            np.savez_compressed(tag, X=np.zeros((0,), np.float16),
                                y=np.zeros((0,), np.int8))
            return "empty " + fname

        # z-score each channel within the file, then store as float16
        mu = data.mean(1, keepdims=True)
        sd = data.std(1, keepdims=True) + 1e-12
        pos_set = set(pos_idx)
        X = np.stack([(data[:, i * w:(i + 1) * w] - mu) / sd for i in keep])
        y = np.array([1 if i in pos_set else 0 for i in keep], np.int8)
        del data
        np.savez_compressed(tag, X=X.astype(np.float16), y=y)
        return "ok %s n=%d pos=%d%s" % (fname, len(y), int(y.sum()),
                                        " [lowmem]" if big else "")
    except Exception as e:
        return "FAIL %s: %s: %s" % (fname, type(e).__name__, e)


def merge():
    Xs, ys, cs, ss = [], [], [], []
    for f in sorted(OUTDIR.glob("*__*.npz")):
        d = np.load(f)
        if d["X"].size == 0:
            continue
        cohort, subj, _ = f.stem.split("__", 2)
        Xs.append(d["X"]); ys.append(d["y"])
        cs += [cohort] * len(d["y"]); ss += [subj] * len(d["y"])
    if not Xs:
        print("nothing to merge")
        return
    X = np.concatenate(Xs); y = np.concatenate(ys)
    out = OUTDIR / "raw_windows.npz"
    np.savez_compressed(out, X=X, y=y,
                        cohort=np.array(cs), subject=np.array(ss))
    import collections
    print("\nmerged: X", X.shape, X.dtype,
          "| pos %d / %d (%.2f%%)" % (y.sum(), len(y), 100 * y.mean()))
    print("file: %s  (%.0f MB)" % (out, out.stat().st_size / 1e6))
    print("per cohort:", dict(collections.Counter(cs)))
    n_sub = len(set(ss))
    print("subjects: %d" % n_sub)


def run(jobs, workers, label):
    if not jobs:
        return
    from multiprocessing import Pool
    print("\n--- %s: %d files, %d workers ---" % (label, len(jobs), workers), flush=True)
    t0 = time.time()
    with Pool(workers, maxtasksperchild=4) as p:
        for i, m in enumerate(p.imap_unordered(process, jobs, chunksize=1), 1):
            if m.startswith("FAIL") or i % 25 == 0 or i == len(jobs):
                el = time.time() - t0
                print("[%d/%d] %.1fm eta %.1fm | %s"
                      % (i, len(jobs), el / 60,
                         el / i * (len(jobs) - i) / 60, m), flush=True)


if __name__ == "__main__":
    all_jobs = xc.build_jobs()
    todo = [j for j in all_jobs
            if not (OUTDIR / ("%s__%s__%s.npz"
                              % (j[0], j[1], Path(j[2]).stem))).exists()]
    print("%d files total, %d still to do" % (len(all_jobs), len(todo)), flush=True)

    small, big = [], []
    for j in todo:
        try:
            (big if _est_gb(j[3]) > BIG_GB else small).append(j)
        except Exception:
            small.append(j)
    print("  %d small, %d large (low-memory path)" % (len(small), len(big)), flush=True)

    run(small, WORKERS_SMALL, "small files")
    run(big, WORKERS_BIG, "large files")
    merge()
