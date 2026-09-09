"""
=============================================================================
REPOINT THE SCRIPTS FROM E: TO D:
=============================================================================
    python fix_paths.py

The external drive that was mounted as E: is now D:. Every script has the
old drive letter hardcoded at the top. This rewrites those constants in
place, after taking a .bak copy of each file, and then checks that the
directories it now points at actually exist.

It only touches paths that begin with a drive letter, so nothing else in
the code is affected. Run it once from the folder holding the scripts.
=============================================================================
"""
from pathlib import Path
import re, shutil, sys

HERE = Path(__file__).resolve().parent

# old -> new. Longest first, so the specific dataset paths are rewritten
# before the bare drive letter.
MAP = [
    (r"E:\CPU-Intel 7\C_Drive\Compressed\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0",
     r"D:\CPU-Intel 7\C_Drive\Compressed\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0"),
    (r"E:\CPU-Intel 7\C_Drive\archive\siena-scalp-eeg-database-1.0.0",
     r"D:\CPU-Intel 7\C_Drive\archive\siena-scalp-eeg-database-1.0.0"),
    (r"E:\xcohort_raw", r"D:\xcohort_raw"),
    (r"E:\xcohort_full", r"D:\xcohort_full"),
    (r"E:\cpu_results", r"D:\cpu_results"),
    (r"E:\figures",     r"D:\figures"),
    (r"E:\xcohort",     r"D:\xcohort"),
    (r"E:\preds",       r"D:\preds"),
    (r"E:\eeg_project", r"D:\eeg_project"),
]

SCRIPTS = ["xcohort.py", "xcohort2.py", "extract_raw.py", "extract_full.py",
           "cpu_analyses.py", "handcrafted_event.py", "preds_analyses.py",
           "make_figures.py", "make_figures_concept.py", "run.py", "sleep.py"]

changed = []
for name in SCRIPTS:
    p = HERE / name
    if not p.exists():
        continue
    t = orig = p.read_text(encoding="utf-8", errors="ignore")
    for old, new in MAP:
        t = t.replace(old, new)
    if t != orig:
        shutil.copy(p, p.with_suffix(p.suffix + ".bak"))
        p.write_text(t, encoding="utf-8")
        n = sum(orig.count(o) for o, _ in MAP)
        changed.append((name, n))

print("rewritten:")
for n, c in changed:
    print("   %-28s %d path(s)" % (n, c))
if not changed:
    print("   nothing to change (already on D:, or scripts not in this folder)")

print("\nchecking the directories those paths now point at:")
targets = [
    (r"D:\xcohort", "feature files"),
    (r"D:\cpu_results", "earlier results"),
    (r"D:\figures", "figures"),
    (r"D:\CPU-Intel 7\C_Drive\Compressed\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0", "CHB-MIT"),
    (r"D:\CPU-Intel 7\C_Drive\archive\siena-scalp-eeg-database-1.0.0", "Siena"),
]
missing = 0
for path, what in targets:
    d = Path(path)
    if d.exists():
        n = len(list(d.glob("*.npz"))) or len(list(d.glob("*.csv"))) or \
            len(list(d.rglob("*.edf"))) or len(list(d.glob("*")))
        print("   OK      %-14s %-70s (%d items)" % (what, path, n))
    else:
        print("   MISSING %-14s %s" % (what, path)); missing += 1

if missing:
    print("\n%d path(s) not found. Edit MAP above and run again." % missing)
    sys.exit(1)

print("""
Ready. Next, in this order:

  python -u handcrafted_event.py "D:\\xcohort"
      native-prevalence and event-level results for the handcrafted
      baseline, on continuous data, ~45 min

  python -u extract_full.py
      every window as raw time series, sharded by subject, 1-2 h
      needs xcohort.py beside it

Both are CPU only and can run at the same time.""")
