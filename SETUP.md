# Putting this on GitHub

From the folder that contains this file:

```bash
git init
git add .
git commit -m "Cross-patient and cross-cohort generalisation in scalp EEG seizure detection"
git branch -M main
git remote add origin https://github.com/<user>/eeg-cross-cohort-seizure-detection.git
git push -u origin main
```

**Keep the repository private until the manuscript is submitted.** The
results and the manuscript are both here, and neither is public yet.

## What is committed and what is not

`.gitignore` excludes every data file — `.npz`, `.npy`, `.edf`, `.pt` — so
the 32 GB of extracted windows and the 390 MB of model weights stay out of
git. Only code, the manuscript and the figure PDFs are committed. That is
deliberate: git handles text well and large binaries badly, and the data can
be regenerated from PhysioNet with the scripts here.

If you want the weights and per-window predictions preserved, put them on
Zenodo alongside the archived repository rather than in git.

## Before making it public

1. Fix the four items under *Known issues* in the README, or leave them
   listed as they are. Listing them is defensible; silently leaving them
   is not, given what this paper argues about evaluation practice.
2. Replace the drive letters with a config file or command-line arguments,
   so the scripts run somewhere other than the machine they were written on.
3. Add `requirements.txt` versions that were actually used:
   `pip freeze > requirements.txt` on the machine that produced the results.
4. Link the repository to Zenodo and take a DOI, then put that DOI in
   `CITATION.cff` and in the manuscript's data-availability statement.
