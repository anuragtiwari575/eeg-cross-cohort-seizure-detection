# eeg-cross-cohort-seizure-detection
# Cross-patient and cross-cohort generalisation in scalp EEG seizure detection

Does an automated seizure detector lose more when it meets a **new patient**
or a **new recording centre**? This repository holds every script used to
answer that on two public scalp EEG cohorts, together with the manuscript.

Manuscript in preparation. This is the working record; it will be tagged and
archived with a DOI at submission.

---

## What is here

```
code/       17 scripts, all of which were run; see the map below
paper/      main.tex, references.bib, figures/ (15 PDFs)
figures/    empty on clone; scripts write here
results/    empty on clone; scripts write here
```

Everything in `code/` produced something that appears in the paper. Scripts
that were written but never used, or that were superseded before producing a
reported number, are not included.

## Which script produced which result

| Script | Stage | Produces | Where it appears |
|---|---|---|---|
| `xcohort.py` | 1–4 | spectral features for every window; linear baselines; cohort-identity probe | Table 1, §Negative controls |
| `xcohort2.py` | 5 | leave-one-out, feature-shift decoding, normalisation ablation | Table 1, §Normalisation |
| `extract_raw.py` | 6 | subsampled raw windows (7.36% positive, 1.5 GB) | input to the GPU sweeps |
| `kaggle_master.py` | 7a | **14 architectures, one seed** | Table 2, Figure 6 |
| `kaggle_sweep_v2.py` | 7b | EEGNet, 3 seeds, with per-window predictions | superseded by v3; its predictions fed the first pass of F10–F12 |
| `kaggle_sweep_v3.py` | 7c | **4 architectures × 5 seeds, 300 runs**, predictions *and weights* | Table 3, §Seed variance |
| `cpu_analyses.py` | 8 | matched-window baseline, negative controls, prevalence sweep, per-subject | Figures 9, 11, 5 |
| `preds_analyses.py` | 9 | matched prevalence, bootstrap CIs, event level, reliability, from saved predictions | Table 4, Figures 7(part), 12, 13 |
| `extract_full.py` | 10 | **every** window as raw time series, 38 shards, 32 GB | input to native-prevalence inference |
| `infer_local.py` | 11 | saved weights scored on the full continuous set, local GPU | Table 5 |
| `score_report.py` | 12 | all metrics recomputed from the saved scores | Tables 5–6, Figures 7, 8 |
| `handcrafted_event.py` | 13 | handcrafted baseline at native prevalence, event level, on continuous data | Tables 5–6, Figure 8 |
| `make_figures.py` | 14 | F3, F4, F5 | Figures 9, 11, 5 |
| `make_figures_concept.py` | 14 | F1, F6, F7, F8, F9 | Figures 1, 3, 2, 10, 4 |
| `make_f2.py` | 14 | F2, from the 14-architecture results | Figure 6 |
| `fix_figures.py` | 14 | redraws F2, F11, F14, F15 with correct curve averaging | Figures 6, 12, 7, 8 |
| `fix_paths.py` | — | rewrites the hardcoded drive letter across every script | utility |

Two of these need a word of explanation.

`kaggle_master.py` and `kaggle_sweep_v3.py` are both sweeps and both are
kept, because they answer different questions and both are cited. The first
covers fourteen architectures at one seed and gives the benchmark in
Table 2. The second covers four architectures at five seeds and gives the
seed analysis in Table 3, which is what establishes that the differences in
Table 2 are within initialisation noise. Neither supersedes the other.

`fix_figures.py` exists because the first version of the event-level figures
averaged the curves wrongly. Each run chooses its thresholds from its own
score quantiles, so grouping rows by threshold and taking a mean interleaves
points from different operating points rather than averaging curves. The fix
interpolates every run onto a common false-alarm grid first. The original
code is not kept; the corrected version is what produced the figures in the
paper.

## Data

Neither dataset is redistributed. Both are public on PhysioNet:

- **CHB-MIT Scalp EEG Database** — 24 recording sets from 23 paediatric
  subjects (chb21 is chb01 recorded about eighteen months later), 256 Hz,
  bipolar longitudinal montage
- **Siena Scalp EEG Database** — 14 adult subjects, 512 Hz, referential
  montage

Paths are hardcoded at the top of each script. `fix_paths.py` rewrites them
all at once, which is how the move from `E:` to `D:` was handled partway
through this work.

## Running it

```bash
pip install -r requirements.txt
```

Then, in order. Each stage consumes the previous one.

| # | Command | Where | Time |
|---|---|---|---|
| 1 | `python -u xcohort.py probe` | CPU | 5 min |
| 2 | `python -u xcohort.py prep` | CPU | ~3 h |
| 3 | `python -u xcohort.py exp` | CPU | 10 min |
| 4 | `python -u xcohort.py probe2` | CPU | 5 min |
| 5 | `python -u xcohort2.py all` | CPU | ~1 h |
| 6 | `python -u extract_raw.py` | CPU | 20 min |
| 7 | `kaggle_master.py`, then `kaggle_sweep_v3.py` | GPU | 10–14 h each |
| 8 | `python -u cpu_analyses.py all` | CPU | 10 min |
| 9 | `python -u preds_analyses.py <preds folder>` | CPU | 15 min |
| 10 | `python -u extract_full.py` | CPU | ~25 min |
| 11 | `python -u infer_local.py` | GPU | ~4 h |
| 12 | `python -u score_report.py` | CPU | 15 min |
| 13 | `python -u handcrafted_event.py` | CPU | 45 min |
| 14 | `make_figures*.py`, `make_f2.py`, `fix_figures.py` | CPU | 5 min |

Stages 7 must be Kaggle **committed** notebooks (Save Version → Save & Run
All), not interactive sessions: those expire after about an hour of
inactivity and their working directory is wiped. Both scripts are
self-contained, write results after every run, echo the full table into the
log, and resume from a manifest if re-committed.

Stage 11 needs the weights from stage 7c. Running it locally rather than on
Kaggle avoids uploading 32 GB and allows both transfer directions to be
scored instead of one.

## Protocol

Three nested settings, identical windows and preprocessing throughout:

- **S1 within-patient** — 70/30 split in time inside one recording set
- **S2 cross-patient** — grouped 5-fold over CHB-MIT recording sets
- **S3 cross-cohort** — CHB-MIT ↔ Siena, both directions

Only CHB-MIT → Siena nests inside S1 and S2, since those are both trained on
CHB-MIT; the reverse direction changes the training cohort and is reported
as a replication rather than as a further step.

Windows are 4 s with no overlap. Positive if at least half the window falls
inside an annotated seizure, negative if none of it does, discarded
otherwise. This gives 1 000 311 windows over 38 subjects, 3 405 of them
positive: 0.34%.

## Settings determined empirically

Recorded because they are not obvious and were established by inspection
rather than convention.

**Band 1–40 Hz.** The two cohorts handle mains interference in opposite
ways. CHB-MIT already carries a notch with a broad stopband spanning roughly
50–70 Hz: power at 60 Hz sits about an order of magnitude below the
surrounding spectrum and rises again by 70 Hz, which cannot happen in an
unfiltered 1/f spectrum. Siena is un-notched with a sharp 50 Hz peak and a
harmonic at 100 Hz. Fitting a spectral model across the CHB-MIT stopband
inflates the estimated aperiodic exponent by about 50% (2.03 over 4–45 Hz
against 3.14 over 4–55 Hz in the same recordings).

**Fixed rather than knee aperiodic model.** The knee model did not converge,
returning exponents above 13 in some fits.

**`T8-P8` excluded.** Duplicated in CHB-MIT; including it would weight one
derivation twice.

**One Siena annotation excluded.** PN00-3 implies a 3660 s seizure and is
treated as a transcription error; 43 of 44 seizures retained. Two other
entries carry free text beside the timestamp, and the parser takes the first
well-formed `HH.MM.SS` token, which preserves the clinical onset.

**S1 admits 8 of 24 CHB-MIT recording sets.** Not only a sample-size limit:
in several sets every annotated seizure falls in the final 30% of the
recording, so a temporal split leaves no positive training examples at all.

## Main results

Handcrafted features, continuous data, native prevalence (AUROC):

| Setting | band power | aperiodic | both |
|---|---|---|---|
| Leave-one-out, CHB-MIT (n=24) | 0.613 | 0.636 | 0.718 |
| Leave-one-out, Siena (n=12) | 0.706 | 0.665 | 0.747 |
| CHB-MIT → Siena | 0.638 | 0.677 | 0.687 |
| Siena → CHB-MIT | 0.612 | 0.617 | 0.680 |

Learned against handcrafted on the same 1 000 311 windows at 0.34%: mean
cross-cohort AUROC 0.843 against 0.686, lift over chance 42.5× against 2.8×.

Architecture spread against seed noise: 0.66 cross-patient, 1.09
cross-cohort, 2.39 within-patient. Which architecture ranked best
cross-cohort changed with the seed.

Cost of each boundary, nested direction: S1→S2 0.063, S2→S3 0.039, ratio
0.62 with a 95% interval of 0.37 to 1.35. The ordering is consistent; the
multiple is not established.

Event level on continuous recording, one false alarm per hour: learned 22.1%
of cross-cohort seizures, handcrafted 1.1%.

## Known issues

1. **`extract_raw.py` seeds negative sampling with `hash()`.** Python
   randomises string hashing between processes unless `PYTHONHASHSEED` is
   set, so the selected negatives are not reproducible across runs. Needs a
   deterministic hash or an exported manifest.
2. **S2 fold membership is not recoverable from a run identifier.** The
   folds were formed inside the sweep, so `infer_local.py` cannot score
   them and skips 100 of 300 runs. Emitting fold membership from the sweep
   would fix this.
3. **chb01 and chb21 are the same patient.** They are treated as separate
   units, as the database itself does, so a cross-patient split that
   separates them is not strictly patient-independent. The gap between the
   recordings is about eighteen months, so the leakage is limited but not
   zero.
4. **Seed variance covers 4 of the 14 architectures.**
5. **F10 and F14 have cosmetic problems.** F10's x-axis extends well below
   the range that carries data; F14 has a spike near 1% from averaging runs
   with differing native prevalences. Neither affects a reported number.

## Citation

Manuscript in preparation. Until it appears, cite this repository by its
archived DOI (to be added on release).

## Licence

MIT for the code. The datasets carry their own terms; see PhysioNet. Nothing
here redistributes patient data.


Code released under the MIT Licence — see LICENSE.

The datasets carry their own terms; consult PhysioNet. Nothing in this repository redistributes patient data.
