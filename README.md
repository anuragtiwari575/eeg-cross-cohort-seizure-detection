# Cross-cohort transfer in scalp EEG seizure detection

Code, corrected annotations and per-fold results for the paper *The cost of
cross-cohort transfer in scalp EEG seizure detection depends more on
protocol choices than on the cohort boundary*.

## What the paper found

A seizure detector can be asked to cross two boundaries: to a new patient
within a cohort, or to a different cohort. Transferring between centres is
usually assumed to cost more. We measured both under one protocol on CHB-MIT
and Siena, scoring every setting on the same 1,000,277 four-second windows
of continuous recording at the native positive rate of 0.35 %.

|                   | learned | handcrafted linear | handcrafted boosted |
| ----------------- | ------- | ------------------ | ------------------- |
| S1 within-patient | 0.958   | —                  | —                   |
| S2 cross-patient  | 0.949   | 0.801              | —                   |
| S3 cross-cohort   | 0.856   | 0.794              | 0.857               |
| CHB-MIT → Siena   | 0.859   | 0.807              | 0.875               |
| Siena → CHB-MIT   | 0.854   | 0.782              | 0.839               |

The cross-patient drop is 0.008 and the additional cross-cohort drop 0.093
for the learned representations. On identical windows and splits the same
quantity is 0.007 for handcrafted features: the two arms disagree
thirteenfold about a number that is supposed to describe the data rather
than the method.

Scoring the cross-patient setting on the class-balanced subsample the models
were trained on, and the cross-cohort setting on continuous recording, gives
0.072 and 0.024 and reverses the ordering. Five defensible ways of measuring
one quantity give answers from −0.054 to 0.093, and none of the choices
separating them is about cohorts.

## Corrected Siena annotations

`data/siena_annotations_corrected.csv` holds all 47 annotated seizures of
the Siena Scalp EEG Database, with start and end times in seconds from the
start of each recording.

A regular expression written against the format most of the files use
recovers 45. The four defects are in the parsing, not in the database:

| Subject | Defect |
| ------- | ------ |
| PN01 | writes `Start time` and `End time` where every other subject writes `Seizure start time` and `Seizure end time`, so both its seizures are lost and 13.5 hours are labelled negative |
| PN01, PN11 | name files that do not exist: `PN01.edf` and `PN11-.edf` against the actual `PN01-1.edf` and `PN11-1.edf` |
| PN06 | writes its own identifier as `PNO6`, with a letter O for the zero, in three of five entries |
| PN00-3 | gives an end time of 19.29.29 against a start of 18.28.29 in a recording of 2509 s; read as 18.29.29 it is a 60 s seizure the recording does contain |

Forty-six are recovered exactly as annotated and one with the corrected end
time. If you are parsing this database, you will meet the same four.

`code/fix_labels.py diagnose` prints what each annotation file contains, and
`rebuild` writes the corrected CSV.

## Running it

Python 3 with numpy, scipy, pandas, scikit-learn, matplotlib, mne, neurodsp,
specparam, torch and braindecode.

```
pip install numpy scipy pandas scikit-learn matplotlib mne neurodsp \
            specparam torch braindecode
```

Each script carries its input and output paths at the top, as absolute paths
from the machine the work was done on. Change those first; nothing is read
from a config file.

The order they were run in, with later stages consuming earlier ones:

| Step | Script | What it does |
| ---- | ------ | ------------ |
| 1 | `fix_labels.py` | diagnose the annotation files, then write the corrected CSV |
| 2 | `rerun_siena.py` | rebuild Siena from those labels and rerun everything that tests on it; seven stages with resumable markers, about six hours |
| — | `resume.py` | run after an interruption: opens every artefact, deletes the ones that will not load, clears markers left by partly finished stages |
| 3 | `refit.py` | GPU. Rescores within-patient runs on the held-out 30 % rather than the whole shard, and reconstructs the cross-patient folds |
| 4 | `analysis.py` | the boundary comparison with one variable changing at a time, blocked permutation, recording-context control, a boosted baseline, dense event grid |
| 5 | `normalisation2.py` | causal normalisation, both feature arms, both boundaries |
| 6 | `retrain.py` | GPU. The learned arm under causal normalisation, and the two architectures never rerun with multiple seeds |
| 7 | `recompute_all.py` | every figure the paper quotes, from one pass |
| 8 | `make_figs.py` | six of the ten figures |
| — | `verify.py` | independent checks: seizure counts, PN00-3, the montage, the feature probes |

`sensitivity.py`, `fix_table1.py`, `table_numbers.py`, `collect.py`,
`make_f2.py` and `fix_figures.py` are earlier stages, kept so that a reader
checking how a figure came about can follow the whole path rather than only
its last step. Some of their numbers were superseded; the paper says where.

## Two things worth knowing before you run any of it

**The feature vector must match the cached CHB-MIT matrices column for
column.** An earlier version of the Siena extraction used different Welch
parameters and grouped bands across channels rather than channels across
bands. The cross-patient figures were unaffected, being CHB-MIT on both
sides, while the cross-cohort figures fell below chance — which is the
signature of two feature spaces being compared rather than of a weak model.
`rerun_siena.py` now checks the layout before computing anything and stops
on a mismatch.

**Spectral parameterisation is the slow part**, roughly two million model
fits for Siena alone. `rerun_siena.py` spreads it over eight processes,
which takes about twenty minutes; single-threaded it takes several hours.

## Data

Both datasets are public on PhysioNet and are not redistributed here.

- CHB-MIT Scalp EEG Database, <https://doi.org/10.13026/C2K01R>
- Siena Scalp EEG Database, <https://doi.org/10.13026/5d4a-j060>

Download them there and point the scripts at your copies.

## Layout

```
code/     analysis scripts
data/     corrected Siena annotations, window manifest, per-fold results
paper/    manuscript source and figures
```

## Citing

The paper is under review. A permanent archived identifier will be deposited
on acceptance and this section updated with it.

## Licence

Code under the MIT licence. The datasets remain under PhysioNet's terms.
