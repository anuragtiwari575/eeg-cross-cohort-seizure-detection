# eeg-cross-cohort-seizure-detection
Does a seizure detector lose more when it meets a new patient or a new recording centre? Multi-architecture evaluation on CHB-MIT and Siena under one protocol, at native prevalence. Code for a manuscript in preparation.
Why this study

Automated seizure detection on CHB-MIT has reported accuracies above 98% for more than a decade, and several recent systems report above 99%. Clinical adoption has not followed. The gap is usually attributed to distribution shift — a detector trained at one centre is expected to fail at another — and a large literature on domain adaptation follows from that expectation.

Two things about that expectation had not been checked.

First, the two shifts have never been measured side by side. Cross-subject and cross-dataset generalisation are treated as related but distinct problems, and the second is generally presented as the harder. But they are rarely evaluated on identical windows, under identical splits, with identical models. Without that comparison you cannot tell whether a method that improves cross-dataset transfer is addressing dataset shift specifically, or is addressing inter-subject variability that is equally present inside a single cohort. If it is the latter, the alignment is operating on the wrong axis.

Second, reported figures are not comparable between studies. The positive class here is rare — interictal activity outnumbers ictal by two to three orders of magnitude — and the common practice of rebalancing before evaluation, whether by synthetic oversampling, by overlapping-window augmentation, or by majority undersampling, changes the conditions under which the metric is computed. Two papers using the same dataset and the same task may not be describing comparable systems.

This repository measures both.

What is measured

Three nested settings, with everything else held fixed:

	Train	Test	Question
S1 within-patient	first 70% of one subject, in time	remaining 30%	performance ceiling; what patient-specific studies report
S2 cross-patient	4/5 of CHB-MIT subjects	held-out 1/5	a new patient, same centre
S3 cross-cohort	all of one cohort	all of the other	a new centre, both directions

The comparison the study turns on is S1→S2 against S2→S3: the first is the cost of a new patient, the second the additional cost of a new centre.

Two feature families are compared under the same protocol — handcrafted spectral features (relative band power, and the aperiodic exponent from spectral parameterisation) and learned representations from published architectures — so that any difference is attributable to the representation rather than to the pipeline.

The evaluation prevalence is then swept from 50% down to the native 0.34%, with the trained model held fixed, to separate what a reported number owes to method from what it owes to protocol.

Findings so far

Handcrafted features, continuous data, native prevalence (AUROC):

Setting	band power	aperiodic	both
Leave-one-subject-out, CHB-MIT (n=24)	0.613	0.636	0.718
Leave-one-subject-out, Siena (n=12)	0.706	0.665	0.747
CHB-MIT → Siena	0.638	0.677	0.687
Siena → CHB-MIT	0.612	0.617	0.680

The central comparison. With band-power features, 13 of 24 CHB-MIT leave-one-subject-out folds scored worse than the cross-cohort transfer, and 11 of 24 in the reverse direction. With the combined feature set it is 8 of 24 and 7 of 24. In a substantial fraction of cases, testing on a new patient from the same centre is harder than testing on a different centre entirely.

What the features encode, measured identically (5-fold logistic regression on the same windows):

target	AUROC
cohort identity	0.92
subject identity (mean of one-vs-rest)	0.95
seizure label	0.81

Identity — both kinds — is more recoverable than the label the model is meant to predict. Cohort and subject are close to each other, which is the mechanism behind S2 ≈ S3: in this feature space the two boundaries are alike.

Cohort shift exists but is largely benign. The per-feature magnitude of the shift between cohorts is uncorrelated with that feature's importance for seizure detection (r = −0.096). The features that move most between cohorts — beta-band power, mean KS 0.249 — are largely not the discriminative ones.

Prevalence dependence. Model held fixed, only the evaluation class balance varied: AUROC moves 0.033 across the whole range while AUPRC moves by a factor of 165 (0.810 at 50% prevalence, 0.005 at the native 0.34%). Precision lift over chance stays between 1.6× and 3.9× throughout, so the collapse reflects the changing base rate rather than a change in discriminative ability. A figure reported at balanced prevalence and one reported on continuous recording cannot be compared on AUPRC, and AUROC does not reveal the difference.

Per-subject variation is wide and includes failures the mean conceals. With band power, 8 of 24 subjects fall below chance and chb14 reaches 0.141 — systematically inverted ranking, not absent signal. Combining feature families reduces this to 2 of 24 but does not remove it.

Per-subject standardisation helps asymmetrically (CHB-MIT → Siena 0.687 → 0.745, but Siena → CHB-MIT 0.680 → 0.632), so it is reported as an observation rather than as a recommendation.

The learned-representation sweep is still running; results will be added as a further column.

Repository layout
code/
    xcohort.py               stages 1-4: features, linear baselines, probes
    xcohort2.py              stage 5: LOSO, shift decoding, normalisation
    extract_raw.py           stage 6: subsampled raw windows for the GPU stage
    kaggle_master.py         stage 7: architecture sweep (Kaggle, self-contained)
    cpu_analyses.py          stage 8: matched baseline, controls, prevalence
    make_figures.py          stage 9: results figures F2-F5
    make_figures_concept.py  stage 10: concept and method figures F1, F6-F9
paper/
    main.tex                 manuscript (Introduction and Related Work drafted)
    references.bib           bibliography; entries marked [CHECK] need verifying
    intro_relatedwork.md     the same text in markdown, for drafting
figures/                     generated; empty on clone
results/                     generated; empty on clone

Only scripts that were actually run are included. Earlier exploratory work on circadian aperiodic dynamics belongs to a different question and is not part of this repository.

Data

Neither dataset is redistributed. Both are public on PhysioNet:

CHB-MIT Scalp EEG Database — 23 paediatric subjects, continuous recording during pre-surgical evaluation with medication tapered, 256 Hz, bipolar longitudinal montage
Siena Scalp EEG Database — 14 adult subjects (20–71 years), 512 Hz, referential montage, 29 EEG channels plus EKG

The two differ in age group, acquisition hardware, sampling rate, montage and mains frequency, which is what makes a transfer between them a genuine dataset-level shift rather than a nominal one.

Paths are set at the top of each script and must be edited before use. The committed values point at the drive the work was developed on.

Installation
bash
git clone <this repository>
cd eeg-cross-cohort-seizure-detection
pip install -r requirements.txt

braindecode changed its API during this work — EEGNetv4 was renamed to EEGNet, and the return format of get_params changed — so the pinned version is not incidental. kaggle_master.py probes several constructor signatures and skips any architecture it cannot build, recording the reason in the results file rather than failing.

Stage 7 needs a GPU. Everything else runs on CPU; stage 2 is the slowest at roughly three hours on 12 cores.

Running the pipeline

Run in order. Each stage consumes the previous one.

#	Command	Where	Time	Produces
1	python -u xcohort.py probe	laptop	5 min	montage and spectral checks, annotation parsing
2	python -u xcohort.py prep	laptop	~3 h	per-window spectral features
3	python -u xcohort.py exp	laptop	10 min	linear baselines, S1/S2/S3
4	python -u xcohort.py probe2	laptop	5 min	cohort-identity probe, feature shift
5	python -u xcohort2.py all	laptop	~1 h	LOSO, shift decoding, normalisation ablation
6	python -u extract_raw.py	laptop	20 min	subsampled raw windows (~1.5 GB)
7	kaggle_master.py	Kaggle GPU	6–11 h	architecture sweep
8	python -u cpu_analyses.py all	laptop	10 min	matched baseline, controls, prevalence, per-subject
9	python -u make_figures.py all	laptop	1 min	F2–F5
10	python -u make_figures_concept.py all	laptop	4 min	F1, F6–F9

Stages 1–6 and 8–10 are ordinary scripts. Stage 7 needs care.

Stage 7: the architecture sweep

kaggle_master.py must run as a Kaggle committed notebook — Save Version → Save & Run All — not as an interactive session. Interactive sessions expire after about an hour of inactivity and their working directory is wiped; a committed run is immune to that and keeps going with the browser closed. Kaggle allows 12 hours per GPU session and 30 hours a week.

Paste the entire file into a single cell. It is self-contained. It writes results three ways after every run — the CSV, a backup copy, and a full dump into the log — so nothing is lost if the working directory disappears, and it skips completed runs on re-execution, so a truncated sweep can be resumed by committing again.

The script stops cleanly at 11.3 hours and prints its summary rather than being cut off at the platform limit.

Protocol in detail

Montage harmonisation. The 17 bipolar derivations of the CHB-MIT longitudinal montage are constructed in both cohorts; for Siena this means deriving bipolar channels from the referential recording, using the older T3/T4/T5/T6 nomenclature those files carry. All 41 Siena recordings and 683 of 686 CHB-MIT recordings support all 17.

Filtering and resampling. Band-pass 1–40 Hz (see the next section for why), Siena resampled from 512 Hz to 256 Hz after filtering, each channel z-scored within its own recording. Per-recording normalisation removes session-level amplitude scaling without using information from any other recording.

Windowing. 4 s windows, no overlap. Overlapping windows drawn from the same seizure are near-duplicates, and their presence on both sides of a split inflates apparent performance; the choice is deliberate and differs from several published pipelines.

Labelling. Positive if at least 50% of the window falls inside an annotated seizure; negative if none of it does. Windows with intermediate overlap are discarded rather than assigned to either class, since their label is genuinely ambiguous.

This yields 1 000 311 windows across 38 subjects, of which 3 405 are positive — a native rate of 0.34%.

Subsampling for the GPU stage. Retaining every window as raw time series would need roughly 70 GB. The reduced set keeps every positive window and 60 randomly chosen negative windows per recording: 46 245 windows at 7.36% positive. No seizure data is discarded; only interictal data is reduced. This changes the evaluation prevalence, which is treated as a variable to be studied rather than a nuisance to be hidden.

Metrics. AUROC and AUPRC, with AUPRC always reported against the base rate it should be read against. At 0.34% positive, accuracy and specificity are uninformative, and AUROC alone is insufficient because it is invariant to prevalence while the achievable precision at a given sensitivity is not.

Settings determined empirically

These were established during development rather than chosen by convention, and are recorded because they are not obvious.

Band 1–40 Hz. The two cohorts handle mains interference in opposite ways. CHB-MIT already carries a notch, and a broad one: power at 60 Hz sits about an order of magnitude below the surrounding spectrum and rises again by 70 Hz, which cannot happen in an unfiltered 1/f spectrum. The stopband spans roughly 50–70 Hz. Siena is un-notched and shows a sharp 50 Hz peak, about fifteen times the adjacent spectrum, with a harmonic at 100 Hz. 1–40 Hz lies below both. This is not cosmetic: fitting a spectral model across the CHB-MIT stopband inflates the estimated aperiodic exponent by about 50% — 2.03 over 4–45 Hz against 3.14 over 4–55 Hz in the same recordings — an artefact of the filter rather than a property of the signal. See figures/F6_spectra.

Fixed rather than knee aperiodic model. The knee model did not converge on this data, returning exponents outside any physiologically plausible range (values above 13 in some fits).

T8-P8 excluded. It is duplicated in CHB-MIT, and including it would weight one derivation twice.

Frontopolar channels are noisy but retained. Quality-control pass rates by channel range from 0.51 (F8-T8, F7-T7, frontopolar) to 0.95 (CZ-PZ, midline), which follows anatomy — eye movement anteriorly, temporalis EMG laterally. Channels are not excluded on this basis; failing fits are set to missing and handled downstream.

One Siena annotation excluded. PN00-3 implies a 3660 s seizure and is treated as a transcription error; 43 of 44 seizures are retained. Two other Siena entries carry free text alongside the timestamp — an alternative time introduced by "oppure", and separate clinical and electrographic onsets — and the parser takes the first well-formed HH.MM.SS token, which preserves the clinical onset.

Figures

Generated into figures/ as 300-dpi PNG and vector PDF.

	Content
F1	Pipeline: two cohorts, one protocol, three evaluation settings
F2	Generalisation by architecture, and the cost of each boundary
F3	Prevalence sweep: AUROC flat, AUPRC across two orders of magnitude
F4	Per-subject leave-one-out, including the below-chance subjects
F5	Negative controls against chance
F6	Spectral evidence for the 1–40 Hz band
F7	The two boundaries, and the hypothesis under test
F8	Feature-space structure by cohort, subject and label, with separability
F9	Window labelling rule, including discarded boundary windows

F2 needs results/results_dl.csv from stage 7; the other results figures run from stage 8 output alone.

Negative controls

Two, both of which must sit at chance for anything else to be believed. Both do.

Label permutation within each subject, preserving per-subject class balance, three repeats: 0.511 (range 0.480–0.541), with AUPRC 0.0013–0.0046 against a base rate of 0.0034.

Recording context. In CHB-MIT every positive window comes from a seizure-containing recording. If those recordings carried a session-level signature — electrode adjustment, arousal, staff activity — a classifier could score well without representing ictal activity at all. Separating non-seizure windows of seizure recordings (n = 165 819) from non-seizure windows of seizure-free recordings (n = 704 791), grouped by subject: 0.522 (range 0.476–0.572). Session-level context is not linearly separable in these features.

Known issues

To be resolved before the repository is tagged for release. They are listed rather than quietly fixed because this study is partly about evaluation practice, and the same standard should apply here.

extract_raw.py seeds negative sampling with hash(). Python randomises string hashing between processes unless PYTHONHASHSEED is set, so the selected negatives are not currently reproducible across runs. Needs a deterministic hash, or an exported manifest.
The window .npz files record cohort and subject but not source file or window index, so a third party cannot reconstruct the exact subset. A manifest export is needed.
Seed variance is not yet characterised. Differences of about 0.02 AUROC between architectures should not be interpreted until repeated seeds have been run.
The identity probes are not equally hard. Subject identity is one-vs-rest across 38 subjects while cohort identity is binary. Both are measured the same way, but the comparison is indicative rather than definitive.
The prevalence sweep currently covers the linear model only. The same sweep is needed for at least two learned architectures.
Event-level metrics are not yet reported. Window-level AUROC is less clinically meaningful than sensitivity per seizure event with false alarms per hour, which is what a clinical reader will look for.
Reproducing a single result

To check the central comparison without running the whole pipeline, you need stages 1, 2 and 5 — roughly four hours, CPU only:

bash
python -u xcohort.py probe     # verify paths and annotations parse
python -u xcohort.py prep      # ~3 h
python -u xcohort2.py loso     # leave-one-subject-out and cross-cohort

The final block of stage 5 prints, for each feature family, how many leave-one-subject-out folds fall below the cross-cohort transfer. That count is the result the study rests on.

Citation

Manuscript in preparation. Until it appears, please cite this repository by its archived DOI (to be added on release).

Related work by the same author, applying a comparable argument to intracranial recordings: Cohort composition confounds the evaluation of interictal iEEG biomarkers for epilepsy surgery (submitted).

Licence

Code released under the MIT Licence — see LICENSE.

The datasets carry their own terms; consult PhysioNet. Nothing in this repository redistributes patient data.
