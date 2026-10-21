# A1 metadata correction — what changed and how to verify

## Official correction (15 October 2026)

The original `namesngenre.npy` did **not** align titles and genres with
rating columns. The instructor's corrected `namesngenre2.npy` has 4,980 rows
and 2 columns (`title`, `genres`), matched in order to the 4,980 rating columns.
The accompanying `movie_ids.npy` contains the original, unique MovieLens ID
for each column. Never compare by title: different IDs can share a title.

Comparison of the old project ZIP with the corrected file:
- Only 648 / 4,980 titles (13.0%) agree at the same position.
- Only ~17.1% of genre strings agree at the same position.
- Train/test/eval rating splits are unchanged by the correction.

## Changes to code

1. `models.movie_metadata()` now loads `namesngenre2.npy`, not the old file.
2. It verifies the exact official contents, the expected 4,980 metadata rows,
   and the unique 4,980 movie IDs. An old file renamed to the new filename
   causes a loud validation error instead of silently corrupting features.
3. For tiny synthetic test matrices (not full A1 data), the loader returns
   `None` so toy regression/sanity tests still work as before.
4. Both item features and user-genre preference features reflect the corrected
   indexing; **no algorithmic change** to Gibbs/ALS/KNN is necessary.
5. All old experimental CSVs and the pre-correction `best_config.json` are
   archived in `results/legacy_pre_metadata_correction/`.

## Correct evaluation procedure

- `python3 verify_metadata.py` checks file alignment, hashes, and both unchanged
  rating matrices; no fitting or evaluation is performed.
- `python3 regression_tests.py` and `python3 checks.py` confirm interface and math.
- `python3 platform_sim.py` performs the fixed disjoint fit/calibration/evaluation
  partition. `results/platform_sim_corrected.csv` contains **new** scores only.
- **Only after reviewing the new results:** `python3 platform_sim.py --write-config`
  recalibrates `best_config.json`; it does not automatically select a better
  model based on evaluation scores.
- `python3 experiments.py` (train-only validation + test) and `python3 ablation.py`
  must also be rerun if their old numbers are to appear in the final report.
- `python3 generate.py --name ratings_train.npy` must produce finite predictions
  of the expected shape and preserve observed entries.

The frozen blend evaluation split is separate from blend calibration, but
train and test labels have already influenced earlier team development.
This is **not** an untouched independent evaluation of the entire project.
Only the professor's hidden evaluation is genuinely external to development.

### Reporting

The corrected metadata may improve OR worsen RMSE, so do not assume the
previous improvement remains. The old local reported BPMF-side (0.83352) and blend (0.83274) scores are
explicitly historical. On the same seed/split after the fix, BPMF-side scored
**0.81513** and blend scored **0.81513**, as recorded in
`results/platform_sim_corrected.csv`. Calibration selected weights BPMF=1,
KNN=0, so `best_config.json` now directly invokes BPMF with side information
(the same predictions without running an unnecessary KNN call). Re-running
`platform_sim.py --write-config` recreates the equivalent 1/0 blend config.
The Bayesian prior `prior_mode='plug_in'` still uses estimated metadata
coefficients instead of drawing them from the full posterior.
