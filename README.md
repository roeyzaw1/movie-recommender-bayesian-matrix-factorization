# DataLab A1 — Collaborative filtering (group 38 · LHOOQ)

Predict the missing ratings of a users × movies matrix (MovieLens subset).
Score = RMSE on hidden ratings (lower is better).

## Quick start

Put `ratings_train.npy` and `ratings_test.npy` in this folder first; they are not committed.
`namesngenre.npy` (movie titles and genres) IS committed: the model reads it at run time.

**Full explanation of the method, with references: [METHOD.md](METHOD.md).**

## Submitted model (short version)

Blend of **Bayesian matrix factorization (BPMF) with side-information priors**
and **user-KNN**. Blend weights are calibrated on a separate holdout with
nonnegative weights summing to one; see `best_config.json`. BPMF averages predictions over many plausible models instead
of trusting one; its priors pull each movie towards what its genres, decade and popularity
predict, and each user towards what their rating behaviour predicts. Runtime depends strongly on the CPU/BLAS configuration; check platform timing.

```bash
pip install -r requirements.txt
python3 checks.py                                # sanity checks on a tiny matrix (seconds)
python3 regression_tests.py                      # test evaluation splitting and posterior sampling
python3 experiments.py                           # local protocol train -> test, writes results/ (~2 min)
python3 platform_sim.py                          # leakage-controlled simulation; does NOT overwrite config
python3 platform_sim.py --write-config           # rerun & explicitly update submitted weights
python3 platform_sim.py --quick                  # fast interface test; do NOT use these scores
python3 sensitivity.py --quick                   # sample-size stability smoke check
python3 ablation.py                              # which side information helps (~5 min)
python3 generate.py --name ratings_train.npy     # platform entry point -> output.npy (~25 s)
```

## Files

| File | What it does |
|---|---|
| `generate.py` | **Platform entry point.** Loads `--name`, fits the model in `best_config.json`, keeps real ratings, fills the gaps, saves `output.npy`. The two load/save lines are untouched. |
| `models.py` | All models, from global mean to Bayesian matrix factorization, each explained in comments. |
| `experiments.py` | Local protocol: tunes every model on a validation split of train, then scores each once on `ratings_test.npy`. Writes `results/`. |
| `checks.py` | Every model: right shape, no NaN, ratings in [0.5, 5], reproducible. |
| `platform_sim.py` | Simulates the platform (train+test merged, 20% hidden, 3 splits); fits blend weights; writes `best_config.json`. |
| `ablation.py` | Which side information helps BPMF. |
| `best_config.json` | Submitted model configuration, read by `generate.py`; inspect after running the corrected simulation. |
| `regression_tests.py` | Disjoint train/calibration/evaluation tests and Bayesian-prior checks. |
| `sensitivity.py` | Optional MCMC burn-in/sample-count sensitivity on a train-derived validation split. |
| `CHANGES_AND_VERIFICATION.md` | Fixes, caveats, and verification procedure. |
| `METHOD.md` | Full explanation of the method, results and references. |
| `namesngenre.npy` | Movie titles and genres (course file), used for the side-information prior. |
| `results/` | `validation.csv`, `test.csv`, `support.csv` (local protocol), `platform_sim.csv`, `ablation.csv`, logs. |

## Data

* `ratings_train.npy`, `ratings_test.npy`: 610 users × 4,980 movies, NaN = not rated.
  31,598 ratings each (≈1% of cells), never the same cell. Ratings 0.5–5 in half stars.
* Movies are very thinly rated: median 3 ratings per movie. Users: median 24.
* `namesngenre.npy`: title and genres of each movie (used by the side-information prior).
* The platform gives `generate.py` train + test merged (≈2% of cells) and scores a hidden eval set.

## Protocol

1. Hold out 20% of the training ratings (seed 2026) as a validation set.
2. Try every setting in `GRID` (experiments.py): fit on 80%, score on the 20%.
3. Lock the best setting per method, refit on all of train, score once on test.
Test scores from `experiments.py` are for reporting, but **the overall team has also
used train+test during platform-simulation/model development**. Therefore, do NOT claim
that ratings_test.npy has remained untouched across the entire project. Only the
instructor's hidden eval is completely external to development. The corrected
`platform_sim.py` uses disjoint training/calibration/evaluation partitions within
one simulation; see `CHANGES_AND_VERIFICATION.md`.

## Results

**HISTORICAL platform simulation before correction** (overlapping holdouts; do NOT cite these as an untouched final estimate)

| Model | RMSE | Time |
|---|---:|---:|
| bias | 0.8561 | 0.2 s |
| user KNN | 0.8491 | 0.5 s |
| ALS matrix factorization | 0.8479 | 1.6 s |
| BPMF, plain | 0.8405 | 23 s |
| BPMF + side information | 0.8338 | 23 s |
| **blend (submitted)** | **0.8317** | 24 s |

**HISTORICAL local protocol** (`results/test.csv`; may no longer represent the updated model): global mean 1.0369 · bias 0.8742 · user KNN 0.8715 ·
ALS 0.8718 · BPMF plain 0.8637 · BPMF + side information 0.8585.

## Verified corrected platform-density simulation

With a **disjoint 64% fit / 16% calibration / 20% evaluation** partition
(seed 2026, merged observed ratings), the historical plug-in BPMF approach got:

| Model | Corrected local evaluation RMSE |
|---|---:|
| Bias | 0.85996 |
| User KNN | 0.85194 |
| ALS | 0.85038 |
| Plain BPMF | 0.83915 |
| BPMF + metadata | 0.83352 |
| **Calibrated blend** | **0.83274** |

Weights are **0.91257 BPMF + 0.08743 KNN**, selected on the calibration data only.
The tiny difference between the last two rows may be noise. These are **local**
scores, not the instructor's official platform score. See `CHANGES_AND_VERIFICATION.md`.

## Important: corrected evaluation vs legacy results

The original CSVs `results/platform_sim.csv`, `results/ablation.csv`, and
`results/test.csv` are retained for provenance; they predate the evaluation
correction. Run `platform_sim.py` to create `results/platform_sim_corrected.csv`.
The MCMC option `prior_mode="sampled"` is scientifically more complete, but changes
model behavior and requires new validation; the default `"plug_in"` retains old
behavior. Do not compare results from different variants as if they came from one model.
