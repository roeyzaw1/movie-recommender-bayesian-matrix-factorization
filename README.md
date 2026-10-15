# DataLab A1 — Collaborative filtering (group 38 · LHOOQ)

Predict the missing ratings of a users × movies matrix (MovieLens subset).
Score = RMSE on hidden ratings (lower is better).

## Quick start

Put `ratings_train.npy` and `ratings_test.npy` in this folder first; they are not committed.
`namesngenre.npy` (movie titles and genres) IS committed: the model reads it at run time.

**Full explanation of the method, with references: [METHOD.md](METHOD.md).**

## Submitted model (short version)

Blend of **Bayesian matrix factorization (BPMF) with side-information priors** (weight 0.81)
and **user-KNN** (weight 0.195). BPMF averages predictions over many plausible models instead
of trusting one; its priors pull each movie towards what its genres, decade and popularity
predict, and each user towards what their rating behaviour predicts. ~24 s, NumPy only.

```bash
pip install -r requirements.txt
python3 checks.py                                # sanity checks on a tiny matrix (seconds)
python3 experiments.py                           # local protocol train -> test, writes results/ (~2 min)
python3 platform_sim.py                          # platform simulation, writes best_config.json (~3 min)
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
| `best_config.json` | The submitted model (blend), read by `generate.py`. |
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
Test scores are never used to choose anything. The submitted model is chosen separately by
`platform_sim.py`, at the platform's data density (see METHOD.md §2).

## Results

**Platform simulation** (train+test merged, 20% hidden, mean of 2 splits not used for any choice)

| Model | RMSE | Time |
|---|---:|---:|
| bias | 0.8561 | 0.2 s |
| user KNN | 0.8491 | 0.5 s |
| ALS matrix factorization | 0.8479 | 1.6 s |
| BPMF, plain | 0.8405 | 23 s |
| BPMF + side information | 0.8338 | 23 s |
| **blend (submitted)** | **0.8317** | 24 s |

**Local protocol** (`results/test.csv`): global mean 1.0369 · bias 0.8742 · user KNN 0.8715 ·
ALS 0.8718 · BPMF plain 0.8637 · BPMF + side information 0.8585.
