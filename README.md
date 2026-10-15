# DataLab A1 — Collaborative filtering (group 38 · LHOOQ)

Predict the missing ratings of a users × movies matrix (MovieLens subset).
Score = RMSE on hidden ratings (lower is better).

## Quick start

Put the course data files (`ratings_train.npy`, `ratings_test.npy`, `namesngenre.npy`)
in this folder first; they are not committed (see `.gitignore`).

```bash
pip install -r requirements.txt
python3 checks.py                                # sanity checks on a tiny matrix (seconds)
python3 experiments.py                           # full comparison, writes results/ (~30 s)
python3 generate.py --name ratings_train.npy     # platform entry point -> output.npy
```

## Files

| File | What it does |
|---|---|
| `generate.py` | **Platform entry point.** Loads `--name`, fits the model in `best_config.json`, keeps real ratings, fills the gaps, saves `output.npy`. The two load/save lines are untouched. |
| `models.py` | All models, from global mean to matrix factorization, each explained in comments. |
| `experiments.py` | Tunes every model on a validation split, then scores each once on `ratings_test.npy`. Writes `best_config.json` and `results/`. |
| `checks.py` | Every model: right shape, no NaN, ratings in [0.5, 5], reproducible. |
| `best_config.json` | Model chosen by validation RMSE (currently ALS). |
| `results/` | `validation.csv` (every setting tried), `test.csv` (final scores), `support.csv` (error by number of ratings per movie), `run.log`. |

## Data

* `ratings_train.npy`, `ratings_test.npy`: 610 users × 4,980 movies, NaN = not rated.
  31,598 ratings each (≈1% of cells), never the same cell. Ratings 0.5–5 in half stars.
* Movies are very thinly rated: median 3 ratings per movie. Users: median 24.
* `namesngenre.npy`: title and genres of each movie (not used yet).
* The platform gives `generate.py` train + test merged (≈2% of cells) and scores a hidden eval set.

## Protocol

1. Hold out 20% of the training ratings (seed 2026) as a validation set.
2. Try every setting in `GRID` (experiments.py): fit on 80%, score on the 20%.
3. Lock the best setting per method, refit on all of train, score once on test.
4. The best method on validation goes into `best_config.json`.

Test scores are never used to choose anything.

## First results (`results/test.csv`)

| Method | Locked setting | Validation RMSE | Test RMSE | Exact accuracy | Time (s) |
|---|---|---:|---:|---:|---:|
| global mean | – | 1.0162 | 1.0369 | 12.9% | 0.0 |
| user mean | – | 0.9378 | 0.9448 | 21.7% | 0.0 |
| movie mean | – | 0.9980 | 0.9996 | 22.1% | 0.0 |
| bias (μ + b_u + b_i) | reg 3 | 0.8825 | 0.8742 | 24.5% | 0.2 |
| user KNN | k 30, damping 3 | 0.8811 | **0.8715** | 24.5% | 0.3 |
| ALS matrix factorization | 32 factors, reg_bias 3, reg_factor 10 | **0.8804** | 0.8718 | 24.6% | 1.6 |

Exact accuracy = predictions rounded to the nearest half star that hit the true rating.
