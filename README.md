# Movie-rating prediction with Bayesian matrix factorisation

*Master IASD (Université Paris Dauphine-PSL), Data Science Lab, assignment A1.
Group project: Roey Zaworbach, [Maryam Shiran](https://github.com/MaryamShiran) and
[MohammadMahdi1128](https://github.com/MohammadMahdi1128).*

## The problem

610 users and 4,980 movies form a table of ratings from 0.5 to 5, and only about 1% of
it is filled in. The task is to predict the missing ratings, i.e. to guess how
much a user would like a movie they haven't rated. Predictions are scored with **RMSE**,
the typical size of the error in stars (lower is better), on ratings hidden by the course.

## What we did

1. **Simple baselines:** the global average, user and movie averages with biases,
   user-to-user nearest neighbours (KNN) and alternating least squares (ALS).
2. **Bayesian probabilistic matrix factorisation (BPMF):** each user and each movie is
   described by 20 hidden "taste" numbers, and a rating is roughly the match between
   the two. Instead of fitting one best answer, the model samples many plausible answers
   (Gibbs sampling) and averages them, which makes it much harder to overfit such a sparse table.
3. **Side information in the prior:** most movies have only a handful of ratings, so we
   also use what we know about each movie (genres, decade, popularity) and each user
   (genre mix, activity) to give the model a sensible starting guess for their hidden
   numbers. A movie with 2 ratings then borrows strength from similar movies. This idea
   follows Agarwal & Chen (2009) and Park et al. (2013).

## Results

| Model | RMSE (local held-out split) |
|---|---:|
| Movie/user bias | 0.860 |
| User KNN | 0.852 |
| ALS | 0.850 |
| Plain BPMF | 0.839 |
| **BPMF + side information** | **0.815** |

On the course's own hidden test set our submission scored **0.806 RMSE**.
The details are in [METHOD.md](METHOD.md), [report.pdf](report.pdf) and [slides.pdf](slides.pdf).

---

*The original course README follows.*

# DataLab A1 — Group 38 (LHOOQ), corrected metadata edition

**Independent code audit (20 October 2026):** See `AUDIT_REPORT.md`. One broken import in the ablation script has been repaired. The original BPMF prediction model and best configuration were left unchanged. Run `python3 -m unittest -v regression_tests.py audit_regression_tests.py` for both original and supplemental tests. With the instructor data, the independently checked train-to-test RMSE is **0.83923**; this is a different evaluation protocol from the corrected platform-density simulation.


This repository is updated for the instructor's **15 October 2026 metadata correction**.
The old `namesngenre.npy` had a positional indexing error and is removed.
`namesngenre2.npy` and `movie_ids.npy` are the official corrected files.
Every metadata row `j` describes movie `movie_ids[j]` and ratings column `j`.
Do **not** use movie title for joins or deduplication.

## Use and reproduce

From a clean clone, place the unchanged `ratings_train.npy` and `ratings_test.npy`
(from the updated A1 ZIP) at repository root; these two large files are deliberately
excluded from this project ZIP. Then run:

```bash
python3 -m pip install -r requirements.txt
python3 verify_metadata.py
python3 regression_tests.py
python3 checks.py
python3 platform_sim.py                 # full corrected-metadata comparison
python3 platform_sim.py --write-config  # only if accepting freshly calibrated weights
python3 generate.py --name ratings_train.npy
```

`generate.py` writes `output.npy`, preserves all observed ratings, and loads the
supplied `--name` file for platform compatibility.

## Method

The submitted predictor is **BPMF with corrected metadata-dependent priors**.
The corrected-data calibration selected blend weights BPMF=1.0 and KNN=0.0,
so `best_config.json` now uses the mathematically identical stand-alone BPMF
predictor; the fitted blend weights are archived in
`results/calibrated_blend_config.json`. Baselines include global/user/movie means, biases, KNN,
ALS, and plain BPMF. See `METHOD.md` for math and model descriptions.

**CAUTION:** Any archived results from before the 15 October fix are invalid as
comparisons for the new metadata-driven model. They are kept in
`results/legacy_pre_metadata_correction/` for provenance, not for reporting.
Prior calibrated weights were **recomputed** with the corrected metadata.
See `METADATA_MIGRATION.md`.

Optional: `python3 experiments.py`, `python3 ablation.py`, and
`python3 sensitivity.py` also use corrected metadata through `models.py`.

For submission, also add `slides.pdf` and `report.pdf` to the repository root.

## Corrected metadata results (full local run, 20 October 2026)

Single disjoint 64% fit / 16% calibration / 20% evaluation split, seed 2026.
The original team's previous development has seen train+test, so this is *not*
a fully untouched test of the complete research process.

| Model | RMSE (corrected dataset) |
|---|---:|
| Bias | 0.85996 |
| User KNN | 0.85194 |
| ALS | 0.85038 |
| Plain BPMF | 0.83915 |
| BPMF + corrected side information | **0.81513** |
| Calibrated blend | **0.81513** |

The same script with the **old, misaligned metadata** scored BPMF+side=0.83352
and blend=0.83274. The new metadata substantially improves this local split,
but independent confirmation requires the instructor's hidden evaluation.
`results/platform_sim_corrected.csv` and
`results/metadata_corrected_simulation.log` contain the new full run.

**Important:** The Bayesian model remains in `prior_mode="plug_in"`; it draws
latent factors and hyperparameters, but uses a fitted (not sampled) regression
of metadata to prior means. Do not describe the feature coefficients as having
a full posterior distribution.
