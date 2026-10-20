> **METADATA CORRECTION — 20 OCT 2026:** The instructor corrected the movie-to-metadata alignment on 15 October. All numeric experiments and blend weights in this historical document were obtained with the OLD, misaligned `namesngenre.npy` and are **not valid evidence for the corrected model**. Do not copy these old results into the final report. See `METADATA_MIGRATION.md` for the updated workflow and new results. The mathematical model descriptions remain relevant.

> **Revision note (October 2026):** This document describes the original experiments.
> The old `results/platform_sim.csv` uses overlapping random held-out sets and
> must not be described as an entirely untouched evaluation. The corrected
> protocol and new results are in `CHANGES_AND_VERIFICATION.md` and
> `results/platform_sim_corrected.csv`. The shipped model defaults to a
> **plug-in** feature-coefficient estimate (a partially Bayesian approximation),
> not a full Gibbs posterior over W. Set `prior_mode="sampled"` to sample W,
> but validate this different model before using it. Historical numerical values
> in this document have not been recomputed for that alternative.

# Method — Bayesian matrix factorization with side-information priors

Group 38 (LHOOQ) · DataLab A1 · Collaborative filtering

This document explains every step of the submitted model, why each step is there, and
what evidence supports it. All numbers come from scripts in this repository
(`experiments.py`, `platform_sim.py`, `ablation.py`); the CSV files are in `results/`.

---

## 1. Problem

We get a ratings matrix $R$ of 610 users × 4,980 movies. Ratings are 0.5–5 in half stars;
`NaN` means "not rated". The task is to predict the missing entries. The score is RMSE on
hidden ratings:

$$\text{RMSE} = \sqrt{\frac{1}{|T|}\sum_{(u,i)\in T} (\hat r_{ui} - r_{ui})^2}$$

— the typical size of an error, in stars.

**What makes this dataset hard**

| | train | train + test (platform input) |
|---|---|---|
| ratings | 31,598 | 63,196 |
| filled cells | 1.0% | 2.1% |
| median ratings per movie | 3 | ~6 |
| median ratings per user | 24 | ~47 |

Most movies have only a handful of ratings. Any model that learns a lot of parameters per
movie must be protected against over-fitting those few ratings. Every design choice below
is about **how much to trust thin evidence**.

## 2. Evaluation protocols

**(a) Local protocol** (`experiments.py`). Hold out 20% of `ratings_train.npy` for validation,
tune each method there, lock its settings, refit on all of train, score once on `ratings_test.npy`.
Test ratings are never used to choose anything.

**(b) Platform simulation** (`platform_sim.py`). The platform fits on train+test merged
(about twice as many ratings) and scores a hidden set. Model rankings and the best amount
of regularisation can change with data density, so we choose the submitted model under the
same conditions: merge train and test, hide 20%, fit on the remaining 80% (≈50k ratings).
We use three random splits: split 0 only to fit the blend weights, splits 1 and 2 only to
report scores.

## 3. Starting point: biases and matrix factorization

**Bias model.** $\hat r_{ui} = \mu + b_u + b_i$: global mean + how generous user $u$ is +
how well liked movie $i$ is. The biases are fitted by regularised least squares; the penalty
acts like a few imaginary average ratings, so a movie with one 5-star rating is not treated as
a masterpiece.

**Matrix factorization (MF).** Add an interaction term:

$$\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i$$

Each movie has a vector $q_i$ of $k$ hidden "taste dials" (e.g. how much action, how dark),
each user has a vector $p_u$ of preferences for those dials, and $p_u^\top q_i$ measures how
well the movie fits the user. The dials are never named; the model invents whichever explain
the data best (Koren, Bell & Volinsky, 2009). We fit it with alternating least squares (ALS):
with all movie vectors fixed, each user's $(b_u, p_u)$ is an ordinary ridge regression, and
vice versa.

ALS produces a single "best guess" for every vector, and its regularisation strengths
must be tuned by hand. Both are weak points when most movies have 1–5 ratings.

## 4. Bayesian probabilistic matrix factorization (BPMF)

BPMF (Salakhutdinov & Mnih, 2008) keeps the same prediction formula but treats every
vector as **uncertain**.

### 4.1 Model

For user $u$ write $U_u = [b_u, p_u]$, and for movie $i$ write $V_i = [b_i, q_i]$
(dimension $d = k+1$).

* **Likelihood** — observed ratings are noisy:
  $r_{ui} \sim \mathcal N(\mu + b_u + b_i + p_u^\top q_i,\ \alpha^{-1})$, where $\alpha$ is the noise precision.
* **Prior** — before seeing any rating, each vector is Gaussian around a centre:
  $U_u \sim \mathcal N(m_u, \Lambda_U^{-1})$, $V_i \sim \mathcal N(m_i, \Lambda_V^{-1})$.
  In plain BPMF all users share one centre and all movies share one centre. §5 changes this.
* **Hyper-priors** — the precision matrices $\Lambda_U, \Lambda_V$ (how tightly vectors cluster
  around their centres = the regularisation strength) and the noise precision $\alpha$ are
  themselves unknown and get Wishart / Gamma priors. **The model learns its own regularisation.**

### 4.2 Prediction = average over plausible models

Instead of one best set of vectors, BPMF predicts with the posterior average

$$\hat r_{ui} = \mathbb E\big[\mu + b_u + b_i + p_u^\top q_i \mid \text{observed ratings}\big]
\approx \frac{1}{S}\sum_{s=1}^{S} \hat r^{(s)}_{ui}$$

over $S$ samples drawn from the posterior. For a movie with two ratings, many dial settings
fit equally well; averaging over them gives a cautious prediction close to the movie's bias,
while well-rated movies get confident, specific predictions. This happens automatically,
without tuning a regularisation constant.

### 4.3 Gibbs sampling (how the samples are drawn)

Gibbs sampling draws from a complicated joint distribution by repeatedly drawing each block
of unknowns from its distribution **given all the others**. Each block here has a closed form.
One sweep (function `_bpmf_chain` in `models.py`):

1. **Prior centres and precisions.** Given the current user vectors, compute their prior
   centres (§5) and draw $\Lambda_U$ from its posterior: a Wishart distribution whose scale
   depends on how far the vectors sit from their centres. Same for movies.
2. **All users.** With movies fixed, each user's $U_u$ is a Bayesian linear regression
   (features of movie $j$: $[1, q_j]$, target: $r_{uj} - \mu - b_j$). Its posterior is Gaussian:
   $$P_u = \Lambda_U + \alpha \sum_{j \in \text{rated}(u)} x_j x_j^\top,\qquad
   \text{mean}_u = P_u^{-1}\Big(\Lambda_U m_u + \alpha \sum_j x_j\,(r_{uj}-\mu-b_j)\Big)$$
   Draw $U_u \sim \mathcal N(\text{mean}_u, P_u^{-1})$.
   This is ALS's ridge regression, except that we **draw** a vector instead of taking the best one.
3. **All movies.** Same with users fixed.
4. **Noise.** Draw $\alpha$ from its Gamma posterior given the current errors on observed ratings.
5. Compute the full prediction matrix and add it to a running sum (after warm-up).

The first `burnin` = 20 sweeps are discarded (the sampler is still moving away from its start).
The next `samples` = 80 are averaged.

**Several chains.** We run 3 independent chains from different random starts and average them.
Separate chains explore different parts of the posterior; averaging 3 chains of 80 samples
was clearly better than 1 longer chain in our tests.

### 4.4 Settings

`factors = 20`, `chains = 3`, `burnin = 20`, `samples = 80`. Chosen on platform-simulation
split 0 (time ↔ accuracy trade-off; measured before the decade feature was added):

| factors | chains × samples | RMSE (split 0) | time |
|---:|---|---:|---:|
| 10 | 3 × 80 | 0.8229 | 13 s |
| **20** | **3 × 80** | **0.8197** | **23 s** |
| 30 | 3 × 80 | 0.8201 | 41 s |
| 20 | 6 × 40 | 0.8224 | 28 s |
| 50 | 2 × 80 | 0.8214 | 64 s |

Unlike ALS, BPMF does not degrade with more factors; the extra cost brings no gain beyond 20.

### 4.5 Implementation

The per-user posteriors are computed for all users at once. $\sum_j x_j x_j^\top$ for every
user is one matrix product (`rated @ outer`), and the 610 (or 4,980) small systems are solved in
one batched `np.linalg.solve` / `cholesky` call. One chain takes ~8 s, the full model ~24 s on
platform-sized input, NumPy only. Initialisation: biases from the bias model, small random dials.

## 5. Our addition: side-information priors

In plain BPMF the prior pulls every movie towards the **same** average movie. A movie with
one rating therefore starts from "average movie" plus one noisy data point.

We centre each vector on what **side information** predicts instead:

$$m_i = W_V^\top f_i \quad (\text{movies}),\qquad m_u = W_U^\top g_u \quad (\text{users})$$

**Movie features $f_i$** (from the corrected `namesngenre2.npy` and the input matrix)
* 20 genre indicators (`Comedy`, `Drama`, …, including `(no genres listed)`)
* release decade (one-hot, parsed from the title, e.g. "Toy Story (1995)")
* $\log(1 + \text{number of ratings})$ — popularity

**User features $g_u$** (from the input matrix only)
* genre mix of the movies the user **chose to rate** (share of Comedy, Drama, …)
* $\log(1 + \text{number of ratings})$ — activity

The user features rely on the fact that *which* movies someone rates is itself informative,
not only the scores: ratings are missing not at random (Marlin & Zemel, 2009). The features use
only which cells are filled in the input, never hidden ratings.

**How $W$ is set.** At every sweep, $W$ is the ridge-regression fit of the current vectors on
the features (`_sample_prior`). The precision $\Lambda$ is then drawn from the spread of the
vectors around these centres. If the features predict well, vectors cluster tightly around
the centres and the prior pulls hard; if they predict poorly, the prior becomes weak on its own.
This is a simplification of Porteous, Asuncion & Welling (2010), who also sample $W$; we use
its point estimate.

**Effect — ablation** (`ablation.py`, platform simulation, splits 1 and 2, 3 chains each)

| prior centre | split 1 | split 2 | mean |
|---|---:|---:|---:|
| none (plain BPMF) | 0.8400 | 0.8411 | 0.8405 |
| + movie metadata (genres, decade) | 0.8398 | 0.8370 | 0.8384 |
| + popularity (log counts) | 0.8373 | 0.8372 | 0.8373 |
| + user behaviour (genre mix) | 0.8398 | 0.8414 | 0.8406 |
| **all together** | **0.8351** | **0.8326** | **0.8338** |

* Every information source alone gives at most ~0.003; together they give 0.0067, on both splits.
  The sources are complementary: genres describe what a movie is, popularity separates
  cult films from blockbusters, and the user's genre mix says which part of the catalogue
  the user's tastes come from.
* Run-to-run variation (different random seeds, same split) is ~0.001 for a single chain,
  so the full-model gain is well above noise.
* In ALS, the same genre prior gave no gain (0.8364 → 0.8376). The Bayesian version can weight
  the prior by how informative it is; ALS cannot without extra tuning.

## 6. Blend with user-KNN

The final prediction is

$$\hat r = 0.81 \cdot \hat r_{\text{BPMF+side}} + 0.195 \cdot \hat r_{\text{user-KNN}}$$

User-KNN (lecture slides 8–9: Pearson-style similarity on baseline residuals, k = 30, with a
damping term) captures local "people like you" patterns that a low-dimensional factor model
smooths away, so its errors are partly different. The weights are least-squares fitted on the
hidden ratings of platform-simulation split 0 and then **fixed**; splits 1 and 2 report the original exploratory result (not strictly independent because the held-out sets overlap)
(Sill et al., 2009, for the general principle of linear blending).

## 7. Results

**Platform simulation** (`results/platform_sim.csv`, mean of legacy splits 1–2; NOT an untouched final evaluation)

| model | RMSE | time |
|---|---:|---:|
| bias | 0.8561 | 0.2 s |
| user-KNN | 0.8491 | 0.5 s |
| ALS (32 factors) | 0.8479 | 1.6 s |
| BPMF, plain | 0.8405 | 23 s |
| BPMF + side information | 0.8338 | 23 s |
| **blend (submitted)** | **0.8317** | **24 s** |

**Local protocol** (`results/test.csv`, fit on train, scored on test)

| model | test RMSE |
|---|---:|
| global mean | 1.0369 |
| bias | 0.8742 |
| user-KNN | 0.8715 |
| ALS | 0.8718 |
| BPMF, plain | 0.8637 |
| BPMF + side information | 0.8585 |

Both protocols give the same ranking. The local protocol has half as much training data,
so all errors are higher; the gains from BPMF and from side information are larger there,
consistent with both helping most when data is scarce.

## 8. Limitations

* Simulated scores cannot be mapped exactly to platform scores (more training data there, a
  different hidden set). Rankings are what we trust, not absolute values.
* Only 2–3 random splits; differences below ~0.002 should not be over-interpreted.
* $W$ is a point estimate (not sampled); the Wishart hyper-prior uses fixed simple defaults.
* Exact-rating accuracy is low (~25% after rounding) for every method: continuous predictions
  minimise RMSE, not exact hits.
* Release year is parsed from titles; 0.1% of titles have none and get the median year.

## 9. References

* Salakhutdinov, R. & Mnih, A. (2008). *Bayesian probabilistic matrix factorization using Markov
  chain Monte Carlo.* ICML 2008. — the BPMF model and Gibbs sampler (our §4).
* Salakhutdinov, R. & Mnih, A. (2007). *Probabilistic matrix factorization.* NeurIPS 2007. —
  the non-Bayesian version; why MAP estimates over-fit sparse users and movies.
* Porteous, I., Asuncion, A. & Welling, M. (2010). *Bayesian matrix factorization with side
  information and Dirichlet process mixtures.* AAAI 2010. — side information in the BPMF prior
  (closest prior work to our §5).
* Adams, R. P., Dahl, G. E. & Murray, I. (2010). *Incorporating side information in probabilistic
  matrix factorization with Gaussian processes.* UAI 2010.
* Koren, Y., Bell, R. & Volinsky, C. (2009). *Matrix factorization techniques for recommender
  systems.* IEEE Computer 42(8). — biases + factors, ALS vs SGD (our §3).
* Koren, Y. (2008). *Factorization meets the neighborhood: a multifaceted collaborative filtering
  model.* KDD 2008. — combining factor and neighbourhood models, implicit feedback.
* Marlin, B. M. & Zemel, R. S. (2009). *Collaborative prediction and ranking with non-random
  missing data.* RecSys 2009. — why *which* items are rated carries information.
* Sill, J., Takács, G., Mackey, L. & Lin, D. (2009). *Feature-weighted linear stacking.*
  arXiv:0911.0460. — blending models (our §6).
* Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer, §2.3 and §3.3. —
  Gaussian/Wishart distributions and Bayesian linear regression (the per-user update).
* Harper, F. M. & Konstan, J. A. (2015). *The MovieLens datasets: history and context.*
  ACM TiiS 5(4). — the data source.
