# Group 38: corrections and verification

This revision preserves the original implementation and old CSVs for reproducibility.
The **new recommended evaluation** is `platform_sim.py`; older numbers are historical
and must not be presented as clean final holdout estimates.

## 1. Evaluation leakage fixed

The original `platform_sim.py` independently hid 20% of the same merged ratings
in three random splits. Some ratings used for fitting blend weights in split 0
also appeared in the reported holdouts of splits 1/2. These later results were
not an untouched assessment of the blend-selection procedure.

The replacement uses a **single, disjoint 64/16/20 partition** (seed 2026):

- 64% FIT: base models fitted to select blend weights.
- 16% CALIBRATION: solely to fit the two-model convex blend weight.
- 20% EVALUATION: never used for fitting the blend weights; base models are refit
  on FIT+CALIBRATION before being evaluated here.

All reported model errors in `platform_sim_corrected.csv` use the same evaluation
partition. The blend is evaluated once using its **frozen** calibration weight.
If you try several schemes then choose based on this evaluation result, it becomes
another tuning set; seek a separate holdout or the instructor's hidden score.

**Important caveat:** The *team's broader development process* has already used
both train and test labels in prior tuning/experiments. These corrections cannot
make all previously seen labels unseen again. Only the instructor's hidden eval
is external to the entire development process. For an untouched research estimate,
collect independent test ratings or freeze a genuinely unseen set before any tuning.

## 2. Blend correction

Blend weights are now constrained to the 2-model simplex:
`w_BPMF ∈ [0, 1]` and `w_KNN = 1 - w_BPMF`. The closed-form constrained
least-squares solution is used instead of unconstrained `np.linalg.lstsq`.
The historical 0.81+0.195=1.005 was an unconstrained linear blend, not necessarily
a bug; constraining weights is a more easily interpreted design choice.

## 3. BPMF prior: explicit approximation / optional blocked Gibbs sampling

The original `_sample_prior` repeatedly plugged in a **Ridge estimate** for W
(metadata -> prior-centre regression), while sampling latent factors and a precision
matrix. This is an approximation, not a sampler from a full posterior over W.

The new `prior_mode='sampled'` implements a Gaussian conjugate hierarchical model:

- Λ ~ Wishart(d, I_d)
- W | Λ ~ MatrixNormal(0, I_feature, Λ^-1)
- X | W,Λ ~ MatrixNormal(FW, I_entity, Λ^-1)

It samples Λ given latent vectors after integrating out W, then samples W from
its matrix-normal conditional posterior. This corrects the missing uncertainty
in feature-coefficient estimates. This is a **different statistical model / sampler**
and must be tuned and evaluated separately.

`prior_mode='plug_in'` is kept as default for backward compatibility with the
originally validated local scores. The supplied config does not silently change
the algorithm when code is updated. To try the hierarchical alternative, run
`python3 platform_sim.py --sampled-prior` and inspect metrics before choosing it.

## 4. MCMC convergence remains an empirical question

`python3 sensitivity.py` compares different burn-in/sample settings and random
seeds on a validation split derived only from the train ratings. This produces
`results/bpmf_sensitivity.csv`. It is **not proof of convergence**: for formal
convergence diagnostics, save per-chain scalar traces and investigate mixing,
autocorrelation, effective sample size and R-hat.

## 5. Tests and steps to submit

1. Install `pip install -r requirements.txt`.
2. Run `python3 regression_tests.py` and `python3 checks.py`.
3. Run `python3 platform_sim.py` for corrected results. It does **not** modify
   `best_config.json` unless you explicitly add `--write-config`.
4. Compare the corrected blend against bias, ALS, KNN, and BPMF alone; decide
   model based on CALIBRATION or new independent evidence, not on the final
   evaluation partition alone.
5. If appropriate, rerun `python3 platform_sim.py --write-config` (slow) to
   write the newly calibrated blend weights; do not use `--quick` for selection.
6. Run `python3 generate.py --name ratings_train.npy` and check `output.npy`
   (shape, finite, observed entries preserved), then test once on the official
   platform after committing the needed files.
7. Update any graphs/report tables using the NEW saved results, and disclose
   both the approximate posterior and the differences among evaluation protocols.

Remember: the original CSVs have not been replaced or retroactively validated.

## 6. Results from a completed corrected full-size simulation

Executed with the original **plug-in** BPMF mode and the predefined blending
candidates (BPMF + side information, user KNN), seed 2026.

| Model | Evaluation RMSE | Runtime of evaluation-model fit |
|---|---:|---:|
| Bias | 0.85996 | 0.29 s |
| User KNN | 0.85194 | 0.99 s |
| ALS | 0.85038 | 2.67 s |
| BPMF (plain) | 0.83915 | 75.25 s |
| BPMF + side information | 0.83352 | 76.14 s |
| Frozen calibrated blend | **0.83274** | Reuses both models |

Calibration weights: **0.91256858 BPMF + 0.08743142 KNN**; sum exactly 1.
These numbers come from `results/platform_sim_corrected.csv`, generated locally,
**not from the instructor's hidden evaluation platform**. The improvement of the
blend over BPMF alone (0.00078) is small and is not evidence of a statistically
reliable improvement; do not overclaim it.

The `prior_mode='sampled'` option passed numerical smoke tests on real matrix
shapes, but its full-length predictive performance has **not** been revalidated.
Consequently the selected final config keeps the previously validated
`prior_mode='plug_in'` default.

## 7. Verification performed

- `python3 checks.py`: all 8 model interfaces passed their tiny-matrix tests.
- `python3 regression_tests.py`: 4 regression checks passed.
- `python3 platform_sim.py --quick`: disjoint split and output smoke tests passed.
- `python3 platform_sim.py --quick --sampled-prior`: optional W-sampling path ran
  on the real matrix (quick settings are **not** accuracy measurements).
- `python3 sensitivity.py --quick`: generated seed/sample-count diagnostics.
- `generate.py --name ratings_demo.npy` on a small example: generated `output.npy`
  with the right shape, finite predictions, and every original rating preserved.
- Full-size corrected `platform_sim.py --write-config`: completed, wrote the
  corrected CSV and updated `best_config.json`.

Still required before relying on a final official score: push the updated code to
your team's GitHub repository, allow the grading platform to execute it, and
check its hidden RMSE and runtime. No official score has been measured here.
