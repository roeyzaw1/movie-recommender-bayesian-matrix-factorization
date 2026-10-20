# Rigorous independent code audit — 20 October 2026

## Scope and source

Compared the corrected Group 38 repository to the instructor's updated `datalab-a1.zip`, including both 610 × 4980 rating arrays, `namesngenre2.npy`, and `movie_ids.npy`. Reviewed `models.py`, generation interface, validation and experiment scripts, current config and result records. The original corrected copy was left untouched; this audited copy includes a minimal repair of an experiment script and supplementary tests.

## Verified outcomes

- Corrected metadata and movie IDs match the instructor's ZIP *byte for byte* (SHA-256 digests match the validation constants in `models.py`). Shapes are `(4980,2)` and `(4980,)`. The dataset is `(610,4980)`, with 31,598 observed ratings in **each** disjoint split.
- `generate.py --name ratings_train.npy` completed on the actual ratings and produced a `(610,4980)` finite dense matrix of predictions within `[0.5,5]`. All 31,598 supplied training ratings were preserved exactly. RMSE on the independently supplied `ratings_test.npy` was **0.8392288908471077** (this is a train→test local score, **not** the platform's hidden evaluation, and not the merged-density 0.81513 score).
- Checked the sparse ALS per-user and per-movie ridge equations, and established a zero-factor consistency test with the bias-coordinate-descent algorithm.
- Checked the factor Gibbs conditionals of BPMF against a separately computed multivariate Gaussian conditional, including its sampling covariance. Verified the integrated ridge-mean scatter identity used by the `prior_mode='sampled'` Wishart hyperparameter update.
- Checked KNN's no-shared-ratings fallback, the disjoint local partitioning, and metadata feature finiteness.
- All **11** current regression tests pass when the official rating arrays are present: five original tests plus six additional independent audit tests.

## Confirmed bug and repair

**`ablation.py` would not run.** It imported `merged_split` from `platform_sim.py`, but that function was removed when the platform simulation was rewritten. In this audited copy it calls `split_observed` from `experiments.py` (20% rating holdout), matching the intended experiment design. The output filename is now `results/ablation_corrected.csv` to prevent confusion with historical, invalid old-metadata results. Import and quick BPMF ablation branches on actual real-dimension data passed. The full ablation grid was **not** rerun. Hence no new numerical ablation claim should be made until it completes.

## Remaining limitations / cautions (not confirmed bugs)

1. **Partially Bayesian default.** `best_config.json` uses `prior_mode='plug_in'`: feature-regression coefficient `W` is re-estimated by ridge regression, not drawn from the full joint posterior. Its factor and precision sampling updates are internally consistent with that approximation. `prior_mode='sampled'` implements a valid blocked update for `W` and the precision, but is a different statistical model requiring new experiments.
2. **MCMC convergence is not proved.** Fixed `burnin=20`, `samples=80`, and three chains do not establish mixing or accurate posterior averages. Consider `sensitivity.py` with multiple seeds/longer chains. This is a statistical validation risk, not evidence of an incorrect Gibbs equation.
3. **Compute budget.** The full submitted BPMF is computationally substantial (300 Gibbs sweeps across 3 chains). On the current CPU its full end-to-end run required approximately a few minutes. Verify the platform's timeout and runtime scoring. `--quick` settings are only for smoke testing and must not replace the submission config without retuning.
4. **Metadata size fallback.** For synthetic inputs not containing exactly 4,980 movies, the model omits movie metadata. This is intentional for small smoke tests; do not mistake a successful smoke test with the wrong dimensions for evidence that the side-information model is active. The genuine `(610,4980)` data path was tested.
5. **Legacy results.** `METHOD.md` still preserves obsolete pre-correction figures (with a warning at the top), including a historic blending equation. Use the corrected report figures and do not copy the old sections verbatim.
6. **No independent reproduction of the corrected merged-density score.** The archived `0.81513` in `results/platform_sim_corrected.csv` comes from an earlier recorded full simulation, not a new independent simulation performed in this audit. It is **not** the same data split or training density as the independently checked `0.83923` train→test RMSE.
7. **External platform not available here.** Only the local required interface and published local data can be checked; the hidden evaluator's running-time limits, exact-rating accuracy definition, and hidden scores cannot be certified.

## Files changed in this audited copy

- `ablation.py`: repair the missing import and data partition; send output to `results/ablation_corrected.csv`.
- `audit_regression_tests.py`: add six independent regression/integration tests.
- `AUDIT_REPORT.md`: this report.
- `README.md`: add this audit's test/run instructions.

**The submitted predictive equations, metadata, main simulation, `generate.py`, and `best_config.json` were not changed.** No numerical results were recomputed and silently substituted in this ZIP.

## How to verify locally

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest -v regression_tests.py audit_regression_tests.py
python3 verify_metadata.py
python3 checks.py
```

Copy `ratings_train.npy` and `ratings_test.npy` from the updated instructor ZIP into this folder to run the rating-array integration check and the full entry point:

```bash
python3 -m unittest -v audit_regression_tests.py
python3 generate.py --name ratings_train.npy
```

The generated `output.npy` is the submitted interface output; it is not shipped in this source ZIP.

To **recompute** corrected ablations, run `python3 ablation.py` after copying rating data; the results should be independently scrutinized before the final report.
