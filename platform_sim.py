"""Leakage-controlled A1 platform-density simulation.

The course platform fits on *all* observed train+test ratings and evaluates
on a separate private set. This script instead uses one frozen, disjoint
local partition: 64% fit, 16% blend-calibration, 20% untouched final evaluation.

* Fit base models on FIT and learn blend weight(s) on CALIBRATION only.
* Refit fixed models on FIT+CALIBRATION and score once on EVALUATION.
* Write an optional submission config ONLY when --write-config is supplied.

IMPORTANT: The official ratings_test.npy has already been used to choose the
original team's approach. Because this simulation merges it with train, these
results are model-development evidence, not an untouched official test set.
Use the course's hidden eval to assess final generalization.

Usage:
    python3 platform_sim.py
    python3 platform_sim.py --quick    # fast interface/data-leakage smoke test
    python3 platform_sim.py --write-config
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from models import predict

MODELS = {
    'bias': {'method': 'bias', 'reg': 3.0},
    'user_knn': {'method': 'user_knn', 'k': 30, 'damping': 3.0},
    'als_mf': {'method': 'als_mf', 'factors': 32, 'reg_bias': 3.0, 'reg_factor': 10.0},
    'bpmf_plain': {'method': 'bpmf', 'side_info': False},
    'bpmf_side': {'method': 'bpmf', 'side_info': True},
}
BLEND_PARTS = ['bpmf_side', 'user_knn']


def rmse(predictions, truth):
    known = np.isfinite(truth)
    if not np.any(known):
        raise ValueError('No held-out ratings to score')
    return float(np.sqrt(np.mean((predictions[known] - truth[known]) ** 2)))


def three_way_split(data, seed=2026, evaluation_fraction=0.2, calibration_fraction=0.2):
    """Return (fit, calibration, evaluation), with no rating in two partitions.

    calibration_fraction is a fraction of non-evaluation observations; default
    fractions of ALL observed ratings are 0.64 / 0.16 / 0.20.
    """
    rng = np.random.default_rng(seed)
    cells = rng.permutation(np.flatnonzero(np.isfinite(data)))
    n_eval = round(len(cells) * evaluation_fraction)
    n_cal = round((len(cells) - n_eval) * calibration_fraction)
    evaluation_cells = cells[:n_eval]
    calibration_cells = cells[n_eval:n_eval+n_cal]
    fit_cells = cells[n_eval+n_cal:]
    outs = []
    for selected in (fit_cells, calibration_cells, evaluation_cells):
        x = np.full_like(data, np.nan, dtype=float)
        x.flat[selected] = data.flat[selected]
        outs.append(x)
    return tuple(outs)


def fit_simplex_blend(p1, p2, truth):
    """Fit a nonnegative two-model blend with weights adding to exactly one.

    min_{0<=w<=1} || w*p1 + (1-w)*p2 - y ||^2
    """
    known = np.isfinite(truth)
    delta = (p1 - p2)[known]
    target = (truth - p2)[known]
    denominator = float(delta @ delta)
    w = float(np.clip((delta @ target) / denominator, 0, 1)) if denominator > 0 else 0.5
    return [w, 1.0 - w]


def configs(quick=False, sampled=False):
    out = {name: dict(cfg) for name, cfg in MODELS.items()}
    for name in ('bpmf_plain', 'bpmf_side'):
        if sampled:
            out[name]['prior_mode'] = 'sampled'
        if quick:
            out[name].update(factors=4, chains=1, burnin=2, samples=3)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--quick', action='store_true', help='Fast smoke test, NOT model-quality evidence')
    parser.add_argument('--sampled-prior', action='store_true', help='Sample feature regression W (slower / different model)')
    parser.add_argument('--write-config', action='store_true', help='Explicitly replace best_config.json')
    args = parser.parse_args()
    train = np.load('ratings_train.npy')
    test = np.load('ratings_test.npy')
    if train.shape != test.shape or np.any(np.isfinite(train) & np.isfinite(test)):
        raise ValueError('Expected same-shape, disjoint train/test rating matrices')
    merged = np.where(np.isfinite(train), train, test)
    fit, cal, evaluation = three_way_split(merged, seed=args.seed)
    train_for_eval = np.where(np.isfinite(cal), cal, fit)
    assert not np.any(np.isfinite(fit) & np.isfinite(cal))
    assert not np.any(np.isfinite(fit) & np.isfinite(evaluation))
    assert not np.any(np.isfinite(cal) & np.isfinite(evaluation))
    for name, data in (('fit', fit), ('calibration', cal), ('evaluation', evaluation)):
        print(name, int(np.isfinite(data).sum()), flush=True)

    model_cfgs = configs(quick=args.quick, sampled=args.sampled_prior)
    calibration_preds = {}
    for name in BLEND_PARTS:
        print('calibrate', name, flush=True)
        calibration_preds[name] = predict(fit, model_cfgs[name])
    weights = fit_simplex_blend(*(calibration_preds[k] for k in BLEND_PARTS), cal)
    print('Frozen weights', dict(zip(BLEND_PARTS, weights)), flush=True)

    rows = []
    final_preds = {}
    for name, cfg in model_cfgs.items():
        start = time.perf_counter()
        final_preds[name] = predict(train_for_eval, cfg)
        secs = time.perf_counter() - start
        rows.append({'model': name, 'eval_rmse': round(rmse(final_preds[name], evaluation), 5),
                     'seconds': round(secs, 2)})
        print(rows[-1], flush=True)
    blended = sum(w * final_preds[name] for w, name in zip(weights, BLEND_PARTS))
    score = rmse(np.clip(blended, 0.5, 5.0), evaluation)
    rows.append({'model': 'blend', 'eval_rmse': round(score, 5), 'seconds': ''})
    print(rows[-1], flush=True)
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    path = results_dir / ('platform_sim_smoke.csv' if args.quick else 'platform_sim_corrected.csv')
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('Results saved:', path)
    if args.write_config:
        if args.quick:
            raise ValueError('Refusing to write a submission config from quick smoke-test settings')
        config = {'method': 'blend', 'parts': [model_cfgs[n] for n in BLEND_PARTS],
                  'weights': weights}
        Path('best_config.json').write_text(json.dumps(config, indent=2) + '\n')
        print('Updated best_config.json (new blend weights)')
    else:
        print('best_config.json unchanged. Use --write-config ONLY after reviewing results.')


if __name__ == '__main__':
    main()
