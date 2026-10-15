"""Simulate the platform and choose the submitted model.  Run:  python3 platform_sim.py  (~3 min)

Why: the platform trains on train+test MERGED (~63k ratings, ~2% of cells) and scores a
hidden set. Our experiments.py trains on 25k-31k ratings. Models behave differently at
different densities, so we pick the submitted model at the platform's density.

Protocol
1. Merge ratings_train.npy and ratings_test.npy.
2. Three random splits (seeds 0, 1, 2): hide 20% of the merged ratings, fit on the other 80%.
3. Split 0 is used ONLY to fit the blend weights (least squares on its hidden ratings).
4. Splits 1 and 2 score every model and the blend with those fixed weights.
5. The blend config is written to best_config.json (read by generate.py).

Outputs: results/platform_sim.csv, best_config.json
"""
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
    return float(np.sqrt(np.mean((predictions[known] - truth[known]) ** 2)))


def merged_split(merged, seed, fraction=0.2):
    rng = np.random.default_rng(seed)
    cells = np.flatnonzero(np.isfinite(merged))
    held_out = rng.choice(cells, round(fraction * cells.size), replace=False)
    fit, hidden = merged.copy(), np.full_like(merged, np.nan)
    hidden.flat[held_out] = merged.flat[held_out]
    fit.flat[held_out] = np.nan
    return fit, hidden


def main():
    train, test = np.load('ratings_train.npy'), np.load('ratings_test.npy')
    merged = np.where(np.isfinite(train), train, test)
    rows, weights = [], None
    for seed in (0, 1, 2):
        fit, hidden = merged_split(merged, seed)
        preds = {}
        for name, config in MODELS.items():
            start = time.perf_counter()
            preds[name] = predict(fit, config)
            seconds = time.perf_counter() - start
            rows.append({'split': seed, 'model': name, 'rmse': round(rmse(preds[name], hidden), 4),
                         'seconds': round(seconds, 1)})
            print(rows[-1], flush=True)
        known = np.isfinite(hidden)
        if seed == 0:   # learn blend weights here only
            X = np.column_stack([preds[n][known] for n in BLEND_PARTS])
            weights = np.linalg.lstsq(X, hidden[known], rcond=None)[0]
            print('blend weights', dict(zip(BLEND_PARTS, np.round(weights, 3))), flush=True)
        else:
            blended = np.clip(sum(w * preds[n] for w, n in zip(weights, BLEND_PARTS)), 0.5, 5.0)
            rows.append({'split': seed, 'model': 'blend', 'rmse': round(rmse(blended, hidden), 4), 'seconds': ''})
            print(rows[-1], flush=True)

    Path('results').mkdir(exist_ok=True)
    with open('results/platform_sim.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print('\nMean RMSE on splits 1-2 (never used for any choice):')
    for name in list(MODELS) + ['blend']:
        scores = [r['rmse'] for r in rows if r['model'] == name and r['split'] != 0]
        print(f'  {name:11s} {np.mean(scores):.4f}')

    config = {'method': 'blend', 'parts': [MODELS[n] for n in BLEND_PARTS],
              'weights': [round(float(w), 3) for w in weights]}
    Path('best_config.json').write_text(json.dumps(config, indent=2) + '\n')
    print('\nbest_config.json ->', config)


if __name__ == '__main__':
    main()
