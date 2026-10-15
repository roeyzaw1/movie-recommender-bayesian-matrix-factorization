"""Compare all models with a clean protocol.  Run:  python3 experiments.py

Protocol (why: never choose settings by looking at the test score)
1. Split the OBSERVED ratings of ratings_train.npy: 80% "fit", 20% "validation".
2. For every method, try each setting in the grid: fit on "fit", score on "validation".
3. Keep the best setting per method (by validation RMSE). This is now locked.
4. Refit each locked method on ALL of ratings_train.npy, score once on ratings_test.npy.
5. The overall best-on-validation method is written to best_config.json for generate.py.

Metrics
* RMSE: typical size of an error, in stars (lower is better). Main platform metric.
* Exact accuracy: share of ratings hit exactly after rounding to the nearest half star.
  The platform reports this too.
* Seconds: fitting time on this machine (the platform's timing will differ).

Outputs (folder results/): validation.csv, test.csv, support.csv
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
from models import predict

GRID = {
    'global_mean': [{}],
    'user_mean': [{}],
    'item_mean': [{}],
    'bias': [{'reg': r} for r in (1, 2, 3, 5, 10)],
    'user_knn': [{'k': k, 'damping': d} for k in (10, 30, 100) for d in (0.0, 1.0, 3.0)],
    'als_mf': [{'factors': f, 'reg_bias': 3.0, 'reg_factor': r}
               for f in (8, 32) for r in (5.0, 10.0, 20.0)],
}


def rmse(predictions, truth):
    known = np.isfinite(truth)
    return float(np.sqrt(np.mean((predictions[known] - truth[known]) ** 2)))


def exact_accuracy(predictions, truth):
    known = np.isfinite(truth)
    rounded = np.round(predictions[known] * 2) / 2      # nearest half star
    return float(np.mean(rounded == truth[known]))


def split_observed(table, fraction=0.2, seed=2026):
    """Move a random `fraction` of the observed ratings into a separate validation matrix."""
    rng = np.random.default_rng(seed)
    cells = np.flatnonzero(np.isfinite(table))
    held_out = rng.choice(cells, round(fraction * cells.size), replace=False)
    fit, validation = table.copy(), np.full_like(table, np.nan)
    validation.flat[held_out] = table.flat[held_out]
    fit.flat[held_out] = np.nan
    return fit, validation


def timed(table, config):
    start = time.perf_counter()
    predictions = predict(table, config)
    return predictions, time.perf_counter() - start


def save_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    out = Path('results')
    out.mkdir(exist_ok=True)
    train = np.load('ratings_train.npy')
    test = np.load('ratings_test.npy')
    assert not np.any(np.isfinite(train) & np.isfinite(test)), 'train and test overlap'

    # Steps 1-3: tune on the validation split only.
    fit, validation = split_observed(train)
    validation_rows, locked = [], {}
    for method, settings in GRID.items():
        for params in settings:
            config = {'method': method, **params}
            predictions, seconds = timed(fit, config)
            score = rmse(predictions, validation)
            validation_rows.append({'config': json.dumps(config), 'validation_rmse': round(score, 4),
                                    'seconds': round(seconds, 2)})
            print(f'validation  {json.dumps(config):70s} RMSE {score:.4f}  ({seconds:.1f}s)', flush=True)
            if method not in locked or score < locked[method][0]:
                locked[method] = (score, config)
    save_csv(out / 'validation.csv', validation_rows)

    best_score, best_config = min(locked.values(), key=lambda pair: pair[0])
    Path('best_config.json').write_text(json.dumps(best_config, indent=2) + '\n')
    print('\nChosen for generate.py:', best_config, f'(validation RMSE {best_score:.4f})\n')

    # Step 4: one final score per locked method on the supplied test set.
    test_rows, test_predictions = [], {}
    for method, (val_score, config) in locked.items():
        predictions, seconds = timed(train, config)
        test_predictions[method] = predictions
        row = {'method': method, 'config': json.dumps(config), 'validation_rmse': round(val_score, 4),
               'test_rmse': round(rmse(predictions, test), 4),
               'test_exact_accuracy': round(exact_accuracy(predictions, test), 4),
               'seconds': round(seconds, 2)}
        test_rows.append(row)
        print(f"test        {method:12s} RMSE {row['test_rmse']:.4f}  exact {row['test_exact_accuracy']:.3f}"
              f"  ({seconds:.1f}s)", flush=True)
    save_csv(out / 'test.csv', test_rows)

    # Where do models fail?  Test RMSE split by how many training ratings the movie has.
    ratings_per_movie = np.isfinite(train).sum(axis=0)
    support_rows = []
    for low, high in ((1, 5), (5, 20), (20, 10 ** 6)):
        subset = test.copy()
        subset[:, (ratings_per_movie < low) | (ratings_per_movie >= high)] = np.nan
        for method, predictions in test_predictions.items():
            support_rows.append({'movie_train_ratings': f'{low}-{high - 1}' if high < 10 ** 6 else f'{low}+',
                                 'n_test': int(np.isfinite(subset).sum()), 'method': method,
                                 'test_rmse': round(rmse(predictions, subset), 4)})
    save_csv(out / 'support.csv', support_rows)


if __name__ == '__main__':
    main()
