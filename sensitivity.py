"""Optional MCMC sensitivity experiment, strictly on a TRAIN-derived validation set.

Run: python3 sensitivity.py [--quick]
Writes results/bpmf_sensitivity.csv. This checks stability only: comparing RMSE
and predictions does not prove posterior convergence or good mixing.
"""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
from models import predict
from experiments import split_observed, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--sampled-prior', action='store_true')
    args = parser.parse_args()
    train = np.load('ratings_train.npy')
    fit, val = split_observed(train, seed=2026)
    settings = [(2, 4), (4, 8)] if args.quick else [(10, 30), (20, 80), (40, 160)]
    rows, predictions = [], []
    for burnin, samples in settings:
        for seed in (0, 1) if args.quick else (0, 1, 2):
            config = {'method': 'bpmf', 'side_info': True, 'chains': 1,
                      'burnin': burnin, 'samples': samples, 'seed': seed,
                      'prior_mode': 'sampled' if args.sampled_prior else 'plug_in'}
            if args.quick:
                config['factors'] = 4
            start = time.perf_counter()
            pred = predict(fit, config)
            seconds = time.perf_counter() - start
            predictions.append((burnin, samples, seed, pred))
            rows.append({'burnin': burnin, 'samples': samples, 'seed': seed,
                         'prior_mode': config['prior_mode'],
                         'validation_rmse': round(rmse(pred, val), 6),
                         'seconds': round(seconds, 2)})
            print(rows[-1], flush=True)
    known = np.isfinite(val)
    for row, (_, _, _, pred) in zip(rows, predictions):
        alternatives = [other[3][known] for other in predictions
                        if other[0:2] == (row['burnin'], row['samples']) and other[2] != row['seed']]
        row['max_prediction_difference_between_seeds'] = (
            round(float(max(np.sqrt(np.mean((pred[known] - x) ** 2)) for x in alternatives)), 6)
            if alternatives else '')
    Path('results').mkdir(exist_ok=True)
    path = Path('results/bpmf_sensitivity_smoke.csv' if args.quick else 'results/bpmf_sensitivity.csv')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('Saved', path)


if __name__ == '__main__':
    main()
