"""Which side information helps BPMF?  Run:  python3 ablation.py  (~5 min)

Legacy exploratory platform-density splits (seeds 1 and 2). These splits
overlap with the original team's model-selection data. These results are NOT
an untouched final held-out evaluation. Only the prior changes:
  none            : every user/movie pulled to the same average (plain BPMF)
  movie metadata  : movies pulled to what genres + decade predict
  popularity      : users/movies pulled to what their number of ratings predicts
  user behaviour  : users pulled to what the genre mix of their rated movies predicts
  all             : everything (the submitted model)
Output: results/ablation.csv
"""
import csv

import numpy as np
from models import _bpmf_chain, clip, movie_metadata, observed_mask
from platform_sim import merged_split, rmse


def features(table, use):
    rated = observed_mask(table).astype(float)
    n_user, n_item = rated.sum(axis=1), rated.sum(axis=0)
    meta = movie_metadata(table.shape[1])
    user_f, item_f = [np.ones(len(n_user))], [np.ones(len(n_item))]
    if 'popularity' in use:
        user_f.append(np.log1p(n_user)); item_f.append(np.log1p(n_item))
    if 'metadata' in use:
        item_f.append(meta)
    if 'behaviour' in use:
        user_f.append((rated @ meta) / np.maximum(n_user, 1)[:, None])
    return np.column_stack(user_f), np.column_stack(item_f)


VARIANTS = {
    'none': (),
    'movie metadata': ('metadata',),
    'popularity': ('popularity',),
    'user behaviour': ('behaviour',),
    'all': ('metadata', 'popularity', 'behaviour'),
}

if __name__ == '__main__':
    merged = np.where(np.isfinite(t := np.load('ratings_train.npy')), t, np.load('ratings_test.npy'))
    rows = []
    for seed in (1, 2):
        fit, hidden = merged_split(merged, seed)
        for name, use in VARIANTS.items():
            uf, itf = features(fit, use)
            runs = [_bpmf_chain(fit, 20, 20, 80, c, uf, itf) for c in range(3)]
            rows.append({'split': seed, 'prior': name, 'rmse': round(rmse(clip(np.mean(runs, 0)), hidden), 4)})
            print(rows[-1], flush=True)
    with open('results/ablation.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in VARIANTS:
        print(f"{name:15s} mean {np.mean([r['rmse'] for r in rows if r['prior'] == name]):.4f}")
