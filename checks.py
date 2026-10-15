"""Quick sanity checks on a tiny matrix.  Run:  python3 checks.py"""
import numpy as np
from models import MODELS, predict

nan = np.nan
tiny = np.array([[5.0, nan, 2.0, nan],
                 [nan, 1.0, 3.0, nan],
                 [4.0, 2.0, nan, nan],
                 [nan, nan, nan, nan]])   # last user and last movie have no ratings at all

small_settings = {'user_knn': {'k': 2}, 'als_mf': {'factors': 2, 'epochs': 3}}

for method in MODELS:
    config = {'method': method, **small_settings.get(method, {})}
    p = predict(tiny, config)
    assert p.shape == tiny.shape, method
    assert np.isfinite(p).all(), f'{method} produced NaN/inf'
    assert p.min() >= 0.5 and p.max() <= 5.0, f'{method} out of range'
    assert np.array_equal(p, predict(tiny, config)), f'{method} not reproducible'
    print(f'ok  {method:12s} unseen user/movie -> {p[-1, -1]:.2f}')
print('PASS: shape, finite, in range, reproducible for every model')
