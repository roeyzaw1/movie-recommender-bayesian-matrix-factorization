"""Additional independent mathematical and integration tests from the code audit.

Run: python3 -m unittest -v audit_regression_tests.py
Rating arrays are optional; when they exist, their integrity is also checked.
"""
import unittest
from pathlib import Path
import numpy as np
from models import _sample_side, _sample_prior, movie_metadata, EXPECTED_MOVIES, predict
from experiments import split_observed
from platform_sim import fit_simplex_blend, three_way_split
import ablation


class MathematicalAuditTests(unittest.TestCase):
    def test_gaussian_posterior_batched_draw_matches_independent_reference(self):
        rng = np.random.default_rng(17)
        n_entities, n_items, factors = 5, 7, 3
        other = rng.normal(size=(n_items, factors + 1))
        centre = rng.normal(size=(n_entities, factors + 1))
        a = rng.normal(size=(factors + 1, factors + 1))
        precision = a @ a.T + 0.7 * np.eye(factors + 1)
        rated = (rng.uniform(size=(n_entities, n_items)) > 0.45).astype(float)
        targets = rated * rng.normal(size=(n_entities, n_items))
        alpha = 1.7
        got = _sample_side(np.random.default_rng(2026), rated, targets, other,
                           centre, precision, alpha)
        z = np.random.default_rng(2026).standard_normal((n_entities, factors + 1, 1))
        x = np.column_stack([np.ones(n_items), other[:, 1:]])
        expected = np.zeros_like(got)
        for j in range(n_entities):
            ids = np.flatnonzero(rated[j])
            xj = x[ids]
            post_prec = precision + alpha * (xj.T @ xj)
            rhs = precision @ centre[j] + alpha * (xj.T @ targets[j, ids])
            mean = np.linalg.solve(post_prec, rhs)
            expected[j] = mean + np.linalg.solve(np.linalg.cholesky(post_prec).T, z[j])[:, 0]
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    def test_matrix_normal_integrated_precision_scatter_identity(self):
        rng = np.random.default_rng(40)
        features = rng.normal(size=(11, 4))
        vec = rng.normal(size=(11, 3))
        reg = features.T @ features + np.eye(features.shape[1])
        w = np.linalg.solve(reg, features.T @ vec)
        residual = vec - features @ w
        scatter = residual.T @ residual + w.T @ w
        exact = vec.T @ (np.eye(len(vec)) - features @ np.linalg.solve(reg, features.T)) @ vec
        np.testing.assert_allclose(scatter, exact, rtol=1e-12, atol=1e-12)
        centres, precision = _sample_prior(np.random.default_rng(7), vec, features, 'sampled')
        self.assertEqual(centres.shape, vec.shape)
        self.assertGreater(np.linalg.eigvalsh(precision).min(), 0)

    def test_ablation_module_loads_and_its_split_is_disjoint(self):
        self.assertEqual(len(ablation.VARIANTS), 5)
        a = np.arange(72., dtype=float).reshape(8, 9)
        a[::2, ::2] = np.nan
        fit, held = split_observed(a, fraction=0.2, seed=2)
        self.assertFalse(np.any(np.isfinite(fit) & np.isfinite(held)))
        self.assertEqual(np.isfinite(fit).sum() + np.isfinite(held).sum(), np.isfinite(a).sum())

    def test_als_zero_latent_factors_matches_bias_coordinate_descent(self):
        from models import als_mf, bias
        data = np.array([[5., np.nan, 2., 1.],
                         [np.nan, 4., np.nan, 3.],
                         [4., np.nan, np.nan, np.nan]])
        np.testing.assert_allclose(als_mf(data, factors=0, reg_bias=3.0, epochs=15),
                                   bias(data, reg=3.0), rtol=1e-12, atol=1e-12)

    def test_knn_no_shared_ratings_reverts_to_bias(self):
        from models import user_knn, bias
        data = np.array([[5., np.nan], [np.nan, 1.]])
        np.testing.assert_allclose(user_knn(data, k=1), bias(data), rtol=1e-12, atol=1e-12)

    def test_actual_dataset_optional_integration(self):
        root = Path(__file__).resolve().parent
        paths = [root / 'ratings_train.npy', root / 'ratings_test.npy']
        if not all(p.exists() for p in paths):
            self.skipTest('Rating arrays not present in this code-only ZIP')
        train, test = (np.load(p, allow_pickle=False) for p in paths)
        self.assertEqual(train.shape, (610, EXPECTED_MOVIES))
        self.assertEqual(test.shape, train.shape)
        self.assertEqual(int(np.isfinite(train).sum()), 31598)
        self.assertEqual(int(np.isfinite(test).sum()), 31598)
        self.assertFalse((np.isfinite(train) & np.isfinite(test)).any())
        features = movie_metadata(train.shape[1])
        self.assertEqual(features.shape, (EXPECTED_MOVIES, 27))
        self.assertTrue(np.isfinite(features).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
