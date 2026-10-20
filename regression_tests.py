"""Fast regression tests for evaluation leakage and Bayesian prior sampling.

Run: python3 regression_tests.py
No ratings files are needed for these checks.
"""
import unittest
import numpy as np
from models import _sample_prior, predict, movie_metadata, EXPECTED_MOVIES
from platform_sim import three_way_split, fit_simplex_blend


class Tests(unittest.TestCase):
    def test_official_metadata_alignment(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parent
        meta = np.load(root / 'namesngenre2.npy', allow_pickle=False)
        ids = np.load(root / 'movie_ids.npy', allow_pickle=False)
        self.assertEqual(meta.shape, (EXPECTED_MOVIES, 2))
        self.assertEqual(len(set(map(int, ids))), EXPECTED_MOVIES)
        self.assertEqual(int(ids[0]), 1)
        self.assertEqual(meta[0, 0], 'Toy Story (1995)')
        features = movie_metadata(EXPECTED_MOVIES)
        self.assertEqual(features.shape[0], EXPECTED_MOVIES)
        self.assertTrue(np.isfinite(features).all())


    def test_disjoint_three_way_partition(self):
        data = np.full((19, 23), np.nan)
        data.flat[np.arange(350)[::3]] = (np.arange(len(np.arange(350)[::3])) % 10 + 1) / 2
        fit, cal, evaluation = three_way_split(data, seed=1234)
        masks = [np.isfinite(x) for x in (fit, cal, evaluation)]
        self.assertTrue(np.array_equal(sum(masks), np.isfinite(data).astype(int)))
        reconstructed = np.where(masks[0], fit, np.where(masks[1], cal, evaluation))
        self.assertTrue(np.array_equal(reconstructed, data, equal_nan=True))
        self.assertGreater(masks[1].sum(), 0)
        self.assertGreater(masks[2].sum(), 0)

    def test_blend_nonnegative_and_sums_to_one(self):
        rng = np.random.default_rng(0)
        truth = np.full((10, 5), np.nan)
        truth.flat[:30] = rng.uniform(1, 5, 30)
        a = np.full_like(truth, 5)
        b = np.full_like(truth, 1)
        weights = fit_simplex_blend(a, b, truth)
        self.assertAlmostEqual(sum(weights), 1)
        self.assertTrue(all(0 <= w <= 1 for w in weights))
        self.assertEqual(fit_simplex_blend(truth * 0 + 5, truth * 0 + 5, truth), [0.5, 0.5])

    def test_prior_sampling_and_determinism(self):
        rng = np.random.default_rng(7)
        v = rng.normal(size=(20, 4))
        f = np.column_stack([np.ones(20), rng.normal(size=(20, 3))])
        for mode in ('plug_in', 'sampled'):
            ans = _sample_prior(np.random.default_rng(17), v, f, mode)
            repeat = _sample_prior(np.random.default_rng(17), v, f, mode)
            for x, y in zip(ans, repeat):
                np.testing.assert_allclose(x, y)
                self.assertTrue(np.isfinite(x).all())
            self.assertGreater(np.linalg.eigvalsh(ans[1]).min(), 0)
        with self.assertRaises(ValueError):
            _sample_prior(rng, v, f, 'nonexistent')

    def test_models_on_small_matrix_with_sampled_prior(self):
        data = np.array([[5., np.nan, 3.], [np.nan, 4., 1.], [2., np.nan, np.nan]])
        for mode in ('plug_in', 'sampled'):
            c = dict(method='bpmf', prior_mode=mode, factors=2, burnin=2, samples=3, chains=1)
            out = predict(data, c)
            self.assertEqual(out.shape, data.shape)
            self.assertTrue(np.isfinite(out).all())
            self.assertTrue(np.all((out >= 0.5) & (out <= 5)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
