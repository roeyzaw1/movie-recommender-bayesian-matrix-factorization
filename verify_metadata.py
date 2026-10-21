"""Verify the corrected instructor metadata and unchanged rating arrays.

Run: python3 verify_metadata.py
"""
from pathlib import Path
import numpy as np
from models import _check_corrected_metadata, movie_metadata, EXPECTED_MOVIES


def main():
    root = Path(__file__).resolve().parent
    metadata = _check_corrected_metadata(EXPECTED_MOVIES)
    ids = np.load(root / 'movie_ids.npy', allow_pickle=False)
    features = movie_metadata(EXPECTED_MOVIES)
    print(f'CORRECT metadata: {metadata.shape}, IDs: {ids.shape}, features: {features.shape}')
    print(f'Column 0: movieId={ids[0]} title={metadata[0,0]!r} genres={metadata[0,1]!r}')
    assert ids[0] == 1 and metadata[0, 0] == 'Toy Story (1995)'
    assert features.shape[0] == EXPECTED_MOVIES and np.isfinite(features).all()
    train_path, test_path = root / 'ratings_train.npy', root / 'ratings_test.npy'
    if train_path.exists() and test_path.exists():
        train, test = np.load(train_path), np.load(test_path)
        assert train.shape == test.shape == (610, EXPECTED_MOVIES)
        assert np.isfinite(train).sum() == np.isfinite(test).sum() == 31598
        assert not np.any(np.isfinite(train) & np.isfinite(test))
        print('CORRECT ratings: shape (610, 4980), 31,598 observations in each split; disjoint')
    else:
        print('Ratings not copied here: metadata validation passed; skip rating checks')
    print('PASS — corrected metadata / movie ID mapping validated')


if __name__ == '__main__':
    main()
