"""Platform entry point: python3 generate.py --name <ratings>.npy  ->  output.npy

The platform gives us ONE matrix (train + test merged) and scores our predictions
on a hidden "eval" set. We fit the model chosen in best_config.json (picked by
experiments.py on a validation split), keep every real rating unchanged and fill
the gaps with predictions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from models import predict

DEFAULT_CONFIG = {"method": "bias", "reg": 3.0}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                        help="Name of the npy of the ratings table to complete")
    args = parser.parse_args()

    # Open Ratings table
    print('Ratings loading...')
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')

    config_file = Path(__file__).with_name('best_config.json')
    config = json.loads(config_file.read_text()) if config_file.exists() else DEFAULT_CONFIG
    print('Model:', config)

    observed = np.isfinite(table)
    predictions = predict(table, config)
    table = np.where(observed, table, predictions)   # keep real ratings, fill only the gaps

    # Save the completed table
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE
