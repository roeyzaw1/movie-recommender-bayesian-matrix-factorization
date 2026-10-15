"""Rating-prediction models for DataLab A1.

Conventions used everywhere in this file
----------------------------------------
* `table` is a (users x movies) matrix. Row u = one user, column i = one movie.
* A rating is a number in [0.5, 5.0] (half stars). NaN = "not rated".
* Every model takes the table and returns a FULL matrix of predictions
  (one number for every user/movie pair). generate.py then keeps the real
  ratings where they exist and uses the predictions for the gaps.
* Models only ever learn from OBSERVED cells. We never replace NaN by 0
  before fitting: 0 would mean "hated it", which is not what a gap means.

Models, from simplest to most complex
-------------------------------------
1. global_mean   : everyone gives every movie the average rating.
2. user_mean     : each user's own average rating.
3. item_mean     : each movie's own average rating.
4. bias          : average + "how generous is this user" + "how liked is this movie".
5. user_knn      : bias, corrected by what similar users thought of the movie.
6. als_mf        : matrix factorization (hidden "taste dials"), alternating least squares.
"""
import numpy as np

MIN_RATING, MAX_RATING = 0.5, 5.0


def clip(predictions):
    """Keep predictions inside the valid rating range."""
    return np.clip(predictions, MIN_RATING, MAX_RATING)


def observed_mask(table):
    """True where a rating exists, False where it is missing (NaN)."""
    return np.isfinite(table)


# ---------------------------------------------------------------------------
# 1-3. Mean baselines
# ---------------------------------------------------------------------------
def global_mean(table):
    """Predict the overall average rating everywhere."""
    mask = observed_mask(table)
    return np.full(table.shape, table[mask].mean())


def user_mean(table):
    """Predict each user's average rating (global average if the user has none)."""
    mask = observed_mask(table)
    mu = table[mask].mean()
    counts = mask.sum(axis=1)
    sums = np.where(mask, table, 0.0).sum(axis=1)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), mu)
    return np.repeat(means[:, None], table.shape[1], axis=1)


def item_mean(table):
    """Predict each movie's average rating (global average if the movie has none)."""
    mask = observed_mask(table)
    mu = table[mask].mean()
    counts = mask.sum(axis=0)
    sums = np.where(mask, table, 0.0).sum(axis=0)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), mu)
    return np.repeat(means[None, :], table.shape[0], axis=0)


# ---------------------------------------------------------------------------
# 4. Bias baseline:  prediction(u, i) = mu + b_u + b_i
# ---------------------------------------------------------------------------
def fit_biases(table, reg=3.0, iterations=15):
    """Learn a user bias b_u and a movie bias b_i.

    b_u > 0 : user u rates higher than average (generous).
    b_i > 0 : movie i is rated higher than average (well liked).

    We minimise   sum over observed (r_ui - mu - b_u - b_i)^2 + reg * (sum b_u^2 + sum b_i^2)
    by alternating two exact updates (fix movies -> solve users, fix users -> solve movies).

    `reg` acts like `reg` imaginary ratings equal to the average: a movie with one
    rating of 5.0 gets b_i = 1.5 / (1 + reg) instead of a full +1.5, so we do not
    trust a single opinion too much. Movies with many ratings are barely affected.
    """
    mask = observed_mask(table)
    mu = table[mask].mean()
    residual = np.where(mask, table - mu, 0.0)   # how far each rating is from the average
    n_user = mask.sum(axis=1)                    # number of ratings per user
    n_item = mask.sum(axis=0)                    # number of ratings per movie
    b_u = np.zeros(table.shape[0])
    b_i = np.zeros(table.shape[1])
    for _ in range(iterations):
        b_u = (residual - mask * b_i[None, :]).sum(axis=1) / (n_user + reg)
        b_i = (residual - mask * b_u[:, None]).sum(axis=0) / (n_item + reg)
    return mu, b_u, b_i


def bias(table, reg=3.0):
    """Predict mu + b_u + b_i for every user/movie pair."""
    mu, b_u, b_i = fit_biases(table, reg)
    return clip(mu + b_u[:, None] + b_i[None, :])


# ---------------------------------------------------------------------------
# 5. User-based nearest neighbours (lecture slides 8-9)
# ---------------------------------------------------------------------------
def user_knn(table, k=30, shrinkage=10.0, damping=1.0, reg=3.0):
    """Bias baseline corrected by the opinions of the k most similar users.

    Steps
    1. Fit the bias baseline and compute each observed rating's deviation from it
       ("Dana liked Alien 0.6 stars MORE than the baseline expected").
    2. Similarity between two users = Pearson-style correlation of their deviations
       on the movies they BOTH rated (lecture slide 9, centred on the baseline
       instead of the user mean).
       Shrinkage: a similarity computed from few shared movies is unreliable, so
       multiply it by n_common / (n_common + shrinkage).
    3. To predict (u, i): take the k users most similar to u who rated movie i and
       average their deviations, weighted by similarity:

           prediction = baseline(u, i) + sum(sim * deviation) / (sum |sim| + damping)

       With damping = 0 this is exactly the lecture formula. damping > 0 pulls the
       correction towards 0 when the neighbours are only weakly similar
       (otherwise one barely-similar neighbour would count as much as ten good ones).
    """
    mask = observed_mask(table)
    mu, b_u, b_i = fit_biases(table, reg)
    baseline = mu + b_u[:, None] + b_i[None, :]
    deviation = np.where(mask, table - baseline, 0.0)
    obs = mask.astype(float)

    # Pearson-style similarity over co-rated movies, for all user pairs at once.
    n_common = obs @ obs.T                           # [u, v] = number of movies both rated
    numerator = deviation @ deviation.T              # sum of products on co-rated movies
    sq = (deviation ** 2) @ obs.T                    # [u, v] = sum of u's squared deviations on movies v rated
    denominator = np.sqrt(sq * sq.T)
    sim = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    sim *= n_common / (n_common + shrinkage)
    np.fill_diagonal(sim, 0.0)                       # a user is not their own neighbour

    predictions = baseline.copy()
    for i in range(table.shape[1]):
        raters = np.flatnonzero(mask[:, i])          # users who rated movie i
        if raters.size == 0:
            continue
        weights = sim[:, raters]                     # (all users) x (raters of i)
        if raters.size > k:                          # keep only the k most similar raters
            drop = np.argpartition(np.abs(weights), raters.size - k, axis=1)[:, :raters.size - k]
            weights = weights.copy()
            np.put_along_axis(weights, drop, 0.0, axis=1)
        correction = weights @ deviation[raters, i]
        predictions[:, i] += correction / (np.abs(weights).sum(axis=1) + damping + 1e-12)
    return clip(predictions)


# ---------------------------------------------------------------------------
# 6. Matrix factorization:  prediction(u, i) = mu + b_u + b_i + p_u . q_i
# ---------------------------------------------------------------------------
# Each movie i gets a vector q_i of `factors` hidden "dials" (e.g. how much action,
# how dark). Each user u gets a vector p_u saying how much they like each dial.
# p_u . q_i (dot product) = how well this movie matches this user's taste.
# The dials are never named: the algorithm invents whichever best explain the data.
#
# The lecture (slide 19) gives two ways to learn p, q, b_u, b_i: gradient descent,
# or ALS (used here): freeze movies -> each user is an exact ridge regression; then swap.


def als_mf(table, factors=32, reg_bias=3.0, reg_factor=10.0, epochs=15, seed=42):
    """Biased matrix factorization trained by alternating least squares.

    Minimises
        sum over observed (r_ui - mu - b_u - b_i - p_u.q_i)^2
        + reg_bias   * (sum b_u^2 + sum b_i^2)
        + reg_factor * (sum |p_u|^2 + sum |q_i|^2)

    With all movie parameters fixed, finding (b_u, p_u) for one user is an ordinary
    ridge regression on the movies that user rated -> one small linear system.
    Same for movies with users fixed. Alternate `epochs` times.

    Biases and dials get SEPARATE penalties: biases are reliable with few ratings
    (small reg_bias), dials overfit easily (larger reg_factor). One shared value
    forces a bad compromise. (The lecture's cost on slide 16 also uses two
    separate penalties, lambda and mu.)
    """
    mask = observed_mask(table)
    mu = table[mask].mean()
    rng = np.random.default_rng(seed)
    p = rng.normal(0.0, 0.1, (table.shape[0], factors))
    q = rng.normal(0.0, 0.1, (table.shape[1], factors))
    b_u = np.zeros(table.shape[0])
    b_i = np.zeros(table.shape[1])
    rated_by_user = [np.flatnonzero(row) for row in mask]     # movies each user rated
    raters_of_item = [np.flatnonzero(col) for col in mask.T]  # users who rated each movie
    penalty = np.diag([reg_bias] + [reg_factor] * factors)    # first coefficient = the bias

    def ridge(features, targets):
        # Solve min |targets - [1, features] @ coef|^2 + coef' penalty coef
        x = np.column_stack((np.ones(len(targets)), features))
        return np.linalg.solve(x.T @ x + penalty, x.T @ targets)

    for _ in range(epochs):
        for u, movies in enumerate(rated_by_user):           # update users, movies fixed
            coef = ridge(q[movies], table[u, movies] - mu - b_i[movies])
            b_u[u], p[u] = coef[0], coef[1:]
        for i, users in enumerate(raters_of_item):           # update movies, users fixed
            coef = ridge(p[users], table[users, i] - mu - b_u[users])
            b_i[i], q[i] = coef[0], coef[1:]
    return clip(mu + b_u[:, None] + b_i[None, :] + p @ q.T)


# ---------------------------------------------------------------------------
MODELS = {
    'global_mean': global_mean,
    'user_mean': user_mean,
    'item_mean': item_mean,
    'bias': bias,
    'user_knn': user_knn,
    'als_mf': als_mf,
}


def predict(table, config):
    """Run the model described by config, e.g. {"method": "bias", "reg": 3.0}."""
    params = dict(config)
    method = params.pop('method')
    return MODELS[method](table, **params)
