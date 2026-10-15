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
7. bpmf          : Bayesian matrix factorization with side-information priors (main model).
8. blend         : weighted average of several models.
"""
import re
from pathlib import Path

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
# 7. Bayesian matrix factorization with side information (our main model)
# ---------------------------------------------------------------------------
# Same prediction as ALS:  mu + b_u + b_i + p_u . q_i
# Two differences (details in METHOD.md):
#
# (a) Bayesian averaging (BPMF, Salakhutdinov & Mnih 2008).
#     ALS returns ONE best set of dials. With 2 ratings for a movie, many dial settings
#     fit equally well, so betting on one is risky. BPMF instead draws many plausible
#     settings (Gibbs sampling) and averages their predictions. The amount of
#     regularisation is also learned from the data instead of being tuned by hand.
#
# (b) Side-information prior (our addition, in the spirit of Porteous et al. 2010).
#     Plain BPMF pulls every movie's (bias, dials) towards the same average movie.
#     We pull each movie towards what its genres, release decade and popularity predict,
#     and each user towards what their rating BEHAVIOUR predicts (which genres they chose
#     to rate, how many ratings they gave). A movie with 1 rating then starts from
#     "a typical 1990s comedy" instead of from nothing.


def movie_metadata(n_movies):
    """Genres + release decade for each movie, from namesngenre.npy (or None if unavailable)."""
    path = Path(__file__).with_name('namesngenre.npy')
    if not path.exists():
        return None
    meta = np.load(path, allow_pickle=True)
    if len(meta) != n_movies:                       # file does not match this matrix
        return None
    genre_lists = [str(g).split('|') for g in meta[:, 1]]
    names = sorted({g for gs in genre_lists for g in gs})
    genres = np.array([[name in gs for name in names] for gs in genre_lists], float)
    years = []
    for title in meta[:, 0]:
        found = re.search(r'\((\d{4})\)\s*$', str(title).strip())   # "Toy Story (1995)" -> 1995
        years.append(float(found.group(1)) if found else np.nan)
    years = np.array(years)
    years[np.isnan(years)] = np.nanmedian(years)
    edges = [-np.inf, 1960, 1970, 1980, 1990, 2000, 2010, np.inf]
    decades = np.column_stack([(years >= a) & (years < b) for a, b in zip(edges[:-1], edges[1:])]).astype(float)
    return np.column_stack((genres, decades))


def side_features(table, metadata):
    """Feature rows used to build the priors. First column = 1 (intercept).

    user: genre mix of the movies they rated, log(1 + number of ratings)
    movie: genres, decade, log(1 + number of ratings)
    Only uses WHICH cells are rated, never the hidden ones.
    """
    rated = observed_mask(table).astype(float)
    n_user, n_item = rated.sum(axis=1), rated.sum(axis=0)
    user_f = [np.ones(len(n_user)), np.log1p(n_user)]
    item_f = [np.ones(len(n_item)), np.log1p(n_item)]
    if metadata is not None:
        user_f.append((rated @ metadata) / np.maximum(n_user, 1)[:, None])
        item_f.append(metadata)
    return np.column_stack(user_f), np.column_stack(item_f)


def _sample_wishart(rng, df, scale):
    """Random precision matrix from a Wishart distribution (Bartlett decomposition)."""
    d = scale.shape[0]
    lower = np.linalg.cholesky(scale)
    a = np.zeros((d, d))
    a[np.diag_indices(d)] = np.sqrt(rng.chisquare(df - np.arange(d)))
    a[np.tril_indices(d, -1)] = rng.standard_normal(d * (d - 1) // 2)
    la = lower @ a
    return la @ la.T


def _sample_prior(rng, vectors, features, mode="plug_in"):
    """Draw a Gaussian-prior precision, with optional sampling of feature coefficients.

    Prior model (when mode='sampled'):
        precision ~ Wishart(df=d, scale=I)
        W | precision ~ MatrixNormal(0, I_features, precision^-1)
        vectors | W, precision ~ MatrixNormal(features @ W, I_rows, precision^-1).

    Integrating W out gives the exact conditional posterior of the precision
    given 'vectors'. We sample it, then sample W conditionally on that precision.
    This is a valid blocked Gibbs update for (W, precision) given the vectors.

    'plug_in' preserves the team's original approximation (ridge-estimate W
    without sampling it). This mode reproduces the supplied historical results,
    but must not be called fully Bayesian for W.
    """
    if mode not in {"plug_in", "sampled"}:
        raise ValueError("mode must be 'plug_in' or 'sampled'")
    d = vectors.shape[1]
    nf = features.shape[1]
    reg = features.T @ features + np.eye(nf)
    fitted_W = np.linalg.solve(reg, features.T @ vectors)
    errors = vectors - features @ fitted_W
    scatter = errors.T @ errors
    if mode == "sampled":
        # The W prior contributes fitted_W.T @ fitted_W when W is marginalized.
        scatter = scatter + fitted_W.T @ fitted_W
    scale = np.linalg.inv(np.eye(d) + scatter)
    precision = _sample_wishart(rng, d + len(vectors), (scale + scale.T) / 2)
    W = fitted_W
    if mode == "sampled":
        z = rng.standard_normal((nf, d))
        # Left covariance reg^-1, right covariance precision^-1.
        left = np.linalg.solve(np.linalg.cholesky(reg).T, z)
        W = fitted_W + np.linalg.solve(np.linalg.cholesky(precision).T, left.T).T
    return features @ W, precision


def _sample_side(rng, rated, targets, other, centre, precision, noise_precision):
    """Draw [bias, dials] for ALL users (or all movies) at once.

    For one user this is Bayesian linear regression: features of movie j = [1, q_j],
    target = rating - mu - b_j. The posterior is Gaussian with
        precision  P = precision_prior + noise_precision * sum_j x_j x_j'
        mean       m = P^-1 (precision_prior @ centre + noise_precision * sum_j x_j * target_j)
    and we draw one sample from it. Matrix products do the sums for every user together.
    """
    x = np.column_stack((np.ones(len(other)), other[:, 1:]))
    d = x.shape[1]
    outer = (x[:, :, None] * x[:, None, :]).reshape(len(x), d * d)       # x_j x_j' for each column j
    xtx = (rated @ outer).reshape(-1, d, d)                              # summed over rated columns
    xty = targets @ x                                                    # targets are 0 where unrated
    post_prec = noise_precision * xtx + precision
    post_mean = np.linalg.solve(post_prec, (noise_precision * xty + centre @ precision)[..., None])[..., 0]
    chol = np.linalg.cholesky(post_prec)
    z = rng.standard_normal((len(post_prec), d, 1))
    return post_mean + np.linalg.solve(np.swapaxes(chol, 1, 2), z)[..., 0]


def _bpmf_chain(table, factors, burnin, samples, seed, user_f, item_f, prior_mode="plug_in"):
    """One Gibbs-sampling run; returns the average prediction over its kept samples."""
    mask = observed_mask(table)
    rated = mask.astype(float)
    mu = table[mask].mean()
    rng = np.random.default_rng(seed)
    _, b_u, b_i = fit_biases(table, 3.0)                                 # start from the bias model
    U = np.column_stack((b_u, rng.normal(0, .1, (table.shape[0], factors))))   # [b_u, p_u]
    V = np.column_stack((b_i, rng.normal(0, .1, (table.shape[1], factors))))   # [b_i, q_i]
    centred = np.where(mask, table - mu, 0.0)
    noise_precision = 2.0                                                # 1 / (rating noise variance)
    total = np.zeros(table.shape)
    for step in range(burnin + samples):
        centre_u, prec_u = _sample_prior(rng, U, user_f, mode=prior_mode)
        centre_v, prec_v = _sample_prior(rng, V, item_f, mode=prior_mode)
        U = _sample_side(rng, rated, centred - rated * V[:, 0], V, centre_u, prec_u, noise_precision)
        V = _sample_side(rng, rated.T, centred.T - rated.T * U[:, 0], U, centre_v, prec_v, noise_precision)
        pred = mu + U[:, :1] + V[:, 0][None, :] + U[:, 1:] @ V[:, 1:].T
        errors = (table - pred)[mask]
        noise_precision = rng.gamma(1 + errors.size / 2, 1 / (1 + 0.5 * errors @ errors))
        if step >= burnin:                                               # skip warm-up samples
            total += pred
    return total / samples


def bpmf(table, factors=20, chains=3, burnin=20, samples=80, seed=0, side_info=True,
         prior_mode="plug_in"):
    """MF averaged over chains. prior_mode='sampled' samples W as well.

    Default plug_in maintains compatibility with existing published local scores.
    For a full conditional update of the feature coefficients, use 'sampled' and
    re-evaluate rather than reusing previously tuned configs or reported scores.
    """
    metadata = movie_metadata(table.shape[1]) if side_info else None
    if side_info:
        user_f, item_f = side_features(table, metadata)
    else:
        user_f, item_f = np.ones((table.shape[0], 1)), np.ones((table.shape[1], 1))
    runs = [_bpmf_chain(table, factors, burnin, samples, seed + c, user_f, item_f, prior_mode) for c in range(chains)]
    return clip(np.mean(runs, axis=0))


# ---------------------------------------------------------------------------
# 8. Blend: weighted average of several models
# ---------------------------------------------------------------------------
def blend(table, parts, weights):
    """sum_k weights[k] * prediction of parts[k].

    Weights are fitted by least squares on held-out ratings (platform_sim.py).
    Models that make DIFFERENT mistakes partly cancel each other out.
    """
    if len(parts) != len(weights) or not parts:
        raise ValueError("blend requires matching, nonempty parts and weights")
    return clip(sum(w * predict(table, part) for w, part in zip(weights, parts)))


# ---------------------------------------------------------------------------
MODELS = {
    'global_mean': global_mean,
    'user_mean': user_mean,
    'item_mean': item_mean,
    'bias': bias,
    'user_knn': user_knn,
    'als_mf': als_mf,
    'bpmf': bpmf,
    'blend': blend,
}


def predict(table, config):
    """Run the model described by config, e.g. {"method": "bias", "reg": 3.0}."""
    params = dict(config)
    method = params.pop('method')
    return MODELS[method](table, **params)
