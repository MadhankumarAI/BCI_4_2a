"""
Classical (non-GAN) augmentation baselines.
===========================================

These exist so the GAN has to earn its place. A GAN that improves kappa by
0.01 over no augmentation but loses to segment-recombination - which costs
milliseconds and has no failure modes - is not a result worth putting in a
medical pipeline. main.py therefore puts all of these in the same per-subject
selection gate as the GAN and lets the validation data pick.

Every function here takes and returns trials in (trials, samples, channels)
order and works on post-CAR data, matching the ensemble's own pipeline stage.
"""

import numpy as np
import scipy.linalg as la
from sklearn.covariance import LedoitWolf


def _rng(seed):
    return np.random.default_rng(seed)


def segment_recombine(X, y, n_new, n_segments=4, seed=0):
    """
    Segmentation & Recombination (Lotte 2015; Schirrmeister et al. 2017).

    Cut every trial into `n_segments` equal time slices, then build a new
    trial by drawing each slice from a different, randomly chosen trial of the
    SAME class. The result is a trial that never existed but whose every
    fragment is real cortical activity with the right class label.

    This is the strongest cheap baseline for motor imagery, because the
    features the downstream models use - band power and spatial covariance -
    are averaged over the whole window, so splicing preserves them while
    decorrelating the trial-specific noise.

    The seam artefacts are the known weakness: a discontinuity at each cut
    point injects broadband energy. We cross-fade over 10 samples to suppress
    it, which matters here because the filter bank runs up to 40 Hz.
    """
    rng = _rng(seed)
    n_trials, n_samples, n_channels = X.shape
    bounds = np.linspace(0, n_samples, n_segments + 1).astype(int)
    fade = min(10, (bounds[1] - bounds[0]) // 4)

    out_X, out_y = [], []
    classes = np.unique(y)
    per_class = n_new // len(classes)

    for c in classes:
        pool = np.where(y == c)[0]
        for _ in range(per_class):
            trial = np.empty((n_samples, n_channels))
            donors = rng.choice(pool, size=n_segments, replace=True)
            for s in range(n_segments):
                a, b = bounds[s], bounds[s + 1]
                seg = X[donors[s], a:b]
                if s > 0 and fade > 0:
                    # Linear cross-fade across the seam so the splice does not
                    # look like a step edge to a 40 Hz filter.
                    w = np.linspace(0, 1, fade).reshape(-1, 1)
                    trial[a:a + fade] = trial[a:a + fade] * (1 - w) + seg[:fade] * w
                    trial[a + fade:b] = seg[fade:]
                else:
                    trial[a:b] = seg
            out_X.append(trial)
            out_y.append(c)

    return np.asarray(out_X), np.asarray(out_y)


def gaussian_noise(X, y, n_new, sigma=0.05, seed=0):
    """
    Copy random trials and add white noise at `sigma` times the trial std.

    The weakest baseline, included as a control: if noise-copies help as much
    as the GAN does, then the "improvement" is just regularisation of the
    downstream classifier and has nothing to do with generative modelling.
    """
    rng = _rng(seed)
    idx = rng.integers(0, len(X), n_new)
    base = X[idx]
    scale = base.std(axis=(1, 2), keepdims=True) * sigma
    return base + rng.standard_normal(base.shape) * scale, y[idx]


def time_shift(X, y, n_new, max_frac=0.1, seed=0):
    """Copy random trials and circularly roll them in time.

    ERD onset latency genuinely varies trial to trial, so a small shift is a
    label-preserving transform rather than a distortion.
    """
    rng = _rng(seed)
    idx = rng.integers(0, len(X), n_new)
    max_shift = max(1, int(X.shape[1] * max_frac))
    out = np.empty((n_new,) + X.shape[1:])
    for i, j in enumerate(idx):
        out[i] = np.roll(X[j], rng.integers(-max_shift, max_shift + 1), axis=0)
    return out, y[idx]


def _inv_sqrt(C):
    w, V = la.eigh(C)
    w = np.clip(w, 1e-10, None)
    return V @ np.diag(w ** -0.5) @ V.T


def _sqrt(C):
    w, V = la.eigh(C)
    w = np.clip(w, 1e-10, None)
    return V @ np.diag(np.sqrt(w)) @ V.T


def riemannian_recolour(X, y, n_new, seed=0):
    """
    Geodesic re-colouring on the SPD manifold.

    Take a real trial, whiten it by its own spatial covariance, then re-colour
    it with a covariance drawn from the geodesic between its own and another
    same-class trial's:

        C_mix = C_a^{1/2} (C_a^{-1/2} C_b C_a^{-1/2})^t C_a^{1/2},  t ~ U(0,1)
        X_new = X_a C_a^{-1/2} C_mix^{1/2}

    Why this baseline specifically: the ensemble is four-fifths Riemannian, so
    what it actually consumes is the spatial covariance of each trial. This
    transform moves a trial to a new, geometrically valid point on the SPD
    manifold - a covariance that is genuinely between two real ones - while
    keeping the real temporal waveform intact. It is the cheapest way to
    populate the exact manifold region the classifiers care about, and it is
    the honest thing for the GAN to be measured against.
    """
    rng = _rng(seed)
    lw = LedoitWolf()
    n_channels = X.shape[2]

    covs = np.empty((len(X), n_channels, n_channels))
    for i in range(len(X)):
        lw.fit(X[i] - X[i].mean(axis=0))
        covs[i] = lw.covariance_

    out_X, out_y = [], []
    for c in np.unique(y):
        pool = np.where(y == c)[0]
        for _ in range(n_new // len(np.unique(y))):
            a, b = rng.choice(pool, size=2, replace=False)
            t = rng.uniform(0.15, 0.85)

            Ca_inv_sqrt = _inv_sqrt(covs[a])
            Ca_sqrt = _sqrt(covs[a])
            M = Ca_inv_sqrt @ covs[b] @ Ca_inv_sqrt
            w, V = la.eigh(M)
            w = np.clip(w, 1e-10, None)
            # Fractional matrix power = geodesic interpolation at parameter t.
            C_mix = Ca_sqrt @ (V @ np.diag(w ** t) @ V.T) @ Ca_sqrt

            out_X.append((X[a] - X[a].mean(axis=0)) @ Ca_inv_sqrt @ _sqrt(C_mix))
            out_y.append(c)

    return np.asarray(out_X), np.asarray(out_y)


#: Name -> callable(X, y, n_new, seed) used by the selection gate in main.py.
CLASSICAL_METHODS = {
    "segrec": lambda X, y, n, s: segment_recombine(X, y, n, seed=s),
    "riemann": lambda X, y, n, s: riemannian_recolour(X, y, n, seed=s),
    "noise": lambda X, y, n, s: gaussian_noise(X, y, n, seed=s),
    "shift": lambda X, y, n, s: time_shift(X, y, n, seed=s),
}
