"""
Quality, fidelity and memorisation metrics for synthetic EEG.
=============================================================

Hartmann et al. (section 5) are explicit that no single metric is trustworthy
for EEG GANs: in their table the model with the best Frechet distance produced
the *least* realistic spectra, and the Inception Score gave no signal at all
about a collapsed model. So this module reports a panel and main.py never
decides anything from it - the decision is made by downstream classifier
performance on held-out real data. The panel is for diagnosis and for the
audit trail.

The panel:

  psd_log_distance   Do the synthetic spectra have EEG shape? Directly checks
                     the aliasing failure that transposed convolutions cause.
  band_power_error   Same question restricted to mu (8-13 Hz) and beta
                     (13-30 Hz), which is where motor imagery actually lives.
  frechet_tangent    Our EEG analogue of FID: Frechet distance between real
                     and synthetic in Riemannian tangent space instead of in
                     Inception feature space. Tangent space is the right
                     choice here because it is literally the feature space
                     three of the five base classifiers consume.
  sliced_wasserstein Hartmann's preferred metric - the model with the best SWD
                     had the most natural spatial and spectral distributions.
  nn_distance_ratio  MEMORISATION / PRIVACY CHECK. See below.
  tstr_accuracy      Train on Synthetic, Test on Real. The gold standard from
                     Brophy et al.'s time-series GAN review.
  trts_accuracy      Train on Real, Test on Synthetic. Catches the opposite
                     failure: synthetic data that is trivially separable
                     because the generator encoded the label too crudely.

On nn_distance_ratio
--------------------
With ~110 training trials the dominant risk is not a bad GAN, it is a GAN that
has memorised the training set and is re-emitting it with jitter. That would
still improve every fidelity metric, would still look fine on TSTR, and would
be worthless as augmentation - and in a clinical setting it is a re-identifi-
cation hazard, since the "synthetic" data would carry a real patient's signal.

We measure it the way Hartmann's Euclidean-distance metric does, but as a
ratio so it is interpretable without a reference:

    ratio = mean_i min_j d(synthetic_i, real_j)  /  mean_i min_{j != i} d(real_i, real_j)

Read it as: how far is a fake trial from the nearest real trial, relative to
how far real trials are from each other.

    ratio << 1   the generator is reproducing training trials. Reject.
    ratio ~= 1   fakes sit at the same spacing as reals. This is the target.
    ratio >> 1   fakes are off-manifold; the generator has not converged.
"""

import numpy as np
import scipy.linalg as la
from scipy.signal import welch
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

FS = 250


# ---------------------------------------------------------------------------
# Spectral fidelity
# ---------------------------------------------------------------------------

def _mean_psd(X, fs=FS, nperseg=256):
    """Mean power spectral density over trials and channels.

    X : (trials, samples, channels)
    """
    f, P = welch(X, fs=fs, nperseg=min(nperseg, X.shape[1]), axis=1)
    return f, P.mean(axis=(0, 2))


def amplitude_ratio(X_real, X_fake):
    """
    Per-channel standard deviation of synthetic over real.

    The cheapest and most revealing single number in the panel, and the one
    this code originally lacked. An under-trained generator gets the spectral
    SHAPE roughly right long before it gets the SCALE right, so the log-PSD
    distance can look unremarkable while the synthetic trials carry a seventh
    of the real power. Covariance-based classifiers - which is most of this
    ensemble - are then handed trials whose spatial covariance is off by a
    factor of ~50, and they will happily fit that.

    Want ~1.0. Outside [0.5, 2.0] the generator is not ready to use.
    """
    r = float(np.mean(X_real.std(axis=(0, 1))))
    f = float(np.mean(X_fake.std(axis=(0, 1))))
    return f / max(r, 1e-12)


def psd_log_distance(X_real, X_fake, fmax=45.0):
    """
    Mean absolute difference of log10 PSD, over 1-45 Hz.

    Log scale because EEG power falls off as roughly 1/f: on a linear scale
    the delta band would swamp everything and a generator could score well
    while emitting nothing at all in mu and beta.
    """
    f, Pr = _mean_psd(X_real)
    _, Pf = _mean_psd(X_fake)
    m = (f >= 1.0) & (f <= fmax)
    return float(np.mean(np.abs(np.log10(Pr[m] + 1e-20) - np.log10(Pf[m] + 1e-20))))


def band_power_error(X_real, X_fake, bands=((8, 13), (13, 30))):
    """Relative error in mean band power, per band. Keys are e.g. "8-13Hz"."""
    f, Pr = _mean_psd(X_real)
    _, Pf = _mean_psd(X_fake)
    out = {}
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        pr, pf = Pr[m].mean(), Pf[m].mean()
        out[f"{lo}-{hi}Hz"] = float(abs(pf - pr) / (pr + 1e-20))
    return out


# ---------------------------------------------------------------------------
# Distribution distance in Riemannian tangent space
# ---------------------------------------------------------------------------

def _logm(C):
    w, V = la.eigh(C)
    return V @ np.diag(np.log(np.clip(w, 1e-10, None))) @ V.T


def _expm(C):
    w, V = la.eigh(C)
    return V @ np.diag(np.exp(w)) @ V.T


def _inv_sqrt(C):
    w, V = la.eigh(C)
    return V @ np.diag(np.clip(w, 1e-10, None) ** -0.5) @ V.T


def _covs(X):
    """Ledoit-Wolf shrunk spatial covariances. X: (trials, samples, channels)."""
    lw = LedoitWolf()
    out = np.empty((len(X), X.shape[2], X.shape[2]))
    for i in range(len(X)):
        lw.fit(X[i] - X[i].mean(axis=0))
        out[i] = lw.covariance_
    return out


def tangent_features(X, ref=None):
    """
    Project trials into the tangent space at the log-Euclidean mean.

    Returns (features, ref). Pass the returned `ref` when projecting a second
    set, so both land in the same tangent space - otherwise the distance
    between them is meaningless.
    """
    C = _covs(X)
    if ref is None:
        ref = _expm(np.mean([_logm(c) for c in C], axis=0))
    R = _inv_sqrt(ref)

    n = ref.shape[0]
    iu = np.triu_indices(n)
    off = iu[0] != iu[1]

    feats = np.empty((len(C), len(iu[0])))
    for i, c in enumerate(C):
        Y = _logm(R @ c @ R)
        v = Y[iu].copy()
        v[off] *= np.sqrt(2.0)  # preserve the Frobenius inner product
        feats[i] = v
    return feats, ref


def frechet_tangent_distance(X_real, X_fake):
    """
    Frechet distance between the two Gaussians fitted in tangent space:

        ||mu_r - mu_f||^2 + tr(S_r + S_f - 2 (S_r S_f)^{1/2})

    Same formula as FID, but the embedding is Riemannian tangent space rather
    than an ImageNet network - which would be meaningless for EEG anyway.
    """
    Fr, ref = tangent_features(X_real)
    Ff, _ = tangent_features(X_fake, ref=ref)

    mu_r, mu_f = Fr.mean(0), Ff.mean(0)
    # Shrunk covariances: the feature dimension (253 for 22 channels) exceeds
    # the trial count, so the raw sample covariance is singular.
    Sr = np.cov(Fr, rowvar=False) + np.eye(Fr.shape[1]) * 1e-6
    Sf = np.cov(Ff, rowvar=False) + np.eye(Ff.shape[1]) * 1e-6

    covmean = la.sqrtm(Sr @ Sf)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu_r - mu_f) ** 2).sum() + np.trace(Sr + Sf - 2 * covmean))


def sliced_wasserstein(X_real, X_fake, n_projections=256, seed=0):
    """
    Sliced Wasserstein distance in tangent space.

    Project both clouds onto many random 1-D directions, take the 1-D
    Wasserstein distance (which is just the mean absolute difference of sorted
    samples), average. Unlike the Frechet distance this makes no Gaussian
    assumption, which is why Hartmann found it caught mode collapse that FID
    missed.
    """
    rng = np.random.default_rng(seed)
    Fr, ref = tangent_features(X_real)
    Ff, _ = tangent_features(X_fake, ref=ref)

    scaler = StandardScaler().fit(Fr)
    Fr, Ff = scaler.transform(Fr), scaler.transform(Ff)

    dirs = rng.standard_normal((Fr.shape[1], n_projections))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)

    pr = np.sort(Fr @ dirs, axis=0)
    pf = np.sort(Ff @ dirs, axis=0)

    # Different sample counts: resample both onto a common quantile grid.
    q = np.linspace(0, 1, min(len(pr), len(pf)))
    ir = (q * (len(pr) - 1)).round().astype(int)
    iff = (q * (len(pf) - 1)).round().astype(int)
    return float(np.mean(np.abs(pr[ir] - pf[iff])))


# ---------------------------------------------------------------------------
# Memorisation
# ---------------------------------------------------------------------------

def nn_distance_ratio(X_real, X_fake):
    """
    Nearest-neighbour distance ratio in tangent space. See module docstring.

    Returns a dict with the ratio and both of its components, because the
    ratio alone hides which side moved.
    """
    Fr, ref = tangent_features(X_real)
    Ff, _ = tangent_features(X_fake, ref=ref)

    scaler = StandardScaler().fit(Fr)
    Fr, Ff = scaler.transform(Fr), scaler.transform(Ff)

    # fake -> nearest real
    d_fr = np.linalg.norm(Ff[:, None, :] - Fr[None, :, :], axis=2)
    fake_to_real = d_fr.min(axis=1).mean()

    # real -> nearest other real (self-distance masked out)
    d_rr = np.linalg.norm(Fr[:, None, :] - Fr[None, :, :], axis=2)
    np.fill_diagonal(d_rr, np.inf)
    real_to_real = d_rr.min(axis=1).mean()

    return {
        "ratio": float(fake_to_real / (real_to_real + 1e-12)),
        "fake_to_real": float(fake_to_real),
        "real_to_real": float(real_to_real),
    }


# ---------------------------------------------------------------------------
# Downstream utility
# ---------------------------------------------------------------------------

def _fit_lda(X, y):
    F, ref = tangent_features(X)
    sc = StandardScaler().fit(F)
    clf = LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(F), y)
    return clf, sc, ref


def tstr_trts(X_real, y_real, X_fake, y_fake, X_holdout, y_holdout):
    """
    Train-on-Synthetic-Test-on-Real, and its mirror.

    TSTR is the metric that actually matters: if a classifier trained purely on
    synthetic trials can classify real held-out trials, the generator captured
    the discriminative structure rather than just the marginal statistics.

    TRTS is the sanity check in the other direction. A TRTS far ABOVE the real
    baseline means the generator is producing exaggerated, unnaturally
    separable class prototypes - which inflates TSTR too, and would quietly
    teach the downstream ensemble a decision boundary that does not transfer.

    `X_holdout` must be real trials that the GAN never saw.
    """
    clf_s, sc_s, ref_s = _fit_lda(X_fake, y_fake)
    Fh, _ = tangent_features(X_holdout, ref=ref_s)
    tstr = accuracy_score(y_holdout, clf_s.predict(sc_s.transform(Fh)))

    clf_r, sc_r, ref_r = _fit_lda(X_real, y_real)
    Ff, _ = tangent_features(X_fake, ref=ref_r)
    trts = accuracy_score(y_fake, clf_r.predict(sc_r.transform(Ff)))

    Fh2, _ = tangent_features(X_holdout, ref=ref_r)
    trtr = accuracy_score(y_holdout, clf_r.predict(sc_r.transform(Fh2)))

    return {"tstr": float(tstr), "trts": float(trts), "trtr_baseline": float(trtr)}


def full_report(X_real, y_real, X_fake, y_fake, X_holdout=None, y_holdout=None):
    """Run the whole panel. Returns a flat dict, ready to print or serialise."""
    rep = {
        "amplitude_ratio": amplitude_ratio(X_real, X_fake),
        "psd_log_distance": psd_log_distance(X_real, X_fake),
        "frechet_tangent": frechet_tangent_distance(X_real, X_fake),
        "sliced_wasserstein": sliced_wasserstein(X_real, X_fake),
    }
    rep.update({f"band_err_{k}": v
                for k, v in band_power_error(X_real, X_fake).items()})
    rep.update({f"nn_{k}": v for k, v in nn_distance_ratio(X_real, X_fake).items()})
    if X_holdout is not None:
        rep.update(tstr_trts(X_real, y_real, X_fake, y_fake, X_holdout, y_holdout))
    return rep
