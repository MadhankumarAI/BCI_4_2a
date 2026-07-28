import numpy as np
import scipy.linalg as la
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import LedoitWolf

def logm_spd(C):
    """
    Compute matrix logarithm for symmetric positive-definite matrix C.
    """
    vals, vecs = la.eigh(C)
    vals = np.clip(vals, 1e-10, None)
    return vecs @ np.diag(np.log(vals)) @ vecs.T

def expm_spd(C):
    """
    Compute matrix exponential for symmetric matrix C.
    """
    vals, vecs = la.eigh(C)
    return vecs @ np.diag(np.exp(vals)) @ vecs.T

def inv_sqrt_spd(C):
    """
    Compute matrix inverse square root for symmetric positive-definite matrix C.
    """
    vals, vecs = la.eigh(C)
    vals = np.clip(vals, 1e-10, None)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

def log_euclidean_mean(covs):
    """
    Compute the Log-Euclidean geometric mean of covariance matrices.
    """
    log_covs = [logm_spd(c) for c in covs]
    mean_log_cov = np.mean(log_covs, axis=0)
    return expm_spd(mean_log_cov)

def vectorize_spd(Y):
    """
    Vectorize the upper triangular part of symmetric matrix Y,
    scaling the off-diagonals by sqrt(2) to preserve inner product.
    """
    n = Y.shape[0]
    idx_row, idx_col = np.triu_indices(n)
    vec = Y[idx_row, idx_col].copy()
    mask = idx_row != idx_col
    vec[mask] *= np.sqrt(2.0)
    return vec

def precompute_covariances(X):
    """
    Compute trial-by-trial covariance matrices using Ledoit-Wolf shrinkage.
    X shape: (trials, bands, samples, channels)
    Returns: (trials, bands, channels, channels)
    """
    n_trials, n_bands, n_samples, n_channels = X.shape
    covs = np.zeros((n_trials, n_bands, n_channels, n_channels))
    lw = LedoitWolf()
    for i in range(n_trials):
        for b in range(n_bands):
            trial = X[i, b]
            trial_centered = trial - np.mean(trial, axis=0)
            lw.fit(trial_centered)
            covs[i, b] = lw.covariance_
    return covs

class TangentSpace(BaseEstimator, TransformerMixin):
    """
    Transformer to project covariance matrices into the tangent space.
    """
    def __init__(self, metric='log-euclidean'):
        self.metric = metric
        self.C_ref_ = None
        
    def fit(self, X, y=None):
        """
        X: covariance matrices shape (trials, channels, channels)
        """
        self.C_ref_ = log_euclidean_mean(X)
        return self
        
    def transform(self, X):
        """
        X: covariance matrices shape (trials, channels, channels)
        """
        C_ref_inv_sqrt = inv_sqrt_spd(self.C_ref_)
        features = []
        for i in range(X.shape[0]):
            C = X[i]
            projected = C_ref_inv_sqrt @ C @ C_ref_inv_sqrt
            Y = logm_spd(projected)
            features.append(vectorize_spd(Y))
        return np.array(features)

class FilterBankTangentSpace(BaseEstimator, TransformerMixin):
    """
    Transformer to project filter-bank covariance matrices to tangent space.
    Input X shape: (trials, bands, channels, channels)
    """
    def __init__(self):
        self.ts_estimators = []
        
    def fit(self, X, y=None):
        """
        X shape: (trials, bands, channels, channels)
        """
        n_bands = X.shape[1]
        self.ts_estimators = []
        for b in range(n_bands):
            ts_est = TangentSpace()
            ts_est.fit(X[:, b])
            self.ts_estimators.append(ts_est)
        return self
        
    def transform(self, X):
        """
        X shape: (trials, bands, channels, channels)
        """
        n_bands = X.shape[1]
        features = []
        for b in range(n_bands):
            ts_feat = self.ts_estimators[b].transform(X[:, b])
            features.append(ts_feat)
        return np.concatenate(features, axis=1)
