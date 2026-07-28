"""
Augmented Covariance Riemannian + MDM + Ensemble Pipeline
=========================================================

This module implements three cutting-edge techniques from the BCI literature:

1. Augmented Covariance Matrix (ACM): Embeds time-delayed copies of signals
   into the covariance matrix, capturing spatio-temporal dynamics (Barachant 2015).

2. Minimum Distance to Mean (MDM): A parameter-free Riemannian classifier
   that computes class-conditional geometric means and classifies based on
   geodesic distance on the SPD manifold.

3. FgMDM: MDM enhanced with Fisher Geodesic Discriminant Analysis filtering
   in the tangent space before classification.
"""

import numpy as np
import scipy.linalg as la
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.covariance import LedoitWolf


# ============================================================================
# Core SPD Matrix Operations
# ============================================================================

def logm_spd(C):
    """Matrix logarithm for SPD matrix."""
    vals, vecs = la.eigh(C)
    vals = np.clip(vals, 1e-10, None)
    return vecs @ np.diag(np.log(vals)) @ vecs.T

def expm_spd(C):
    """Matrix exponential for symmetric matrix."""
    vals, vecs = la.eigh(C)
    return vecs @ np.diag(np.exp(vals)) @ vecs.T

def sqrtm_spd(C):
    """Matrix square root for SPD matrix."""
    vals, vecs = la.eigh(C)
    vals = np.clip(vals, 1e-10, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T

def inv_sqrt_spd(C):
    """Matrix inverse square root for SPD matrix."""
    vals, vecs = la.eigh(C)
    vals = np.clip(vals, 1e-10, None)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

def inv_spd(C):
    """Matrix inverse for SPD matrix."""
    vals, vecs = la.eigh(C)
    vals = np.clip(vals, 1e-10, None)
    return vecs @ np.diag(1.0 / vals) @ vecs.T

def geodesic_distance(A, B):
    """
    Compute the affine-invariant Riemannian geodesic distance between
    two SPD matrices A and B.
    d(A, B) = ||log(A^{-1/2} B A^{-1/2})||_F
    """
    A_inv_sqrt = inv_sqrt_spd(A)
    M = A_inv_sqrt @ B @ A_inv_sqrt
    log_M = logm_spd(M)
    return la.norm(log_M, 'fro')

def log_euclidean_mean(covs):
    """Log-Euclidean geometric mean of SPD matrices."""
    log_covs = np.array([logm_spd(c) for c in covs])
    return expm_spd(np.mean(log_covs, axis=0))

def affine_invariant_mean(covs, max_iter=50, tol=1e-9):
    """
    Compute the affine-invariant (Karcher/Frechet) mean of SPD matrices.
    Uses iterative gradient descent on the manifold.
    """
    n = len(covs)
    # Initialize with log-euclidean mean
    G = log_euclidean_mean(covs)
    
    for _ in range(max_iter):
        G_inv_sqrt = inv_sqrt_spd(G)
        G_sqrt = sqrtm_spd(G)
        
        # Compute tangent vectors
        S = np.zeros_like(G)
        for c in covs:
            S += logm_spd(G_inv_sqrt @ c @ G_inv_sqrt)
        S /= n
        
        # Check convergence
        if la.norm(S, 'fro') < tol:
            break
        
        # Update mean
        G = G_sqrt @ expm_spd(S) @ G_sqrt
    
    return G

def vectorize_spd(Y):
    """
    Vectorize upper triangle of symmetric matrix, scaling off-diagonals
    by sqrt(2) to preserve inner product.
    """
    n = Y.shape[0]
    idx_row, idx_col = np.triu_indices(n)
    vec = Y[idx_row, idx_col].copy()
    mask = idx_row != idx_col
    vec[mask] *= np.sqrt(2.0)
    return vec


# ============================================================================
# Augmented Covariance Matrix
# ============================================================================

class AugmentedCovariances(BaseEstimator, TransformerMixin):
    """
    Compute Augmented Covariance Matrices by embedding time-delayed
    copies of the signal into the spatial covariance.
    
    This captures spatio-temporal dynamics that standard spatial
    covariance cannot represent (Barachant 2015).
    
    Parameters
    ----------
    n_delays : int
        Number of time delays to embed. Each delay adds n_channels
        dimensions to the covariance matrix.
    delay_step : int
        Number of samples between consecutive delays.
    """
    def __init__(self, n_delays=3, delay_step=4):
        self.n_delays = n_delays
        self.delay_step = delay_step
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """
        X: (trials, samples, channels)
        Returns: (trials, aug_channels, aug_channels) where
                 aug_channels = channels * (1 + n_delays)
        """
        n_trials, n_samples, n_channels = X.shape
        max_delay = self.n_delays * self.delay_step
        effective_samples = n_samples - max_delay
        
        lw = LedoitWolf()
        covs = []
        
        for i in range(n_trials):
            # Build augmented signal matrix
            blocks = [X[i, max_delay:, :]]  # Original signal (trimmed)
            for d in range(1, self.n_delays + 1):
                offset = max_delay - d * self.delay_step
                blocks.append(X[i, offset:offset + effective_samples, :])
            
            # Concatenate along channel axis: (effective_samples, aug_channels)
            augmented = np.concatenate(blocks, axis=1)
            
            # Compute covariance
            augmented_centered = augmented - np.mean(augmented, axis=0)
            lw.fit(augmented_centered)
            covs.append(lw.covariance_)
        
        return np.array(covs)


# ============================================================================
# Tangent Space Projection
# ============================================================================

class TangentSpace(BaseEstimator, TransformerMixin):
    """Project SPD matrices to tangent space at the geometric mean."""
    
    def __init__(self, metric='log-euclidean'):
        self.metric = metric
        self.C_ref_ = None
        
    def fit(self, X, y=None):
        if self.metric == 'affine-invariant':
            self.C_ref_ = affine_invariant_mean(X)
        else:
            self.C_ref_ = log_euclidean_mean(X)
        return self
        
    def transform(self, X):
        C_ref_inv_sqrt = inv_sqrt_spd(self.C_ref_)
        features = []
        for i in range(X.shape[0]):
            projected = C_ref_inv_sqrt @ X[i] @ C_ref_inv_sqrt
            Y = logm_spd(projected)
            features.append(vectorize_spd(Y))
        return np.array(features)


# ============================================================================
# MDM Classifier (Minimum Distance to Mean)
# ============================================================================

class MDM(BaseEstimator, ClassifierMixin):
    """
    Minimum Distance to Mean classifier on the SPD manifold.
    
    For each class, computes the geometric mean of the training
    covariance matrices. At test time, assigns the class whose
    mean is closest in geodesic distance.
    
    This is parameter-free and extremely robust to noise.
    """
    def __init__(self, metric='affine-invariant'):
        self.metric = metric
        self.class_means_ = {}
        self.classes_ = None
        
    def fit(self, X, y):
        """
        X: (trials, channels, channels) - SPD matrices
        y: (trials,) - class labels
        """
        self.classes_ = np.unique(y)
        self.class_means_ = {}
        
        for c in self.classes_:
            mask = y == c
            if self.metric == 'affine-invariant':
                self.class_means_[c] = affine_invariant_mean(X[mask])
            else:
                self.class_means_[c] = log_euclidean_mean(X[mask])
        
        return self
    
    def predict(self, X):
        """Predict class labels based on minimum geodesic distance."""
        predictions = []
        for i in range(X.shape[0]):
            distances = {}
            for c in self.classes_:
                distances[c] = geodesic_distance(X[i], self.class_means_[c])
            predictions.append(min(distances, key=distances.get))
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Estimate class probabilities using softmin of geodesic distances.
        """
        probas = []
        for i in range(X.shape[0]):
            distances = np.array([
                geodesic_distance(X[i], self.class_means_[c])
                for c in self.classes_
            ])
            # Softmin: convert distances to probabilities
            neg_distances = -distances
            exp_neg = np.exp(neg_distances - np.max(neg_distances))
            proba = exp_neg / np.sum(exp_neg)
            probas.append(proba)
        return np.array(probas)


# ============================================================================
# FgMDM (Fisher Geodesic MDM)
# ============================================================================

class FgMDM(BaseEstimator, ClassifierMixin):
    """
    Fisher Geodesic MDM: MDM enhanced with Fisher Geodesic Discriminant
    Analysis (FGDA) filtering in the tangent space.
    
    Steps:
    1. Project training SPD matrices to tangent space
    2. Apply Fisher LDA in tangent space to find discriminant directions
    3. Project back to manifold using filtered tangent vectors
    4. Classify using MDM on the filtered covariance matrices
    """
    def __init__(self, n_filters=None):
        self.n_filters = n_filters
        self.C_ref_ = None
        self.W_ = None
        self.mdm_ = None
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        # Step 1: Compute reference mean
        self.C_ref_ = affine_invariant_mean(X)
        
        # Step 2: Project to tangent space
        ts = TangentSpace(metric='affine-invariant')
        ts.C_ref_ = self.C_ref_
        S = ts.transform(X)
        
        # Step 3: Fisher LDA in tangent space
        n_features = S.shape[1]
        n_filters = self.n_filters or min(len(self.classes_) - 1, n_features)
        
        # Between-class scatter
        overall_mean = np.mean(S, axis=0)
        S_b = np.zeros((n_features, n_features))
        S_w = np.zeros((n_features, n_features))
        
        for c in self.classes_:
            mask = y == c
            S_c = S[mask]
            n_c = len(S_c)
            mean_c = np.mean(S_c, axis=0)
            diff = (mean_c - overall_mean).reshape(-1, 1)
            S_b += n_c * (diff @ diff.T)
            
            centered = S_c - mean_c
            S_w += centered.T @ centered
        
        # Regularize within-class scatter
        S_w += np.eye(n_features) * 1e-6
        
        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = la.eigh(S_b, S_w)
        idx = np.argsort(eigenvalues)[::-1]
        self.W_ = eigenvectors[:, idx[:n_filters]]
        
        # Step 4: Filter tangent vectors and project back to manifold
        S_filtered = S @ self.W_ @ self.W_.T
        
        # Reconstruct filtered SPD matrices
        X_filtered = self._tangent_to_spd(S_filtered)
        
        # Step 5: Fit MDM on filtered matrices
        self.mdm_ = MDM(metric='affine-invariant')
        self.mdm_.fit(X_filtered, y)
        
        return self
    
    def _tangent_to_spd(self, S_vectors):
        """Convert tangent space vectors back to SPD matrices."""
        C_ref_sqrt = sqrtm_spd(self.C_ref_)
        n = self.C_ref_.shape[0]
        n_features = n * (n + 1) // 2
        
        covs = []
        for vec in S_vectors:
            # Unvectorize: reconstruct symmetric matrix from upper triangle
            # We need to handle the sqrt(2) scaling of off-diagonals
            Y = np.zeros((n, n))
            idx_row, idx_col = np.triu_indices(n)
            
            # Trim or pad vector to expected size
            v = vec[:n_features] if len(vec) >= n_features else np.pad(vec, (0, n_features - len(vec)))
            
            Y[idx_row, idx_col] = v
            mask = idx_row != idx_col
            Y[idx_row[mask], idx_col[mask]] /= np.sqrt(2.0)
            Y = Y + Y.T - np.diag(np.diag(Y))
            
            # Map back to manifold: C = C_ref^{1/2} exp(Y) C_ref^{1/2}
            C = C_ref_sqrt @ expm_spd(Y) @ C_ref_sqrt
            covs.append(C)
        
        return np.array(covs)
    
    def predict(self, X):
        # Filter test data through the same pipeline
        ts = TangentSpace(metric='affine-invariant')
        ts.C_ref_ = self.C_ref_
        S = ts.transform(X)
        S_filtered = S @ self.W_ @ self.W_.T
        X_filtered = self._tangent_to_spd(S_filtered)
        return self.mdm_.predict(X_filtered)
    
    def predict_proba(self, X):
        ts = TangentSpace(metric='affine-invariant')
        ts.C_ref_ = self.C_ref_
        S = ts.transform(X)
        S_filtered = S @ self.W_ @ self.W_.T
        X_filtered = self._tangent_to_spd(S_filtered)
        return self.mdm_.predict_proba(X_filtered)


# ============================================================================
# Filter Bank Wrappers
# ============================================================================

class FilterBankCovariances(BaseEstimator, TransformerMixin):
    """Compute covariances per filter bank band."""
    
    def __init__(self, estimator='ledoit-wolf'):
        self.estimator = estimator
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """
        X: (trials, bands, samples, channels)
        Returns: (trials, bands, channels, channels)
        """
        lw = LedoitWolf()
        n_trials, n_bands = X.shape[0], X.shape[1]
        n_channels = X.shape[3]
        covs = np.zeros((n_trials, n_bands, n_channels, n_channels))
        
        for i in range(n_trials):
            for b in range(n_bands):
                trial = X[i, b]
                trial_centered = trial - np.mean(trial, axis=0)
                lw.fit(trial_centered)
                covs[i, b] = lw.covariance_
        
        return covs

class FilterBankTangentSpace(BaseEstimator, TransformerMixin):
    """Project multi-band covariance matrices to tangent space."""
    
    def __init__(self):
        self.ts_estimators = []
        
    def fit(self, X, y=None):
        """X: (trials, bands, channels, channels)"""
        n_bands = X.shape[1]
        self.ts_estimators = []
        for b in range(n_bands):
            ts = TangentSpace()
            ts.fit(X[:, b])
            self.ts_estimators.append(ts)
        return self
        
    def transform(self, X):
        features = []
        for b in range(X.shape[1]):
            features.append(self.ts_estimators[b].transform(X[:, b]))
        return np.concatenate(features, axis=1)

class FilterBankMDM(BaseEstimator, ClassifierMixin):
    """
    MDM classifier applied independently per frequency band.
    Final prediction uses sum of geodesic distances across bands.
    """
    def __init__(self):
        self.mdms_ = []
        self.classes_ = None
        
    def fit(self, X, y):
        """X: (trials, bands, channels, channels)"""
        self.classes_ = np.unique(y)
        n_bands = X.shape[1]
        self.mdms_ = []
        for b in range(n_bands):
            mdm = MDM(metric='affine-invariant')
            mdm.fit(X[:, b], y)
            self.mdms_.append(mdm)
        return self
        
    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
    
    def predict_proba(self, X):
        n_bands = X.shape[1]
        all_probas = np.zeros((X.shape[0], len(self.classes_)))
        for b in range(n_bands):
            all_probas += self.mdms_[b].predict_proba(X[:, b])
        all_probas /= n_bands
        return all_probas
