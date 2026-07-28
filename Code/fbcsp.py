import numpy as np
import scipy.linalg as la
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif

class CSP(BaseEstimator, TransformerMixin):
    """
    Common Spatial Pattern (CSP) feature extraction.
    Vectorized and optimized for high-performance execution.
    """
    def __init__(self, m_components=4):
        self.m_components = m_components
        self.filters_ = None

    def fit(self, X, y):
        """
        X: array of shape (trials, samples, channels)
        y: array of shape (trials,) containing class labels (1 and 2)
        """
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP requires exactly two classes.")
            
        c1, c2 = classes[0], classes[1]
        
        X1 = X[y == c1]
        X2 = X[y == c2]
        
        # Calculate covariance matrices
        cov1 = self._compute_covariance(X1)
        cov2 = self._compute_covariance(X2)
        
        # Solve generalized eigenvalue problem: cov1 * W = lambda * (cov1 + cov2) * W
        eigen_values, eigen_vectors = la.eigh(cov1, cov1 + cov2)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigen_values)[::-1]
        self.filters_ = eigen_vectors[:, idx]
        return self
        
    def transform(self, X):
        """
        Extract log-variance features. Vectorized across all trials.
        X: (trials, samples, channels)
        """
        if self.filters_ is None:
            raise RuntimeError("CSP not fitted.")
            
        # Select first m/2 and last m/2 filters
        m_half = self.m_components // 2
        W = np.concatenate([self.filters_[:, :m_half], self.filters_[:, -m_half:]], axis=1)
        
        # Vectorized projection: (trials, samples, channels) @ (channels, m) -> (trials, samples, m)
        projected = np.matmul(X, W)
        
        # Calculate variance along the time axis (samples, axis=1)
        var = np.var(projected, axis=1)
        
        # Calculate log-variance features normalized by sum of variances
        log_var = np.log(var / np.sum(var, axis=1, keepdims=True))
        return log_var
        
    def _compute_covariance(self, X_class):
        """
        Compute average covariance matrix across trials using vectorized np.einsum.
        X_class: (trials, samples, channels)
        """
        n_samples = X_class.shape[1]
        X_centered = X_class - np.mean(X_class, axis=1, keepdims=True)
        
        # Covariance: (1/(samples-1)) * (X_centered.T @ X_centered) for each trial
        covs = np.einsum('tsf,tsc->tfc', X_centered, X_centered) / (n_samples - 1)
        
        # Mean across trials
        cov_mean = np.mean(covs, axis=0)
        
        # Add small regularization to diagonal for numerical stability
        cov_mean += np.eye(cov_mean.shape[0]) * 1e-6
        return cov_mean

class FBCSP(BaseEstimator, TransformerMixin):
    """
    Filter Bank Common Spatial Pattern (FBCSP)
    """
    def __init__(self, m_components=4, k_features=4):
        self.m_components = m_components
        self.k_features = k_features
        self.csps = []
        self.feature_selector = SelectKBest(f_classif, k=self.k_features)
        
    def set_params(self, **params):
        super().set_params(**params)
        self.feature_selector = SelectKBest(f_classif, k=self.k_features)
        return self
        
    def fit(self, X, y):
        """
        X: array of shape (trials, bands, samples, channels)
        """
        self.csps = []
        n_bands = X.shape[1]
        
        all_features = []
        for b in range(n_bands):
            csp = CSP(m_components=self.m_components)
            csp.fit(X[:, b], y)
            self.csps.append(csp)
            
            features_b = csp.transform(X[:, b])
            all_features.append(features_b)
            
        # Concatenate features from all bands (trials, bands * m_components)
        X_features = np.concatenate(all_features, axis=1)
        
        # Fit feature selector
        self.feature_selector.fit(X_features, y)
        return self
        
    def transform(self, X):
        """
        X: array of shape (trials, bands, samples, channels)
        """
        n_bands = X.shape[1]
        
        all_features = []
        for b in range(n_bands):
            features_b = self.csps[b].transform(X[:, b])
            all_features.append(features_b)
            
        X_features = np.concatenate(all_features, axis=1)
        
        # Select best features
        return self.feature_selector.transform(X_features)
