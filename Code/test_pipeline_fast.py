import numpy as np
from pathlib import Path
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
import time

# Precompute LedoitWolf covariance for each trial
def precompute_covariances(X):
    """
    X: (trials, bands, samples, channels)
    Returns: (trials, bands, channels, channels)
    """
    X_centered = X - np.mean(X, axis=2, keepdims=True)
    n_samples = X.shape[2]
    covs = np.einsum('tbsf,tbsc->tbfc', X_centered, X_centered) / (n_samples - 1)
    return covs

# Fast CSP using precomputed covariances
class FastCSP(object):
    def __init__(self, m_components=4):
        self.m_components = m_components
        self.filters_ = None

    def fit_with_covs(self, covs, y):
        """
        covs: (trials, channels, channels)
        y: (trials,)
        """
        classes = np.unique(y)
        c1, c2 = classes[0], classes[1]
        
        cov1_mean = np.mean(covs[y == c1], axis=0)
        cov2_mean = np.mean(covs[y == c2], axis=0)
        
        # Regularization
        cov1_mean += np.eye(cov1_mean.shape[0]) * 1e-6
        cov2_mean += np.eye(cov2_mean.shape[0]) * 1e-6
        
        import scipy.linalg as la
        eigen_values, eigen_vectors = la.eigh(cov1_mean, cov1_mean + cov2_mean)
        idx = np.argsort(eigen_values)[::-1]
        self.filters_ = eigen_vectors[:, idx]
        return self

    def transform(self, X):
        """
        X: (trials, samples, channels)
        """
        m_half = self.m_components // 2
        W = np.concatenate([self.filters_[:, :m_half], self.filters_[:, -m_half:]], axis=1)
        projected = np.matmul(X, W)
        var = np.var(projected, axis=1)
        log_var = np.log(var / np.sum(var, axis=1, keepdims=True))
        return log_var

class FastFBCSP(object):
    def __init__(self, m_components=4, k_features=4):
        self.m_components = m_components
        self.k_features = k_features
        self.csps = []
        self.feature_selector = SelectKBest(f_classif, k=self.k_features)

    def fit_with_covs(self, X, covs, y):
        """
        X: (trials, bands, samples, channels)
        covs: (trials, bands, channels, channels)
        """
        self.csps = []
        n_bands = X.shape[1]
        all_features = []
        for b in range(n_bands):
            csp = FastCSP(m_components=self.m_components)
            csp.fit_with_covs(covs[:, b], y)
            self.csps.append(csp)
            features_b = csp.transform(X[:, b])
            all_features.append(features_b)
            
        X_features = np.concatenate(all_features, axis=1)
        self.feature_selector.fit(X_features, y)
        return self
        
    def transform(self, X):
        n_bands = X.shape[1]
        all_features = []
        for b in range(n_bands):
            features_b = self.csps[b].transform(X[:, b])
            all_features.append(features_b)
        X_features = np.concatenate(all_features, axis=1)
        return self.feature_selector.transform(X_features)

# Load data with artifacts filtered
def load_and_epoch_data_new(file_path: Path, start_sec=0.5, end_sec=2.5, fs=250):
    mat = sio.loadmat(file_path)
    data = mat['data']
    
    is_eval = 'E.mat' in file_path.name
    
    start_offset = int(start_sec * fs)
    end_offset = int(end_sec * fs)
    
    all_trials = []
    all_labels = []
    trial_indices = []
    
    global_trial_idx = 0
    for i in range(data.shape[1]):
        run_data = data[0, i]
        if 'trial' not in run_data.dtype.names:
            continue
        trial_val = run_data['trial']
        if trial_val.size == 0:
            continue
            
        X_run = run_data['X'][0, 0]
        trial_pos = trial_val[0, 0].flatten()
        
        y_run = None
        if 'y' in run_data.dtype.names and run_data['y'].size > 0:
            y_run = run_data['y'][0, 0].flatten()
            
        artifacts = None
        if 'artifacts' in run_data.dtype.names and run_data['artifacts'].size > 0:
            artifacts = run_data['artifacts'][0, 0].flatten()
            
        for j, pos in enumerate(trial_pos):
            is_artifact = (artifacts is not None and artifacts[j] == 1)
            label = y_run[j] if y_run is not None else 0
            
            if not is_artifact:
                if is_eval or (label in [1, 2]):
                    start = pos + start_offset
                    end = pos + end_offset
                    trial = np.nan_to_num(X_run[start:end, :22])
                    all_trials.append(trial)
                    all_labels.append(label)
                    trial_indices.append(global_trial_idx)
                
            global_trial_idx += 1
            
    return np.array(all_trials), np.array(all_labels), np.array(trial_indices)

def load_true_labels(file_path: Path):
    mat = sio.loadmat(file_path)
    return mat['classlabel'].flatten()

from preprocessing import apply_filter_bank, common_average_reference

def run_subject_eval(subject, start_sec, end_sec):
    data_dir = Path(r"c:\Users\jaip7\Downloads\madhan\BCI\BCICIV-2a-mat")
    labels_dir = Path(r"c:\Users\jaip7\Downloads\madhan\BCI\true_labels")
    
    train_file = data_dir / f"{subject}T.mat"
    test_file = data_dir / f"{subject}E.mat"
    true_labels_file = labels_dir / f"{subject}E.mat"
    
    X_train_raw, y_train, _ = load_and_epoch_data_new(train_file, start_sec=start_sec, end_sec=end_sec)
    
    # CAR & Filter Bank
    X_train_car = common_average_reference(X_train_raw)
    X_train_fb = apply_filter_bank(X_train_car)
    
    # Precompute trial covariances
    covs_train = precompute_covariances(X_train_fb)
    
    # Grid search cross-validation on train set
    best_score = -1
    best_k = 4
    best_m = 4
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Fast manual grid search
    for k in [4, 6, 8, 10, 12, 16]:
        for m in [2, 4, 6]:
            scores = []
            for train_idx, val_idx in cv.split(X_train_fb, y_train):
                X_tr, X_val = X_train_fb[train_idx], X_train_fb[val_idx]
                covs_tr = covs_train[train_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                fbcsp = FastFBCSP(m_components=m, k_features=k)
                fbcsp.fit_with_covs(X_tr, covs_tr, y_tr)
                
                scaler = StandardScaler()
                X_tr_f = scaler.fit_transform(fbcsp.transform(X_tr))
                X_val_f = scaler.transform(fbcsp.transform(X_val))
                
                lda = LDA(solver='lsqr', shrinkage='auto')
                lda.fit(X_tr_f, y_tr)
                
                scores.append(lda.score(X_val_f, y_val))
                
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
                best_m = m
                
    # Fit on full training set using best parameters
    fbcsp = FastFBCSP(m_components=best_m, k_features=best_k)
    fbcsp.fit_with_covs(X_train_fb, covs_train, y_train)
    scaler = StandardScaler()
    X_train_f = scaler.fit_transform(fbcsp.transform(X_train_fb))
    lda = LDA(solver='lsqr', shrinkage='auto')
    lda.fit(X_train_f, y_train)
    
    # Evaluate on test set
    X_test_raw, _, test_indices = load_and_epoch_data_new(test_file, start_sec=start_sec, end_sec=end_sec)
    y_test_all = load_true_labels(true_labels_file)
    y_test_all = y_test_all[test_indices]
    
    mask = (y_test_all == 1) | (y_test_all == 2)
    X_test_raw_filtered = X_test_raw[mask]
    y_test = y_test_all[mask]
    
    X_test_car = common_average_reference(X_test_raw_filtered)
    X_test_fb = apply_filter_bank(X_test_car)
    
    X_test_f = scaler.transform(fbcsp.transform(X_test_fb))
    y_pred = lda.predict(X_test_f)
    
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    return acc, kappa, best_k, best_m, best_score

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    # Let's test a few time windows
    windows = [(2.5, 4.5), (2.5, 5.0), (2.5, 5.5), (2.5, 6.0)]
    
    for w in windows:
        t0 = time.time()
        print(f"\n--- Testing window: {w[0]-2.0}s to {w[1]-2.0}s after cue ---")
        accuracies = []
        kappas = []
        for i in range(1, 10):
            subject = f"A0{i}"
            acc, kappa, k, m, val_score = run_subject_eval(subject, w[0], w[1])
            accuracies.append(acc)
            kappas.append(kappa)
            print(f"  {subject}: Acc: {acc*100:.2f}% | Kappa: {kappa:.4f} (best_k={k}, best_m={m}, val={val_score:.4f})")
        print(f"  Mean Accuracy: {np.mean(accuracies)*100:.2f}%")
        print(f"  Mean Kappa: {np.mean(kappas):.4f}")
        print(f"  Time taken: {time.time() - t0:.2f}s")
