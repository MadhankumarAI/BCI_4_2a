"""
Ensemble Pipeline: FBCSP + Riemannian TSM + MDM + Augmented Covariance
======================================================================

Combines four complementary classifiers using soft-voting (probability averaging):

1. FBCSP + Shrinkage LDA (captures band-specific spatial patterns)
2. Filter Bank Riemannian TSM + Logistic Regression (manifold-based features)
3. Filter Bank MDM (parameter-free geodesic classifier)
4. Augmented Covariance TSM + Logistic Regression (spatio-temporal dynamics)

Each model captures different aspects of the EEG signal, and their
ensemble is more robust than any individual model.
"""

import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_loader import load_and_epoch_data, load_true_labels
from preprocessing import apply_filter_bank, common_average_reference
from fbcsp import FBCSP

from advanced_riemann import (
    FilterBankCovariances, FilterBankTangentSpace, FilterBankMDM,
    AugmentedCovariances, TangentSpace
)

warnings.filterwarnings("ignore")


def train_fbcsp_model(X_train_fb, y_train):
    """Train FBCSP + LDA model with inner CV for hyperparameter selection."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -1
    best_k, best_m = 4, 4
    
    for k in [4, 6, 8, 10, 12, 16]:
        for m in [2, 4, 6]:
            scores = []
            for train_idx, val_idx in cv.split(X_train_fb, y_train):
                X_tr, X_val = X_train_fb[train_idx], X_train_fb[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                fbcsp = FBCSP(m_components=m, k_features=k)
                fbcsp.fit(X_tr, y_tr)
                
                scaler = StandardScaler()
                X_tr_f = scaler.fit_transform(fbcsp.transform(X_tr))
                X_val_f = scaler.transform(fbcsp.transform(X_val))
                
                lda = LDA(solver='lsqr', shrinkage='auto')
                lda.fit(X_tr_f, y_tr)
                scores.append(lda.score(X_val_f, y_val))
            
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_k, best_m = k, m
    
    # Refit on full training set
    fbcsp = FBCSP(m_components=best_m, k_features=best_k)
    fbcsp.fit(X_train_fb, y_train)
    scaler = StandardScaler()
    X_f = scaler.fit_transform(fbcsp.transform(X_train_fb))
    lda = LDA(solver='lsqr', shrinkage='auto')
    lda.fit(X_f, y_train)
    
    return fbcsp, scaler, lda, best_k, best_m


def train_riemannian_tsm(covs_train, y_train):
    """Train Filter Bank Tangent Space + Logistic Regression with inner CV."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -1
    best_k, best_C = 'all', 0.1
    
    for k in [50, 100, 150, 200, 'all']:
        for C in [0.01, 0.1, 1.0, 10.0]:
            scores = []
            for train_idx, val_idx in cv.split(covs_train[:, 0, :, :].reshape(len(covs_train), -1), y_train):
                covs_tr = covs_train[train_idx]
                covs_val = covs_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                fbts = FilterBankTangentSpace()
                fbts.fit(covs_tr)
                X_tr_f = fbts.transform(covs_tr)
                X_val_f = fbts.transform(covs_val)
                
                actual_k = k if k != 'all' else X_tr_f.shape[1]
                actual_k = min(actual_k, X_tr_f.shape[1])
                
                selector = SelectKBest(f_classif, k=actual_k)
                X_tr_f = selector.fit_transform(X_tr_f, y_tr)
                X_val_f = selector.transform(X_val_f)
                
                scaler = StandardScaler()
                X_tr_f = scaler.fit_transform(X_tr_f)
                X_val_f = scaler.transform(X_val_f)
                
                clf = LogisticRegression(solver='liblinear', penalty='l2', C=C, random_state=42)
                clf.fit(X_tr_f, y_tr)
                scores.append(clf.score(X_val_f, y_val))
            
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_k, best_C = k, C
    
    # Refit on full training set
    fbts = FilterBankTangentSpace()
    fbts.fit(covs_train)
    X_f = fbts.transform(covs_train)
    
    actual_k = best_k if best_k != 'all' else X_f.shape[1]
    actual_k = min(actual_k, X_f.shape[1])
    
    selector = SelectKBest(f_classif, k=actual_k)
    X_f = selector.fit_transform(X_f, y_train)
    scaler = StandardScaler()
    X_f = scaler.fit_transform(X_f)
    clf = LogisticRegression(solver='liblinear', penalty='l2', C=best_C, random_state=42)
    clf.fit(X_f, y_train)
    
    return fbts, selector, scaler, clf, best_k, best_C


def train_mdm(covs_train, y_train):
    """Train Filter Bank MDM classifier (parameter-free)."""
    mdm = FilterBankMDM()
    mdm.fit(covs_train, y_train)
    return mdm


def train_augmented_riemann(X_train_car, y_train):
    """Train Augmented Covariance TSM + Logistic Regression with inner CV."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = -1
    best_delays, best_step, best_C = 3, 4, 0.1
    
    for n_delays in [2, 3, 4]:
        for delay_step in [3, 5]:
            # Compute augmented covariances
            aug_cov = AugmentedCovariances(n_delays=n_delays, delay_step=delay_step)
            covs = aug_cov.fit_transform(X_train_car)
            
            for C in [0.01, 0.1, 1.0]:
                scores = []
                for train_idx, val_idx in cv.split(covs.reshape(len(covs), -1), y_train):
                    covs_tr, covs_val = covs[train_idx], covs[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    
                    ts = TangentSpace()
                    ts.fit(covs_tr)
                    X_tr_f = ts.transform(covs_tr)
                    X_val_f = ts.transform(covs_val)
                    
                    scaler = StandardScaler()
                    X_tr_f = scaler.fit_transform(X_tr_f)
                    X_val_f = scaler.transform(X_val_f)
                    
                    clf = LogisticRegression(solver='liblinear', penalty='l2', C=C, random_state=42)
                    clf.fit(X_tr_f, y_tr)
                    scores.append(clf.score(X_val_f, y_val))
                
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_delays, best_step, best_C = n_delays, delay_step, C
    
    # Refit on full training set
    aug_cov = AugmentedCovariances(n_delays=best_delays, delay_step=best_step)
    covs = aug_cov.fit_transform(X_train_car)
    
    ts = TangentSpace()
    ts.fit(covs)
    X_f = ts.transform(covs)
    
    scaler = StandardScaler()
    X_f = scaler.fit_transform(X_f)
    clf = LogisticRegression(solver='liblinear', penalty='l2', C=best_C, random_state=42)
    clf.fit(X_f, y_train)
    
    return aug_cov, ts, scaler, clf, best_delays, best_step, best_C


def main():
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent.parent / "BCICIV-2a-mat"
    labels_dir = current_dir.parent.parent / "true_labels"
    
    subjects = [f"A0{i}" for i in range(1, 10)]
    accuracies = []
    kappas = []
    
    start_sec = 2.5
    end_sec = 6.0
    
    print("=" * 60)
    print("ENSEMBLE: FBCSP + Riemannian TSM + MDM + Augmented Cov")
    print(f"Time Window: {start_sec-2.0}s to {end_sec-2.0}s post-cue")
    print("=" * 60)
    
    for subject in subjects:
        print(f"\nProcessing Subject {subject}...")
        
        train_file = data_dir / f"{subject}T.mat"
        test_file = data_dir / f"{subject}E.mat"
        true_labels_file = labels_dir / f"{subject}E.mat"
        
        # --- Load & Preprocess Training Data ---
        X_train_raw, y_train, _ = load_and_epoch_data(train_file, start_sec=start_sec, end_sec=end_sec)
        X_train_car = common_average_reference(X_train_raw)
        X_train_fb = apply_filter_bank(X_train_car)
        
        # Precompute filter bank covariances
        fb_cov = FilterBankCovariances()
        covs_train = fb_cov.transform(X_train_fb)
        
        # --- Train 4 Models ---
        print("  Training FBCSP + LDA...", end=" ", flush=True)
        fbcsp, fbcsp_scaler, fbcsp_lda, bk, bm = train_fbcsp_model(X_train_fb, y_train)
        print(f"(k={bk}, m={bm})")
        
        print("  Training Riemannian TSM + LR...", end=" ", flush=True)
        fbts, ts_selector, ts_scaler, ts_clf, tk, tC = train_riemannian_tsm(covs_train, y_train)
        print(f"(k={tk}, C={tC})")
        
        print("  Training Filter Bank MDM...", end=" ", flush=True)
        mdm = train_mdm(covs_train, y_train)
        print("(parameter-free)")
        
        print("  Training Augmented Cov TSM + LR...", end=" ", flush=True)
        aug_cov, aug_ts, aug_scaler, aug_clf, ad, ast, aC = train_augmented_riemann(X_train_car, y_train)
        print(f"(delays={ad}, step={ast}, C={aC})")
        
        # --- Load & Preprocess Test Data ---
        X_test_raw, _, test_indices = load_and_epoch_data(test_file, start_sec=start_sec, end_sec=end_sec)
        y_test_all = load_true_labels(true_labels_file)
        y_test_all = y_test_all[test_indices]
        
        mask = (y_test_all == 1) | (y_test_all == 2)
        X_test_raw_filtered = X_test_raw[mask]
        y_test = y_test_all[mask]
        
        X_test_car = common_average_reference(X_test_raw_filtered)
        X_test_fb = apply_filter_bank(X_test_car)
        covs_test = fb_cov.transform(X_test_fb)
        
        # --- Predict with each model ---
        # Model 1: FBCSP
        X_test_fbcsp = fbcsp_scaler.transform(fbcsp.transform(X_test_fb))
        proba_fbcsp = fbcsp_lda.predict_proba(X_test_fbcsp)
        
        # Model 2: Riemannian TSM
        X_test_ts = ts_scaler.transform(ts_selector.transform(fbts.transform(covs_test)))
        proba_tsm = ts_clf.predict_proba(X_test_ts)
        
        # Model 3: MDM
        proba_mdm = mdm.predict_proba(covs_test)
        
        # Model 4: Augmented Covariance
        covs_aug_test = aug_cov.transform(X_test_car)
        X_test_aug = aug_scaler.transform(aug_ts.transform(covs_aug_test))
        proba_aug = aug_clf.predict_proba(X_test_aug)
        
        # --- Soft Voting Ensemble ---
        # Weighted average: give more weight to models with higher expected accuracy
        proba_ensemble = (
            1.0 * proba_fbcsp + 
            1.0 * proba_tsm + 
            0.8 * proba_mdm + 
            1.0 * proba_aug
        ) / 3.8
        
        classes = fbcsp_lda.classes_
        y_pred = classes[np.argmax(proba_ensemble, axis=1)]
        
        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        accuracies.append(acc)
        kappas.append(kappa)
        
        # Also show individual model results
        y_fbcsp = classes[np.argmax(proba_fbcsp, axis=1)]
        y_tsm = classes[np.argmax(proba_tsm, axis=1)]
        y_mdm = mdm.classes_[np.argmax(proba_mdm, axis=1)]
        y_aug = aug_clf.classes_[np.argmax(proba_aug, axis=1)]
        
        k_fbcsp = cohen_kappa_score(y_test, y_fbcsp)
        k_tsm = cohen_kappa_score(y_test, y_tsm)
        k_mdm = cohen_kappa_score(y_test, y_mdm)
        k_aug = cohen_kappa_score(y_test, y_aug)
        
        print(f"  Individual Kappas: FBCSP={k_fbcsp:.4f} | TSM={k_tsm:.4f} | MDM={k_mdm:.4f} | AugCov={k_aug:.4f}")
        print(f"  >>> ENSEMBLE: Accuracy: {acc*100:.2f}% | Kappa: {kappa:.4f}")
        
    print("\n" + "=" * 60)
    print(f"ENSEMBLE Mean Accuracy: {np.mean(accuracies)*100:.2f}%")
    print(f"ENSEMBLE Mean Kappa: {np.mean(kappas):.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
