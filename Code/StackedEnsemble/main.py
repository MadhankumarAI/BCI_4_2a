"""
Stacked Ensemble with Multi-Scale Temporal Windows (Optimized)
==============================================================

This ultimate pipeline combines:
1. STACKING META-LEARNER (Logistic Regression on OOF predictions)
2. MULTI-SCALE TEMPORAL WINDOWS (Early, Full, Late phases)
3. DIVERSE BASE MODELS (FBCSP, Riemannian TSM, MDM, AugCov, SVM-RBF)

Optimization: Inner grid search is only performed once on the full training 
set. The found hyperparameters are then reused for the Out-Of-Fold (OOF) 
cross-validation, reducing training time from hours to minutes.
"""

import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from data_loader import load_and_epoch_data, load_true_labels
from preprocessing import apply_filter_bank, common_average_reference
from fbcsp import FBCSP

sys.path.append(str(Path(__file__).parent.parent / "Ensemble"))
from advanced_riemann import (
    FilterBankCovariances, FilterBankTangentSpace, FilterBankMDM,
    AugmentedCovariances, TangentSpace
)

warnings.filterwarnings("ignore")

# ============================================================================
# Time windows: (start_sec, end_sec) relative to trial onset (cue at 2.0s)
# ============================================================================
TIME_WINDOWS = [
    (2.5, 4.5),   # Early:  0.5s - 2.5s post-cue (ERD onset)
    (2.5, 6.0),   # Full:   0.5s - 4.0s post-cue (complete window)
    (3.5, 6.0),   # Late:   1.5s - 4.0s post-cue (sustained / ERS)
]

def precompute_fb_covariances(X_fb):
    fb_cov = FilterBankCovariances()
    return fb_cov.transform(X_fb)


# ============================================================================
# Base Model Factories
# ============================================================================

class FBCSPModel:
    def __init__(self, k=None, m=None):
        self.k = k
        self.m = m
        self.fbcsp = None
        self.scaler = None
        self.clf = None
    
    def fit(self, X_fb, y):
        if self.k is None or self.m is None:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            best_score, self.k, self.m = -1, 4, 2
            for k in [4, 8, 12]:
                for m in [2, 4]:
                    scores = []
                    for tr, va in cv.split(X_fb, y):
                        try:
                            fb = FBCSP(m_components=m, k_features=k)
                            fb.fit(X_fb[tr], y[tr])
                            sc = StandardScaler()
                            Xtr = sc.fit_transform(fb.transform(X_fb[tr]))
                            Xva = sc.transform(fb.transform(X_fb[va]))
                            lda = LDA(solver='lsqr', shrinkage='auto')
                            lda.fit(Xtr, y[tr])
                            scores.append(lda.score(Xva, y[va]))
                        except:
                            scores.append(0.0)
                    ms = np.mean(scores)
                    if ms > best_score:
                        best_score, self.k, self.m = ms, k, m
        
        self.fbcsp = FBCSP(m_components=self.m, k_features=self.k)
        self.fbcsp.fit(X_fb, y)
        self.scaler = StandardScaler()
        Xf = self.scaler.fit_transform(self.fbcsp.transform(X_fb))
        self.clf = LDA(solver='lsqr', shrinkage='auto')
        self.clf.fit(Xf, y)
    
    def predict_proba(self, X_fb):
        Xf = self.scaler.transform(self.fbcsp.transform(X_fb))
        return self.clf.predict_proba(Xf)


class RiemannTSMModel:
    def __init__(self, k=None, C=None):
        self.k = k
        self.C = C
        self.fbts = None
        self.selector = None
        self.scaler = None
        self.clf = None
    
    def fit(self, covs, y):
        if self.k is None or self.C is None:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            best_score, self.k, self.C = -1, 'all', 0.1
            for k in [100, 200, 'all']:
                for C in [0.01, 0.1, 1.0]:
                    scores = []
                    for tr, va in cv.split(covs.reshape(len(covs), -1), y):
                        fb = FilterBankTangentSpace()
                        fb.fit(covs[tr])
                        Xtr = fb.transform(covs[tr])
                        Xva = fb.transform(covs[va])
                        ak = k if k != 'all' else Xtr.shape[1]
                        ak = min(ak, Xtr.shape[1])
                        sel = SelectKBest(f_classif, k=ak)
                        Xtr = sel.fit_transform(Xtr, y[tr])
                        Xva = sel.transform(Xva)
                        sc = StandardScaler()
                        Xtr = sc.fit_transform(Xtr)
                        Xva = sc.transform(Xva)
                        clf = LogisticRegression(solver='saga', penalty='elasticnet', 
                                               l1_ratio=0.5, C=C, random_state=42, max_iter=500)
                        clf.fit(Xtr, y[tr])
                        scores.append(clf.score(Xva, y[va]))
                    ms = np.mean(scores)
                    if ms > best_score:
                        best_score, self.k, self.C = ms, k, C
        
        self.fbts = FilterBankTangentSpace()
        self.fbts.fit(covs)
        Xf = self.fbts.transform(covs)
        ak = self.k if self.k != 'all' else Xf.shape[1]
        ak = min(ak, Xf.shape[1])
        self.selector = SelectKBest(f_classif, k=ak)
        Xf = self.selector.fit_transform(Xf, y)
        self.scaler = StandardScaler()
        Xf = self.scaler.fit_transform(Xf)
        self.clf = LogisticRegression(solver='saga', penalty='elasticnet',
                                     l1_ratio=0.5, C=self.C, random_state=42, max_iter=500)
        self.clf.fit(Xf, y)
    
    def predict_proba(self, covs):
        Xf = self.scaler.transform(self.selector.transform(self.fbts.transform(covs)))
        return self.clf.predict_proba(Xf)


class MDMModel:
    def __init__(self):
        self.mdm = None
    
    def fit(self, covs, y):
        self.mdm = FilterBankMDM()
        self.mdm.fit(covs, y)
    
    def predict_proba(self, covs):
        return self.mdm.predict_proba(covs)


class AugCovModel:
    def __init__(self, d=None, s=None, C=None):
        self.d = d
        self.s = s
        self.C = C
        self.aug_cov = None
        self.ts = None
        self.scaler = None
        self.clf = None
    
    def fit(self, X_car, y):
        if self.d is None or self.s is None or self.C is None:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            best_score, self.d, self.s, self.C = -1, 3, 4, 0.1
            for d in [2, 4]:
                for s in [3, 5]:
                    aug = AugmentedCovariances(n_delays=d, delay_step=s)
                    covs = aug.fit_transform(X_car)
                    for C in [0.01, 0.1, 1.0]:
                        scores = []
                        for tr, va in cv.split(covs.reshape(len(covs), -1), y):
                            ts = TangentSpace()
                            ts.fit(covs[tr])
                            Xtr = ts.transform(covs[tr])
                            Xva = ts.transform(covs[va])
                            sc = StandardScaler()
                            Xtr = sc.fit_transform(Xtr)
                            Xva = sc.transform(Xva)
                            clf = LogisticRegression(solver='liblinear', C=C, random_state=42)
                            clf.fit(Xtr, y[tr])
                            scores.append(clf.score(Xva, y[va]))
                        ms = np.mean(scores)
                        if ms > best_score:
                            best_score, self.d, self.s, self.C = ms, d, s, C
        
        self.aug_cov = AugmentedCovariances(n_delays=self.d, delay_step=self.s)
        covs = self.aug_cov.fit_transform(X_car)
        self.ts = TangentSpace()
        self.ts.fit(covs)
        Xf = self.ts.transform(covs)
        self.scaler = StandardScaler()
        Xf = self.scaler.fit_transform(Xf)
        self.clf = LogisticRegression(solver='liblinear', C=self.C, random_state=42)
        self.clf.fit(Xf, y)
    
    def predict_proba(self, X_car):
        covs = self.aug_cov.transform(X_car)
        Xf = self.scaler.transform(self.ts.transform(covs))
        return self.clf.predict_proba(Xf)


class SVMRBFModel:
    def __init__(self, C=None):
        self.C = C
        self.fbts = None
        self.selector = None
        self.scaler = None
        self.clf = None
    
    def fit(self, covs, y):
        self.fbts = FilterBankTangentSpace()
        self.fbts.fit(covs)
        Xall = self.fbts.transform(covs)
        
        self.selector = SelectKBest(f_classif, k=min(100, Xall.shape[1]))
        Xall = self.selector.fit_transform(Xall, y)
        self.scaler = StandardScaler()
        Xall = self.scaler.fit_transform(Xall)
        
        if self.C is None:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            best_score, self.C = -1, 1.0
            for C in [0.1, 1.0, 10.0]:
                scores = []
                for tr, va in cv.split(Xall, y):
                    clf = SVC(kernel='rbf', C=C, gamma='scale', probability=True, random_state=42)
                    clf.fit(Xall[tr], y[tr])
                    scores.append(clf.score(Xall[va], y[va]))
                ms = np.mean(scores)
                if ms > best_score:
                    best_score, self.C = ms, C
        
        self.clf = SVC(kernel='rbf', C=self.C, gamma='scale', probability=True, random_state=42)
        self.clf.fit(Xall, y)
    
    def predict_proba(self, covs):
        Xf = self.scaler.transform(self.selector.transform(self.fbts.transform(covs)))
        return self.clf.predict_proba(Xf)


def main():
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent.parent / "BCICIV-2a-mat"
    labels_dir = current_dir.parent.parent / "true_labels"
    
    subjects = [f"A0{i}" for i in range(1, 10)]
    accuracies = []
    kappas = []
    
    print("=" * 65)
    print("STACKED ENSEMBLE: 5 Models × 3 Time Windows = 15 Base Learners")
    print("Optimization: Hyperparameters found on full train set are reused in CV")
    print("=" * 65)
    
    for subject in subjects:
        print(f"\n{'='*65}")
        print(f"Processing Subject {subject}...")
        print(f"{'='*65}")
        
        train_file = data_dir / f"{subject}T.mat"
        test_file = data_dir / f"{subject}E.mat"
        true_labels_file = labels_dir / f"{subject}E.mat"
        
        # Meta-features
        all_train_meta = []
        all_test_meta = []
        y_train = None
        y_test = None
        
        # Store found hyperparameters
        models_hyperparams = {i: {} for i in range(len(TIME_WINDOWS))}
        
        # Phase 1: Train on full training set to find hyperparameters and generate test predictions
        for wi, (t_start, t_end) in enumerate(TIME_WINDOWS):
            win_name = f"Window {wi+1} ({t_start-2.0:.1f}s-{t_end-2.0:.1f}s)"
            print(f"\n  {win_name}")
            
            X_train_raw, y_train_w, _ = load_and_epoch_data(train_file, start_sec=t_start, end_sec=t_end)
            y_train = y_train_w
            X_train_car = common_average_reference(X_train_raw)
            X_train_fb = apply_filter_bank(X_train_car)
            covs_train = precompute_fb_covariances(X_train_fb)
            
            X_test_raw, _, test_indices = load_and_epoch_data(test_file, start_sec=t_start, end_sec=t_end)
            y_test_all = load_true_labels(true_labels_file)
            y_test_all = y_test_all[test_indices]
            mask = (y_test_all == 1) | (y_test_all == 2)
            X_test_raw_filtered = X_test_raw[mask]
            y_test = y_test_all[mask]
            
            X_test_car = common_average_reference(X_test_raw_filtered)
            X_test_fb = apply_filter_bank(X_test_car)
            covs_test = precompute_fb_covariances(X_test_fb)
            
            # M1: FBCSP
            print(f"    Training FBCSP...", end=" ", flush=True)
            m1 = FBCSPModel(); m1.fit(X_train_fb, y_train)
            models_hyperparams[wi]['fbcsp'] = (m1.k, m1.m)
            all_train_meta.append(m1.predict_proba(X_train_fb))
            all_test_meta.append(m1.predict_proba(X_test_fb))
            print("done")
            
            # M2: TSM
            print(f"    Training TSM+EN...", end=" ", flush=True)
            m2 = RiemannTSMModel(); m2.fit(covs_train, y_train)
            models_hyperparams[wi]['tsm'] = (m2.k, m2.C)
            all_train_meta.append(m2.predict_proba(covs_train))
            all_test_meta.append(m2.predict_proba(covs_test))
            print("done")
            
            # M3: MDM
            print(f"    Training MDM...", end=" ", flush=True)
            m3 = MDMModel(); m3.fit(covs_train, y_train)
            all_train_meta.append(m3.predict_proba(covs_train))
            all_test_meta.append(m3.predict_proba(covs_test))
            print("done")
            
            # M4: AugCov
            print(f"    Training AugCov...", end=" ", flush=True)
            m4 = AugCovModel(); m4.fit(X_train_car, y_train)
            models_hyperparams[wi]['augcov'] = (m4.d, m4.s, m4.C)
            all_train_meta.append(m4.predict_proba(X_train_car))
            all_test_meta.append(m4.predict_proba(X_test_car))
            print("done")
            
            # M5: SVM
            print(f"    Training SVM-RBF...", end=" ", flush=True)
            m5 = SVMRBFModel(); m5.fit(covs_train, y_train)
            models_hyperparams[wi]['svm'] = (m5.C,)
            all_train_meta.append(m5.predict_proba(covs_train))
            all_test_meta.append(m5.predict_proba(covs_test))
            print("done")
            
        X_meta_train_full = np.concatenate(all_train_meta, axis=1)
        X_meta_test = np.concatenate(all_test_meta, axis=1)
        
        # Phase 2: Generate OOF predictions for Meta-Learner (using found hyperparams)
        print("\n  Generating OOF predictions (fast mode)...", end=" ", flush=True)
        cv_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_meta = np.zeros_like(X_meta_train_full)
        
        for fold_idx, (tr_idx, va_idx) in enumerate(cv_meta.split(np.zeros(len(y_train)), y_train)):
            fold_val_meta = []
            
            for wi, (t_start, t_end) in enumerate(TIME_WINDOWS):
                X_train_raw, y_train_w, _ = load_and_epoch_data(train_file, start_sec=t_start, end_sec=t_end)
                X_train_car = common_average_reference(X_train_raw)
                X_train_fb = apply_filter_bank(X_train_car)
                covs_train_all = precompute_fb_covariances(X_train_fb)
                
                X_fb_tr, X_fb_va = X_train_fb[tr_idx], X_train_fb[va_idx]
                covs_tr, covs_va = covs_train_all[tr_idx], covs_train_all[va_idx]
                X_car_tr, X_car_va = X_train_car[tr_idx], X_train_car[va_idx]
                y_tr = y_train_w[tr_idx]
                
                k, m = models_hyperparams[wi]['fbcsp']
                fm1 = FBCSPModel(k=k, m=m); fm1.fit(X_fb_tr, y_tr)
                fold_val_meta.append(fm1.predict_proba(X_fb_va))
                
                k_ts, c_ts = models_hyperparams[wi]['tsm']
                fm2 = RiemannTSMModel(k=k_ts, C=c_ts); fm2.fit(covs_tr, y_tr)
                fold_val_meta.append(fm2.predict_proba(covs_va))
                
                fm3 = MDMModel(); fm3.fit(covs_tr, y_tr)
                fold_val_meta.append(fm3.predict_proba(covs_va))
                
                d_ag, s_ag, c_ag = models_hyperparams[wi]['augcov']
                fm4 = AugCovModel(d=d_ag, s=s_ag, C=c_ag); fm4.fit(X_car_tr, y_tr)
                fold_val_meta.append(fm4.predict_proba(X_car_va))
                
                c_svm, = models_hyperparams[wi]['svm']
                fm5 = SVMRBFModel(C=c_svm); fm5.fit(covs_tr, y_tr)
                fold_val_meta.append(fm5.predict_proba(covs_va))
                
            oof_meta[va_idx] = np.concatenate(fold_val_meta, axis=1)
        print("done")
        
        # Phase 3: Train Meta-Learner
        meta_scaler = StandardScaler()
        oof_scaled = meta_scaler.fit_transform(oof_meta)
        X_meta_test_scaled = meta_scaler.transform(X_meta_test)
        
        best_meta_score, best_meta_C = -1, 1.0
        for C in [0.01, 0.1, 1.0, 10.0]:
            scores = []
            for tr_idx, va_idx in cv_meta.split(oof_scaled, y_train):
                clf = LogisticRegression(solver='liblinear', C=C, random_state=42)
                clf.fit(oof_scaled[tr_idx], y_train[tr_idx])
                scores.append(clf.score(oof_scaled[va_idx], y_train[va_idx]))
            ms = np.mean(scores)
            if ms > best_meta_score:
                best_meta_score, best_meta_C = ms, C
                
        meta_clf = LogisticRegression(solver='liblinear', C=best_meta_C, random_state=42)
        meta_clf.fit(oof_scaled, y_train)
        
        y_pred = meta_clf.predict(X_meta_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        accuracies.append(acc)
        kappas.append(kappa)
        
        print(f"  >>> STACKED ENSEMBLE: Accuracy: {acc*100:.2f}% | Kappa: {kappa:.4f}")
    
    print("\n" + "=" * 65)
    print(f"STACKED ENSEMBLE Mean Accuracy: {np.mean(accuracies)*100:.2f}%")
    print(f"STACKED ENSEMBLE Mean Kappa:    {np.mean(kappas):.4f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
