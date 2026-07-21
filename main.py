import numpy as np
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings

from data_loader import load_and_epoch_data, load_true_labels
from preprocessing import apply_filter_bank, common_average_reference
from fbcsp import FBCSP

# Suppress some potential sklearn warnings about rank-deficient covariance matrices during CV
warnings.filterwarnings("ignore")

def main():
    data_dir = Path(r"c:\Users\jaip7\Downloads\madhan\BCI\BCICIV-2a-mat")
    labels_dir = Path(r"c:\Users\jaip7\Downloads\madhan\BCI\true_labels")
    
    subjects = [f"A0{i}" for i in range(1, 10)]
    accuracies = []
    kappas = []
    
    for subject in subjects:
        print(f"Processing Subject {subject}...")
        
        train_file = data_dir / f"{subject}T.mat"
        test_file = data_dir / f"{subject}E.mat"
        true_labels_file = labels_dir / f"{subject}E.mat"
        
        # --- Training ---
        # Extract trials (2.5 to 5.5s relative to trial start is 0.5 to 3.5s after cue)
        X_train_raw, y_train = load_and_epoch_data(train_file, start_sec=2.5, end_sec=5.5)
        
        # Apply Common Average Reference (CAR)
        X_train_car = common_average_reference(X_train_raw)
        
        # Apply Filter Bank
        X_train_fb = apply_filter_bank(X_train_car)
        
        # Setup Pipeline with FBCSP and Shrinkage LDA
        pipeline = Pipeline([
            ('fbcsp', FBCSP(m_components=4)),
            ('scaler', StandardScaler()),
            ('lda', LDA(solver='lsqr', shrinkage='auto'))
        ])
        
        # Grid Search for best k_features using inner Cross-Validation
        param_grid = {
            'fbcsp__k_features': [4, 8, 12, 16, 20]
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=1)
        
        # Fit Grid Search
        grid_search.fit(X_train_fb, y_train)
        best_model = grid_search.best_estimator_
        
        print(f"  Best params: {grid_search.best_params_}")
        
        # --- Evaluation ---
        X_test_raw, _ = load_and_epoch_data(test_file, start_sec=2.5, end_sec=5.5)
        
        # Load true labels for evaluation
        y_test_all = load_true_labels(true_labels_file)
        
        # Filter evaluation trials to only Class 1 (Left) and Class 2 (Right)
        mask = (y_test_all == 1) | (y_test_all == 2)
        X_test_raw_filtered = X_test_raw[mask]
        y_test = y_test_all[mask]
        
        # Apply CAR and Filter Bank
        X_test_car = common_average_reference(X_test_raw_filtered)
        X_test_fb = apply_filter_bank(X_test_car)
        
        # Predict and evaluate
        y_pred = best_model.predict(X_test_fb)
        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        accuracies.append(acc)
        kappas.append(kappa)
        
        print(f"  Accuracy: {acc*100:.2f}% | Kappa: {kappa:.4f}")
        
    print("-" * 30)
    print(f"Mean Accuracy: {np.mean(accuracies)*100:.2f}%")
    print(f"Mean Kappa: {np.mean(kappas):.4f}")

if __name__ == "__main__":
    main()
