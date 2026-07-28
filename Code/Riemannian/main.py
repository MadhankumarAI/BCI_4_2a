import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score
import warnings

# Add parent directory to path to import data_loader and preprocessing
sys.path.append(str(Path(__file__).parent.parent))

from data_loader import load_and_epoch_data, load_true_labels
from preprocessing import apply_filter_bank, common_average_reference
from riemann import FilterBankTangentSpace, precompute_covariances

warnings.filterwarnings("ignore")

def main():
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent.parent / "BCICIV-2a-mat"
    labels_dir = current_dir.parent.parent / "true_labels"
    
    subjects = [f"A0{i}" for i in range(1, 10)]
    accuracies = []
    kappas = []
    
    # 0.5s to 4.0s post-cue (optimal time window)
    start_sec = 2.5
    end_sec = 6.0
    
    print(f"Running Riemannian Tangent Space Pipeline...")
    print(f"Time Window: {start_sec-2.0}s to {end_sec-2.0}s post-cue")
    print("-" * 50)
    
    for subject in subjects:
        print(f"Processing Subject {subject}...")
        
        train_file = data_dir / f"{subject}T.mat"
        test_file = data_dir / f"{subject}E.mat"
        true_labels_file = labels_dir / f"{subject}E.mat"
        
        # --- Training ---
        X_train_raw, y_train, _ = load_and_epoch_data(train_file, start_sec=start_sec, end_sec=end_sec)
        
        # Apply CAR and Filter Bank
        X_train_car = common_average_reference(X_train_raw)
        X_train_fb = apply_filter_bank(X_train_car)
        
        # Precompute covariances
        covs_train = precompute_covariances(X_train_fb)
        
        # Setup Pipeline with Filter Bank Tangent Space + regularized Logistic Regression
        pipeline = Pipeline([
            ('fbts', FilterBankTangentSpace()),
            ('select', SelectKBest(f_classif)),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(solver='liblinear', penalty='l2', random_state=42))
        ])
        
        # Grid Search over feature selection k and classifier C
        param_grid = {
            'select__k': [50, 100, 150, 200, 'all'],
            'clf__C': [0.01, 0.1, 1.0, 10.0]
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=1)
        
        # Fit Grid Search using precomputed covariances
        grid_search.fit(covs_train, y_train)
        best_model = grid_search.best_estimator_
        
        print(f"  Best params: {grid_search.best_params_}")
        
        # --- Evaluation ---
        X_test_raw, _, test_indices = load_and_epoch_data(test_file, start_sec=start_sec, end_sec=end_sec)
        y_test_all = load_true_labels(true_labels_file)
        
        # Align evaluation labels with clean trials
        y_test_all = y_test_all[test_indices]
        
        # Filter evaluation trials to only Class 1 (Left) and Class 2 (Right)
        mask = (y_test_all == 1) | (y_test_all == 2)
        X_test_raw_filtered = X_test_raw[mask]
        y_test = y_test_all[mask]
        
        # Apply CAR and Filter Bank
        X_test_car = common_average_reference(X_test_raw_filtered)
        X_test_fb = apply_filter_bank(X_test_car)
        
        # Precompute test covariances
        covs_test = precompute_covariances(X_test_fb)
        
        # Predict and evaluate
        y_pred = best_model.predict(covs_test)
        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        accuracies.append(acc)
        kappas.append(kappa)
        
        print(f"  Accuracy: {acc*100:.2f}% | Kappa: {kappa:.4f}")
        
    print("-" * 50)
    print(f"Mean Accuracy: {np.mean(accuracies)*100:.2f}%")
    print(f"Mean Kappa: {np.mean(kappas):.4f}")

if __name__ == "__main__":
    main()
