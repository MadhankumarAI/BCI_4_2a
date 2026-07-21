import numpy as np
import scipy.io as sio
from pathlib import Path

def load_and_epoch_data(file_path: Path, start_sec=0.5, end_sec=2.5, fs=250):
    """
    Load data from BCI Competition IV 2a MAT file and epoch it into trials.
    Extracts only Left (class 1) and Right (class 2) trials.
    """
    mat = sio.loadmat(file_path)
    data = mat['data']
    
    is_eval = 'E.mat' in file_path.name
    
    start_offset = int(start_sec * fs)
    end_offset = int(end_sec * fs)
    
    all_trials = []
    all_labels = []
    
    for i in range(data.shape[1]):
        run_data = data[0, i]
        
        # Check if this run has trials
        if 'trial' not in run_data.dtype.names:
            continue
            
        trial_val = run_data['trial']
        if trial_val.size == 0:
            continue
            
        X_run = run_data['X'][0, 0] # shape (samples, channels)
        trial_pos = trial_val[0, 0].flatten()
        
        # y might be empty for evaluation set, or contain NaNs/unknowns
        # If it's a training file, we can filter by class 1 and 2 right away.
        if not is_eval and 'y' in run_data.dtype.names and run_data['y'].size > 0:
            y_run = run_data['y'][0, 0].flatten()
            
            # Extract artifacts if present
            artifacts = None
            if 'artifacts' in run_data.dtype.names and run_data['artifacts'].size > 0:
                artifacts = run_data['artifacts'][0, 0].flatten()
                
            for j, pos in enumerate(trial_pos):
                label = y_run[j]
                if label in [1, 2]: # Left or Right
                    if artifacts is not None and artifacts[j] == 1:
                        continue # Skip artifact
                        
                    start = pos + start_offset
                    end = pos + end_offset
                    # Only use first 22 channels (EEG) and handle NaNs
                    trial = np.nan_to_num(X_run[start:end, :22])
                    all_trials.append(trial)
                    all_labels.append(label)
        else:
            # For evaluation set, we load all trials.
            # Filtering will happen in main.py using true_labels.
            for j, pos in enumerate(trial_pos):
                start = pos + start_offset
                end = pos + end_offset
                trial = np.nan_to_num(X_run[start:end, :22])
                all_trials.append(trial)
                all_labels.append(0) # Dummy label for now
                
    return np.array(all_trials), np.array(all_labels)

def load_true_labels(file_path: Path):
    """
    Load true labels for the evaluation set.
    """
    mat = sio.loadmat(file_path)
    return mat['classlabel'].flatten()
