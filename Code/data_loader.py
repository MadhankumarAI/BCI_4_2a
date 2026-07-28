import numpy as np
import scipy.io as sio
from pathlib import Path

def load_and_epoch_data(file_path: Path, start_sec=0.5, end_sec=2.5, fs=250):
    """
    Load data from BCI Competition IV 2a MAT file and epoch it into trials.
    Extracts only Left (class 1) and Right (class 2) trials.
    Filters out EOG/muscle artifacts and returns the trial indices to align labels.
    """
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
        
        # Check if this run has trials
        if 'trial' not in run_data.dtype.names:
            continue
            
        trial_val = run_data['trial']
        if trial_val.size == 0:
            continue
            
        X_run = run_data['X'][0, 0] # shape (samples, channels)
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
            
            # If training, only keep class 1 and 2 non-artifact trials
            # If evaluation, keep all non-artifact trials (labels are loaded from true_labels in main.py)
            if not is_artifact:
                if is_eval or (label in [1, 2]):
                    start = pos + start_offset
                    end = pos + end_offset
                    # Only use first 22 channels (EEG) and handle NaNs
                    trial = np.nan_to_num(X_run[start:end, :22])
                    all_trials.append(trial)
                    all_labels.append(label)
                    trial_indices.append(global_trial_idx)
                    
            global_trial_idx += 1
                
    return np.array(all_trials), np.array(all_labels), np.array(trial_indices)

def load_true_labels(file_path: Path):
    """
    Load true labels for the evaluation set.
    """
    mat = sio.loadmat(file_path)
    return mat['classlabel'].flatten()
