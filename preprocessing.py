import numpy as np
from scipy.signal import butter, filtfilt

def common_average_reference(data):
    """
    Apply Common Average Reference (CAR) spatial filter.
    data shape: (trials, samples, channels)
    Returns CAR filtered data.
    """
    # Calculate the mean across all channels (axis=2) and subtract it
    mean_signal = np.mean(data, axis=2, keepdims=True)
    return data - mean_signal

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a zero-phase Butterworth bandpass filter.
    data shape: (trials, samples, channels) or (samples, channels)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    # Check data shape
    if data.ndim == 3:
        # (trials, samples, channels)
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                filtered_data[i, :, j] = filtfilt(b, a, data[i, :, j])
    elif data.ndim == 2:
        # (samples, channels)
        filtered_data = np.zeros_like(data)
        for j in range(data.shape[1]):
            filtered_data[:, j] = filtfilt(b, a, data[:, j])
    else:
        raise ValueError("Data must be 2D or 3D")
        
    return filtered_data

def apply_filter_bank(data, fs=250):
    """
    Apply multiple bandpass filters (filter banks).
    Common setup for MI: 9 bands (4-8, 8-12, 12-16, 16-20, 20-24, 24-28, 28-32, 32-36, 36-40)
    data shape: (trials, samples, channels)
    Returns: (bands, trials, samples, channels)
    """
    bands = [
        (4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
        (24, 28), (28, 32), (32, 36), (36, 40)
    ]
    
    fb_data = []
    for low, high in bands:
        fb_data.append(bandpass_filter(data, low, high, fs))
        
    # Return shape (trials, bands, samples, channels)
    return np.swapaxes(np.array(fb_data), 0, 1)
