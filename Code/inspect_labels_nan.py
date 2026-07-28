import scipy.io as sio
import numpy as np

true_labels = sio.loadmat(r'c:\Users\jaip7\Downloads\madhan\BCI\true_labels\A01E.mat')['classlabel'].flatten()
print("Unique values in A01E true labels:", np.unique(true_labels))
print("Indices where label is NaN or negative:", np.where(np.isnan(true_labels) | (true_labels < 0)))
