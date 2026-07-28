import scipy.io as sio
import numpy as np

mat = sio.loadmat(r'c:\Users\jaip7\Downloads\madhan\BCI\BCICIV-2a-mat\A01T.mat')
data = mat['data']
run = data[0, 3] # run 3 (first task run)
trial = run['trial'][0, 0].flatten()
print("First 10 trial positions:", trial[:10])
print("Differences between consecutive trials:", np.diff(trial[:10]))
print("Fs:", run['fs'][0, 0][0, 0])
if 'y' in run.dtype.names:
    print("Y shape:", run['y'][0, 0].shape)
