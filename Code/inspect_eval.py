import scipy.io as sio
import numpy as np

mat = sio.loadmat(r'c:\Users\jaip7\Downloads\madhan\BCI\BCICIV-2a-mat\A01E.mat')
data = mat['data']
run = data[0, 3] # run 3
print("Fields in eval run:", run.dtype.names)
if 'artifacts' in run.dtype.names:
    print("Artifacts shape:", run['artifacts'][0, 0].shape)
    print("Num artifacts in this run:", np.sum(run['artifacts'][0, 0]))
