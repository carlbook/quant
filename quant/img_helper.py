#!/home/carl/miniconda3/envs/trading_pytorch/bin/python
import os
import h5py
import numpy as np
import math

dest_dir = 'C004_images/'
os.chdir('/mnt/data/trading/datasets/')

# i made this because the 2d stock chart images did not fit in RAM. This will make ~2.1M files occupying ~40GB

with h5py.File('CRSPdsf62_trn_C004.hdf5', 'r') as datafile:
    os.chdir(os.path.join(dest_dir))
    dates = datafile['date'][...]
    block_size = 50000
    num_blocks = math.ceil(float(len(dates)) / block_size)
    for b in range(num_blocks):
        hist_block = datafile['hist_2d'][block_size * b: block_size * (b + 1)]
        for i in range(len(hist_block)):
            np.save(str(block_size * b + i), hist_block[i])

datafile.close()
