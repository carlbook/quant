#!/home/carl/miniconda3/envs/trading_pytorch/bin/python
import h5py
import configs
import numpy as np
from collections import deque
from utils import to_sequences
from datetime import datetime as dt


# noinspection PyMethodMayBeStatic
class CRSPdsf62InMemory:
    """constructed on the assumption that the resultant datasets can all fit in RAM. Uses larger chunksizes"""

    def __init__(self, src):
        self.source = src
        self.trn_fout = '/mnt/data/trading/datasets/CRSPdsf62_trn_{}.hdf5'
        self.vec_fout = '/mnt/data/trading/datasets/CRSPdsf62_vec_{}.hdf5'
        # self.price_hist_window = np.arange(CFG.stk_hist_periods)
        self.sv_hist_window = np.arange(CFG.sv_seq_len)
        self.conv2d_base_img = \
            np.ones(CFG.conv2d_pixel_width) * \
            np.linspace(-CFG.conv2d_axis_range, CFG.conv2d_axis_range, CFG.conv2d_pixel_height)[..., None]
        self.trn_vec_means = None
        self.trn_vec_stdvs = None
        self.stk_blk_size = 400  # set this such that the RAM usage is reasonable
        self.suffix = int(dt.now().timestamp())
        self.__check_inputs()

    def __check_inputs(self):
        """DON'T FORGET there are NaN vals at the begining of each list of vectors"""
        if CFG.sv_seq_len >= CFG.stk_hist_periods - 20:
            raise ValueError('The stock vector sequence length is too long relative to the price history lookback')
        elif CFG.stk_hist_periods < 50:
            raise ValueError('The price history lookback is too short. Recall the sma50 makes NaN for first N vectors')

    def __prep_outputs(self, data_dict):
        # make training chunk size comparable to ML algo batch_size so that we're not reading huge excess amount
        # of data on each batch retrieval. Especially important if batches are retrieved in random order.
        trn_chunk = 2**15
        t_out = h5py.File(self.trn_fout.format(self.suffix), 'x')
        for k, v in CFG.to_h5py_attrs().items():
            t_out.attrs.create(k, v)
        # t_out.attrs.create('num_long_classes', len(CFG.md_bins) + 1)
        # t_out.attrs.create('num_short_classes', len(CFG.md_bins) + 1)
        t_out.attrs.create('source_data', self.source)
        t_out.attrs.create('trn_vec_means', self.trn_vec_means)
        t_out.attrs.create('trn_vec_stdvs', self.trn_vec_stdvs)
        for name, data in data_dict.items():
            # size can vary along axis 0 but is fixed to data dims along other axes
            dims, dtype = data.shape[1:], data.dtype
            t_sh = (trn_chunk, *dims)
            init_sh = (0, *dims)  # it is essential that the init shape have zero rows
            max_sh = (None, *dims)
            t_out.create_dataset(name, compression='lzf', dtype=dtype, shape=init_sh, maxshape=max_sh, chunks=t_sh)
        return t_out

    def build_datasets(self):
        """In prior versions of this class I had the option for multiple folds, and also a contiguity kwarg, and
        validation dataset. Those are no longer relevant because I will be performing 'roll-forward' cross validation
        dynamically during model training"""
        src = h5py.File(self.source, 'r')
        keys_dict = src.keys()
        # get data from first source table
        data_dict = to_sequences(
            src[next(iter(keys_dict))], CFG, self.sv_hist_window, self.conv2d_base_img,
            self.trn_vec_means, self.trn_vec_stdvs)
        t_out = self.__prep_outputs(data_dict)
        # dictionaries to hold chunks of data to be written to output dataset
        trn_data = {n: [] for n in data_dict.keys()}
        key_stack = deque(keys_dict)
        ctr = 0  # ctr tracks num stocks' worth data blocked together
        while key_stack:
            ctr += 1
            k = key_stack.popleft()
            data_dict = to_sequences(
                src[k], CFG, self.sv_hist_window, self.conv2d_base_img, self.trn_vec_means, self.trn_vec_stdvs)
            for name, data in data_dict.items():
                trn_data[name].append(data)
            # condition "not key_stack" accounts for possibility that final block of data occurs at ctr < stk_blk_size
            if ctr >= self.stk_blk_size or not key_stack:
                ctr = 0
                print('\n\n--------NEW BLOCK--------')
                for ds_name in trn_data.keys():
                    # populate training datasets
                    data_block = np.concatenate(trn_data[ds_name])
                    if ds_name != 'date':
                        assert not np.isnan(data_block).any()
                    trn_data[ds_name] = []  # reset for next iteration; clear up memory
                    dset_rowcount = t_out[ds_name].shape[0]
                    t_out[ds_name].resize(dset_rowcount + data_block.shape[0], axis=0)
                    t_out[ds_name][dset_rowcount:] = data_block
                    print(f'loading {ds_name} trn block with {data_block.shape[0]} rows at index {dset_rowcount}')
        src.close()
        t_out.close()
        self.__standardize_nonvec_inputs()

    def __standardize_nonvec_inputs(self):
        """open the newly created files and standardize the inputs (other than vectors) which have not yet been
        standardized. Use only training datasets to determine mean and stdev. Note hdf5 dataset dtypes cannot
        be changed. If you want to downcast to float16 you need to delete and recreate the dset"""
        trn_file = self.trn_fout.format(self.suffix)
        print('\n\n--------standardizing non-vector inputs using training data--------')
        datasets = ('c50', 'c100', 'c200')
        with h5py.File(trn_file, 'r+') as trn:
            for ds in datasets:
                trn_data = trn[ds][...]
                m, s = trn_data.mean(axis=0), trn_data.std(axis=0)
                trn_data = (trn_data - m) / s
                assert not np.isnan(trn_data).any()
                trn[ds][...] = trn_data
                trn.attrs.create(f'{ds}_means', m)
                trn.attrs.create(f'{ds}_stdvs', s)
        trn.close()

    def compute_vec_stats(self, return_vec_mtrx=False):
        print('\n\n--------computing vector means and stdevs--------')
        vec = []
        with h5py.File(self.source, 'r') as src:
            for grp in src.values():
                vec.append(grp['Vectors'][...])  # still has NaNs at the beginning
        src.close()
        vec = np.concatenate(vec)
        vec = vec[~np.isnan(vec).any(axis=1)].astype(np.float64)
        self.trn_vec_means = vec.mean(axis=0).astype(np.float32)
        self.trn_vec_stdvs = vec.std(axis=0).astype(np.float32)
        if return_vec_mtrx: return vec.astype(np.float32)
        else: del vec

    def extract_vec_as_dataset(self):
        """create a separate h5py file containing the stock vectors (individual, not sequences of vectors) to be used
        in clustering analysis and for the computation of mean and stdev for standardization. This data is NOT being
        standardized, however the mean and stdev will be calculated."""
        vec = self.compute_vec_stats(return_vec_mtrx=True)
        with h5py.File(self.vec_fout.format(self.suffix), 'x') as f_out:
            print(f'loading vectors into standalone vec dataset, shape {vec.shape}')
            for k, v in CFG.to_h5py_attrs().items():
                f_out.attrs.create(k, v)
            f_out.attrs.create('trn_vector_means', self.trn_vec_means)
            f_out.attrs.create('trn_vector_stdevs', self.trn_vec_stdvs)
            f_out.attrs.create('source_data', self.source)
            f_out.create_dataset('stock_vectors', data=vec, compression='lzf', shuffle=True)
        f_out.close()


if __name__ == '__main__':
    source = '/mnt/data/trading/datasets/CRSPdsf62_cln_aug_vec_C004.hdf5'
    with h5py.File(source, 'r') as source_data:
        CFG = getattr(configs, source_data.attrs['config_class_name'])()
        print(f'\nWorking with config class {CFG.__class__.__name__}')
        assert CFG.__class__.__name__ == source_data.attrs['config_class_name']
    source_data.close()

    builder = CRSPdsf62InMemory(source)
    builder.compute_vec_stats()
    builder.build_datasets()
