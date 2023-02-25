import h5py
import numpy as np
from random import sample
import pandas as pd
import torch
from torch.utils.data import Dataset


class CRSPdsf62InMemoryMaster:
    """Intended to be used with hdf5 dataset where each feature (e.g. vec) is a single ndarray and the entire dataset
     fits in RAM. Can only use one target (training signal). Validation split is done dynamically day-by-days"""

    def __init__(self, data_fpath, with_sp500=False, val_sample=100, subset_samples=None, num_sv=None, target='md'):
        self.data_fpath = data_fpath
        self.img_fpath = '/mnt/data/trading/datasets/C004_images/'  # TODO: hard-coding to test something
        self.with_sp500 = with_sp500
        self.val_sample = val_sample  # number of days worth of samples to use in the validaion set
        self.all_data, self.trn_data, self.val_data, self.attrs = {}, {}, {}, {}
        self.unique_dates = None  # will be a list of str YYYY-MM-DD therefore greater/less than comparisons are valid
        self.trn_date = None  # threshold btw trn and val for roll-forward cross validation. Inclusive to trn
        self.trn_idx = -1  # the index in self.all_data corresponding to the cutoff date
        self.subset_samples = subset_samples  # in case you want to retrieve a subset of the data for testing/dev
        self.num_sv = num_sv  # the number of stock vectors to retain
        self.target = target  # indicate which training data you want
        self.num_classes = 0
        self.num_trn_samples = 0
        self.num_val_samples = 0
        self.__setup()

    def __setup(self):
        with h5py.File(self.data_fpath, 'r') as datafile:
            self.attrs = dict(datafile.attrs)  # contains config used to create the dataset
            if self.target not in datafile.keys():
                datafile.close()
                raise ValueError('The selected target dataset does not exist')
            if not self.subset_samples: self.subset_samples = datafile['vec'].shape[0]
            if not self.num_sv: self.num_sv = datafile['vec'].shape[1]
            self.all_data['date'] = datafile['date'][:self.subset_samples].astype(np.str_)
            self.unique_dates = list(np.unique(self.all_data['date']))  # np.unique also sorts the values
            # PyTorch expects dims (batch, channels, sequence len) so we must use swapaxes
            self.all_data['vec'] = torch.from_numpy(
                datafile['vec'][:self.subset_samples, -self.num_sv:, :].swapaxes(1, 2))
            # self.all_data['hist_2d'] = torch.from_numpy(
            #     datafile['hist_2d'][:self.subset_samples][:, None, :, :].astype(np.float16))
            self.all_data['cSMA'] = torch.from_numpy(np.transpose(np.vstack((
                datafile['c50'][:self.subset_samples],
                # datafile['c100'][:self.subset_samples],
                datafile['c200'][:self.subset_samples]))))
            self.attrs['vector_dims'] = self.all_data['vec'].shape[1:]
            self.num_classes = 3  # self.attrs[f'num_{self.target}_classes']  # expects one of cc, oc, md
            self.all_data['target'] = torch.from_numpy(datafile[self.target][:self.subset_samples]).to(torch.long)
            if self.with_sp500:
                self.all_data['sp500_cc'] = torch.from_numpy(datafile['sp500_cc'][:self.subset_samples])
        datafile.close()
        # self.step_date()  # start at earliest available day
        self.__split_data()  # the step method is not needed for single-epoch training

    # def step_date(self):
    #     # roll-forward training and validation, one trading day at a time
    #     self.trn_idx += 1
    #     self.trn_date = self.unique_dates[self.trn_idx]
    #     self.__split_data()

    def __split_data(self):
        tensor_keys = [k for k in self.all_data.keys() if k != 'date']
        # First sort the main dataset
        print('Sorting training samples')
        sort_idxs = np.argsort(self.all_data['date'])
        self.all_data = {k: v[sort_idxs] for k, v in self.all_data.items()}
        print('Slicing training subset')
        mask = self.all_data['date'] < self.unique_dates[-self.val_sample]
        self.trn_data = {k: self.all_data[k][mask] for k in tensor_keys}
        self.trn_img_idxs = sort_idxs[mask]
        self.num_trn_samples = self.trn_data['vec'].shape[0]
        # Validation data.
        print('Slicing validation subset')
        future_mask = ~mask
        self.val_data = {k: self.all_data[k][future_mask] for k in tensor_keys}
        self.val_img_idxs = sort_idxs[future_mask]
        self.num_val_samples = self.val_data['vec'].shape[0]
        print('Deleting bulk data object')
        del self.all_data

    def get_trn_data(self, idx):
        vec = self.trn_data['vec'][idx]
        sma = self.trn_data['cSMA'][idx]
        img = torch.from_numpy(
            np.load(f'{self.img_fpath}{self.trn_img_idxs[idx]}.npy')[None, :, :].astype(np.float16))
        trgt = self.trn_data['target'][idx]
        return (vec, sma, img), trgt

    def get_val_data(self, idx):
        vec = self.val_data['vec'][idx]
        sma = self.val_data['cSMA'][idx]
        img = torch.from_numpy(
            np.load(f'{self.img_fpath}{self.val_img_idxs[idx]}.npy')[None, :, :].astype(np.float16))
        trgt = self.val_data['target'][idx]
        return (vec, sma, img), trgt

    def get_trn_data_sp500(self, idx):
        vec = self.trn_data['vec'][idx]
        sma = self.trn_data['cSMA'][idx]
        img = self.trn_data['hist_2d'][idx]
        sp5 = self.trn_data['sp500_cc'][idx]
        trgt = self.trn_data['target'][idx]
        return (vec, sma, img, sp5), trgt

    def get_val_data_sp500(self, idx):
        vec = self.val_data['vec'][idx]
        sma = self.val_data['cSMA'][idx]
        img = self.val_data['hist_2d'][idx]
        sp5 = self.val_data['sp500_cc'][idx]
        trgt = self.val_data['target'][idx]
        return (vec, sma, img, sp5), trgt

    def count_trn_samples(self):
        return self.num_trn_samples

    def count_val_samples(self):
        return self.num_val_samples


class SequenceCRSPdsf62InMemory(Dataset):
    """Intended to be used with hdf5 dataset where each feature (e.g. vec) is a single ndarray and the entire dataset
     fits in RAM. Can only use one target (training signal). Validation split is done dynamically day-by-days"""

    def __init__(self, datamaster: CRSPdsf62InMemoryMaster, is_validation=False):
        self.master = datamaster
        self.attrs = datamaster.attrs
        self.with_sp500 = datamaster.with_sp500
        self.target = datamaster.target
        self.num_classes = datamaster.num_classes
        self.is_validation = is_validation
        if is_validation:
            self.get_samples = self.master.count_val_samples
            if datamaster.with_sp500: self.loader_func = self.master.get_val_data_sp500
            else: self.loader_func = self.master.get_val_data
        else:
            self.get_samples = self.master.count_trn_samples
            if datamaster.with_sp500: self.loader_func = self.master.get_trn_data_sp500
            else: self.loader_func = self.master.get_trn_data

    def get_trn_idx(self):
        return self.master.trn_idx

    def get_trn_date(self):
        return self.master.trn_date

    def count_unique_dates(self):
        return len(self.master.unique_dates)

    # def step_date(self):
    #     self.master.step_date()

    def __len__(self):
        return self.get_samples()

    def __getitem__(self, idx):
        return self.loader_func(idx)


class InferenceInMemory(Dataset):
    """Similar to SequenceCRSPdsf62InMemory in terms of the outputs provided; works with `new` data from IB, Yahoo,
    and other sources, assembled in individual h5py groups to be used for inference or model evaluation outside of
    training. The input data has no train/validation split and no cross validation folds. We will be stacking
    ndarrays to be consumed by the ML model, but we do not want to lose index (trading date) info so downstream make
    sure to NOT use shuffle=True in the pytorch Dataloader constructor"""

    def __init__(self, data_fpath, num_syms=None):
        self.data_fpath = data_fpath
        self.df, self.data, self.attrs = None, {}, None  # the df data is for reference, not for the ML model
        self.num_syms = num_syms  # in case you want a subset of the data for testing/dev
        self.target = None  # will be one of oc, cc, md
        self.with_sp500 = False  # will be set to True if sp500 data is found in the dataset
        self.num_classes = 0  # needs to be calc'd based on selected target
        self.num_samples = 0
        self.loader_func = None
        self.__load_data()

    def __load_data(self):
        # note that an inference dataset will not have ML target data (oc, cc, md, sp500_cc)
        with h5py.File(self.data_fpath, 'r') as datafile:
            self.attrs = dict(datafile.attrs)
            sym_list = list(sample(datafile.keys(), self.num_syms) if self.num_syms else datafile.keys())
            ref_data = {k: [] for k in datafile[sym_list[0]]['raw_df'].keys()}
            mdl_data = {k: [] for k in datafile[sym_list[0]].keys() if k != 'raw_df'}
            ref_data_keys = tuple(ref_data.keys())
            mdl_data_keys = tuple(mdl_data.keys())
            ref_data['sym'] = []  # this line must be after ref_data_keys assignment
            # it is imperative to maintain index alignment among all the arrays involved here
            for sym in sym_list:
                numel = datafile[sym]['raw_df']['Date'].shape[0]
                ref_data['sym'].append(np.array(numel * [sym], dtype=object))
                for k in ref_data_keys:
                    ref_data[k].append(datafile[sym]['raw_df'][k][...])
                for k in mdl_data_keys:
                    mdl_data[k].append(datafile[sym][k][...])
        datafile.close()
        ref_data = {k: np.concatenate(arr_list) for k, arr_list in ref_data.items()}
        mdl_data = {k: np.concatenate(arr_list) for k, arr_list in mdl_data.items()}
        # build the dataframe to hold the reference data, and any cols we might add later after inference
        converted_dts = pd.to_datetime(ref_data['Date'].astype(np.str_), format='%Y-%m-%d')
        midx = pd.MultiIndex.from_arrays([ref_data['sym'], converted_dts], names=('sym', 'Date'))
        self.df = pd.DataFrame(index=midx)
        cols = [k for k in ref_data_keys if k not in ('sym', 'Date')]
        for c in cols: self.df[c] = ref_data[c]
        shapes = [arr.shape[0] for arr in mdl_data.values()]
        assert self.df.shape[0] == min(shapes) == max(shapes)
        self.num_samples = self.df.shape[0]
        # convert to tensors the arrays required by the pytorch model
        self.data['vec'] = torch.from_numpy(mdl_data['vec'].swapaxes(1, 2))
        self.data['cSMA'] = torch.from_numpy(np.transpose(np.vstack(
            (mdl_data['c50'], mdl_data['c100'], mdl_data['c200']))))
        self.attrs['vector_dims'] = self.data['vec'].shape[1:]
        targets = {'cc', 'md', 'oc'} & set(mdl_data_keys)  # will be empty when running inference
        for t in targets:
            self.data[t] = torch.from_numpy(mdl_data[t]).to(torch.long)
        if 'sp500_cc' in mdl_data_keys:
            self.with_sp500 = True
            self.data['sp500_cc'] = torch.from_numpy(mdl_data['sp500_cc'])
            self.loader_func = self.__getter_with_sp500
        else: self.loader_func = self.__getter

    def set_target(self, target, with_sp500=False):
        self.target = target
        self.with_sp500 = with_sp500
        self.num_classes = int(self.attrs[f'num_{target}_classes'])  # it's a str in the h5py file attrs
        if with_sp500: self.loader_func = self.__getter_with_sp500
        else: self.loader_func = self.__getter

    def inject_sp500_estimate(self):
        # TODO: dont forget to standardize using vals in self.attrs
        raise NotImplementedError

    # TODO: build and incorporate getters for inference wherein there are no targets

    def __getter(self, idx):
        vec = self.data['vec'][idx]
        sma = self.data['cSMA'][idx]
        trgt = self.data[self.target][idx]
        return (vec, sma), trgt

    def __getter_with_sp500(self, idx):
        vec = self.data['vec'][idx]
        sma = self.data['cSMA'][idx]
        sp5 = self.data['sp500_cc'][idx]
        trgt = self.data[self.target][idx]
        return (vec, sma, sp5), trgt

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.loader_func(idx)
