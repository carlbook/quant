import utils
import numpy as np
import pandas as pd
from typing import Callable
from dataclasses import dataclass


@dataclass
class BaseConfig:

    def to_h5py_attrs(self):
        # h5py attributes must be scalars or numpy arrays
        attr_dict = {'config_class_name': self.__class__.__name__}
        for k, v in self.__dict__.items():
            if isinstance(v, int) or isinstance(v, float) or isinstance(v, np.ndarray):
                attr_dict[k] = v
            elif isinstance(v, Callable):
                attr_dict[k] = v.__name__
            else:
                attr_dict[k] = str(v)
        return attr_dict

# CONFIG CLASSES 001 AND 002 HAVE BEEN REMOVED FROM THIS BRANCH DUE TO OTHER CODE CHANGES RENDERING THEM UNUSABLE
@dataclass
class Config003(BaseConfig):
    """If you change this dataclass you must delete and rebuild all datasets and models on which it depended"""
    gr_window: int = 10  # periods over wich good-run criteria will be calc'd for CRSP data preprocessing
    gr_min_dollar_vol: int = 10000000
    gr_min_runlength: int = 300
    gr_max_abs_change: float = 0.50
    min_rolling_ret: float = 0.001  # min rolling 10 day mean of SP500 daily return magnitudes is 0.00098
    min_rolling_range: float = 0.002  # bar height div by closing price. Filters out unusually low volatility
    volsma50_minper: int = 40  # minper integers give the number of days of contiguous data which will be NaN
    num_synth: int = 2  # number of synthetic datasets to create for each company
    volatility_norm: bool = False
    close_norm_minper: int = 60  # okay to make this same as the widow b/c the dropped data due to cl hist is larger
    close_norm_window: int = 60  # window size for the closing price volatility normalization
    vec_func: Callable[[str, pd.DataFrame, int, dataclass], tuple] = utils.stock_vector_5
    vec_func_return_range: float = 0.14  # 7% loss to 7% gain
    vec_func_openchg_range: float = 0.08  # 4% loss to 4% gain
    vec_func_barsize_range: float = 0.22
    vec_func_volvssma50_range: float = 2.0  # vol ratio as low as 10% of sma50, as high as 10x sma50
    vec_func_sp500ret_range: float = 0.1  # you can do json.loads(z.replace("'", '"')) where z is a str repr of a dict
    sv_seq_len: int = 48  # stock-vector sequence length. ML model may not use full length
    stk_hist_periods: int = 170  # min warmup for 200-day SMA on cl price hist
    close_hist_func: Callable[[np.ndarray, dataclass], tuple] = utils.proc_close_hist_sma
    c50_softclip: float = 0.35  # note these three SMA softclips are applied after log10()
    c100_softclip: float = 0.50
    c200_softclip: float = 0.70
    pr_span: float = 0.004  # affects magnitude of random price variation in synthetic data
    vol_span: float = 0.1  # affects magnitude of random volume variation in synthetic data
    val_split: float = 0.0  # the final val_split fraction of periods for each stock reserved for validation
    # bin boundaries for market returns, stock O->C and C->C price changes, and multiday signal.
    # The multiday signal will be in the range -0.5 to 0.5
    simulation_len: int = 20  # number of future trading days over which to simulate a trade
    sim_stoploss: float = 3  # scaling factor to be multiplied by avg daily return to determine stop loss amount
    sim_close_r_mult: float = 2  # close simulated positions at gain amount given by this value * risk amount
    md_bins: np.ndarray = np.arange(-0.5, sim_close_r_mult, 0.5)


@dataclass
class Config003VtyNorm(Config003):
    """If you change this dataclass you must delete and rebuild all datasets and models on which it depended"""
    volatility_norm: bool = True
    vec_func: Callable[[str, pd.DataFrame, int, dataclass], tuple] = utils.stock_vector_5_volatility_norm
    vec_func_return_range: float = 10.0
    vec_func_openchg_range: float = 5.0
    vec_func_barsize_range: float = 11.0
    vec_func_volvssma50_range: float = 2.0
    vec_func_sp500ret_range: float = 0.1


@dataclass
class Config004(Config003):
    """If you change this dataclass you must delete and rebuild all datasets and models on which it depended"""
    gr_min_dollar_vol: int = 50000000
    gr_min_runlength: int = 400
    gr_max_abs_change: float = 0.50
    num_synth: int = 0  # number of synthetic datasets to create for each company
    vec_func: Callable[[str, pd.DataFrame, int, dataclass], tuple] = utils.stock_vector_6
    sv_seq_len: int = 8  # stock-vector sequence length. ML model may not use full length
    stk_hist_periods: int = 170  # min warmup for 200-day SMA on cl price hist
    simulation_len: int = 10  # number of future trading days over which to simulate a trade
    sim_stoploss: float = 2  # scaling factor to be multiplied by avg daily return to determine stop loss amount
    sim_close_r_mult: float = 2  # close simulated positions at gain amount given by this value * risk amount
    md_bins: np.ndarray = None
    conv2d_pixel_height: int = 144
    conv2d_pixel_width: int = 144  # MUST NOT BE LARGER THAN stk_hist_periods
    conv2d_axis_range: float = 0.5  # e.g. 0.3 indicates +- 30%
    conv2d_hist_window: np.ndarray = np.arange(conv2d_pixel_width)


# TODO: recompute conv2d_axis_range for normalized data!
# @dataclass
# class Config004VtyNorm(Config003VtyNorm):
#     """If you change this dataclass you must delete and rebuild all datasets and models on which it depended"""
