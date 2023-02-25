import pywt
import pandas as pd
import numpy as np
from random import random, sample
from scipy.fft import dct


def runs_of_ones(bits):
    """input must be 1D array of 0s and 1s"""
    difs = np.diff(np.hstack(([0], bits, [0])))
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    return run_starts, run_ends, run_ends - run_starts


def runs_by_criteria(df, bool_mask, good_run_len, merge_limit=0):
    """
    Finds runs in df where bool_mask is true, run length meets or exceeds a minimum, and optionally concatenate
    adjacent runs when they are separated by an acceptably small number distance.

    Parameters:
    df (Pandas DataFrame): source data
    bool_mask (Pandas Series of bools): True/False mask over the source data (df) indicating each row's adherence
        to some criteria.
    good_run_len (int): The minimum acceptable run length
    merge_limit (int): If adjacent runs are separated by merge_limit or fewer rows where bool_mask is False then
        concatenate the runs

    Returns:
    good_runs (list): List of data frames, each of which is a good run
    """
    if merge_limit < 0:
        raise(ValueError, 'merge_limit must be greater than or equal to zero')
    starts, ends, lengths = runs_of_ones(bool_mask.values.astype(int).flatten())
    good_runs = []
    if merge_limit == 0:
        for st, en, ln in zip(starts, ends, lengths):
            if ln >= good_run_len:
                good_runs.append(df.loc[df.index[st: en]])
    else:
        prev_run_end = -np.inf
        for st, en, ln in zip(starts, ends, lengths):
            if ln >= good_run_len:
                df_subset = df.loc[df.index[st: en]]
                if st - prev_run_end <= merge_limit:
                    good_runs[-1] = pd.concat([good_runs[-1], df_subset])
                else:
                    good_runs.append(df_subset)
                prev_run_end = en
    return good_runs


def runs_of_consecutive_vals(x):
    """Find runs of consecutive items in a numpy array. Used this in Lam Research Capacity Analysis tool"""
    n = x.shape[0]
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    else:
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]
        run_values = x[loc_run_start]
        run_lengths = np.diff(np.append(run_starts, n))
        return run_values, run_starts, run_lengths


def generate_synthetic(df, pr_span, vol_span, rng):
    """
    Generates synthetic data through small random perturbations of the input dataframe.

    Parameters:
    df (Pandas DataFrame): stock data OHLCV + market
    pr_span (float): random price variation will be in the range +- span/2, e.g. 0.001 --> 0.9995 <= var < 1.0005
    vol_span (float): span for variation on volume
    rng (np.random.default_rng): instance of numpy random number generator

    Returns:
    df_syn (Pandas DataFrame): new synthetic data represeting perturbation of input df
    """
    nrows = df.shape[0]
    df_syn = df.copy(deep=True)
    df_syn['Return_noDiv_Level_S&P500'] *= 1 + pr_span * (rng.random(nrows) - 0.5)
    df_syn['Open'] *= 1 + pr_span * (rng.random(nrows) - 0.5)
    df_syn['Close'] *= 1 + pr_span * (rng.random(nrows) - 0.5)
    df_syn['Ask_High'] = pd.concat([
        df_syn['Ask_High'] * (1 + pr_span * (rng.random(nrows) - 0.5)),
        df_syn['Open'] * 1.0000001,
        df_syn['Close'] * 1.0000001
    ], axis=1).max(axis=1)  # High cannot be lower than close or open
    df_syn['Bid_Low'] = pd.concat([
        df_syn['Bid_Low'] * (1 + pr_span * (rng.random(nrows) - 0.5)),
        df_syn['Open'] * 0.9999999,
        df_syn['Close'] * 0.9999999
    ], axis=1).min(axis=1)  # Low cannot be higher than close or open
    df_syn['Volume'] *= 1 + vol_span * (rng.random(nrows) - 0.5)
    # recalc the returns using new stock and market close values
    df_syn['Return_noDiv'] = df_syn['Close'] / df_syn['Close'].shift(1) - 1
    df_syn['Return_noDiv_S&P500'] = \
        df_syn['Return_noDiv_Level_S&P500'] / df_syn['Return_noDiv_Level_S&P500'].shift(1) - 1
    # fill in the NaN values at iloc[0]
    df_syn['Return_noDiv'].iloc[0] = \
        df['Return_noDiv'].iloc[0] * (1 + 0.1 * (random() - 0.5))
    df_syn['Return_noDiv_S&P500'].iloc[0] = \
        df['Return_noDiv_S&P500'].iloc[0] * (1 + 0.1 * (random() - 0.5))
    return df_syn


def derived_fields(df_stk, config):
    """
    Compute aggregate fields such as moving averages and also compute training signal(s). Data Standardization
    cannot be done here (e.g. standardizing closing prices) since each training sample will require a window calc,
    which is not the same thing as rolling calcs. Volatility normalization can be done here. The fields computed
    here are required for vectorization.

    Parameters:
    df_stk (Pandas DataFrame): OHLCV + market data with CRSP style column headers
    config (Dataclass): parameters
    """
    df_stk['OC_Change'] = df_stk['Close'] / df_stk['Open'] - 1  # same-day O->C change
    df_stk['CO_Change'] = df_stk['Open'] / df_stk['Close'].shift(1) - 1  # O vs prev C
    hi_lo = df_stk['Ask_High'] - df_stk['Bid_Low']
    # should not be possible for the close to be "outside" the bar range, but clip just in case
    df_stk['Close_In_Bar'] = np.clip((df_stk['Close'] - df_stk['Bid_Low']) / hi_lo, 0.0, 1.0)
    df_stk['Bar_Size'] = hi_lo / df_stk['Close'].shift(1)
    # vectorization scheme requires 50-day SMA of trading volume
    df_stk['volsma50'] = df_stk['Volume'].rolling(
        50, min_periods=config.volsma50_minper, closed='left').mean()
    ret_mag = df_stk['Return_noDiv'].abs().rolling(
        config.close_norm_window, min_periods=config.close_norm_minper, closed='left').mean()
    df_stk['Rolling_Ret_Mag'] = ret_mag
    if config.volatility_norm:
        # These series reassignments are faster than using DataFrame.multiply on all 4 cols at once
        df_stk['Return_noDiv'] = df_stk['Return_noDiv'] / ret_mag
        df_stk['OC_Change'] = df_stk['OC_Change'] / ret_mag
        df_stk['CO_Change'] = df_stk['CO_Change'] / ret_mag
        df_stk['Bar_Size'] = df_stk['Bar_Size'] / ret_mag
    return df_stk


# noinspection PyUnresolvedReferences
def simulate_trade(cl_rmult, op_rmult, config):
    """
    This function does not account for gaps or slippage which prevent closing at exactly the desired gain or loss R
    multiple. Nor does it account for large intraday price ranges which could hit both stopping criteria in the same
    day.

    Returns:
        results (array of ints): will be one of -1, 0, 1 indicating short, no-trade, long
    """
    res = {'long': 1, 'short': -1}
    for side, sign in res.items():
        # argmax will hit first true value, indicating the first day on which an exit criteria is met
        check_risk = (
                ((sign * op_rmult) <= -1) |
                ((sign * cl_rmult) <= -1)
        ).argmax(axis=1)
        check_gain = (
                ((sign * op_rmult) >= config.sim_close_r_mult) |
                ((sign * cl_rmult) >= config.sim_close_r_mult)
        ).argmax(axis=1)
        maybe_stopped = check_risk > 0
        maybe_full_gain = check_gain > 0
        is_stopped_out = maybe_stopped & ((check_risk < check_gain) | ~maybe_full_gain)
        is_full_gain = ~is_stopped_out & maybe_full_gain
        res[side] = is_full_gain.astype(np.int8)  # will be 1s or 0s
    return res['long'] - res['short']


# noinspection PyArgumentList
def derived_fields_and_targets(df_dict, config):
    """
    Compute aggregate fields such as moving averages and also compute training signal(s). Data Standardization
    cannot be done here (e.g. standardizing closing prices) since each training sample will require a window calc,
    which is not the same thing as rolling calcs. It is essential that the training signal dicts use same keys as
    df_dict. Volatility normalization can be done here. NOTE that the computation of the training signals will
    produce NaN values in the final few rows. These will need to be dropped, therefore this function cannot be
    used to predict tomorrow's prices.

    Parameters:
    df_dict (dict of Pandas DataFrames): good runs of stock data, including synthetic
    config (Dataclass): parameters

    Returns:
    trn_long (dict of numpy ndarrays): simulated buy-side trading outcome as R-multiple
    trn_shrt (dict of numpy ndarrays): simulated sell-side trading outcome as R-multiple
    *** All outputs MUST have the same number of rows (same shape along first axis) ***
    """
    nan_strt = -1 * config.simulation_len
    trade_sim = {}
    # note that pandas rolling.mean() will forward-fill nans
    for k in df_dict:
        df_dict[k] = derived_fields(df_dict[k], config)
        # NOTE that some columns in the dataframe have been modified during volatility normalization
        cl_subsets = stack_rolling_window(df_dict[k]['Close'].values, window_arange=np.arange(config.simulation_len))
        op_subsets = stack_rolling_window(df_dict[k]['Open'].values, window_arange=np.arange(config.simulation_len))
        opens = df_dict[k]['Open'].values[:-config.simulation_len + 1]
        rets = df_dict[k]['Rolling_Ret_Mag'].values[:-config.simulation_len + 1]
        risk_arr = config.sim_stoploss * rets * df_dict[k]['Close'].values[:-config.simulation_len + 1]
        cl_rmult = (cl_subsets - opens[..., None]) / risk_arr[..., None]
        op_rmult = (op_subsets - opens[..., None]) / risk_arr[..., None]
        outcomes = simulate_trade(cl_rmult, op_rmult, config)
        # today's signal is simualted trading starting at TOMORROW'S open
        trade_sim[k] = outcomes[1:]
        df_dict[k].drop(index=df_dict[k].index[nan_strt:], inplace=True)
        assert len(df_dict[k]) == len(trade_sim[k])
    return df_dict, trade_sim


def nsphere_embed(df, dtype):
    """
    Calculate rectangular coodinates unit-vectors in n-spherical Euclidean space for n-1 features. Features should have
    reviously been mapped onto angular ranges per https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates. The
    final angular coordinate is permitted to span [0, 2*pi) but I will not use the full range because values greater
    than pi would cause large dot products between stock vectors at the low end and high end of this feature's range
    and this is not appropriate to the intent of the mapping.

    Parameters:
    df: (Pandas Dataframe with RangeIndex columns): contains source features as columns. COLUMN ORDER IS IMPORTANT!
    dtype (numpy dtype): desired return data dtype

    Returns:
    vec_df: a dataframe where each column corresponds to one of the dimensions of the embedding space.
    """
    num_cols = len(df.columns)
    cosines = df.apply(np.cos)
    sines = df.apply(np.sin).cumprod(axis=1)
    vec_df = pd.DataFrame(data=1.0,
                          index=df.index,
                          columns=range(num_cols + 1))
    vec_df.loc[:, 1:] = sines.values
    vec_df.loc[:, :num_cols - 1] *= cosines.values  # pandas column slicing is inclusive at upper bound
    return vec_df.astype(dtype)


def safe_norm(df):
    """
    Compute the square root of the sum of the squares. Replace zeroes with a small finite number to avoid div by 0 error

    Parameters:
    df (Pandas Dataframe with RangeIndex columns): The set of values in each row represents the components of a vector

    Returns:
    n: a dataframe with a single column
    """
    n = (df ** 2).sum(axis=1).pow(0.5)
    n.loc[n == 0.0] = np.finfo(n.dtype).eps
    return n


def symmetric_soft_clip(data, width):
    """Modified sigmoid. Input data must be centered about zero in terms of the width argument. The width
    argument determines the data range which is mapped to ~95% of the sigmoid range"""
    rescaled = -3.66356 * data / (0.5 * width)
    return 2. / (1. + np.exp(rescaled)) - 1.0


# noinspection PyTypeChecker
def stock_vector_6(desc: str, df: pd.DataFrame, sma_min_periods: int, config):
    """
    Compute vector embeddings for stock data in a region of 6-dimensional Euclidean space

    Parameters:
    desc (str): a description / identifier for the dataframe
    df (Pandas DataFrame): dataframe must have columns `Open`, `Ask_High`, `Bid_Low`, `Close`, `Volume`, `volsma50`,
    `Return_noDiv`, `Return_noDiv_S&P500`, `Return_noDiv_Level_S&P500` where volsma50 is a 50-period simple moving
    average on the stock's trading volume.
    sma_min_periods (int): the number of periods at which the 50-period SMA is useable.
    config (dataclass): configuration including softclip ranges

    Returns:
    df (Pandas DataFrame): Input dataframe df with a new column for the vector embedding for each index value for which
    the embedding could be calculated. If it is not possible to produce the embedding then return None
    messages (str): newline-separated messages regarding the data processing. Useful for logging
    """
    messages = list()
    # clip the data so outliers don't cause the algo to focus on large regions of space with few samples
    cl_vs_prev_cl = symmetric_soft_clip(df['Return_noDiv'], config.vec_func_return_range)
    op_vs_prev_cl = symmetric_soft_clip(df['CO_Change'], config.vec_func_openchg_range)
    bar_size = symmetric_soft_clip(df['Bar_Size'], config.vec_func_barsize_range)
    # use of log10 for vol versus its sma50 implies that the 'range' refers to orders of magnitude centered at 0
    vol_vs_sma50 = symmetric_soft_clip(np.log10(df['Volume'] / df['volsma50']), config.vec_func_volvssma50_range)
    sp500ret = symmetric_soft_clip(df['Return_noDiv_S&P500'], config.vec_func_sp500ret_range)

    # # op_vs_prev_cl will have a NaN at iloc[0]. We could fill it in with some random fluctuation around 1.0,
    # # but the vector for this day will get dropped anyway because of the SMA
    # op_vs_prev_cl.iloc[0] = gauss(1, 0.01)

    # package the collection of series into a dataframe. ORDER IS IMPORTANT!
    df_vec = pd.concat([cl_vs_prev_cl, op_vs_prev_cl, bar_size, df['Close_In_Bar'], vol_vs_sma50, sp500ret], axis=1)
    df_vec.columns = range(len(df_vec.columns))
    # Drop the days prior to SMA being useable
    df_vec = df_vec.iloc[sma_min_periods:].astype(np.float32)
    if df_vec.isna().any(axis=None):
        messages.append('NaN values detected in embedding vectors for {}. Abandoning'.format(desc))
        return None, '\n'.join(messages)
    # add the vectors as a column of tuples. Reindex ensures we get tuples of NaN which makes unpacking easy later
    df = df.join(df_vec.reindex(index=df.index).apply(tuple, axis=1).to_frame('Vectors'), how='left')
    return df, '\n'.join(messages)


# noinspection PyTypeChecker
def stock_vector_6_volatility_norm(desc: str, df: pd.DataFrame, min_periods: int, config):
    """
    Compute vector embeddings for stock data in a region of 6-dimensional Euclidean space. Expects incoming returns and
    O->C changes to have been normalized by the rolling mean of the absolute value of daily returns.

    Parameters:
    desc (str): a description / identifier for the dataframe
    df (Pandas DataFrame): dataframe must have columns `Open`, `Ask_High`, `Bid_Low`, `Close`, `Volume`, `volsma50`,
    `Return_noDiv`, `Return_noDiv_S&P500`, `Return_noDiv_Level_S&P500`, `OC_Change`, `CO_Change`, `Bar_Size',
    `Close_In_Bar` where volsma50 is a 50-period simple moving average on the stock's trading volume.
    min_periods (int): the number of periods over which the rolling stats produced NaNs
    config (dataclass): configuration including softclip ranges


    Returns:
    df (Pandas DataFrame): Input dataframe df with a new column for the vector embedding for each index value for which
    the embedding could be calculated. If it is not possible to produce the embedding then return None
    messages (str): newline-separated messages regarding the data processing. Useful for logging
    """
    messages = list()
    # use of log10 for vol versus its sma50 implies that the 'range' refers to orders of magnitude centered at 0
    cl_vs_prev_cl = symmetric_soft_clip(df['Return_noDiv'], config.vec_func_return_range)
    op_vs_prev_cl = symmetric_soft_clip(df['CO_Change'], config.vec_func_openchg_range)
    barsize = symmetric_soft_clip(df['Bar_Size'], config.vec_func_barsize_range)
    # use of log10 for vol versus its sma50 implies that the 'range' refers to orders of magnitude centered at 0
    vol_vs_sma50 = symmetric_soft_clip(np.log10(df['Volume'] / df['volsma50']), config.vec_func_volvssma50_range)
    sp500ret = symmetric_soft_clip(df['Return_noDiv_S&P500'], config.vec_func_sp500ret_range)
    # package the collection of series into a dataframe. ORDER IS IMPORTANT!
    df_vec = pd.concat(
        [cl_vs_prev_cl, op_vs_prev_cl, barsize, df['Close_In_Bar'], vol_vs_sma50, sp500ret], axis=1)
    df_vec.columns = range(len(df_vec.columns))
    # Drop the days prior to SMA being useable
    df_vec = df_vec.iloc[min_periods:].astype(np.float32)
    if df_vec.isna().any(axis=None):
        messages.append('NaN values detected in embedding vectors for {}. Abandoning'.format(desc))
        return None, '\n'.join(messages)
    # add the vectors as a column of tuples. Reindex ensures we get tuples of NaN which makes unpacking easy later
    df = df.join(df_vec.reindex(index=df.index).apply(tuple, axis=1).to_frame('Vectors'), how='left')
    return df, '\n'.join(messages)


# noinspection PyTypeChecker
def stock_vector_5(desc: str, df: pd.DataFrame, sma_min_periods: int, config):
    """Same as the 6D version but with no SP500 data"""
    messages = list()
    # clip the data so outliers don't cause the algo to focus on large regions of space with few samples
    cl_vs_prev_cl = symmetric_soft_clip(df['Return_noDiv'], config.vec_func_return_range)
    op_vs_prev_cl = symmetric_soft_clip(df['CO_Change'], config.vec_func_openchg_range)
    bar_size = symmetric_soft_clip(df['Bar_Size'], config.vec_func_barsize_range)
    # use of log10 for vol versus its sma50 implies that the 'range' refers to orders of magnitude centered at 0
    vol_vs_sma50 = symmetric_soft_clip(np.log10(df['Volume'] / df['volsma50']), config.vec_func_volvssma50_range)

    # package the collection of series into a dataframe. ORDER IS IMPORTANT!
    df_vec = pd.concat([cl_vs_prev_cl, op_vs_prev_cl, bar_size, df['Close_In_Bar'], vol_vs_sma50], axis=1)
    df_vec.columns = range(len(df_vec.columns))
    # Drop the days prior to SMA being useable
    df_vec = df_vec.iloc[sma_min_periods:].astype(np.float32)
    if df_vec.isna().any(axis=None):
        messages.append('NaN values detected in embedding vectors for {}. Abandoning'.format(desc))
        return None, '\n'.join(messages)
    # add the vectors as a column of tuples. Reindex ensures we get tuples of NaN which makes unpacking easy later
    df = df.join(df_vec.reindex(index=df.index).apply(tuple, axis=1).to_frame('Vectors'), how='left')
    return df, '\n'.join(messages)


# noinspection PyTypeChecker
def stock_vector_5_volatility_norm(desc: str, df: pd.DataFrame, min_periods: int, config):
    """Same as the 6D version but with no SP500 data"""
    messages = list()
    # use of log10 for vol versus its sma50 implies that the 'range' refers to orders of magnitude centered at 0
    cl_vs_prev_cl = symmetric_soft_clip(df['Return_noDiv'], config.vec_func_return_range)
    op_vs_prev_cl = symmetric_soft_clip(df['CO_Change'], config.vec_func_openchg_range)
    barsize = symmetric_soft_clip(df['Bar_Size'], config.vec_func_barsize_range)
    # use of log10 for vol versus its sma50 implies that the 'range' refers to orders of magnitude centered at 0
    vol_vs_sma50 = symmetric_soft_clip(np.log10(df['Volume'] / df['volsma50']), config.vec_func_volvssma50_range)
    # package the collection of series into a dataframe. ORDER IS IMPORTANT!
    df_vec = pd.concat(
        [cl_vs_prev_cl, op_vs_prev_cl, barsize, df['Close_In_Bar'], vol_vs_sma50], axis=1)
    df_vec.columns = range(len(df_vec.columns))
    # Drop the days prior to SMA being useable
    df_vec = df_vec.iloc[min_periods:].astype(np.float32)
    if df_vec.isna().any(axis=None):
        messages.append('NaN values detected in embedding vectors for {}. Abandoning'.format(desc))
        return None, '\n'.join(messages)
    # add the vectors as a column of tuples. Reindex ensures we get tuples of NaN which makes unpacking easy later
    df = df.join(df_vec.reindex(index=df.index).apply(tuple, axis=1).to_frame('Vectors'), how='left')
    return df, '\n'.join(messages)


def bin_midpoints(bin_boundaries):
    """
    Compute the widths and midpoints of a numpy array of bin boundaries. This construction inludes the two bins spanning
    (-inf, bin_boundaries[0]) and (bin_boundaries[-1], inf) therefore there are len(bin_boundaries) + 1 elements in the
    output arrays. This function is intended to be used with soft_bin(); the treatment of the 0th and -1st bins herein
    is necessary to facilitate the distance calculation and subsequent quasi-softmax calc in soft_bin.
    """
    padded = np.hstack((
        bin_boundaries[0] - (bin_boundaries[1] - bin_boundaries[0]),
        bin_boundaries,
        bin_boundaries[-1] + (bin_boundaries[-1] - bin_boundaries[-2])
    ))
    widths = np.diff(padded)
    midpoints = widths / 2.0 + padded[:-1]
    return midpoints, widths


def soft_bin(bin_midpts, values, min_bin_wdth=0.0, sharpness=1.0):
    """
    Calculate soft-binning for input scalar(s) using a modified softmax formula. The distribution of each input
    value over the bins is calculated based on the distance to the midpoint of each bin, e.g. a value will
    achieve the max possible allocation to a given bin if the value is exactly centered on that bin.

    Note that some variation in bin width is okay, but if the bin spacing varies greatly then the more closely
    spaced bins will be difficult to resolve unless the sharpness is set high, in which case there will be very little
    spreading of values across adjacent large bins

    Parameters:
    bins_midpts: numpy array of bin midpoints, sorted ascending. Note the handling of the rightmost and leftmost bins
    values: the values to be "binned"
    sharpness: a multiplier used to manage the width of the distribution of each value over the bins

    Returns:
    out: array bin-loadings. Rows correspond to input values, cols correspond to bins
    """
    # ensure values is a column-array
    values = values.flatten()[..., None]
    # prevent numerical underflow where values are very far outside the bin range
    lwr_bound = bin_midpts[0] - 2 * (bin_midpts[1] - bin_midpts[0])
    upr_bound = bin_midpts[-1] + 2 * (bin_midpts[-1] - bin_midpts[-2])
    values[values < lwr_bound] = lwr_bound
    values[values > upr_bound] = upr_bound
    logits = np.exp(-sharpness * np.abs((values - bin_midpts) / min_bin_wdth))
    result = logits / logits.sum(axis=1)[..., None]
    return result


def stack_rolling_window(arr, window_arange=np.arange(1)):
    """
    Compute a vertical stack of arrays consisting of windowed subsets of the input array where the windows are stride 1
    apart from each other. My use case for this function is inside a long loop and the window width doesn't change, so
    I've precomputed window_arange such that it doesn't need to be repeatedly computed here

    Parameters:
    arr (1D numpy array): Source data to be segmented by a series of adjacent windows
    window_arange (1D numpy array): monotonically increasing array of integers starting at 0, number of elements is the
    desired window width

    Returns:
    (2D numpy array): stack of windowed subsets from arr
    """
    numel = arr.shape[0]
    samples = numel - window_arange.shape[0] + 1  # num useable windows we can create based on arr len and window width
    idxs = window_arange * np.ones((samples, 1), dtype=np.int32) + np.arange(samples)[..., None]
    return arr[idxs]


def sample_with_min_distance(n, num_elements=4, min_distance=10):
    """Sample num_elements from range(n), with a minimum distance. https://stackoverflow.com/questions/51918580/
    python-random-list-of-numbers-in-a-range-keeping-with-a-minimum-distance"""
    def ranks(smpl):
        indices = sorted(range(len(smpl)), key=lambda i: smpl[i])
        return sorted(indices, key=lambda i: indices[i])

    s = sample(range(n - (num_elements - 1) * (min_distance - 1)), num_elements)
    return [s + (min_distance - 1) * r for s, r in zip(s, ranks(s))]


def bucket(data_array, bins_array):
    # intended to be used for training signals and one-hot input signals like SP500
    return np.searchsorted(bins_array, data_array, side='left').astype(np.int8)


def proc_close_hist_wavelet(cl_array, config, price_hist_window):
    """must have same signature as other funcs which transform closing price histories"""
    hists = stack_rolling_window(cl_array, window_arange=price_hist_window)
    hists, _ = pywt.dwt(hists, config.wavelet, mode='constant', axis=1)  # dimensionality reduction
    hists = (hists - hists.mean(axis=1, keepdims=True)) / hists.std(axis=1, keepdims=True)  # standardize
    return hists


def proc_close_hist_dct(cl_array, config, price_hist_window):
    """must have same signature as other funcs which transform closing price histories"""
    # NOTE you prob don't want to use this. The variance on the coeffs is huge; can cause neural net to blow up
    hists = stack_rolling_window(cl_array, window_arange=price_hist_window)
    hists = hists / hists[..., 0, None]  # important for DCT to have reasonably small coeffs
    # we do not need to retain the first coefficient. It conveys the offset, but we're interested in the shape
    hists = dct(hists)[..., 1: config.num_dct_coeffs + 1]
    return hists


def proc_close_hist_sma(cl_array, config):
    shp = config.stk_hist_periods
    cl_series = pd.Series(data=cl_array, dtype=np.float32)
    # the `closed` kwarg must NOT be used on the biggest SMA. That way we only lose first stk_hist_periods - 1 values
    # thus maintaining alignment with the behavior of stack_rolling_window. Note rolling() returns float64
    cl_vs_sma50 = np.log10((cl_series / cl_series.rolling(50).mean().values)[shp - 1:])
    cl_vs_sma100 = np.log10((cl_series / cl_series.rolling(100).mean().values)[shp - 1:])
    cl_vs_sma200 = np.log10((cl_series / cl_series.rolling(200, min_periods=shp).mean().values)[shp - 1:])
    cl_vs_sma50 = symmetric_soft_clip(cl_vs_sma50.values, config.c50_softclip).astype(np.float32)
    cl_vs_sma100 = symmetric_soft_clip(cl_vs_sma100.values, config.c100_softclip).astype(np.float32)
    cl_vs_sma200 = symmetric_soft_clip(cl_vs_sma200.values, config.c200_softclip).astype(np.float32)
    return cl_vs_sma50, cl_vs_sma100, cl_vs_sma200


def proc_close_hist_2d(hi, lo, cl, config, base_img):
    """create a series of 2D arrays consisting of 0s and 1s, each of which is a simple stock chart."""
    if len(cl) < config.conv2d_pixel_width:
        delta = config.conv2d_pixel_width - len(cl)
        hi = np.pad(hi, (delta, 0))
        lo = np.pad(lo, (delta, 0))
        cl = np.pad(cl, (delta, 0))
    hi_stack = stack_rolling_window(hi, window_arange=config.conv2d_hist_window)
    lo_stack = stack_rolling_window(lo, window_arange=config.conv2d_hist_window)
    cl_stack = stack_rolling_window(cl, window_arange=config.conv2d_hist_window)
    hi_stack = (hi_stack / cl_stack[:, -1][..., None] - 1)[:, None, :]  # reshape to make the broadcasting work
    lo_stack = (lo_stack / cl_stack[:, -1][..., None] - 1)[:, None, :]
    img_stack = np.broadcast_to(
        base_img,
        (len(cl_stack), config.conv2d_pixel_height, config.conv2d_pixel_width))
    mask_stack = (img_stack >= lo_stack) & (img_stack <= hi_stack)
    return mask_stack.astype(np.int8)


def proc_sp500_ret(data_array, config):
    tmrw_sp500_ret = symmetric_soft_clip(data_array, config.vec_func_sp500ret_range)
    return tmrw_sp500_ret


def proc_stk_vectors(vec_mtrx, config, sv_hist_window):
    stacked_indices = stack_rolling_window(np.arange(len(vec_mtrx)), window_arange=sv_hist_window)
    vector_seq = vec_mtrx[stacked_indices, :]
    alignment_idx = config.stk_hist_periods - 1 - (len(vec_mtrx) - len(vector_seq))
    return vector_seq[alignment_idx:]


def to_sequences(data, config, sv_hist_window, base_img, means, stdvs):
    """produce sequences of samples to be consumed by ML model. Thus function expects
    component-wise means and standard deviations for vector standarization. Transformed
    closing price history standardization is handled differently in create_datasets"""
    c50, c100, c200 = config.close_hist_func(data['Close'][...], config)
    vec = proc_stk_vectors((data['Vectors'][...] - means) / stdvs, config, sv_hist_window)
    align_idx = config.stk_hist_periods - 1
    trade = data['Trade'][align_idx:] + 1  # maps -1, 0, 1 to 0, 1, 2
    # TODO: NOTE that proper alignment of indexes from proc_close_hist_2d relies on the fact that
    #  config.stk_hist_periods is equal to conv2d_pixel_width
    img_align_idx = config.stk_hist_periods - config.conv2d_pixel_width
    hist_2d = proc_close_hist_2d(
        data['Ask_High'][...],
        data['Bid_Low'][...],
        data['Close'][...],
        config,
        base_img)[img_align_idx:]
    ret = data['Return_noDiv'][align_idx:]
    date = data['Date'][align_idx:]
    data_dict = dict(date=date, c50=c50, c100=c100, c200=c200, vec=vec, ret=ret, trade=trade, hist_2d=hist_2d)
    return data_dict


def to_seq_inference(data, config, sv_hist_window, means, stdvs):
    """MUST be the same as to_sequences except for the absence of the ML training targets"""
    c50, c100, c200 = config.close_hist_func(data['Close'][...], config)
    vec = proc_stk_vectors((data['Vectors'][...] - means) / stdvs, config, sv_hist_window)
    date = data['Date'][config.stk_hist_periods - 1:]
    data_dict = dict(date=date, c50=c50, c100=c100, c200=c200, vec=vec)
    return data_dict
