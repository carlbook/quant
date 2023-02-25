#!/home/carl/miniconda3/envs/trading_pytorch/bin/python
import os
import h5py
import utils
import sqlite3
import configs
import numpy as np
import pandas as pd
from random import sample
from contextlib import suppress
from configparser import ConfigParser
from datetime import datetime as dt


# the purpose of this file is to provide tooling to identify CRSP data useable for ML training, perform split
# adjustments, build moving avgs, and organize the data for vectorization or other input to ML training e.g. 1D conv
CRSP_MARKET_TABLES = {'CRSP_NYSE_ValueWeighted': '1000200',
                      'CRSP_NYSE_EqualWeighted': '1000201',
                      'S&P500':                  '1000502',
                      'Nasdaq':                  '1000503'}
CRSP_MARKET_DB = 'StockData_CRSP62_DailyIndexTimeSeries.db'
CRSP_MARKET_DB_PATH = '/mnt/data/stocks/' + CRSP_MARKET_DB


class PrepCRSP:

    def __init__(self, path_to_stock_db, tables=None, mode='random', num_tables=5):
        """Run processing pipeline to transform CRSP stock data into ML training set. Note that additional processing
        might be required in the DataLoader for the ML model, for example to compute wavelet xform over Close price
        history, or to compute soft-binned values for next-day's Close vs Open as a training signal

        Parameters:
            path_to_stock_db (str): path to one of StockData_CRSPdsf62_DailyOHLCV.db, StockData_CRSP62_DailyOHLCV.db
            tables (int or str or list of int or str): stock table(s) to process
            mode (str): can be 'random' or 'all'. Ignored if tables is not None
            num_tables (int): specifies the number of tables to process when mode is 'random'
        """
        self.log_file = './logs/crsp_preprocessing_{}.log'.format(int(dt.now().timestamp()))
        with suppress(FileNotFoundError):
            os.remove(self.log_file)
        self.stock_db_path = path_to_stock_db
        self.stk_db_fname = self.stock_db_path.rsplit('/', maxsplit=1)[-1]
        self.df_mkt_idxs = pd.DataFrame()  # df to hold index histories with appropriately named columns
        self.tables_to_process = list()  # list of CRSP stock history tables to transform
        self.rng = np.random.default_rng()
        self.__load_config('crsp_proc_config.cfg')
        self.__parse_table_selections(tables, mode, num_tables)
        self.__load_market_indexes()
        self.__run_pipeline()

    def __load_config(self, fname):
        io_config = ConfigParser()
        io_config.read(fname)
        self.stk_query_template = io_config[self.stk_db_fname].get('query_all_pricehist')
        self.idx_query_template = io_config[CRSP_MARKET_DB].get('query_market_index')
        if CFG.close_norm_minper is not None:
            self.keep_vect_lwrbound = max(CFG.volsma50_minper, CFG.close_norm_minper)
        else:
            self.keep_vect_lwrbound = CFG.volsma50_minper
        out_dir = io_config['output'].get('dir')
        out_fname = io_config['output'].get('fname')
        # do not overwrite or modify an existing output dataset
        if os.path.isfile(os.path.join(out_dir, out_fname)):
            name_pieces = out_fname.split('.')
            out_fname = '{}_{}.{}'.format(name_pieces[0], dt.now().strftime('%d%b%Y_%H%M%S'), name_pieces[-1])
        self.output_fpath = os.path.join(out_dir, out_fname)
        self.__log(f'Output file for this run: {self.output_fpath}')

    def __parse_table_selections(self, tables, mode, num_tables):
        if tables:
            self.tables_to_process = [str(tables)] if not isinstance(tables, list) else [str(t) for t in tables]
        elif mode == 'random':
            if not num_tables > 0:
                raise ValueError('num_tables must be a positive integer when using random table selection')
            self.tables_to_process = sample(self.__read_table_names(), num_tables)
        elif mode == 'all':
            self.tables_to_process = self.__read_table_names()
        else:
            raise RuntimeError('provide a valid table, table list, or mode selection')

    def __read_table_names(self):
        """Get list of table names, each containing the price history of a security"""
        with sqlite3.connect(self.stock_db_path) as conn:
            df_tables = pd.read_sql_query("SELECT name FROM sqlite_schema WHERE type='table';", conn)
        conn.close()
        return df_tables['name'].tolist()

    def __load_market_indexes(self):
        # The indexes loaded here must be those which are needed to create ML training data
        idx_list = ['S&P500']
        list_of_dfs = list()
        with sqlite3.connect(CRSP_MARKET_DB_PATH) as conn:
            for idx in idx_list:
                df = pd.read_sql_query(
                    self.idx_query_template.format(CRSP_MARKET_TABLES[idx]), conn, index_col='Date')
                # noinspection PyTypeChecker
                df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
                df.rename(columns={'Return_noDiv': 'Return_noDiv_' + idx,
                                   'Return_noDiv_Level': 'Return_noDiv_Level_' + idx},
                          inplace=True)
                list_of_dfs.append(df)
        conn.close()
        # outer join the chosen indexes to create a single df to join to our stock history later
        self.df_mkt_idxs = pd.DataFrame().join(list_of_dfs, how='outer')

    # noinspection PyTypeChecker
    def __read_stock_table(self, tbl, conn):
        df_stk = pd.read_sql_query(self.stk_query_template.format(tbl), conn, index_col='Date')
        df_stk.index = pd.to_datetime(df_stk.index, format='%Y-%m-%d')
        return df_stk

    def __run_pipeline(self):
        with sqlite3.connect(self.stock_db_path) as conn, h5py.File(self.output_fpath, 'x') as f_out:
            # load config data into h5py file attributes
            for k, v in CFG.to_h5py_attrs().items():
                f_out.attrs.create(k, v)
            for table in self.tables_to_process:
                try:
                    df_stk = self.__read_stock_table(table, conn)
                    df_stk = self.__apply_price_adj(df_stk)
                    df_stk = self.__join_market(df_stk)
                    df_stk = self.__find_good_runs(df_stk, table)  # df_stk becomes a dictionary of dataframes
                    if not df_stk:
                        continue
                    df_stk = self.__generate_synthetic(df_stk)
                    df_stk, trade_sim = utils.derived_fields_and_targets(df_stk, CFG)
                    # df_stk = self.__create_multiday_bars(df_stk)
                    df_stk = self.__vector_embedding(df_stk)
                    for k in df_stk:
                        # unpack the tuples of vectors into a 2D numpy array
                        vec = pd.DataFrame.from_records(df_stk[k]['Vectors'].values).astype(np.float32).values
                        group = f_out.create_group('t' + k)
                        group.create_dataset('Date', data=df_stk[k].index.strftime('%Y-%m-%d').values)
                        group.create_dataset('Open', data=df_stk[k]['Open'].values.astype(np.float32))
                        group.create_dataset('Ask_High', data=df_stk[k]['Ask_High'].values.astype(np.float32))
                        group.create_dataset('Bid_Low', data=df_stk[k]['Bid_Low'].values.astype(np.float32))
                        group.create_dataset('Close', data=df_stk[k]['Close'].values.astype(np.float32))
                        group.create_dataset('Return_noDiv', data=df_stk[k]['Return_noDiv'].values.astype(np.float16))
                        group.create_dataset('Vectors', data=vec)
                        group.create_dataset('Trade', data=trade_sim[k])
                except Exception as EX:
                    er_mesg = '\nError while processing table {}: {}'.format(table, EX)
                    print(er_mesg)
                    self.__log(er_mesg)
        conn.close()

    # noinspection PyMethodMayBeStatic
    def __apply_price_adj(self, stk):
        stk['Open'] /= stk['Cum_Price_Adjust_Factor']
        stk['Ask_High'] /= stk['Cum_Price_Adjust_Factor']
        stk['Bid_Low'] /= stk['Cum_Price_Adjust_Factor']
        stk['Close'] /= stk['Cum_Price_Adjust_Factor']
        stk['Volume'] *= stk['Cum_Price_Adjust_Factor']
        stk.drop(columns='Cum_Price_Adjust_Factor', inplace=True)
        return stk

    def __join_market(self, stk):
        """join on market index. NaN will be removed later when we find good runs of data"""
        stk = stk.join(
            self.df_mkt_idxs.loc[stk.index[0]: stk.index[-1] + pd.Timedelta(1, unit='days')],
            how='right')
        return stk

    # noinspection PyMethodMayBeStatic
    def __find_good_runs(self, stk, tbl_name):
        """break the input df into subsets meeting certain criteria. The return object is a dictionary wherein the
        keys are strings structured as CrspTableName_StartDate_RunLength, e.g. 10372_20061123_445. Most of the filtering
        here should focus on eliminating non-physical data. Removal of large movements isn't appropriate here if
        merge_limit is greater than 0. Larege movements will be clipped during vectorization"""
        gr_dict = dict()
        if stk.shape[0] < CFG.gr_min_runlength:
            return gr_dict
        dollar_volume = (stk['Volume'] * stk['Close']).rolling(CFG.gr_window, min_periods=1, closed='right').mean()
        ret_magnitude = stk['Return_noDiv'].abs().rolling(CFG.gr_window, min_periods=1, closed='right').mean()
        rg = (stk['Ask_High'] - stk['Bid_Low']) / stk['Close'].shift(1)
        rg.iloc[0] = rg.iloc[1]  # remove the NaN from shift(1)
        rg = rg.rolling(CFG.gr_window, min_periods=1, closed='both').mean()
        mask = ((stk[['Open', 'Ask_High', 'Bid_Low', 'Close', 'Volume']] > 0).all(axis=1)) & \
               (stk['Return_noDiv_S&P500'].notna()) & \
               (dollar_volume >= CFG.gr_min_dollar_vol) & \
               (stk['Return_noDiv'].abs() < CFG.gr_max_abs_change) & \
               ((stk['Close'] / stk['Open'] - 1).abs() < CFG.gr_max_abs_change) & \
               ((stk['Open'] / stk['Close'].shift(1) - 1).abs() < CFG.gr_max_abs_change) & \
               ((stk['Close'] / stk['Close'].shift(1) - 1).abs() < CFG.gr_max_abs_change) & \
               (stk[['Open', 'Close', 'Bid_Low']].le(stk['Ask_High'], axis='index').all(axis=1)) & \
               (stk[['Open', 'Close']].ge(stk['Bid_Low'], axis='index').all(axis=1)) & \
               (ret_magnitude >= CFG.min_rolling_ret) & \
               (rg >= CFG.min_rolling_range) & \
               (stk['Ask_High'].gt(stk['Bid_Low'], axis='index'))
        good_runs = utils.runs_by_criteria(stk, mask, CFG.gr_min_runlength, merge_limit=1)
        for gr in good_runs:
            # This assertion should never hit since the SQL responses are sorted. But if it does, I want to know
            assert gr.index.is_monotonic_increasing
            # don't worry about dropping the NaN rows here. This will happen later
            k = '{}_{}_{}'.format(tbl_name, gr.index[0].strftime('%Y%m%d'), gr.shape[0])
            gr_dict[k] = gr
        return gr_dict

    def __generate_synthetic(self, df_dict):
        df_dict_synth = {}
        for key, df_stk in df_dict.items():
            for i in range(CFG.num_synth):
                df_syn = utils.generate_synthetic(df_stk, CFG.pr_span, CFG.vol_span, self.rng)
                new_key = '{}_syn{}'.format(key, i + 1)
                df_dict_synth[new_key] = df_syn
                # plot_ohlcvm_comparison(df_stk, temp, new_key, source='crsp')  # testing
        return {**df_dict, **df_dict_synth}

    def __create_multiday_bars(self, df_dict, n=5):
        """form an n-period OHLCV + V/Vsma50 bar for each row. Note this means the bars do not strictly represent
        weeks Mon-Fri. Avoid large NaN impact by not calculating volume moving average across the full history"""
        raise NotImplementedError
        # # I would need to make a new vector embedding scheme and vocabulary for this. Not sure if it's worth it
        # lbl_o = f'{n}_Open'
        # lbl_h = f'{n}_Ask_High'
        # lbl_l = f'{n}_Bid_Low'
        # lbl_c = f'{n}_Close'
        # lbl_v = f'{n}_Volume'
        # lbl_v50 = f'{n}_volsma50'
        # lbl_r = f'{n}_Return_noDiv'
        # for k in df_dict:
        #     df_dict[k][lbl_o] = df_dict[k]['Open'].rolling(n).agg(lambda r: r.iloc[0])
        #     df_dict[k][lbl_h] = df_dict[k]['Ask_High'].rolling(n).max()
        #     df_dict[k][lbl_l] = df_dict[k]['Bid_Low'].rolling(n).min()
        #     df_dict[k][lbl_c] = df_dict[k]['Close'].rolling(n).agg(lambda r: r.iloc[-1])
        #     df_dict[k][lbl_v] = df_dict[k]['Volume'].rolling(n).sum()
        #     df_dict[k][lbl_v50] = df_dict[k][lbl_v] / df_dict[k]['volsma50'].rolling(n).sum()
        #     df_dict[k][lbl_r] = df_dict[k]['Close'] / df_dict[k]['Close'].rolling(n + 1).agg(lambda r: r.iloc[0]) - 1
        # return df_dict

    def __vector_embedding(self, df_dict):
        """The returned df_dict will probably have NaNs for the first ~self.keep_vect_lwrbound rows of Vectors"""
        discard_tables = list()
        for k, df in df_dict.items():
            df, message = CFG.vec_func(k, df, self.keep_vect_lwrbound, CFG)
            if df is None:
                discard_tables.append(k)
                # if a source-data table gets deleted then also delete the associated synthetic data
                if 'syn' not in k: discard_tables += [f'{k}_syn{s}' for s in range(1, CFG.num_synth + 1)]
            else:
                df_dict[k] = df
            if len(message) > 0:
                self.__log(message)
        for k in discard_tables:
            del df_dict[k]
        return df_dict

    def __log(self, txt):
        with open(self.log_file, 'a') as f:
            f.writelines(txt + '\n')


if __name__ == '__main__':
    stocks = {
        '89393': 'Netflix',
        '18163': 'Proctor and Gamble',
        '25785': 'Ford',
        '14593': 'Apple'}

    CFG = configs.Config004()

    PrepCRSP('/mnt/data/stocks/StockData_CRSPdsf62_DailyOHLCV.db', mode='all')
    # PrepCRSP('/mnt/data/stocks/StockData_CRSPdsf62_DailyOHLCV.db', num_tables=100)
