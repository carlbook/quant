#!/home/carl/miniconda3/envs/trading/bin/python
import pandas as pd
import os
import sqlite3

db_dir = '/mnt/data/stocks/'
base_dir = '/mnt/data/stocks/kaggle_data/crsp/quarterly/'


def overview():
    # This will grab some data from the CRSP database files so i can figure out what data i need and how to join it
    sub_dirs = ['q_stock62/', 'q_indexes/', 'q_stock/']
    os.chdir(base_dir)
    skip = ['asio.sas7bdat', 'dsbo.sas7bdat', 'dsia.sas7bdat', 'dsib.sas7bdat',
            'dsic.sas7bdat', 'dsio.sas7bdat', 'dsir.sas7bdat', 'dsix.sas7bdat',
            'dsiy.sas7bdat', 'dssc.sas7bdat', 'dsso.sas7bdat', 'msp500.sas7bdat']
    for s in sub_dirs:
        files = os.listdir(base_dir + s)
        for f in files:
            if f[-3:] == '.lz' or f in skip:
                continue  # if i didn't decompress a file then im not interested in it here
            else:
                df = pd.read_sas(base_dir + s + f, chunksize=100)
                dfr = df.read()
                str_cols = dfr.select_dtypes(object).columns
                dfr.loc[:, str_cols] = dfr.loc[:, str_cols].apply(lambda z: z.str.decode('utf-8'))
                with open(base_dir + 'data_overview_decoded.txt', 'a') as out:
                    out.write(f.split(sep='.')[0] + '\n')
                    out.write(dfr.to_string() + '\n\n')


# Don't forget the details about negative numbers etc. See Trilium notes.
#
# Also NOTE: the code in this file, when creating the sqlite .db tables, does not set any field as "primary key".
# It does not enforce non-null, it does not enforce uniqueness.
# However, i have gone into some of the tables in DB Browser for SQLite and done these things manually


def s6z_dp_dly():
    f_name = 'q_stock62/s6z_dp_dly.sas7bdat'
    # the field KYPERNO is the permanent identifier for the stock
    db_name = 'StockData_CRSP62_DailyCloses.db'
    if os.path.isfile(db_dir + db_name):
        print('WARNING: database file already exists. Exiting')
        return
    # provide all the columns you intend to keep, along with their sqlite type
    sql_types = {'Date':           'DATE',
                 'Close':          'REAL',
                 'Return':         'REAL',
                 'Return_noDiv':   'REAL',
                 'Capitalization': 'REAL',
                 'Volume':         'REAL'
                 }
    group_key = 'KYPERMNO'  # IF you intend to group the data, this is where you specify which field to use
    with sqlite3.connect(db_dir + db_name) as conn:
        df = pd.read_sas(base_dir + f_name)
        key_list = df[group_key].unique().tolist()
        df.rename(columns={'CALDT': 'Date',
                           'PRC':   'Close',
                           'RET':   'Return',
                           'RETX':  'Return_noDiv',
                           'TCAP':  'Capitalization',
                           'VOL':   'Volume'
                           },
                  inplace=True
                  )
        df.loc[:, 'Date'] = df['Date'].dt.date
        group = df.groupby(by=group_key, sort=False)
        for k in key_list:
            group.get_group(k)[list(sql_types.keys())].to_sql(int(k),
                                                              conn,
                                                              if_exists='fail',
                                                              index=False,
                                                              dtype=sql_types,
                                                              method='multi')


def s6z_ds_dly():
    f_name = 'q_stock62/s6z_ds_dly.sas7bdat'
    db_name = 'StockData_CRSP62_DailyOHL.db'
    if os.path.isfile(db_dir + db_name):
        print('WARNING: database file already exists. Exiting')
        return
    # provide all the columns you intend to keep, along with their sqlite type
    sql_types = {'Date':             'DATE',
                 'Bid_Low':          'REAL',
                 'Ask_High':         'REAL',
                 'Bid':              'REAL',
                 'Ask':              'REAL',
                 'Nasdaq_NumTrades': 'INTEGER',
                 'Open':             'REAL'
                 }
    group_key = 'KYPERMNO'  # IF you intend to group the data, this is where you specify which field to use
    with sqlite3.connect(db_dir + db_name) as conn:
        df = pd.read_sas(base_dir + f_name)
        key_list = df[group_key].unique().tolist()
        df.rename(columns={'CALDT':   'Date',
                           'BIDLO':   'Bid_Low',
                           'ASKHI':   'Ask_High',
                           'BID':     'Bid',
                           'ASK':     'Ask',
                           'NUMTRD':  'Nasdaq_NumTrades',
                           'OPENPRC': 'Open'
                           },
                  inplace=True
                  )
        df.loc[:, 'Date'] = df['Date'].dt.date
        group = df.groupby(by=group_key, sort=False)
        for k in key_list:
            group.get_group(k)[list(sql_types.keys())].to_sql(int(k),
                                                              conn,
                                                              if_exists='fail',
                                                              index=False,
                                                              dtype=sql_types,
                                                              method='multi')


def s6z_del():
    f_name = 'q_stock62/s6z_del.sas7bdat'
    db_name = 'StockData_CRSP62_misc.db'
    # provide all the columns you intend to keep, along with their sqlite type
    sql_types = {'KYPERMNO':         'INTEGER',
                 'Delist_Date':      'DATE',
                 'Delist_Code':      'INTEGER',
                 'New_PERMNO':       'INTEGER',
                 'New_PERMCO':       'INTEGER',
                 'Next_Info_DT':     'DATE',
                 'Del_Price':        'REAL',
                 'Del_Pymnt_Date':   'DATE',
                 'Del_Amount':       'REAL',
                 'Del_Return':       'REAL',
                 'Del_Return_NoDiv': 'REAL'
                 }
    with sqlite3.connect(db_dir + db_name) as conn:
        df = pd.read_sas(base_dir + f_name)
        df.rename(columns={'DLSTDT': 'Delist_Date',
                           'DLSTCD': 'Delist_Code',
                           'NWPERM': 'New_PERMNO',
                           'NWCOMP': 'New_PERMCO',
                           'NEXTDT': 'Next_Info_DT',
                           'DLPRC':  'Del_Price',
                           'DLPDT':  'Del_Pymnt_Date',
                           'DLAMT':  'Del_Amount',
                           'DLRET':  'Del_Return',
                           'DLRETX': 'Del_Return_NoDiv'
                           },
                  inplace=True
                  )
        df.loc[:, 'Delist_Date'] = df['Delist_Date'].dt.date
        df.loc[:, 'Next_Info_DT'] = df['Next_Info_DT'].dt.date
        df.loc[:, 'Del_Pymnt_Date'] = df['Del_Pymnt_Date'].dt.date
        df.to_sql('Daily_Delist_Returns', conn, if_exists='fail', index=False, dtype=sql_types)


def s6z_dind():
    f_name = 'q_stock62/s6z_dind.sas7bdat'
    db_name = 'StockData_CRSP62_DailyIndexTimeSeries.db'
    if os.path.isfile(db_dir + db_name):
        print('WARNING: database file already exists. Exiting')
        return
    # provide all the columns you intend to keep, along with their sqlite type
    sql_types = {'Date':                  'DATE',
                 'Tot_Return':            'REAL',
                 'Tot_Return_Level':      'REAL',
                 'Return_noDiv':          'REAL',
                 'Return_noDiv_Level':    'REAL',
                 'Income_Return':         'REAL',
                 'Income_Return_Level':   'REAL',
                 'Count_Securities_Used': 'INTEGER',
                 'Val_Securities_Used':   'REAL',
                 'Count_Securities_Tot':  'INTEGER',
                 'Val_Securities_Tot':    'REAL'
                 }
    group_key = 'KYINDNO'  # IF you intend to group the data, this is where you specify which field to use
    with sqlite3.connect(db_dir + db_name) as conn:
        df = pd.read_sas(base_dir + f_name)
        key_list = df[group_key].unique().tolist()
        df.rename(columns={'CALDT':  'Date',
                           'TRET':   'Tot_Return',
                           'TIND':   'Tot_Return_Level',
                           'ARET':   'Return_noDiv',
                           'AIND':   'Return_noDiv_Level',
                           'IRET':   'Income_Return',
                           'IIND':   'Income_Return_Level',
                           'USDCNT': 'Count_Securities_Used',
                           'USDVAL': 'Val_Securities_Used',
                           'TOTCNT': 'Count_Securities_Tot',
                           'TOTVAL': 'Val_Securities_Tot'
                           },
                  inplace=True
                  )
        df.loc[:, 'Date'] = df['Date'].dt.date
        group = df.groupby(by=group_key, sort=False)
        for k in key_list:
            group.get_group(k)[list(sql_types.keys())].to_sql(int(k),
                                                              conn,
                                                              if_exists='fail',
                                                              index=False,
                                                              dtype=sql_types,
                                                              method='multi')


def s6z_hdr():
    f_name = 'q_stock62/s6z_hdr.sas7bdat'
    db_name = 'StockData_CRSP62_misc.db'
    # provide all the columns you intend to keep, along with their sqlite type
    sql_types = {'KYPERMNO':         'INTEGER',
                 'CUSIP':            'TEXT',
                 'CUSIP9':           'TEXT',
                 'PERMCO':           'INTEGER',
                 'COMPNO':           'INTEGER',
                 'ISSUNO':           'INTEGER',
                 'Ticker_Symbol':    'TEXT',
                 'Exchange_Code':    'INTEGER',
                 'SIC_Code':         'INTEGER',
                 'Data_Begin_Date':  'DATE',
                 'Data_End_Date':    'DATE',
                 'Delist_Code':      'INTEGER',
                 'Company_Name':     'TEXT',
                 'Trading_Symbol':   'TEXT',
                 'NAICS_Code':       'TEXT',
                 'Share_Code':       'TEXT',
                 'Primary_Exchange': 'TEXT',
                 'Trading_Status':   'TEXT',
                 'Security_Status':  'TEXT'
                 }
    with sqlite3.connect(db_dir + db_name) as conn:
        df = pd.read_sas(base_dir + f_name)
        df.rename(columns={'HTICK':     'Ticker_Symbol',
                           'HEXCD':     'Exchange_Code',
                           'HSICCD':    'SIC_Code',
                           'BEGDT':     'Data_Begin_Date',
                           'ENDDT':     'Data_End_Date',
                           'HDLSTCD':   'Delist_Code',
                           'HCOMNAM':   'Company_Name',
                           'HTSYMBOL':  'Trading_Symbol',
                           'HSNAICS':   'NAICS_Code',
                           'HSHRCD':    'Share_Code',
                           'HPRIMEXCH': 'Primary_Exchange',
                           'HTRDSTAT':  'Trading_Status',
                           'HSECSTAT':  'Security_Status'
                           },
                  inplace=True
                  )
        df.loc[:, 'Data_Begin_Date'] = df['Data_Begin_Date'].dt.date
        df.loc[:, 'Data_End_Date'] = df['Data_End_Date'].dt.date
        df.to_sql('Security_Header', conn, if_exists='fail', index=False, dtype=sql_types)


def s6z_indhdr():
    f_name = 'q_stock62/s6z_indhdr.sas7bdat'
    db_name = 'StockData_CRSP62_misc.db'
    # provide all the columns you intend to keep, along with their sqlite type
    sql_types = {'KYINDNO':                'INTEGER',
                 'Index_Name':             'TEXT',
                 'Begin_Date':             'DATE',
                 'End_Date':               'DATE',
                 'Index_Family':           'INTEGER',
                 'Portfolio_Num':          'INTEGER',
                 'Idx_Base_Level':         'REAL',
                 'Idx_Base_Date':          'DATE',
                 'Idx_Item_Availability':  'TEXT',
                 'Idx_Rules':              'INTEGER',
                 'Idx_Listing_Exceptions': 'INTEGER',
                 'Idx_Methodology':        'INTEGER',
                 'Idx_Rebalance_Rule':     'INTEGER',
                 'Universe_Subset':        'INTEGER',
                 'Universe':               'INTEGER'
                 }
    with sqlite3.connect(db_dir + db_name) as conn:
        df = pd.read_sas(base_dir + f_name)
        df.rename(columns={'INDNAME':      'Index_Name',
                           'INDBEGDT':     'Begin_Date',
                           'INDENDDT':     'End_Date',
                           'INDFAM':       'Index_Family',
                           'PORTNUM':      'Portfolio_Num',
                           'BASELVL':      'Idx_Base_Level',
                           'BASEDT':       'Idx_Base_Date',
                           'AVAILABILITY': 'Idx_Item_Availability',
                           'CALCRULE':     'Idx_Rules',
                           'LISTRULE':     'Idx_Listing_Exceptions',
                           'METHOD':       'Idx_Methodology',
                           'REBALRULE':    'Idx_Rebalance_Rule',
                           'PUNIVERSE':    'Universe_Subset',
                           'UNIVERSE':     'Universe'
                           },
                  inplace=True
                  )
        df.loc[:, 'Begin_Date'] = df['Begin_Date'].dt.date
        df.loc[:, 'End_Date'] = df['End_Date'].dt.date
        df.loc[:, 'Idx_Base_Date'] = df['Idx_Base_Date'].dt.date
        df.to_sql('Index_Header', conn, if_exists='fail', index=False, dtype=sql_types)


def s6z_nam():
    f_name = 'q_stock62/s6z_nam.sas7bdat'
    db_name = 'StockData_CRSP62_misc.db'
    # provide all the columns you intend to keep, along with their sqlite type
    sql_types = {'KYPERMNO':         'INTEGER',
                 'NCUSIP':           'TEXT',
                 'NCUSIP9':          'TEXT',
                 'Name_Start_Date':  'DATE',
                 'Name_End_Date':    'DATE',
                 'Ticker_Symbol':    'TEXT',
                 'Company_Name':     'TEXT',
                 'Share_Class':      'TEXT',
                 'Share_Code':       'TEXT',
                 'Exchange_Code':    'INTEGER',
                 'SIC_Code':         'INTEGER',
                 'Trading_Symbol':   'TEXT',
                 'NAICS_Code':       'TEXT',
                 'Primary_Exchange': 'TEXT',
                 'Trading_Status':   'TEXT',
                 'Security_Status':  'TEXT'
                 }
    with sqlite3.connect(db_dir + db_name) as conn:
        df = pd.read_sas(base_dir + f_name)
        df.rename(columns={'NAMEDT':    'Name_Start_Date',
                           'NAMEENDDT': 'Name_End_Date',
                           'TICKER':    'Ticker_Symbol',
                           'COMNAM':    'Company_Name',
                           'SHRCLS':    'Share_Class',
                           'SHRCD':     'Share_Code',
                           'EXCHCD':    'Exchange_Code',
                           'SICCD':     'SIC_Code',
                           'TSYMBOL':   'Trading_Symbol',
                           'SNAICS':    'NAICS_Code',
                           'PRIMEXCH':  'Primary_Exchange',
                           'TRDSTAT':   'Trading_Status',
                           'SECSTAT':   'Security_Status'
                           },
                  inplace=True
                  )
        df.loc[:, 'Name_Start_Date'] = df['Name_Start_Date'].dt.date
        df.loc[:, 'Name_End_Date'] = df['Name_End_Date'].dt.date
        df.to_sql('Name_History', conn, if_exists='fail', index=False, dtype=sql_types)


def s6z_dis():
    f_name = 'q_stock62/s6z_dis.sas7bdat'
    db_name = 'StockData_CRSP62_misc.db'
    # provide all the columns you intend to keep, along with their sqlite type
    sql_types = {'KYPERMNO':             'INTEGER',
                 'Distribution_Code':    'INTEGER',
                 'Dividend_Amount':      'REAL',
                 'Price_Adjust_Factor':  'REAL',
                 'Shares_Adjust_Factor': 'REAL',
                 'Declaration_Date':     'DATE',
                 'Ex_Distrib_Date':      'DATE',
                 'Record_Date':          'DATE',
                 'Payment_Date':         'DATE',
                 'Acquiring_PERMNO':     'INTEGER',
                 'Acquiring_PERMCO':     'INTEGER'
                 }
    with sqlite3.connect(db_dir + db_name) as conn:
        df = pd.read_sas(base_dir + f_name)
        df.rename(columns={'DISTCD': 'Distribution_Code',
                           'DIVAMT': 'Dividend_Amount',
                           'FACPR':  'Price_Adjust_Factor',
                           'FACSHR': 'Shares_Adjust_Factor',
                           'DCLRDT': 'Declaration_Date',
                           'EXDT':   'Ex_Distrib_Date',
                           'RCRDDT': 'Record_Date',
                           'PAYDT':  'Payment_Date',
                           'ACPERM': 'Acquiring_PERMNO',
                           'ACCOMP': 'Acquiring_PERMCO'
                           },
                  inplace=True
                  )
        df.loc[:, 'Declaration_Date'] = df['Declaration_Date'].dt.date
        df.loc[:, 'Ex_Distrib_Date'] = df['Ex_Distrib_Date'].dt.date
        df.loc[:, 'Record_Date'] = df['Record_Date'].dt.date
        df.loc[:, 'Payment_Date'] = df['Payment_Date'].dt.date
        df.to_sql('Distribution_Events', conn, if_exists='fail', index=False, dtype=sql_types)


# noinspection PyTypeChecker
def dsf62():
    f_name = 'q_stock62/dsf62.sas7bdat'
    db_name = 'StockData_dsf62_DailyOHLCV.db'
    if os.path.isfile(db_dir + db_name):
        print('WARNING: database file already exists. Exiting')
        return

    creation_stmnt = "CREATE TABLE IF NOT EXISTS '{}' (" \
                     + "Date DATE UNIQUE NOT NULL," \
                     + "Open REAL," \
                     + "Ask_High REAL," \
                     + "Bid_Low REAL," \
                     + "Close REAL," \
                     + "Return REAL," \
                     + "Return_noDiv REAL," \
                     + "Volume REAL," \
                     + "Shares_Outstanding REAL," \
                     + "Cum_Price_Adjust_Factor REAL," \
                     + "Cum_Shares_Adjust_Factor REAL," \
                     + "Bid REAL," \
                     + "Ask REAL," \
                     + "Nasdaq_NumTrades INTEGER," \
                     + "PRIMARY KEY (Date)" \
                     + ");"

    group_key = 'PERMNO'  # IF you intend to group the data, this is where you specify which field to use
    with sqlite3.connect(db_dir + db_name) as conn:
        cur = conn.cursor()
        for df in pd.read_sas(base_dir + f_name, chunksize=5000000):
            print()
            print(df.iloc[-1][['PERMNO', 'DATE']])
            key_list = df[group_key].unique().tolist()
            df.loc[:, 'DATE'] = df['DATE'].dt.date
            group = df.groupby(by=group_key, sort=False)
            for k in key_list:
                cur.execute(creation_stmnt.format(int(k)))
                conn.commit()
                df = group.get_group(k)
                list_of_tuples = list(zip(
                    df['DATE'],
                    df['OPENPRC'],
                    df['ASKHI'],
                    df['BIDLO'],
                    df['PRC'],
                    df['RET'],
                    df['RETX'],
                    df['VOL'],
                    df['SHROUT'],
                    df['CFACPR'],
                    df['CFACSHR'],
                    df['BID'],
                    df['ASK'],
                    df['NUMTRD']
                ))
                cur.executemany("INSERT INTO '" + str(int(k)) + "' VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                list_of_tuples)
        conn.commit()
        cur.close()


if __name__ == '__main__':
    # overview()
    dsf62()
