#!/home/carl/miniconda3/envs/trading/bin/python
import os
import sqlite3
import pandas as pd

primary = '/mnt/data/stocks/StockData_CRSP62_DailyCloses.db'
secondary = '/mnt/data/stocks/StockData_CRSP62_DailyOHL.db'
joined_db = '/mnt/data/stocks/StockData_CRSP62_DailyOHLCV.db'
dtype_map = {'Close': float,
             'Return': float,
             'Return_noDiv': float,
             'Capitalization': float,
             'Volume': float,
             'Bid_Low': float,
             'Ask_High': float,
             'Bid': float,
             'Ask': float,
             'Nasdaq_NumTrades': int,
             'Open': float}

creation_stmnt = "CREATE TABLE IF NOT EXISTS '{}' (" \
    + "Date DATE UNIQUE NOT NULL," \
    + "Open REAL," \
    + "Ask_High REAL," \
    + "Bid_Low REAL," \
    + "Close REAL," \
    + "Return REAL," \
    + "Return_noDiv REAL," \
    + "Volume REAL," \
    + "Capitalization REAL," \
    + "Bid REAL," \
    + "Ask REAL," \
    + "Nasdaq_NumTrades INTEGER," \
    + "PRIMARY KEY (Date)" \
    + ");"


def main():
    with sqlite3.connect(primary) as con_primary, sqlite3.connect(secondary) as con_secondary, sqlite3.connect(joined_db) as con_final:
        tables = pd.read_sql_query("SELECT name FROM sqlite_schema WHERE type='table';", con_primary)['name'].tolist()
        cur = con_final.cursor()
        for t in tables:
            # do creation separately so i can batch the commits on the inserts for speed
            cur.execute(creation_stmnt.format(t))
        ctr = 0
        con_final.commit()
        for t in tables:
            df = pd.read_sql_query("SELECT * FROM '{}'".format(t), con_primary, index_col='Date')
            df_secondary = pd.read_sql_query("SELECT * FROM '{}'".format(t), con_secondary, index_col='Date')
            if df.shape[0] != df_secondary.shape[0]:
                print('\ndifferent record count for PERMNO ' + t)
            df = df.join(df_secondary, on=None).astype(dtype_map, copy=False, errors='ignore')
            check_uniqueness = df.index.difference(df.index.unique())
            if check_uniqueness.shape[0] > 0:
                print('\nfound duplicate dates for PERMNO ' + t)
                print(check_uniqueness)
            list_of_tuples = list(zip(
                df.index,
                df['Open'],
                df['Ask_High'],
                df['Bid_Low'],
                df['Close'],
                df['Return'],
                df['Return_noDiv'],
                df['Volume'],
                df['Capitalization'],
                df['Bid'],
                df['Ask'],
                df['Nasdaq_NumTrades']
            ))
            cur.executemany("INSERT INTO '" + t + "' VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", list_of_tuples)
            ctr += 1
            if ctr % 500 == 0:
                con_final.commit()
            del df, df_secondary, list_of_tuples
        else:
            con_final.commit()
            cur.close()

# putting this here after i finished with the creation and data upload for the joined_db file so i
# dont accidentally modify that file by running this script.
# It shouldn't modify because of the uniqueness constraints but this is more explicit
if __name__ == '__main__':
    # main()
    pass
