
import sqlite3
import pandas as pd
import concurrent.futures
import time
import random
import logging
import argparse
import requests
import json
import numpy as np
import concurrent.futures
import time
import random
import logging
import argparse
import requests
import json
import numpy as np
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from isyatirimhisse import fetch_stock_data, fetch_financials, fetch_index_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fetch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

FAILED_TICKERS_FILE = "failed_tickers.log"

SIMPLE_FINANCIAL_CODES = [] # Deprecated

INDUSTRIAL_CODES = [
    '1BL',  # TOPLAM VARLIKLAR (Total Assets)
    '1A',   # DÖNEN VARLIKLAR (Current Assets)
    '2N',   # ÖZKAYNAKLAR (Total Equity)
    '3C',   # SATIŞ GELİRLERİ (Revenue)
    '3D',   # BRÜT KAR (Gross Profit)
    '3DF',  # FAALİYET KARI (Operating Income)
    '3L',   # DÖNEM KARI (Net Income)
    '4C',   # İŞLETME FAALİYETLERİNDEN NAKİT (Operating Cash Flow)
    '2A',   # KISA VADELİ YÜKÜMLÜLÜKLER (Short Term Liabilities)
    '2B'    # UZUN VADELİ YÜKÜMLÜLÜKLER (Long Term Liabilities)
]

BANK_CODES = [
    '1Z',   # TOTAL ASSETS
    '2O',   # SHAREHOLDERS' EQUITY
    '3A',   # INTEREST INCOME (Revenue Proxy)
    '3C',   # NET INTEREST INCOME (Gross Profit Proxy)
    '3CH',  # NET OPERATING INCOME (Op Income Proxy)
    '3Z',   # NET PROFIT/LOSSES
    '2Z'    # TOTAL LIABILITIES
]
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def setup_databases(source_db_path, target_db_path):
    """
    Connect to source DB to get tickers and setup target DB for historical data.
    """
    try:
        conn_source = sqlite3.connect(source_db_path)
        cursor_source = conn_source.cursor()
        cursor_source.execute("SELECT ticker FROM companies")
        tickers = [row[0] for row in cursor_source.fetchall()]
        conn_source.close()
        logger.info(f"Found {len(tickers)} tickers in {source_db_path}")
    except sqlite3.Error as e:
        logger.error(f"Error reading from {source_db_path}: {e}")
        raise
    return tickers

def fetch_and_store(ticker, start_date, conn, append_mode=False):
    """
    Fetch data for a single ticker and store it in the database.
    """
    try:
        df = fetch_stock_data(symbols=ticker, start_date=start_date)
        
        if df is not None and not df.empty:
            # Mode handling
            if_exists_mode = 'append' if append_mode else 'replace'
            
            df.to_sql(ticker, conn, if_exists=if_exists_mode, index=False)
            logger.info(f"{ticker}: Saved {len(df)} rows ({if_exists_mode}).")
            return True

    except Exception as e:
        logger.error(f"Error fetching/storing data for {ticker}: {e}")
        return False

# --- CONCURRENCY HELPER ---
def process_ticker_concurrent(ticker, start_date, db_target, append_mode, only_financials, sector_map):
    """
    Worker function for ThreadPoolExecutor.
    Opens its own DB connection to avoid thread safety issues.
    """
    # Create thread-local connection
    local_conn = sqlite3.connect(db_target, check_same_thread=False)
    
    try:
        # 1. Fetch Price
        if not only_financials:
            success = fetch_and_store(ticker, start_date, local_conn, append_mode=append_mode)
            if not success:
                logger.warning(f"{ticker}: Price fetch failed.")
                local_conn.close()
                return False
        
        # 2. Fetch Financials
        start_year = start_date.split('-')[-1]
        
        # Smart Sector Logic
        sector_group = '1' # Default Industrial
        if ticker in sector_map:
            sec_name = sector_map[ticker].lower()
            if 'banka' in sec_name or 'bank' in sec_name or 'sigorta' in sec_name:
                sector_group = '3'
            
        fetch_financials_history(ticker, start_year, local_conn, initial_group=sector_group)
        
        # Rate Limit (Per thread)
        time.sleep(0.2) 
        
        local_conn.close()
        return True
    except Exception as e:
        logger.error(f"Thread Error {ticker}: {e}")
        try:
            local_conn.close()
        except:
            pass
        return False

def fetch_financials_history(ticker, start_year, conn, initial_group='1'):
    cursor = conn.cursor()
    table_name = f"{ticker}_fin"
    # Ensure table
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (period TEXT, item_code TEXT, value_try REAL, value_usd REAL, PRIMARY KEY (period, item_code))")
    conn.commit()

    current_year = datetime.now().year
    start_year_int = int(start_year)
    chunk_size = 10
    
    for year_chunk_start in range(start_year_int, current_year + 1, chunk_size):
        year_chunk_end = min(year_chunk_start + chunk_size - 1, current_year)
        y_start, y_end = str(year_chunk_start), str(year_chunk_end)
        
        # Optimization: Try provided group first
        groups_to_try = [initial_group]
        if initial_group == '1': 
            groups_to_try.append('3')
        else:
            groups_to_try.append('1')
            
        valid_fetch = False
        
        for grp in groups_to_try:
            try:
                df_try = fetch_financials(ticker, y_start, y_end, 'TRY', grp)
                
                if df_try is not None and not df_try.empty:
                    # Found it! Fetch USD
                    try:
                        df_usd = fetch_financials(ticker, y_start, y_end, 'USD', grp)
                    except:
                        df_usd = None
                        
                    # Define Mappings
                    MAP_G1 = {
                        '1BL': 'TOTAL_ASSETS', '1A': 'CURRENT_ASSETS', '2N': 'EQUITY',
                        '3C': 'REVENUE', '3D': 'GROSS_PROFIT', '3DF': 'OPERATING_PROFIT',
                        '3L': 'NET_INCOME', '4C': 'OPERATING_CASH_FLOW',
                        '2A': 'SHORT_TERM_LIABILITIES', '2B': 'LONG_TERM_LIABILITIES'
                    }
                    MAP_G3 = {
                        '1Z': 'TOTAL_ASSETS', '2O': 'EQUITY', '3CH': 'OPERATING_PROFIT', 
                        '3K': 'OPERATING_PROFIT', '3Z': 'NET_INCOME', '2Z': 'TOTAL_LIABILITIES'
                    }
                    mapping = MAP_G3 if grp == '3' else MAP_G1
                    
                    # Filtering Logic
                    base_exclusions = ['FINANCIAL_ITEM_CODE', 'item_code', 'SYMBOL', 'FINANCIAL_ITEM_DESC', 'FINANCIAL_ITEM_NAME_TR', 'FINANCIAL_ITEM_NAME_EN']
                    
                    df_try = df_try[df_try['FINANCIAL_ITEM_CODE'].isin(mapping.keys())].copy()
                    df_try['item_code'] = df_try['FINANCIAL_ITEM_CODE'].map(mapping)
                    vals = [c for c in df_try.columns if c not in base_exclusions]
                    melted_try = df_try.melt(id_vars=['item_code'], value_vars=vals, var_name='period', value_name='value_try')
                    
                    if df_usd is not None:
                        df_usd = df_usd[df_usd['FINANCIAL_ITEM_CODE'].isin(mapping.keys())].copy()
                        df_usd['item_code'] = df_usd['FINANCIAL_ITEM_CODE'].map(mapping)
                        vals_u = [c for c in df_usd.columns if c not in base_exclusions]
                        melted_usd = df_usd.melt(id_vars=['item_code'], value_vars=vals_u, var_name='period', value_name='value_usd')
                        final_df = pd.merge(melted_try, melted_usd, on=['item_code', 'period'], how='left')
                    else:
                        final_df = melted_try
                        final_df['value_usd'] = np.nan
                        
                    # Write
                    records = final_df.to_dict('records')
                    cursor.execute("BEGIN TRANSACTION")
                    for r in records:
                         cursor.execute(f"INSERT OR REPLACE INTO {table_name} (period, item_code, value_try, value_usd) VALUES (?, ?, ?, ?)",
                                       (r['period'], r['item_code'], r['value_try'], r['value_usd']))
                    conn.commit()
                    valid_fetch = True
                    break 
            except Exception:
                continue # Try next group
        
        if not valid_fetch:
             pass # No data for chunk

def add_foreign_rate_column(conn, tickers):
    """
    Ensure all ticker tables have a 'foreign_rate' column.
    """
    cursor = conn.cursor()
    logger.info("Ensuring 'foreign_rate' column exists in all tables...")
    for ticker in tickers:
        try:
            # Check if table exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{ticker}'")
            if not cursor.fetchone():
                continue

            # Check if column exists
            cursor.execute(f"PRAGMA table_info({ticker})")
            columns = [info[1] for info in cursor.fetchall()]
            if 'foreign_rate' not in columns:
                cursor.execute(f"ALTER TABLE {ticker} ADD COLUMN foreign_rate REAL")
        except Exception as e:
            logger.error(f"Error adding column to {ticker}: {e}")
    conn.commit()

def fetch_foreign_ownership_history(start_date_str, conn, tickers):
    """
    Iterate daily from start_date to today, bulk fetch foreign ownership, and update tables.
    """
    logger.info("Starting Foreign Ownership History Fetch...")
    
    start_date = datetime.strptime(start_date_str, "%d-%m-%Y")
    end_date = datetime.now()
    
    session = requests.Session()
    url = "https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website/StockInfo/CompanyInfoAjax.aspx/GetYabanciOranlarXHR"
    headers = {
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Content-Type": "application/json; charset=utf-8"
    }

    # Cache which tables exist to avoid updating non-existent tables
    existing_tables = set()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for row in cursor.fetchall():
        existing_tables.add(row[0])

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%d-%m-%Y")

        # Bulk Request
        payload = {
            "baslangicTarih": date_str,
            "bitisTarihi": date_str,
            "sektor": None,
            "endeks": "09", # 09 = All items usually
            "hisse": None
        }

        try:
            response = session.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                items = data.get("d", [])
                
                if items:
                    logger.info(f"Date {date_str}: Found {len(items)} records. Updating DB...")
                    
                    # Batch Update
                    # HGDG_TARIH format in DB is "YYYY-MM-DD 00:00:00"
                    db_date_str = current_date.strftime("%Y-%m-%d 00:00:00")
                    
                    cursor = conn.cursor()
                    cursor.execute("BEGIN TRANSACTION")
                    
                    for item in items:
                        ticker = item.get("HISSE_KODU")
                        rate = item.get("YAB_ORAN_END")

                        if rate is not None:
                            if isinstance(rate, str):
                                rate = rate.replace(',', '.')
                            try:
                                rate = float(rate)
                            except ValueError:
                                rate = None
                        
                        if ticker in existing_tables and rate is not None:
                            # Update specific row
                            cursor.execute(f"UPDATE {ticker} SET foreign_rate = ? WHERE HGDG_TARIH = ?", (rate, db_date_str))
                    
                    conn.commit()
                else:
                    logger.debug(f"Date {date_str}: No data (d is empty).")
            else:
                logger.warning(f"Date {date_str}: HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"Error fetching/updating foreign ownership for {date_str}: {e}")

        # Rate Limit
        current_date += timedelta(days=1)
        time.sleep(random.uniform(0.5, 1.5))

def fetch_macro_data(conn, start_date=None):
    """
    Fetches Macro/Index data (e.g. GLDGR) and stores in 'macro_indicators' table.
    """
    logger.info("Fetching Macro Data (GLDGR)...")
    try:
        # Force full history if no date provided, otherwise use provided date
        s_date = start_date if start_date else '01-04-2008'
        logger.info(f"Using Start Date for Macro: {s_date}")
        
        df = fetch_index_data(indices=['GLDGR'], start_date=s_date)
        
        if df is not None and not df.empty:
            # Rename for consistency
            # returned cols: INDEX, DATE, VALUE
            df = df.rename(columns={'INDEX': 'code', 'DATE': 'date', 'VALUE': 'value'})
            
            # Ensure date is standard string YYYY-MM-DD or similar
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # Write to DB
            df.to_sql('macro_indicators', conn, if_exists='replace', index=False)
            logger.info(f"Saved {len(df)} macro records.")
        else:
            logger.warning("No macro data returned.")

    except Exception as e:
        logger.error(f"Error fetching macro data: {e}")

def get_latest_date_from_db(conn):
    """
    Finds the latest date present in the database to resume fetching from.
    Checks a few major tickers to be sure.
    """
    check_tickers = ['THYAO', 'GARAN', 'AKBNK', 'EREGL']
    latest_date = None
    
    cursor = conn.cursor()
    # first check valid tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    
    for ticker in check_tickers:
        if ticker in tables:
            try:
                cursor.execute(f"SELECT MAX(HGDG_TARIH) FROM {ticker}")
                res = cursor.fetchone()
                if res and res[0]:
                    d_str = res[0] # YYYY-MM-DD HH:MM:SS
                    # Parse
                    if " " in d_str:
                        d_obj = datetime.strptime(d_str.split(" ")[0], "%Y-%m-%d")
                    else:
                        d_obj = datetime.strptime(d_str, "%Y-%m-%d")
                        
                    if latest_date is None or d_obj > latest_date:
                        latest_date = d_obj
            except Exception:
                continue
    
    return latest_date

def load_sector_map_db(db_source):
    s_map = {}
    try:
        c = sqlite3.connect(db_source)
        cur = c.cursor()
        cur.execute("SELECT ticker, sector FROM companies")
        for t, s in cur.fetchall():
            s_map[t] = s
        c.close()
    except:
        pass
    return s_map

def main():
    parser = argparse.ArgumentParser(description='Fetch BIST historical data.')
    parser.add_argument('--limit', type=int, help='Limit number of tickers to process (for testing)')
    parser.add_argument('--start-date', default=None, help='Start date (DD-MM-YYYY). If not set, resumes from DB.')
    parser.add_argument("--only-financials", action="store_true", help="Skip price data, fetch only financials.")
    parser.add_argument('--db-source', default='bist_companies.db', help='Source DB for tickers')
    parser.add_argument('--db-target', default='bist_historicdata.db', help='Target DB for historical data')
    parser.add_argument('--skip-prices', action='store_true', help='Skip price fetching, only do foreign ownership')
    parser.add_argument('--only-macro', action='store_true', help='Skip everything else, ONLY fetch Gold/Macro data.')
    
    args = parser.parse_args()

    if args.only_macro:
        conn = sqlite3.connect(args.db_target)
        # Use provided start date or default to full history
        s_date = args.start_date if args.start_date else '01-04-2008'
        fetch_macro_data(conn, start_date=s_date)
        conn.close()
        logger.info("Macro Data Update Complete.")
        return

    try:
        tickers = setup_databases(args.db_source, args.db_target)
    except Exception:
        return

    # Load Sector Map for Smart Fetching
    sector_map = load_sector_map_db(args.db_source)

    if args.limit:
        tickers = tickers[:args.limit]
        logger.info(f"Limiting to first {args.limit} tickers")

    conn_target = sqlite3.connect(args.db_target)
    
    # Determine Start Date
    start_date_str = args.start_date
    append_mode = False
    
    if not start_date_str:
        logger.info("No start date provided. Checking DB for latest data to resume...")
        last_db_date = get_latest_date_from_db(conn_target)
        
        if last_db_date:
            # Resume from next day
            resume_date = last_db_date + timedelta(days=1)
            start_date_str = resume_date.strftime("%d-%m-%Y")
            logger.info(f"Resuming from {start_date_str} (Last DB Date: {last_db_date.strftime('%Y-%m-%d')})")
            append_mode = True
        else:
            # Default fallback
            start_date_str = "01-01-2010"
            logger.info(f"No existing data found. Defaulting to {start_date_str}")
            
    # Assign back to args for consistency in calls below
    args.start_date = start_date_str
    
    if append_mode:
        logger.info("Running in APPEND mode.")
    else:
        logger.info("Running in REPLACE/FULL mode.")

    conn_target.close() # Close main thread connection before threads start

    # 1. Fetch Prices & Financials (Concurrent)
    if not args.skip_prices:
        # Clean failed tickers log
        with open(FAILED_TICKERS_FILE, 'w') as f:
            f.write("")

        logger.info(f"Starting Turbo Fetch (5 Threads)... Mode: {'Append' if append_mode else 'Full'}")
    
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(
                    process_ticker_concurrent, 
                    ticker, args.start_date, args.db_target, append_mode, args.only_financials, sector_map
                ): ticker 
                for ticker in tickers
            }
            
            count = 0
            total = len(tickers)
            for future in concurrent.futures.as_completed(futures):
                tck = futures[future]
                try:
                    if future.result():
                        count += 1
                    if count % 10 == 0:
                        logger.info(f"Progress: {count}/{total}")
                except Exception as e:
                    logger.error(f"Failed {tck}: {e}")
                    with open(FAILED_TICKERS_FILE, 'a') as f:
                        f.write(f"{ticker}\n")

    # Re-open for sequential parts
    conn_final = sqlite3.connect(args.db_target)

    # 3. Add Column for Foreign Rate
    add_foreign_rate_column(conn_final, tickers)

    # 4. Fetch Foreign Ownership
    # Note: We loop dates, not tickers.
    fetch_foreign_ownership_history(args.start_date, conn_final, tickers)

    # 5. Fetch Macro Data
    fetch_macro_data(conn_final, start_date='01-04-2008')
    
    conn_final.close()
    logger.info("Done.")

if __name__ == "__main__":
    main()
