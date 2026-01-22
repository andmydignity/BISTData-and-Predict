
import sqlite3
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SECTOR_FILE = "Sektörler.csv"
DB_PATH = "bist_historicdata.db"
START_DATE = "2008-04-01"

def load_sector_mapping(csv_path):
    sector_map = {}
    current_sector = None
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line: continue
        if line[0].isdigit():
            import csv
            reader = csv.reader([line])
            row = next(reader)
            if len(row) >= 2:
                tickers_raw = row[1]
                ts = [t.strip() for t in tickers_raw.split(',')]
                for t in ts:
                    if t and current_sector:
                        sector_map[t] = current_sector
        else:
            if line.startswith("Sıra,Kod"): continue
            clean_line = line.replace(",,", "").strip()
            if clean_line:
                current_sector = clean_line.replace('"', '').strip()
    logger.info(f"Loaded {len(sector_map)} tickers mapped to sectors.")
    return sector_map

def load_financials(conn, ticker):
    try:
        query = f"SELECT * FROM {ticker}_fin"
        df = pd.read_sql(query, conn)
    except:
        return None
    if df.empty: return None

    df['value'] = df['value_try']
    
    def parse_period(p):
        try:
            y, q = map(int, p.split('/'))
            m = q
            import calendar
            last_day = calendar.monthrange(y, m)[1]
            # Lag 90 days
            return pd.Timestamp(year=y, month=m, day=last_day) + pd.Timedelta(days=90)
        except:
            return pd.NaT

    df['date'] = df['period'].apply(parse_period)
    df = df.dropna(subset=['date'])
    
    # --- ANNUALIZATION LOGIC ---
    # Parse period month to determine multiplier
    def get_month(p):
        try:
            return int(p.split('/')[1])
        except:
            return 12

    df['month'] = df['period'].apply(get_month)
    
    # Flow items that need annualization
    flow_items = [
        'REVENUE', 'GROSS_PROFIT', 'OPERATING_PROFIT', 'NET_INCOME', 
        'OPERATING_CASH_FLOW', 'INTEREST_INCOME', 'NET_INTEREST_INCOME', 
        'NET_OPERATING_INCOME', 'NET_PROFIT_LOSSES'
    ]
    
    # Apply multiplier only to flow items
    # Multiplier = 12 / month
    # e.g. Month 3 => 12/3 = 4. Month 12 => 1.
    
    mask_flow = df['item_code'].isin(flow_items)
    
    # We must be careful about negative values? Yes, Math works same.
    # Ex: -1B in 3M => -4B Annualized. Correct.
    
    df.loc[mask_flow, 'value'] = df.loc[mask_flow, 'value'] * (12.0 / df.loc[mask_flow, 'month'])
    
    df = df.drop_duplicates(subset=['date', 'item_code'])
    return df.pivot(index='date', columns='item_code', values='value')

def load_macro_data(conn):
    try:
        df = pd.read_sql("SELECT * FROM macro_indicators WHERE code='GLDGR'", conn)
    except:
        return None
    if df.empty: return None
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    df = df.rename(columns={'value': 'Gold_Price'})
    return df[['Gold_Price']]

def process_ticker(conn, ticker, df_macro):
    # 1. Load ALL source columns
    try:
        df = pd.read_sql(f"SELECT * FROM {ticker}", conn)
    except:
        return None
        
    if df.empty: return None

    # Identify Date column
    if 'HGDG_TARIH' in df.columns:
        df['date'] = pd.to_datetime(df['HGDG_TARIH'])
    else:
        return None
        
    df = df.sort_values('date').set_index('date')

    # Forward Fill ALL source columns (Project requirement)
    df = df.ffill()

    # 2. Join Macro
    if df_macro is not None:
        # Drop duplicative Gold_Price if it exists in source from previous runs
        if 'Gold_Price' in df.columns:
            df = df.drop(columns=['Gold_Price'])
            
        df = df.join(df_macro, how='left')
        df['Gold_Price'] = df['Gold_Price'].ffill()
    else:
        df['Gold_Price'] = np.nan

    # Helper for Safe Division
    def safe_div(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = a / b
        # Replace Inf/-Inf with NaN
        if isinstance(result, (pd.Series, pd.DataFrame)):
            result = result.replace([np.inf, -np.inf], np.nan)
        return result

    # 3. Calculate Base Metrics
    # Using specific column names known from schema
    # "HGDG_KAPANIS" -> Price_TL
    # "PD" -> PD_TL
    # "PD_USD" -> PD_USD
    
    p_tl = df['HGDG_KAPANIS'] if 'HGDG_KAPANIS' in df.columns else None
    pd_tl = df['PD'] if 'PD' in df.columns else None
    pd_usd = df['PD_USD'] if 'PD_USD' in df.columns else None
    
    if p_tl is None or pd_tl is None:
        return None

    df['Price_Gold'] = safe_div(p_tl, df['Gold_Price'])
    df['PD_Gold'] = safe_div(pd_tl, df['Gold_Price'])

    # 4. Load Financials & Merge
    df_fin = load_financials(conn, ticker)
    if df_fin is not None:
        df_fin = df_fin.sort_index()
        
        # Drop duplicates if exist in src
        cols_to_drop = [c for c in df_fin.columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            
        df = pd.merge_asof(
            df, df_fin, 
            left_index=True, right_index=True, 
            direction='backward', 
            tolerance=pd.Timedelta(days=365*2)
        )
    else:
        for c in ['EQUITY', 'NET_INCOME', 'OPERATING_PROFIT', 'TOTAL_ASSETS']:
            df[c] = np.nan

    # 5. Features
    # Helpers
    def get_col(name):
        if name in df.columns:
            return df[name]
        else:
            return pd.Series(np.nan, index=df.index)

    equity = get_col('EQUITY')
    net_inc = get_col('NET_INCOME')
    op_inc = get_col('OPERATING_PROFIT')
    assets = get_col('TOTAL_ASSETS')
    # New Fundamentals
    revenue = get_col('REVENUE')
    gross_profit = get_col('GROSS_PROFIT')
    op_cash_flow = get_col('OPERATING_CASH_FLOW')
    short_liab = get_col('SHORT_TERM_LIABILITIES')
    long_liab = get_col('LONG_TERM_LIABILITIES')
    curr_assets = get_col('CURRENT_ASSETS')

    # Ratios (Validation)
    # Ratios (Validation)
    df['FK'] = safe_div(pd_tl, net_inc)
    df['PD_DD'] = safe_div(pd_tl, equity)
    df['FD_FAVOK'] = safe_div(pd_tl, op_inc)
    df['B_M'] = safe_div(equity, pd_tl)
    df['OP_E'] = safe_div(op_inc, equity)
    
    # --- NEW: Advanced Fundamentals ---
    # 1. Margins
    df['Gross_Margin'] = safe_div(gross_profit, revenue)
    df['Net_Margin'] = safe_div(net_inc, revenue)
    df['Operating_Margin'] = safe_div(op_inc, revenue)
    
    # 2. Financial Health
    total_debt = short_liab + long_liab
    df['Debt_Equity'] = safe_div(total_debt, equity)
    df['Current_Ratio'] = safe_div(curr_assets, short_liab)
    df['Leverage'] = safe_div(assets, equity) # NEW
    
    # 3. Valuation & Efficiency Expanded
    # Price / Cash Flow (Market Cap / OCF)
    df['P_CF'] = safe_div(pd_tl, op_cash_flow)
    # Returns (NEW)
    df['ROE'] = safe_div(net_inc, equity)
    df['ROA'] = safe_div(net_inc, assets)
    # Earnings Quality (NEW)
    df['Earnings_Quality'] = safe_div(op_cash_flow, net_inc)
    # Efficiency (NEW)
    df['Asset_Turnover'] = safe_div(revenue, assets)
    
    # 3.1 Financial Health Expanded
    df['ST_Debt_To_Total'] = safe_div(short_liab, total_debt)
    
    # 4. Growth
    # Revenue Growth (Year over Year).
    df['Revenue_Growth'] = revenue.pct_change(periods=252)
    # Net Income Growth (YoY)
    df['Net_Income_Growth'] = net_inc.pct_change(periods=252)
    # ----------------------------------
    # ----------------------------------

    # Growth / Lagged
    # Calculate BEFORE filter
    df['Asset_Growth'] = assets.pct_change(periods=252)
    df['MOM_TL'] = p_tl.shift(21) / p_tl.shift(252) - 1
    
    # MOM USD
    if pd_usd is not None:
        # Price_USD Proxy = PD_USD (since Price_USD data not explicit, PD_USD tracks USD value)
        # Using derived Price_USD = Price_TL * (PD_USD/PD_TL)
        price_usd = p_tl * (pd_usd / pd_tl)
        df['MOM_USD'] = price_usd.shift(21) / price_usd.shift(252) - 1
        df['Size_USD'] = np.log(pd_usd)
        df['Vol_USD'] = price_usd.pct_change().rolling(21).std() * np.sqrt(252)
    else:
        df['MOM_USD'] = np.nan
        df['Size_USD'] = np.nan
        df['Vol_USD'] = np.nan
        
    # MOM Gold
    p_gold = df['Price_Gold']
    df['MOM_Gold'] = p_gold.shift(21) / p_gold.shift(252) - 1
    
    # Size
    df['Size_TL'] = np.log(pd_tl)
    df['Size_Gold'] = np.log(df['PD_Gold'])
    
    # Volatility
    df['Vol_TL'] = p_tl.pct_change().rolling(21).std() * np.sqrt(252)
    df['Vol_Gold'] = p_gold.pct_change().rolling(21).std() * np.sqrt(252)

    # --- NEW: Technical Indicators ---
    
    # 1. RSI (14)
    # 1. RSI (14)
    delta = p_tl.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = safe_div(gain, loss)
    # If loss is 0, RS is Inf (handled by safe_div as NaN or Inf, but ideally RSI should be 100)
    # If we specifically want RSI=100 when loss=0:
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # Fill cases where loss was 0 (resulting in RSI NaN or Inf issues)
    # If only gain > 0 and loss == 0, RSI -> 100
    df.loc[loss == 0, 'RSI_14'] = 100
    # If gain is also 0, RSI -> 50 (no move)
    df.loc[(gain == 0) & (loss == 0), 'RSI_14'] = 50
    
    # 2. SMA Ratio (Price / 50-day SMA)
    sma_50 = p_tl.rolling(window=50).mean()
    df['SMA_Ratio'] = safe_div(p_tl, sma_50)
    

    
    # 6. Filter Date
    df_final = df[df.index >= START_DATE]
    
    # Drop temp Join columns if desired?
    # User said "Add ... for each ticker". Probably keep Financial items too?
    # I'll keep everything in df_final. 
    # except maybe 'date' duplicate if index is reset
    
    # Reset index to restore HGDG_TARIH/date
    # HGDG_TARIH was original column. df.index is 'date'. 
    # to_sql will write index unless index=False. 
    # Let's update HGDG_TARIH with valid date and drop 'date' index.
    df_final = df_final.reset_index(drop=True)
    
    # Ensure HGDG_TARIH is preserved and correct? 
    # It exists in source columns. 
    # Since we filtered rows, HGDG_TARIH column values are also filtered.
    
    return df_final

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    logger.info("Parsing Sector Map...")
    sector_map = load_sector_mapping(SECTOR_FILE)
    
    conn = sqlite3.connect(DB_PATH)
    # Enable WAL for speed
    conn.execute("PRAGMA journal_mode=WAL;")
    
    logger.info("Loading Macro Data...")
    df_macro = load_macro_data(conn)
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%_fin' AND name NOT LIKE 'macro_indicators' AND name NOT LIKE 'sqlite_%'")
    tables = sorted([r[0] for r in cursor.fetchall()]) # Sort for deterministic order
    
    total = len(tables)
    count = 0
    
    logger.info(f"Processing {total} tables...")
    
    for ticker in tables:
        # Check if it's a company ticker
        if ticker not in sector_map:
            continue
            
        # logger.debug(f"Processing {ticker}...")
        # Check if it's a company ticker (in sector map or generally looks like one)
        # Using sector_map as a filter is good practice, but user said "for each ticker". 
        # Some tickers might not be in sector file? 
        # Let's process valid ones.
        if ticker not in sector_map:
            # Skip non-mapped tickers to be safe, or process all?
            # User previously focused on mapped ones.
            # I will process ONLY mapped ones to avoid corrupting meta-tables if any exist.
            continue
            
        try:
            df_new = process_ticker(conn, ticker, df_macro)
            if df_new is not None and not df_new.empty:
                # OVERWRITE Table
                # df_new.to_sql(ticker, conn, if_exists='replace', index=False)
                # But we can't write to same DB while reading? SQLite allows it.
                df_new.to_sql(ticker, conn, if_exists='replace', index=False)
                count += 1
                if count % 20 == 0:
                    logger.info(f"Updated {count}/{total} tickers")
        except Exception as e:
            logger.error(f"Failed {ticker}: {e}")
            
    conn.commit()
    
    # 2. Sector Normalization
    logger.info("Starting Sector Normalization...")
    calculate_sector_relatives(conn, sector_map)
    
    conn.close()
    logger.info("Done.")

def calculate_sector_relatives(conn, sector_map):
    import gc
    # Reverse map: Sector -> [Tickers]
    sectors = {}
    for t, s in sector_map.items():
        if s not in sectors: sectors[s] = []
        sectors[s].append(t)
        
    metrics = [
        'FK', 'PD_DD', 'FD_FAVOK', 'B_M', 'OP_E', 
        'Gross_Margin', 'Net_Margin', 'Operating_Margin',
        'Debt_Equity', 'Current_Ratio', 'Leverage',
        'P_CF', 'Revenue_Growth', 'Asset_Turnover', 'Net_Income_Growth', 'ST_Debt_To_Total',
        'Asset_Growth', 'ROE', 'ROA', 'Earnings_Quality'
    ]
    
    # Helper for Safe Division (reused logic)
    def safe_div_series(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            res = a / b
        res = res.replace([np.inf, -np.inf], np.nan)
        return res
        
    for sec_name, tickers in sectors.items():
        # Filter tickers that actually exist in DB
        valid_tickers = []
        cursor = conn.cursor()
        for t in tickers:
            try:
                # Check table existence
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{t}'")
                if cursor.fetchone():
                    valid_tickers.append(t)
            except: pass
            
        if len(valid_tickers) < 2:
            continue

        # SKIP COMPLETED SECTORS (Recovery Mode) - REMOVED for Full Run

        logger.info(f"Processing Sector: {sec_name} ({len(valid_tickers)} tickers) - Pass 1: Calculating Averages")
        
        # --- PASS 1: Calculate Sector Means Iteratively (Low RAM) ---
        sector_sum = None
        sector_count = None
        
        for t in valid_tickers:
            try:
                # Load only counts needed for mean
                cols = ", ".join(['HGDG_TARIH'] + metrics)
                q = f"SELECT {cols} FROM {t}"
                df = pd.read_sql(q, conn)
                df['HGDG_TARIH'] = pd.to_datetime(df['HGDG_TARIH'])
                df = df.set_index('HGDG_TARIH')
                
                # Deduplicate Index to prevent Alignment Explosion
                df = df[~df.index.duplicated(keep='first')]
                
                # Ensure numeric
                for m in metrics:
                   df[m] = pd.to_numeric(df[m], errors='coerce')
                
                if sector_sum is None:
                    # Initialize accumulator with aligned index (union of all dates will happen naturally if we reindex? No.)
                    # Easier: Just accumulate in aligned dataframes. 
                    # Use 'add' with fill_value=0, but we need to handle NaNs correctly for Mean.
                    # count should only incr if not NaN.
                    
                    sector_sum = df[metrics].fillna(0)
                    sector_count = df[metrics].notna().astype(int)
                else:
                    # We need to align indices (dates). outer join sum.
                    # align() is expensive? 
                    # efficient way: add(..., fill_value=0)
                    
                    sector_sum = sector_sum.add(df[metrics].fillna(0), fill_value=0)
                    sector_count = sector_count.add(df[metrics].notna().astype(int), fill_value=0)
                    
                del df
                gc.collect()
            except Exception as e:
                # logger.warning(f"Error reading {t} in Pass 1: {e}")
                pass
                
        if sector_sum is None:
            continue
            
        # Calculate Mean
        sector_mean = sector_sum / sector_count
        sector_mean = sector_mean.replace([np.inf, -np.inf], np.nan)
        
        # --- PASS 2: Calculate Relatives and Update (Iterative) ---
        logger.info(f"Processing Sector: {sec_name} - Pass 2: Updating DB")
        
        for t in valid_tickers:
            try:
                full_df = pd.read_sql(f"SELECT * FROM {t}", conn)
                full_df['HGDG_TARIH'] = pd.to_datetime(full_df['HGDG_TARIH'])
                full_df = full_df.set_index('HGDG_TARIH')
                
                # Deduplicate Index to prevent Join Explosion
                full_df = full_df[~full_df.index.duplicated(keep='first')]

                # We need to join with sector_mean
                # sector_mean has index 'HGDG_TARIH'
                
                # Join logic
                merged = full_df.join(sector_mean, rsuffix='_Avg')
                
                # Calculate Relatives
                for m in metrics:
                    avg_col = f"{m}_Avg"
                    rel_col = f"Rel_{m}"
                    
                    if avg_col in merged.columns:
                        val = pd.to_numeric(merged[m], errors='coerce')
                        avg = merged[avg_col]
                        merged[rel_col] = safe_div_series(val, avg)
                
                # Cleanup: remove _Avg columns if any remain (join might have added them)
                # Actually, we just need to save the new columns.
                # full_df now has keys.
                
                # Prepare to save
                # Reset index
                final_df = merged.reset_index()
                
                # Drop temp columns (like _Avg)
                cols_to_save = [c for c in final_df.columns if not c.endswith('_Avg')]
                final_df = final_df[cols_to_save]
                
                final_df.to_sql(t, conn, if_exists='replace', index=False)
                
                del full_df, merged, final_df
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error updating sector rels for {t} in Pass 2: {e}")
        
    logger.info(f"Completed Sector Normalization.")

if __name__ == "__main__":
    main()
