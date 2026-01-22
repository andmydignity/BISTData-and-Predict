import argparse
import sqlite3
import pandas as pd
import numpy as np
import joblib
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

DB_PATH = "bist_historicdata.db"
MODEL_PATH = "bist_model_sector_3m.pkl"
FEATURES_PATH = "model_features.pkl"
SECTOR_DB_PATH = "bist_companies.db"


def load_sector_map():
    sector_map = {}
    try:
        conn = sqlite3.connect(SECTOR_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, sector FROM companies")
        rows = cursor.fetchall()
        for ticker, sector in rows:
            if ticker and sector:
                sector_map[ticker] = sector.strip()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to load sector map: {e}")
    return sector_map


def get_latest_data():
    logger.info("Loading latest data for all tickers...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%_fin' AND name NOT LIKE 'macro_indicators' AND name NOT LIKE 'sqlite_%'"
    )
    tables = [r[0] for r in cursor.fetchall()]

    sector_map = load_sector_map()

    data_list = []

    # Prepare a cursor for financial queries
    fin_cursor = conn.cursor()

    for tick in tables:
        if tick not in sector_map:
            continue

        try:
            # Fetch last 40 rows
            query = f"""
                SELECT HGDG_TARIH, HGDG_KAPANIS, HGDG_HACIM, HGDG_MAX, HGDG_MIN,
                       FK, PD_DD, FD_FAVOK, B_M, OP_E, Asset_Growth, MOM_TL, MOM_USD, MOM_Gold,
                       Size_TL, Size_USD, Size_Gold, Vol_TL, Vol_USD, Vol_Gold, RSI_14, SMA_Ratio, foreign_rate,
                       Gross_Margin, Net_Margin, Operating_Margin, Debt_Equity, Current_Ratio, Leverage, P_CF, Revenue_Growth, ROE, ROA, Earnings_Quality,
                       Asset_Turnover, Net_Income_Growth, ST_Debt_To_Total,
                       Rel_FK, Rel_PD_DD, Rel_FD_FAVOK, Rel_B_M, Rel_OP_E,
                       Rel_Gross_Margin, Rel_Net_Margin, Rel_Operating_Margin, Rel_Debt_Equity, Rel_Current_Ratio, Rel_Leverage,
                       Rel_P_CF, Rel_Revenue_Growth, Rel_Asset_Turnover, Rel_Net_Income_Growth, Rel_ST_Debt_To_Total,
                       Rel_Asset_Growth, Rel_ROE, Rel_ROA, Rel_Earnings_Quality
                FROM {tick} 
                ORDER BY HGDG_TARIH DESC 
                LIMIT 40
            """
            df_hist = pd.read_sql(query, conn)

            if df_hist.empty:
                continue

            # Sort ascending for calculation
            df_hist = df_hist.sort_values("HGDG_TARIH")

            # --- Calculate Technicals ---
            # 1. Volume Average (20)
            avg_vol_20 = df_hist["HGDG_HACIM"].tail(20).mean()
            current_vol = df_hist["HGDG_HACIM"].iloc[-1]

            # 2. ATR(14)
            high = df_hist["HGDG_MAX"]
            low = df_hist["HGDG_MIN"]
            close = df_hist["HGDG_KAPANIS"]
            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean().iloc[-1]

            # Get latest row
            row = df_hist.iloc[-1].to_dict()

            # Filter Checks
            price = row["HGDG_KAPANIS"]

            # Store calculated metrics
            row["ATR_Pct"] = atr_14 / price if price > 0 else 0
            row["Avg_Vol_20"] = avg_vol_20
            row["Vol_Ratio"] = current_vol / avg_vol_20 if avg_vol_20 > 0 else 0
            row["StopLoss"] = price - (2 * atr_14)

            # Map columns
            row["Ticker"] = tick
            row["Sector"] = sector_map[tick]
            row["ForeignRate"] = row.pop("foreign_rate")
            row["Price"] = row.pop("HGDG_KAPANIS")
            row["Date"] = row.pop("HGDG_TARIH")
            row["Volume"] = row.pop("HGDG_HACIM")

            # --- Fetch Financial Date ---
            try:
                fin_cursor.execute(
                    f"SELECT period FROM {tick}_fin ORDER BY period DESC LIMIT 1"
                )
                fin_row = fin_cursor.fetchone()
                row["Financial_Date"] = fin_row[0] if fin_row else "N/A"
            except:
                row["Financial_Date"] = "N/A"
            # -----------------------------

            data_list.append(row)

        except Exception as e:
            # logger.error(f"Error {tick}: {e}")
            pass

    conn.close()
    return pd.DataFrame(data_list)


def predict():
    parser = argparse.ArgumentParser(description="Predict Sector Leaders")
    parser.add_argument(
        "-n", "--num", type=int, default=20, help="Number of results to display"
    )
    parser.add_argument(
        "-if",
        "--ignore_filters",
        action="store_true",
        help="Ignore all filters except Volume",
    )
    args = parser.parse_args()

    # 1. Load Model & Features
    try:
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        logger.info(f"Loaded model and {len(features)} features.")
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        return

    # 2. Get Data
    df = get_latest_data()
    logger.info(f"Got data for {len(df)} tickers. Latest Date: {df['Date'].max()}")

    # 3. Prepare Features
    # Ensure foreign_rate exists (model uses lowercase, we have Capitalized)
    if "ForeignRate" in df.columns and "foreign_rate" not in df.columns:
        df["foreign_rate"] = df["ForeignRate"]

    for f in features:
        if f in df.columns:
            df[f] = df[f].astype(float)
        else:
            logger.warning(f"Feature {f} missing from data!")
            df[f] = np.nan

    # Clean Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 4. Predict
    X = df[features]
    logger.info("Predicting...")
    # Add check for model feature count vs X count? LightGBM usually handles it if names match.
    # But dropped features might cause issues if model expects them.
    # We should assume model_features.pkl matches the loaded model.
    probs = model.predict_proba(X)[:, 1]
    df["Win_Probability"] = probs

    # 5. Apply Filters
    # User Request:
    # SMA_Ratio < 1.30
    # RSI_14 < 80
    # Volume > (Avg_Volume_20d * 1.2)
    # ATR(14)/Price < 0.05
    # Plus Liquidity > 30M

    MIN_VOLUME = 30_000_000

    # Create boolean masks for reporting
    df["ForeignRate"] = pd.to_numeric(df["ForeignRate"], errors="coerce")
    df["SMA_Ratio"] = pd.to_numeric(df["SMA_Ratio"], errors="coerce")
    df["RSI_14"] = pd.to_numeric(df["RSI_14"], errors="coerce")
    df["PD_DD"] = pd.to_numeric(df["PD_DD"], errors="coerce")
    df["FD_FAVOK"] = pd.to_numeric(df["FD_FAVOK"], errors="coerce")

    mask_liquidity = df["Volume"] >= MIN_VOLUME
    mask_sma = df["SMA_Ratio"] < 1.50
    mask_rsi = (df["RSI_14"] < 85) | ((df["RSI_14"] >= 85) & (df["SMA_Ratio"] < 1.15))
    mask_vol_breakout = df["Volume"] > (df["Avg_Vol_20"] * 0.8)
    mask_atr = df["ATR_Pct"] < 0.07
    mask_pddd = df["PD_DD"] < 20
    mask_quality = df["ForeignRate"] > 1.0
    # Exempt Financials from EV/EBITDA filter
    mask_fdfavok = (df["Sector"] == "MALİ KURULUŞLAR") | (
        (df["FD_FAVOK"] > 0) & (df["FD_FAVOK"] < 35)
    )

    if args.ignore_filters:
        df_filtered = df[mask_liquidity].copy()
        filter_msg = f"Filters: Vol>60M | **OTHER FILTERS IGNORED (-if)**"
    else:
        df_filtered = df[
            mask_liquidity
            & mask_sma
            & mask_rsi
            & mask_vol_breakout
            & mask_atr
            & mask_pddd
            & mask_fdfavok
            & mask_quality
        ].copy()
        filter_msg = f"Filters: Vol>60M | SMA<1.5 | RSI<85 | Vol>0.8xAvg | ATR<7% | PD/DD<20 | 0<FD/FAVOK<35 (Excl. Financials)"

    # df_filtered = df[mask_liquidity & mask_pddd & mask_quality].copy()

    logger.info(f"Original: {len(df)}")
    logger.info(f"Filtered: {len(df_filtered)}")
    logger.info(f" - Low Liquidity (<30M): {len(df) - mask_liquidity.sum()}")
    # logger.info(f" - High SMA (>=1.3): {len(df) - mask_sma.sum()}")
    # logger.info(f" - High RSI (>=80): {len(df) - mask_rsi.sum()}")
    # logger.info(f" - No Vol Breakout (<1.2x): {len(df) - mask_vol_breakout.sum()}")
    # logger.info(f" - High ATR (>=5%): {len(df) - mask_atr.sum()}")
    # logger.info(f" - High Valuation (PD/DD>=5): {len(df) - mask_pddd.sum()}")
    # logger.info(f" - Bad Valuation (FD/FAVOK not 0-15): {len(df) - mask_fdfavok.sum()}")

    # 6. Output Top N
    top_n = args.num
    top_results = df_filtered.sort_values("Win_Probability", ascending=False).head(
        top_n
    )

    print("\n" + "=" * 160)
    print(f"TOP {top_n} PREDICTIONS (Model: **SECTOR LEADER** / 3 Month Horizon)")
    print(filter_msg)
    print(f"Date: {df['Date'].max()}")
    print("=" * 160)

    # Format for display
    display_cols = [
        "Ticker",
        "Sector",
        "Price",
        "StopLoss",
        "Win_Prob",
        "FK",
        "PD_DD",
        "EV_EBITDA",
        "P_CF",
        "RSI",
        "Vol_X",
        "For_Rate",
        "Curr_Ratio",
        "Debt_Tot",
        "Fin_Date",
    ]

    # Rename columns for compact display
    top_results = top_results.rename(
        columns={
            "Win_Probability": "Win_Prob",
            "RSI_14": "RSI",
            "Vol_Ratio": "Vol_X",
            "ForeignRate": "For_Rate",
            "FD_FAVOK": "EV_EBITDA",
            "Current_Ratio": "Curr_Ratio",
            "ST_Debt_To_Total": "Debt_Tot",
            "Financial_Date": "Fin_Date",
        }
    )

    # Add rank
    top_results = top_results.reset_index(drop=True)
    top_results.index += 1

    print(
        top_results[display_cols].to_string(
            formatters={
                "Win_Prob": "{:.2%}".format,
                "Price": "{:.2f}".format,
                "StopLoss": "{:.2f}".format,
                "RSI": "{:.1f}".format,
                "FK": "{:.2f}".format,
                "PD_DD": "{:.2f}".format,
                "EV_EBITDA": "{:.2f}".format,
                "P_CF": "{:.2f}".format,
                "Vol_X": "{:.2f}x".format,
                "For_Rate": "{:.2f}%".format,
                "Curr_Ratio": "{:.2f}".format,
                "Debt_Tot": "{:.2%}".format,
            }
        )
    )
    print("=" * 160)


if __name__ == "__main__":
    predict()
