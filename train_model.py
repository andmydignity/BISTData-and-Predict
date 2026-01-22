
import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import logging
from datetime import datetime
import os
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

DB_PATH = "bist_historicdata.db"
SECTOR_FILE = "Sektörler.csv"
MODEL_PATH = "bist_model_3m.pkl"
START_DATE = "2009-07-06"

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
                # Clean up sector name if needed (similar to CSV logic to be safe)
                clean_sector = sector.strip()
                sector_map[ticker] = clean_sector
        conn.close()
    except Exception as e:
        logger.error(f"Failed to load sector map from DB: {e}")
        return {}

    print(f"DEBUG: Loaded {len(sector_map)} tickers in map from DB.")
    if len(sector_map) > 0:
        print(f"DEBUG: Sample: {list(sector_map.keys())[:5]}")
    return sector_map

def load_data():
    logger.info("Connecting to DB...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%_fin' AND name NOT LIKE 'macro_indicators' AND name NOT LIKE 'sqlite_%'")
    tables = [r[0] for r in cursor.fetchall()]

    print(f"DEBUG: Found {len(tables)} tables in DB.")
    if len(tables) > 0:
        print(f"DEBUG: DB Table Sample: {tables[:5]}")

    sector_map = load_sector_map()
    
    all_dfs = []
    
    logger.info(f"Loading data for {len(tables)} tickers...")
    count = 0
    for tick in tables:
        if tick not in sector_map: continue
        
        try:
            # We need Price for Target, and Features for Training
            # Note: foreign_rate is lowercase in DB schema
            cols = [
                "HGDG_TARIH", "HGDG_KAPANIS", 
                "FK", "PD_DD", "FD_FAVOK", "B_M", "OP_E", "Asset_Growth", 
                "MOM_TL", "MOM_USD", "MOM_Gold", 
                "Size_TL", "Size_USD", "Size_Gold", 
                "Vol_TL", "Vol_USD", "Vol_Gold", 
                "RSI_14", "SMA_Ratio", "foreign_rate",
                "Gross_Margin", "Net_Margin", "Operating_Margin",
                "Debt_Equity", "Current_Ratio", "Leverage",
                "P_CF", "Revenue_Growth", "ROE", "ROA", "Earnings_Quality",
                "Asset_Turnover", "Net_Income_Growth", "ST_Debt_To_Total",
                "Rel_FK", "Rel_PD_DD", "Rel_FD_FAVOK", "Rel_B_M", "Rel_OP_E",
                "Rel_Gross_Margin", "Rel_Net_Margin", "Rel_Operating_Margin",
                "Rel_Debt_Equity", "Rel_Current_Ratio", "Rel_Leverage",
                "Rel_P_CF", "Rel_Revenue_Growth", "Rel_Asset_Turnover", "Rel_Net_Income_Growth", "Rel_ST_Debt_To_Total",
                "Rel_Asset_Growth", "Rel_ROE", "Rel_ROA", "Rel_Earnings_Quality"
            ]
            query = f"SELECT {', '.join(cols)} FROM {tick}"
            df = pd.read_sql(query, conn)
            
            # Rename for consistency if needed, model uses 'ForeignRate' in feature list
            # df = df.rename(columns={'foreign_rate': 'ForeignRate'}) 

            if df.empty: continue
            
            df['date'] = pd.to_datetime(df['HGDG_TARIH'])
            df = df.set_index('date').sort_index()
            df = df[df.index >= START_DATE]
            
            df['Ticker'] = tick
            df['Sector'] = sector_map[tick]
            
            # Rename Price
            df = df.rename(columns={'HGDG_KAPANIS': 'Price'})
            
            all_dfs.append(df)
            count += 1
        except Exception as e:
            logger.warning(f"Error loading {tick}: {e}")
            pass
            
    conn.close()
    logger.info(f"Loaded {count} tickers. Concatenating...")
    full_df = pd.concat(all_dfs)
    return full_df

def create_target(df):
    logger.info("Creating Target (Top 10% BIST Performer vs All)...")
    
    # 1. Calculate Individual Forward Return (3 Months = 63 Trading Days)
    df = df.sort_values(['Ticker', 'date'])
    
    df['Price_Fwd'] = df.groupby('Ticker')['Price'].shift(-63)
    df['Return_3M'] = (df['Price_Fwd'] / df['Price']) - 1
    
    # Clean Infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    df = df[df['Return_3M'].notna()]
    
    # 2. Rank Returns Daily WITHIN EACH SECTOR
    # We want top 10% of the SECTOR
    # Use transform to calculate percentile per date AND Sector
    # Ensure date is available for grouping (it might be index)
    if 'date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
         df = df.reset_index()
         
    df['Return_Rank'] = df.groupby(['date', 'Sector'])['Return_3M'].transform(lambda x: x.rank(pct=True))
    
    # 3. Target = 1 if Rank > 0.90, else 0
    df['Target'] = (df['Return_Rank'] > 0.90).astype(int)
    
    logger.info(f"Target Distribution (Sector-Relative): {df['Target'].value_counts(normalize=True).to_dict()}")
    
    return df

def train(df):
    logger.info("Preparing Train/Test Split...")
    
    features = [
        'FK', 'PD_DD', 'FD_FAVOK', 
        'B_M', 'OP_E', 'Asset_Growth',
        'MOM_TL', 'MOM_USD', 'MOM_Gold', 
        'Size_TL', 'Size_USD', 'Size_Gold',
        'Vol_TL', 'Vol_USD', 'Vol_Gold', 
        'foreign_rate',
        'RSI_14', 'SMA_Ratio',
        # NEW FEATURES
        "Gross_Margin", "Net_Margin", "Operating_Margin",
        "Debt_Equity", "Current_Ratio", "Leverage",
        "P_CF", "Revenue_Growth", "ROE", "ROA", "Earnings_Quality",
        "Asset_Turnover", "Net_Income_Growth", "ST_Debt_To_Total",
        # RELATIVE FEATURES
        "Rel_FK", "Rel_PD_DD", "Rel_FD_FAVOK", "Rel_B_M", "Rel_OP_E",
        "Rel_Gross_Margin", "Rel_Net_Margin", "Rel_Operating_Margin",
        "Rel_Debt_Equity", "Rel_Current_Ratio", "Rel_Leverage",
        "Rel_P_CF", "Rel_Revenue_Growth", "Rel_Asset_Turnover", "Rel_Net_Income_Growth", "Rel_ST_Debt_To_Total",
        "Rel_Asset_Growth", "Rel_ROE", "Rel_ROA", "Rel_Earnings_Quality"
    ]
    
    # Ensure features are float and clean infs
    for f in features:
        df[f] = pd.to_numeric(df[f], errors='coerce')
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df = df.dropna(subset=features)
        
    # Time Series Split
    cutoff = pd.Timestamp("2024-01-01")
    
    train_df = df[df['date'] < cutoff]
    test_df = df[df['date'] >= cutoff]
    
    X_train = train_df[features]
    y_train = train_df['Target']
    
    X_test = test_df[features]
    y_test = test_df['Target']
    
    # Explicit Print
    print(f"DEBUG: Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    logger.info(f"Train Size: {len(X_train)}, Test Size: {len(X_test)}")
    
    # LightGBM Classifier
    logger.info("Training LightGBM Classifier...")
    
    params = {
        'objective': 'binary',
        'metric': 'auc', # Area Under Curve preferred for imbalance
        'boosting_type': 'gbdt',
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'num_leaves': 96,
        'min_data_in_leaf': 120,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'is_unbalance': False, # pct=0.10 is not super unbalanced, but scale_pos_weight is better
        'scale_pos_weight': 3, 
        'device': 'gpu', 
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbose': -1,
        'seed': 42
    }
    
    try:
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        )
    except Exception as e:
        logger.warning(f"GPU Failed ({e}). Switching to CPU.")
        params['device'] = 'cpu'
        del params['gpu_platform_id']
        del params['gpu_device_id']
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        )
        
    # Eval
    score = model.score(X_test, y_test) # Accuracy
    logger.info(f"Accuracy Score on Test: {score:.4f}")
    
    # Precision @ 10%
    probs = model.predict_proba(X_test)[:, 1]
    res_df = pd.DataFrame({'Target': y_test, 'Prob': probs})
    res_df = res_df.sort_values('Prob', ascending=False)
    top_k = int(len(res_df) * 0.10)
    if top_k > 0:
        prec_k = res_df.iloc[:top_k]['Target'].mean()
    else:
        prec_k = 0.0
    logger.info(f"Precision@10% on Test: {prec_k:.4f}")
    
    # Feature Importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importance)
    
    # Save
    logger.info(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(features, "model_features.pkl")
    
    return model

def main():
    df = load_data()
    df_valid = create_target(df)
    train(df_valid)
    logger.info("Done.")

if __name__ == "__main__":
    main()
