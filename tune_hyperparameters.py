import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import json

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

DB_PATH = "bist_historicdata.db"
SECTOR_DB_PATH = "bist_companies.db"
RESULTS_FILE = "tuning_results.json"
PLOT_FILE = "tuning_results.png"
ben = [
    0,  # FK
    0,  # PD_DD
    0,  # FD_FAVOK
    0,  # B_M
    0,  # OP_E
    1,  # Asset_Growth (Expansion is good)
    1,  # MOM_TL (Trend)
    1,  # MOM_USD
    1,  # MOM_Gold
    0,  # Size_TL
    0,  # Size_USD
    0,  # Size_Gold
    0,  # Vol_TL
    0,  # Vol_USD
    0,  # Vol_Gold
    0,  # foreign_rate
    0,  # RSI_14 (Mean Reversion?)
    1,  # SMA_Ratio (Above SMA is bullish)
    # NEW FEATURES
    1,  # Gross_Margin
    1,  # Net_Margin
    1,  # Operating_Margin
    0,  # Debt_Equity (Ambiguous)
    0,  # Current_Ratio
    0,  # Leverage
    0,  # P_CF
    1,  # Revenue_Growth
    1,  # ROE
    1,  # ROA
    1,  # Earnings_Quality
    1,  # Asset_Turnover
    1,  # Net_Income_Growth
    0,  # ST_Debt_To_Total
    # RELATIVE FEATURES (Replicate logic)
    0,  # Rel_FK
    0,  # Rel_PD_DD
    0,  # Rel_FD_FAVOK
    0,  # Rel_B_M
    0,  # Rel_OP_E
    1,  # Rel_Gross_Margin
    1,  # Rel_Net_Margin
    1,  # Rel_Operating_Margin
    0,  # Rel_Debt_Equity
    0,  # Rel_Current_Ratio
    0,  # Rel_Leverage
    0,  # Rel_P_CF
    1,  # Rel_Revenue_Growth
    1,  # Rel_Asset_Turnover
    1,  # Rel_Net_Income_Growth
    0,  # Rel_ST_Debt_To_Total
    1,  # Rel_Asset_Growth
    1,  # Rel_ROE
    1,  # Rel_ROA
    1,  # Rel_Earnings_Quality
]

gemini = [
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    1,  # 16. foreign_rate -> ARTIK 1 (Yabancı iyidir)
    0,
    1,
    1,
    1,
    1,
    -1,  # 22. Debt_Equity -> ARTIK -1 (Borç kötüdür)
    1,  # 23. Current_Ratio -> ARTIK 1 (Likidite iyidir)
    -1,  # 24. Leverage -> ARTIK -1 (Kaldıraç risktir)
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    -1,  # 32. ST_Debt_To_Total -> ARTIK -1 (KV Borç risktir)
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    -1,  # 41. Rel_Debt_Equity -> -1
    1,  # 42. Rel_Current_Ratio -> 1
    -1,  # 43. Rel_Leverage -> -1
    0,
    1,
    1,
    1,
    -1,  # 48. Rel_ST_Debt_To_Total -> -1
    1,
    1,
    1,
    1,
]


# --- 1. Data Loading (Same as train_sector_full.py) ---
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


def load_data():
    logger.info("Loading Data...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%_fin' AND name NOT LIKE 'macro_indicators' AND name NOT LIKE 'sqlite_%'"
    )
    tables = [r[0] for r in cursor.fetchall()]

    sector_map = load_sector_map()
    logger.info(f"DEBUG: Found {len(tables)} tables in DB.")
    logger.info(f"DEBUG: Found {len(sector_map)} tickers in Sector Map.")
    all_dfs = []

    # Feature Columns to Load
    features_sql = """
        FK, PD_DD, FD_FAVOK, B_M, OP_E, Asset_Growth,
        MOM_TL, MOM_USD, MOM_Gold, 
        Size_TL, Size_USD, Size_Gold,
        Vol_TL, Vol_USD, Vol_Gold, 
        foreign_rate,
        RSI_14, SMA_Ratio,
        Gross_Margin, Net_Margin, Operating_Margin,
        Debt_Equity, Current_Ratio, Leverage, P_CF, Revenue_Growth,
        ROE, ROA, Earnings_Quality,
        Asset_Turnover, Net_Income_Growth, ST_Debt_To_Total,
        Rel_FK, Rel_PD_DD, Rel_FD_FAVOK, Rel_B_M, Rel_OP_E,
        Rel_Gross_Margin, Rel_Net_Margin, Rel_Operating_Margin,
        Rel_Debt_Equity, Rel_Current_Ratio, Rel_Leverage,
        Rel_P_CF, Rel_Revenue_Growth, Rel_Asset_Turnover, Rel_Net_Income_Growth, Rel_ST_Debt_To_Total,
        Rel_Asset_Growth, Rel_ROE, Rel_ROA, Rel_Earnings_Quality
    """

    cols = ["HGDG_TARIH", "HGDG_KAPANIS"] + [x.strip() for x in features_sql.split(",")]

    for tick in tables:
        if tick not in sector_map:
            continue
        try:
            query = f"SELECT {','.join(cols)} FROM {tick}"
            df = pd.read_sql(query, conn)
            if df.empty:
                continue

            df.columns = ["date", "Price"] + [
                c.strip() for c in features_sql.split(",")
            ]
            df["Ticker"] = tick
            df["Sector"] = sector_map[tick]
            df["date"] = pd.to_datetime(df["date"])
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {tick}: {e}")
            pass

    conn.close()
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs)


def create_target(df):
    logger.info("Creating Targets...")
    df = df.sort_values(["Ticker", "date"])
    df["Price_Fwd"] = df.groupby("Ticker")["Price"].shift(-63)  # 3 Months
    df["Return_3M"] = (df["Price_Fwd"] / df["Price"]) - 1
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["Return_3M"])

    # Sector Relative Rank
    df["Return_Rank"] = df.groupby(["date", "Sector"])["Return_3M"].transform(
        lambda x: x.rank(pct=True)
    )
    df["Target"] = (df["Return_Rank"] > 0.90).astype(int)
    return df


# --- Device Detection ---
def get_device_params():
    try:
        # Try simple test: Does LightGBM accept gpu?
        # Or faster: check if nvidia-smi works
        import subprocess

        subprocess.check_call(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        logger.info("GPU Detected. Using device='gpu'.")
        return {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}
    except:
        logger.info("GPU not detected or nvidia-smi failed. Using device='cpu'.")
        return {"device": "cpu"}


# --- Progress Callback ---
def simple_progress_callback(period=50):
    def callback(env):
        if (env.iteration + 1) % period == 0:
            import sys

            sys.stdout.write(f"\r[Fold Progress] Iteration {env.iteration + 1}...")
            sys.stdout.flush()

    return callback


# --- 2. Tuning Loop ---
def run_tuning():
    df = load_data()
    if df.empty:
        logger.error("No data found!")
        return

    df = create_target(df)

    # FILTER DATE (Foreign Rate Reliable Start)
    START_DATE = "2009-07-06"
    df = df[df["date"] >= START_DATE]
    logger.info(f"Filtered Data Start Date: {START_DATE} (Rows: {len(df)})")

    # Clean Features
    features = [
        "FK",
        "PD_DD",
        "FD_FAVOK",
        "B_M",
        "OP_E",
        "Asset_Growth",
        "MOM_TL",
        "MOM_USD",
        "MOM_Gold",
        "Size_TL",
        "Size_USD",
        "Size_Gold",
        "Vol_TL",
        "Vol_USD",
        "Vol_Gold",
        "foreign_rate",
        "RSI_14",
        "SMA_Ratio",
        "Gross_Margin",
        "Net_Margin",
        "Operating_Margin",
        "Debt_Equity",
        "Current_Ratio",
        "Leverage",
        "P_CF",
        "Revenue_Growth",
        "ROE",
        "ROA",
        "Earnings_Quality",
        "Asset_Turnover",
        "Net_Income_Growth",
        "ST_Debt_To_Total",
        "Rel_FK",
        "Rel_PD_DD",
        "Rel_FD_FAVOK",
        "Rel_B_M",
        "Rel_OP_E",
        "Rel_Gross_Margin",
        "Rel_Net_Margin",
        "Rel_Operating_Margin",
        "Rel_Debt_Equity",
        "Rel_Current_Ratio",
        "Rel_Leverage",
        "Rel_P_CF",
        "Rel_Revenue_Growth",
        "Rel_Asset_Turnover",
        "Rel_Net_Income_Growth",
        "Rel_ST_Debt_To_Total",
        "Rel_Asset_Growth",
        "Rel_ROE",
        "Rel_ROA",
        "Rel_Earnings_Quality",
    ]

    for f in features:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before_drop = len(df)
    # df = df.dropna(subset=features) # LightGBM handles NaNs natively
    after_drop = len(df)
    logger.info(f"Data Rows: {before_drop} -> {after_drop} (Kept NaNs for LightGBM)")

    # Sort for Time Series Split (Critical!)
    df = df.sort_values("date").reset_index(drop=True)

    X = df[features]
    y = df["Target"]
    dates = df["date"]

    # Parameter Grid (Scale_pos_weight added for imbalance)
    param_grid = [
        {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 3000,
            "learning_rate": 0.01,
            "num_leaves": 96,
            "min_data_in_leaf": 120,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "is_unbalance": False,
            "scale_pos_weight": 2,
            "monotone_constraints": ben,
            "monotone_constraints_method": "advanced",
        },
        {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 3000,
            "learning_rate": 0.01,
            "num_leaves": 96,
            "min_data_in_leaf": 120,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "is_unbalance": False,
            "scale_pos_weight": 3,
            "monotone_constraints": ben,
            "monotone_constraints_method": "advanced",
        },
    ]

    tscv = TimeSeriesSplit(n_splits=5)

    results = []

    device_params = get_device_params()
    logger.info("Starting 5-Fold Walk-Forward Validation...")

    for param_idx, params in enumerate(param_grid):
        logger.info(f"Testing Config {param_idx + 1}/{len(param_grid)}: {params}")

        # Merge device params
        current_params = {**params, **device_params}

        fold_aucs = []
        fold_precs = []

        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Sub-sampled dates for weight calculation
            train_dates = pd.to_datetime(dates.iloc[train_index])
            min_date = train_dates.min()
            max_date = train_dates.max()
            days_diff = (train_dates - min_date).dt.days
            max_diff = (max_date - min_date).days
            # Avoid division by zero if single day (unlikely but safe)
            if max_diff == 0:
                train_weights = np.ones(len(train_dates))
            else:
                train_weights = 0.5 + 0.5 * (days_diff / max_diff)

            # Train with Fallback
            try:
                # Add callbacks
                cbs = [simple_progress_callback(50)]

                model = lgb.LGBMClassifier(
                    random_state=42, n_jobs=-1, verbose=-1, **current_params
                )

                model.fit(X_train, y_train, callbacks=cbs, sample_weight=train_weights)
                import sys

                sys.stdout.write("\n")  # Newline after fold finishes
            except Exception as e:
                if current_params.get("device") == "gpu":
                    logger.warning(f"GPU Failed in Tuning ({e}). Retrying with CPU.")
                    current_params["device"] = "cpu"
                    current_params.pop("gpu_platform_id", None)
                    current_params.pop("gpu_device_id", None)

                    model = lgb.LGBMClassifier(
                        random_state=42, n_jobs=-1, verbose=-1, **current_params
                    )
                    model.fit(
                        X_train, y_train, callbacks=cbs, sample_weight=train_weights
                    )
                    import sys

                    sys.stdout.write("\n")
                else:
                    raise e

            # Predict
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
            fold_aucs.append(auc)

            # Precision @ 10% (Top Decile Precision)
            res_df = pd.DataFrame({"Target": y_test, "Prob": probs})
            res_df = res_df.sort_values("Prob", ascending=False)
            top_k = int(len(res_df) * 0.10)  # Top 10%
            if top_k > 0:
                top_preds = res_df.iloc[:top_k]
                prec_k = top_preds["Target"].mean()
            else:
                prec_k = 0.0
            fold_precs.append(prec_k)

            logger.info(f"  Fold {fold + 1} AUC: {auc:.4f} | Prec@10%: {prec_k:.4f}")

        avg_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        avg_prec = np.mean(fold_precs)

        logger.info(
            f"Config {param_idx + 1} Result: Avg AUC={avg_auc:.4f}, Prec@10%={avg_prec:.4f}"
        )

        results.append(
            {
                "params": params,
                "fold_scores": fold_aucs,
                "fold_precs": fold_precs,
                "avg_auc": avg_auc,
                "avg_prec": avg_prec,
                "std_auc": std_auc,
                "label": f"Est={params['n_estimators']}, LR={params['learning_rate']}, L={params['num_leaves']}",
            }
        )

    # Save Results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 6))

    # Plot AUC per fold for each config
    folds_x = range(1, 6)

    for res in results:
        plt.plot(
            folds_x,
            res["fold_scores"],
            marker="o",
            label=f"{res['label']} (Avg: {res['avg_auc']:.3f})",
        )

    plt.title("5-Fold Walk-Forward Validation: Model Stability")
    plt.xlabel("Fold Number (Time ->)")
    plt.ylabel("AUC Score")
    plt.xticks(folds_x, [f"Fold {i} (Latest)" for i in folds_x])  # Rough labels
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    logger.info(f"Saved plot to {PLOT_FILE}")

    # Find Best
    best_res = max(results, key=lambda x: x["avg_auc"])
    logger.info("=" * 60)
    logger.info(f"🏆 BEST CONFIGURATION: {best_res['label']}")
    logger.info(f"   Avg AUC: {best_res['avg_auc']:.4f}")
    logger.info(f"   Params: {best_res['params']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_tuning()
