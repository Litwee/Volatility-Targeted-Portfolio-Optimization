from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import numpy as np
from itertools import product

#Settings
TRADING_DAYS = 252
TRAIN_START = "2010-01-01"
TEST_START = "2022-01-01"
FIT_YEARS = 5
VALIDATION_YEARS = 1
FEATURES = {"Vol_5d", "Vol_10d", "Vol_21d", "SMA_ratio", "Cash_returns"}
HORIZON = 5
LOG_TARGET = True
SEED = 25
MAX_DEPTH = [3, 5, 8]
MIN_SAMPLES_LEAF = [10, 25, 50, 100]


try:
    CSV_FILE = Path(__file__).resolve().parent / "market_data.csv"
except NameError:
    CSV_FILE = Path.cwd() / "market_data.csv"

def load_data(path = CSV_FILE):
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = yf.download(["SPY", "^VIX", "^IRX"], start="2009-01-01", auto_adjust=True, progress=False)["Close"]
        df.rename(columns = {"^IRX": "IRX", "^VIX": "VIX"}, inplace = True)
        df.ffill(inplace = True) # missing: irx - 4, spy-2
        df.to_csv(path)
    return df

def build_features(df):
    df = df.copy()
    df["Returns"] = df["SPY"].pct_change()
    df["Vol_5d"] = df["Returns"].rolling(5).std() * np.sqrt(TRADING_DAYS)
    df["Vol_10d"] = df["Returns"].rolling(10).std() * np.sqrt(TRADING_DAYS)
    df["Vol_21d"] = df["Returns"].rolling(21).std() * np.sqrt(TRADING_DAYS)
    df["SMA_ratio"] = df["SPY"] / df["SPY"].rolling(200).mean()
    df["Cash_returns"] = (df["IRX"] / 100 / TRADING_DAYS).shift(1)
    df["VIX_5d"] = df["VIX"].diff(5)
    df["VIX_over_5d"] = df["VIX"] - df["VIX"].rolling(5).mean()
    df["VIX_level"] = df["VIX"] / 100
    df = df.loc[TRAIN_START:]
    return df

def create_target(df):
    df = df.copy()
    df["future_vol"] = (df["Returns"].rolling(HORIZON).std() * np.sqrt(TRADING_DAYS)).shift(-HORIZON)
    df["log_future_vol"] = np.log(df["future_vol"])
    return df

def split_data(df):
    df_train = df[df.index < TEST_START].iloc[:-HORIZON]
    df_test = df[df.index >= TEST_START]
    return df_train, df_test

def cluster_data(df_train):
    fold_no = pd.Timestamp(TEST_START).year - pd.Timestamp(TRAIN_START).year - FIT_YEARS - VALIDATION_YEARS + 1
    if fold_no < 1:
        raise ValueError("Not possible to train on current settings, reduce the training and/or validation periods")
    folds = []
    for fold in range(fold_no):
        fit_start = pd.Timestamp(TRAIN_START) + pd.DateOffset(years = fold)
        val_start = pd.Timestamp(TRAIN_START) + pd.DateOffset(years = fold + FIT_YEARS)
        fit_end = df_train.loc[fit_start:val_start - pd.DateOffset(days = 1)].index[-(HORIZON+1)]
        val_end = val_start + pd.DateOffset(years = VALIDATION_YEARS) - pd.DateOffset(days = 1)
        folds.append({"fit_start": fit_start, "fit_end": fit_end, "val_start": val_start, "val_end": val_end})
    return folds

def train_model(df_train, max_depth_var, min_samples_leaf_var, fold):
    train = df_train[fold["fit_start"]:fold["fit_end"]]
    model = RandomForestRegressor(
        n_estimators= 100,
        max_depth= max_depth_var,
        min_samples_leaf= min_samples_leaf_var,
        random_state= SEED
    )
    if LOG_TARGET:
        target = train["log_future_vol"]
    else:
        target = train["future_vol"]
    model.fit(train[FEATURES], target)
    return model

def optimize_variables(df_train, folds):
    if LOG_TARGET:
        target = "log_future_vol"
    else:
        target = "future_vol"
    for depth, leaf in product[tuple[int, int]](MAX_DEPTH, MIN_SAMPLES_LEAF):
        scores = []
        for fold in folds:
            model = train_model(df_train, depth, leaf, fold)
            val = df_train[fold["val_start"]:fold["val_end"]]
            rmse = np.sqrt(((model.predict(val[FEATURES])-val[target]) ** 2).mean())
            scores.append([fold["fold_number"], depth, leaf, rmse])
    df_scores = pd.DataFrame(scores, columns=["fold", "depth", "leaf", "rmse"])
    mean_rmse = df_scores.groupby(["depth", "leaf"])["rmse"].mean()
    best_depth, best_leaf = mean_rmse.idxmin()
    best_rmse = mean_rmse.min()
    return best_depth, best_leaf, best_rmse, mean_rmse, df_scores



df = build_features(load_data())
df = create_target(df)
df_train, df_test = split_data(df)
clusters = cluster_data(df_train)
print(clusters)
