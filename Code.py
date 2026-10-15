# Volatility-targeted SPY strategy.
# A random forest forecasts SPY's volatility over the next 5 trading days. Exposure to SPY is scaled
# down when forecast volatility is high. The forecast is benchmarked against calibrated VIX and
# naive baselines, and the strategy against buy-and-hold, on a held-out test period (2022 onward).

from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import numpy as np
from itertools import product
from sklearn.linear_model import LinearRegression

#Settings
LIMIT_DATE = "2026-10-10"
TRADING_DAYS = 252
TRAIN_START = "2010-01-01"      # analysis starts here; 2009 data is only a warm-up for rolling windows
TEST_START = "2022-01-01"       # everything from this date on is the untouched test period
FIT_YEARS = 5
VALIDATION_YEARS = 1
FEATURES = ["Vol_5d", "Vol_10d", "Vol_21d", "SMA_ratio", "VIX_level", "VIX_over_5d", "VIX_5d"]
HORIZON = 5                     # forecast horizon in trading days
TRANSACTION_COST = 0.0005       # 0.05% per unit of position change
SEED = 25
MAX_DEPTH = [3, 5, 8]
MIN_SAMPLES_LEAF = [10, 25, 50, 100]
N_ESTIMATORS = 100


try:
    CSV_FILE = Path(__file__).resolve().parent / "market_data.csv"
except NameError:               # notebook / interactive shell
    CSV_FILE = Path.cwd() / "market_data.csv"

def load_data(path = CSV_FILE):
    # Prices are cached in a CSV; delete it to re-download.
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = yf.download(["SPY", "^VIX", "^IRX"], start="2009-01-01", end=LIMIT_DATE, auto_adjust=True, progress=False)["Close"]
        df.rename(columns = {"^IRX": "IRX", "^VIX": "VIX"}, inplace = True)
        df.ffill(inplace = True) # missing: irx - 4, spy-2
        df.to_csv(path)
    return df

def build_features(df):
    # Every feature uses data up to and including day t only (no look-ahead).
    df = df.copy()
    df["Returns"] = df["SPY"].pct_change()
    df["Vol_5d"] = df["Returns"].rolling(5).std() * np.sqrt(TRADING_DAYS)       # annualized realized volatility
    df["Vol_10d"] = df["Returns"].rolling(10).std() * np.sqrt(TRADING_DAYS)
    df["Vol_21d"] = df["Returns"].rolling(21).std() * np.sqrt(TRADING_DAYS)
    df["SMA_ratio"] = df["SPY"] / df["SPY"].rolling(200).mean()                 # trend: price vs 200-day average
    # T-bill yield (% per year) -> daily decimal; lagged one day because it is only known after the fact.
    df["Cash_returns"] = (df["IRX"] / 100 / TRADING_DAYS).shift(1)
    df["VIX_5d"] = df["VIX"].diff(5)
    df["VIX_over_5d"] = df["VIX"] - df["VIX"].rolling(5).mean()
    df["VIX_level"] = df["VIX"] / 100                                           # decimal, same units as realized volatility
    df = df.loc[TRAIN_START:]                                                   # drop the warm-up after the rolling windows are computed
    return df

def create_target(df):
    # Volatility over days t+1 ... t+5: the rolling value at t+5 is shifted back to row t.
    # The last HORIZON rows have no full future window and stay NaN.
    df = df.copy()
    df["future_vol"] = (df["Returns"].rolling(HORIZON).std() * np.sqrt(TRADING_DAYS)).shift(-HORIZON)
    return df

def split_data(df):
    # Split by time. Purge: the last HORIZON training rows have targets built from test-period returns.
    df_train = df[df.index < TEST_START].iloc[:-HORIZON]
    df_test = df[df.index >= TEST_START]
    return df_train, df_test

def cluster_data(df_train):
    # Rolling-window folds inside the training years: fit on 5 years, validate on the next year.
    fold_no = pd.Timestamp(TEST_START).year - pd.Timestamp(TRAIN_START).year - FIT_YEARS - VALIDATION_YEARS + 1
    if fold_no < 1:
        raise ValueError("Not possible to train on current settings, reduce the training and/or validation periods")
    folds = []
    for fold in range(fold_no):
        fit_start = pd.Timestamp(TRAIN_START) + pd.DateOffset(years = fold)
        val_start = pd.Timestamp(TRAIN_START) + pd.DateOffset(years = fold + FIT_YEARS)
        # Purge by rows, not calendar days: drop the last HORIZON trading days of the fit window.
        fit_end = df_train.loc[fit_start:val_start - pd.DateOffset(days = 1)].index[-(HORIZON+1)]
        val_end = val_start + pd.DateOffset(years = VALIDATION_YEARS) - pd.DateOffset(days = 1)
        folds.append({"fit_start": fit_start, "fit_end": fit_end, "val_start": val_start, "val_end": val_end, "fold_number": fold})
    return folds

def train_model(df_train, max_depth_var, min_samples_leaf_var, fold):
    train = df_train[fold["fit_start"]:fold["fit_end"]]
    model = RandomForestRegressor(
        n_estimators = N_ESTIMATORS,
        max_depth = max_depth_var,
        min_samples_leaf = min_samples_leaf_var,
        random_state = SEED
    )
    target = train["future_vol"]
    model.fit(train[FEATURES], target)
    return model

def optimize_variables(df_train, folds):
    # Choose depth and leaf size by average validation RMSE. The test period is never used here.
    target = "future_vol"
    scores = []
    for depth, leaf in product(MAX_DEPTH, MIN_SAMPLES_LEAF):
        for fold in folds:
            model = train_model(df_train, depth, leaf, fold)
            val = df_train[fold["val_start"]:fold["val_end"]]
            rmse = np.sqrt(((model.predict(val[FEATURES])-val[target]) ** 2).mean())
            scores.append([fold["fold_number"], depth, leaf, rmse])
    df_scores = pd.DataFrame(scores, columns=["fold", "depth", "leaf", "rmse"])
    mean_rmse = df_scores.groupby(["depth", "leaf"])["rmse"].mean()
    best_depth, best_leaf = mean_rmse.idxmin()
    return best_depth, best_leaf

def backtest(prediction, df_test, target_vol, vix_calibrated):
    # Position = share of the portfolio held in SPY, the rest in cash.
    # Volatility targeting: position = target volatility / forecast volatility, capped at 1 (no leverage).
    # shift(1): the position decided at the close of day t is held on day t+1. Day one starts fully invested.
    test = df_test.copy()
    signals = {"Buy_and_hold_SPY": pd.Series(1.0, index = test.index),
               "VIX": pd.Series(target_vol/vix_calibrated, index = test.index).clip(upper = 1.0).shift(1).fillna(1.0),
               "Prediction": pd.Series(target_vol/prediction, index = test.index).clip(upper = 1.0).shift(1).fillna(1.0),
               "Fixed_exposure": pd.Series(0.8, index = test.index),            # arbitrary 80% benchmark
               "Cash_only": pd.Series(0.0, index = test.index)
    }
    # Daily return = position * SPY + (1 - position) * cash - cost * size of the position change.
    daily_returns = {"Buy_and_hold_SPY": signals["Buy_and_hold_SPY"] * test["Returns"] + (1-signals["Buy_and_hold_SPY"]) * test["Cash_returns"],
                     "VIX": signals["VIX"] * test["Returns"] + (1-signals["VIX"]) * test["Cash_returns"] - signals["VIX"].diff().abs().fillna(0.0) * TRANSACTION_COST,
                     "Prediction": signals["Prediction"] * test["Returns"] + (1-signals["Prediction"]) * test["Cash_returns"] - signals["Prediction"].diff().abs().fillna(0.0) * TRANSACTION_COST,
                     "Fixed_exposure": signals["Fixed_exposure"] * test["Returns"] + (1-signals["Fixed_exposure"]) * test["Cash_returns"],
                     "Cash_only": signals["Cash_only"] * test["Returns"] + (1-signals["Cash_only"]) * test["Cash_returns"]
    }

    return pd.DataFrame(daily_returns), pd.DataFrame(signals)

def get_metrics(daily_returns, signals, cash_returns):
    metrics = {}
    for strategy in signals:
        r = daily_returns[strategy]
        p = signals[strategy]
        metrics[strategy] = {"Annual_Return": (1+r).prod() ** (TRADING_DAYS / len(r)) - 1,         # compounded, annualized
                             "Volatility": r.std() * np.sqrt(TRADING_DAYS),
                             "Sharpe": ((r - cash_returns).mean() / (r - cash_returns).std()) * np.sqrt(TRADING_DAYS),   # excess return over cash; NaN for Cash_only
                             "Max_drawdown": ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min(),   # worst fall from a running peak
                             "Avg_exposure": p.mean(),
                             "Turnover": p.diff().abs().mean()
                            }
    return pd.DataFrame(metrics).T

def score_forecasts(df_test, forecasts):
    # The last HORIZON test rows have no realized volatility yet, so they are excluded from scoring only.
    test = df_test.dropna(subset=["future_vol"])
    scores = {}
    for name in forecasts:
        error = pd.Series(forecasts[name], index=df_test.index).loc[test.index] - test["future_vol"]
        scores[name] = {"RMSE": np.sqrt((error ** 2).mean()),
                        "MAPE": (error.abs() / test["future_vol"]).mean()}
    return pd.DataFrame(scores).T

def print_results(metrics, forecast_scores):
    pct = "{:.1%}".format
    print("\nStrategy results (test period, after costs)")
    print(metrics.to_string(
        formatters={"Annual_Return": pct, "Volatility": pct, "Max_drawdown": pct,
                    "Avg_exposure": pct, "Turnover": pct, "Sharpe": "{:.2f}".format},
        na_rep="n/a"))
    print("\nForecast accuracy (test period, lower is better)")
    print(forecast_scores.sort_values("RMSE").to_string(
        formatters={"RMSE": "{:.4f}".format, "MAPE": pct}))

def main():
    df = create_target(build_features(load_data()))
    df_train, df_test = split_data(df)
    # Raw VIX overstates realized volatility, so calibrate it with a regression fitted on training data only.
    train_vix = df_train.dropna(subset=["future_vol"])
    reg = LinearRegression().fit(train_vix[["VIX_level"]], train_vix["future_vol"])
    vix_calibrated = reg.predict(df_test[["VIX_level"]]).clip(min = 0.01)   # floor avoids dividing by ~0
    # Target volatility is also a training-data statistic.
    target_vol = df_train["future_vol"].median()
    folds = cluster_data(df_train)
    final_training_fold = {"fit_start": pd.Timestamp(TEST_START) - pd.DateOffset(years = FIT_YEARS), "fit_end": pd.Timestamp(TEST_START) - pd.DateOffset(days = 1)}
    max_depth, min_samples_leaf = optimize_variables(df_train, folds)
    # Ensemble: one model per validation window plus a final model on the latest 5 years, averaged.
    folds.append(final_training_fold)
    models = []
    predictions = []
    for fold in folds:
        model = train_model(df_train, max_depth, min_samples_leaf, fold)
        models.append(model)
    for model in models:
        prediction = model.predict(df_test[FEATURES]).clip(min = 0.01)
        predictions.append(prediction)
    model_prediction = np.mean(predictions, axis=0)
    daily_returns_df, signals_df = backtest(model_prediction, df_test, target_vol, vix_calibrated)
    metrics = get_metrics(daily_returns_df, signals_df, df_test["Cash_returns"])
    forecasts = {"Model": model_prediction,
                 "VIX_calibrated": vix_calibrated,
                 "VIX_raw": df_test["VIX_level"],
                 "Vol_21d": df_test["Vol_21d"],
                 "Train_mean": df_train["future_vol"].mean()
    }
    forecast_scores = score_forecasts(df_test, forecasts)
    print_results(metrics, forecast_scores)

if __name__ == "__main__":
    main()