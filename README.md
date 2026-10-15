# Volatility Targeted Portfolio Optimization

Can a machine-learning forecast of SPY's volatility help manage risk, and does it add anything beyond VIX?

I train a random forest to forecast SPY's realized volatility over the next 5 trading days. A volatility-targeting rule then scales the share of the portfolio held in SPY (the rest sits in cash): hold less SPY when forecast volatility is high, more when it is low. I compare the forecast against VIX-based and naive baselines, and the strategy against buy-and-hold and other benchmarks, on a test period the model never touched.

## Key findings

- **Volatility targeting reduced risk.** Compared with buy-and-hold, the random-forest strategy cut volatility from 17.3% to 10.7% and max drawdown from -24.5% to -14.2%. It gave up return (9.1% vs 12.4% a year), and its Sharpe ratio was close (0.50 vs 0.54).
- **The model adds little beyond VIX.** Its forecast was close to, but not better than, a linear regression on VIX (RMSE 0.0742 vs 0.0716). The differences between the model and calibrated VIX, in both forecast error and strategy results, are too small for the model to be considered better.
- **It does beat naive forecasts.** Its RMSE is about 22% lower than a constant training-mean forecast and about 18% lower than 21-day realized volatility.
- **Timing did not improve risk-adjusted return.** A constant 80% SPY / 20% cash allocation matched buy-and-hold's Sharpe ratio (0.54), which was higher than the model's (0.50).
- **Results are stable across random seeds.** Rerunning with seeds 25, 11, 39 and 48 changed forecast RMSE by less than 0.0001 and MAPE by less than 0.3 percentage points, and the strategy metrics were similar.

## Results

Test period: 3 January 2022 to 9 October 2026 (about 4.75 years). Costs included.

**Strategies**

| Strategy | Annual return | Volatility | Sharpe | Max drawdown | Avg exposure | Turnover |
|---|---|---|---|---|---|---|
| Buy & hold SPY | 12.4% | 17.3% | 0.54 | -24.5% | 100.0% | 0.0% |
| Vol targeting, calibrated VIX | 8.6% | 11.0% | 0.45 | -15.8% | 78.0% | 4.1% |
| Vol targeting, random forest | 9.1% | 10.7% | 0.50 | -14.2% | 75.4% | 4.8% |
| Fixed 80% SPY | 11.0% | 13.8% | 0.54 | -19.7% | 80.0% | 0.0% |
| Cash only | 4.0% | 0.1% | n/a | 0.0% | 0.0% | 0.0% |

- *Avg exposure* is the average share of the portfolio held in SPY. *Turnover* is the average daily change in that share.
- Sharpe is computed on returns above the T-bill return, so it is undefined for cash only.
- The random-forest strategy has lower volatility and drawdown than the fixed 80% allocation, but it also holds less SPY on average (75.4% vs 80%), so the comparison is not exposure-matched. The fixed 80% was derived from the training data (an average over the whole training period), not from the test period.

**Forecast accuracy** (against realized 5-day volatility, lower is better)

| Forecast | RMSE | MAPE |
|---|---|---|
| Calibrated VIX | 0.0716 | 37.9% |
| Random forest | 0.0742 | 40.9% |
| Raw VIX | 0.0839 | 67.2% |
| 21-day realized volatility | 0.0905 | 48.4% |
| Training mean | 0.0955 | 50.2% |

RMSE is in annualized volatility units (0.0742 means about 7.4 percentage points). MAPE penalizes over-forecasts more heavily, so RMSE is the primary measure.

## Method

**Data.** SPY (dividend-adjusted close), VIX and the 13-week T-bill yield (`^IRX`) from Yahoo Finance. Analysis starts on 1 January 2010; 2009 data is used only as a warm-up so the rolling windows are valid from the start. A few missing prices were forward-filled.

**Target.** Annualized standard deviation of SPY's daily returns over the next 5 trading days.

**Features** (all known at the close of day t): 5-, 10- and 21-day realized volatility; price relative to its 200-day average; VIX level; VIX change over 5 days; VIX relative to its 5-day average.

**Train / test split.** By date, never random. Training is 2010 to 2021, test is 2022 onward. The last 5 training rows are dropped (purged), because their targets are built from test-period returns.

**Validation and tuning.** Seven rolling folds inside the training years: fit on 5 years, validate on the next year (2015 to 2021), with the last 5 trading days of each fit window purged. Tree depth (3, 5, 8) and minimum leaf size (10, 25, 50, 100) were chosen by average validation RMSE. Validation RMSE varied little across the 12 combinations (about 0.082 to 0.089), so these settings matter little.

**Model.** Random forest regressor with 100 trees. The final forecast is the average of eight forests: one per validation window, plus one fitted on the five years before the test period.

**Calibrated VIX.** Raw VIX tends to sit above subsequent realized volatility. I therefore fit a linear regression of realized volatility on VIX using training data only, and use its output as the VIX forecast.

**Position sizing.** Position = target volatility / forecast volatility, capped at 100% (no leverage). The target volatility is the median realized volatility in the training data. The model and calibrated VIX go through the same rule, so differences come from the forecast.

**Timing and costs.** A position decided at the close of day t is held on day t+1. Cash earns the T-bill yield, lagged one day. Each change in position costs 0.05% of the amount traded.

**Benchmarks.** Buy & hold, a fixed 80% SPY allocation, and cash only. Forecast baselines are calibrated VIX, raw VIX, 21-day realized volatility and the training mean.

**Evaluation design.** Test-period results were not used to adjust any part of the model. Hyperparameters were chosen on validation folds, and the target volatility, VIX calibration and fixed-exposure level come from training data.

## Limitations

- One asset and one test period of about 4.75 years. Daily 5-day windows overlap, so there are fewer independent observations than rows. The results are weak evidence either way.
- The 5-day volatility target is noisy.
- The final models are not retrained during the test period, and seven of the eight forests were fitted on windows that end before 2021.
- The position is decided at a close and applied from that same close, which assumes trading at the price the signal was computed from. There is no slippage or tax, and the first day is assumed fully invested.
- Returns are nominal (not adjusted for inflation).
- Standard volatility benchmarks such as EWMA or HAR-RV are not included.
- This is a test of a volatility forecast, not evidence of a tradable edge.

## Run it

```bash
pip install pandas numpy scikit-learn yfinance
python Code.py
```

The first run downloads the data and saves it to `market_data.csv` next to the script, which ends on 9 October 2026. To reproduce the numbers above, use that same file: a fresh download can differ slightly, because Yahoo Finance adjusts historical prices. The script prints the strategy table and the forecast table.

## Development notes

I wrote the code myself. I used AI (Claude) for debugging and learned from its suggestions while building the project on my own. The `print_results` function and the code comments were written by the AI. The methodology was developed with the assistance of Professor Christopher Volinsky of NYU Stern.
