# ============================================================
# DATA → RISK METRICS → COMPOSITE SCORE (FINAL STABLE VERSION)
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import time
import os
from curl_cffi import requests

# ============================================================
# 1. SETTINGS
# ============================================================

os.chdir(r"C:/Users/Gaurav/OneDrive/Desktop/SAPM all/DFRM")

SYMBOL_FILE = "NIFTY_500s.csv"   # must contain 'Symbol'
INTERVAL = "1d"                 # "1d" or "1wk"
ROLLING_WINDOW = 60

START_DATE = dt.datetime(2009, 1, 1)
END_DATE   = dt.datetime(2024, 1, 31)

ANNUAL_FACTOR = 252 if INTERVAL == "1d" else 52

# ============================================================
# 2. LOAD SYMBOLS
# ============================================================

symbols_df = pd.read_csv(SYMBOL_FILE)

symbols = (
    symbols_df["Symbol"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

tickers = [s + ".NS" for s in symbols]
tickers.append("^NSEI")   # market index

# ============================================================
# 3. FETCH DATA (CLOSING PRICES ONLY)
# ============================================================

session = requests.Session(impersonate="chrome")
all_prices = {}

BATCH_SIZE = 10

for i in range(0, len(tickers), BATCH_SIZE):
    batch = tickers[i:i + BATCH_SIZE]
    print(f"Fetching batch {i // BATCH_SIZE + 1}")

    for ticker in batch:
        try:
            hist = yf.Ticker(ticker, session=session).history(
                start=START_DATE,
                end=END_DATE + dt.timedelta(days=1),
                interval=INTERVAL
            )

            if "Close" in hist.columns and hist["Close"].notna().sum() > 5:
                all_prices[ticker] = hist["Close"]

        except Exception as e:
            print(f"Failed for {ticker}: {e}")

        time.sleep(0.1)

# ============================================================
# 3A. SAVE RAW DOWNLOADED PRICES (UNTOUCHED)
# ============================================================

raw_prices = pd.DataFrame(all_prices).sort_index()

raw_prices.to_csv(
    "nifty500_raw_prices_unadjusted.csv"
)

print("✅ Raw downloaded prices saved")

# ============================================================
# 4. PRICE MATRIX (CLEANED DATA USED IN MODEL)
# ============================================================

prices = raw_prices.copy()

# 🔥 DROP stocks with NO data at all
prices = prices.dropna(axis=1, how="all")

# Forward-fill ONLY after first valid price
prices = prices.ffill()

# Separate market index
nifty = prices["^NSEI"]
prices = prices.drop(columns="^NSEI")

# ============================================================
# 4A. SAVE CLEANED PRICE MATRIX
# ============================================================

prices.to_csv(
    "nifty500_prices_cleaned_ffill.csv"
)

print("✅ Cleaned price matrix saved")

print("Stocks with usable data:", prices.shape[1])
print("Date range:", prices.index.min(), "to", prices.index.max())

# ============================================================
# 5. RETURNS (FROM WHERE DATA EXISTS)
# ============================================================

returns = prices.pct_change(fill_method=None)
market_returns = nifty.pct_change(fill_method=None)

# ============================================================
# 6. ROLLING BETA → AVERAGE BETA (CORRECT PER STOCK)
# ============================================================

market_var = market_returns.rolling(ROLLING_WINDOW).var()

rolling_beta = pd.DataFrame(index=returns.index, columns=returns.columns)

for stock in returns.columns:
    rolling_beta[stock] = (
        returns[stock]
        .rolling(ROLLING_WINDOW)
        .cov(market_returns)
        / market_var
    )

avg_beta = rolling_beta.mean(axis=0)

# ============================================================
# 7. CORE RISK METRICS
# ============================================================

volatility = returns.std() * np.sqrt(ANNUAL_FACTOR)
downside_vol = returns.where(returns < 0).std() * np.sqrt(ANNUAL_FACTOR)

cum_returns = (1 + returns).cumprod()
drawdown = cum_returns / cum_returns.cummax() - 1
max_drawdown = drawdown.min()

var_99 = returns.quantile(0.01)
cvar_99 = returns[returns.le(var_99)].mean()

avg_corr = returns.corr().mean()

# ============================================================
# 8. NORMALIZATION FUNCTION
# ============================================================

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

beta_n = normalize(avg_beta)
vol_n = normalize(volatility)
down_n = normalize(downside_vol)
mdd_n = normalize(abs(max_drawdown))
cvar_n = normalize(abs(cvar_99))
corr_n = normalize(avg_corr)

# ============================================================
# 9. COMPOSITE RISK SCORE (RISK ONLY)
# ============================================================

risk_score = (
    0.20 * beta_n +
    0.15 * vol_n +
    0.20 * down_n +
    0.20 * mdd_n +
    0.15 * cvar_n +
    0.10 * corr_n
)

# ============================================================
# 9A. RETURNS FOR SELECTION (SEPARATE)
# ============================================================

annual_return = returns.mean() * ANNUAL_FACTOR
return_score = normalize(annual_return)

selection_score = return_score / (risk_score + 1e-6)

# ============================================================
# 10. FINAL OUTPUT TABLE
# ============================================================

final_df = pd.DataFrame({
    "Stock": avg_beta.index.astype(str),

    "Avg_Beta": avg_beta.values,
    "Volatility": volatility.reindex(avg_beta.index).values,
    "Downside_Vol": downside_vol.reindex(avg_beta.index).values,
    "Max_Drawdown": max_drawdown.reindex(avg_beta.index).values,
    "CVaR_99": cvar_99.reindex(avg_beta.index).values,
    "Avg_Correlation": avg_corr.reindex(avg_beta.index).values,
    "Composite_Risk_Score": risk_score.reindex(avg_beta.index).values,

    "Annual_Return": annual_return.reindex(avg_beta.index).values,
    "Return_Score": return_score.reindex(avg_beta.index).values,
    "Selection_Score": selection_score.reindex(avg_beta.index).values
})

# ============================================================
# 11. SAVE FINAL OUTPUT
# ============================================================

final_df.to_csv(
    "nifty500_composite_risk_output.csv",
    index=False
)

print("✅ Final risk & selection output saved")

print(final_df.head())
print(final_df.columns)


# ============================================================
# 12. BETA CATEGORIES
# ============================================================

def beta_bucket(beta):
    if beta < 0.8:
        return "Low_Beta"
    elif 0.8 <= beta <= 1.2:
        return "Mid_Beta"
    else:
        return "High_Beta"

final_df["Beta_Category"] = final_df["Avg_Beta"].apply(beta_bucket)

# ============================================================
# 13. TOP 10 STOCKS PER BETA CATEGORY
# ============================================================

top10_low_beta = (
    final_df[final_df["Beta_Category"] == "Low_Beta"]
    .sort_values("Selection_Score", ascending=False)
    .head(10)
)

top10_mid_beta = (
    final_df[final_df["Beta_Category"] == "Mid_Beta"]
    .sort_values("Selection_Score", ascending=False)
    .head(10)
)

top10_high_beta = (
    final_df[final_df["Beta_Category"] == "High_Beta"]
    .sort_values("Selection_Score", ascending=False)
    .head(10)
)


# ============================================================
# 14. COMBINED SELECTION UNIVERSE
# ============================================================

selected_stocks_df = pd.concat(
    [top10_low_beta, top10_mid_beta, top10_high_beta],
    axis=0
).reset_index(drop=True)

print("Selected stocks for optimization:", selected_stocks_df.shape[0])
print(selected_stocks_df[["Stock", "Avg_Beta", "Beta_Category", "Selection_Score"]])

# ============================================================
# 15. RETURNS FOR SELECTED STOCKS
# ============================================================

selected_symbols = selected_stocks_df["Stock"].tolist()

selected_returns = returns[selected_symbols].dropna(how="any")

# ============================================================
# 16. COVARIANCE MATRIX
# ============================================================

cov_matrix = selected_returns.cov() * ANNUAL_FACTOR

# ============================================================
# 17. PERMUTATION SEARCH FOR BEST 10-STOCK LONG-ONLY MVP
# ============================================================

import random
from scipy.optimize import minimize

NUM_TRIALS = 10000   # can increase to 10000 if system allows

best_variance = np.inf
best_subset = None
best_weights = None

candidate_stocks = selected_symbols

# Keep dates where at least 80% stocks have data (robust covariance)
candidate_returns = returns[candidate_stocks].dropna(
    thresh=int(0.8 * len(candidate_stocks))
)

def long_only_mvp(cov_matrix):
    n = cov_matrix.shape[0]

    MIN_WEIGHT = 0.02
    MAX_WEIGHT = 1 - MIN_WEIGHT * (n - 1)

    def portfolio_variance(w):
        return w.T @ cov_matrix @ w

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    )

    bounds = [(MIN_WEIGHT, MAX_WEIGHT) for _ in range(n)]

    w0 = np.ones(n) / n

    result = minimize(
        portfolio_variance,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result.x if result.success else None



for _ in range(NUM_TRIALS):
    subset = random.sample(candidate_stocks, 10)

    sub_returns = candidate_returns[subset].dropna(how="any")
    if sub_returns.shape[0] < 60:   # ensure enough observations
        continue

    cov = sub_returns.cov() * ANNUAL_FACTOR

    try:
        weights = long_only_mvp(cov.values)
        if weights is None:
            continue

        portfolio_variance = weights.T @ cov.values @ weights

        if portfolio_variance < best_variance:
            best_variance = portfolio_variance
            best_subset = subset
            best_weights = weights

    except Exception:
        continue


# ============================================================
# 18. FINAL 10-STOCK MVP PORTFOLIO
# ============================================================

final_weights = pd.Series(
    best_weights,
    index=best_subset,
    name="Weight"
)

# ============================================================
# 19. CAPITAL ALLOCATION
# ============================================================

TOTAL_CAPITAL = 10_00_00_000  # ₹10 crore

final_portfolio = (
    final_df
    .set_index("Stock")
    .loc[best_subset]
    .join(final_weights)
)

final_portfolio["Investment_₹"] = final_portfolio["Weight"] * TOTAL_CAPITAL

final_portfolio = final_portfolio.sort_values("Weight", ascending=False)

print("\n✅ FINAL 10-STOCK MINIMUM VARIANCE PORTFOLIO")
print(final_portfolio[
    ["Avg_Beta", "Beta_Category", "Selection_Score", "Weight", "Investment_₹"]
])

# ============================================================
# 20. SAVE FINAL PORTFOLIO
# ============================================================

final_portfolio.to_csv("final_10_stock_mvp_portfolio.csv")
print("✅ Final 10-stock MVP saved")

final_portfolio["Weight"] = final_portfolio["Weight"] / final_portfolio["Weight"].sum()
final_portfolio["Weight"] = final_portfolio["Weight"].round(4)
final_portfolio["Investment_₹"] = final_portfolio["Investment_₹"].round(0)
