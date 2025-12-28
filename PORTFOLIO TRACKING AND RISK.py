# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 12:22:28 2025

@author: Gaurav
"""


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
from curl_cffi import requests

# ============================================================
# 1. SETTINGS
# ============================================================

PORTFOLIO_FILE = r"C:/Users/Gaurav/OneDrive/Desktop/SAPM all/DFRM/final_10_stock_mvp_portfolio.csv"

START_DATE = dt.datetime(2024, 2, 1)
END_DATE   = dt.datetime(2024, 12, 31)

TOTAL_CAPITAL = 10_00_00_000   # ₹10 crore (used only for value curve)
BENCHMARK = "^CRSLDX"          # NIFTY 500
INTERVAL = "1d"

# ============================================================
# 2. LOAD FINAL PORTFOLIO
# ============================================================

portfolio_df = pd.read_csv(PORTFOLIO_FILE)

portfolio_df = portfolio_df.dropna(subset=["Stock", "Weight"])
portfolio_df = portfolio_df[portfolio_df["Weight"] > 0].reset_index(drop=True)

assert portfolio_df.shape[0] == 10, "Portfolio must have exactly 10 stocks"

stocks = portfolio_df["Stock"].tolist()
weights = portfolio_df.set_index("Stock")["Weight"]

print("✅ Portfolio loaded")
print(portfolio_df[["Stock", "Weight"]])

# ============================================================
# 3. DOWNLOAD PRICE DATA (CHROME IMPERSONATION)
# ============================================================

session = requests.Session(impersonate="chrome")
prices = {}

for ticker in stocks + [BENCHMARK]:
    print(f"Downloading {ticker}")
    hist = yf.Ticker(ticker, session=session).history(
        start=START_DATE,
        end=END_DATE + dt.timedelta(days=1),
        interval=INTERVAL
    )

    if hist.empty:
        raise ValueError(f"No price data for {ticker}")

    prices[ticker] = hist["Close"]
    time.sleep(0.1)

prices_df = pd.DataFrame(prices).dropna(how="all")

print("✅ Price data downloaded")

# ============================================================
# 4. DAILY RETURNS
# ============================================================

stock_returns = prices_df[stocks].pct_change(fill_method=None)
benchmark_returns = prices_df[BENCHMARK].pct_change(fill_method=None)

# Portfolio daily returns (WEIGHTED % RETURNS)
portfolio_returns = stock_returns @ weights

# ============================================================
# 5. CUMULATIVE % PERFORMANCE
# ============================================================

portfolio_cum = (1 + portfolio_returns).cumprod() - 1
benchmark_cum = (1 + benchmark_returns).cumprod() - 1

# ============================================================
# 6. PERFORMANCE SUMMARY
# ============================================================

print("\n📊 PERFORMANCE SUMMARY (1 Feb – 31 Dec 2024)")
print(f"Portfolio Return : {portfolio_cum.iloc[-1]:.2%}")
print(f"NIFTY 500 Return : {benchmark_cum.iloc[-1]:.2%}")

comparison_df = pd.DataFrame({
    "Portfolio_%_Change": portfolio_cum,
    "NIFTY500_%_Change": benchmark_cum
})

comparison_df.to_csv("portfolio_vs_nifty500_percentage_comparison.csv")

# ============================================================
# 7. CUMULATIVE % GROWTH PLOT
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(portfolio_cum.index, portfolio_cum * 100, label="Portfolio", linewidth=2)
plt.plot(benchmark_cum.index, benchmark_cum * 100, label="NIFTY 500", linestyle="--")

plt.title("Cumulative % Growth: Portfolio vs NIFTY 500")
plt.ylabel("Cumulative Return (%)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 8. PORTFOLIO VALUE GROWTH (₹)
# ============================================================

portfolio_value = TOTAL_CAPITAL * (1 + portfolio_returns).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(portfolio_value.index, portfolio_value / 1e7, linewidth=2)
plt.title("Portfolio Value Growth (₹ Crore)")
plt.ylabel("Portfolio Value (₹ Crore)")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 9. DAILY PORTFOLIO RETURNS
# ============================================================

plt.figure(figsize=(12, 4))
plt.plot(portfolio_returns.index, portfolio_returns * 100)
plt.title("Daily Portfolio Returns (%)")
plt.ylabel("Daily Return (%)")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 10. 99% DAILY VaR (HISTORICAL & PARAMETRIC)
# ============================================================

# Historical VaR
var_99_hist = -np.percentile(portfolio_returns.dropna(), 1)

# Parametric VaR
mu_p = portfolio_returns.mean()
sigma_p = portfolio_returns.std()
z_99 = norm.ppf(0.99)

var_99_param = -(mu_p - z_99 * sigma_p)

print(f"\n99% Historical Daily VaR : {var_99_hist:.2%}")
print(f"99% Parametric Daily VaR : {var_99_param:.2%}")

# ============================================================
# 11. VaR BREACH VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    portfolio_returns.index,
    portfolio_returns * 100,
    label="Portfolio Returns",
    alpha=0.7
)

plt.axhline(
    -var_99_hist * 100,
    color="red",
    linestyle="--",
    label="99% Historical VaR"
)

plt.axhline(
    -var_99_param * 100,
    color="black",
    linestyle=":",
    label="99% Parametric VaR"
)

plt.title("Daily Portfolio Returns with 99% VaR")
plt.ylabel("Return (%)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("✅ Comparison file saved: portfolio_vs_nifty500_percentage_comparison.csv")

# ============================================================
# HISTORICAL VOLATILITY (FEB 2024 – DEC 2024)
# ============================================================

daily_volatility = portfolio_returns.std()
annualized_volatility = daily_volatility * np.sqrt(252)

print("\n📉 HISTORICAL VOLATILITY (Feb–Dec 2024)")
print(f"Daily Volatility       : {daily_volatility:.4%}")
print(f"Annualized Volatility  : {annualized_volatility:.2%}")

# ============================================================
# ROLLING HISTORICAL VOLATILITY (21-DAY)
# ============================================================

rolling_vol_21d = portfolio_returns.rolling(21).std() * np.sqrt(252)

plt.figure(figsize=(12, 5))
plt.plot(rolling_vol_21d.index, rolling_vol_21d * 100, linewidth=2)

plt.title("Rolling 21-Day Annualized Historical Volatility (%)")
plt.ylabel("Volatility (%)")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.show()

benchmark_vol = benchmark_returns.std() * np.sqrt(252)

print("\n📊 VOLATILITY COMPARISON")
print(f"Portfolio Volatility : {annualized_volatility:.2%}")
print(f"NIFTY 500 Volatility : {benchmark_vol:.2%}")

benchmark_rolling_vol = benchmark_returns.rolling(21).std() * np.sqrt(252)

plt.figure(figsize=(12, 5))
plt.plot(rolling_vol_21d.index, rolling_vol_21d * 100, label="Portfolio", linewidth=2)
plt.plot(benchmark_rolling_vol.index, benchmark_rolling_vol * 100, 
         linestyle="--", label="NIFTY 500")

plt.title("Rolling 21-Day Historical Volatility Comparison")
plt.ylabel("Volatility (%)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 12. IMPLIED VOLATILITY USING INDIA VIX (NO UNDEFINED VARS)
# ============================================================

# Use SAME chrome impersonation session
vix = yf.Ticker("^INDIAVIX", session=session).history(
    start=START_DATE,
    end=END_DATE + dt.timedelta(days=1),
    auto_adjust=True
)["Close"].dropna()

# Average India VIX (convert % to decimal)
india_vix_avg = vix.mean() / 100

print("\n📈 IMPLIED VOLATILITY (VIX-BASED)")
print(f"Average India VIX (Feb–Dec 2024): {india_vix_avg:.2%}")

# ============================================================
# SCALE VIX TO PORTFOLIO (VOLATILITY RATIO METHOD)
# ============================================================
# annualized_volatility  -> Portfolio historical vol (already computed)
# benchmark_vol          -> NIFTY 500 historical vol (already computed)

portfolio_implied_vol = india_vix_avg * (
    annualized_volatility / benchmark_vol
)

print(
    f"Portfolio Implied Volatility "
    f"(Options expiring Mar 2025 reference): "
    f"{portfolio_implied_vol:.2%}"
)
# plot vix 
plt.figure(figsize=(12, 5))
plt.plot(vix.index, vix, label="India VIX", linewidth=2)
plt.axhline(vix.mean(), linestyle="--", color="red", label="Average VIX")

plt.title("India VIX – Implied Volatility (Feb–Dec 2024)")
plt.ylabel("Implied Volatility (%)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# SETTINGS
# ==============================

PORTFOLIO_VALUE = 10_00_00_000   # ₹10 crore
PORTFOLIO_BETA = 1.0
NIFTY_LOT_SIZE = 50

session = requests.Session(impersonate="chrome")

# ==============================
# 1. FETCH REAL-TIME DATA
# ==============================

nifty = yf.Ticker("^NSEI", session=session).history(period="1d")
vix = yf.Ticker("^INDIAVIX", session=session).history(period="1d")

nifty_spot = float(nifty["Close"].iloc[-1])
india_vix = float(vix["Close"].iloc[-1])

print(f"NIFTY Spot Price : {nifty_spot:.2f}")
print(f"India VIX        : {india_vix:.2f}")

# ============================================================
# OPTION-BASED RISK MANAGEMENT DECISION ENGINE
# ============================================================

# Inputs already available from earlier code
# nifty_spot  -> current NIFTY spot
# india_vix   -> current India VIX

strike_rounding = 50  # NIFTY strikes rounded to nearest 50

def round_strike(x):
    return int(round(x / strike_rounding) * strike_rounding)

print("\n" + "="*70)
print("🔐 RISK MANAGEMENT STRATEGY DECISION (OPTIONS)")
print("="*70)

print(f"NIFTY Spot Price : {nifty_spot:.2f}")
print(f"India VIX        : {india_vix:.2f}\n")

# ============================================================
# SCENARIO 1: VERY LOW VOLATILITY (NO HEDGE)
# ============================================================

if india_vix < 12:
    print("🟢 SCENARIO: VERY LOW VOLATILITY")
    print("Market Interpretation:")
    print("- Calm market")
    print("- Low probability of sharp weekly drawdown")
    print("- Hedging cost is inefficient\n")

    print("Recommended Strategy:")
    print("✔ NO OPTIONS HEDGE")
    print("✔ Stay fully invested in equity portfolio\n")

    print("Reasoning:")
    print("- India VIX < 12 historically indicates stability")
    print("- Buying puts would lead to premium decay losses\n")

    print("Exit / Review Condition:")
    print("- Re-evaluate hedge if VIX rises above 12")

# ============================================================
# SCENARIO 2: LOW–MODERATE VOLATILITY (LIGHT PROTECTION)
# ============================================================

elif 12 <= india_vix < 16:
    put_strike = round_strike(nifty_spot * 0.97)

    print("🟡 SCENARIO: LOW–MODERATE VOLATILITY")
    print("Market Interpretation:")
    print("- Mild uncertainty")
    print("- Portfolio downside risk increasing\n")

    print("Recommended Strategy: LIGHT PROTECTIVE PUT")

    print("Action:")
    print(f"✔ BUY NIFTY {put_strike} PUT (Out-of-the-Money ~3%)")
    print("✖ Do NOT sell any calls\n")

    print("Why this works:")
    print("- Limits downside beyond ~3% weekly fall")
    print("- Cheap insurance due to moderate VIX")
    print("- Retains full upside participation\n")

    print("When to Exit:")
    print("- If VIX falls below 12 → exit hedge")
    print("- If VIX rises above 16 → upgrade hedge")

# ============================================================
# SCENARIO 3: HIGH VOLATILITY (FULL PROTECTION)
# ============================================================

elif 16 <= india_vix < 20:
    put_strike = round_strike(nifty_spot * 0.95)

    print("🟠 SCENARIO: HIGH VOLATILITY")
    print("Market Interpretation:")
    print("- Elevated risk of sharp weekly losses")
    print("- Drawdowns likely\n")

    print("Recommended Strategy: FULL PROTECTIVE PUT")

    print("Action:")
    print(f"✔ BUY NIFTY {put_strike} PUT (Out-of-the-Money ~5%)")
    print("✖ Avoid selling calls (preserve convexity)\n")

    print("Why this works:")
    print("- Caps weekly loss close to 5%")
    print("- Aligns with portfolio risk constraint")
    print("- High VIX justifies insurance cost\n")

    print("When to Exit:")
    print("- If VIX drops below 16")
    print("- Or after volatility event passes")

# ============================================================
# SCENARIO 4: EXTREME VOLATILITY (AGGRESSIVE HEDGE)
# ============================================================

else:
    atm_put = round_strike(nifty_spot)
    otm_call = round_strike(nifty_spot * 1.03)

    print("🔴 SCENARIO: EXTREME VOLATILITY / MARKET STRESS")
    print("Market Interpretation:")
    print("- Panic conditions")
    print("- High probability of >5% weekly move\n")

    print("Recommended Strategy: COLLAR STRATEGY")

    print("Action:")
    print(f"✔ BUY NIFTY {atm_put} PUT (ATM Protection)")
    print(f"✔ SELL NIFTY {otm_call} CALL (3% OTM)")
    print("✔ Net hedge cost reduced via call writing\n")

    print("Why this works:")
    print("- Guarantees strict downside protection")
    print("- Call premium offsets expensive put")
    print("- Suitable only in crisis regimes\n")

    print("Trade-off:")
    print("- Upside capped beyond ~3%")
    print("- Acceptable during stress periods\n")

    print("When to Exit:")
    print("- Unwind collar when VIX < 20")
    print("- Shift back to protective put")

print("\n" + "="*70)
print("📌 NOTE:")
print("- Strategy is volatility-driven, not prediction-based")
print("- Reviewed weekly or on major VIX movement")
print("- Designed to cap weekly loss at ~5%")
print("="*70)
