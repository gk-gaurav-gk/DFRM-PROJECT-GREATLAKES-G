# DFRM-PROJECT-GREATLAKES-G
This project constructs risk-optimized 10-stock equity portfolio using historical NIFTY 500 data (2009–2024) evaluates its performance against the NIFTY 500 index, computes advanced risk metrics (VaR, volatility, CVaR), estimates implied volatility using India VIX and implements volatility-based options hedging framework to cap weekly losses at 5%.
#  Optimum Portfolio Construction & Risk Management (NIFTY 500)

##  Project Overview

This project builds, evaluates, and risk-manages an **optimal equity portfolio** using **NIFTY 500 stocks**.  
It follows a **fully data-driven and reproducible approach**, covering:

- Portfolio construction using **historical volatility**
- Risk-adjusted stock selection
- Minimum Variance Portfolio (MVP) optimization
- Performance tracking vs **NIFTY 500**
- Risk measurement using **VaR, CVaR, volatility**
- **Implied volatility estimation** using India VIX
- **Options-based risk management strategy** to cap weekly losses at 5%

The project is designed to meet **academic finance standards** while remaining **practically realistic**.

---

## 🧠 Methodology Summary

### 1️⃣ Data Collection
- **Universe**: NIFTY 500 stocks
- **Source**: Yahoo Finance (`yfinance`)
- **Technique**: Chrome impersonation (`curl_cffi`) to avoid request blocking
- **Data Used**:
  - Daily closing prices
  - Period: **Jan 2009 – Jan 2024** (portfolio construction)
  - **Feb 2024 – Dec 2024** (performance tracking)

---

### 2️⃣ Risk Metrics Computed

Each stock is evaluated using multiple risk dimensions:

| Metric | Purpose |
|------|--------|
| Rolling Beta (60-day) | Market sensitivity |
| Annualized Volatility | Total risk |
| Downside Volatility | Loss-focused risk |
| Maximum Drawdown | Worst historical loss |
| 99% CVaR | Tail risk severity |
| Average Correlation | Diversification benefit |

All metrics are **normalized** to ensure comparability.

---

### 3️⃣ Composite Risk & Selection Score

- **Composite Risk Score**: Weighted combination of normalized risk metrics
- **Return Score**: Annualized historical return
- **Selection Score**:


This avoids return-chasing and ensures **risk-adjusted selection**.

---

### 4️⃣ Stock Selection Logic

Stocks are divided into **three beta regimes**:

- **Low Beta** (< 0.8)
- **Mid Beta** (0.8 – 1.2)
- **High Beta** (> 1.2)

Top 10 stocks from each bucket are selected → **30 candidates**, ensuring diversification across risk regimes.

---

### 5️⃣ Portfolio Optimization

- **Objective**: Minimum Variance Portfolio (MVP)
- **Constraints**:
  - Long-only
  - Fully invested
  - Minimum 2% weight per stock
- **Method**:
  - Permutation-based subset search
  - Constrained optimization (SLSQP)

📌 **Final Portfolio**: 10 stocks, ₹10 crore exposure

---

### 6️⃣ Performance Tracking (Feb–Dec 2024)

- Portfolio tracked **daily**
- Compared against **NIFTY 500 (^CRSLDX)**
- Returns measured on **percentage basis**

📈 **Results**:
- Portfolio Return: **19.16%**
- NIFTY 500 Return: **12.54%**
- Portfolio outperformed with **lower volatility**

---

### 7️⃣ Risk Analysis

#### 🔹 Value at Risk (99%)
- Historical VaR: **1.91%**
- Parametric VaR: **1.94%**

#### 🔹 Historical Volatility
- Daily Volatility: **0.87%**
- Annualized Volatility: **13.81%**
- Lower than NIFTY 500 volatility (**15.23%**)

---

### 8️⃣ Implied Volatility (March 2025 Reference)

- **Proxy Used**: India VIX (^INDIAVIX)
- Average India VIX (Feb–Dec 2024): **14.69%**
- Portfolio Implied Volatility (scaled): **13.32%**

📌 NSE option chain history is not publicly available, so **India VIX** is used as a forward-looking implied volatility proxy.

---

### 9️⃣ Risk Management Strategy (Core Highlight)

🎯 **Objective**: Ensure portfolio never loses more than **5% in any given week**

#### Strategy Type:
**Volatility-Regime-Based Options Hedging**

| India VIX Level | Strategy |
|----------------|----------|
| < 12 | No hedge |
| 12 – 16 | Light OTM Protective Put |
| 16 – 20 | Full OTM Protective Put |
| > 20 | Collar Strategy (Put + Call Write) |

- Hedge instrument: **NIFTY index options**
- Rationale: Index options are liquid, cheap, and effective
- Strategy is **rules-based**, not discretionary

---

## ▶️ How to Run the Project

### Step 1: Install Dependencies
```bash
pip install pandas numpy yfinance matplotlib scipy curl-cffi
Step 2: Run Portfolio Construction
python portfolio_construction.py
This generates:
Risk scores
Final optimized portfolio
CSV outputs

Step 3: Run Performance & Risk Analysis
python portfolio_tracking_and_risk.py
This:
Tracks performance
Computes VaR & volatility
Estimates implied volatility
Outputs hedge recommendation
