import yfinance as yf  # Downloads Bursa Malaysia stock prices and FBMKLCI benchmark data
import pandas as pd
import numpy as np
import os

# =========================================
# 1. STOCK LIST (CANDIDATE POOL)
# =========================================
# A diversified pool of Bursa Malaysia stocks including mix of sectors 
# (banking, utilities, tech, industrials, etc.) and different market 
# capitalisations (large, mid, small).
# This avoids selection bias and allows volatility classification to emerge from data.

stocks = [
    "1155.KL","1295.KL","1023.KL","5347.KL","5225.KL",
    "5819.KL","1961.KL","2445.KL","3816.KL","6012.KL",
    "4863.KL","6888.KL","1082.KL","5183.KL","6033.KL",
    "1818.KL","4707.KL","4197.KL","3182.KL","4715.KL",
    "1066.KL","2488.KL","4677.KL","6742.KL","1562.KL",
    "5296.KL","7113.KL","7106.KL","5168.KL","6947.KL",
    "0138.KL","5216.KL","7277.KL","7084.KL","7203.KL",
    "5285.KL","5681.KL","1619.KL","5211.KL","0166.KL"
]

# Benchmark index for beta calculation later
benchmark = "^KLSE"  # FTSE Bursa Malaysia KLCI

# =========================================
# 2. DOWNLOAD DATA
# =========================================
# We use 2 years of daily data to balance:
# - market relevance (recent conditions)
# - data consistency across all stocks
# Aligns with assignment requirement (minimum 2 years)

print("Downloading data...")
data = yf.download(stocks + [benchmark], period="2y", auto_adjust=True, progress=True)

# Extract closing prices (standard for return calculations)
prices = data["Close"]

# Separate benchmark and stock prices (important for beta calculation later)
benchmark_prices = prices[benchmark]
stock_prices = prices.drop(columns=[benchmark])

# =========================================
# 3. DATA CLEANING
# =========================================
# Allow up to 10% missing values to retain more assets
# This improves diversification while maintaining acceptable data quality
# Strict removal (0% missing) would reduce the investment universe too much

stock_prices = stock_prices.dropna(axis=1, thresh=len(stock_prices) * 0.9)

print(f"\nRemaining stocks after cleaning: {len(stock_prices.columns)}")
print(stock_prices.columns.tolist())

# =========================================
# 4. RETURNS CALCULATION
# =========================================
# Convert prices into percentage returns:
# r_t = (P_t - P_{t-1}) / P_{t-1}
# This standardises performance across different stock price levels

returns = stock_prices.pct_change().dropna()
benchmark_returns = benchmark_prices.pct_change().dropna()

# =========================================
# 5. DAILY STATISTICS
# =========================================
# Mean Return = expected daily return
# Volatility = standard deviation of daily returns (risk)

mean_returns = returns.mean()
volatility = returns.std()

stats_full = pd.DataFrame({
    "Mean Return": mean_returns,
    "Volatility": volatility,
    "Annual Return": mean_returns * 252,    # annualised return, 252 trading days
    "Annual Risk": volatility * np.sqrt(252)# annualised risk
})

# Remove zero volatility (avoid division errors)
stats_full = stats_full[stats_full["Volatility"] > 0]

# Return/Risk Ratio measures efficiency (return per unit of risk)
stats_full["Return/Risk Ratio"] = stats_full["Mean Return"] / stats_full["Volatility"]

# =========================================
# 6. LOW VOLATILITY STOCKS (STAGE 1)
# =========================================
# Income objective → select only stocks with positive returns
# Negative-return stocks are excluded as they contradict income generation goals

stats_positive = stats_full[stats_full["Mean Return"] > 0].copy()

print(f"\nStocks with positive returns: {len(stats_positive)}")
print(f"Stocks excluded (negative returns): {len(stats_full) - len(stats_positive)}")

# Select top 10 stocks with highest return-to-risk ratio
# This proxies stable and efficient income-generating stocks

# Step 1: Try assignment condition
low_vol_condition = stats_positive[
    stats_positive["Mean Return"] > stats_positive["Volatility"]
]

# Step 2: Apply logic
if len(low_vol_condition) >= 10:
    low_vol = low_vol_condition.head(10)
    method_used = "Assignment condition: Mean return > Volatility"
else:
    print("⚠️ No stocks met strict condition → using lowest volatility fallback")
    low_vol = stats_positive.nsmallest(10, "Volatility")
    method_used = "Fallback: Lowest volatility selection"

# =========================================
# 7. HIGH VOLATILITY STOCKS (STAGE 2)
# =========================================
# Use FULL universe (including negative returns)
# Mandate defines high volatility as:
# volatility > 3 × mean return

high_vol_candidates = stats_full[
    stats_full["Volatility"] > 3 * stats_full["Mean Return"]
]

print(f"\nStocks meeting high-vol condition: {len(high_vol_candidates)}")

# Select top 5 highest volatility stocks
high_vol = high_vol_candidates.sort_values(
    by="Volatility", ascending=False
).head(5)

# Fallback if insufficient stocks meet condition
if len(high_vol) < 5:
    print("⚠️ Not enough stocks met high-vol condition. Using highest volatility stocks instead.")
    high_vol = stats_full.sort_values(
        by="Volatility", ascending=False
    ).head(5)

# Flag whether each stock strictly meets condition
high_vol["Meets_Condition"] = high_vol["Volatility"] > 3 * high_vol["Mean Return"]

# =========================================
# 8. OUTPUT RESULTS
# =========================================
print("\n" + "="*70)
print("LOW VOLATILITY STOCKS (Stage 1)")
print("="*70)
print(low_vol[["Annual Return", "Annual Risk", "Return/Risk Ratio"]])

print("\n" + "="*70)
print("HIGH VOLATILITY STOCKS (Stage 2 Addition)")
print("="*70)
print(high_vol[["Annual Return", "Annual Risk", "Volatility", "Mean Return", "Meets_Condition"]])

# Highlight high-vol stocks with negative returns (important insight)
negative_high_vol = high_vol[high_vol["Mean Return"] < 0]
if len(negative_high_vol) > 0:
    print("\n⚠️ High-volatility stocks with NEGATIVE returns:")
    print(negative_high_vol.index.tolist())

# =========================================
# 9. SAVE DATA (REPRODUCIBILITY)
# =========================================
# This stores processed datasets for analysis and next stages

os.makedirs("output", exist_ok=True)

# Save candidate pool (IMPORTANT for assignment requirement)
pd.Series(stocks, name="Candidate Pool").to_csv("output/candidate_pool.csv")
print(f"Candidate pool size: {len(stocks)} stocks")

# Save processed datasets
stock_prices.to_csv("output/cleaned_prices.csv")
returns.to_csv("output/returns.csv")
stats_full.to_csv("output/stock_statistics.csv")

# Save selected stocks
low_vol.to_csv("output/low_vol_stocks.csv")
high_vol.to_csv("output/high_vol_stocks.csv")

# Combine selected stocks for next stage (portfolio optimisation)
selected_stocks = list(low_vol.index) + list(high_vol.index)
pd.Series(selected_stocks, name="Selected Stocks").to_csv("output/selected_stocks.csv")

print("\n✅ Core outputs saved")


# =========================================
# 10. SAVE RAW DATA (RUN ONCE ONLY)
# =========================================
# This section exports individual stock dataset

SAVE_DATA = False  # Change to True ONLY when exporting

if SAVE_DATA:

    print("Saving raw datasets...")

    os.makedirs("raw_data", exist_ok=True)

    for ticker in stock_prices.columns:
        df = stock_prices[[ticker]].dropna()
        df.to_csv(f"raw_data/{ticker}.csv")

    benchmark_prices.to_csv("raw_data/fbmklci.csv")

    print("✅ Raw data saved")

else:
    print("ℹ️ Raw data export skipped")