import yfinance as yf
import numpy as np
import math as m
import datetime as datetime
import pandas as pd
from scipy.stats import norm

#Function for Black-Scholes Formula (Euro Option)
def black_scholes_call(S,K,T,r,sigma):
    """
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to maturity (in years)
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying asset (annualized)

    Returns:
    float: Call Option Price 
    """

# Get stock price BHP on 16 May 2025
ticker = "BHP"
stock = yf.Ticker(ticker)
#16 May Historical Data
target_date = "2025-05-16"
hist = stock.history(start="2025-05-16", end="2025-05-17")

#Closing price extract
if hist.empty:
    raise ValueError(f"No stock data found for {ticker} on {target_date}.")
S = hist["Close"].iloc[0]  # Closing price on 16 May 2025

# Calculate K 
K = 0.98*S  

# Calculate T 
Valuation_Date = datetime.datetime(2025, 5, 16)
Expiration_Date = datetime.datetime(2025, 9, 15)
T = (Expiration_Date - Valuation_Date).days / 365

# Calculate R 
# Loading RBA Daily Gov yield curve from Excel 
# Read the Excel file
df = pd.read_excel('Risk Free - Yield Curve.xlsx')  # You can add sheet_name='Sheet1' if needed

# Show the first 5 rows
print(df.head())

# Calculating call price 
d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
call_price


