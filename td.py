import yfinance as yf
import pandas as pd

# Define tickers
tickers = {"Gold_ETF": "TATAGOLD.NS", "Silver_FoF": "TATSILV.NS"}

for name, ticker in tickers.items():
    print(f"Fetching and cleaning data for {name}...")
    
    # yf.Ticker object allows extraction of basic info metadata fields
    fund = yf.Ticker(ticker)
    info = fund.info
    
    # 1. Download all available historical time-series data
    df = fund.history(period="max")
    
    # 2. Data Cleaning Process
    df.dropna(how="all", inplace=True) # Drop completely empty rows
    df.ffill(inplace=True) # Forward-fill any accidental missing pricing ticks
    
    # 3. Add static metadata fields as columns for a flat file database structure
    df['Ticker'] = ticker
    df['Fund_Name'] = info.get('longName', name)
    df['Category'] = info.get('category', 'Commodity ETF')
    df['Total_Assets_Metadata'] = info.get('totalAssets', 'N/A')
    
    # Reset index to turn Date into a visible spreadsheet column
    df.reset_index(inplace=True)
    
    # 4. Export clean CSV
    file_name = f"cleaned_{name}_fundamentals.csv"
    df.to_csv(file_name, index=False)
    print(f"Successfully saved {len(df)} rows of clean data to {file_name}\n")
