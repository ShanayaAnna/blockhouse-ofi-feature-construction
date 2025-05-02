import pandas as pd

def preprocess(filepath: str) -> pd.DataFrame:
    """
    Loads and preprocesses LOB data:
    - Parses timestamps
    - Sorts by event time
    - Groups by symbol
    """
    df = pd.read_csv(filepath)

    # Parse timestamps
    df["ts_event"] = pd.to_datetime(df["ts_event"])
    df.sort_values(["symbol", "ts_event"], inplace=True) #Dataset contains only 1 symbol (cross-asset?)

    return df

def compute_best_level_ofi(grouped_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the BestLevel OFI per symbol(only AAPL as of now) per event.
    This is using Eq(1)

    The core idea is: we detect whether there's net buying or selling pressure at the top of the order book.
    """

    df = grouped_df.copy()
    df["best_level_ofi"] = 0.0

    for idx in range(1, len(df)):
        curr = df.iloc[idx]
        prev = df.iloc[idx - 1]

        # Bid Logic 
        if curr["bid_px_00"] > prev["bid_px_00"]:
            bid_ofi = curr["bid_sz_00"]
        elif curr["bid_px_00"] == prev["bid_px_00"]:
            bid_ofi = curr["bid_sz_00"] - prev["bid_sz_00"]
        else:
            bid_ofi = -prev["bid_sz_00"]

        # Ask Logic
        if curr["ask_px_00"] > prev["ask_px_00"]:
            ask_ofi = -prev["ask_sz_00"]
        elif curr["ask_px_00"] == prev["ask_px_00"]:
            ask_ofi = prev["ask_sz_00"] - curr["ask_sz_00"]
        else:
            ask_ofi = curr["ask_sz_00"]

        # Best Level Ofi = is bidOFI - askOFI 
        df.at[df.index[idx], "best_level_ofi"] = bid_ofi - ask_ofi

    return df


if __name__ == "__main__":
    df = preprocess("data/first_25000_rows.csv")
    print("Data shape:", df.shape)
    print("Symbols in dataset:", df["symbol"].unique())
    
    results = []

    # Group by symbol so we can handle each stock independently
    for symbol, symbol_df in df.groupby("symbol"):
        print(f"Computing Best-Level OFI for symbol: {symbol}")
        symbol_best_level_ofi = compute_best_level_ofi(symbol_df)
        print(symbol_best_level_ofi.head(10))
        results.append(symbol_best_level_ofi)

    df_with_best_level_ofi = pd.concat(results, ignore_index=True)



