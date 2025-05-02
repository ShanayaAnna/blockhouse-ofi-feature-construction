import pandas as pd

def preprocess(filepath: str) -> pd.DataFrame:
    """
    Loads and preprocesses LOB data:
    - Parses timestamps
    - Sorts by event time
    - Groups by symbol (for downstream logic)
    """
    df = pd.read_csv(filepath)

    # Parse timestamps
    df["ts_event"] = pd.to_datetime(df["ts_event"])
    df.sort_values(["ts_event"], inplace=True)

    return df

if __name__ == "__main__":
    df = preprocess("data/first_25000_rows.csv")
    print("Data shape:", df.shape)
    print("Symbols in dataset:", df["symbol"].unique())
    print(df[["ts_event", "symbol"]].head())

