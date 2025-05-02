import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

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

def compute_multi_level_ofi(grouped_df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    """
    Computes Multi Level OFI (The dataset conatins 10 levels 0-9).
    Uses the same logic from Best-Level OFI across bid/ask prices and sizes.

    """
    df = grouped_df.copy()

    # Initializing all level-wise OFI columns
    for level in range(levels):
        df[f"ofi_level_{level}"] = 0.0

    for idx in range(1, len(df)):
        curr = df.iloc[idx]
        prev = df.iloc[idx - 1]

        for level in range(levels):
            # Build column names for this level
            bid_px_col = f"bid_px_0{level}"
            bid_sz_col = f"bid_sz_0{level}"
            ask_px_col = f"ask_px_0{level}"
            ask_sz_col = f"ask_sz_0{level}"

            # Bid Logic
            if curr[bid_px_col] > prev[bid_px_col]:
                bid_ofi = curr[bid_sz_col]
            elif curr[bid_px_col] == prev[bid_px_col]:
                bid_ofi = curr[bid_sz_col] - prev[bid_sz_col]
            else:
                bid_ofi = -prev[bid_sz_col]

            # Ask Logic
            if curr[ask_px_col] > prev[ask_px_col]:
                ask_ofi = -prev[ask_sz_col]
            elif curr[ask_px_col] == prev[ask_px_col]:
                ask_ofi = prev[ask_sz_col] - curr[ask_sz_col]
            else:
                ask_ofi = curr[ask_sz_col]

            level_ofi = bid_ofi - ask_ofi
            df.at[df.index[idx], f"ofi_level_{level}"] = level_ofi

    return df

def compute_integrated_ofi(multi_level_df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    """
    Computes the Integrated OFI using the first principal component of multi-level OFIs.
    Equation (4)
    """
    df = multi_level_df.copy()

    # Extracting the level-wise OFI columns
    ofi_matrix = df[[f"ofi_level_{i}" for i in range(levels)]].fillna(0).values

    # Fit PCA on the entire OFI matrix 
    pca = PCA(n_components=1)
    first_pc_scores = pca.fit_transform(ofi_matrix)  
    first_pc_weights = pca.components_[0]           

    # Normalize the weights
    l1_norm = np.sum(np.abs(first_pc_weights))
    normalized_weights = first_pc_weights / l1_norm

    df["integrated_ofi"] = ofi_matrix.dot(normalized_weights)

    return df

def simulate_other_symbols(original_df: pd.DataFrame, symbol_variants: list) -> pd.DataFrame:
    """
    Since the data set provided consists of only one symbol
    We are simulation additional symbols.
    This function adds small noise to bid/ask prices and sizes It is used only for testing cross-asset OFI logic.

    Note: We use MSFT and GOOG because they were part of the Nasdaq 100 dataset used in the paper
    """
    simulated_dfs = [original_df]

    for name in symbol_variants:
        sim_df = original_df.copy()
        sim_df["symbol"] = name

        # Adding small noise 
        for col in sim_df.columns:
            if "px" in col:
                sim_df[col] += np.random.normal(0.01, 0.005, size=len(sim_df))  # bump the prices slightly
            elif "sz" in col:
                sim_df[col] = sim_df[col] * (1 + np.random.normal(0.01, 0.005, size=len(sim_df)))  # adding jitter

        simulated_dfs.append(sim_df)

    combined = pd.concat(simulated_dfs, ignore_index=True)
    combined.sort_values(["symbol", "ts_event"], inplace=True)
    return combined

def compute_cross_asset_ofi(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    """
    Runs the full OFI feature extraction pipeline:
    - Best-Level OFI
    - Multi-Level OFI
    - Integrated OFI
    - Cross-Asset OFI (sum of other symbols' integrated OFI at same ts_event)
    Equation (CI[1])
    """
    full_results = []

    for symbol, symbol_df in df.groupby("symbol"):
        print(f"Running OFI pipeline for {symbol}")
        best_df = compute_best_level_ofi(symbol_df)
        multi_df = compute_multi_level_ofi(best_df, levels=levels)
        integrated_df = compute_integrated_ofi(multi_df, levels=levels)
        full_results.append(integrated_df)

    combined = pd.concat(full_results, ignore_index=True)

    # Cross-Asset OFI Logic: sum of other symbols' integrated OFI at same timestamp 
    # For each symbol, subtract their own contribution from total sum at that timestamp
    grouped = combined.groupby("ts_event")
    combined["cross_asset_ofi"] = combined.apply(
        lambda row: grouped["integrated_ofi"].get_group(row["ts_event"]).sum() - row["integrated_ofi"],
        axis=1
    )

    return combined

if __name__ == "__main__":
    df = preprocess("data/first_25000_rows.csv")
    print("Data shape:", df.shape)
    print("Symbols in dataset:", df["symbol"].unique())

    # Step 1–3: Best-Level, Multi-Level, Integrated OFI
    combined_results = []

    for symbol, symbol_df in df.groupby("symbol"):
        print(f"\nComputing OFI features for symbol: {symbol}")

        # 1: Best-Level
        best_df = compute_best_level_ofi(symbol_df)
        
        # 2: Multi-Level
        multi_df = compute_multi_level_ofi(best_df)
        
        # 3: Integrated OFI
        integrated_df = compute_integrated_ofi(multi_df)

        combined_results.append(integrated_df)

    final_ofi_df = pd.concat(combined_results, ignore_index=True)

    print("\n Base OFI Features (Best, Multi-Level, Integrated)")
    print(final_ofi_df[["ts_event", "symbol", "best_level_ofi", "ofi_level_0", "ofi_level_1", "ofi_level_2", "ofi_level_3", "ofi_level_4", "ofi_level_5", "ofi_level_6", "ofi_level_7", "ofi_level_8", "ofi_level_9", "integrated_ofi"]].head(10))
    final_ofi_df.to_csv("output/ofi_features_base.csv", index=False)

    # 4: Cross-Asset OFI with simulated data
    print("\nSimulating a multi-asset dataset...")
    df_multi_asset = simulate_other_symbols(df, symbol_variants=["MSFT", "GOOG"])

    print("\nComputing Cross-Asset OFI for the simulated dataset...")
    final_cross_asset_df = compute_cross_asset_ofi(df_multi_asset)

    # Print preview and save cross-asset output
    print("\nCross-Asset OFI Output")
    print(final_cross_asset_df[["ts_event", "symbol", "best_level_ofi", "integrated_ofi", "cross_asset_ofi"]].head(10))
    final_cross_asset_df.to_csv("output/ofi_features_with_cross_asset.csv", index=False)

