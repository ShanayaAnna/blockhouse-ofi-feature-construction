# OFI Feature Construction

This repo contains a Python script to compute the following Order Flow Imbalance (OFI) features from limit order book data:

- Best-Level OFI
- Multi-Level OFI
- Integrated OFI (via PCA)
- Cross-Asset OFI

## Note 1
Dataset contains 10 levels of bid/ask prices and sizes per timestamp.


## Note 2: Note on Dataset Discrepancy:
The dataset provided for this task was titled `first_25000_rows.csv`.

However, upon inspection:
- The actual **data shape is**: `(5000, 74)`
- It contains **only one symbol**: `AAPL`
  
So to construct the Cross-Asset OFI Feature
- We simulated **two additional symbols** — `MSFT` and `GOOG`
- These were derived from the existing `AAPL` data by applying **slight, randomized noise** to:
  - Bid/ask prices
  - Bid/ask sizes
- The resulting dataset mimics the structure of **Nasdaq-100** data used in the paper

# Output:
There are two output CSVs:
1. output/ofi_features_base.csv
This file includes per-symbol feature columns for:
best_level_ofi 
ofi_level_0 to ofi_level_9 
integrated_ofi 
Each row corresponds to one event in the dataset.

2. output/ofi_features_with_cross_asset.csv
This file includes the same fields as above plus:
cross_asset_ofi 
This is executed on the simulated dataset as described above

## Sample Output
Data shape: (5000, 74)
Symbols in dataset: ['AAPL']

Computing OFI features for symbol: AAPL

Base OFI Features (Best, Multi-Level, Integrated)
ts_event                          symbol  best_level_ofi  ofi_level_0  ...  integrated_ofi
2024-10-21 11:54:29.221064336+00:00 AAPL             0.0          0.0       0.000000
2024-10-21 11:54:29.223769812+00:00 AAPL             2.0          2.0       0.035961
2024-10-21 11:54:29.225030400+00:00 AAPL             3.0          3.0       0.053942
2024-10-21 11:54:29.712434212+00:00 AAPL             0.0          0.0      21.060680
2024-10-21 11:54:29.764673165+00:00 AAPL             0.0          0.0     -21.060680
...


[10 rows x 14 columns]

Simulating a multi-asset dataset...

Computing Cross-Asset OFI for the simulated dataset...
Running OFI pipeline for AAPL
Running OFI pipeline for GOOG
Running OFI pipeline for MSFT

Cross-Asset OFI Output
ts_event                          symbol  best_level_ofi  integrated_ofi  cross_asset_ofi
2024-10-21 11:54:29.221064336+00:00 AAPL             0.0         0.000000         0.000000
2024-10-21 11:54:29.223769812+00:00 AAPL             2.0         0.035961        30.826914
2024-10-21 11:54:29.225030400+00:00 AAPL             3.0         0.053942      -174.127164
2024-10-21 11:54:29.712434212+00:00 AAPL             0.0        21.060680       102.847409
2024-10-21 11:54:29.764673165+00:00 AAPL             0.0       -21.060680      -176.684852
...


Author: Shanaya Anna Varkey
