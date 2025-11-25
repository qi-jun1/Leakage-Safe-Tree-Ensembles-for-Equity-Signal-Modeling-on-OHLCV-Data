import numpy as np
import pandas as pd
from itertools import product

#Function to create purged K Folds determined by Date, so purging is correctly applied based on date, given that some Features are computed over larger time window
def purged_kfold_indices_by_date(df, date_col='Date', n_splits=5, purge_days=5):
    # Ensure sorted and datetime
    df = df.sort_values(date_col).copy()
    df[date_col] = pd.to_datetime(df[date_col])

    n_samples = len(df)
    pos_idx = np.arange(n_samples)     
    dates = df[date_col].to_numpy()         

    # Unique dates in order
    unique_dates = np.unique(dates)
    date_folds = np.array_split(unique_dates, n_splits)

    folds = []
    for fold_dates in date_folds:
        if len(fold_dates) == 0:
            continue

        # --- test: positions whose date is in this fold ---
        test_mask = np.isin(dates, fold_dates)
        test_idx = pos_idx[test_mask]

        # --- purge window in calendar days ---
        min_test_date = fold_dates[0]
        max_test_date = fold_dates[-1]

        purge_start = min_test_date - pd.Timedelta(days=purge_days)
        purge_end   = max_test_date + pd.Timedelta(days=purge_days)

        purge_mask = (dates >= purge_start) & (dates <= purge_end)
        purge_idx = pos_idx[purge_mask]

        # --- train: everything except test+purge ---
        train_idx = np.setdiff1d(pos_idx, np.union1d(test_idx, purge_idx))

        folds.append((train_idx, test_idx))

    return folds

# Function to test wether purged CV by Date successfully created Train/Test Splits
def test_purged_cv_by_date(df: pd.DataFrame,
                           folds,
                           purge_days = 60):

    purge_days = 60  # or 20, 40, whatever you used

    df_cv = df.sort_values('Date').copy()  # the same df you passed to CV
    folds = purged_kfold_indices_by_date(df_cv, date_col='Date', 
                                        n_splits=5, purge_days=purge_days)

    for i, (train_idx, test_idx) in enumerate(folds):
        train_dates = df_cv.iloc[train_idx]['Date']
        test_dates  = df_cv.iloc[test_idx]['Date']

        min_test = test_dates.min()
        max_test = test_dates.max()

        lower_purge = min_test - pd.Timedelta(days=purge_days)
        upper_purge = max_test + pd.Timedelta(days=purge_days)

        # test if any train dates for the fold are within the purged window around the Test Set
        violation = train_dates[(train_dates >= lower_purge) & (train_dates <= upper_purge)]

        print(f"Fold {i}: violations = {len(violation)}")

# Helper Function to create Dictionary with possible Hyperparaemter combinations
def param_grid_dicts(param_dict): 
    keys = list(param_dict.keys())
    values_product = product(*[param_dict[k] for k in keys]) 
    for combo in values_product:
        yield dict(zip(keys, combo))