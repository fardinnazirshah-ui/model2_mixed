"""
scripts/utils/data_helpers.py
Data loading, alignment, and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_csv(file_path, date_column=None):
    """
    Load CSV file with automatic date parsing.
    
    Args:
        file_path (str): Path to CSV file
        date_column (str): Name of date column (auto-detected if None)
    
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(file_path)
        
        # Auto-detect date column
        if date_column is None:
            date_cols = ['date', 'Date', 'DATE', 'time', 'Time', 'datetime']
            for col in date_cols:
                if col in df.columns:
                    date_column = col
                    break
        
        # Parse dates
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise


def align_dataframes(dataframes, on_column='date', method='inner'):
    """
    Align multiple dataframes to common date range.
    
    Args:
        dataframes (list): List of dataframes
        on_column (str): Column to align on
        method (str): 'inner' (latest start, earliest end) or 'outer' (expand range)
    
    Returns:
        pd.DataFrame: Merged aligned dataframe
    """
    if not dataframes:
        raise ValueError("No dataframes provided")
    
    merged = dataframes[0]
    for df in dataframes[1:]:
        merged = pd.merge(merged, df, on=on_column, how=method)
    
    logger.info(f"Aligned {len(dataframes)} dataframes: {len(merged)} rows remaining")
    return merged


def handle_missing_values(df, strategy='forward_fill', limit=3):
    """
    Handle missing values with specified strategy.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): 'forward_fill', 'backward_fill', 'interpolate', 'drop'
        limit (int): Max consecutive fills
    
    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    missing_before = df.isnull().sum().sum()
    
    if strategy == 'forward_fill':
        df = df.fillna(method='ffill', limit=limit).fillna(method='bfill', limit=limit)
    elif strategy == 'backward_fill':
        df = df.fillna(method='bfill', limit=limit).fillna(method='ffill', limit=limit)
    elif strategy == 'interpolate':
        df = df.interpolate(method='linear', limit=limit)
    elif strategy == 'drop':
        df = df.dropna()
    
    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} → {missing_after}")
    
    return df


def normalize_features(X_train, X_test, method='standard', fit_on_train=True):
    """
    Normalize features to prevent data leakage.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        method (str): 'standard' or 'minmax'
        fit_on_train (bool): Fit scaler on training data only
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Fit ONLY on training data
    scaler.fit(X_train)
    
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    logger.info(f"Normalized features using {method} scaler")
    return X_train_scaled, X_test_scaled, scaler


def create_lagged_features(df, column, periods=[1, 3, 6, 12], prefix=''):
    """Create lagged features from time series column."""
    for period in periods:
        df[f"{prefix}{column}_lag{period}"] = df[column].shift(period)
    return df


def create_moving_averages(df, column, windows=[3, 6, 12], prefix=''):
    """Create moving average features."""
    for window in windows:
        df[f"{prefix}{column}_ma{window}"] = df[column].rolling(window).mean()
    return df


def create_growth_rates(df, column, periods=[1, 4, 12], prefix=''):
    """Create growth rate features."""
    for period in periods:
        df[f"{prefix}{column}_growth{period}"] = df[column].pct_change(period) * 100
    return df
