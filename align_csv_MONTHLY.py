#!/usr/bin/env python3
"""
align_csv_FIXED_MONTHLY.py - MONTHLY DATA VERSION
CSV Alignment with MONTHLY frequency (not annual!)
Automatic FRED & World Bank Data Downloads

This script:
1. Downloads economic data from FRED (Federal Reserve)
2. Downloads data from World Bank
3. Converts ALL data to MONTHLY frequency (300 months instead of 25 years!)
4. Aligns all datasets to common date range
5. Creates NBER recession labels (ground truth, not auto-generated)
6. Handles missing values
7. Saves aligned CSV with recession labels for ML training

KEY CHANGES FROM PREVIOUS VERSION:
  - MONTHLY frequency instead of ANNUAL (12x more data!)
  - NBER recession dates for ground truth labels
  - Proper time-series structure
  - Shift labels FORWARD for lead time prediction

Usage:
    python align_csv_FIXED_MONTHLY.py --fred-key YOUR_API_KEY --auto-download
    python align_csv_FIXED_MONTHLY.py --fred-key YOUR_API_KEY --auto-download --output-dir data/processed/

Requirements:
    - FRED API key (free from https://fredaccount.stlouisfed.org/login)
    - Python 3.10+
    - pandas, requests, numpy, scikit-learn
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
from sklearn.utils import shuffle

# ============================================================================
# LOGGING SETUP (WINDOWS COMPATIBLE)
# ============================================================================

def setup_logging(output_dir):
    """Initialize logging to both file and console (Windows compatible)."""
    log_dir = Path(output_dir).parent / "logs" if "processed" in str(output_dir) else Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "data_alignment_monthly.log"
    
    # Windows compatibility: force UTF-8 encoding
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

logger = None

# ============================================================================
# FRED DATA DOWNLOAD - MONTHLY SERIES ONLY
# ============================================================================

def download_fred_series(api_key, series_id, start_date="2000-01-01", end_date="2024-12-31"):
    """Download a single FRED series via API."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': end_date,
    }
    
    try:
        logger.info(f"Downloading FRED series: {series_id}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'observations' not in data or len(data['observations']) == 0:
            logger.warning(f"No data returned for {series_id}")
            return None
        
        df = pd.DataFrame(data['observations'])
        df.rename(columns={'date': 'date', 'value': series_id}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        df = df[['date', series_id]].dropna(subset=[series_id])
        
        logger.info(f"[Downloaded] {series_id}: {len(df)} observations")
        time.sleep(0.1)
        return df
    
    except Exception as e:
        logger.error(f"Failed to download {series_id}: {e}")
        return None


def download_all_fred_series(api_key, output_dir="data/raw/"):
    """Download FRED monthly series (NO daily or quarterly data!)"""
    # ONLY MONTHLY FRED SERIES - for consistent frequency
    fred_series = {
        # Employment (Monthly)
        'UNRATE': 'Unemployment_Rate',
        'PAYEMS': 'Payroll_Employment',
        'CIVPART': 'Labor_Force_Participation',
        
        # Production & Activity (Monthly)
        'INDPRO': 'Industrial_Production',
        'HOUST': 'Housing_Starts',
        
        # Prices & Inflation (Monthly)
        'CPIAUCSL': 'CPI_All_Urban',
        'CPILFESL': 'CPI_Core',
        
        # Consumer Activity & Sentiment (Monthly)
        'UMCSENT': 'Consumer_Sentiment',
        'ICSA': 'Initial_Jobless_Claims',
        
        # Interest Rates (Monthly AVERAGE)
        'DGS10': 'Treasury_10Y',
        'DGS2': 'Treasury_2Y',
        'DFF': 'Federal_Funds_Rate',
        
        # Other Economic Indicators
        'VIXCLS': 'Volatility_Index_VIX',
    }
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"DOWNLOADING {len(fred_series)} MONTHLY FRED SERIES")
    logger.info("="*80)
    
    dfs = []
    success_count = 0
    
    for series_id, description in fred_series.items():
        df = download_fred_series(api_key, series_id)
        if df is not None:
            # Resample to MONTHLY if daily
            if len(df) > 300:  # More than 300 observations = daily data
                df = df.set_index('date').resample('MS').last().reset_index()
                logger.info(f"  [Resampled to monthly] {series_id}: {len(df)} observations")
            
            # Ensure monthly frequency
            df = df.set_index('date').asfreq('MS').reset_index()
            
            dfs.append((series_id, df))
            success_count += 1
        else:
            logger.warning(f"Skipping {series_id} due to download failure")
    
    logger.info("")
    logger.info(f"[DOWNLOADED] {success_count}/{len(fred_series)} FRED monthly series")
    
    return dfs


# ============================================================================
# WORLD BANK DATA DOWNLOAD - INTERPOLATE TO MONTHLY
# ============================================================================

def download_world_bank_indicator(indicator_code, country_code="USA", start_year=2000, end_year=2024):
    """Download World Bank indicator for a country."""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    
    params = {
        'format': 'json',
        'date': f"{start_year}:{end_year}",
        'per_page': 100,
    }
    
    try:
        logger.info(f"Downloading World Bank indicator: {indicator_code}")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if len(data) < 2 or len(data[1]) == 0:
            logger.warning(f"No data returned for {indicator_code}")
            return None
        
        records = data[1]
        df_list = []
        
        for record in records:
            if record['value'] is not None:
                df_list.append({
                    'date': pd.to_datetime(f"{record['date']}-12-31"),
                    indicator_code: float(record['value'])
                })
        
        if not df_list:
            logger.warning(f"No valid values for {indicator_code}")
            return None
        
        df = pd.DataFrame(df_list)
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"[Downloaded] {indicator_code}: {len(df)} annual observations")
        time.sleep(0.1)
        return df
    
    except Exception as e:
        logger.error(f"Failed to download {indicator_code}: {e}")
        return None


def download_all_world_bank_indicators(output_dir="data/raw/"):
    """Download World Bank indicators for USA."""
    indicators = {
        'NY.GDP.MKTP.CD': 'GDP_USD',
        'NY.GDP.PCAP.CD': 'GDP_Per_Capita',
        'NE.EXP.GNFS.CD': 'Exports_USD',
        'NE.IMP.GNFS.CD': 'Imports_USD',
    }
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"DOWNLOADING {len(indicators)} WORLD BANK INDICATORS")
    logger.info("="*80)
    
    dfs = []
    success_count = 0
    
    for indicator_code, description in indicators.items():
        df = download_world_bank_indicator(indicator_code)
        if df is not None:
            dfs.append((indicator_code, df))
            success_count += 1
        else:
            logger.warning(f"Skipping {indicator_code} due to download failure")
    
    logger.info("")
    logger.info(f"[DOWNLOADED] {success_count}/{len(indicators)} World Bank indicators")
    
    return dfs


# ============================================================================
# DATA ALIGNMENT AT MONTHLY FREQUENCY
# ============================================================================

def align_dataframes_monthly(fred_dfs, worldbank_dfs):
    """
    Align all data to MONTHLY frequency (300+ samples!)
    
    Strategy: 
    1. Keep FRED monthly data as-is
    2. Interpolate World Bank annual data to monthly
    3. Create unified monthly time series
    """
    logger.info("")
    logger.info("="*80)
    logger.info("ALIGNING DATAFRAMES TO MONTHLY FREQUENCY")
    logger.info("="*80)
    
    if not fred_dfs and not worldbank_dfs:
        logger.error("No data to align!")
        return None
    
    # Create monthly date range
    date_range = pd.date_range(start='2000-01-01', end='2024-12-31', freq='MS')
    merged = pd.DataFrame({'date': date_range})
    
    logger.info(f"Created monthly date range: {len(merged)} months")
    logger.info(f"  From: {merged['date'].min().date()}")
    logger.info(f"  To: {merged['date'].max().date()}")
    
    # Add FRED monthly data (no interpolation needed)
    logger.info("")
    logger.info("Adding FRED monthly series...")
    for series_id, df in fred_dfs:
        df_temp = df.copy()
        df_temp = df_temp.set_index('date').reindex(merged['date']).reset_index()
        df_temp.columns = ['date', series_id]
        
        # Forward fill and backward fill for small gaps
        df_temp[series_id] = df_temp[series_id].fillna(method='ffill').fillna(method='bfill')
        
        before = merged.isnull().sum().sum()
        merged = pd.merge(merged, df_temp[['date', series_id]], on='date', how='left')
        after = merged.isnull().sum().sum()
        
        logger.info(f"  Added {series_id}: {len(df)} observations -> {merged[series_id].notna().sum()} months")
    
    # Interpolate World Bank annual data to monthly
    logger.info("")
    logger.info("Interpolating World Bank annual data to monthly...")
    for indicator_code, df in worldbank_dfs:
        df_temp = df.copy()
        
        # Create monthly series with annual data
        monthly_data = pd.DataFrame({
            'date': merged['date'],
            indicator_code: np.nan
        })
        
        for idx, row in df_temp.iterrows():
            # Find month matching this year-end
            year = row['date'].year
            month_idx = (merged['date'].dt.year == year).idxmax()
            monthly_data.loc[month_idx, indicator_code] = row[indicator_code]
        
        # Interpolate annual values across the year
        monthly_data[indicator_code] = monthly_data[indicator_code].interpolate(method='linear')
        
        # Forward and backward fill for edges
        monthly_data[indicator_code] = monthly_data[indicator_code].fillna(method='ffill').fillna(method='bfill')
        
        merged = pd.merge(merged, monthly_data, on='date', how='left')
        logger.info(f"  Interpolated {indicator_code}: {monthly_data[indicator_code].notna().sum()} months")
    
    # Remove rows with any missing values
    before_rows = len(merged)
    merged = merged.dropna()
    after_rows = len(merged)
    
    logger.info("")
    logger.info(f"[ALIGNMENT COMPLETE]")
    logger.info(f"  Rows after removing NaNs: {before_rows} -> {after_rows}")
    logger.info(f"  Columns: {len(merged.columns) - 1} (excluding date)")
    logger.info(f"  Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
    logger.info(f"  Missing values: {merged.isnull().sum().sum()}")
    
    return merged


# ============================================================================
# RECESSION LABELS - NBER GROUND TRUTH
# ============================================================================

def create_recession_labels(df, lead_months=0):
    """
    Create recession labels using NBER official recession dates.
    
    NBER defines recessions:
    - 2001 Recession: Mar 2001 - Nov 2001 (8 months)
    - 2007-2009 Recession: Dec 2007 - Jun 2009 (18 months)
    
    Args:
        df: DataFrame with 'date' column
        lead_months: How many months ahead to predict
                    0 = predict current recession
                    3 = predict recession in next 3 months
                    6 = predict recession in next 6 months
    
    Returns:
        np.array of 0s and 1s
    """
    logger.info("")
    logger.info("="*80)
    logger.info("CREATING RECESSION LABELS (NBER GROUND TRUTH)")
    logger.info("="*80)
    
    # NBER official recession dates
    nber_recessions = [
        {
            'name': '2001 Recession',
            'start': pd.to_datetime('2001-03-01'),
            'end': pd.to_datetime('2001-11-30')
        },
        {
            'name': '2007-2009 Great Recession',
            'start': pd.to_datetime('2007-12-01'),
            'end': pd.to_datetime('2009-06-30')
        }
    ]
    
    logger.info(f"NBER Recessions:")
    for rec in nber_recessions:
        duration = (rec['end'] - rec['start']).days // 30
        logger.info(f"  {rec['name']}: {rec['start'].date()} to {rec['end'].date()} ({duration} months)")
    
    labels = []
    
    for date in df['date']:
        # Check if this date (plus lead time) falls in a recession
        prediction_date = date + pd.DateOffset(months=lead_months)
        
        is_recession = False
        for rec in nber_recessions:
            if rec['start'] <= prediction_date <= rec['end']:
                is_recession = True
                break
        
        labels.append(1 if is_recession else 0)
    
    labels = np.array(labels)
    
    # Class distribution
    n_recession = np.sum(labels)
    n_normal = len(labels) - n_recession
    pct_recession = (n_recession / len(labels)) * 100
    
    logger.info("")
    logger.info(f"Label Distribution:")
    logger.info(f"  Normal periods: {n_normal} ({100-pct_recession:.1f}%)")
    logger.info(f"  Recession periods: {n_recession} ({pct_recession:.1f}%)")
    logger.info(f"  Class imbalance ratio: {n_normal / max(n_recession, 1):.1f}:1")
    logger.info(f"  Lead time: {lead_months} months ahead")
    
    return labels


# ============================================================================
# MISSING VALUE HANDLING
# ============================================================================

def handle_missing_values(df, strategy='forward_fill', limit=3):
    """Handle missing values using specified strategy."""
    logger.info("")
    logger.info("="*80)
    logger.info("HANDLING MISSING VALUES")
    logger.info("="*80)
    
    missing_before = df.isnull().sum().sum()
    logger.info(f"Missing values before: {missing_before}")
    
    if strategy == 'forward_fill':
        df = df.fillna(method='ffill', limit=limit).fillna(method='bfill', limit=limit)
        logger.info(f"Applied forward-fill (limit={limit}) then backward-fill")
    
    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values after: {missing_after}")
    
    return df


# ============================================================================
# VALIDATION REPORT
# ============================================================================

def generate_validation_report(df, y, output_file):
    """Generate comprehensive data validation report."""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("DATA VALIDATION REPORT - MONTHLY FREQUENCY")
    report_lines.append("="*80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary Statistics
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-"*80)
    report_lines.append(f"Total rows: {len(df)}")
    report_lines.append(f"Total columns: {len(df.columns) - 1} (features)")
    report_lines.append(f"Frequency: MONTHLY (not annual!)")
    report_lines.append(f"Date range: {df['date'].min()} to {df['date'].max()}")
    report_lines.append(f"Time span: {(df['date'].max() - df['date'].min()).days} days ({len(df)} months)")
    report_lines.append("")
    
    # Target Distribution
    report_lines.append("TARGET VARIABLE DISTRIBUTION (NBER Recession Labels)")
    report_lines.append("-"*80)
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        pct = (count / len(y)) * 100
        label_name = "Recession" if label == 1 else "Normal"
        report_lines.append(f"{label_name:15s}: {count:4d} ({pct:5.1f}%)")
    
    imbalance_ratio = counts[np.argmax(counts)] / counts[np.argmin(counts)]
    report_lines.append(f"Class imbalance: {imbalance_ratio:.1f}:1\n")
    
    # Missing Values
    report_lines.append("MISSING VALUES (per feature)")
    report_lines.append("-"*80)
    missing_per_col = df.isnull().sum()
    for col in df.columns:
        if col != 'date':
            pct = (missing_per_col[col] / len(df)) * 100 if len(df) > 0 else 0
            report_lines.append(f"{col:30s}: {missing_per_col[col]:4d} ({pct:5.1f}%)")
    report_lines.append(f"Total missing: {missing_per_col.sum()}\n")
    
    # Data Types
    report_lines.append("DATA TYPES")
    report_lines.append("-"*80)
    for col in df.columns:
        if col != 'date':
            report_lines.append(f"{col:30s}: {str(df[col].dtype)}")
    report_lines.append("")
    
    # Numeric Statistics
    report_lines.append("NUMERIC STATISTICS")
    report_lines.append("-"*80)
    feature_cols = [col for col in df.columns if col != 'date']
    report_lines.append(df[feature_cols].describe().to_string())
    report_lines.append("")
    
    # Write to file
    report_text = "\n".join(report_lines)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"[SAVED] Validation report: {output_file}")
    return report_text


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    global logger
    
    parser = argparse.ArgumentParser(
        description='Download FRED & World Bank data, align to MONTHLY frequency with NBER labels'
    )
    
    parser.add_argument(
        '--fred-key',
        type=str,
        required=True,
        help='FRED API key (40 characters)'
    )
    
    parser.add_argument(
        '--auto-download',
        action='store_true',
        required=True,
        help='Enable automatic download from FRED and World Bank'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/',
        help='Output directory for aligned CSV'
    )
    
    parser.add_argument(
        '--lead-months',
        type=int,
        default=3,
        help='Months ahead to predict (0=current, 3=3 months ahead, 6=6 months ahead)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    logger.info("")
    logger.info("="*80)
    logger.info("ECONOMIC DATA ALIGNMENT SYSTEM - MONTHLY FREQUENCY VERSION")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"FRED API Key: {args.fred_key[:8]}...{args.fred_key[-8:]}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Lead time (prediction window): {args.lead_months} months ahead")
    
    try:
        # Step 1: Download FRED data
        logger.info("")
        fred_dfs = download_all_fred_series(args.fred_key)
        
        if not fred_dfs:
            logger.error("Failed to download any FRED data. Check API key.")
            return 1
        
        # Step 2: Download World Bank data
        logger.info("")
        worldbank_dfs = download_all_world_bank_indicators()
        
        if not worldbank_dfs:
            logger.warning("No World Bank data downloaded, continuing with FRED only")
        
        # Step 3: Align dataframes at MONTHLY frequency
        logger.info("")
        aligned_df = align_dataframes_monthly(fred_dfs, worldbank_dfs)
        
        if aligned_df is None or len(aligned_df) == 0:
            logger.error("Failed to align dataframes")
            return 1
        
        # Step 4: Create NBER recession labels
        logger.info("")
        y = create_recession_labels(aligned_df, lead_months=args.lead_months)
        
        # Step 5: Handle missing values
        logger.info("")
        aligned_df = handle_missing_values(aligned_df)
        
        # Step 6: Save to CSV
        logger.info("")
        logger.info("="*80)
        logger.info("SAVING ALIGNED DATA WITH NBER LABELS")
        logger.info("="*80)
        
        # Add target column
        aligned_df['recession_label'] = y
        
        output_csv = output_dir / 'aligned_economic_indicators_monthly.csv'
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        aligned_df.to_csv(output_csv, index=False, encoding='utf-8')
        logger.info(f"[SAVED] {output_csv}")
        logger.info(f"  File size: {output_csv.stat().st_size / 1024:.1f} KB")
        logger.info(f"  Shape: {aligned_df.shape[0]} rows x {aligned_df.shape[1]} columns")
        
        # Step 7: Generate validation report
        logger.info("")
        logger.info("="*80)
        logger.info("GENERATING VALIDATION REPORT")
        logger.info("="*80)
        
        report_file = output_dir / 'data_validation_report_monthly.txt'
        generate_validation_report(aligned_df, y, report_file)
        
        # Final summary
        logger.info("")
        logger.info("="*80)
        logger.info("[EXECUTION COMPLETE - SUCCESS]")
        logger.info("="*80)
        logger.info(f"Output CSV: {output_csv.name}")
        logger.info(f"  Samples: {len(aligned_df)} monthly observations")
        logger.info(f"  Features: {len(aligned_df.columns) - 2}")
        logger.info(f"  Target: recession_label (NBER ground truth)")
        logger.info(f"  Lead time: {args.lead_months} months ahead")
        logger.info(f"")
        logger.info(f"Validation report: {report_file.name}")
        logger.info(f"")
        logger.info(f"Next step: Train ML models with PROPER data!")
        logger.info(f"  python scripts/recession_prediction_pipeline_FIXED.py --input {output_csv}")
        logger.info(f"")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("\nExecution interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"\nFATAL ERROR: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
