#!/usr/bin/env python3
"""
align_csv.py - FIXED FOR WINDOWS COMPATIBILITY
CSV Alignment with Automatic FRED & World Bank Data Downloads

This script:
1. Automatically downloads economic data from FRED (Federal Reserve)
2. Automatically downloads data from World Bank
3. Aligns all datasets to common date range (MONTHLY resampling for daily data)
4. Handles missing values
5. Saves aligned CSV for ML training

Usage:
    python align_csv.py --fred-key YOUR_API_KEY --auto-download
    python align_csv.py --fred-key YOUR_API_KEY --auto-download --output-dir data/processed/

Requirements:
    - FRED API key (free from https://fredaccount.stlouisfed.org/login)
    - Python 3.10+
    - pandas, requests, numpy
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

# ============================================================================
# LOGGING SETUP (WINDOWS COMPATIBLE)
# ============================================================================

def setup_logging(output_dir):
    """Initialize logging to both file and console (Windows compatible)."""
    log_dir = Path(output_dir).parent / "logs" if "processed" in str(output_dir) else Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "data_alignment.log"
    
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
# FRED DATA DOWNLOAD
# ============================================================================

def download_fred_series(api_key, series_id, start_date="2000-01-01", end_date="2024-12-31"):
    """
    Download a single FRED series via API.
    
    Args:
        api_key (str): FRED API key (40 characters)
        series_id (str): FRED series ID (e.g., 'UNRATE', 'A191RL1Q225SBEA')
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: DataFrame with date and value columns, or None if failed
    """
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
    """Download all configured FRED series."""
    # FRED series to download (MONTHLY & QUARTERLY ONLY - no daily data!)
    fred_series = {
        # GDP & Growth
        'A191RL1Q225SBEA': 'Real_GDP_Quarterly',
        
        # Employment
        'UNRATE': 'Unemployment_Rate_Monthly',
        'PAYEMS': 'Payroll_Employment_Monthly',
        'CIVPART': 'Labor_Force_Participation_Monthly',
        
        # Interest Rates (MONTHLY AVERAGE)
        'DGS10': 'Treasury_10Y_Monthly',
        'DGS2': 'Treasury_2Y_Monthly',
        
        # Industrial & Production
        'INDPRO': 'Industrial_Production_Monthly',
        'HOUST': 'Housing_Starts_Monthly',
        
        # Consumer Activity
        'UMCSENT': 'Consumer_Sentiment_Monthly',
        
        # Inflation
        'CPIAUCSL': 'CPI_Monthly',
    }
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"DOWNLOADING {len(fred_series)} FRED SERIES")
    logger.info("="*80)
    
    dfs = []
    success_count = 0
    
    for series_id, description in fred_series.items():
        df = download_fred_series(api_key, series_id)
        if df is not None:
            # Resample daily data to monthly
            if len(df) > 365:  # Likely daily data
                df = df.set_index('date').resample('MS').last().reset_index()
                logger.info(f"  [Resampled to monthly] {series_id}: {len(df)} observations")
            
            dfs.append((series_id, df))
            success_count += 1
        else:
            logger.warning(f"Skipping {series_id} due to download failure")
    
    logger.info("")
    logger.info(f"[DOWNLOADED] {success_count}/{len(fred_series)} FRED series")
    
    return dfs


# ============================================================================
# WORLD BANK DATA DOWNLOAD
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
        
        logger.info(f"[Downloaded] {indicator_code}: {len(df)} observations")
        time.sleep(0.1)
        return df
    
    except Exception as e:
        logger.error(f"Failed to download {indicator_code}: {e}")
        return None


def download_all_world_bank_indicators(output_dir="data/raw/"):
    """Download World Bank indicators for USA."""
    indicators = {
        'NY.GDP.MKTP.CD': 'GDP_USD',
        'NY.GDP.PCAP.CD': 'GDP_Per_Capita_USD',
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
# DATA ALIGNMENT
# ============================================================================

def align_dataframes(fred_dfs, worldbank_dfs):
    """
    Align all downloaded dataframes to common date range.
    
    Strategy: Use INNER join on date. World Bank data is annual (end of year),
    so convert monthly FRED data to annual average.
    """
    logger.info("")
    logger.info("="*80)
    logger.info("ALIGNING DATAFRAMES")
    logger.info("="*80)
    
    if not fred_dfs and not worldbank_dfs:
        logger.error("No data to align!")
        return None
    
    # Convert all to annual frequency (World Bank constraint)
    annual_dfs = []
    
    # Process FRED data - convert to annual averages
    for series_id, df in fred_dfs:
        df_annual = df.copy()
        df_annual['year'] = df_annual['date'].dt.year
        df_annual = df_annual.groupby('year')[series_id].mean().reset_index()
        df_annual['date'] = pd.to_datetime(df_annual['year'].astype(str) + '-12-31')
        df_annual = df_annual[['date', series_id]]
        annual_dfs.append(df_annual)
        logger.info(f"  Converted {series_id} to annual: {len(df_annual)} observations")
    
    # Process World Bank data - already annual
    for indicator_code, df in worldbank_dfs:
        annual_dfs.append(df)
        logger.info(f"  World Bank {indicator_code}: {len(df)} observations")
    
    logger.info("")
    logger.info(f"Aligning {len(annual_dfs)} dataframes...")
    
    # Merge all dataframes
    merged = annual_dfs[0]
    logger.info(f"  Starting with {len(merged)} rows")
    
    for i, df in enumerate(annual_dfs[1:], 1):
        before = len(merged)
        merged = pd.merge(merged, df, on='date', how='inner')
        after = len(merged)
        logger.info(f"  After merge {i}: {before} -> {after} rows")
    
    if len(merged) == 0:
        logger.error("ERROR: No overlapping dates found after alignment!")
        logger.error("Possible cause: Date mismatch between FRED and World Bank data")
        return None
    
    # Sort by date
    merged = merged.sort_values('date').reset_index(drop=True)
    
    date_range = f"{merged['date'].min().date()} to {merged['date'].max().date()}"
    logger.info("")
    logger.info(f"[ALIGNMENT COMPLETE]")
    logger.info(f"  Rows: {len(merged)}")
    logger.info(f"  Columns: {len(merged.columns)}")
    logger.info(f"  Date range: {date_range}")
    logger.info(f"  Missing values: {merged.isnull().sum().sum()}")
    
    return merged


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

def generate_validation_report(df, output_file):
    """Generate comprehensive data validation report."""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("DATA VALIDATION REPORT")
    report_lines.append("="*80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary Statistics
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-"*80)
    report_lines.append(f"Total rows: {len(df)}")
    report_lines.append(f"Total columns: {len(df.columns)}")
    report_lines.append(f"Date range: {df['date'].min()} to {df['date'].max()}")
    report_lines.append(f"Time span: {(df['date'].max() - df['date'].min()).days} days\n")
    
    # Missing Values
    report_lines.append("MISSING VALUES")
    report_lines.append("-"*80)
    missing_per_col = df.isnull().sum()
    for col in df.columns:
        pct = (missing_per_col[col] / len(df)) * 100 if len(df) > 0 else 0
        report_lines.append(f"{col:30s}: {missing_per_col[col]:4d} ({pct:5.1f}%)")
    report_lines.append(f"Total missing: {missing_per_col.sum()}\n")
    
    # Data Types
    report_lines.append("DATA TYPES")
    report_lines.append("-"*80)
    for col in df.columns:
        report_lines.append(f"{col:30s}: {str(df[col].dtype)}")
    report_lines.append("")
    
    # Numeric Statistics
    report_lines.append("NUMERIC STATISTICS")
    report_lines.append("-"*80)
    report_lines.append(df.describe().to_string())
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
        description='Download FRED & World Bank data, align, and save CSV'
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
    
    args = parser.parse_args()
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    logger.info("")
    logger.info("="*80)
    logger.info("ECONOMIC DATA ALIGNMENT SYSTEM")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"FRED API Key: {args.fred_key[:8]}...{args.fred_key[-8:]}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Step 1: Download FRED data
        fred_dfs = download_all_fred_series(args.fred_key)
        
        if not fred_dfs:
            logger.error("Failed to download any FRED data. Check API key.")
            return 1
        
        # Step 2: Download World Bank data
        worldbank_dfs = download_all_world_bank_indicators()
        
        if not worldbank_dfs:
            logger.warning("No World Bank data downloaded, continuing with FRED only")
        
        # Step 3: Align dataframes
        aligned_df = align_dataframes(fred_dfs, worldbank_dfs)
        
        if aligned_df is None or len(aligned_df) == 0:
            logger.error("Failed to align dataframes")
            return 1
        
        # Step 4: Handle missing values
        aligned_df = handle_missing_values(aligned_df)
        
        # Step 5: Save to CSV
        logger.info("")
        logger.info("="*80)
        logger.info("SAVING ALIGNED DATA")
        logger.info("="*80)
        
        output_csv = output_dir / 'aligned_economic_indicators.csv'
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        aligned_df.to_csv(output_csv, index=False, encoding='utf-8')
        logger.info(f"[SAVED] {output_csv}")
        logger.info(f"  File size: {output_csv.stat().st_size / 1024:.1f} KB")
        
        # Step 6: Generate validation report
        logger.info("")
        logger.info("="*80)
        logger.info("GENERATING VALIDATION REPORT")
        logger.info("="*80)
        
        report_file = output_dir / 'data_validation_report.txt'
        generate_validation_report(aligned_df, report_file)
        
        # Final summary
        logger.info("")
        logger.info("="*80)
        logger.info("[EXECUTION COMPLETE - SUCCESS]")
        logger.info("="*80)
        logger.info(f"Output CSV: {output_csv.name}")
        logger.info(f"Validation report: {report_file.name}")
        logger.info(f"")
        logger.info(f"Next step: Train ML models")
        logger.info(f"  python scripts/recession_prediction_pipeline.py --input {output_csv}")
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
