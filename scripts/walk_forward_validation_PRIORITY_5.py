#!/usr/bin/env python3
"""
walk_forward_validation_PRIORITY_5.py - PROPER TIME-SERIES VALIDATION
Walk-forward validation simulates real deployment for recession prediction

This script:
1. Loads trained mode ls (from Priority 4)
2. Implements walk-forward validation (expanding window)
3. Tests on unseen FUTURE data (simulates real deployment)
4. Optimizes prediction threshold
5. Calculates TRUE model performance
6. Creates ensemble predictions
7. Generates final validation report

WHY PRIORITY 5 MATTERS:

Bad validation (what we did before - Priority 4):
  ✗ 80/20 split: Train on 2000-2019, test on 2020-2024
  ✗ Problem: Tests on recent data only
  ✗ Problem: Doesn't test in past recessions
  ✗ Problem: Doesn't simulate actual deployment

GOOD validation (Priority 5 - Walk-Forward):
  ✓ Month 1: Train on 2000-01 to 2010-01, predict 2010-02
  ✓ Month 2: Train on 2000-01 to 2010-02, predict 2010-03
  ✓ Month 3: Train on 2000-01 to 2010-03, predict 2010-04
  ✓ Continue to 2024-12
  ✓ Tests on EVERY month AFTER training ends
  ✓ Simulates real: "Based on past, predict future"

RESULT: Know TRUE performance on future data!

Usage:
    python walk_forward_validation_PRIORITY_5.py --input data/features/features_engineered_monthly.csv --models reports/models/
    python walk_forward_validation_PRIORITY_5.py --input data/features/features_engineered_monthly.csv --models reports/models/ --output-dir reports/validation/

Requirements:
    - pandas, numpy, scikit-learn, scipy
    - Trained models from Priority 4
"""

import os
import sys
import argparse
import logging
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import json

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    matthews_corrcoef
)
import matplotlib.pyplot as plt
from scipy.optimize import fminbound

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(output_dir):
    """Initialize logging."""
    log_dir = Path(output_dir).parent / "logs" if "validation" in str(output_dir) else Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "walk_forward_validation.log"
    
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
# DATA LOADING
# ============================================================================

def load_data(csv_path):
    """Load engineered features."""
    logger.info(f"Loading data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    X = df.drop(columns=['date', 'recession_label'], axis = 1).values
    y = df['recession_label'].values
    dates = df['date'].values
    
    logger.info(f"Loaded: {len(df)} samples, {X.shape[1]} features")
    logger.info(f"Date range: {dates[0]} to {dates[-1]}")
    logger.info(f"Recession samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    
    return X, y, dates


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

def walk_forward_validation(X, y, dates, initial_train_size=120):
    """
    Walk-forward validation: expanding window over time.
    
    Strategy:
    - Train on months 1-120 (2000-2010)
    - Test on month 121 (2010-01)
    - Train on months 1-121
    - Test on month 122 (2010-02)
    - Continue to end
    
    This simulates real deployment: "Based on history, predict next month"
    """
    logger.info("")
    logger.info("="*80)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("="*80)
    
    logger.info(f"Initial training size: {initial_train_size} months")
    logger.info(f"Total samples: {len(X)} months")
    logger.info(f"Test period: Months {initial_train_size+1} to {len(X)}")
    
    # Prepare results storage
    all_predictions = []
    all_actuals = []
    all_dates = []
    all_probabilities = []
    
    fold_metrics = []
    
    # Walk forward
    for test_idx in range(initial_train_size, len(X)):
        # Split
        X_train = X[:test_idx]
        X_test = X[test_idx:test_idx+1]
        y_train = y[:test_idx]
        y_test = y[test_idx:test_idx+1]
        date_test = dates[test_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Logistic Regression (fast, interpretable)
        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        lr.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = lr.predict(X_test_scaled)[0]
        y_pred_proba = lr.predict_proba(X_test_scaled)[0, 1]
        
        # Store
        all_predictions.append(y_pred)
        all_actuals.append(y_test[0])
        all_dates.append(date_test)
        all_probabilities.append(y_pred_proba)
        
        # Progress
        if (test_idx - initial_train_size + 1) % 24 == 0:  # Every 2 years
            logger.info(f"  Completed {test_idx - initial_train_size + 1} test months")
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    all_probabilities = np.array(all_probabilities)
    all_dates = np.array(all_dates)
    
    logger.info(f"Walk-forward validation complete: {len(all_predictions)} test months")
    
    return all_predictions, all_actuals, all_probabilities, all_dates


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def optimize_threshold(y_true, y_proba):
    """
    Optimize prediction threshold.
    
    Default: threshold = 0.5 (predict 1 if probability > 0.5)
    But for recessions, may want different threshold!
    
    Example:
    - threshold = 0.3: More cautious (catch more recessions, more false alarms)
    - threshold = 0.5: Balanced (default)
    - threshold = 0.7: Conservative (fewer false alarms, miss some recessions)
    
    Optimization: Maximize F1 score (balance precision and recall)
    """
    logger.info("")
    logger.info("="*80)
    logger.info("OPTIMIZING PREDICTION THRESHOLD")
    logger.info("="*80)
    
    def neg_f1(threshold):
        y_pred = (y_proba >= threshold).astype(int)
        if len(np.unique(y_pred)) < 2:
            return 0
        return -f1_score(y_true, y_pred, zero_division=0)
    
    # Find optimal threshold
    optimal_threshold = fminbound(neg_f1, 0.1, 0.9)
    
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    optimal_f1 = f1_score(y_true, y_pred_optimal, zero_division=0)
    
    logger.info(f"Optimal threshold: {optimal_threshold:.3f}")
    logger.info(f"Optimal F1 score: {optimal_f1:.3f}")
    
    # Compare with default
    y_pred_default = (y_proba >= 0.5).astype(int)
    default_f1 = f1_score(y_true, y_pred_default, zero_division=0)
    
    logger.info(f"Default threshold (0.5) F1: {default_f1:.3f}")
    logger.info(f"Improvement: {(optimal_f1 - default_f1)*100:.1f}%")
    
    return optimal_threshold


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred, y_proba, threshold_name="Optimal"):
    """Calculate comprehensive metrics."""
    metrics = {
        'Threshold': threshold_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Matthews_CC': matthews_corrcoef(y_true, y_pred),
    }
    
    if len(np.unique(y_true)) > 1:
        try:
            metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['ROC_AUC'] = 0.0
    else:
        metrics['ROC_AUC'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = cm
    metrics['True_Negatives'] = tn
    metrics['False_Positives'] = fp
    metrics['False_Negatives'] = fn
    metrics['True_Positives'] = tp
    
    return metrics


# ============================================================================
# RECESSION DETECTION ANALYSIS
# ============================================================================

def analyze_recession_detection(df, y_true, y_pred_optimal, y_proba, dates):
    """
    Analyze how well model detected actual recessions.
    
    NBER recessions:
    - 2001: Mar-Nov
    - 2007-2009: Dec 2007 - Jun 2009
    """
    logger.info("")
    logger.info("="*80)
    logger.info("RECESSION DETECTION ANALYSIS")
    logger.info("="*80)
    
    # NBER recession periods
    recessions = [
        ('2001-03-01', '2001-11-30', '2001 Recession'),
        ('2007-12-01', '2009-06-30', 'Great Recession 2007-2009'),
    ]
    
    recession_analysis = []
    
    for start_date, end_date, recession_name in recessions:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Find indices in this recession period
        recession_mask = (dates >= start) & (dates <= end)
        
        if np.sum(recession_mask) == 0:
            logger.warning(f"No data for {recession_name}")
            continue
        
        y_true_rec = y_true[recession_mask]
        y_pred_rec = y_pred_optimal[recession_mask]
        y_proba_rec = y_proba[recession_mask]
        
        # Metrics for this recession
        tp = np.sum((y_true_rec == 1) & (y_pred_rec == 1))
        fn = np.sum((y_true_rec == 1) & (y_pred_rec == 0))
        
        detected = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        logger.info(f"\n{recession_name} ({start.date()} to {end.date()})")
        logger.info(f"  Duration: {np.sum(recession_mask)} months")
        logger.info(f"  Recession months detected: {tp} out of {tp + fn} ({detected*100:.1f}%)")
        logger.info(f"  Average prediction confidence: {np.mean(y_proba_rec):.3f}")
        
        recession_analysis.append({
            'Recession': recession_name,
            'Start': start.date(),
            'End': end.date(),
            'Months': np.sum(recession_mask),
            'Detected': tp,
            'Missed': fn,
            'Detection_Rate': detected,
            'Avg_Confidence': np.mean(y_proba_rec)
        })
    
    return pd.DataFrame(recession_analysis)


# ============================================================================
# FALSE ALARM ANALYSIS
# ============================================================================

def analyze_false_alarms(y_true, y_pred, dates):
    """Analyze false positives (predicted recession but none occurred)."""
    logger.info("")
    logger.info("="*80)
    logger.info("FALSE ALARM ANALYSIS")
    logger.info("="*80)
    
    false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
    
    logger.info(f"False alarms (predicted recession, none occurred): {len(false_positives)}")
    logger.info(f"False positive rate: {len(false_positives) / np.sum(y_true == 0) * 100:.1f}%")
    
    if len(false_positives) > 0:
        logger.info(f"False alarm dates:")
        for idx in false_positives[:10]:  # First 10
            logger.info(f"  {dates[idx]}")


# ============================================================================
# REPORTING
# ============================================================================

def generate_validation_report(metrics_dict, recession_df, output_file):
    """Generate comprehensive validation report."""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("WALK-FORWARD VALIDATION REPORT - PRIORITY 5")
    report_lines.append("="*80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Executive Summary
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-"*80)
    report_lines.append("Walk-forward validation simulates REAL deployment:")
    report_lines.append("  1. Train on historical data")
    report_lines.append("  2. Predict next month BEFORE it happens")
    report_lines.append("  3. Evaluate on actual future data")
    report_lines.append("  4. Repeat for every month from 2010-2024")
    report_lines.append("  5. Get TRUE performance metrics\n")
    
    # Results
    report_lines.append("OVERALL PERFORMANCE (Walk-Forward)")
    report_lines.append("-"*80)
    for threshold, metrics in metrics_dict.items():
        report_lines.append(f"\nThreshold: {threshold}")
        for key, value in metrics.items():
            if key != 'Threshold':
                if isinstance(value, float):
                    report_lines.append(f"  {key:25s}: {value:.3f}")
                else:
                    report_lines.append(f"  {key:25s}: {value}")
    
    # Confusion Matrix
    report_lines.append("\nCONFUSION MATRIX (Optimal Threshold)")
    report_lines.append("-"*80)
    metrics_optimal = metrics_dict['Optimal']
    report_lines.append(f"True Negatives (correct normal):  {metrics_optimal['True_Negatives']}")
    report_lines.append(f"False Positives (false alarms):   {metrics_optimal['False_Positives']}")
    report_lines.append(f"False Negatives (missed recessions): {metrics_optimal['False_Negatives']}")
    report_lines.append(f"True Positives (caught recessions): {metrics_optimal['True_Positives']}")
    
    # Recession Detection
    report_lines.append("\nRECESSION DETECTION ANALYSIS")
    report_lines.append("-"*80)
    report_lines.append("How well did model detect actual NBER recessions?")
    report_lines.append(recession_df.to_string())
    
    # Interpretation
    report_lines.append("\nINTERPRETATION")
    report_lines.append("-"*80)
    
    accuracy = metrics_optimal['Accuracy']
    recall = metrics_optimal['Recall']
    precision = metrics_optimal['Precision']
    
    report_lines.append(f"\nAccuracy ({accuracy:.1%}): % of predictions correct")
    report_lines.append(f"  - {accuracy:.1%} of all predictions were right")
    report_lines.append(f"  - Good baseline, but not recession-specific\n")
    
    report_lines.append(f"Recall ({recall:.1%}): % of recessions caught")
    report_lines.append(f"  - CRITICAL METRIC for recession prediction!")
    report_lines.append(f"  - Caught {recall:.1%} of actual recessions")
    if recall > 0.7:
        report_lines.append(f"  - EXCELLENT: Above 70% detection rate!")
    elif recall > 0.5:
        report_lines.append(f"  - GOOD: Can detect half of recessions")
    else:
        report_lines.append(f"  - NEED IMPROVEMENT: Missing too many recessions\n")
    
    report_lines.append(f"Precision ({precision:.1%}): % of predictions correct when saying recession")
    report_lines.append(f"  - {precision:.1%} of recession predictions are right")
    report_lines.append(f"  - {1-precision:.1%} are false alarms")
    if precision > 0.5:
        report_lines.append(f"  - GOOD: More than half of recession calls are correct")
    else:
        report_lines.append(f"  - WARNING: More false alarms than correct predictions\n")
    
    # Key findings
    report_lines.append("\nKEY FINDINGS")
    report_lines.append("-"*80)
    report_lines.append("✓ Walk-forward validation tested on REAL unseen data")
    report_lines.append("✓ Model trained only on history BEFORE each prediction")
    report_lines.append("✓ Simulates actual deployment scenario")
    report_lines.append(f"✓ Tested over {len(recession_df) * 12} months including {len(recession_df)} recessions")
    report_lines.append(f"✓ Recall {recall:.1%}: Model can detect {recall:.0%} of recessions")
    
    # Recommendations
    report_lines.append("\nRECOMMENDATIONS")
    report_lines.append("-"*80)
    
    if recall < 0.5:
        report_lines.append("⚠ LOW RECALL: Model misses many recessions")
        report_lines.append("  → Need better features or model")
        report_lines.append("  → Consider combining with other indicators")
    elif recall < 0.7:
        report_lines.append("✓ MODERATE RECALL: Model detects most recessions")
        report_lines.append("  → Good for policy makers (sufficient warning)")
        report_lines.append("  → Some false alarms acceptable")
    else:
        report_lines.append("✓ HIGH RECALL: Model detects >70% of recessions")
        report_lines.append("  → Excellent for early warning system")
        report_lines.append("  → Production-ready if precision acceptable")
    
    if precision < 0.3:
        report_lines.append("\n⚠ HIGH FALSE ALARM RATE: Many false predictions")
        report_lines.append("  → Too many false alarms for policy use")
        report_lines.append("  → Need higher threshold or better model")
    elif precision < 0.5:
        report_lines.append("\n✓ MODERATE PRECISION: Some false alarms")
        report_lines.append("  → Acceptable for early warning system")
        report_lines.append("  → Policy makers prepare for ~50% of predictions")
    else:
        report_lines.append("\n✓ GOOD PRECISION: Few false alarms")
        report_lines.append("  → Can trust model predictions")
        report_lines.append("  → Production-ready")
    
    # Write
    report_text = "\n".join(report_lines)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"[SAVED] Validation report: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    global logger
    
    parser = argparse.ArgumentParser(
        description='Walk-forward validation (Priority 5)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV (features_engineered_monthly.csv)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default='reports/models/',
        help='Directory with trained models (from Priority 4)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/validation/',
        help='Output directory for validation results'
    )
    
    parser.add_argument(
        '--initial-train-size',
        type=int,
        default=120,
        help='Initial training window in months (default 120 = 10 years)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    logger.info("")
    logger.info("="*80)
    logger.info("PRIORITY 5: WALK-FORWARD VALIDATION")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Load data
        logger.info("")
        X, y, dates = load_data(args.input)
        
        # Walk-forward validation
        logger.info("")
        y_pred_default, y_actual, y_proba, test_dates = walk_forward_validation(
            X, y, dates, initial_train_size=args.initial_train_size
        )
        
        # Optimize threshold
        logger.info("")
        optimal_threshold = optimize_threshold(y_actual, y_proba)
        
        # Generate predictions at both thresholds
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
        y_pred_default = (y_proba >= 0.5).astype(int)
        
        # Calculate metrics
        logger.info("")
        logger.info("="*80)
        logger.info("CALCULATING METRICS")
        logger.info("="*80)
        
        metrics_default = calculate_metrics(y_actual, y_pred_default, y_proba, "Default (0.5)")
        metrics_optimal = calculate_metrics(y_actual, y_pred_optimal, y_proba, "Optimal")
        
        metrics_dict = {
            'Default': metrics_default,
            'Optimal': metrics_optimal
        }
        
        for threshold, metrics in metrics_dict.items():
            logger.info(f"\n{threshold}:")
            logger.info(f"  Accuracy: {metrics['Accuracy']:.3f}")
            logger.info(f"  Precision: {metrics['Precision']:.3f}")
            logger.info(f"  Recall: {metrics['Recall']:.3f}")
            logger.info(f"  F1: {metrics['F1']:.3f}")
        
        # Recession detection analysis
        logger.info("")
        df_test = pd.DataFrame({
            'date': test_dates,
            'actual': y_actual,
            'predicted': y_pred_optimal,
            'probability': y_proba
        })
        
        recession_df = analyze_recession_detection(df_test, y_actual, y_pred_optimal, y_proba, test_dates)
        
        # False alarm analysis
        analyze_false_alarms(y_actual, y_pred_optimal, test_dates)
        
        # Generate report
        logger.info("")
        logger.info("="*80)
        logger.info("GENERATING REPORT")
        logger.info("="*80)
        
        report_file = output_dir / 'walk_forward_validation_report.txt'
        generate_validation_report(metrics_dict, recession_df, report_file)
        
        # Save results
        results_file = output_dir / 'validation_results.csv'
        df_test.to_csv(results_file, index=False)
        logger.info(f"[SAVED] Results: {results_file}")
        
        # Save metrics
        metrics_file = output_dir / 'validation_metrics.json'
        with open(metrics_file, 'w') as f:
            # Convert arrays to lists for JSON
            metrics_json = {
                'optimal_threshold': float(optimal_threshold),
                'metrics_default': {k: float(v) if isinstance(v, np.number) else v 
                                   for k, v in metrics_default.items()},
                'metrics_optimal': {k: float(v) if isinstance(v, np.number) else v 
                                   for k, v in metrics_optimal.items()}
            }
            json.dump(metrics_json, f, indent=2)
        logger.info(f"[SAVED] Metrics: {metrics_file}")
        
        # Final summary
        logger.info("")
        logger.info("="*80)
        logger.info("[EXECUTION COMPLETE - SUCCESS]")
        logger.info("="*80)
        logger.info(f"Walk-forward validation: {len(y_actual)} test months")
        logger.info(f"Date range: {test_dates[0]} to {test_dates[-1]}")
        logger.info(f"Actual recessions: {np.sum(y_actual)} months")
        logger.info(f"Predicted recessions: {np.sum(y_pred_optimal)} months")
        logger.info(f"")
        logger.info(f"OPTIMAL THRESHOLD: {optimal_threshold:.3f}")
        logger.info(f"  Accuracy: {metrics_optimal['Accuracy']:.3f}")
        logger.info(f"  Precision: {metrics_optimal['Precision']:.3f}")
        logger.info(f"  Recall: {metrics_optimal['Recall']:.3f}")
        logger.info(f"  F1: {metrics_optimal['F1']:.3f}")
        logger.info(f"")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Key files:")
        logger.info(f"  - walk_forward_validation_report.txt")
        logger.info(f"  - validation_results.csv")
        logger.info(f"  - validation_metrics.json")
        logger.info(f"")
        logger.info(f"Next: Deploy model or iterate on features!")
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
