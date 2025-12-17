"""
scripts/utils/metrics.py
Model evaluation and metrics computation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, auc, precision_recall_curve, confusion_matrix,
    matthews_corrcoef, roc_curve
)
from sklearn.model_selection import cross_validate
import logging

logger = logging.getLogger(__name__)


def compute_evaluation_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Compute comprehensive evaluation metrics for classification.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_pred_proba (array): Predicted probabilities (for ROC-AUC, PR-AUC)
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
    }
    
    # Probability-based metrics
    if y_pred_proba is not None:
        # Ensure 2D array and take positive class probabilities
        if len(y_pred_proba.shape) == 1:
            y_pred_proba_pos = y_pred_proba
        else:
            y_pred_proba_pos = y_pred_proba[:, 1]
        
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba_pos)
        
        # Precision-Recall AUC
        precision, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba_pos)
        metrics['pr_auc'] = auc(recall_vals, precision)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['tp'] = int(tp)
    
    return metrics


def compute_cross_validation_scores(model, X, y, cv, scoring='roc_auc'):
    """
    Compute cross-validation scores.
    
    Args:
        model: Scikit-learn compatible model
        X (array): Features
        y (array): Labels
        cv: Cross-validation splitter
        scoring (str): Scoring metric
    
    Returns:
        dict: CV scores summary
    """
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        return_train_score=True
    )
    
    summary = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        test_key = f"test_{metric}"
        train_key = f"train_{metric}"
        
        summary[metric] = {
            'test_mean': scores[test_key].mean(),
            'test_std': scores[test_key].std(),
            'train_mean': scores[train_key].mean(),
            'test_scores': scores[test_key].tolist(),
        }
    
    logger.info(f"Cross-validation complete: {len(scores['test_accuracy'])} folds")
    return summary


def generate_confusion_matrix(y_true, y_pred):
    """
    Generate detailed confusion matrix report.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
    
    Returns:
        pd.DataFrame: Confusion matrix as dataframe
    """
    cm = confusion_matrix(y_true, y_pred)
    
    df_cm = pd.DataFrame(
        cm,
        index=['Actual Non-Recession', 'Actual Recession'],
        columns=['Predicted Non-Recession', 'Predicted Recession']
    )
    
    return df_cm


def compute_roc_curve(y_true, y_pred_proba):
    """Compute ROC curve coordinates."""
    if len(y_pred_proba.shape) > 1:
        y_pred_proba = y_pred_proba[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc,
    }


def compute_pr_curve(y_true, y_pred_proba):
    """Compute Precision-Recall curve coordinates."""
    if len(y_pred_proba.shape) > 1:
        y_pred_proba = y_pred_proba[:, 1]
    
    precision, recall_vals, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_vals, precision)
    
    return {
        'precision': precision,
        'recall': recall_vals,
        'thresholds': thresholds,
        'auc': pr_auc,
    }
