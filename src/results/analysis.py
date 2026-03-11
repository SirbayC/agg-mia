import logging
import os

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def save_predictions(
    output_dir: str,
    results_df: pd.DataFrame,
) -> str:
    """
    Save predictions (blob_id, label, score) to CSV.

    Args:
        output_dir: Output directory path
        results_df: DataFrame with columns ['blob_id', 'label', 'score']

    Returns:
        Path to the saved predictions CSV file
    """
    predictions_file = os.path.join(output_dir, "predictions.csv")
    results_df.to_csv(predictions_file, index=False)
    logger.info(f"Predictions saved to {predictions_file}")

    return predictions_file


def compute_and_save_metrics(
    output_dir: str,
    results_df: pd.DataFrame,
) -> str:
    """
    Compute binary classification metrics and save to CSV.

    Args:
        output_dir: Output directory path
        results_df: DataFrame with columns ['label', 'score']

    Returns:
        Path to the saved metrics CSV file
    """
    labels_array = np.array(results_df['label'])
    scores_array = np.array(results_df['score'])

    # ROC-based metrics required by downstream reporting.
    try:
        roc_auc = roc_auc_score(labels_array, scores_array)
        fpr_curve, tpr_curve, _ = roc_curve(labels_array, scores_array)
        tpr_at_1pct_fpr = float(np.interp(0.01, fpr_curve, tpr_curve))
        tpr_at_0_1pct_fpr = float(np.interp(0.001, fpr_curve, tpr_curve))
    except Exception as e:
        logger.warning(f"Could not compute ROC-based metrics: {e}")
        roc_auc = None
        tpr_at_1pct_fpr = None
        tpr_at_0_1pct_fpr = None

    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'metric': [
            'roc auc',
            'TPR @ 1% FPR',
            'TPR @ 0.1% FPR',
        ],
        'value': [
            roc_auc,
            tpr_at_1pct_fpr,
            tpr_at_0_1pct_fpr,
        ]
    })

    metrics_file = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Metrics saved to {metrics_file}")

    # Log metrics
    logger.info("ROC Metrics:")
    if roc_auc is not None:
        logger.info(f"  ROC AUC:         {roc_auc:.4f}")
    if tpr_at_1pct_fpr is not None:
        logger.info(f"  TPR @ 1% FPR:    {tpr_at_1pct_fpr:.4f}")
    if tpr_at_0_1pct_fpr is not None:
        logger.info(f"  TPR @ 0.1% FPR:  {tpr_at_0_1pct_fpr:.4f}")

    return metrics_file
