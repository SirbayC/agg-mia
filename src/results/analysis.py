import logging
import os

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
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

    # Convert scores to binary predictions using 0.5 threshold
    predictions = (scores_array >= 0.5).astype(int)

    # Compute metrics
    accuracy = accuracy_score(labels_array, predictions)
    precision = precision_score(labels_array, predictions, zero_division=0)
    recall = recall_score(labels_array, predictions, zero_division=0)
    f1 = f1_score(labels_array, predictions, zero_division=0)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(labels_array, scores_array)
    except Exception as e:
        logger.warning(f"Could not compute ROC-AUC: {e}")
        roc_auc = None

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels_array, predictions).ravel()

    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'metric': [
            'accuracy',
            'precision',
            'recall',
            'f1_score',
            'specificity',
            'false_positive_rate',
            'roc_auc',
            'true_positives',
            'true_negatives',
            'false_positives',
            'false_negatives',
        ],
        'value': [
            accuracy,
            precision,
            recall,
            f1,
            specificity,
            fpr,
            roc_auc,
            tp,
            tn,
            fp,
            fn,
        ]
    })

    metrics_file = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Metrics saved to {metrics_file}")

    # Log metrics
    logger.info(f"Binary Classification Metrics (threshold=0.5):")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    if roc_auc is not None:
        logger.info(f"  ROC-AUC:   {roc_auc:.4f}")

    return metrics_file
