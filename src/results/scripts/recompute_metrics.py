"""
Recompute metrics.csv from an existing predictions.csv.

Usage:
    python -m src.results.scripts.recompute_metrics <predictions_csv>

The metrics.csv is written next to the predictions file.
"""

import logging
import sys

import pandas as pd

from src.results.analysis import compute_and_save_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.results.scripts.recompute_metrics <predictions_csv>")
        sys.exit(1)

    predictions_path = sys.argv[1]
    results_df = pd.read_csv(predictions_path)

    output_dir = str(__import__("pathlib").Path(predictions_path).parent)
    metrics_file = compute_and_save_metrics(output_dir=output_dir, results_df=results_df)
    logger.info(f"Done. Metrics written to {metrics_file}")


if __name__ == "__main__":
    main()
