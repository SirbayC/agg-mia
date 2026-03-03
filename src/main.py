"""
Main entry point for AGG-MIA experiments.
Handles argument parsing, data loading, and MIA execution.
"""

import argparse
import logging
import os
import sys
import traceback
from typing import Type

from src.models.loader import load_model_and_tokenizer
from src.datasets.data_loader import load_data
from src.mias.mia_interface import MIAttack

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AGG-MIA: Membership Inference Attack on Code Models"
    )

    # Model and dataset arguments
    parser.add_argument(
        "--mia",
        type=str,
        choices=["trawic", "ezmia", "miaadv"],
        required=True,
        help="Which MIA method to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Target model hf id",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to data directory (contains seen/ and unseen/ subdirs with parquet files)",
    )

    # Data loading arguments
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="Fraction of data to load (0.0 to 1.0). E.g., 0.5 loads only 50%% of samples",
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (rest goes to test)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results",
    )

    return parser.parse_args()


def load_mia_class(mia_name: str) -> Type[MIAttack]:
    """Dynamically load the appropriate MIA class."""
    if mia_name == "trawic":
        from src.mias.trawic.trawic_mia import TraWiCMIA
        return TraWiCMIA
    elif mia_name == "ezmia":
        from src.mias.ez_mia.ez_mia_mia import EZMIAMia
        return EZMIAMia
    elif mia_name == "miaadv":
        from src.mias.mia_adv.mia_adv_mia import MIAAdvMIA
        return MIAAdvMIA
    else:
        raise ValueError(f"Unknown MIA: {mia_name}")


def main():
    args = parse_args()

    logger.info(f"Starting AGG-MIA experiment")
    logger.info(f"  MIA: {args.mia}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Sample fraction: {args.sample_fraction}")
    logger.info(f"  Train/test split: {args.train_test_split}")


    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Load data
    logger.info("Loading data...")
    try:
        train_df, test_df = load_data(
            data_dir=args.data_dir,
            sample_fraction=args.sample_fraction,
            train_test_split=args.train_test_split,
        )
        logger.info(
            f"Data loaded successfully:"
            f"\n  Train: {len(train_df)} samples "
            f"({(train_df['label'] == 1).sum()} seen, {(train_df['label'] == 0).sum()} unseen)"
            f"\n  Test: {len(test_df)} samples "
            f"({(test_df['label'] == 1).sum()} seen, {(test_df['label'] == 0).sum()} unseen)"
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        traceback.print_exc()
        return 1

    # Load target model once and pass it into MIA
    logger.info("Loading target model and tokenizer...")
    try:
        model, tokenizer = load_model_and_tokenizer(args.model)
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {e}")
        traceback.print_exc()
        return 1

    # Load MIA class
    logger.info(f"Loading {args.mia} MIA...")
    try:
        MIAClass = load_mia_class(args.mia)
        mia = MIAClass(
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
        )
        logger.info(f"MIA loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load MIA: {e}")
        traceback.print_exc()
        return 1

    # Compute scores on test set
    logger.info("Computing MIA scores on test set...")
    try:
        test_scores = mia.compute_scores(test_df['text'].tolist())
        logger.info(f"Scores computed: {len(test_scores)} scores")

        # Add scores to test dataframe
        test_df['score'] = test_scores

        # Save results as CSV
        results_file = os.path.join(args.output_dir, "results.csv")
        test_df.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")

    except Exception as e:
        logger.error(f"Failed to compute scores: {e}")
        traceback.print_exc()
        return 1

    logger.info("Experiment completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
