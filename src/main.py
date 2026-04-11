"""
Main entry point for AGG-MIA experiments.
Handles argument parsing, data loading, and MIA execution.
"""

import argparse
import logging
import os
import random
import sys
import traceback
from typing import Type

import numpy as np
import pandas as pd

from src.models.loader import load_model_and_tokenizer
from src.datasets.data_loader import load_data
from src.mias.mia_interface import MIAttack
from src.results.analysis import save_predictions, compute_and_save_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        logger.warning("Torch not available while setting seed; continuing with Python/NumPy seed only")


def parse_args():
    parser = argparse.ArgumentParser(
        description="AGG-MIA: Membership Inference Attack on Code Models"
    )

    # Model and dataset arguments
    parser.add_argument(
        "--mia",
        type=str,
        choices=["trawic", "miaadv", "loss", "mkp", "pac", "bow"],
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used across Python, NumPy, PyTorch, and data splitting/sampling",
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
        from src.mias.trawic.trawic import TraWiCMIA
        return TraWiCMIA
    elif mia_name == "miaadv":
        from src.mias.mia_adv.adv import AdvMIA
        return AdvMIA
    elif mia_name == "loss":
        from src.mias.loss_mia.loss import LossMIA
        return LossMIA
    elif mia_name == "mkp":
        from src.mias.mkp_mia.mkp import MinKProbMIA
        return MinKProbMIA
    elif mia_name == "pac":
        from src.mias.pac_mia.pac import PACAttack
        return PACAttack
    elif mia_name == "bow":
        from src.mias.bow_mia.bow import BoWMIA
        return BoWMIA
    else:
        raise ValueError(f"Unknown MIA: {mia_name}")


def main():
    args = parse_args()

    set_global_seed(args.seed)

    logger.info(f"Starting AGG-MIA experiment")
    logger.info("Experiment parameters:\n%s", "\n".join(f"  {k}: {v}" for k, v in vars(args).items()))

    # Load data
    logger.info("Loading data...")
    try:
        train_df, test_df = load_data(
            data_dir=args.data_dir,
            sample_fraction=args.sample_fraction,
            train_fraction=args.train_test_split,
            seed=args.seed,
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

    # Load target model once and pass it into MIA (skipped for model-free attacks)
    logger.info(f"Loading {args.mia} MIA class to check model requirement...")
    MIAClass = load_mia_class(args.mia)

    model, tokenizer = None, None
    if MIAClass.requires_model:
        logger.info("Loading target model and tokenizer...")
        try:
            model, tokenizer = load_model_and_tokenizer(args.model)
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model/tokenizer: {e}")
            traceback.print_exc()
            return 1
    else:
        logger.info("MIA does not require a model — skipping model load.")

    # Instantiate MIA
    logger.info(f"Instantiating {args.mia} MIA...")
    try:
        mia = MIAClass(
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        logger.info(f"MIA loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load MIA: {e}")
        traceback.print_exc()
        return 1

    # Train MIA on training set
    logger.info("Training MIA on training set...")
    try:
        mia.train(train_df)
        logger.info("MIA training completed")
    except Exception as e:
        logger.error(f"Failed to train MIA: {e}")
        traceback.print_exc()
        return 1

    # Evaluate MIA on test set
    logger.info("Evaluating MIA on test set...")
    try:
        test_scores = mia.evaluate(test_df)
        logger.info(f"Scores computed: {len(test_scores)} scores")

        # Create results dataframe with blob_id, label, score
        results_df = pd.DataFrame({
            'blob_id': test_df['blob_id'],
            'label': test_df['label'],
            'score': test_scores
        })

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Created output directory: {args.output_dir}")

        # Save predictions
        save_predictions(
            output_dir=args.output_dir,
            results_df=results_df,
        )

        # Compute and save metrics
        compute_and_save_metrics(
            output_dir=args.output_dir,
            results_df=results_df,
        )

    except Exception as e:
        logger.error(f"Failed to compute scores: {e}")
        traceback.print_exc()
        return 1

    logger.info("Experiment completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
