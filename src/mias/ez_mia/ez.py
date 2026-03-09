import logging
from typing import List

import pandas as pd

from src.mias.mia_interface import MIAttack

logger = logging.getLogger(__name__)


class EZMIA(MIAttack):
    def __init__(self, model, tokenizer, batch_size: int = 1, seed: int = 42):
        super().__init__(model=model, tokenizer=tokenizer, batch_size=batch_size, seed=seed)
        self.reference_model = None

    @property
    def name(self) -> str:
        return "ezmia"

    def train(self, train_df: pd.DataFrame) -> None:
        """
        Train EZ-MIA (build reference model if needed).

        Args:
            train_df: DataFrame with columns ['text', 'blob_id', 'label']
        """
        logger.warning(
            "EZMIAMia placeholder train() is active. "
            "Wire this to the EZ-MIA pipeline to build/train a reference model."
        )
        # TODO: Wire to EZ-MIA pipeline
        # May need to train a reference model on non-members
        # or use distillation from target model
        self.reference_model = "placeholder_reference"

    def evaluate(self, test_df: pd.DataFrame) -> List[float]:
        """
        Evaluate EZ-MIA on test data.

        Args:
            test_df: DataFrame with columns ['text', 'blob_id', 'label']

        Returns:
            List of membership scores
        """
        logger.warning(
            "EZMIAMia placeholder evaluate() is active; returning 0.0 scores. "
            "Wire this to the EZ-MIA pipeline to compute error zone scores."
        )
        # TODO: Wire to EZ-MIA pipeline
        # Compute log-probabilities using target and reference models
        # Calculate error zone scores (positive - negative log-prob differences)
        return [0.0 for _ in range(len(test_df))]
