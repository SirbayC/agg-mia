import logging
from typing import List

import pandas as pd

from src.mias.mia_interface import MIAttack

logger = logging.getLogger(__name__)


class TraWiCMIA(MIAttack):
    def __init__(self, model, tokenizer, batch_size: int = 1):
        super().__init__(model=model, tokenizer=tokenizer, batch_size=batch_size)
        self.classifier = None

    @property
    def name(self) -> str:
        return "trawic"

    def train(self, train_df: pd.DataFrame) -> None:
        """
        Train the TraWiC classifier on training data.

        Args:
            train_df: DataFrame with columns ['text', 'blob_id', 'label']
        """
        logger.warning(
            "TraWiCMIA placeholder train() is active. "
            "Wire this to the TraWiC pipeline to train a Random Forest classifier."
        )
        # TODO: Wire to TraWiC pipeline
        # Extract features from texts using the model
        # Train Random Forest classifier on features + labels
        self.classifier = "placeholder_trained"

    def evaluate(self, test_df: pd.DataFrame) -> List[float]:
        """
        Evaluate TraWiC on test data.

        Args:
            test_df: DataFrame with columns ['text', 'blob_id', 'label']

        Returns:
            List of membership scores
        """
        logger.warning(
            "TraWiCMIA placeholder evaluate() is active; returning 0.0 scores. "
            "Wire this to the TraWiC pipeline to use the trained classifier."
        )
        # TODO: Wire to TraWiC pipeline
        # Extract features from test texts
        # Use trained classifier to predict membership probabilities
        return [0.0 for _ in range(len(test_df))]
