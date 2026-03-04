import logging
from typing import List

import pandas as pd

from src.mias.mia_interface import MIAttack

logger = logging.getLogger(__name__)


class MIAAdvMIA(MIAttack):
    def __init__(self, model, tokenizer, batch_size: int = 1):
        super().__init__(model=model, tokenizer=tokenizer, batch_size=batch_size)
        self.mlp_classifier = None

    @property
    def name(self) -> str:
        return "miaadv"

    def train(self, train_df: pd.DataFrame) -> None:
        """
        Train MIA-Adv (generate adversarial prompts and train MLP classifier).

        Args:
            train_df: DataFrame with columns ['text', 'blob_id', 'label']
        """
        logger.warning(
            "MIAAdvMIA placeholder train() is active. "
            "Wire this to the mia_adv pipeline to train an MLP on perturbed prompts."
        )
        # TODO: Wire to mia_adv pipeline
        # Generate adversarial perturbations of training samples
        # Extract features from perturbed prompts
        # Train MLP classifier on features + labels
        self.mlp_classifier = "placeholder_trained"

    def evaluate(self, test_df: pd.DataFrame) -> List[float]:
        """
        Evaluate MIA-Adv on test data.

        Args:
            test_df: DataFrame with columns ['text', 'blob_id', 'label']

        Returns:
            List of membership scores
        """
        logger.warning(
            "MIAAdvMIA placeholder evaluate() is active; returning 0.0 scores. "
            "Wire this to the mia_adv pipeline to use the trained MLP."
        )
        # TODO: Wire to mia_adv pipeline
        # Generate adversarial perturbations of test samples
        # Extract features from perturbed prompts
        # Use trained MLP to predict membership probabilities
        return [0.0 for _ in range(len(test_df))]
