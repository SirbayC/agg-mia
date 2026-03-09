from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class MIAttack(ABC):
    """Abstract base class for Membership Inference Attacks."""

    def __init__(self, model, tokenizer, batch_size: int, seed: int):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seed = seed

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the attack."""
        pass

    @abstractmethod
    def train(self, train_df: pd.DataFrame) -> None:
        """
        Train the MIA (e.g., train a classifier).

        Args:
            train_df: DataFrame with columns ['text', 'blob_id', 'label']
                     where label is 1 for seen (member) and 0 for unseen (non-member)
        """
        pass

    @abstractmethod
    def evaluate(self, test_df: pd.DataFrame) -> List[float]:
        """
        Evaluate the MIA on test samples.

        Args:
            test_df: DataFrame with columns ['text', 'blob_id', 'label']

        Returns:
            List of scores (higher score = more likely to be a member)
        """
        pass