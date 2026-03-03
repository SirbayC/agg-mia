from abc import ABC, abstractmethod
from typing import List

class MIAttack(ABC):
    """Abstract base class for Membership Inference Attacks."""

    def __init__(self, model, tokenizer, batch_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the attack."""
        pass

    @abstractmethod
    def compute_scores(self, texts: List[str]) -> List[float]:
        """
        Compute MIA scores for a list of texts.

        Args:
            texts: List of text samples to score

        Returns:
            List of scores (higher score = more likely to be a member)
        """
        pass