import logging
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.mias.mia_interface import MIAttack
from src.mias.loss_mia.config import LossMIAParams

logger = logging.getLogger(__name__)


class LossMIA(MIAttack):
    """Loss-based MIA: lower loss (higher negative loss) indicates membership."""

    def __init__(self, model, tokenizer, batch_size: int, seed: int):
        super().__init__(model=model, tokenizer=tokenizer, batch_size=batch_size, seed=seed)
        self.params = LossMIAParams()
        logger.info("LossMIA parameters:\n%s", "\n".join(f"  {k}: {v}" for k, v in vars(self.params).items()))

    @property
    def name(self) -> str:
        return "loss"

    def train(self, train_df: pd.DataFrame) -> None:
        """No training required for loss-based MIA."""
        pass

    def evaluate(self, test_df: pd.DataFrame) -> List[float]:
        """
        Compute negative loss scores for each sample.

        Args:
            test_df: DataFrame with columns ['text', 'blob_id', 'label']

        Returns:
            List of scores (higher = more likely member).
        """
        texts = test_df["text"].tolist()
        scores = []

        self.model.eval()
        for text in tqdm(texts, desc="Computing loss scores"):
            try:
                inputs = self.tokenizer(
                    text,
                    max_length=self.params.sequence_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"].clone())

                # Negative loss: higher score = lower loss = more likely member
                scores.append(-outputs.loss.item())
            except Exception as e:
                logger.warning(f"Error computing loss for sample: {e}")
                scores.append(float("nan"))

        return scores
