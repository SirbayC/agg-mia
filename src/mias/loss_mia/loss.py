import logging
from typing import List

import pandas as pd
from tqdm import tqdm

from src.mias.mia_interface import MIAttack
from src.mias.loss_mia.config import LossMIAParams
from src.models.vllm_backend import get_prompt_token_logprobs, is_vllm_model

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

        if not is_vllm_model(self.model):
            raise RuntimeError("LossMIA requires the vLLM backend.")

        for text in tqdm(texts, desc="Computing loss scores"):
            try:
                token_log_probs = get_prompt_token_logprobs(
                    self.model,
                    self.tokenizer,
                    text,
                    self.params.sequence_length,
                )
                scores.append(float(token_log_probs.mean()) if len(token_log_probs) else float("nan"))
            except Exception as e:
                logger.warning(f"Error computing loss for sample: {e}")
                scores.append(float("nan"))

        return scores
