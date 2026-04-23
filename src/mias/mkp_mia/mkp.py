import logging
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.mias.mia_interface import MIAttack
from src.mias.mkp_mia.config import MKPMIAParams
from src.models.vllm_backend import get_prompt_token_logprobs, is_vllm_model

logger = logging.getLogger(__name__)


class MinKProbMIA(MIAttack):
    """
    Min-K% Prob Attack: membership score is the mean log-probability
    of the k% least-likely tokens. Members tend to have higher scores
    (less surprising tokens) than non-members.
    """

    def __init__(self, model, tokenizer, batch_size: int, seed: int):
        super().__init__(model=model, tokenizer=tokenizer, batch_size=batch_size, seed=seed)
        self.params = MKPMIAParams()
        logger.info("MinKProbMIA parameters:\n%s", "\n".join(f"  {k}: {v}" for k, v in vars(self.params).items()))

    @property
    def name(self) -> str:
        return "mkp"

    def train(self, train_df: pd.DataFrame) -> None:
        """No training required for Min-K% Prob MIA."""
        pass

    def evaluate(self, test_df: pd.DataFrame) -> List[float]:
        """
        Compute Min-K% Prob scores for each sample.

        Args:
            test_df: DataFrame with columns ['text', 'blob_id', 'label']

        Returns:
            List of scores (higher = more likely member).
        """
        texts = test_df["text"].tolist()
        all_sorted_probs = self._get_token_probs(texts)
        return [self._score(probs) for probs in all_sorted_probs]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_token_probs(self, texts: List[str]) -> List[np.ndarray]:
        """Return per-text arrays of sorted (ascending) token log-probabilities."""
        if not is_vllm_model(self.model):
            raise RuntimeError("MinKProbMIA requires the vLLM backend.")

        all_probs = []
        for text in tqdm(texts, desc="Computing Min-K% token log-probs"):
            if self.params.use_sliding_window:
                probs = self._token_probs_sliding(text)
            else:
                probs = self._token_probs_truncation(text)
            all_probs.append(np.sort(probs))  # ascending: lowest probs first
        return all_probs

    def _score(self, sorted_probs: np.ndarray) -> float:
        """Mean of the k% lowest log-probabilities (the 'error zone')."""
        if len(sorted_probs) == 0:
            return float("nan")
        k_length = max(1, int(len(sorted_probs) * self.params.k))
        return float(np.mean(sorted_probs[:k_length]))

    def _token_probs_truncation(self, text: str) -> np.ndarray:
        """Per-token log-probs with simple truncation."""
        return get_prompt_token_logprobs(self.model, self.tokenizer, text, self.params.sequence_length)

    def _token_probs_sliding(self, text: str) -> np.ndarray:
        """Per-token log-probs via non-overlapping sliding window (scores every token once)."""
        token_ids = self.tokenizer(text, add_special_tokens=True).input_ids
        window = self.params.sequence_length
        token_log_probs = []

        for start_index in range(0, len(token_ids), window):
            chunk_ids = token_ids[start_index : start_index + window]
            if len(chunk_ids) < 2:
                continue
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=False)
            chunk_probs = get_prompt_token_logprobs(
                self.model,
                self.tokenizer,
                chunk_text,
                window,
            )
            token_log_probs.extend(chunk_probs.tolist())

        return np.array(token_log_probs)
