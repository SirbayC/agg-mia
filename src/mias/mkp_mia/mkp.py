import logging
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm

from src.mias.mia_interface import MIAttack
from src.mias.mkp_mia.config import MKPMIAParams

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
        self.model.eval()
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
        inputs = self.tokenizer(
            text,
            max_length=self.params.sequence_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            lp = log_softmax(logits, dim=-1)

        token_log_probs = []
        for i in range(inputs["input_ids"].shape[1] - 1):
            token_id = inputs["input_ids"][0, i + 1]
            token_log_probs.append(lp[0, i, token_id].item())

        return np.array(token_log_probs)

    def _token_probs_sliding(self, text: str) -> np.ndarray:
        """Per-token log-probs via non-overlapping sliding window (scores every token once)."""
        input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids[0]
        window = self.params.sequence_length
        token_log_probs = []

        for i in range(0, len(input_ids), window):
            chunk = input_ids[i : i + window]
            if len(chunk) < 2:
                continue
            chunk_tensor = chunk.unsqueeze(0).to(self.model.device)
            with torch.no_grad():
                logits = self.model(chunk_tensor).logits
                lp = log_softmax(logits, dim=-1)
            for j in range(len(chunk) - 1):
                token_log_probs.append(lp[0, j, chunk[j + 1]].item())

        return np.array(token_log_probs)
