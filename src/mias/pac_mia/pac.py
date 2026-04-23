import logging
import random
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.mias.mia_interface import MIAttack
from src.mias.pac_mia.config import PACMIAParams
from src.models.vllm_backend import get_prompt_token_logprobs, is_vllm_model

logger = logging.getLogger(__name__)


def _token_log_probs(text: str, model, tokenizer, max_length: int) -> np.ndarray:
    """Per-token log-probabilities for a single text (sequential, no padding)."""
    try:
        if not is_vllm_model(model):
            raise RuntimeError("PACAttack requires the vLLM backend.")

        return get_prompt_token_logprobs(model, tokenizer, text, max_length)
    except Exception as e:
        logger.warning(f"Error computing token log-probs: {e}")
        return np.array([])


def _compute_polarized_distance(probs: List[float], near_count: int, far_count: int) -> float:
    """Mean of the far_count highest probs minus mean of the near_count lowest probs."""
    if len(probs) == 0:
        return 0.0
    n = len(probs)
    far = max(1, min(far_count, n))
    near = max(1, min(near_count, n))
    if near + far > n:
        scale = n / (near + far)
        near = max(1, int(near * scale))
        far = max(1, int(far * scale))
    sorted_probs = np.sort(probs)
    return float(np.mean(sorted_probs[::-1][:far]) - np.mean(sorted_probs[:near]))


def _generate_mutants(text: str, tokenizer, m_ratio: float, n_samples: int) -> List[str]:
    """Generate adjacent samples by randomly swapping token pairs."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    mutants = []
    for _ in range(n_samples):
        swapped = tokens.copy()
        for _ in range(int(m_ratio * len(swapped))):
            if len(swapped) >= 2:
                i, j = random.sample(range(len(swapped)), 2)
                swapped[i], swapped[j] = swapped[j], swapped[i]
        mutants.append(tokenizer.decode(swapped, skip_special_tokens=True))
    return mutants


def _pac_score(text: str, model, tokenizer, params: PACMIAParams) -> float:
    """PAC score = polarized_distance(original) - mean(polarized_distance(mutants))."""
    mutants = _generate_mutants(text, tokenizer, params.m_ratio, params.n_samples)
    all_probs = [
        _token_log_probs(t, model, tokenizer, params.sequence_length).tolist()
        for t in [text] + mutants
    ]
    original_pd = _compute_polarized_distance(all_probs[0], params.near_count, params.far_count)
    mutant_pds = [
        _compute_polarized_distance(p, params.near_count, params.far_count)
        for p in all_probs[1:]
    ]
    return original_pd - float(np.mean(mutant_pds)) if mutant_pds else 0.0


class PACAttack(MIAttack):
    """
    PAC (Polarized-Augment Calibration) MIA.

    Score = polarized distance of the original text minus the mean polarized
    distance of token-swap mutants. Members score higher because the target
    model assigns a more polarized distribution to memorised content.
    """

    def __init__(self, model, tokenizer, batch_size: int, seed: int):
        super().__init__(model=model, tokenizer=tokenizer, batch_size=batch_size, seed=seed)
        self.params = PACMIAParams()
        logger.info("PACAttack parameters:\n%s", "\n".join(f"  {k}: {v}" for k, v in vars(self.params).items()))

    @property
    def name(self) -> str:
        return "pac"

    def train(self, train_df: pd.DataFrame) -> None:
        """No training required for PAC MIA."""
        pass

    def evaluate(self, test_df: pd.DataFrame) -> List[float]:
        """
        Compute PAC scores for each sample.

        Args:
            test_df: DataFrame with columns ['text', 'blob_id', 'label']

        Returns:
            List of scores (higher = more likely member).
        """
        self.model.eval()
        scores = []
        for text in tqdm(test_df["text"].tolist(), desc="Computing PAC scores"):
            try:
                scores.append(_pac_score(text, self.model, self.tokenizer, self.params))
            except Exception as e:
                logger.warning(f"Error computing PAC score: {e}")
                scores.append(float("nan"))
        return scores
