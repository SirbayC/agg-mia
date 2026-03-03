import logging
from typing import List

from src.mias.mia_interface import MIAttack

logger = logging.getLogger(__name__)


class TraWiCMIA(MIAttack):
    def __init__(self, model, tokenizer, batch_size: int = 1):
        super().__init__(model=model, tokenizer=tokenizer, batch_size=batch_size)

    @property
    def name(self) -> str:
        return "trawic"

    def compute_scores(self, texts: List[str]) -> List[float]:
        logger.warning(
            "TraWiCMIA placeholder is active; returning 0.0 scores. "
            "Wire this class to the TraWiC pipeline next."
        )
        return [0.0 for _ in texts]
