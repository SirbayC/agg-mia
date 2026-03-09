import logging
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.mias.mia_interface import MIAttack
from src.mias.bow_mia.config import BoWMIAParams

logger = logging.getLogger(__name__)


class BoWMIA(MIAttack):
    """
    Bag-of-Words distribution-shift detector.

    Trains a TF-IDF + logistic regression classifier solely on surface text
    features — no LLM involved. A high AUC means the seen/unseen split is
    statistically distinguishable without any model knowledge, indicating a
    distribution shift that would confound other MIA results.
    """

    def __init__(self, model, tokenizer, batch_size: int, seed: int):
        super().__init__(model=model, tokenizer=tokenizer, batch_size=batch_size, seed=seed)
        self.params = BoWMIAParams()
        self._clf: Pipeline | None = None
        logger.info("BoWMIA parameters:\n%s", "\n".join(f"  {k}: {v}" for k, v in vars(self.params).items()))

    @property
    def name(self) -> str:
        return "bow"

    def train(self, train_df: pd.DataFrame) -> None:
        """Fit TF-IDF + logistic regression on the training split."""
        logger.info("BoWMIA: fitting TF-IDF + logistic regression on %d samples...", len(train_df))
        p = self.params
        self._clf = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=p.max_features,
                ngram_range=p.ngram_range,
                sublinear_tf=p.sublinear_tf,
            )),
            ("lr", LogisticRegression(
                C=p.C,
                max_iter=p.max_iter,
                solver=p.solver,
                n_jobs=p.n_jobs,
                random_state=self.seed,
            )),
        ])
        self._clf.fit(train_df["text"].tolist(), train_df["label"].tolist())
        logger.info("BoWMIA: training complete.")
        self.log_top_features()

    def log_top_features(self, n: int = 20) -> None:
        """Log the top n ngrams most predictive of seen (member) and unseen (non-member)."""
        if self._clf is None:
            raise RuntimeError("BoWMIA.train() must be called first.")
        feature_names = self._clf.named_steps["tfidf"].get_feature_names_out()
        coefs = self._clf.named_steps["lr"].coef_[0]

        top_seen_idx = coefs.argsort()[-n:][::-1]
        top_unseen_idx = coefs.argsort()[:n]

        logger.info("Top %d features predicting SEEN (member):", n)
        for i in top_seen_idx:
            logger.info("  %+.4f  %s", coefs[i], feature_names[i])

        logger.info("Top %d features predicting UNSEEN (non-member):", n)
        for i in top_unseen_idx:
            logger.info("  %+.4f  %s", coefs[i], feature_names[i])

    def evaluate(self, test_df: pd.DataFrame) -> List[float]:
        """
        Return the classifier's predicted probability of class 1 (seen/member)
        for each sample, using only surface text features.

        Args:
            test_df: DataFrame with columns ['text', 'blob_id', 'label']

        Returns:
            List of scores in [0, 1] (higher = more likely member per BoW).
        """
        if self._clf is None:
            raise RuntimeError("BoWMIA.train() must be called before evaluate().")
        probs = self._clf.predict_proba(test_df["text"].tolist())
        return probs[:, 1].tolist()
