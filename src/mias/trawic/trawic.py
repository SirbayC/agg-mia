import logging
from typing import List

import pandas as pd
import torch 
from sklearn.ensemble import RandomForestClassifier 
from tqdm import tqdm 

from src.mias.mia_interface import MIAttack
from src.mias.trawic.config import TraWiCParams
from src.mias.trawic.feature_extractor import extract_features

logger = logging.getLogger(__name__)

class TraWiCMIA(MIAttack):
    def __init__(self, model, tokenizer, batch_size: int, seed: int):
        super().__init__(model=model, tokenizer=tokenizer, batch_size=batch_size, seed=seed)
        self.classifier = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.params = TraWiCParams()
        logger.info("TraWiC parameters:\n%s", "\n".join(f"  {k}: {v}" for k, v in vars(self.params).items()))

    @property
    def name(self) -> str:
        return "trawic"

    def train(self, train_df: pd.DataFrame) -> None:
        """
        Train the TraWiC classifier on training data.

        Args:
            train_df: DataFrame with columns ['text', 'blob_id', 'label']
        """
        logger.info(f"Training TraWiC MIA on {len(train_df)} samples...")
        
        # Extract features for all training samples
        feature_list = []
        labels = []
        
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting features"):
            try:
                features = extract_features(
                    code=row['text'],
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    params=self.params
                )
                feature_list.append(list(features.values()))
                labels.append(row['label'])
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                # Add zero features for failed samples
                feature_list.append([0.0] * 12)
                labels.append(row['label'])
        
        # Train Random Forest classifier
        logger.info("Training Random Forest classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=self.params.n_estimators,
            max_depth=self.params.max_depth,
            max_features=self.params.max_features,
            criterion=self.params.criterion,
            random_state=self.params.random_state,
            n_jobs=self.params.n_jobs
        )
        self.classifier.fit(feature_list, labels)
        logger.info("TraWiC classifier training completed")

    def evaluate(self, test_df: pd.DataFrame) -> List[float]:
        """
        Evaluate TraWiC on test data.

        Args:
            test_df: DataFrame with columns ['text', 'blob_id', 'label']

        Returns:
            List of membership scores (probability of being a member)
        """
        if self.classifier is None:
            logger.error("Classifier not trained! Call train() first.")
            return [0.0] * len(test_df)
        
        logger.info(f"Evaluating TraWiC MIA on {len(test_df)} samples...")
        
        # Extract features for all test samples
        feature_list = []
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Extracting features"):
            try:
                features = extract_features(
                    code=row['text'],
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    params=self.params
                )
                feature_list.append(list(features.values()))
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                feature_list.append([0.0] * 12)
        
        # Predict membership probabilities
        logger.info("Predicting membership probabilities...")
        proba = self.classifier.predict_proba(feature_list)
        
        # Return probability of class 1 (member)
        scores = [p[1] for p in proba]
        logger.info("TraWiC evaluation completed")
        
        return scores
