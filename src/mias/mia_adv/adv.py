import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.mias.mia_interface import MIAttack
from src.mias.mia_adv.config import MIAAdvParams
from src.mias.mia_adv.perturb import traverse_all_variants, PERTURBATIONS
from src.mias.mia_adv.classifier import compute_features, CustomMLP, train_mlp

logger = logging.getLogger(__name__)

# Fixed ordering of the 11 perturbation keys (matches traverse_all_variants output)
_PERTURBATION_KEYS = list(PERTURBATIONS.keys())

class AdvMIA(MIAttack):
    def __init__(self, model, tokenizer, batch_size: int = 1, seed: int = 42):
        super().__init__(model=model, tokenizer=tokenizer, batch_size=batch_size, seed=seed)
        self.params = MIAAdvParams()
        self.mlp_classifier = None
        self.scaler = None
        logger.info("MIAAdv parameters:\n%s", "\n".join(f"  {k}: {v}" for k, v in vars(self.params).items()))

    @property
    def name(self) -> str:
        return "miaadv"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_perturbations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of df with columns text_0 (original) and text_1..text_11
        (one column per perturbation variant).
        """
        df = df.copy()
        df['text_0'] = df['text']
        for idx, row in df.iterrows():
            variants = traverse_all_variants(row['text'])
            for j, key in enumerate(_PERTURBATION_KEYS, start=1):
                df.at[idx, f'text_{j}'] = variants[key]
        return df

    def _generate_single(self, text: str) -> str:
        """Greedy decode on one text, returning only the newly generated tokens."""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.params.max_input_tokens,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        newline_token_id = self.tokenizer.encode('\n', add_special_tokens=False)[-1]
        break_ids = [self.tokenizer.eos_token_id, newline_token_id]
        if self.tokenizer.sep_token_id is not None:
            break_ids.append(self.tokenizer.sep_token_id)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.params.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
        
                eos_token_id=break_ids,
                early_stopping=True,
                do_sample=True,      
                top_k=50,            
                temperature=0.8      
            )

        input_len = inputs['input_ids'].shape[1]
        return self.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

    def _generate_outputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run model inference for every text_0..text_11 column.
        Returns a copy of df with added output_text_0..output_text_11 columns.
        """
        output_df = df.copy()
        total = len(df)
        for col_idx in range(12):
            col = f'text_{col_idx}'
            outputs = []
            for text in tqdm(df[col], desc=f"Generating variant {col_idx}", total=total):
                outputs.append(self._generate_single(text))
            output_df[f'output_text_{col_idx}'] = outputs
        return output_df

    def _extract_features(self, output_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a feature vector per row via compute_features().
          - ground truth  : original text (text column)
          - base output   : output_text_0  (model output for unperturbed input)
          - perturbed outs: output_text_1..output_text_11
        """
        features = []
        labels = []
        for _, row in output_df.iterrows():
            gt = row['text']
            y_pred = row['output_text_0']
            y_preds_perturbed = [row[f'output_text_{i}'] for i in range(1, 12)]
            features.append(compute_features(gt, y_pred, y_preds_perturbed))
            labels.append(int(row['label']))
        return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)

    # ------------------------------------------------------------------
    # MIAttack interface
    # ------------------------------------------------------------------

    def train(self, train_df: pd.DataFrame) -> None:
        """
        Full MIA-Adv training pipeline:
          1. Generate 11 perturbations per sample (text_0..text_11 columns).
          2. Run StarCoder2 inference on each variant (output_text_0..output_text_11).
          3. Extract feature vectors via CodeBERT similarity + CodeGPT perplexity.
          4. Fit a StandardScaler and train a 3-layer MLP binary classifier.

        Args:
            train_df: DataFrame with columns ['text', 'blob_id', 'label']
        """
        logger.info("MIAAdv train: applying perturbations to %d samples...", len(train_df))
        perturbed_df = self._apply_perturbations(train_df)

        logger.info("MIAAdv train: running model inference (12 variants x %d samples)...", len(train_df))
        output_df = self._generate_outputs(perturbed_df)

        logger.info("MIAAdv train: extracting features...")
        X, y = self._extract_features(output_df)

        # Stratified split to balance classes and produce a validation set,
        # matching the original MIA_Adv training protocol.
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.params.val_split,
            stratify=y,
            random_state=self.seed,
        )
        logger.info(
            "MIAAdv train: stratified split -> train=%d, val=%d (pos: %d/%d)",
            len(y_train), len(y_val), y_train.sum(), y_val.sum(),
        )

        self.scaler = StandardScaler().fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        X_t = torch.from_numpy(X_train_scaled)
        y_t = torch.from_numpy(y_train)
        X_v = torch.from_numpy(X_val_scaled)
        y_v = torch.from_numpy(y_val)

        input_dim = X_train_scaled.shape[1]
        self.mlp_classifier = CustomMLP(
            input_dim=input_dim,
            hidden_dims=self.params.hidden_dims,
            num_classes=self.params.num_classes,
            dropout=self.params.dropout,
        )

        logger.info("MIAAdv train: training MLP (input_dim=%d)...", input_dim)
        train_mlp(self.mlp_classifier, X_t, y_t, X_val=X_v, y_val=y_v, batch_size=self.params.batch_size, lr=self.params.lr, num_epochs=self.params.num_epochs)
        logger.info("MIAAdv train: done.")

    def evaluate(self, test_df: pd.DataFrame) -> List[float]:
        """
        Evaluate MIA-Adv on test samples.

        Args:
            test_df: DataFrame with columns ['text', 'blob_id', 'label']

        Returns:
            List of membership scores (probability of being a member, range [0, 1]).
        """
        if self.mlp_classifier is None or self.scaler is None:
            raise RuntimeError("MIAAdvMIA.train() must be called before evaluate()")

        logger.info("MIAAdv evaluate: applying perturbations to %d samples...", len(test_df))
        perturbed_df = self._apply_perturbations(test_df)

        logger.info("MIAAdv evaluate: running model inference...")
        output_df = self._generate_outputs(perturbed_df)

        logger.info("MIAAdv evaluate: extracting features...")
        X, _ = self._extract_features(output_df)

        X_scaled = self.scaler.transform(X)
        mlp_device = next(self.mlp_classifier.parameters()).device
        X_t = torch.from_numpy(X_scaled).to(mlp_device)

        self.mlp_classifier.eval()
        with torch.no_grad():
            probs = F.softmax(self.mlp_classifier(X_t), dim=1)[:, 1]
        return probs.cpu().numpy().tolist()
