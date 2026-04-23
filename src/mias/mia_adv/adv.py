import logging
import random
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
from src.mias.mia_adv.classifier import init_feature_models, compute_features, CustomMLP, train_mlp
from src.models.vllm_backend import generate_text, is_vllm_model

logger = logging.getLogger(__name__)

# Fixed ordering of the 11 perturbation keys (matches traverse_all_variants output)
_PERTURBATION_KEYS = list(PERTURBATIONS.keys())

class AdvMIA(MIAttack):
    def __init__(self, model, tokenizer, batch_size: int, seed: int):
        super().__init__(model=model, tokenizer=tokenizer, batch_size=batch_size, seed=seed)
        self.params = MIAAdvParams()
        self.mlp_classifier = None
        self.scaler = None
        logger.info("MIAAdv parameters:\n%s", "\n".join(f"  {k}: {v}" for k, v in vars(self.params).items()))
        logger.info("MIAAdv: loading CodeBERT and CodeGPT feature models...")
        init_feature_models(target_device=getattr(model, "device", None))
        logger.info("MIAAdv: feature models ready.")
        # Keep the tail of long prefixes so the model sees the most recent context
        self.tokenizer.truncation_side = 'left'

    @property
    def name(self) -> str:
        return "miaadv"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_perturbations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split each text into a prefix (~40-60% of chars) and a ground truth
        continuation. Perturbations are applied only to the prefix, so the
        model is prompted with the prefix and its output is compared against
        the ground truth continuation.

        Adds columns: prefix, gt, text_0 (= prefix), text_1..text_11.
        """
        df = df.copy()

        prefixes, gts = [], []
        for text in df['text']:
            split_idx = int(len(text) * random.uniform(0.4, 0.6))
            prefixes.append(text[:split_idx])
            gts.append(text[split_idx:])

        df['prefix'] = prefixes
        df['gt'] = gts
        df['text_0'] = df['prefix']

        for idx, row in df.iterrows():
            variants = traverse_all_variants(row['prefix'])
            for j, key in enumerate(_PERTURBATION_KEYS, start=1):
                df.at[idx, f'text_{j}'] = variants[key]

        return df

    def _generate_single(self, text: str) -> str:
        """Greedy decode on one text, returning only the newly generated tokens."""
        if not is_vllm_model(self.model):
            raise RuntimeError("MIAAdv requires the vLLM backend.")

        newline_token_id = self.tokenizer.encode('\n', add_special_tokens=False)[-1]
        break_ids = [self.tokenizer.eos_token_id, newline_token_id]
        if self.tokenizer.sep_token_id is not None:
            break_ids.append(self.tokenizer.sep_token_id)

        generated_text = generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=text,
            max_input_tokens=self.params.max_input_tokens,
            max_generated_tokens=self.params.max_new_tokens,
            temperature=self.params.temperature,
            top_p=1.0,
            do_sample=self.params.do_sample,
            top_k=self.params.top_k,
            stop_token_ids=break_ids,
        )
        return generated_text

    def _generate_outputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run model inference for every text_0..text_11 column.
        Returns a copy of df with added output_text_0..output_text_11 columns.
        """
        output_df = df.copy()
        for idx, row in tqdm(df.iterrows(), desc="Generating outputs", total=len(df)):
            for col_idx in range(12):
                text = row[f'text_{col_idx}']
                output = self._generate_single(text)
                output_df.at[idx, f'output_text_{col_idx}'] = output
                logger.debug(
                    "Sample %s | variant %d\n  INPUT:  %s\n  OUTPUT: %s",
                    idx, col_idx, text, output,
                )
        return output_df

    def _extract_features(self, output_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a feature vector per row via compute_features().
          - ground truth  : gt column (the held-out continuation after the prefix)
          - base output   : output_text_0  (model output for unperturbed prefix)
          - perturbed outs: output_text_1..output_text_11
        """
        features = []
        labels = []
        for _, row in output_df.iterrows():
            gt = row['gt']
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
