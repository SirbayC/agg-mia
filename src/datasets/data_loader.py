"""
Data loader for AGG-MIA experiments.
Loads parquet files from data/seen and data/unseen directories.
Handles sampling and train/test splitting.
"""

import logging
import os
import random
from typing import List, Tuple

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def find_first_parquet_file(directory: str) -> str:
    """Find the first (alphabetically) parquet file in a directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    parquet_files = sorted([f for f in os.listdir(directory) if f.endswith(".parquet")])
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {directory}")

    path = os.path.join(directory, parquet_files[0])
    logger.info(f"Found parquet file: {path}")
    return path


def load_parquet_samples(
    file_path: str,
    sample_fraction: float = 1.0,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Load samples from a parquet file.

    Args:
        file_path: Path to parquet file
        sample_fraction: Fraction of data to load (0.0 to 1.0)
        seed: Random seed for deterministic sampling

    Returns:
        Tuple of (texts, blob_ids) where texts are code content strings and blob_ids are identifiers
    """
    logger.info(f"Loading parquet file: {file_path}")

    ds = load_dataset("parquet", data_files=file_path, split="train")
    logger.info(f"Total samples in file: {len(ds)}")

    # Get all samples
    samples = []
    blob_ids = []
    for row in ds:
        content = row.get("content", "")
        # Only include samples with non-empty content
        if content.strip():
            samples.append(content)
            # Try to get blob_id, fall back to repo+path or just use index
            blob_id = row.get("blob_id") or f"{row.get('repo', 'unknown')}/{row.get('path', 'unknown')}"
            blob_ids.append(str(blob_id))

    logger.info(f"Samples with non-empty content: {len(samples)}")

    # Sample fraction
    if sample_fraction < 1.0:
        num_samples = max(1, int(len(samples) * sample_fraction))
        sampler = random.Random(seed)
        indices = sampler.sample(range(len(samples)), num_samples)
        samples = [samples[i] for i in indices]
        blob_ids = [blob_ids[i] for i in indices]
        logger.info(f"After sampling ({sample_fraction:.1%}): {len(samples)} samples")

    return samples, blob_ids


def load_data(
    data_dir: str = "./data",
    sample_fraction: float = 1.0,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from seen and unseen parquet files.

    Args:
        data_dir: Path to data directory (contains seen/ and unseen/ subdirs)
        sample_fraction: Fraction of data to load (0.0 to 1.0)
        train_fraction: Fraction to use for training (rest for testing)
        seed: Random seed for sampling and train/test split

    Returns:
        Tuple of (train_df, test_df) pandas DataFrames with columns ['text', 'blob_id', 'label']
        where label is 1 for seen and 0 for unseen
    """
    if not (0.0 < sample_fraction <= 1.0):
        raise ValueError(f"sample_fraction must be between 0.0 and 1.0, got {sample_fraction}")

    if not (0.0 < train_fraction < 1.0):
        raise ValueError(
            f"train_fraction must be between 0.0 and 1.0, got {train_fraction}"
        )

    # Load seen samples
    seen_dir = os.path.join(data_dir, "seen")
    logger.info(f"Loading seen samples from {seen_dir}")
    seen_parquet = find_first_parquet_file(seen_dir)
    seen_samples, seen_blob_ids = load_parquet_samples(
        seen_parquet,
        sample_fraction,
        seed=seed,
    )
    seen_labels = [1] * len(seen_samples)  # Label: 1 = seen

    # Load unseen samples
    unseen_dir = os.path.join(data_dir, "unseen")
    logger.info(f"Loading unseen samples from {unseen_dir}")
    unseen_parquet = find_first_parquet_file(unseen_dir)
    unseen_samples, unseen_blob_ids = load_parquet_samples(
        unseen_parquet,
        sample_fraction,
        seed=seed,
    )
    unseen_labels = [0] * len(unseen_samples)  # Label: 0 = unseen

    # Combine all samples into a DataFrame
    df = pd.DataFrame({
        'text': seen_samples + unseen_samples,
        'blob_id': seen_blob_ids + unseen_blob_ids,
        'label': seen_labels + unseen_labels
    })

    logger.info(f"Total samples: {len(df)} ({(df['label'] == 1).sum()} seen, {(df['label'] == 0).sum()} unseen)")

    # Stratified split into train and test (maintains same seen/unseen ratio in both splits)
    train_df, test_df = train_test_split(
        df,
        train_size=train_fraction,
        stratify=df['label'],
        random_state=seed
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        f"Train split: {len(train_df)} samples "
        f"({(train_df['label'] == 1).sum()} seen, {(train_df['label'] == 0).sum()} unseen)"
    )
    logger.info(
        f"Test split: {len(test_df)} samples "
        f"({(test_df['label'] == 1).sum()} seen, {(test_df['label'] == 0).sum()} unseen)"
    )

    return train_df, test_df
