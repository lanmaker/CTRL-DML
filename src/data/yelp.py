"""
Yelp review loader using HuggingFace datasets.
"""
from typing import List, Tuple
from pathlib import Path

import numpy as np


def load_yelp_reviews(n_samples: int = 2000, seed: int = 42) -> Tuple[List[str], np.ndarray]:
    """
    Load Yelp review texts and star ratings.

    Uses the yelp_review_full dataset from HuggingFace.

    Returns:
        texts: list of review texts
        ratings: array of star ratings in [1, 5]
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required for Yelp loader. Install with pip install datasets") from exc

    cache_dir = Path(__file__).parents[2] / "data" / "yelp_cache"
    ds = load_dataset("yelp_review_full", split="train", cache_dir=str(cache_dir))

    rng = np.random.default_rng(seed)
    if n_samples and n_samples < len(ds):
        idx = rng.choice(len(ds), size=n_samples, replace=False)
        ds = ds.select(idx.tolist())

    texts = ds["text"]
    # Dataset labels are 0-4; convert to 1-5 for interpretability
    ratings = np.array(ds["label"], dtype=np.float32) + 1.0
    return texts, ratings
