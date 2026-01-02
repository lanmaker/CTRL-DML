"""
Synthetic multimodal data generator for text + tabular confounding.

This module generates data where:
- Tabular features X_tab contain confounders (first 2 features affect T/Y)
- Text features contain a latent confounder (word "Severe" vs "Mild")
- Treatment and outcome depend on BOTH text and tabular confounders

This setup tests whether models can capture confounding from multiple modalities.
"""
import numpy as np
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer


def get_multimodal_data(
    n: int = 3000,
    vocab_size: int = 1000,
    p_noise: float = 0.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data with multimodal confounders (text + tabular).

    Scenario:
    - Tabular X_tab: First 2 features are confounders, rest are noise.
    - Text X_text: Doctor's diagnostic notes with latent confounder.
    - Key Setup: Both modalities affect T and Y.

    Args:
        n: Number of samples
        vocab_size: Size of vocabulary
        p_noise: Probability of masking the confounding word (for ablation)
        seed: Random seed

    Returns:
        Tuple of (X_tab, X_text_indices, Y, T, true_te)
            X_tab: Tabular features (n, 10) - first 2 are confounders
            X_text_indices: Text token indices (n, 10)
            Y: Outcomes (n,)
            T: Treatment indicators (n,)
            true_te: True treatment effects (n,) - constant 3.0
    """
    rng = np.random.default_rng(seed)

    # 1. Tabular Data (10 dim)
    # First 2 features are confounders, rest are noise
    X_tab = rng.normal(0, 1, size=(n, 10)).astype(np.float32)

    # Extract tabular confounders (normalized to [0, 1] for effect computation)
    tab_conf1 = (X_tab[:, 0] > 0).astype(np.float32)  # Binary from first feature
    tab_conf2 = (X_tab[:, 1] > 0).astype(np.float32)  # Binary from second feature

    # 2. Text Data Generation
    # Vocab size 1000.
    # Word 1: "SEVERE" -> Causes T=1, Y increases
    # Word 2: "MILD"   -> Causes T=0, Y decreases
    # Others: Random noise

    vocab = np.arange(vocab_size)
    texts = []
    has_severe = []  # Track who has "SEVERE"

    for i in range(n):
        # Randomly generate a 10-word note
        doc = rng.choice(vocab, 10, replace=True)

        # 50% probability of being severe (insert word 1)
        is_severe = rng.random() > 0.5
        if is_severe:
            doc[0] = 1  # Force "Severe"
        else:
            doc[0] = 2  # Force "Mild"

        # With probability p_noise, mask the confounding word with a neutral token
        if rng.random() < p_noise:
            doc[0] = 0

        texts.append(doc)
        has_severe.append(is_severe)

    X_text_indices = np.array(texts, dtype=np.int64)

    # 3. Generate T and Y (depends on BOTH text and tabular confounders)
    text_conf = np.array(has_severe).astype(np.float32)

    # Combined confounding effect: text (weight 0.4) + tabular (weight 0.3 each)
    # Total confounding strength ~= 1.0 when all high
    combined_conf = 0.4 * text_conf + 0.3 * tab_conf1 + 0.3 * tab_conf2

    # T depends on combined confounder
    # Propensity ranges from 0.1 (all low) to 0.9 (all high)
    propensity = 0.1 + 0.8 * combined_conf
    propensity = np.clip(propensity, 0.05, 0.95)
    T = rng.binomial(1, propensity).astype(np.float32)

    # Y depends on combined confounder and T
    # Y = confounding_effect + treatment_effect + noise
    # True CATE = 3.0
    Y = (2.0 * combined_conf + 3.0 * T + rng.normal(0, 0.5, n)).astype(np.float32)

    true_te = np.ones(n, dtype=np.float32) * 3.0

    return X_tab, X_text_indices, Y, T, true_te


def convert_text_to_bow(
    X_text_indices: np.ndarray,
    vocab_size: int
) -> np.ndarray:
    """
    Convert text indices to bag-of-words features for tree-based methods.

    Args:
        X_text_indices: Token indices (n, seq_len)
        vocab_size: Maximum vocabulary size

    Returns:
        Bag-of-words feature matrix (n, vocab_size)
    """
    n_samples = X_text_indices.shape[0]
    # Create fixed-size bag-of-words representation
    # Count occurrences of each token index per sample
    X_bow = np.zeros((n_samples, vocab_size), dtype=np.float32)
    for i, row in enumerate(X_text_indices):
        for token_idx in row:
            if 0 <= token_idx < vocab_size:
                X_bow[i, token_idx] += 1
    return X_bow


# Alias for backward compatibility
convert_text_to_tfidf = convert_text_to_bow


def get_combined_features(
    X_tab: np.ndarray,
    X_text_indices: np.ndarray,
    vocab_size: int = 1000
) -> np.ndarray:
    """
    Combine tabular and text features for traditional ML methods.

    Args:
        X_tab: Tabular features
        X_text_indices: Text token indices
        vocab_size: Vocabulary size for bag-of-words

    Returns:
        Combined feature matrix
    """
    X_bow = convert_text_to_bow(X_text_indices, vocab_size)
    return np.concatenate([X_tab, X_bow], axis=1)


if __name__ == "__main__":
    X_tab, X_text, Y, T, cate = get_multimodal_data(n=1000)
    print(f"Multimodal data: X_tab={X_tab.shape}, X_text={X_text.shape}")
    print(f"T mean={T.mean():.3f}, Y mean={Y.mean():.3f}, CATE={cate.mean():.3f}")
