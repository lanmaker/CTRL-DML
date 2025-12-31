"""
Synthetic multimodal data generator for text + tabular confounding.

This module generates data where:
- Tabular features X_tab are mostly noise
- Text features contain a latent confounder (word "Severe" vs "Mild")
- Treatment and outcome depend on the text confounder

This setup tests whether models can capture confounding from unstructured text.
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
    Generate data with 'Latent Text Confounder'.

    Scenario:
    - Tabular X_tab: Basic physical indicators (mostly noise).
    - Text X_text: Doctor's diagnostic notes.
    - Key Setup: Word "Severe" (id=1) is a strong confounder.
      If present, T and Y both increase.

    Args:
        n: Number of samples
        vocab_size: Size of vocabulary
        p_noise: Probability of masking the confounding word (for ablation)
        seed: Random seed

    Returns:
        Tuple of (X_tab, X_text_indices, Y, T, true_te)
            X_tab: Tabular features (n, 10) - mostly noise
            X_text_indices: Text token indices (n, 10)
            Y: Outcomes (n,)
            T: Treatment indicators (n,)
            true_te: True treatment effects (n,) - constant 3.0
    """
    rng = np.random.default_rng(seed)

    # 1. Tabular Data (10 dim, mostly noise to confuse the model)
    X_tab = rng.normal(0, 1, size=(n, 10)).astype(np.float32)

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

    # 3. Generate T and Y (Determined purely by 'Severe' in text)
    # Note: Tabular data X_tab has NO effect here!
    # If the model can't read text, it's blind.

    confounder = np.array(has_severe).astype(np.float32)

    # T depends on confounder
    propensity = 0.2 + 0.6 * confounder  # Severe -> 0.8 prob treatment, Mild -> 0.2
    T = rng.binomial(1, propensity).astype(np.float32)

    # Y depends on confounder and T
    # True CATE = 3.0
    Y = (2 * confounder + 3.0 * T + rng.normal(0, 0.5, n)).astype(np.float32)

    true_te = np.ones(n, dtype=np.float32) * 3.0

    return X_tab, X_text_indices, Y, T, true_te


def convert_text_to_tfidf(
    X_text_indices: np.ndarray,
    vocab_size: int
) -> np.ndarray:
    """
    Convert text indices to TF-IDF features for tree-based methods.

    Args:
        X_text_indices: Token indices (n, seq_len)
        vocab_size: Maximum vocabulary size

    Returns:
        TF-IDF feature matrix (n, vocab_size)
    """
    # Convert indices back to string "1 45 99..."
    corpus = [" ".join(map(str, row)) for row in X_text_indices]
    vec = CountVectorizer(max_features=vocab_size)
    X_tfidf = vec.fit_transform(corpus).toarray().astype(np.float32)
    return X_tfidf


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
        vocab_size: Vocabulary size for TF-IDF

    Returns:
        Combined feature matrix
    """
    X_tfidf = convert_text_to_tfidf(X_text_indices, vocab_size)
    return np.concatenate([X_tab, X_tfidf], axis=1)


if __name__ == "__main__":
    X_tab, X_text, Y, T, cate = get_multimodal_data(n=1000)
    print(f"Multimodal data: X_tab={X_tab.shape}, X_text={X_text.shape}")
    print(f"T mean={T.mean():.3f}, Y mean={Y.mean():.3f}, CATE={cate.mean():.3f}")
