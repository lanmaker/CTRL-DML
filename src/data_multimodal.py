import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer

def get_multimodal_data(n=3000, vocab_size=1000, p_noise: float = 0.0):
    """
    Generate data with 'Latent Text Confounder'.
    Scenario:
    - Tabular X_tab: Basic physical indicators (mostly noise).
    - Text X_text: Doctor's diagnostic notes.
    - Key Setup: Word "Severe" (id=1) is a strong confounder. 
      If present, T and Y both increase.
    """
    # 1. Tabular Data (10 dim, mostly noise to confuse the model)
    X_tab = np.random.normal(0, 1, size=(n, 10))
    
    # 2. Text Data Generation
    # Vocab size 1000.
    # Word 1: "SEVERE" -> Causes T=1, Y increases
    # Word 2: "MILD"   -> Causes T=0, Y decreases
    # Others: Random noise
    
    vocab = np.arange(vocab_size)
    texts = []
    has_severe = [] # Track who has "SEVERE"
    
    for i in range(n):
        # Randomly generate a 10-word note
        doc = np.random.choice(vocab, 10, replace=True)
        
        # 50% probability of being severe (insert word 1)
        is_severe = np.random.rand() > 0.5
        if is_severe:
            doc[0] = 1 # Force "Severe"
        else:
            doc[0] = 2 # Force "Mild"

        # With probability p_noise, mask the confounding word with a neutral token
        if np.random.rand() < p_noise:
            doc[0] = 0
            
        texts.append(doc) # Keep as indices for Embedding
        has_severe.append(is_severe)
    
    X_text_indices = np.array(texts) # shape: (N, 10)
    
    # 3. Generate T and Y (Determined purely by 'Severe' in text)
    # Note: Tabular data X_tab has NO effect here!
    # If the model can't read text, it's blind.
    
    confounder = np.array(has_severe).astype(float)
    
    # T depends on confounder
    propensity = 0.2 + 0.6 * confounder # Severe -> 0.8 prob treatment, Mild -> 0.2
    T = np.random.binomial(1, propensity)
    
    # Y depends on confounder and T
    # True CATE = 3.0
    Y = 2 * confounder + 3.0 * T + np.random.normal(0, 0.5, n)
    
    true_te = np.ones(n) * 3.0
    
    return X_tab, X_text_indices, Y, T, true_te

# Helper: Convert text indices to TF-IDF for Causal Forest
def convert_text_to_tfidf(X_text_indices, vocab_size):
    # Convert indices back to string "1 45 99..."
    corpus = [" ".join(map(str, row)) for row in X_text_indices]
    vec = CountVectorizer(max_features=vocab_size)
    X_tfidf = vec.fit_transform(corpus).toarray()
    return X_tfidf
