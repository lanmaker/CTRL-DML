import numpy as np

def generate_complex_data(n=5000, seed=42):
    """
    Generates synthetic data for Causal Inference with three types of features:
    1. Confounders (C): Affect both Treatment (T) and Outcome (Y).
    2. Instruments (I): Affect only Treatment (T).
    3. Noise (N): Affect neither.
    
    Args:
        n (int): Number of samples.
        seed (int): Random seed for reproducibility.
        
    Returns:
        dict: A dictionary containing:
            - 'X': The feature matrix (concatenated C, I, N).
            - 'T': The binary treatment assignment.
            - 'Y': The observed outcome.
            - 'tau': The true treatment effect (constant = 3).
            - 'C': The confounders (for debugging/oracle).
            - 'I': The instruments (for debugging/oracle).
            - 'N': The noise (for debugging/oracle).
    """
    np.random.seed(seed)
    
    # 1. Confounders (C): 5 features
    # These are the variables we MUST control for.
    C = np.random.normal(0, 1, (n, 5))
    
    # 2. Instruments (I): 5 features
    # These affect T but not Y directly. Controlling for them increases variance.
    I = np.random.normal(0, 1, (n, 5))
    
    # 3. Noise (N): 10 features
    # Irrelevant features.
    N = np.random.normal(0, 1, (n, 10))
    
    # Construct the full feature matrix X
    # The model sees this, but doesn't know which column is which type.
    X = np.concatenate([C, I, N], axis=1)
    
    # Treatment Assignment Mechanism
    # T depends on C and I.
    # We make I have a strong effect to tempt the model to over-rely on it.
    logit = np.sum(C, axis=1) + 2 * np.sum(I, axis=1)
    T_prob = 1 / (1 + np.exp(-logit))
    T = np.random.binomial(1, T_prob)
    
    # Outcome Mechanism
    # Y depends on C and T.
    # CRITICAL: I does NOT appear here.
    # True treatment effect (tau) is set to 3.
    tau = 3
    # Non-linear relationship for C to make it harder
    Y = tau * T + np.sum(C, axis=1)**2 + np.random.normal(0, 1, n)
    
    return {
        'X': X,
        'T': T,
        'Y': Y,
        'tau': tau,
        'C': C,
        'I': I,
        'N': N
    }

if __name__ == "__main__":
    # Simple test
    data = generate_complex_data(n=10)
    print("Data shapes:")
    print(f"X: {data['X'].shape}")
    print(f"T: {data['T'].shape}")
    print(f"Y: {data['Y'].shape}")
    print(f"True Effect: {data['tau']}")
    print("Sample T:", data['T'][:5])
