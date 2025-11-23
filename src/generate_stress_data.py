import numpy as np
import sys
import os

# Add parent directory to path to import data_gen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_gen import generate_complex_data

def create_stress_data(n=5000, filename="stress_data.npz"):
    print(f"Generating stress data with n={n}...")
    data = generate_complex_data(n=n)
    
    X = data['X']
    T = data['T']
    Y = data['Y']
    tau = data['tau']
    
    # True Treatment Effect (constant in this case)
    true_te = np.full_like(Y, tau)
    
    print(f"Saving to {filename}...")
    np.savez(filename, X=X, T=T, y=Y, true_te=true_te)
    print("Done.")

if __name__ == "__main__":
    create_stress_data()
