import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Add parent directory to path to import data_gen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_gen import generate_complex_data

def run_baseline(n=5000):
    print("Generating data...")
    data = generate_complex_data(n=n)
    X = data['X']
    T = data['T']
    Y = data['Y']
    true_tau = data['tau']
    
    # Split data
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=42)
    
    print("Training Baseline DML (Double Machine Learning)...")
    
    # Stage 1: Nuisance Estimation
    # Model T ~ X (Propensity Score)
    # We use Random Forest which is robust but can overfit to instruments
    model_t = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model_t.fit(X_train, T_train)
    T_pred_train = model_t.predict_proba(X_train)[:, 1]
    T_pred_test = model_t.predict_proba(X_test)[:, 1]
    
    # Model Y ~ X (Outcome Model)
    model_y = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model_y.fit(X_train, Y_train)
    Y_pred_train = model_y.predict(X_train)
    Y_pred_test = model_y.predict(X_test)
    
    # Stage 2: Residualization (Partialling Out)
    # T_res = T - E[T|X]
    # Y_res = Y - E[Y|X]
    T_res_train = T_train - T_pred_train
    Y_res_train = Y_train - Y_pred_train
    
    T_res_test = T_test - T_pred_test
    Y_res_test = Y_test - Y_pred_test
    
    # Stage 3: Causal Effect Estimation
    # Regress Y_res on T_res
    # Since we assume constant treatment effect, this is just a simple linear regression without intercept
    # tau = (T_res' T_res)^(-1) T_res' Y_res
    
    # Using simple OLS on residuals
    tau_hat = np.linalg.lstsq(T_res_test.reshape(-1, 1), Y_res_test, rcond=None)[0][0]
    
    print(f"True Tau: {true_tau}")
    print(f"Estimated Tau: {tau_hat:.4f}")
    
    mse = (tau_hat - true_tau) ** 2
    print(f"MSE (on Tau): {mse:.6f}")
    
    return mse

if __name__ == "__main__":
    run_baseline()
