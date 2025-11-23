import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_gen import generate_complex_data
from src.model import CTRL_Network

def train_ctrl_dml(n=5000, epochs=100, batch_size=64, lr=0.001):
    # 1. Data Preparation
    data = generate_complex_data(n=n)
    X = torch.FloatTensor(data['X'])
    T = torch.FloatTensor(data['T']).unsqueeze(1)
    Y = torch.FloatTensor(data['Y']).unsqueeze(1)
    true_tau = data['tau']
    
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=42)

    # Normalize data for Neural Network
    from sklearn.preprocessing import StandardScaler
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    
    Y_train = scaler_y.fit_transform(Y_train)
    Y_test_orig = Y_test.numpy() # Keep original for final calc
    Y_test = scaler_y.transform(Y_test)
    
    # Convert back to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    Y_train = torch.FloatTensor(Y_train)
    Y_test = torch.FloatTensor(Y_test)
    
    # 2. Model Initialization
    input_dim = X.shape[1]
    model = CTRL_Network(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 3. Training Loop
    print("Training CTRL-DML Network...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        t_pred, y_pred, _ = model(X_train)
        
        # Loss calculation
        loss_y = nn.MSELoss()(y_pred, Y_train)
        loss_t = nn.BCELoss()(t_pred, T_train)
        
        # Simple Regularization
        l1_reg = 0.0
        for param in model.shared_encoder.parameters():
            l1_reg += torch.sum(torch.abs(param))
            
        loss = loss_y + loss_t + 0.001 * l1_reg
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f} (Y: {loss_y.item():.4f}, T: {loss_t.item():.4f})")
            
    # 4. Evaluation (Causal Effect Estimation)
    model.eval()
    with torch.no_grad():
        t_pred_test, y_pred_test, _ = model(X_test)
        
    # Convert to numpy
    T_test_np = T_test.numpy()
    # Inverse transform Y predictions to original scale
    y_pred_np = scaler_y.inverse_transform(y_pred_test.numpy())
    Y_test_np = Y_test_orig
    
    t_pred_np = t_pred_test.numpy()
    
    # Residualization
    T_res = T_test_np - t_pred_np
    Y_res = Y_test_np - y_pred_np
    
    # Estimate Tau using OLS on residuals
    # lstsq returns solution, residuals, rank, s
    tau_hat = np.linalg.lstsq(T_res, Y_res, rcond=None)[0][0]
    
    if isinstance(tau_hat, np.ndarray):
        tau_hat = tau_hat.item()
    
    print("-" * 30)
    print(f"True Tau: {true_tau}")
    print(f"CTRL-DML Estimated Tau: {tau_hat:.4f}")
    
    mse = (tau_hat - true_tau) ** 2
    print(f"MSE (on Tau): {mse:.6f}")
    
    return mse

if __name__ == "__main__":
    train_ctrl_dml()
