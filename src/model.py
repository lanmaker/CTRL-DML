import torch
import torch.nn as nn
import torch.nn.functional as F

class CTRL_Network(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(CTRL_Network, self).__init__()
        
        # Shared Encoder: Learns common representation Phi(X)
        # In a more advanced version, this could be a TabNet or Transformer
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # T-Head: Predicts Propensity Score P(T=1|X)
        self.t_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Y-Head: Predicts Outcome E[Y|X]
        # Note: In standard DML, we often predict Y directly or Y given T.
        # Here we predict Y directly from shared representation. 
        # Ideally, we might want to feed T into this head as well, 
        # but for feature disentanglement, we want to see if Phi(X) captures enough.
        # Let's stick to the plan: Head Y predicts Outcome.
        self.y_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        embed = self.shared_encoder(x)
        t_pred = self.t_head(embed)
        y_pred = self.y_head(embed)
        return t_pred, y_pred, embed

class OrthogonalRegularization(nn.Module):
    """
    Penalizes if the gradients of T-Head and Y-Head w.r.t Shared Encoder are too aligned.
    This is a simplified proxy for 'disentanglement'.
    Alternatively, we can penalize the correlation of the hidden representations of the heads.
    """
    def __init__(self, lambda_reg=0.1):
        super().__init__()
        self.lambda_reg = lambda_reg
        
    def forward(self, t_head_params, y_head_params):
        # Placeholder for a more complex regularization.
        # For now, we can just use L1/L2 on the weights to encourage sparsity.
        pass
