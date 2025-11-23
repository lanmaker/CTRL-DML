import torch
import torch.nn as nn
from my_dragonnet import TabularAttention # Reuse previous module

class MultimodalCTRL(nn.Module):
    def __init__(self, n_tab_input, vocab_size, embed_dim=32, hidden_dim=200):
        super().__init__()
        
        # === 1. Left Tower: Tabular ===
        # Reuse Attention mechanism
        self.tab_tower = nn.Sequential(
            TabularAttention(n_tab_input, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.2)
        )
        
        # === 2. Right Tower: Text ===
        # EmbeddingBag is perfect for Bag-of-Words tasks
        # Input is word indices, output is mean embedding
        self.text_tower = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean')
        self.text_linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ELU()
        )
        
        # === 3. Fusion Layer ===
        # Concatenate features
        self.fusion = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        
        # === 4. DragonNet Heads (Predict T, Y0, Y1) ===
        self.head_t = nn.Linear(hidden_dim, 1)
        self.head_y0 = nn.Linear(hidden_dim, 1)
        self.head_y1 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_tab, x_text):
        # x_tab: [batch, n_tab]
        # x_text: [batch, seq_len] (indices)
        
        h_tab = self.tab_tower(x_tab)
        h_text = self.text_tower(x_text) # Output: [batch, embed_dim]
        h_text = self.text_linear(h_text)
        
        # Concatenate
        h_cat = torch.cat([h_tab, h_text], dim=1)
        rep = torch.relu(self.fusion(h_cat))
        
        t_logit = self.head_t(rep)
        t_prob = torch.sigmoid(t_logit)
        y0 = self.head_y0(rep)
        y1 = self.head_y1(rep)
        
        return y0, y1, t_prob
