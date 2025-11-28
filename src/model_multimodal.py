import torch
import torch.nn as nn
from my_dragonnet import TabularAttention  # Reuse previous module


class CrossAttentionFusion(nn.Module):
    """Lightweight cross-attention: tabular queries attend over text tokens."""

    def __init__(self, hidden_dim: int, n_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True, dropout=dropout)

    def forward(self, h_tab: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        # h_tab: (B, H), text_tokens: (B, L, H)
        q = self.query_proj(h_tab).unsqueeze(1)  # (B,1,H)
        k = self.key_proj(text_tokens)
        v = self.value_proj(text_tokens)
        out, _ = self.attn(q, k, v, need_weights=False)
        return out.squeeze(1)


class MultimodalCTRL(nn.Module):
    def __init__(
        self,
        n_tab_input: int,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: int = 200,
        fusion: str = "bag",  # "bag" (EmbeddingBag) or "cross_attn"
        dropout: float = 0.2,
    ):
        super().__init__()

        # === 1. Left Tower: Tabular ===
        self.tab_tower = nn.Sequential(
            TabularAttention(n_tab_input, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # === 2. Right Tower: Text ===
        self.fusion_type = fusion
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if fusion == "bag":
            self.text_pool = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
            self.text_proj = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ELU())
        elif fusion == "cross_attn":
            self.text_proj = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ELU(), nn.Dropout(dropout))
            self.cross_attn = CrossAttentionFusion(hidden_dim, n_heads=1, dropout=dropout)
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")

        # === 3. Fusion Layer ===
        self.fusion = nn.Linear(hidden_dim + hidden_dim, hidden_dim)

        # === 4. Heads (Predict T, Y0, Y1) ===
        self.head_t = nn.Linear(hidden_dim, 1)
        self.head_y0 = nn.Linear(hidden_dim, 1)
        self.head_y1 = nn.Linear(hidden_dim, 1)

    def forward(self, x_tab: torch.Tensor, x_text: torch.Tensor):
        # x_tab: [batch, n_tab]; x_text: [batch, seq_len] indices
        h_tab = self.tab_tower(x_tab)

        if self.fusion_type == "bag":
            h_text = self.text_pool(x_text)
            h_text = self.text_proj(h_text)
        else:
            # Embed tokens and apply cross-attention
            tok = self.embedding(x_text)  # (B, L, E)
            tok_proj = self.text_proj(tok)  # (B, L, H)
            h_text = self.cross_attn(h_tab, tok_proj)

        h_cat = torch.cat([h_tab, h_text], dim=1)
        rep = torch.relu(self.fusion(h_cat))

        t_logit = self.head_t(rep)
        t_prob = torch.sigmoid(t_logit)
        y0 = self.head_y0(rep)
        y1 = self.head_y1(rep)
        return y0, y1, t_prob
