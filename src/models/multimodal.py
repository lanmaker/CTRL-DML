"""
Multimodal CTRL-DML models for text + tabular data.

This module contains:
- CrossAttentionFusion: Tabular-to-text cross-attention
- MultimodalCTRL: Two-tower architecture for multimodal CATE estimation
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .dragonnet import TabularAttention


class CrossAttentionFusion(nn.Module):
    """
    Lightweight cross-attention: tabular queries attend over text tokens.

    This allows the model to focus on treatment-relevant phrases in the text
    based on the tabular representation.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=dropout
        )

    def forward(
        self,
        h_tab: torch.Tensor,
        text_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_tab: Tabular representation (B, H)
            text_tokens: Text token embeddings (B, L, H)

        Returns:
            Cross-attended text representation (B, H)
        """
        q = self.query_proj(h_tab).unsqueeze(1)  # (B, 1, H)
        k = self.key_proj(text_tokens)
        v = self.value_proj(text_tokens)
        out, _ = self.attn(q, k, v, need_weights=False)
        return out.squeeze(1)


class MultimodalCTRL(nn.Module):
    """
    Two-tower architecture for multimodal CATE estimation.

    Left tower: Tabular features with TabularAttention gating
    Right tower: Text embeddings (bag-of-words or cross-attention fusion)

    Outputs predictions for Y(0), Y(1), and propensity P(T=1|X).
    """

    def __init__(
        self,
        n_tab_input: int,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: int = 200,
        fusion: str = "bag",  # "bag" (EmbeddingBag) or "cross_attn"
        dropout: float = 0.2,
    ):
        """
        Args:
            n_tab_input: Number of tabular input features
            vocab_size: Size of text vocabulary
            embed_dim: Dimension of word embeddings
            hidden_dim: Hidden layer dimension
            fusion: Fusion type - "bag" for EmbeddingBag, "cross_attn" for cross-attention
            dropout: Dropout probability
        """
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
            self.text_proj = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ELU()
            )
        elif fusion == "cross_attn":
            self.text_proj = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout)
            )
            self.cross_attn = CrossAttentionFusion(
                hidden_dim,
                n_heads=1,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")

        # === 3. Fusion Layer ===
        self.fusion_layer = nn.Linear(hidden_dim + hidden_dim, hidden_dim)

        # === 4. Heads (Predict T, Y0, Y1) ===
        self.head_t = nn.Linear(hidden_dim, 1)
        self.head_y0 = nn.Linear(hidden_dim, 1)
        self.head_y1 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_tab: torch.Tensor,
        x_text: torch.Tensor,
        text_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_tab: Tabular features (B, n_tab)
            x_text: Text token indices (B, seq_len)
            text_weights: Optional TF-IDF weights (B, seq_len)

        Returns:
            Tuple of (y0_pred, y1_pred, t_prob)
        """
        h_tab = self.tab_tower(x_tab)

        if self.fusion_type == "bag":
            if text_weights is None:
                h_text = self.text_pool(x_text)
            else:
                tok = self.text_pool.weight[x_text]
                w = text_weights.unsqueeze(-1)
                h_text = (tok * w).sum(dim=1) / (w.sum(dim=1) + 1e-8)
            h_text = self.text_proj(h_text)
        else:
            # Embed tokens and apply cross-attention
            tok = self.embedding(x_text)  # (B, L, E)
            tok_proj = self.text_proj(tok)  # (B, L, H)
            h_text = self.cross_attn(h_tab, tok_proj)

        h_cat = torch.cat([h_tab, h_text], dim=1)
        rep = torch.relu(self.fusion_layer(h_cat))

        t_logit = self.head_t(rep)
        t_prob = torch.sigmoid(t_logit)
        y0 = self.head_y0(rep)
        y1 = self.head_y1(rep)

        return y0, y1, t_prob

    def predict_cate(
        self,
        x_tab: torch.Tensor,
        x_text: torch.Tensor,
        text_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict CATE = E[Y(1) - Y(0) | X]."""
        y0, y1, _ = self.forward(x_tab, x_text, text_weights=text_weights)
        return (y1 - y0).squeeze(-1)

    @property
    def mask_penalty(self) -> torch.Tensor:
        """Get sparsity penalty from tabular tower."""
        for module in self.tab_tower:
            if isinstance(module, TabularAttention):
                return module.current_mask_penalty
        return torch.tensor(0.0)
