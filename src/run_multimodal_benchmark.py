import argparse
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from data_multimodal import get_multimodal_data, convert_text_to_tfidf
from model_multimodal import MultimodalCTRL


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_once(
    n_samples: int,
    vocab_size: int,
    p_noise: float,
    seed: int = 42,
    epochs: int = 70,
    lr: float = 0.002,
    weight_decay: float = 0.002,
    embed_dim: int = 16,
    fusion: str = "bag",
    run_cf: bool = True,
):
    set_seed(seed)
    print(f"Generating Multimodal Data (p_noise={p_noise})...")
    X_tab, X_text, Y, T, true_te = get_multimodal_data(n=n_samples, vocab_size=vocab_size, p_noise=p_noise)

    pehe_cf = None
    if run_cf:
        print("Running Causal Forest (Baseline)...")
        X_tfidf = convert_text_to_tfidf(X_text, vocab_size)
        X_combined = np.concatenate([X_tab, X_tfidf], axis=1)
        est = CausalForestDML(
            model_y=LassoCV(),
            model_t=LogisticRegressionCV(max_iter=1000),
            n_estimators=100,
            discrete_treatment=True,
            random_state=seed,
        )
        est.fit(Y, T, X=X_combined)
        pred_cf = est.effect(X_combined).flatten()
        pehe_cf = float(np.sqrt(np.mean((true_te - pred_cf) ** 2)))
        print(f"Causal Forest PEHE: {pehe_cf:.4f}")

    print("Running Multimodal CTRL-DML (Ours)...")
    xt_tab = torch.from_numpy(X_tab).float()
    xt_text = torch.from_numpy(X_text).long()
    yt = torch.from_numpy(Y).float().unsqueeze(1)
    tt = torch.from_numpy(T).float().unsqueeze(1)

    model = MultimodalCTRL(n_tab_input=10, vocab_size=vocab_size, embed_dim=embed_dim, fusion=fusion)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        optimizer.zero_grad()
        y0, y1, t_prob = model(xt_tab, xt_text)
        loss_t = F.binary_cross_entropy(t_prob, tt)
        y_pred = tt * y1 + (1 - tt) * y0
        loss_y = F.mse_loss(y_pred, yt)
        loss = loss_y + loss_t
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch} loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        y0, y1, _ = model(xt_tab, xt_text)
        pred_ours = (y1 - y0).cpu().numpy().flatten()
    pehe_ours = float(np.sqrt(np.mean((true_te - pred_ours) ** 2)))
    print(f"Multimodal CTRL-DML PEHE: {pehe_ours:.4f}")
    return pehe_cf, pehe_ours


def run_once_crossattn(
    n_samples: int,
    vocab_size: int,
    p_noise: float,
    seed: int = 42,
    epochs: int = 80,
    lr: float = 0.002,
    weight_decay: float = 0.002,
    embed_dim: int = 16,
    run_cf: bool = False,
):
    """Variant with cross-attention fusion (tabular queries over text tokens)."""
    return run_once(
        n_samples=n_samples,
        vocab_size=vocab_size,
        p_noise=p_noise,
        seed=seed,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        embed_dim=embed_dim,
        fusion="cross_attn",
        run_cf=run_cf,
    )


def plot_bar(pehe_cf: float, pehe_ours: float, out_path: str = "multimodal_result.pdf"):
    plt.figure(figsize=(8, 5))
    models = ["Causal Forest\n(TF-IDF)", "Multimodal\nCTRL-DML"]
    scores = [pehe_cf, pehe_ours]
    colors = ["#1f77b4", "#2ca02c"]
    plt.bar(models, scores, color=colors, alpha=0.8)
    plt.ylabel("PEHE (lower is better)")
    plt.title("Performance on Text-Confounded Data")
    plt.ylim(0, max(scores) * 1.3)
    for i, v in enumerate(scores):
        plt.text(i, v + 0.03, f"{v:.2f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved bar plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--p-noise", type=float, default=0.0, help="Probability of masking the confounding token.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.002)
    parser.add_argument("--embed-dim", type=int, default=16)
    args = parser.parse_args()

    cf, ours = run_once(
        args.n_samples,
        args.vocab_size,
        args.p_noise,
        args.seed,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
    )
    plot_bar(cf, ours)
