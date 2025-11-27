import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoTokenizer
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV, LogisticRegressionCV

from data_multimodal import get_multimodal_data


class TabTextTARNet(nn.Module):
    """Simple TARNet that consumes tabular + frozen text embeddings."""

    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.y0 = nn.Linear(hidden, 1)
        self.y1 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.y0(h), self.y1(h)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_text_strings(X_text: np.ndarray) -> list[str]:
    vocab = {0: "neutral", 1: "severe", 2: "mild"}
    return [" ".join([vocab.get(int(t), f"noise{int(t)}") for t in row]) for row in X_text]


def compute_bert_embeddings(
    texts,
    model_name: str,
    device: torch.device,
    max_length: int = 32,
    tokenizer: AutoTokenizer | None = None,
    model: AutoModel | None = None,
):
    """Compute CLS embeddings with an optional preloaded model/tokenizer to avoid re-downloads."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    with torch.no_grad():
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        outputs = model(**enc)
        cls = outputs.last_hidden_state[:, 0, :]
        return cls.cpu().numpy()


def run_baselines(
    n_samples: int,
    p_noise: float,
    seeds: list[int],
    model_name: str,
    device: torch.device,
    epochs: int = 120,
):
    cf_scores, tarnet_scores = [], []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name).to(device)
    bert_model.eval()
    for seed in seeds:
        set_seed(seed)
        X_tab, X_text, Y, T, true_te = get_multimodal_data(n=n_samples, vocab_size=1000, p_noise=p_noise)
        texts = to_text_strings(X_text)
        text_emb = compute_bert_embeddings(
            texts,
            model_name=model_name,
            device=device,
            tokenizer=tokenizer,
            model=bert_model,
        )
        X_concat = np.concatenate([X_tab, text_emb], axis=1)

        est = CausalForestDML(
            model_y=LassoCV(),
            model_t=LogisticRegressionCV(max_iter=1000),
            n_estimators=100,
            discrete_treatment=True,
            random_state=seed,
        )
        est.fit(Y, T, X=X_concat)
        pehe_cf = float(np.sqrt(np.mean((true_te - est.effect(X_concat).flatten()) ** 2)))

        x_t = torch.from_numpy(X_concat).float().to(device)
        y_t = torch.from_numpy(Y).float().unsqueeze(1).to(device)
        t_t = torch.from_numpy(T).float().unsqueeze(1).to(device)
        model = TabTextTARNet(input_dim=X_concat.shape[1], hidden=64, dropout=0.25).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

        ds = TensorDataset(x_t, y_t, t_t)
        loader = DataLoader(ds, batch_size=128, shuffle=True)
        model.train()
        for _ in range(epochs):
            for bx, by, bt in loader:
                opt.zero_grad()
                y0, y1 = model(bx)
                y_pred = bt * y1 + (1 - bt) * y0
                loss = F.mse_loss(y_pred, by)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            y0, y1 = model(x_t)
            tau_pred = (y1 - y0).cpu().numpy().flatten()
        pehe_tarnet = float(np.sqrt(np.mean((true_te - tau_pred) ** 2)))

        cf_scores.append(pehe_cf)
        tarnet_scores.append(pehe_tarnet)
    return float(np.mean(cf_scores)), float(np.mean(tarnet_scores)), cf_scores, tarnet_scores


def plot_bar(cf, tarnet, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    models = ["BERT embed + CF", "BERT embed + TARNet"]
    scores = [cf, tarnet]
    colors = ["#1f77b4", "#ff7f0e"]
    bars = ax.bar(models, scores, color=colors, alpha=0.85)
    ax.set_ylabel("PEHE (lower is better)")
    ax.set_title("Frozen BERT/ClinicalBERT baselines", pad=20)
    ax.set_ylim(0, max(scores) * 1.35)
    ax.margins(y=0.05)
    ax.bar_label(
        bars,
        labels=[f"{v:.2f}" for v in scores],
        label_type="center",
        fontweight="bold",
        fontsize=11,
        color="white",
    )
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--p-noise", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1024, 2023])
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cf, tarnet, cf_raw, tarnet_raw = run_baselines(
        n_samples=args.n_samples, p_noise=args.p_noise, seeds=args.seeds, model_name=args.model_name, device=device
    )
    print(f"PEHE | BERT+CF: {cf:.3f} | BERT+TARNet: {tarnet:.3f}")
    np.savez(
        "multimodal_bert_baselines_values.npz",
        p_noise=args.p_noise,
        cf=np.array(cf_raw),
        tarnet=np.array(tarnet_raw),
        seeds=np.array(args.seeds),
    )
    plot_bar(cf, tarnet, out_path="multimodal_bert_baselines.pdf")
