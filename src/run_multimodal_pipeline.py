import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch

from run_multimodal_benchmark import run_once as run_ctrl_vs_cf_once, run_once_crossattn
from run_multimodal_bert_baseline import run_baselines

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "CTRL-DML-Paper"


def save_fig(fig: plt.Figure, filename: str):
    """Write a figure to both the repo root and paper directory."""
    for base in (ROOT, PAPER_DIR):
        out_path = base / filename
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["method", "p_noise"])
        .agg(
            mean_pehe=("pehe", "mean"),
            std_pehe=("pehe", "std"),
            sem_pehe=("pehe", "sem"),
            n=("pehe", "size"),
        )
        .reset_index()
    )
    return summary


def plot_sweep(summary: pd.DataFrame, methods: Iterable[str], filename: str = "multimodal_sweep.pdf"):
    plt.figure(figsize=(6.6, 4))
    label_map = {
        "TFIDF_CF": "TF-IDF Causal Forest",
        "CTRL_DML": "CTRL-DML (dense text)",
        "CTRL_DML_XATTN": "CTRL-DML (cross-attn)",
    }
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    for method, color in zip(methods, colors):
        sub = summary[summary["method"] == method].sort_values("p_noise")
        plt.errorbar(
            sub["p_noise"],
            sub["mean_pehe"],
            yerr=sub["std_pehe"],
            marker="o",
            color=color,
            label=label_map.get(method, method),
        )
    plt.xlabel(r"Text noise ratio $p_{\mathrm{noise}}$")
    plt.ylabel("PEHE (lower is better)")
    plt.title("Text signal ablation (fixed seeds/hyperparams)")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    save_fig(plt.gcf(), filename)


def plot_bar(summary: pd.DataFrame, methods: list[str], p_noise: float, filename: str):
    sub = summary[(summary["p_noise"] == p_noise) & (summary["method"].isin(methods))]
    sub = sub.set_index("method").loc[methods].reset_index()
    label_map = {
        "TFIDF_CF": "TF-IDF Causal Forest",
        "CTRL_DML": "CTRL-DML (dense text)",
        "CTRL_DML_XATTN": "CTRL-DML (cross-attn)",
        "BERT_TARNet": "DistilBERT + TARNet",
        "BERT_CF": "DistilBERT + CF",
    }
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"][: len(sub)]
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [label_map.get(m, m) for m in sub["method"]]
    bars = ax.bar(labels, sub["mean_pehe"], yerr=sub["std_pehe"], color=colors, alpha=0.85)
    ax.set_ylabel("PEHE (lower is better)")
    ax.set_title(f"Multimodal baselines (p_noise={p_noise})", pad=12)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in sub["mean_pehe"]], label_type="edge", padding=4)
    plt.tight_layout()
    save_fig(fig, filename)


def main():
    parser = argparse.ArgumentParser(description="Run multimodal sweeps and aggregate all baselines into a single CSV.")
    parser.add_argument("--p-noises", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1024, 2023])
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.002)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--bert-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--bert-p-noise", type=float, default=0.0)
    parser.add_argument("--bert-epochs", type=int, default=90)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-bert", action="store_true", help="Skip frozen BERT upper-bound baselines.")
    args = parser.parse_args()

    rows = []
    for p in args.p_noises:
        for seed in args.seeds:
            print(f"\n=== CTRL-DML vs TF-IDF Forest | p_noise={p} | seed={seed} ===")
            pehe_cf, pehe_ctrl = run_ctrl_vs_cf_once(
                n_samples=args.n_samples,
                vocab_size=args.vocab_size,
                p_noise=p,
                seed=seed,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                embed_dim=args.embed_dim,
            )
            rows.append({"method": "TFIDF_CF", "p_noise": p, "seed": seed, "pehe": pehe_cf})
            rows.append({"method": "CTRL_DML", "p_noise": p, "seed": seed, "pehe": pehe_ctrl})
            # Cross-attention variant (lightweight)
            _, pehe_ctrl_xa = run_once_crossattn(
                n_samples=args.n_samples,
                vocab_size=args.vocab_size,
                p_noise=p,
                seed=seed,
                epochs=args.epochs + 10,
                lr=args.lr,
                weight_decay=args.weight_decay,
                embed_dim=args.embed_dim,
            )
            rows.append({"method": "CTRL_DML_XATTN", "p_noise": p, "seed": seed, "pehe": pehe_ctrl_xa})

    if not args.skip_bert:
        device = torch.device(args.device)
        print(f"\n=== Running frozen BERT baselines on {device} ===")
        cf_mean, tarnet_mean, cf_scores, tarnet_scores = run_baselines(
            n_samples=args.n_samples,
            p_noise=args.bert_p_noise,
            seeds=args.seeds,
            model_name=args.bert_model,
            device=device,
            epochs=args.bert_epochs,
        )
        for seed, score in zip(args.seeds, cf_scores):
            rows.append({"method": "BERT_CF", "p_noise": args.bert_p_noise, "seed": seed, "pehe": score})
        for seed, score in zip(args.seeds, tarnet_scores):
            rows.append({"method": "BERT_TARNet", "p_noise": args.bert_p_noise, "seed": seed, "pehe": score})

    df = pd.DataFrame(rows)
    for base in (ROOT, PAPER_DIR):
        out = base / "multimodal_results.csv"
        df.to_csv(out, index=False)
        print(f"Wrote rows to {out}")

    summary = summarize(df)
    for base in (ROOT, PAPER_DIR):
        out = base / "multimodal_results_summary.csv"
        summary.to_csv(out, index=False)
        print(f"Wrote summary to {out}")

    # Plot sweep for TF-IDF CF vs CTRL-DML
    plot_sweep(summary, methods=["TFIDF_CF", "CTRL_DML", "CTRL_DML_XATTN"], filename="multimodal_sweep.pdf")

    # Bar for clean text (p_noise=0) CF vs CTRL-DML
    plot_bar(summary, methods=["TFIDF_CF", "CTRL_DML"], p_noise=0.0, filename="multimodal_result.pdf")

    # Bar showing lightweight CTRL-DML vs BERT upper bound
    upper_methods = ["TFIDF_CF", "CTRL_DML", "CTRL_DML_XATTN"] + ([] if args.skip_bert else ["BERT_TARNet"])
    plot_bar(summary, methods=upper_methods, p_noise=args.bert_p_noise, filename="multimodal_bert_baselines.pdf")


if __name__ == "__main__":
    main()
