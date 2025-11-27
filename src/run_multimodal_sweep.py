import argparse
import numpy as np
import matplotlib.pyplot as plt

from run_multimodal_benchmark import run_once


def sweep(p_noises, n_samples: int, vocab_size: int, seed: int):
    cf_scores, ctrl_scores = [], []
    for p in p_noises:
        print(f"\n=== Running p_noise={p} ===")
        cf, ctrl = run_once(n_samples=n_samples, vocab_size=vocab_size, p_noise=p, seed=seed)
        cf_scores.append(cf)
        ctrl_scores.append(ctrl)
    return np.array(cf_scores), np.array(ctrl_scores)


def plot_sweep(p_noises, cf_scores, ctrl_scores, out_path="multimodal_sweep.pdf"):
    plt.figure(figsize=(7, 4))
    plt.plot(p_noises, cf_scores, marker="o", color="#1f77b4", label="Causal Forest (TF-IDF)")
    plt.plot(p_noises, ctrl_scores, marker="o", color="#2ca02c", label="CTRL-DML (multimodal)")
    plt.xlabel("Text noise ratio $p_{noise}$")
    plt.ylabel("PEHE (lower is better)")
    plt.title("Text signal ablation (higher $p_{noise}$ = weaker confounding signal)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved sweep plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-noises", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cf, ctrl = sweep(args.p_noises, n_samples=args.n_samples, vocab_size=args.vocab_size, seed=args.seed)
    plot_sweep(args.p_noises, cf, ctrl)
