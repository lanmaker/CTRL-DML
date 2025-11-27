import argparse
import numpy as np
import statsmodels.api as sm


def bootstrap_ate(y, t, n_boot=200, seed=42):
    rng = np.random.default_rng(seed)
    ates = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b, t_b = y[idx], t[idx]
        ate = float(np.mean(y_b[t_b == 1]) - np.mean(y_b[t_b == 0]))
        ates.append(ate)
    return float(np.mean(ates)), np.percentile(ates, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser(description="Lalonde ATE demo with bootstrap CI")
    parser.add_argument("--n-boot", type=int, default=200)
    args = parser.parse_args()
    lalonde = sm.datasets.get_rdataset("lalonde", "MatchIt").data
    y = lalonde["re78"].to_numpy()
    t = lalonde["treat"].to_numpy()
    ate = float(np.mean(y[t == 1]) - np.mean(y[t == 0]))
    ate_boot, ci = bootstrap_ate(y, t, n_boot=args.n_boot)
    print(f"Lalonde ATE (diff-in-means): {ate:.2f}")
    print(f"Lalonde ATE bootstrap mean: {ate_boot:.2f}, 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")


if __name__ == "__main__":
    main()
