import argparse
import numpy as np
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description="Project STAR ATE demo (small vs regular classes)")
    parser.add_argument("--n-boot", type=int, default=200)
    args = parser.parse_args()
    star = sm.datasets.get_rdataset("STAR", "AER").data
    # Treatment: small class vs regular (drop other categories)
    star = star[star["classtype"].isin(["small.class", "regular.class"])]
    star["treat"] = (star["classtype"] == "small.class").astype(int)
    y = star["read"].to_numpy()
    t = star["treat"].to_numpy()
    ate = float(np.mean(y[t == 1]) - np.mean(y[t == 0]))

    rng = np.random.default_rng(42)
    ates = []
    n = len(y)
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        y_b, t_b = y[idx], t[idx]
        ates.append(float(np.mean(y_b[t_b == 1]) - np.mean(y_b[t_b == 0])))
    lo, hi = np.percentile(ates, [2.5, 97.5])

    print(f"Project STAR (small vs regular) ATE (diff-in-means): {ate:.2f}")
    print(f"Bootstrap 95% CI: [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
