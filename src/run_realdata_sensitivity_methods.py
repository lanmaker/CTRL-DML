import math
import pandas as pd
import statsmodels.api as sm


def load_lalonde():
    data = sm.datasets.get_rdataset("lalonde", "MatchIt").data
    y = data["re78"].to_numpy().astype(float) / 1000.0  # scale to thousands
    return y


def load_star():
    data = sm.datasets.get_rdataset("STAR", "AER").data
    data = data[["star1", "read1"]].dropna()
    data = data[data["star1"].isin(["small", "regular"])]
    y = data["read1"].to_numpy().astype(float)
    return y


def evalue_for_difference(ate: float, sd: float) -> float:
    if sd <= 0:
        return float("nan")
    d = abs(ate) / sd
    rr = math.exp(math.pi * d / math.sqrt(3.0))
    if rr < 1:
        rr = 1 / rr
    return rr + math.sqrt(rr * max(rr - 1, 0))


def compute_for_dataset(name: str, sd: float, ate_csv: str):
    df = pd.read_csv(ate_csv)
    rows = []
    for _, row in df.iterrows():
        ate = float(row["ate"])
        evalue = evalue_for_difference(ate, sd)
        rows.append(
            {
                "dataset": name,
                "method": row["method"],
                "ate": ate,
                "ci_lower": float(row["ci_lower"]),
                "ci_upper": float(row["ci_upper"]),
                "evalue": evalue,
            }
        )
    return rows


def main():
    y_l = load_lalonde()
    y_s = load_star()
    sd_l = float(y_l.std(ddof=1))
    sd_s = float(y_s.std(ddof=1))
    rows = []
    rows += compute_for_dataset("lalonde", sd_l, "realdata_lalonde_ate.csv")
    rows += compute_for_dataset("star", sd_s, "realdata_star_ate.csv")
    out = pd.DataFrame(rows)
    out.to_csv("realdata_sensitivity_methods.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
