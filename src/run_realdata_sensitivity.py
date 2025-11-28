import math
import pandas as pd
import numpy as np
import statsmodels.api as sm


def load_lalonde():
    data = sm.datasets.get_rdataset("lalonde", "MatchIt").data
    y = data["re78"].to_numpy().astype(float) / 1000.0  # scale to thousands
    t = data["treat"].to_numpy().astype(float)
    return y, t


def load_star():
    data = sm.datasets.get_rdataset("STAR", "AER").data
    data = data[["star1", "read1"]].dropna()
    data = data[data["star1"].isin(["small", "regular"])]
    y = data["read1"].to_numpy().astype(float)
    t = (data["star1"] == "small").to_numpy().astype(float)
    return y, t


def evalue_for_difference(ate: float, sd: float) -> float:
    """
    VanderWeele & Ding E-value approximation for mean difference:
    convert standardized mean difference to an approximate risk ratio,
    then compute E-value = RR + sqrt(RR * (RR - 1)).
    """
    if sd <= 0:
        return math.nan
    d = abs(ate) / sd
    rr = math.exp(math.pi * d / math.sqrt(3.0))
    if rr < 1:
        rr = 1 / rr
    return rr + math.sqrt(rr * max(rr - 1, 0))


def summarize(name: str, y: np.ndarray, t: np.ndarray):
    diff = float(np.mean(y[t == 1]) - np.mean(y[t == 0]))
    sd = float(np.std(y, ddof=1))
    evalue = evalue_for_difference(diff, sd)
    return {"dataset": name, "diff_in_means": diff, "sd": sd, "evalue": evalue}


def main():
    rows = []
    for name, loader in [("lalonde", load_lalonde), ("star", load_star)]:
        y, t = loader()
        rows.append(summarize(name, y, t))
    df = pd.DataFrame(rows)
    df.to_csv("realdata_sensitivity.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
