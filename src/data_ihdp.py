import pandas as pd
from typing import List


COLUMNS = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]
BASE_URL = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_{}.csv"


def load_ihdp_replicate(rep: int = 1) -> pd.DataFrame:
    """
    Load a single IHDP semi-synthetic replicate with ground-truth potential outcomes.

    Columns: treatment, y_factual, y_cfactual, mu0, mu1, x1..x25
    """
    df = pd.read_csv(BASE_URL.format(rep), header=None)
    df.columns = COLUMNS
    df["replicate"] = rep
    return df


def load_ihdp_replicates(reps: List[int]) -> pd.DataFrame:
    """Load multiple IHDP replicates and concatenate."""
    dfs = [load_ihdp_replicate(r) for r in reps]
    return pd.concat(dfs, ignore_index=True)

