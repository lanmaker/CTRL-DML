import csv
import numpy as np
import matplotlib.pyplot as plt


def load_results(path="ablation_results.csv"):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "variant": r["variant"],
                    "pehe_dml": float(r["pehe_dml"]),
                    "pehe_plugin": float(r["pehe_plugin"]),
                }
            )
    return rows


def aggregate(rows):
    agg = {}
    for r in rows:
        agg.setdefault(r["variant"], {"dml": [], "plugin": []})
        agg[r["variant"]]["dml"].append(r["pehe_dml"])
        agg[r["variant"]]["plugin"].append(r["pehe_plugin"])
    variants = []
    dml_mean, dml_std = [], []
    plug_mean, plug_std = [], []
    for v, vals in agg.items():
        variants.append(v)
        dml_mean.append(np.mean(vals["dml"]))
        dml_std.append(np.std(vals["dml"]))
        plug_mean.append(np.mean(vals["plugin"]))
        plug_std.append(np.std(vals["plugin"]))
    return variants, np.array(dml_mean), np.array(dml_std), np.array(plug_mean), np.array(plug_std)


def plot_ablation(variants, dml_mean, dml_std, plug_mean, plug_std, out_path="ablation_plot.pdf"):
    x = np.arange(len(variants))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, dml_mean, width, yerr=dml_std, label="DML (orthogonal)", color="#9467bd", alpha=0.85, capsize=4)
    ax.bar(x + width / 2, plug_mean, width, yerr=plug_std, label="Plug-in (TARNet)", color="#1f77b4", alpha=0.85, capsize=4)
    ax.set_ylabel("PEHE (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20)
    ax.set_title("Ablation: gating and DML toggles")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    rows = load_results()
    variants, dml_mean, dml_std, plug_mean, plug_std = aggregate(rows)
    plot_ablation(variants, dml_mean, dml_std, plug_mean, plug_std)
