"""
LaTeX table and macro generation utilities for CTRL-DML.
"""
import math
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import re

MISSING_LATEX = r"\ensuremath{\text{--}}"


def df_to_latex_table(
    df: pd.DataFrame,
    output_path: Path,
    caption: str,
    label: str,
    columns: Optional[List[str]] = None,
    column_format: Optional[str] = None,
    float_format: str = "%.2f",
    escape: bool = False,
    position: str = "t",
    centering: bool = True,
    booktabs: bool = True,
    header_map: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Convert DataFrame to a LaTeX table file.

    Args:
        df: pandas DataFrame
        output_path: Path to save .tex file
        caption: Table caption
        label: LaTeX label (e.g., "tab:ablation")
        columns: Subset of columns to include
        column_format: LaTeX column format (e.g., "lcc")
        float_format: Format string for floats
        escape: Whether to escape special characters
        position: Table position (t, h, b, p)
        centering: Whether to center the table
        booktabs: Use booktabs style (toprule, midrule, bottomrule)
        header_map: Rename columns in header

    Returns:
        Path to saved file
    """
    if columns:
        df = df[columns]

    if header_map:
        df = df.rename(columns=header_map)

    # Generate LaTeX
    if column_format is None:
        column_format = "l" + "c" * (len(df.columns) - 1)

    latex_body = df.to_latex(
        index=False,
        escape=escape,
        float_format=float_format,
        column_format=column_format,
    )

    # Remove default tabular environment to customize
    lines = latex_body.strip().split('\n')

    # Find content between begin/end tabular
    content_lines = []
    in_tabular = False
    for line in lines:
        if '\\begin{tabular}' in line:
            in_tabular = True
            continue
        if '\\end{tabular}' in line:
            in_tabular = False
            continue
        if in_tabular:
            content_lines.append(line)

    # Build custom table
    table_lines = [
        f"\\begin{{table}}[{position}]",
    ]

    if centering:
        table_lines.append("\\centering")

    table_lines.extend([
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_format}}}",
    ])

    if booktabs:
        # Replace hlines with booktabs commands
        for line in content_lines:
            if '\\hline' in line:
                continue  # Skip hlines, we'll use booktabs
            table_lines.append(line)

        # Insert booktabs rules
        table_lines.insert(len(table_lines) - len(content_lines) + 4, "\\toprule")
        # Find header row and add midrule after
        for i, line in enumerate(table_lines):
            if '&' in line and 'toprule' not in line and i < len(table_lines) - 2:
                table_lines.insert(i + 1, "\\midrule")
                break
    else:
        table_lines.extend(content_lines)

    table_lines.extend([
        "\\bottomrule" if booktabs else "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ])

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        f.write('\n'.join(table_lines))

    print(f"Saved LaTeX table: {output_path}")
    return output_path


class MacroGenerator:
    """Generate LaTeX macros from experiment results."""

    def __init__(self):
        self.macros: Dict[str, str] = {}
        self.sections: Dict[str, List[str]] = {}

    def add(self, name: str, value: Any, section: str = "General", fmt: str = "auto"):
        """
        Add a macro.

        Args:
            name: Macro name (will be converted to valid LaTeX command)
            value: Value (will be formatted appropriately)
            section: Section name for organization
            fmt: Format string or "auto"
        """
        # Sanitize macro name (remove invalid characters, ensure starts with letter)
        clean_name = re.sub(r'[^a-zA-Z0-9]', '', name)
        if clean_name and not clean_name[0].isalpha():
            clean_name = 'M' + clean_name

        # Format value
        if fmt == "auto":
            formatted = self._auto_format(value)
        else:
            formatted = fmt % value if '%' in fmt else str(value)

        self.macros[clean_name] = formatted

        if section not in self.sections:
            self.sections[section] = []
        if clean_name not in self.sections[section]:
            self.sections[section].append(clean_name)

    def _auto_format(self, value: Any) -> str:
        """Auto-format a value for LaTeX."""
        if value is None:
            return MISSING_LATEX
        if isinstance(value, float):
            if math.isnan(value):
                return MISSING_LATEX
            if abs(value) < 0.01 and value != 0:
                return f"{value:.2e}"
            elif abs(value) >= 100:
                return f"{value:.1f}"
            else:
                return f"{value:.2f}"
        elif isinstance(value, int):
            return str(value)
        else:
            return str(value)

    def add_from_csv(
        self,
        csv_path: Path,
        prefix: str,
        column_map: Dict[str, str],
        section: str = "General",
        row_filter: Optional[Dict[str, Any]] = None,
        agg: str = "mean"
    ):
        """
        Add macros from a CSV file.

        Args:
            csv_path: Path to CSV file
            prefix: Prefix for macro names
            column_map: Map column names to macro suffixes
            section: Section name
            row_filter: Filter rows (column: value dict)
            agg: Aggregation method ("mean", "first", "last")
        """
        df = pd.read_csv(csv_path)

        if row_filter:
            for col, val in row_filter.items():
                df = df[df[col] == val]

        for col, suffix in column_map.items():
            if col not in df.columns:
                continue

            values = df[col].values
            if agg == "mean":
                value = float(values.mean())
            elif agg == "std":
                value = float(values.std())
            elif agg == "first":
                value = values[0]
            elif agg == "last":
                value = values[-1]
            else:
                value = float(values.mean())

            macro_name = f"{prefix}{suffix}"
            self.add(macro_name, value, section)

    def add_from_dict(self, data: Dict[str, Any], prefix: str = "", section: str = "General"):
        """Add macros from a dictionary."""
        for key, value in data.items():
            name = f"{prefix}{key}" if prefix else key
            self.add(name, value, section)

    def generate(self, output_path: Path, include_timestamp: bool = True) -> Path:
        """
        Generate the macros file.

        Args:
            output_path: Path to save .tex file
            include_timestamp: Whether to include generation timestamp

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)

        lines = [
            "% Auto-generated LaTeX macros for CTRL-DML results",
            "% DO NOT EDIT MANUALLY - regenerate using: python -m src.utils.latex",
        ]

        if include_timestamp:
            lines.append(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        lines.append("")

        # Group by section
        for section, macro_names in self.sections.items():
            lines.append(f"% === {section} ===")
            for name in sorted(macro_names):
                value = self.macros[name]
                lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
            lines.append("")

        with open(output_path, "w") as f:
            f.write('\n'.join(lines))

        print(f"Saved LaTeX macros: {output_path} ({len(self.macros)} macros)")
        return output_path


def _try_read_csv(csv_dir: Path, *filenames: str) -> Optional[pd.DataFrame]:
    """Try to read CSV from multiple possible filenames."""
    for fname in filenames:
        path = csv_dir / fname
        if path.exists():
            return pd.read_csv(path)
    return None


def generate_all_macros(output_dir: Path, csv_dir: Optional[Path] = None) -> Path:
    """
    Generate all macros from available CSV results.

    Supports both old and new CSV filenames for backward compatibility.

    Args:
        output_dir: Output directory (for results_macros.tex)
        csv_dir: Directory containing CSV files (defaults to output_dir/tables)

    Returns:
        Path to generated macros file
    """
    output_dir = Path(output_dir)
    if csv_dir is None:
        csv_dir = output_dir / "tables"

    gen = MacroGenerator()

    # === Ablation Results ===
    df = _try_read_csv(csv_dir, "ablation_results.csv")
    if df is not None and "variant" in df.columns:
        variant_aliases = {
            "no_gating": "NoGating",
            "gating_no_L1": "Gating",
            "gating_no_l1": "Gating",
            "gating_L1": "GatingLone",
            "gating_l1": "GatingLone",
        }
        for variant in df["variant"].unique():
            vdf = df[df["variant"] == variant]
            prefix = variant_aliases.get(variant)
            if prefix is None:
                prefix = variant.replace("_", " ").title().replace(" ", "")
            prefix = re.sub(r"[^A-Za-z]", "", prefix)
            gen.add(f"Abl{prefix}Dml", vdf["pehe_dml"].mean(), "Ablation Results")
            gen.add(f"Abl{prefix}Plugin", vdf["pehe_plugin"].mean(), "Ablation Results")

    # === Cross-fitting Ablation ===
    df = _try_read_csv(csv_dir, "crossfit_ablation.csv")
    if df is not None:
        if "pehe_crossfit" in df.columns:
            gen.add("CrossfitPehe", df["pehe_crossfit"].mean(), "Cross-fit Ablation")
        if "pehe_nocf" in df.columns:
            gen.add("NoCrossfitPehe", df["pehe_nocf"].mean(), "Cross-fit Ablation")
        if "pehe_plugin" in df.columns:
            gen.add("CrossfitPluginPehe", df["pehe_plugin"].mean(), "Cross-fit Ablation")
    else:
        gen.add("CrossfitPehe", MISSING_LATEX, "Cross-fit Ablation")
        gen.add("NoCrossfitPehe", MISSING_LATEX, "Cross-fit Ablation")
        gen.add("CrossfitPluginPehe", MISSING_LATEX, "Cross-fit Ablation")
    for key in ["CrossfitPehe", "NoCrossfitPehe", "CrossfitPluginPehe"]:
        if key not in gen.macros:
            gen.add(key, MISSING_LATEX, "Cross-fit Ablation")

    # === Benchmark Results (IHDP) ===
    df = _try_read_csv(csv_dir, "benchmark_results.csv")
    if df is not None and len(df) > 0 and "dataset" in df.columns:
        # IHDP - PEHE metrics (paper uses IHDP prefix)
        ihdp = df[df["dataset"] == "ihdp"]
        if len(ihdp) > 0:
            if "pehe_dml_mean" in ihdp.columns:
                gen.add("IHDPCtrlPehe", ihdp["pehe_dml_mean"].mean(), "IHDP Benchmark")
                gen.add("IHDPCtrlStd", ihdp["pehe_dml_std"].mean(), "IHDP Benchmark")
                gen.add("IHDPCfPehe", ihdp["pehe_plugin_mean"].mean(), "IHDP Benchmark")
                gen.add("IHDPCfStd", ihdp["pehe_plugin_std"].mean(), "IHDP Benchmark")
            elif "pehe_dml" in ihdp.columns:
                gen.add("IHDPCtrlPehe", ihdp["pehe_dml"].mean(), "IHDP Benchmark")
                gen.add("IHDPCtrlStd", ihdp["pehe_dml"].std(), "IHDP Benchmark")
                gen.add("IHDPCfPehe", ihdp["pehe_plugin"].mean(), "IHDP Benchmark")
                gen.add("IHDPCfStd", ihdp["pehe_plugin"].std(), "IHDP Benchmark")
            # Paper-facing macros (mean/std/median/IQR for plug-in vs DML)
            plugin = ihdp["pehe_plugin"].dropna()
            dml = ihdp["pehe_dml"].dropna()
            if len(plugin) > 0:
                gen.add("IhdpPluginPehe", plugin.mean(), "IHDP Benchmark")
                gen.add("IhdpPluginStd", plugin.std(), "IHDP Benchmark")
                gen.add("IhdpPluginMedian", plugin.median(), "IHDP Benchmark")
                gen.add("IhdpPluginIqr", plugin.quantile(0.75) - plugin.quantile(0.25), "IHDP Benchmark")
            if len(dml) > 0:
                gen.add("IhdpDmlPehe", dml.mean(), "IHDP Benchmark")
                gen.add("IhdpDmlStd", dml.std(), "IHDP Benchmark")
                gen.add("IhdpDmlMedian", dml.median(), "IHDP Benchmark")
                gen.add("IhdpDmlIqr", dml.quantile(0.75) - dml.quantile(0.25), "IHDP Benchmark")

    # === Public Benchmarks (TWINS/ACIC baselines) ===
    df = _try_read_csv(csv_dir, "public_benchmarks.csv")
    if df is not None and len(df) > 0 and {"dataset", "method", "ate"}.issubset(df.columns):
        def _norm_method(value: str) -> str:
            return re.sub(r"\\s+", "", str(value)).lower()

        def _pick(dataset_prefix: str, method_key: str):
            mask = df["dataset"].astype(str).str.lower().str.startswith(dataset_prefix)
            sub = df[mask & (df["method"].map(_norm_method) == method_key)]
            return sub.iloc[0] if len(sub) > 0 else None

        row = _pick("twins", "causalforest")
        if row is not None:
            gen.add("TwinsCfAte", row["ate"], "TWINS Benchmark")
            if "ci_lower" in row:
                gen.add("TwinsCfCiLo", row["ci_lower"], "TWINS Benchmark")
                gen.add("TwinsCfCiHi", row["ci_upper"], "TWINS Benchmark")

        row = _pick("twins", "tarnet")
        if row is not None:
            gen.add("TwinsTarnetAte", row["ate"], "TWINS Benchmark")
            if "ci_lower" in row:
                gen.add("TwinsTarnetCiLo", row["ci_lower"], "TWINS Benchmark")
                gen.add("TwinsTarnetCiHi", row["ci_upper"], "TWINS Benchmark")

        row = _pick("acic", "causalforest")
        if row is not None:
            gen.add("AcicCfAte", row["ate"], "ACIC Benchmark")
            if "ci_lower" in row:
                gen.add("AcicCfCiLo", row["ci_lower"], "ACIC Benchmark")
                gen.add("AcicCfCiHi", row["ci_upper"], "ACIC Benchmark")

        row = _pick("acic", "tarnet")
        if row is not None:
            gen.add("AcicTarnetAte", row["ate"], "ACIC Benchmark")
            if "ci_lower" in row:
                gen.add("AcicTarnetCiLo", row["ci_lower"], "ACIC Benchmark")
                gen.add("AcicTarnetCiHi", row["ci_upper"], "ACIC Benchmark")

    # Ensure public benchmark macros exist for table rendering.
    for prefix, section in [("Twins", "TWINS Benchmark"), ("Acic", "ACIC Benchmark")]:
        for method in ["Cf", "Tarnet"]:
            for suffix in ["Ate", "CiLo", "CiHi"]:
                key = f"{prefix}{method}{suffix}"
                if key not in gen.macros:
                    gen.add(key, MISSING_LATEX, section)

    # === Scaling Results ===
    df = _try_read_csv(csv_dir, "scaling_dml.csv")
    if df is not None:
        # New format: n_samples, pehe_cf, pehe_ctrl
        if "n_samples" in df.columns:
            for n in df["n_samples"].unique():
                ndf = df[df["n_samples"] == n]
                suffix = {500: "FiveHundred", 1000: "OneK", 2000: "TwoK", 5000: "FiveK"}.get(int(n), str(int(n)))
                if "pehe_cf" in ndf.columns:
                    gen.add(f"ScaleCfN{suffix}", ndf["pehe_cf"].mean(), "Scaling Results")
                if "pehe_ctrl" in ndf.columns:
                    gen.add(f"ScaleCtrlN{suffix}", ndf["pehe_ctrl"].mean(), "Scaling Results")
                if "pehe_dml" in ndf.columns:
                    gen.add(f"ScaleDmlN{suffix}", ndf["pehe_dml"].mean(), "Scaling Results")

    # === Nuisance Misspecification ===
    df = _try_read_csv(csv_dir, "nuisance_misspec.csv")
    if df is not None and "nuisance_strength" in df.columns:
        for strength in ["strong", "weak"]:
            mask = df["nuisance_strength"] == strength
            if mask.any():
                suffix = strength.title()
                gen.add(f"NuisPlugin{suffix}", df.loc[mask, "pehe_plugin"].mean(), "Nuisance Misspecification")
                gen.add(f"NuisDml{suffix}", df.loc[mask, "pehe_dml"].mean(), "Nuisance Misspecification")

    # === Multimodal Results ===
    df = _try_read_csv(csv_dir, "multimodal_benchmark.csv")
    if df is not None:
        # Paper uses: MmCfPehe, MmPluginPehe (dense), MmBowPluginPehe, MmDmlPehe, MmCtrlBagPehe
        if "pehe_cf" in df.columns:
            gen.add("MmCfPehe", df["pehe_cf"].mean(), "Multimodal Results")
        else:
            gen.add("MmCfPehe", MISSING_LATEX, "Multimodal Results")

        if "pehe_plugin" in df.columns:
            gen.add("MmPluginPehe", df["pehe_plugin"].mean(), "Multimodal Results")
        elif "pehe_multimodal" in df.columns:
            gen.add("MmPluginPehe", df["pehe_multimodal"].mean(), "Multimodal Results")
        else:
            gen.add("MmPluginPehe", MISSING_LATEX, "Multimodal Results")

        if "pehe_concat_bow" in df.columns:
            gen.add("MmBowPluginPehe", df["pehe_concat_bow"].mean(), "Multimodal Results")
        else:
            gen.add("MmBowPluginPehe", MISSING_LATEX, "Multimodal Results")

        if "pehe_dml" in df.columns:
            gen.add("MmDmlPehe", df["pehe_dml"].mean(), "Multimodal Results")
        else:
            gen.add("MmDmlPehe", MISSING_LATEX, "Multimodal Results")

        if "pehe_ctrl_bag" in df.columns:
            gen.add("MmCtrlBagPehe", df["pehe_ctrl_bag"].mean(), "Multimodal Results")
        else:
            gen.add("MmCtrlBagPehe", MISSING_LATEX, "Multimodal Results")
    for key in ["MmCfPehe", "MmPluginPehe", "MmBowPluginPehe", "MmDmlPehe", "MmCtrlBagPehe"]:
        if key not in gen.macros:
            gen.add(key, MISSING_LATEX, "Multimodal Results")

    # Cross-attention results
    df_ca = _try_read_csv(csv_dir, "multimodal_cross_attention.csv")
    if df_ca is not None and "pehe" in df_ca.columns:
        gen.add("MmCrossAttnPehe", df_ca["pehe"].mean(), "Multimodal Results")
    elif df is not None and "pehe_cross_attn" in df.columns:
        gen.add("MmCrossAttnPehe", df["pehe_cross_attn"].mean(), "Multimodal Results")
    else:
        gen.add("MmCrossAttnPehe", MISSING_LATEX, "Multimodal Results")

    # BERT baselines
    df_bert = _try_read_csv(csv_dir, "multimodal_bert_baselines.csv")
    if df_bert is not None:
        if "pehe_tarnet" in df_bert.columns:
            gen.add("MmBertTarnetPehe", df_bert["pehe_tarnet"].mean(), "Multimodal Results")
        if "pehe_cf" in df_bert.columns:
            gen.add("MmBertCfPehe", df_bert["pehe_cf"].mean(), "Multimodal Results")
    else:
        gen.add("MmBertTarnetPehe", MISSING_LATEX, "Multimodal Results")
        gen.add("MmBertCfPehe", MISSING_LATEX, "Multimodal Results")
    for key in ["MmBertTarnetPehe", "MmBertCfPehe"]:
        if key not in gen.macros:
            gen.add(key, MISSING_LATEX, "Multimodal Results")

    # === Multimodal Noise Sweep ===
    # Paper uses word-form names: Zero (0.0), Three (0.3), Six (0.6)
    noise_label_map = {
        0.0: "Zero", 0.3: "Three", 0.5: "Five", 0.6: "Six"
    }
    df = _try_read_csv(csv_dir, "multimodal_noise_sweep.csv")
    if df is not None and "p_noise" in df.columns:
        for p in df["p_noise"].unique():
            pdf = df[df["p_noise"] == p]
            p_val = round(float(p), 1)
            p_label = noise_label_map.get(p_val, str(int(p_val * 100)))
            if "pehe_multimodal" in pdf.columns:
                gen.add(f"MmNoise{p_label}Ctrl", pdf["pehe_multimodal"].mean(), "Multimodal Noise Sweep")
            if "pehe_tabular" in pdf.columns:
                gen.add(f"MmNoise{p_label}Tab", pdf["pehe_tabular"].mean(), "Multimodal Noise Sweep")
            elif "pehe_tabular_only" in pdf.columns:
                gen.add(f"MmNoise{p_label}Tab", pdf["pehe_tabular_only"].mean(), "Multimodal Noise Sweep")
    # Ensure paper-expected noise levels have macros (use placeholders if missing).
    for p_val, p_label in [(0.0, "Zero"), (0.3, "Three"), (0.6, "Six")]:
        ctrl_key = f"MmNoise{p_label}Ctrl"
        tab_key = f"MmNoise{p_label}Tab"
        if ctrl_key not in gen.macros:
            gen.add(ctrl_key, MISSING_LATEX, "Multimodal Noise Sweep")
        if tab_key not in gen.macros:
            gen.add(tab_key, MISSING_LATEX, "Multimodal Noise Sweep")

    # === UQ Metrics ===
    df = _try_read_csv(csv_dir, "uq_metrics.csv")
    if df is not None and len(df) > 0:
        gen.add("UqMcCov", df["mc_coverage"].mean(), "Uncertainty Quantification")
        gen.add("UqMcWidth", df["mc_width"].mean(), "Uncertainty Quantification")
        gen.add("UqConfCov", df["conf_coverage"].mean(), "Uncertainty Quantification")
        gen.add("UqConfWidth", df["conf_width"].mean(), "Uncertainty Quantification")
    else:
        gen.add("UqMcCov", MISSING_LATEX, "Uncertainty Quantification")
        gen.add("UqMcWidth", MISSING_LATEX, "Uncertainty Quantification")
        gen.add("UqConfCov", MISSING_LATEX, "Uncertainty Quantification")
        gen.add("UqConfWidth", MISSING_LATEX, "Uncertainty Quantification")

    # === Training Dynamics ===
    df = _try_read_csv(csv_dir, "training_dynamics.csv")
    if df is not None and "epoch" in df.columns:
        df = df.sort_values("epoch")
        last = df.iloc[-1]
        gen.add("DynConfWeight", last["conf_weight"], "Training Dynamics")
        gen.add("DynInstWeight", last["inst_weight"], "Training Dynamics")
        gen.add("DynNoiseWeight", last["noise_weight"], "Training Dynamics")
        gen.add("DynDecayEpoch", int(last["epoch"]), "Training Dynamics")
    else:
        gen.add("DynConfWeight", MISSING_LATEX, "Training Dynamics")
        gen.add("DynInstWeight", MISSING_LATEX, "Training Dynamics")
        gen.add("DynNoiseWeight", MISSING_LATEX, "Training Dynamics")
        gen.add("DynDecayEpoch", MISSING_LATEX, "Training Dynamics")

    # === Feature Roles ===
    df = _try_read_csv(csv_dir, "feature_roles.csv")
    if df is not None and "true_role" in df.columns:
        # Precision@10 for confounders
        conf_df = df.sort_values("s_conf", ascending=False).head(10)
        prec_conf = (conf_df["true_role"] == "confounder").mean()
        gen.add("FeatConfPrecTen", prec_conf, "Feature Roles")
        total_conf = (df["true_role"] == "confounder").sum()
        if total_conf > 0:
            recall_conf = (conf_df["true_role"] == "confounder").sum() / total_conf
            gen.add("FeatConfRecallTen", recall_conf, "Feature Roles")
        # Precision@10 for instruments
        inst_df = df.sort_values("s_inst", ascending=False).head(10)
        prec_inst = (inst_df["true_role"] == "instrument").mean()
        gen.add("FeatInstPrecTen", prec_inst, "Feature Roles")

    # === Bias-Variance ===
    df = _try_read_csv(csv_dir, "bias_variance.csv")
    if df is not None and "confounders" in df.columns:
        for conf in ["with_conf", "no_conf"]:
            mask = df["confounders"] == conf
            if mask.any():
                suffix = "".join(w.title() for w in conf.split("_"))
                gen.add(f"BvPlugin{suffix}", df.loc[mask, "pehe_plugin"].mean(), "Bias-Variance")
                gen.add(f"BvOrth{suffix}", df.loc[mask, "pehe_dml"].mean(), "Bias-Variance")

    # === Yelp Semi-synthetic ===
    df = _try_read_csv(csv_dir, "yelp_semisynth.csv")
    if df is not None and len(df) > 0:
        if "pehe_cf" in df.columns:
            gen.add("YelpCfPehe", df["pehe_cf"].mean(), "Yelp Semi-synthetic")
        if "pehe_plugin" in df.columns:
            gen.add("YelpPluginPehe", df["pehe_plugin"].mean(), "Yelp Semi-synthetic")
        if "pehe_dml" in df.columns:
            gen.add("YelpDmlPehe", df["pehe_dml"].mean(), "Yelp Semi-synthetic")

    # === Real Data ATE ===
    df = _try_read_csv(csv_dir, "realdata_ate.csv")
    if df is not None and "dataset" in df.columns:
        for dataset in df["dataset"].unique():
            ddf = df[df["dataset"] == dataset]
            prefix = dataset.title()
            for _, row in ddf.iterrows():
                method_raw = str(row.get("method", ""))
                method_norm = method_raw.lower().replace("-", "").replace(" ", "")
                if method_norm == "ols":
                    method = "Ols"
                elif method_norm in {"causalforest", "cf"}:
                    method = "Cf"
                elif method_norm in {"ctrldml", "ctrl"}:
                    method = "Ctrl"
                else:
                    method = method_raw.replace(" ", "")
                gen.add(f"{prefix}{method}Ate", row["ate"], f"{prefix} Real Data")
                if "ci_lower" in row:
                    gen.add(f"{prefix}{method}CiLo", row["ci_lower"], f"{prefix} Real Data")
                    gen.add(f"{prefix}{method}CiHi", row["ci_upper"], f"{prefix} Real Data")

    # Ensure ATE macros exist for expected datasets/methods.
    for prefix in ["Lalonde", "Star"]:
        for method in ["Ols", "Cf", "Ctrl"]:
            for suffix in ["Ate", "CiLo", "CiHi"]:
                key = f"{prefix}{method}{suffix}"
                if key not in gen.macros:
                    gen.add(key, MISSING_LATEX, f"{prefix} Real Data")

    # === Real Data Balance (SMD) ===
    df = _try_read_csv(csv_dir, "realdata_balance.csv")
    if df is not None and "dataset" in df.columns:
        for dataset in df["dataset"].unique():
            ddf = df[df["dataset"] == dataset]
            prefix = dataset.title()
            for _, row in ddf.iterrows():
                weighting_raw = str(row.get("weighting", ""))
                weighting_key = weighting_raw.lower().replace("-", "").replace(" ", "")
                if "unweighted" in weighting_key:
                    weighting = "Unweighted"
                elif "ctrl" in weighting_key:
                    weighting = "CTRLIPW"
                else:
                    weighting = weighting_raw.replace("-", " ").title().replace(" ", "")
                if "max_smd" in row:
                    gen.add(f"{prefix}{weighting}Smd", row["max_smd"], f"{prefix} Balance")
                if "n_imbalanced" in row:
                    gen.add(f"{prefix}{weighting}Count", int(row["n_imbalanced"]), f"{prefix} Balance")

    # Ensure balance macros exist for expected datasets/weightings.
    for prefix in ["Lalonde", "Star"]:
        for weighting in ["Unweighted", "CTRLIPW"]:
            smd_key = f"{prefix}{weighting}Smd"
            count_key = f"{prefix}{weighting}Count"
            if smd_key not in gen.macros:
                gen.add(smd_key, MISSING_LATEX, f"{prefix} Balance")
            if count_key not in gen.macros:
                gen.add(count_key, MISSING_LATEX, f"{prefix} Balance")

    # === TWINS/ACIC Fallback Placeholders ===
    # Check if TWINS data exists in benchmark_results
    df_bench = _try_read_csv(csv_dir, "benchmark_results.csv")
    df_public = _try_read_csv(csv_dir, "public_benchmarks.csv")
    twins_exists = (
        df_bench is not None and "twins" in df_bench.get("dataset", pd.Series()).values
    ) or (
        df_public is not None and any(str(d).startswith("twins") for d in df_public.get("dataset", pd.Series()).values)
    )
    acic_exists = (
        df_bench is not None and any(str(d).startswith("acic") for d in df_bench.get("dataset", pd.Series()).values)
    ) or (
        df_public is not None and any(str(d).startswith("acic") for d in df_public.get("dataset", pd.Series()).values)
    )

    if not twins_exists:
        gen.add("TwinsCfAte", MISSING_LATEX, "TWINS Benchmark")
        gen.add("TwinsCfCiLo", MISSING_LATEX, "TWINS Benchmark")
        gen.add("TwinsCfCiHi", MISSING_LATEX, "TWINS Benchmark")
        gen.add("TwinsTarnetAte", MISSING_LATEX, "TWINS Benchmark")

    if not acic_exists:
        gen.add("AcicCfAte", MISSING_LATEX, "ACIC Benchmark")
        gen.add("AcicCfCiLo", MISSING_LATEX, "ACIC Benchmark")
        gen.add("AcicCfCiHi", MISSING_LATEX, "ACIC Benchmark")
        gen.add("AcicTarnetAte", MISSING_LATEX, "ACIC Benchmark")

    # === Yelp Semi-synthetic Fallback Placeholders ===
    df_yelp = _try_read_csv(csv_dir, "yelp_semisynth.csv")
    if df_yelp is None:
        gen.add("YelpCfPehe", MISSING_LATEX, "Yelp Semi-synthetic")
        gen.add("YelpPluginPehe", MISSING_LATEX, "Yelp Semi-synthetic")
        gen.add("YelpDmlPehe", MISSING_LATEX, "Yelp Semi-synthetic")
    for key in ["YelpCfPehe", "YelpPluginPehe", "YelpDmlPehe"]:
        if key not in gen.macros:
            gen.add(key, MISSING_LATEX, "Yelp Semi-synthetic")

    # === Training Parameters (from CTRLConfig) ===
    def _latex_sci(value: float) -> str:
        mant, exp = f"{value:.0e}".split("e")
        return rf"{mant}\times10^{{{int(exp)}}}"

    try:
        from src.models.orthogonal_learner import CTRLConfig
        cfg = CTRLConfig()
    except Exception:
        cfg = None

    if cfg is not None:
        gen.add("ParamKFold", cfg.k_folds, "Training Parameters")
        gen.add("ParamNuisanceEpochs", cfg.nuisance_epochs, "Training Parameters")
        gen.add("ParamTauEpochs", cfg.tau_epochs, "Training Parameters")
        gen.add("ParamPluginEpochs", cfg.plugin_epochs, "Training Parameters")
        gen.add("ParamDropout", cfg.dropout_p, "Training Parameters")
        gen.add("ParamLambdaTau", _latex_sci(cfg.lambda_tau), "Training Parameters")
        gen.add("ParamGradClip", cfg.grad_clip, "Training Parameters")
        gen.add("ParamHiddenNuisance", cfg.hidden_dim, "Training Parameters")
        gen.add("ParamHiddenTau", cfg.hidden_tau, "Training Parameters")
        gen.add("ParamLrDml", _latex_sci(cfg.lr_tau), "Training Parameters")
        gen.add("ParamLrPlugin", _latex_sci(cfg.lr_plugin), "Training Parameters")
        gen.add("ParamWClip", cfg.w_clip, "Training Parameters")
        gen.add("ParamZClip", cfg.z_clip, "Training Parameters")
        gen.add("ParamEps", cfg.eps, "Training Parameters")
        gen.add("ParamAuxBetaStart", cfg.aux_beta_start, "Training Parameters")
        gen.add("ParamAuxBetaEnd", cfg.aux_beta_end, "Training Parameters")
    else:
        gen.add("ParamKFold", MISSING_LATEX, "Training Parameters")
        gen.add("ParamNuisanceEpochs", MISSING_LATEX, "Training Parameters")
        gen.add("ParamTauEpochs", MISSING_LATEX, "Training Parameters")
        gen.add("ParamPluginEpochs", MISSING_LATEX, "Training Parameters")
        gen.add("ParamDropout", MISSING_LATEX, "Training Parameters")
        gen.add("ParamLambdaTau", MISSING_LATEX, "Training Parameters")
        gen.add("ParamGradClip", MISSING_LATEX, "Training Parameters")
        gen.add("ParamHiddenNuisance", MISSING_LATEX, "Training Parameters")
        gen.add("ParamHiddenTau", MISSING_LATEX, "Training Parameters")
        gen.add("ParamLrDml", MISSING_LATEX, "Training Parameters")
        gen.add("ParamLrPlugin", MISSING_LATEX, "Training Parameters")
        gen.add("ParamWClip", MISSING_LATEX, "Training Parameters")
        gen.add("ParamZClip", MISSING_LATEX, "Training Parameters")
        gen.add("ParamEps", MISSING_LATEX, "Training Parameters")
        gen.add("ParamAuxBetaStart", MISSING_LATEX, "Training Parameters")
        gen.add("ParamAuxBetaEnd", MISSING_LATEX, "Training Parameters")

    gen.add("ParamPropClipLo", 0.01, "Training Parameters")
    gen.add("ParamPropClipHi", 0.99, "Training Parameters")

    # Generate file
    output_path = output_dir / "results_macros.tex"
    return gen.generate(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate LaTeX assets")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--generate-macros", action="store_true", help="Generate macros file")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Default behavior: always generate macros when run as module
    generate_all_macros(output_dir)
