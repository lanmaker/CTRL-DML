"""
LaTeX table and macro generation utilities for CTRL-DML.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import re


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
        if isinstance(value, float):
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


def generate_all_macros(output_dir: Path, csv_dir: Optional[Path] = None) -> Path:
    """
    Generate all macros from available CSV results.

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

    # Ablation results
    ablation_csv = csv_dir / "ablation_results.csv"
    if ablation_csv.exists():
        df = pd.read_csv(ablation_csv)
        for variant in df["variant"].unique():
            vdf = df[df["variant"] == variant]
            prefix = variant.replace("_", "").title().replace(" ", "")
            gen.add(f"Abl{prefix}Dml", vdf["pehe_dml"].mean(), "Ablation Results")
            gen.add(f"Abl{prefix}Plugin", vdf["pehe_plugin"].mean(), "Ablation Results")

    # Scaling results
    scaling_csv = csv_dir / "scaling_dml.csv"
    if scaling_csv.exists():
        df = pd.read_csv(scaling_csv)
        for _, row in df.iterrows():
            dataset = row.get("dataset", "")
            method = row.get("method", "").replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
            if "pehe" in row:
                gen.add(f"Scale{dataset}{method}", row["pehe"], "Scaling Results")

    # UQ metrics
    uq_csv = csv_dir / "uq_metrics.csv"
    if uq_csv.exists():
        df = pd.read_csv(uq_csv)
        if len(df) > 0:
            row = df.iloc[0]
            gen.add("UqMcCov", row.get("mc_coverage", 0), "Uncertainty Quantification")
            gen.add("UqMcWidth", row.get("mc_width", 0), "Uncertainty Quantification")
            gen.add("UqConfCov", row.get("conf_coverage", 0), "Uncertainty Quantification")
            gen.add("UqConfWidth", row.get("conf_width", 0), "Uncertainty Quantification")

    # Multimodal results
    mm_csv = csv_dir / "multimodal_dml.csv"
    if mm_csv.exists():
        df = pd.read_csv(mm_csv)
        if "pehe_cf" in df.columns:
            gen.add("MmCfPehe", df["pehe_cf"].mean(), "Multimodal Results")
        if "pehe_plugin" in df.columns:
            gen.add("MmPluginPehe", df["pehe_plugin"].mean(), "Multimodal Results")
        if "pehe_dml" in df.columns:
            gen.add("MmDmlPehe", df["pehe_dml"].mean(), "Multimodal Results")

    # Training parameters (hardcoded for reference)
    gen.add("ParamKFold", 3, "Training Parameters")
    gen.add("ParamNuisanceEpochs", 150, "Training Parameters")
    gen.add("ParamTauEpochs", 300, "Training Parameters")
    gen.add("ParamDropout", 0.4, "Training Parameters")
    gen.add("ParamLambdaTau", "5\\times10^{-4}", "Training Parameters")

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

    if args.generate_macros:
        generate_all_macros(output_dir)
