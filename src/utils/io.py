"""
Output path management for CTRL-DML project.
Centralizes all output paths to ensure consistent structure.
"""
from pathlib import Path
from typing import Optional
import pandas as pd


class OutputManager:
    """Centralized output path management."""

    _instance: Optional["OutputManager"] = None

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize OutputManager.

        Args:
            base_dir: Base output directory. Defaults to project_root/output
        """
        if base_dir is None:
            # Find project root (parent of src/)
            self.project_root = Path(__file__).parents[2]
            self.base = self.project_root / "output"
        else:
            self.base = Path(base_dir)
            self.project_root = self.base.parent

        # Define subdirectories
        self.figures = self.base / "figures"
        self.tables = self.base / "tables"

        # Create directories if they don't exist
        for d in [self.figures, self.tables]:
            d.mkdir(parents=True, exist_ok=True)

    def figure_path(self, name: str, ext: str = "pdf") -> Path:
        """
        Get path for saving a figure.

        Args:
            name: Figure name (without extension)
            ext: File extension (default: pdf)

        Returns:
            Full path to figure file
        """
        return self.figures / f"{name}.{ext}"

    def table_path(self, name: str, ext: str = "tex") -> Path:
        """
        Get path for saving a LaTeX table.

        Args:
            name: Table name (without extension)
            ext: File extension (default: tex)

        Returns:
            Full path to table file
        """
        return self.tables / f"{name}.{ext}"

    def csv_path(self, name: str) -> Path:
        """
        Get path for saving a CSV file (in tables folder for convenience).

        Args:
            name: CSV name (without extension)

        Returns:
            Full path to CSV file
        """
        return self.tables / f"{name}.csv"

    def macro_path(self) -> Path:
        """Get path for the LaTeX macros file."""
        return self.base / "results_macros.tex"

    def paper_path(self) -> Path:
        """Get path to the paper directory."""
        return self.project_root / "CTRL-DML-Paper"

    def save_figure(self, fig, name: str, dpi: int = 300, **kwargs):
        """
        Save a matplotlib figure.

        Args:
            fig: matplotlib figure object
            name: Figure name
            dpi: Resolution
            **kwargs: Additional arguments to savefig
        """
        path = self.figure_path(name)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
        print(f"Saved figure: {path}")
        return path

    def save_csv(self, df: pd.DataFrame, name: str, **kwargs) -> Path:
        """
        Save a DataFrame as CSV.

        Args:
            df: pandas DataFrame
            name: File name
            **kwargs: Additional arguments to to_csv

        Returns:
            Path to saved file
        """
        path = self.csv_path(name)
        df.to_csv(path, index=False, **kwargs)
        print(f"Saved CSV: {path}")
        return path


# Singleton instance
_output_manager: Optional[OutputManager] = None


def get_output_manager(base_dir: Optional[Path] = None) -> OutputManager:
    """
    Get the global OutputManager instance.

    Args:
        base_dir: Optional base directory (only used on first call)

    Returns:
        OutputManager instance
    """
    global _output_manager
    if _output_manager is None:
        _output_manager = OutputManager(base_dir)
    return _output_manager


# Convenience functions
def figure_path(name: str, ext: str = "pdf") -> Path:
    """Shortcut to get figure path."""
    return get_output_manager().figure_path(name, ext)


def table_path(name: str, ext: str = "tex") -> Path:
    """Shortcut to get table path."""
    return get_output_manager().table_path(name, ext)


def csv_path(name: str) -> Path:
    """Shortcut to get CSV path."""
    return get_output_manager().csv_path(name)
