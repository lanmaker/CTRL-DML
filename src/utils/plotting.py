"""
Unified plotting style for CTRL-DML figures.

This module provides a consistent academic style:
- Clean serif fonts (Times New Roman style)
- Muted color palette with dark red as primary color
- No grid, minimal spines
- High-quality output (300 DPI)

Usage:
    from src.utils.plotting import set_style, COLORS, create_figure

    set_style()
    fig, ax = create_figure()
    ax.plot(x, y, color=COLORS['primary'])
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np


# =============================================================================
# Color Palette (Academic/Professional)
# =============================================================================
COLORS = {
    # Primary colors
    'primary': '#8B1C1C',       # Dark red/maroon - main color for points
    'primary_light': '#B85450', # Lighter red - for lines
    'secondary': '#2C5F8A',     # Steel blue - secondary color
    'secondary_light': '#5B8DB8', # Lighter blue

    # Accent colors
    'accent1': '#4A7C59',       # Forest green
    'accent2': '#8B6914',       # Golden brown
    'accent3': '#6B4C8A',       # Purple

    # Neutral colors
    'gray': '#808080',          # Medium gray - reference lines
    'dark_gray': '#4A4A4A',     # Dark gray - text, control group
    'light_gray': '#D3D3D3',    # Light gray

    # Fill colors (for CI bands, bars)
    'fill_primary': '#F2D7D5',  # Light pink
    'fill_secondary': '#D6EAF8', # Light blue
    'fill_accent': '#D5E8D4',   # Light green

    # For comparisons (method colors)
    'method1': '#8B1C1C',       # CTRL-DML (primary)
    'method2': '#2C5F8A',       # Causal Forest (blue)
    'method3': '#4A7C59',       # DragonNet (green)
    'method4': '#8B6914',       # TARNet (gold)
    'method5': '#6B4C8A',       # XGBoost (purple)
    'method6': '#808080',       # Baseline/OLS (gray)
}

# Named method colors for consistency across figures
METHOD_COLORS = {
    'CTRL-DML': COLORS['method1'],
    'CTRL': COLORS['method1'],
    'DML': COLORS['method1'],
    'Causal Forest': COLORS['method2'],
    'CF': COLORS['method2'],
    'DragonNet': COLORS['method3'],
    'TARNet': COLORS['method4'],
    'XGBoost': COLORS['method5'],
    'OLS': COLORS['method6'],
    'Baseline': COLORS['method6'],
    'Plugin': COLORS['primary_light'],
    'Plug-in': COLORS['primary_light'],
}

# Color cycle for automatic coloring
COLOR_CYCLE = [
    COLORS['primary'],
    COLORS['secondary'],
    COLORS['accent1'],
    COLORS['accent2'],
    COLORS['accent3'],
    COLORS['dark_gray'],
]


# =============================================================================
# Style Configuration
# =============================================================================
STYLE_CONFIG = {
    # Font settings
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",

    # Font sizes
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,

    # Figure settings
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "figure.edgecolor": "white",

    # Save settings
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",

    # Axes settings
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "axes.grid": False,
    "axes.axisbelow": True,
    "axes.facecolor": "white",
    "axes.edgecolor": COLORS['gray'],
    "axes.labelcolor": COLORS['dark_gray'],
    "axes.prop_cycle": plt.cycler(color=COLOR_CYCLE),

    # Tick settings
    "xtick.color": COLORS['gray'],
    "ytick.color": COLORS['gray'],
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,

    # Grid settings (off by default)
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "grid.color": COLORS['light_gray'],

    # Legend settings
    "legend.frameon": False,
    "legend.loc": "best",

    # Line settings
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
}


def set_style():
    """Apply the unified academic plotting style."""
    plt.rcdefaults()
    mpl.rcParams.update(STYLE_CONFIG)


def reset_style():
    """Reset to matplotlib defaults."""
    plt.rcdefaults()


# =============================================================================
# Figure Creation Helpers
# =============================================================================
def create_figure(
    figsize: Tuple[float, float] = (6, 4),
    nrows: int = 1,
    ncols: int = 1,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure with the unified style applied.

    Args:
        figsize: Figure size in inches (width, height)
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        **kwargs: Additional arguments for plt.subplots

    Returns:
        Tuple of (figure, axes)
    """
    set_style()
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, ax


def save_figure(fig: plt.Figure, path: Path, **kwargs):
    """
    Save figure with consistent settings.

    Args:
        fig: Figure to save
        path: Output path (will be converted to PDF)
        **kwargs: Additional arguments for savefig
    """
    path = Path(path)
    if path.suffix != '.pdf':
        path = path.with_suffix('.pdf')

    fig.savefig(path, **kwargs)
    plt.close(fig)


# =============================================================================
# Common Plot Types
# =============================================================================
def plot_bar_comparison(
    ax: plt.Axes,
    categories: List[str],
    values_dict: dict,
    yerr_dict: Optional[dict] = None,
    width: float = 0.25,
    ylabel: str = "PEHE",
    title: str = "",
):
    """
    Create a grouped bar chart for method comparison.

    Args:
        ax: Matplotlib axes
        categories: Category labels (x-axis)
        values_dict: Dict mapping method names to lists of values
        yerr_dict: Dict mapping method names to error bars
        width: Bar width
        ylabel: Y-axis label
        title: Plot title
    """
    x = np.arange(len(categories))
    n_methods = len(values_dict)
    offsets = np.linspace(-(n_methods-1)*width/2, (n_methods-1)*width/2, n_methods)

    for i, (method, values) in enumerate(values_dict.items()):
        color = METHOD_COLORS.get(method, COLOR_CYCLE[i % len(COLOR_CYCLE)])
        yerr = yerr_dict.get(method) if yerr_dict else None
        ax.bar(
            x + offsets[i], values, width,
            label=method, color=color,
            yerr=yerr, capsize=3, edgecolor='white', linewidth=0.5
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()


def plot_line_with_ci(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    ci_low: Optional[np.ndarray] = None,
    ci_high: Optional[np.ndarray] = None,
    color: str = None,
    label: str = None,
    fill_alpha: float = 0.3,
):
    """
    Plot a line with confidence interval band.

    Args:
        ax: Matplotlib axes
        x: X values
        y: Y values (center line)
        ci_low: Lower CI bound
        ci_high: Upper CI bound
        color: Line color (defaults to primary)
        label: Legend label
        fill_alpha: Alpha for CI fill
    """
    color = color or COLORS['primary']
    fill_color = COLORS['fill_primary'] if color == COLORS['primary'] else color

    # CI band
    if ci_low is not None and ci_high is not None:
        ax.fill_between(x, ci_low, ci_high, alpha=fill_alpha, color=fill_color, edgecolor='none')

    # Line
    ax.plot(x, y, '-', color=color, linewidth=1.5, label=label)

    # Points
    ax.scatter(x, y, s=25, c=color, edgecolors=color, linewidth=0.5, zorder=5)


def plot_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    xticklabels: List[str],
    yticklabels: List[str],
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    cmap: str = "RdYlGn_r",
    vmin: float = None,
    vmax: float = None,
    fmt: str = ".2f",
):
    """
    Create a heatmap with annotations.

    Args:
        ax: Matplotlib axes
        data: 2D array of values
        xticklabels: Labels for x-axis
        yticklabels: Labels for y-axis
        xlabel, ylabel, title: Axis labels and title
        cmap: Colormap name
        vmin, vmax: Color scale bounds
        fmt: Number format for annotations
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Custom colormap option
    if cmap == "academic_diverging":
        cmap = LinearSegmentedColormap.from_list(
            "academic",
            [COLORS['secondary'], '#FFFFFF', COLORS['primary']]
        )
    elif cmap == "academic_sequential":
        cmap = LinearSegmentedColormap.from_list(
            "academic_seq",
            ['#FFFFFF', COLORS['fill_primary'], COLORS['primary_light'], COLORS['primary']]
        )

    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    # Tick labels
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax.set_yticklabels(yticklabels)

    # Annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text_color = 'white' if (vmax and val > (vmax + vmin) / 2) else COLORS['dark_gray']
            ax.text(j, i, f"{val:{fmt}}", ha='center', va='center', fontsize=8, color=text_color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def add_reference_lines(ax: plt.Axes, hline: float = None, vline: float = None):
    """Add reference lines to a plot."""
    if hline is not None:
        ax.axhline(hline, color=COLORS['gray'], linewidth=0.5, linestyle='--', alpha=0.7)
    if vline is not None:
        ax.axvline(vline, color=COLORS['gray'], linewidth=0.5, linestyle='--', alpha=0.7)


# =============================================================================
# Style Context Manager
# =============================================================================
class academic_style:
    """Context manager for temporarily applying academic style."""

    def __enter__(self):
        self._original = mpl.rcParams.copy()
        set_style()
        return self

    def __exit__(self, *args):
        mpl.rcParams.update(self._original)


# =============================================================================
# Convenience Functions
# =============================================================================
def get_method_color(method: str) -> str:
    """Get color for a method name."""
    return METHOD_COLORS.get(method, COLORS['gray'])


def get_color_cycle(n: int) -> List[str]:
    """Get n colors from the color cycle."""
    return [COLOR_CYCLE[i % len(COLOR_CYCLE)] for i in range(n)]


if __name__ == "__main__":
    # Demo plot
    set_style()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Bar comparison
    ax = axes[0, 0]
    plot_bar_comparison(
        ax,
        categories=['Dataset 1', 'Dataset 2', 'Dataset 3'],
        values_dict={
            'CTRL-DML': [1.2, 1.5, 1.3],
            'Causal Forest': [1.8, 2.0, 1.9],
            'DragonNet': [1.5, 1.7, 1.6],
        },
        title='Method Comparison'
    )

    # Line with CI
    ax = axes[0, 1]
    x = np.linspace(0, 10, 20)
    y = np.sin(x) + 2
    ci_low = y - 0.3
    ci_high = y + 0.3
    plot_line_with_ci(ax, x, y, ci_low, ci_high, label='CTRL-DML')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Line with CI')
    ax.legend()

    # Heatmap
    ax = axes[1, 0]
    data = np.random.rand(5, 5) * 3
    plot_heatmap(
        ax, data,
        xticklabels=[f'X{i}' for i in range(5)],
        yticklabels=[f'Y{i}' for i in range(5)],
        title='Heatmap Demo',
        cmap='academic_sequential'
    )

    # Color palette demo
    ax = axes[1, 1]
    for i, (name, color) in enumerate(list(COLORS.items())[:8]):
        ax.barh(i, 1, color=color, edgecolor='white')
        ax.text(0.5, i, name, ha='center', va='center', fontsize=9,
                color='white' if i < 4 else COLORS['dark_gray'])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 7.5)
    ax.set_title('Color Palette')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('/Users/dingshen/Desktop/ProjeKCt/CTRL-DML/output/figures/style_demo.pdf')
    plt.close()
    print("Style demo saved to output/figures/style_demo.pdf")
