# CTRL-DML Utilities
from .io import OutputManager, get_output_manager
from .metrics import compute_pehe, compute_ate, bootstrap_mean, compute_smd
from .latex import df_to_latex_table, MacroGenerator, generate_all_macros
from .training import set_seed, train_epoch, EarlyStopping
from .plotting import (
    set_style, reset_style, create_figure, save_figure,
    COLORS, METHOD_COLORS, COLOR_CYCLE,
    plot_bar_comparison, plot_line_with_ci, plot_heatmap,
    add_reference_lines, get_method_color, get_color_cycle,
    academic_style,
)

__all__ = [
    # IO
    "OutputManager",
    "get_output_manager",
    # Metrics
    "compute_pehe",
    "compute_ate",
    "bootstrap_mean",
    "compute_smd",
    # LaTeX
    "df_to_latex_table",
    "MacroGenerator",
    "generate_all_macros",
    # Training
    "set_seed",
    "train_epoch",
    "EarlyStopping",
    # Plotting
    "set_style",
    "reset_style",
    "create_figure",
    "save_figure",
    "COLORS",
    "METHOD_COLORS",
    "COLOR_CYCLE",
    "plot_bar_comparison",
    "plot_line_with_ci",
    "plot_heatmap",
    "add_reference_lines",
    "get_method_color",
    "get_color_cycle",
    "academic_style",
]
