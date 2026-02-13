"""
Plotting subpackage for toy_datasets.

Provides dataset visualisation and terminal rendering utilities.
"""
from data_loaders.plotting.visualisation import plot_dataset
from data_loaders.plotting.terminal_plots import terminal_show, enable_terminal_show, TerminalPlotter

__all__ = [
    'plot_dataset',
    'terminal_show',
    'enable_terminal_show',
    'TerminalPlotter',
]
